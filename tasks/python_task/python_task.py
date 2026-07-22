import ast
import math
import random
import os
import signal
import tempfile
from contextlib import contextmanager

from tasks.math_task.math_helpers import ( adapt_scalar_compute, adapt_list_compute, adapt_scalar_prompt, adapt_list_prompt)
from tasks.python_task.rename_input_output import ( FunctionAnalyzer, rename_and_chain,)

class TimeoutException(Exception):
    pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def check_type_annotation(node):
    """Check if an AST annotation node represents a 'list' or 'scalar' type."""
    if node is None:
        return 'scalar'
    if isinstance(node, ast.Name):
        return 'scalar'
    elif isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in ["list", "List"]:
            return 'list'
        elif isinstance(node.value, ast.Name) and node.value.id in ["tuple", "Tuple"]:
            if isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    if check_type_annotation(elt) == 'list':
                        return 'list'
            return 'scalar'
    return 'scalar'

def extract_function_info(code_str):
    """Extract function info (name, params, types, code) from a code string."""
    try:
        tree = ast.parse(code_str)
    except Exception:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args_to_check = node.args.posonlyargs + node.args.args
            if args_to_check and args_to_check[0].arg == 'self':
                args_to_check = args_to_check[1:]

            has_list_in = False
            has_scalar_in = False
            all_params = []
            for arg in args_to_check:
                t = check_type_annotation(arg.annotation)
                all_params.append((arg.arg, t))
                if t == 'list':
                    has_list_in = True
                else:
                    has_scalar_in = True

            has_list_out = check_type_annotation(node.returns) == 'list'

            return {
                'name': node.name,
                'has_list_in': has_list_in,
                'has_scalar_in': has_scalar_in,
                'has_list_out': has_list_out,
                'params': all_params,
                'code': code_str,
            }
    return None


def load_candidate_algos(filepath):
    """Load and parse all candidate algorithms from the leetcode file."""
    with open(filepath, "r") as f:
        content = f.read()

    blocks = content.split("-" * 80)
    all_funcs = []

    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        code = "\n".join(lines[1:])
        info = extract_function_info(code)
        if info:
            all_funcs.append(info)

    return all_funcs


def generate_random_value(type_hint, list_length=10, val_min=-100, val_max=100):
    """Generate a random int scalar or list[int]."""
    if type_hint == 'list':
        return [random.randint(val_min, val_max) for _ in range(list_length)]
    else:
        return random.randint(val_min, val_max)


def execute_function(code_str, func_name, inputs, timeout):
    """Execute a leetcode Solution method with the given inputs dict."""
    local_ns = {}
    exec("from typing import *", local_ns)
    exec("import math", local_ns)

    try:
        exec(code_str, local_ns)
        if 'Solution' in local_ns:
            obj = local_ns['Solution']()
            func = getattr(obj, func_name)
        else:
            func = local_ns[func_name]

        with time_limit(timeout):
            return func(**inputs)
    except Exception as e:
        raise e


def _rename_code(source_code, new_input_name, new_output_name, prefer_type='list', new_method_name=None):
    """Run rename_and_chain via temp files and return the transformed code."""
    tmp_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', dir=tmp_dir, delete=False
    ) as f_in:
        f_in.write(source_code)
        tmp_in = f_in.name
    tmp_out = tmp_in + ".out.py"
    try:
        rename_and_chain(
            tmp_in, tmp_out,
            new_input_name, new_output_name,
            prefer_type=prefer_type,
            new_method_name=new_method_name,
        )
        with open(tmp_out, "r") as f:
            return f.read()
    finally:
        for p in (tmp_in, tmp_out):
            if os.path.exists(p):
                os.remove(p)


class PythonTraceTask:
    """A single task in a Python-trace chain.

    Randomly selects a compatible leetcode algorithm, executes it on the
    chained input (plus randomly-generated static inputs), applies bounding,
    and generates a human-readable "trace this code" prompt.

    Parameters
    ----------
    task_ind : int
        1-based task index in the chain.
    inp : dict
        ``{"chained": {"chain_name": str, "value": int | float | list}}``.
    scalar_max_mag : int
        Modulo for bounding scalar outputs and list elements.
    list_len_max : int
        Maximum allowed list length for outputs.
    algos_file : str | None
        Path to ``leetcode_candidate_algos.txt``.  Defaults to the file
        next to this module.
    rand_static_val : Callable[[str], int | list[int]]
        Called as ``rand_static_val(type_hint)`` where *type_hint* is
        ``'list'`` or ``'scalar'``.  Returns a random value suitable for
        a non-chained parameter.  Constructed and passed in by the caller
        (e.g. ``generate_chain.py``) so the class stays range-agnostic.
    timeout : int
        Per-function execution timeout in seconds.
    """

    def __init__(
        self,
        task_ind,
        inp,
        scalar_max_mag,
        list_len_max,
        rand_static_val,
        algos_file=None,
        timeout=5,
        force_output_kind=None,
        func_index=None,
        static_attempts_per_func=20,
    ):
        self.task_ind = task_ind
        self.inp = inp
        self.scalar_max_mag = scalar_max_mag
        self.list_len_max = list_len_max
        self.rand_static_val = rand_static_val
        self.timeout = timeout
        self.force_output_kind = force_output_kind
        self.func_index = func_index
        self.static_attempts_per_func = static_attempts_per_func

        # Resolve algos file
        if algos_file is None:
            algos_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "leetcode_candidate_algos.txt",
            )
        self._all_funcs = load_candidate_algos(algos_file)

        # Populated by gen_params
        self.selected_func = None   # function info dict
        self.chained_param = None   # original param name receiving the chain
        self.static_values = {}     # {param_name: value} for non-chained params
        self.name = None            # selected function name
        self._raw_result = None
        self._prefer_type = None    # 'list' or 'scalar'

        # Populated lazily by renamed_func_code
        self._renamed_func_code = None

        self.gen_params()

        self.out = self.get_output()
        if self.out is None:
            raise ValueError(f"PythonTraceTask {self.task_ind} returned None.")

        self.prompt = self.gen_prompt()

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _get_input_type(self):
        val = self.inp['chained']['value']
        return 'list' if isinstance(val, list) else 'scalar'

    def _get_compatible_funcs(self):
        return [self._all_funcs[index] for index in self._get_compatible_func_indices()]

    def _get_compatible_func_indices(self):
        input_type = self._get_input_type()
        indices = []
        for index, func in enumerate(self._all_funcs):
            if input_type == 'list' and not func['has_list_in']:
                continue
            if input_type != 'list' and not func['has_scalar_in']:
                continue
            if self.force_output_kind == 'list' and not func['has_list_out']:
                continue
            if self.force_output_kind == 'scalar' and func['has_list_out']:
                continue
            indices.append(index)
            
        return indices

    def _find_chained_param(self, func_info):
        """Pick the parameter that should receive the chained input."""
        input_type = self._get_input_type()
        for pname, ptype in func_info['params']:
            if ptype == input_type:
                return pname
        # Fallback: first param
        return func_info['params'][0][0] if func_info['params'] else None

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def gen_params(self):
        if hasattr(self, 'func_index') and self.func_index is not None:
            candidate_indices = [self.func_index]
            if not self._all_funcs[self.func_index]:
                raise ValueError(f"Function at index {self.func_index} is invalid")
        else:
            candidate_indices = self._get_compatible_func_indices()
            if not candidate_indices:
                raise ValueError(
                    f"No compatible functions for input type "
                    f"{self._get_input_type()}"
                )
            random.shuffle(candidate_indices)

        self._prefer_type = self._get_input_type()
        chained_value = self.inp['chained']['value']
        if self._prefer_type == 'list':
            adapted_chained_value = adapt_list_compute(chained_value, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max, to_int=True)
        else:
            adapted_chained_value = adapt_scalar_compute(chained_value, mod_value=self.scalar_max_mag, to_int=True)

        attempts_per_func = max(1, int(self.static_attempts_per_func))

        for func_index in candidate_indices:
            func_info = self._all_funcs[func_index]
            chained_param = self._find_chained_param(func_info)
            if chained_param is None:
                continue

            for _ in range(attempts_per_func):
                try:
                    task_inputs = {}
                    static_vals = {}

                    for pname, ptype in func_info['params']:
                        if pname == chained_param:
                            task_inputs[pname] = adapted_chained_value
                        else:
                            val = self.rand_static_val(ptype)
                            task_inputs[pname] = val
                            static_vals[pname] = val
                    result = execute_function(
                        func_info['code'], func_info['name'],
                        task_inputs, self.timeout,
                    )

                    if result is None:
                        continue
                    if isinstance(result, (list, tuple)) and not result:
                        continue

                    self.selected_func = func_info
                    self._raw_result = result
                
                    bounded_out = self.get_output()
                    if isinstance(bounded_out, list) and not bounded_out:
                        continue
                    if isinstance(bounded_out, (int, float)) and bounded_out in [0, 1, -1]:
                        continue

                    # Success
                    self.func_index = func_index
                    self.chained_param = chained_param
                    self.static_values = static_vals
                    self.name = func_info['name']
                    return

                except Exception:
                    continue

        raise ValueError(
            f"Could not find a valid function for task {self.task_ind}"
        )

    def get_output(self):
        result = self._raw_result

        if self.selected_func['has_list_out']:
            result = list(result)
            return result
        else:
            return result

    # ------------------------------------------------------------------
    # Code transformation
    # ------------------------------------------------------------------

    @property
    def renamed_func_code(self):
        """Standalone function with renamed input, output, and method name."""
        if self._renamed_func_code is not None:
            return self._renamed_func_code

        idx = self.task_ind
        method_name = f"task_{idx:d}"
        
        if self._prefer_type == 'list':
            adapted_inp_name = f"list_{idx:d}"
        else:
            adapted_inp_name = f"val_{idx:d}"

        out_name = f"task_{idx:d}_out"

        transformed = _rename_code(
            self.selected_func['code'],
            new_input_name=adapted_inp_name,
            new_output_name=out_name,
            prefer_type=self._prefer_type,
            new_method_name=method_name,
        )

        # Strip class wrapper and de-indent the method into a top-level function
        lines = transformed.split('\n')
        func_lines = []
        for line in lines:
            if line.startswith("class Solution:"):
                continue
            if line.startswith(f"    def {method_name}"):
                line = line.replace("(self, ", "(").replace("(self)", "()")
                func_lines.append(line[4:])
            elif line.startswith("    "):
                func_lines.append(line[4:])
            elif not line.strip():
                func_lines.append(line)

        self._renamed_func_code = "\n".join(func_lines).strip() + "\n"
        return self._renamed_func_code

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def gen_prompt(self):
        idx = self.task_ind
        method_name = f"task_{idx:d}"
        out_var_name = f"task_{idx:d}_out"

        func_code = self.renamed_func_code

        # Build parameter mapping strings for the prompt
        param_mappings = []
        for pname, _ptype in self.selected_func['params']:
            if pname == self.chained_param:
                continue
            static_var = f"task_{idx:d}_{pname}_static"
            param_mappings.append(f"{pname} = {static_var}")

        inputs_str = ", ".join(param_mappings)
        inputs_clause = f" with inputs {inputs_str}" if inputs_str else ""

        # Bounding instructions
        bounding_extra = ""
        if self._prefer_type == 'list':
            adapted_inp_name = f"list_{idx:d}"
            bounding_extra = " " + adapt_list_prompt(self.inp['chained']['chain_name'], adapted_inp_name, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max, to_int=True).strip()
        else:
            adapted_inp_name = f"val_{idx:d}"
            bounding_extra = " " + adapt_scalar_prompt(self.inp['chained']['chain_name'], adapted_inp_name, mod_value=self.scalar_max_mag, to_int=True).strip()

        prompt = (
            f"Task {idx:d}:\n"
            f"{bounding_extra.strip() + chr(10) if bounding_extra.strip() else ''}"
            f"Trace the execution of {method_name}{inputs_clause}"
            f" and save the result in {out_var_name}.\n"
            f"```python\n"
            f"{func_code}"
            f"```\n"
        )
        return prompt
