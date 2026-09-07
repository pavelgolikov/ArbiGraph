from __future__ import annotations

import ast
import collections
import copy
import json
import math
import os
import random
import signal
from typing import Any

import libcst as cst

from tasks.adapters import (
    AddScalarsAdapter,
    ListToListAdapter,
    ScalarToScalarAdapter,
    SumListsAdapter,
)
from tasks.task import Task


DEFAULT_ALGOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leetcode_candidate_algos.txt")
LIST_LEN_MAX = 10
SCALAR_MAX_MAG = 100
STATIC_ATTEMPTS = 5
TIMEOUT_SECONDS = 5

# =============================================================================
# Base Class
# =============================================================================

class PythonTask(Task):
    def __init__(
        self,
        task: dict[str, Any],
        task_ind: int,
        input_names: list[str],
        input_value: Any,
        scalar_max_mag: int,
        list_len_max: int,
        num_parents: int = 1,
    ):
        super().__init__("python", task["task_id"], task["task_name"], task["input_type"], task["output_type"])
        self.params = task["func_info"]["params"]
        self.chained_param = task["chained_param"]
        self.scalar_max_mag = scalar_max_mag
        self.list_len_max = list_len_max
        self.num_parents = num_parents
        self.result_var_name = f"python_result_{task_ind:d}"
        self.method_name = f"task_{task_ind:d}"
        self.adapted_input_name = f"list_{task_ind:d}" if self.input_type == "list" else f"val_{task_ind:d}"
        if self.input_type == "list":
            self.input_adapter = ListToListAdapter(
                mod_value=scalar_max_mag,
                list_len_max=list_len_max,
                to_int=True,
                num_parents=self.num_parents,
            )
        else:
            self.input_adapter = ScalarToScalarAdapter(mod_value=scalar_max_mag, to_int=True)
        adapted_input = self.input_adapter.compute(input_value)
        self.func_code = transform_code(
            task["func_info"]["code"],
            task["func_info"]["name"],
            self.chained_param,
            self.adapted_input_name,
            self.result_var_name,
            self.method_name,
            task["func_info"]["return_tuple_index"],
        )
        self.out, self.static_values = self._try_solution(adapted_input)
        self.prompt = self.prompt_generator(task_ind, input_names, f"task_{task_ind:d}_out")

    def _try_solution(self, adapted_input: Any) -> tuple[Any, dict[str, Any]]:
        last_error = None
        for _ in range(STATIC_ATTEMPTS):
            inputs = {self.adapted_input_name: copy.deepcopy(adapted_input)}
            static_values = {}
            for name, param_type in self.params:
                if name == self.chained_param:
                    continue
                if param_type == "list":
                    value = [random.randint(-self.scalar_max_mag, self.scalar_max_mag) for _ in range(self.list_len_max)]
                else:
                    value = random.randint(-self.scalar_max_mag, self.scalar_max_mag)
                inputs[name] = value
                static_values[name] = value
            try:
                output = execute_function(self.func_code, self.method_name, inputs)
            except Exception as exc:
                last_error = exc
                continue
            if output is None or (isinstance(output, (list, tuple)) and not output):
                continue
            return (list(output) if self.output_type == "list" else output), static_values
        raise ValueError(f"Could not execute {self.task_name}: {last_error}")

    def solution_generator(self, input_value: Any) -> Any:
        output = self.out
        return output

    def prompt_generator(self, task_ind: int, input_names: list[str], output_name: str) -> str:
        prompt = self.input_adapter.prompt(input_names[0], self.adapted_input_name)

        for name, value in self.static_values.items():
            prompt += f"Define task_{task_ind:d}_{name}_static = {json.dumps(value, ensure_ascii=True)}.\n"

        mappings = [
            f'{name} = task_{task_ind:d}_{name}_static'
            for name, _param_type in self.params
            if name != self.chained_param
        ]
        inputs_clause = f" with inputs {', '.join(mappings)}" if mappings else ""
        func_code = self.func_code
        if "List[" in func_code:
            func_code = f"from typing import List\n{func_code}"
        prompt += (
            f"Trace the execution of {self.method_name}{inputs_clause}.\n"
            f"```python\n{func_code}```\n"
            f"Let {self.result_var_name} be the value returned by {self.method_name}.\n"
            f"Output the result as {output_name} = "
            f'{{"result": {self.result_var_name}}}, '
            f"replacing {self.result_var_name} with its calculated value."
        )
        return f"Task {task_ind:d}:\n{prompt.rstrip()}"



def load_pool() -> list[dict[str, Any]]:
    tasks = []
    task_id = 0
    for func_info in load_candidate_algos():
        output_type = func_info["output_type"]
        for input_type in ("list", "scalar"):
            chained_param = next((name for name, param_type in func_info["params"] if param_type == input_type), None)
            if chained_param is None:
                continue
            tasks.append({
                "category": "python",
                "task_id": task_id,
                "task_name": f"{func_info['name']}:{input_type}",
                "input_type": input_type,
                "output_type": output_type,
                "task_implementation": PythonTask,
                "func_info": func_info,
                "chained_param": chained_param,
            })
            task_id += 1

    return tasks


def make_stage(
    task: dict[str, Any],
    task_number: int,
    input_names: list[str],
    _input: Any,
) -> tuple[Any, str, list[str], Any]:
    adapter_prompt = ""
    num_parents = 0 if _input is None else len(input_names)
    if _input is None:
        if task["input_type"] == "list":
            task_input = [random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG) for _ in range(LIST_LEN_MAX)]
        else:
            task_input = random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)
    else:
        task_input = _input

    references = [f'{input_name}["result"]' for input_name in input_names]
    if len(input_names) > 1:
        if task["input_type"] == "scalar":
            adapter_name = f"val_{task_number:d}_join"
            join_adapter = AddScalarsAdapter()
        elif task["input_type"] == "list":
            adapter_name = f"list_{task_number:d}_join"
            join_adapter = SumListsAdapter()
        else:
            raise ValueError(f"Cannot join inputs for input_type={task['input_type']!r}.")
        task_input = join_adapter.compute(task_input)
        adapter_prompt = join_adapter.prompt(references, adapter_name)
        references = [adapter_name]

    task_instance = task["task_implementation"](
        task,
        task_number,
        references,
        task_input,
        SCALAR_MAX_MAG,
        LIST_LEN_MAX,
        num_parents,
    )
    prompt = task_instance.prompt.rstrip()
    if adapter_prompt:
        task_header, task_body = prompt.split("\n", 1)
        prompt = f"{task_header}\n{adapter_prompt.rstrip()}\n{task_body}"

    if bad_output(task_instance.out):
        raise ValueError(f"python:{task['task_id']}:{task['task_name']} produced an unusable output.")

    if _input is None:
        input_json = json.dumps({"result": task_input}, ensure_ascii=True)
        prompt = f"Define {input_names[0]} = {input_json}.\n{prompt}"

    return task_instance.out, prompt, [], task_input



# =============================================================================
# Internal Helpers
# =============================================================================

def bad_output(value: Any) -> bool:
    if isinstance(value, list):
        return (
            len(value) <= 1
            or collections.Counter(value).most_common(1)[0][1] > len(value) / 2
            or value == list(range(len(value)))
            or value == list(range(1, len(value) + 1))
        )
    return isinstance(value, (int, float)) and (
        not math.isfinite(value) or value in (0, 1, -1) or abs(value) >= 10**10
    )


def check_type_annotation(node: ast.AST | None) -> str:
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {"list", "List"}:
            return "list"
        if isinstance(node.value, ast.Name) and node.value.id in {"tuple", "Tuple"}:
            if isinstance(node.slice, ast.Tuple):
                return "list" if any(check_type_annotation(item) == "list" for item in node.slice.elts) else "scalar"
    return "scalar"


def extract_function_info(code_str: str) -> dict[str, Any] | None:
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = node.args.posonlyargs + node.args.args
            if args and args[0].arg == "self":
                args = args[1:]

            params = [(arg.arg, check_type_annotation(arg.annotation)) for arg in args]
            output_type = check_type_annotation(node.returns)
            return_tuple_index = -1
            if (
                isinstance(node.returns, ast.Subscript)
                and isinstance(node.returns.value, ast.Name)
                and node.returns.value.id in {"tuple", "Tuple"}
                and isinstance(node.returns.slice, ast.Tuple)
            ):
                return_tuple_index = 0
                for index, item in enumerate(node.returns.slice.elts):
                    if check_type_annotation(item) == "list":
                        return_tuple_index = index
                        break

            return {
                "name": node.name,
                "params": params,
                "output_type": output_type,
                "return_tuple_index": return_tuple_index,
                "code": code_str,
            }
    return None


def load_candidate_algos(filepath: str = DEFAULT_ALGOS_FILE) -> list[dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as handle:
        content = handle.read()

    funcs = []
    for block in content.split("-" * 80):
        if not block.strip():
            continue
        info = extract_function_info("\n".join(block.strip().split("\n")[1:]))
        if info is not None:
            funcs.append(info)
    return funcs


def execute_function(code_str: str, func_name: str, inputs: dict[str, Any], timeout: int = TIMEOUT_SECONDS) -> Any:
    namespace: dict[str, Any] = {}
    exec("from typing import *", namespace)
    exec("import math", namespace)
    exec(code_str, namespace)
    func = namespace[func_name]

    def signal_handler(_signum, _frame):
        raise TimeoutError("Timed out.")

    previous_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout)
    try:
        return func(**copy.deepcopy(inputs))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


class FunctionTransformer(cst.CSTTransformer):
    def __init__(
        self,
        old_function_name: str,
        new_function_name: str,
        old_input_name: str,
        new_input_name: str,
        output_name: str,
        return_tuple_index: int,
    ):
        self.old_function_name = old_function_name
        self.new_function_name = new_function_name
        self.old_input_name = old_input_name
        self.new_input_name = new_input_name
        self.output_name = output_name
        self.return_tuple_index = return_tuple_index
        self.in_function = False
        self.done = False
        self.nested_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self.in_function:
            self.nested_depth += 1
            return False
        if node.name.value == self.old_function_name and self.done:
            return False
        if node.name.value == self.old_function_name:
            self.in_function = True
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
        params = [param for param in updated_node.params.params if param.name.value != "self"]
        updated_node = updated_node.with_changes(params=updated_node.params.with_changes(params=params))
        if self.nested_depth:
            self.nested_depth -= 1
            return updated_node
        if original_node.name.value == self.old_function_name:
            if self.done and not self.in_function:
                return cst.RemoveFromParent()
            self.in_function = False
            self.done = True
            return updated_node.with_changes(name=cst.Name(self.new_function_name))
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name):
        if self.in_function and not self.nested_depth and updated_node.value == self.old_input_name:
            return updated_node.with_changes(value=self.new_input_name)
        return updated_node

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute):
        if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "self":
            if updated_node.attr.value == self.old_function_name:
                return cst.Name(self.new_function_name)
            return cst.Name(updated_node.attr.value)
        return updated_node

    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine):
        if not self.in_function or self.nested_depth:
            return updated_node

        for statement in updated_node.body:
            if isinstance(statement, cst.Return) and statement.value is not None:
                value = statement.value
                if self.return_tuple_index != -1:
                    value = cst.Subscript(
                        value=cst.Name("_res"),
                        slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer(str(self.return_tuple_index))))],
                    )
                    return cst.FlattenSentinel([
                        cst.SimpleStatementLine(body=[cst.Assign(targets=[cst.AssignTarget(cst.Name("_res"))], value=statement.value)]),
                        cst.SimpleStatementLine(body=[cst.Assign(targets=[cst.AssignTarget(cst.Name(self.output_name))], value=value)]),
                        cst.SimpleStatementLine(body=[cst.Return(value=cst.Name(self.output_name))]),
                    ])
                return cst.FlattenSentinel([
                    cst.SimpleStatementLine(body=[cst.Assign(targets=[cst.AssignTarget(cst.Name(self.output_name))], value=value)]),
                    cst.SimpleStatementLine(body=[cst.Return(value=cst.Name(self.output_name))]),
                ])
        return updated_node


def transform_code(
    source_code: str,
    old_function_name: str,
    old_input_name: str,
    new_input_name: str,
    new_output_name: str,
    new_method_name: str,
    return_tuple_index: int,
) -> str:
    tree = cst.parse_module(source_code)
    code = tree.visit(FunctionTransformer(
        old_function_name,
        new_method_name,
        old_input_name,
        new_input_name,
        new_output_name,
        return_tuple_index,
    )).code
    lines = []
    for line in code.splitlines():
        if line.startswith("class Solution:"):
            continue
        if line.startswith("    "):
            lines.append(line[4:])
        elif line.strip():
            lines.append(line)
    return "\n".join(lines).strip() + "\n"
