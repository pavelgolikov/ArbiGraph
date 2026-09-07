"""
GSMSymbolicTask wraps GSM-Symbolic templates as framework tasks.

Each template from ml-gsm-symbolic/templates/symbolic/ contains:
  * question_annotated  a question string with {var, default} placeholders,
    followed by #init, #conditions, and #answer sections.
  * answer_annotated  step-by-step answer template (unused, we compute directly).

This task class:
  1. Uses one selected template from the GSM task pool.
  2. Parses the #init / #conditions / #answer DSL.
  3. Binds the chained input to one numeric variable.
  4. Rejection-samples values for the remaining variables.
  5. Checks conditions and computes the final scalar answer.
"""

import json
import math
import os
import random
import re
from fractions import Fraction
from typing import Any, Dict, List, Tuple

from tasks.adapters import (
    AddScalarsAdapter,
    ScalarToScalarAdapter,
)
from tasks.task import Task


SCALAR_MAX_MAG = 100


# ============================================================================
# Lookup tables referenced by sample(...) in the GSM-Symbolic DSL
# ============================================================================

# Each "display" table is a list of (display_text, numeric_value) tuples.
# String-only tables are plain lists of strings.

NAMES = [
    "Liam", "Noah", "Oliver", "James", "Elijah", "William", "Henry", "Lucas",
    "Benjamin", "Theodore", "Jack", "Aiden", "Owen", "Samuel", "Ryan",
    "Emma", "Olivia", "Ava", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Evelyn", "Luna", "Ella", "Scarlett", "Grace", "Lily",
    "Rania", "Jamal", "Haruka", "Mei", "Carlos", "Diego", "Priya", "Aisha",
    "Kenji", "Yuki", "Fatima", "Omar", "Sakura", "Arjun", "Tariq", "Chen",
]

NAMES_MALE = [
    "Liam", "Noah", "Oliver", "James", "Elijah", "William", "Henry", "Lucas",
    "Benjamin", "Theodore", "Jack", "Aiden", "Owen", "Samuel", "Ryan",
    "Jamal", "Carlos", "Diego", "Kenji", "Arjun", "Tariq", "Omar", "Chen",
    "Ethan", "Mason", "Logan", "Alexander", "Daniel", "Matthew", "David",
]

NAMES_FEMALE = [
    "Emma", "Olivia", "Ava", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Evelyn", "Luna", "Ella", "Scarlett", "Grace", "Lily",
    "Rania", "Haruka", "Mei", "Priya", "Aisha", "Yuki", "Fatima", "Sakura",
    "Chloe", "Zoe", "Nora", "Hannah", "Aria", "Riley", "Victoria",
]

FRUITS = [
    "apples", "oranges", "bananas", "strawberries", "grapes", "peaches",
    "pears", "mangoes", "watermelons", "blueberries", "cherries", "lemons",
]

SPORTS = [
    "soccer", "basketball", "tennis", "baseball", "swimming",
    "volleyball", "hockey", "cricket", "golf", "rugby",
]

COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "pink", "white", "black", "brown",
]

CITIES = [
    "New York", "London", "Tokyo", "Paris", "Sydney",
    "Toronto", "Berlin", "Madrid", "Rome", "Seoul",
]

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

CURRENCIES_SYM = ["$", "€", "£", "¥"]

WEIGHTS_MED = ["pound", "kilogram"]

WEIGHTS_SM = ["ounce", "gram"]

LENGTH_LG = ["mile", "kilometer"]

# --- Numeric display tables: list of (display_text, numeric_value) ---

MULTI_TIMES: List[Tuple[str, float]] = [
    ("twice", 2), ("three times", 3), ("four times", 4),
    ("five times", 5), ("six times", 6), ("seven times", 7),
    ("eight times", 8), ("nine times", 9), ("ten times", 10),
]

# "multiple_ice" is the same concept but phrased as "double", "triple", etc.
MULTIPLE_ICE: List[Tuple[str, float]] = [
    ("double", 2), ("triple", 3), ("quadruple", 4),
]

MULTIPLE: List[Tuple[str, float]] = MULTI_TIMES + MULTIPLE_ICE

FRACTION_ALNUM: List[Tuple[str, float]] = [
    ("half", 0.5), ("a third", 1/3), ("a quarter", 0.25),
    ("a fifth", 0.2), ("a sixth", 1/6), ("a tenth", 0.1),
    ("two-thirds", 2/3), ("three-quarters", 0.75),
    ("three-fifths", 0.6), ("two-fifths", 0.4),
]

FRACTION_ALPH = FRACTION_ALNUM  # alias

FRACTIONS: List[Tuple[str, float]] = [
    ("1/2", Fraction(1, 2)), ("1/3", Fraction(1, 3)), ("1/4", Fraction(1, 4)),
    ("1/5", Fraction(1, 5)), ("1/6", Fraction(1, 6)), ("1/8", Fraction(1, 8)),
    ("1/10", Fraction(1, 10)), ("2/3", Fraction(2, 3)), ("3/4", Fraction(3, 4)),
    ("2/5", Fraction(2, 5)), ("3/5", Fraction(3, 5)), ("3/8", Fraction(3, 8)),
]

FRACTION_NUMS: List[Tuple[str, float]] = FRACTIONS  # alias

FRACTION_DECIMALS: List[Tuple[str, float]] = [
    ("0.5", 0.5), ("0.25", 0.25), ("0.75", 0.75), ("0.1", 0.1),
    ("0.2", 0.2), ("0.125", 0.125), ("0.375", 0.375),
    ("0.333", 1/3), ("0.667", 2/3),
]

# Registry mapping DSL names → (is_numeric, values)
# For numeric tables, values are (display, number) tuples.
# For string tables, values are plain strings.
LOOKUP_TABLES: Dict[str, Tuple[bool, list]] = {
    "names":             (False, NAMES),
    "names_male":        (False, NAMES_MALE),
    "names_female":      (False, NAMES_FEMALE),
    "fruits":            (False, FRUITS),
    "sports":            (False, SPORTS),
    "colors":            (False, COLORS),
    "cities":            (False, CITIES),
    "weekdays":          (False, WEEKDAYS),
    "currencies_sym":    (False, CURRENCIES_SYM),
    "weights_med":       (False, WEIGHTS_MED),
    "weights_sm":        (False, WEIGHTS_SM),
    "length_lg":         (False, LENGTH_LG),
    # Numeric display tables
    "multi_times":       (True, MULTI_TIMES),
    "multiple_ice":      (True, MULTIPLE_ICE),
    "multiple":          (True, MULTIPLE),
    "fraction_alnum":    (True, FRACTION_ALNUM),
    "fraction_alph":     (True, FRACTION_ALPH),
    "fractions":         (True, FRACTIONS),
    "fraction_nums":     (True, FRACTION_NUMS),
    "fraction_decimals": (True, FRACTION_DECIMALS),
}


# ============================================================================
# Template parser
# ============================================================================

def _parse_template(question_annotated: str):
    """
    Parse a GSM-Symbolic question_annotated string into:
      question_template, init_lines, condition_lines, answer_expr
    """
    init_idx = question_annotated.find('#init:')
    cond_idx = question_annotated.find('#conditions:')
    ans_idx = question_annotated.find('#answer:')
    
    init_end = len(question_annotated)
    cond_end = len(question_annotated)
    
    if cond_idx != -1:
        init_end = cond_idx
    elif ans_idx != -1:
        init_end = ans_idx
        
    if ans_idx != -1:
        cond_end = ans_idx
        
    question_template = question_annotated[:init_idx].strip() if init_idx != -1 else question_annotated.strip()
    
    init_block = ""
    if init_idx != -1:
        init_block = question_annotated[init_idx + len('#init:'):init_end].strip()
        
    cond_block = ""
    if cond_idx != -1:
        cond_block = question_annotated[cond_idx + len('#conditions:'):cond_end].strip()
        
    answer_expr = ""
    if ans_idx != -1:
        answer_expr = question_annotated[ans_idx + len('#answer:'):].strip()

    init_lines = [l.strip().lstrip("- ") for l in init_block.split("\n") if l.strip().lstrip("- ")]
    cond_lines = [l.strip().lstrip("- ") for l in cond_block.split("\n") if l.strip().lstrip("- ")]

    return question_template, init_lines, cond_lines, answer_expr


def _parse_init_line(line: str):
    """
    Parse a single init line and return a variable spec dict.
    Returns: {
        "names": [str],       # variable name(s)
        "is_numeric": bool,
        "domain_type": str,   # "range", "frange", "numbers_within", "sample_named",
                              # "sample_inline", "sample_multi_named"
        "domain_args": ...,   # depends on type
    }
    """
    line = line.strip()
    if not line:
        return None

    # Check for $ prefix (numeric variable)
    is_numeric = line.startswith("$")
    if is_numeric:
        line = line[1:]

    # Split on '='
    eq_idx = line.index("=")
    lhs = line[:eq_idx].strip()
    rhs = line[eq_idx+1:].strip()

    # Handle multi-assignment: "o1, o2, o4 = sample([...], 3)"
    names = [n.strip() for n in lhs.split(",")]

    spec = {"names": names, "is_numeric": is_numeric}

    # Parse RHS
    if rhs.startswith("range("):
        args_str = rhs[len("range("):-1]
        args = [int(a.strip()) for a in args_str.split(",")]
        spec["domain_type"] = "range"
        spec["domain_args"] = args
    elif rhs.startswith("frange("):
        args_str = rhs[len("frange("):-1]
        args = [float(a.strip()) for a in args_str.split(",")]
        spec["domain_type"] = "frange"
        spec["domain_args"] = args
    elif rhs.startswith("numbers_within("):
        args_str = rhs[len("numbers_within("):-1]
        args = [int(a.strip()) for a in args_str.split(",")]
        spec["domain_type"] = "numbers_within"
        spec["domain_args"] = args
    elif rhs.startswith("sample_sequential("):
        # e.g. sample_sequential(weekdays, 2)
        inner = rhs[len("sample_sequential("):-1].strip()
        parts = [p.strip() for p in inner.split(",")]
        list_name = parts[0]
        count = int(parts[1]) if len(parts) > 1 else 1
        spec["domain_type"] = "sample_sequential"
        spec["domain_args"] = {"list_name": list_name, "count": count}
    elif rhs.startswith("sample("):
        inner = rhs[len("sample("):-1].strip()
        if inner.startswith("["):
            # Inline list
            items = re.findall(r'"([^"]*)"', inner) or re.findall(r"'([^']*)'", inner)
            # Check for count argument after the list
            count_match = re.search(r'\]\s*,\s*(\d+)', inner)
            count = int(count_match.group(1)) if count_match else 1
            spec["domain_type"] = "sample_inline"
            spec["domain_args"] = {"items": items, "count": count}
        else:
            # Named list(s), possibly with + concatenation and optional count
            # e.g. "names, 2" or "multiple_ice+multi_times" or "names"
            # Extract trailing count: check if last comma-separated token is a digit
            count = 1
            count_match = re.match(r'^(.+),\s*(\d+)$', inner)
            if count_match:
                inner_no_count = count_match.group(1).strip()
                count = int(count_match.group(2))
            else:
                inner_no_count = inner
            list_names = [n.strip() for n in re.split(r'\s*\+\s*', inner_no_count)]
            spec["domain_type"] = "sample_named"
            spec["domain_args"] = {"list_names": list_names, "count": count}
    elif rhs.startswith("fix_floats(np.arange("):
        args_str = rhs[len("fix_floats(np.arange("):].replace(")", "")
        args = [float(a.strip()) for a in args_str.split(",")]
        spec["domain_type"] = "frange"
        spec["domain_args"] = args
    else:
        # Fallback: treat as literal
        spec["domain_type"] = "literal"
        spec["domain_args"] = rhs

    return spec


def _resolve_named_lists(list_names: List[str], is_numeric: bool):
    """Resolve and concatenate named lookup tables."""
    combined = []
    for name in list_names:
        name = name.strip()
        if name not in LOOKUP_TABLES:
            raise RuntimeError(
                f"Unknown lookup table '{name}'. "
                f"Available tables: {sorted(LOOKUP_TABLES.keys())}"
            )
        _is_num, values = LOOKUP_TABLES[name]
        combined.extend(values)
    return combined


def _sample_from_domain(spec: dict, rng: random.Random) -> Any:
    """Sample a single value from a variable's domain spec."""
    dt = spec["domain_type"]
    args = spec["domain_args"]

    if dt == "range":
        return rng.choice(range(*args))
    elif dt == "frange":
        start, stop, step = args[0], args[1], args[2] if len(args) > 2 else 1.0
        values = []
        v = start
        while v < stop:
            values.append(round(v, 6))
            v += step
        return rng.choice(values) if values else start
    elif dt == "numbers_within":
        lo, hi = args[0], args[1]
        return rng.randint(lo, hi)
    elif dt == "sample_named":
        list_names = args["list_names"]
        count = args["count"]
        pool = _resolve_named_lists(list_names, spec["is_numeric"])
        if not pool:
            raise RuntimeError(
                f"Empty pool after resolving named lists {list_names} "
                f"for variable(s) {spec['names']}"
            )
        if count > 1:
            if len(pool) < count:
                raise RuntimeError(
                    f"Pool from {list_names} has only {len(pool)} items "
                    f"but {count} distinct samples requested for {spec['names']}"
                )
            return rng.sample(pool, count)
        item = rng.choice(pool)
        return item  # (display, value) for numeric, or string
    elif dt == "sample_sequential":
        list_name = args["list_name"]
        count = args["count"]
        if list_name not in LOOKUP_TABLES:
            raise RuntimeError(
                f"Unknown lookup table '{list_name}' in sample_sequential "
                f"for variable(s) {spec['names']}. "
                f"Available tables: {sorted(LOOKUP_TABLES.keys())}"
            )
        _is_num, pool = LOOKUP_TABLES[list_name]
        # Pick a random starting index so that `count` consecutive items fit
        max_start = len(pool) - count
        if max_start < 0:
            raise RuntimeError(
                f"Table '{list_name}' has only {len(pool)} items "
                f"but {count} consecutive items requested for {spec['names']}"
            )
        start = rng.randint(0, max_start)
        return pool[start:start + count]
    elif dt == "sample_inline":
        items = args["items"]
        count = args["count"]
        if not items:
            raise RuntimeError(
                f"Empty inline item list for variable(s) {spec['names']}"
            )
        if count > 1:
            if len(items) < count:
                raise RuntimeError(
                    f"Inline list has only {len(items)} items "
                    f"but {count} distinct samples requested for {spec['names']}"
                )
            return rng.sample(items, count)
        return rng.choice(items)
    elif dt == "literal":
        try:
            return int(args)
        except ValueError:
            try:
                return float(args)
            except ValueError:
                return args
    raise RuntimeError(
        f"Unrecognized domain type '{dt}' for variable(s) {spec['names']}"
    )


# ============================================================================
# Condition / answer evaluator
# ============================================================================

class _SmartInt(int):
    def __getitem__(self, idx):
        return self


class _SmartFloat(float):
    def __getitem__(self, idx):
        return self

def _build_eval_namespace(var_values: dict) -> dict:
    """Build a namespace dict for evaluating conditions and answer expressions."""
    ns = {}
    for name, val in var_values.items():
        # For numeric display vars, extract the numeric value
        if isinstance(val, tuple) and len(val) == 2:
            num = val[1]
            if isinstance(num, int):
                ns[name] = _SmartInt(num)
            else:
                ns[name] = _SmartFloat(num)
        else:
            ns[name] = val

    # Add helper functions used in conditions
    ns["round"] = round
    ns["divides"] = lambda a, b: (b != 0) and (a % b == 0)
    ns["is_int"] = lambda x: isinstance(x, int) or (isinstance(x, float) and x == int(x))
    ns["Fraction"] = Fraction
    ns["int"] = int
    ns["abs"] = abs
    ns["min"] = min
    ns["max"] = max
    ns["math"] = math
    return ns


def _eval_conditions(cond_lines: List[str], var_values: dict) -> bool:
    """Evaluate all condition expressions. Returns True if all pass."""
    ns = _build_eval_namespace(var_values)
    for cond in cond_lines:
        cond = cond.strip()
        if not cond:
            continue
        try:
            if not eval(cond, {"__builtins__": {}}, ns):
                return False
        except Exception:
            return False
    return True


def _eval_answer(answer_expr: str, var_values: dict):
    """Evaluate the answer expression and return the result."""
    ns = _build_eval_namespace(var_values)
    try:
        result = eval(answer_expr, {"__builtins__": {}}, ns)
        # Convert to int if it's a clean integer
        if isinstance(result, float) and result == int(result):
            return int(result)
        if isinstance(result, Fraction):
            if result.denominator == 1:
                return int(result.numerator)
            return float(result)
        return result
    except Exception as e:
        raise ValueError(f"Failed to evaluate answer expression '{answer_expr}': {e}")


# ============================================================================
# Question rendering
# ============================================================================

def _render_question(template: str, var_values: dict) -> str:
    """
    Replace all {var, default} placeholders in the question template.
    For numeric display vars (tuples), use the display text.
    For plain numbers, use str(number).
    For strings, use the string directly.
    """
    def _display(val):
        if isinstance(val, tuple) and len(val) == 2:
            return str(val[0])  # display text
        if isinstance(val, float):
            if val == int(val):
                return str(int(val))
            return str(val)
        return str(val)

    def replacer(m):
        var_name = m.group(1).strip()
        if var_name in var_values:
            return _display(var_values[var_name])
        # Fall back to default value
        default = m.group(2).strip() if m.group(2) else var_name
        return default

    # Match {var, default} or {var}
    rendered = re.sub(r'\{(\w+)\s*,\s*([^}]*)\}', replacer, template)
    rendered = re.sub(r'\{(\w+)\}', replacer, rendered)
    return rendered


# ============================================================================
# Template loader
# ============================================================================

_TEMPLATES_CACHE: dict[str, List[dict]] = {}


TARGET_TEMPLATE_FILES = (
    "0000.json", "0003.json", "0004.json", "0007.json", "0010.json", "0015.json",
    "0016.json", "0018.json", "0019.json", "0024.json", "0027.json", "0028.json",
    "0032.json", "0034.json", "0036.json", "0039.json", "0042.json", "0047.json",
    "0048.json", "0050.json", "0051.json", "0054.json", "0058.json", "0059.json",
    "0060.json", "0062.json", "0066.json", "0067.json", "0071.json", "0072.json",
    "0075.json", "0078.json", "0082.json", "0084.json", "0086.json", "0088.json",
    "0093.json", "0094.json", "0095.json", "0096.json", "0099.json",
)

def load_templates(templates_dir: str) -> List[dict]:
    """Load all GSM-Symbolic template JSON files, overlaying corrected ones."""
    templates_dir = os.path.abspath(templates_dir)
    if templates_dir in _TEMPLATES_CACHE:
        return _TEMPLATES_CACHE[templates_dir]

    corrected_dir = os.path.join(os.path.dirname(__file__), "corrected_templates")
    templates = []
    for fname in sorted(os.listdir(templates_dir)):
        if not fname.endswith(".json"):
            continue
            
        corrected_path = os.path.join(corrected_dir, fname)
        if os.path.exists(corrected_path):
            file_path = corrected_path
        else:
            file_path = os.path.join(templates_dir, fname)
            
        with open(file_path, encoding="utf-8") as f:
            tpl = json.load(f)
        if "question_annotated" in tpl:
            tpl["_filename"] = fname
            templates.append(tpl)

    _TEMPLATES_CACHE[templates_dir] = templates
    return templates


def get_default_templates_dir() -> str:
    """Return the default path to the GSM-Symbolic templates directory.

    The checkout lives outside the repository. ARBIGRAPH_DEPS overrides the
    directory holding it; the default is the directory containing the repo.
    """
    deps_dir = os.environ.get(
        "ARBIGRAPH_DEPS",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
    )
    return os.path.abspath(os.path.join(deps_dir, "ml-gsm-symbolic", "templates", "symbolic"))


# ============================================================================
# GSMSymbolicTask
# ============================================================================

class GSMSymbolicTask(Task):
    """
    A Task that wraps one GSM-Symbolic word-problem template.

    Input: scalar.
    Output: scalar answer to the word problem.
    """

    MAX_REJECTION_ATTEMPTS = 200

    def __init__(
        self,
        task: dict[str, Any],
        task_ind: int,
        input_names: list[str],
        input_value: Any,
        scalar_max_mag: int,
    ):
        super().__init__("gsm", task["task_id"], task["task_name"], task["input_type"], task["output_type"])
        self.scalar_max_mag = scalar_max_mag
        self.result_var_name = f"gsm_result_{task_ind:d}"
        self.adapted_input_name = f"val_{task_ind:d}"
        self.var_values = {}
        self.question_template = ""
        self.answer_expr = ""
        self.chained_var_name = None
        self.clamped_add_val = 0
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=self.scalar_max_mag,
            to_int=True,
            needs_abs=True,
        )

        self._bind_input_to_template(task["template"], task["template_filename"], input_value)
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=self.scalar_max_mag,
            to_int=True,
            needs_abs=True,
            add_val=self.clamped_add_val,
        )
        self.out = self.solution_generator(input_value)
        if self.out is None:
            raise ValueError(f"{self.task_name} returned None.")
        self.prompt = self.prompt_generator(task_ind, input_names, f"task_{task_ind:d}_out")

    def _extract_chained_value(self, input_value: Any) -> int:
        return int(self.input_adapter.compute(input_value))

    def _bind_input_to_template(self, template: dict[str, Any], template_filename: str, input_value: Any) -> None:
        qa = template["question_annotated"]
        qt, init_lines, cond_lines, answer_expr = _parse_template(qa)
        specs = []
        for line in init_lines:
            spec = _parse_init_line(line)
            if spec:
                specs.append(spec)

        numeric_specs = [
            spec for spec in specs
            if spec["is_numeric"]
            and spec["domain_type"] in ("range", "numbers_within", "frange")
            and len(spec["names"]) == 1
        ]
        if not numeric_specs:
            raise ValueError(f"{self.task_name} has no bindable scalar variable.")

        # Only a variable the answer actually reads may carry the chained value.
        # Some templates declare a variable that appears solely in a condition --
        # 0054's `total` bounds n1..n3 but is absent from `n1 + (n1+n2) + (n1+n2+n3)`.
        # Binding to one of those silently breaks the chain: the child's answer no
        # longer depends on its parent, so it can be solved without it. Match whole
        # identifiers so `n1` does not match `n10`.
        answer_names = set(re.findall(r"[A-Za-z_]\w*", answer_expr))
        numeric_specs = [spec for spec in numeric_specs if spec["names"][0] in answer_names]
        if not numeric_specs:
            raise ValueError(
                f"{self.task_name} has no bindable scalar variable in its answer "
                f"expression: {answer_expr!r}"
            )

        chained_int = self._extract_chained_value(input_value)

        for bind_spec in numeric_specs:
            result = self._try_sample(
                specs,
                cond_lines,
                answer_expr,
                bind_spec,
                chained_int,
            )
            if result is not None:
                self.question_template = qt
                self.answer_expr = answer_expr
                self.var_values = result
                self.chained_var_name = bind_spec["names"][0]
                self.clamped_add_val = self.var_values[self.chained_var_name] - chained_int
                return

        raise ValueError(
            f"GSMSymbolicTask: Could not find a valid variable assignment "
            f"for chained input value {chained_int} using {template_filename}."
        )

    def _try_sample(
        self,
        specs: list[dict[str, Any]],
        cond_lines: list[str],
        answer_expr: str,
        bind_spec: dict[str, Any],
        chained_int: int,
    ) -> dict[str, Any] | None:
        """
        Try to sample valid values for all variables, with bind_spec's variable
        set to chained_int.  Returns var_values dict on success, None on failure.
        """
        bind_name = bind_spec["names"][0]

        for attempt in range(self.MAX_REJECTION_ATTEMPTS):
            var_values = {}
            use_abs = attempt >= self.MAX_REJECTION_ATTEMPTS // 2

            for spec in specs:
                names = spec["names"]
                if bind_name in names:
                    for name in names:
                        if name == bind_name:
                            val = abs(chained_int) if use_abs else chained_int
                            # Clamp to domain range if possible
                            if spec["domain_type"] == "range":
                                r_args = spec["domain_args"]
                                lo = r_args[0]
                                hi = r_args[1] - 1
                                step = r_args[2] if len(r_args) > 2 else 1
                                # Snap to valid step value
                                if step > 1 and (val - lo) % step != 0:
                                    val = lo + round((val - lo) / step) * step
                                val = max(lo, min(hi, val))
                            elif spec["domain_type"] == "numbers_within":
                                lo, hi = spec["domain_args"]
                                val = max(lo, min(hi, val))
                            var_values[name] = val
                        else:
                            var_values[name] = _sample_from_domain(spec, random)
                else:
                    sampled = _sample_from_domain(spec, random)
                    if len(names) > 1 and isinstance(sampled, list):
                        for i, name in enumerate(names):
                            if i < len(sampled):
                                var_values[name] = sampled[i]
                    else:
                        for name in names:
                            var_values[name] = sampled

            # Check conditions
            if _eval_conditions(cond_lines, var_values):
                # Compute answer
                try:
                    answer = _eval_answer(answer_expr, var_values)
                    if isinstance(answer, (int, float)) and not math.isnan(answer) and not math.isinf(answer):
                        return var_values
                except Exception:
                    continue

        return None

    def solution_generator(self, _input_value: Any) -> Any:
        answer = _eval_answer(self.answer_expr, self.var_values)
        if isinstance(answer, float) and answer == int(answer):
            return int(answer)
        return answer

    def prompt_generator(self, task_number: int, input_names: list[str], output_name: str) -> str:
        input_reference = input_names[0]
        preprocess = self.input_adapter.prompt(input_reference, self.adapted_input_name)
        orig_val = self.var_values.get(self.chained_var_name)
        if self.chained_var_name is not None:
            self.var_values[self.chained_var_name] = self.adapted_input_name
        question = _render_question(self.question_template, self.var_values)
        if self.chained_var_name is not None:
            self.var_values[self.chained_var_name] = orig_val

        return (
            f"Task {task_number:d}:\n"
            f"{preprocess}"
            f"{question}\n"
            f"Compute the final numerical answer. Let {self.result_var_name} be that answer.\n"
            f"Output the result as {output_name} = "
            f'{{"result": {self.result_var_name}}}, '
            f"replacing {self.result_var_name} with its calculated value."
        )


def load_pool() -> list[dict[str, Any]]:
    templates = load_templates(get_default_templates_dir())
    templates_by_filename = {template["_filename"]: template for template in templates}
    tasks = []

    for template_filename in TARGET_TEMPLATE_FILES:
        if template_filename not in templates_by_filename:
            raise ValueError(f"Missing GSM template {template_filename!r}.")
        task_id = int(template_filename.removesuffix(".json"))
        tasks.append({
            "category": "gsm",
            "task_id": task_id,
            "task_name": f"GSM_{template_filename.removesuffix('.json')}",
            "input_type": "scalar",
            "output_type": "scalar",
            "task_implementation": GSMSymbolicTask,
            "template": templates_by_filename[template_filename],
            "template_filename": template_filename,
        })

    return tasks


def make_stage(
    task: dict[str, Any],
    task_number: int,
    input_names: list[str],
    _input: Any,
) -> tuple[Any, str, list[str], Any]:
    adapter_prompt = ""
    if _input is None:
        task_input = random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)
    else:
        task_input = _input

    references = [f'{input_name}["result"]' for input_name in input_names]
    if len(input_names) > 1:
        adapter_name = f"val_{task_number:d}_join"
        join_adapter = AddScalarsAdapter()
        task_input = join_adapter.compute(task_input)
        adapter_prompt = join_adapter.prompt(references, adapter_name)
        references = [adapter_name]

    task_instance = task["task_implementation"](
        task,
        task_number,
        references,
        task_input,
        SCALAR_MAX_MAG,
    )
    prompt = task_instance.prompt.rstrip()
    if adapter_prompt:
        task_header, task_body = prompt.split("\n", 1)
        prompt = f"{task_header}\n{adapter_prompt.rstrip()}\n{task_body}"

    if isinstance(task_instance.out, (int, float)) and (
        not math.isfinite(task_instance.out)
        or task_instance.out in (0, 1, -1)
        or abs(task_instance.out) >= 10**10
    ):
        raise ValueError(f"gsm:{task['task_id']}:{task['task_name']} produced an unusable output.")

    if _input is None:
        input_json = json.dumps({"result": task_input}, ensure_ascii=True)
        prompt = f"Define {input_names[0]} = {input_json}.\n{prompt}"

    return task_instance.out, prompt, [], task_input
