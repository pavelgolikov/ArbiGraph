"""
GSMSymbolicTask  wraps GSM-Symbolic templates as a MathTask.

Each template from ml-gsm-symbolic/templates/symbolic/ contains:
  * question_annotated  a question string with {var, default} placeholders,
    followed by #init, #conditions, and #answer sections.
  * answer_annotated  step-by-step answer template (unused, we compute directly).

This task class:
  1. Randomly selects a template.
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
from typing import Any, Dict, List, Optional, Tuple

from tasks.math_task.math_task import MathTask
from tasks.math_task.math_helpers import (
    adapt_scalar_compute,
    adapt_list_compute,
    adapt_scalar_prompt,
    adapt_element,
)


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
    def __getitem__(self, idx): return self
class _SmartFloat(float):
    def __getitem__(self, idx): return self

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

_TEMPLATES_CACHE: Optional[List[dict]] = None

def load_templates(templates_dir: str) -> List[dict]:
    """Load all GSM-Symbolic template JSON files, overlaying corrected ones."""
    global _TEMPLATES_CACHE
    if _TEMPLATES_CACHE is not None:
        return _TEMPLATES_CACHE

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
            
        with open(file_path) as f:
            tpl = json.load(f)
        if "question_annotated" in tpl:
            tpl["_filename"] = fname
            templates.append(tpl)

    _TEMPLATES_CACHE = templates
    return templates


def get_default_templates_dir() -> str:
    """Return the default path to the GSM-Symbolic templates directory."""
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "ml-gsm-symbolic", "templates", "symbolic"
    ))


# ============================================================================
# GSMSymbolicTask
# ============================================================================

class GSMSymbolicTask(MathTask):
    """
    A MathTask that wraps GSM-Symbolic word-problem templates.

    Input:  scalar or list (if list, uses 8th element, i.e. index 7).
    Output: scalar integer (the answer to the word problem).
    """

    MAX_REJECTION_ATTEMPTS = 200

    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max,
                 templates_dir=None, template_index=None):
        self.templates_dir = templates_dir or get_default_templates_dir()
        self.template_index = template_index  # None = random
        self.selected_template = None
        self.var_values = {}      # all variable assignments
        self.var_specs = []       # parsed init specs
        self.question_template = ""
        self.cond_lines = []
        self.answer_expr = ""
        self.chained_var_name = None  # which numeric var is bound to input
        self._rng = random.Random()

        super().__init__('gsm_symbolic', task_ind, inp, scalar_max_mag, list_len_max)

    def _extract_chained_value(self) -> int:
        """
        Extract a single integer from the chained input.
        Scalar -> use directly.  List -> take 8th element (index 7).
        """
        # The chained value is stored under the first key in self.inp
        first_key = next(iter(self.inp))
        value = self.inp[first_key]["value"]

        if isinstance(value, list):
            idx = min(7, len(value) - 1)
            raw = value[idx]
        else:
            raw = value

        # Convert to int via adapt_scalar_compute
        return int(adapt_scalar_compute(raw, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag))

    def gen_params(self):
        """Select a template, bind chained input, and rejection-sample other vars."""
        templates = load_templates(self.templates_dir)
        if not templates:
            raise ValueError(f"No templates found in {self.templates_dir}")

        if self.template_index is not None:
            if not (0 <= self.template_index < len(templates)):
                raise ValueError(
                    f"Template index {self.template_index} is out of bounds (0-{len(templates)-1})"
                )
            template_order = [self.template_index]
        else:
            template_order = list(range(len(templates)))
            self._rng.shuffle(template_order)

        chained_int = self._extract_chained_value()

        for ti in template_order:
            tpl = templates[ti]
            qa = tpl["question_annotated"]
            qt, init_lines, cond_lines, answer_expr = _parse_template(qa)

            # Parse init specs
            specs = []
            for line in init_lines:
                s = _parse_init_line(line)
                if s:
                    specs.append(s)

            # Find numeric variables to bind the chained input to
            numeric_specs = [s for s in specs if s["is_numeric"]
                             and s["domain_type"] in ("range", "numbers_within", "frange")
                             and len(s["names"]) == 1]

            if not numeric_specs:
                continue  # no numeric range vars to bind to

            # Try binding to each numeric var
            for bind_spec in numeric_specs:
                result = self._try_sample(
                    qt, specs, cond_lines, answer_expr,
                    bind_spec, chained_int
                )
                if result is not None:
                    self.selected_template = tpl
                    self.question_template = qt
                    self.var_specs = specs
                    self.cond_lines = cond_lines
                    self.answer_expr = answer_expr
                    self.var_values = result
                    self.chained_var_name = bind_spec["names"][0]
                    self.clamped_add_val = self.var_values[self.chained_var_name] - chained_int
                    return

        if self.template_index is not None:
            raise ValueError(
                f"GSMSymbolicTask: Could not find a valid variable assignment "
                f"for chained input value {chained_int} using template index {self.template_index}."
            )
        else:
            raise ValueError(
                f"GSMSymbolicTask: Could not find a valid template/variable assignment "
                f"for chained input value {chained_int} after trying all templates."
            )

    def _try_sample(self, qt, specs, cond_lines, answer_expr,
                    bind_spec, chained_int) -> Optional[dict]:
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
                            var_values[name] = _sample_from_domain(spec, self._rng)
                else:
                    sampled = _sample_from_domain(spec, self._rng)
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

    def get_output(self):
        """Compute and return the scalar answer."""
        answer = _eval_answer(self.answer_expr, self.var_values)
        if isinstance(answer, float) and answer == int(answer):
            return int(answer)
        return answer

    def gen_prompt(self) -> str:
        """Generate the question prompt with variables substituted."""
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"

        # Build the chained input variable reference
        first_key = next(iter(self.inp))
        chain_name = self.inp[first_key]["chain_name"]
        value = self.inp[first_key]["value"]

        # Preprocessing instruction for extracting the chained value
        preprocess = ""
        add_val = getattr(self, "clamped_add_val", 0)
        if isinstance(value, list):
            idx = min(7, len(value) - 1)
            ops = adapt_element(needs_abs=True, mod_value=self.scalar_max_mag, to_int=True, add_val=add_val)
            ops_capitalized = ops[0].upper() + ops[1:] if ops else ""
            preprocess = (
                f"Let {adapted_inp_name} be the result of preprocessing {chain_name} as follows - "
                f"take the element at index {idx} (0-indexed) from {chain_name}. "
                f"{ops_capitalized}.\n"
            )
        else:
            preprocess = adapt_scalar_prompt(chain_name, adapted_inp_name, needs_abs=True, mod_value=self.scalar_max_mag, to_int=True, add_val=add_val)

        # Render the question with substituted variables
        orig_val = self.var_values.get(self.chained_var_name)
        if self.chained_var_name is not None:
            self.var_values[self.chained_var_name] = adapted_inp_name
        question = _render_question(self.question_template, self.var_values)
        if self.chained_var_name is not None:
            self.var_values[self.chained_var_name] = orig_val

        prompt = f"Task {self.task_ind:d}:\n"
        prompt += preprocess
        prompt += question + "\n"
        prompt += f"Save the final numerical answer in {out_var_name}.\n"
        return prompt
