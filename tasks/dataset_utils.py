"""Small shared pieces used by dataset generators."""

from __future__ import annotations

import collections
import math
import os
import random
import sys
from typing import Any


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.gsm_task.gsm_symbolic_task import GSMSymbolicTask, get_default_templates_dir, load_templates
from tasks.math_task.math_task import MATH_TASK_SPECS, _build_math_input
from tasks.python_task.python_task import PythonTraceTask, load_candidate_algos


LIST_LEN_MAX = 10
SCALAR_MAX_MAG = 100
PYTHON_STATIC_ATTEMPTS_PER_FUNC = 200
PREAMBLE = (
    "Solve all tasks below. Some tasks depend on outputs from earlier tasks. "
    "Put every answer in its own \\boxed{} block with the exact requested output name, "
    "for example \\boxed{task_1_out = ...}."
)


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


def load_pool(family: str) -> list[dict[str, Any]]:
    if family == "math":
        specs = sorted(
            MATH_TASK_SPECS,
            key=lambda spec: (spec.input_kind, spec.output_kind, spec.cls.__name__),
        )
        return [
            {
                "family": "math",
                "task_id": task_id,
                "task_name": spec.cls.__name__,
                "input_kinds": [spec.input_kind],
                "output_kind": spec.output_kind,
                "spec": spec,
            }
            for task_id, spec in enumerate(specs)
        ]

    if family == "python":
        algos_file = os.path.join(REPO_ROOT, "tasks", "python_task", "leetcode_candidate_algos.txt")
        return [
            {
                "family": "python",
                "task_id": task_id,
                "task_name": func["name"],
                "input_kinds": [
                    kind
                    for kind, enabled in (
                        ("list", func["has_list_in"]),
                        ("scalar", func["has_scalar_in"]),
                    )
                    if enabled
                ],
                "output_kind": "list" if func["has_list_out"] else "scalar",
                "algos_file": algos_file,
            }
            for task_id, func in enumerate(load_candidate_algos(algos_file))
        ]

    if family == "gsm":
        return [
            {
                "family": "gsm",
                "task_id": task_id,
                "task_name": f"GSM_{template['_filename']}",
                "input_kinds": ["scalar"],
                "output_kind": "scalar",
                "template_index": task_id,
            }
            for task_id, template in enumerate(load_templates(get_default_templates_dir()))
        ]

    raise ValueError(f"Unknown family {family!r}. Use exactly 'math', 'python', or 'gsm'.")


def make_stage(
    family: str,
    task: dict[str, Any],
    task_number: int,
    input_name: str,
    value: Any,
) -> tuple[Any, str, list[str]]:
    if family == "math":
        spec = task["spec"]
        inp = _build_math_input(spec.adapter, input_name, value, LIST_LEN_MAX)
        obj = spec.cls(task_number, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
        output = obj.out
        prompt = obj.prompt.rstrip()
        static_inputs = []
    elif family == "python":
        obj = PythonTraceTask(
            task_ind=task_number,
            inp={"chained": {"chain_name": input_name, "value": value}},
            scalar_max_mag=SCALAR_MAX_MAG,
            list_len_max=LIST_LEN_MAX,
            rand_static_val=lambda kind: (
                [random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG) for _ in range(LIST_LEN_MAX)]
                if kind == "list"
                else random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)
            ),
            algos_file=task["algos_file"],
            force_output_kind=task["output_kind"],
            func_index=task["task_id"],
            static_attempts_per_func=PYTHON_STATIC_ATTEMPTS_PER_FUNC,
        )
        output = obj.out
        prompt = obj.prompt.rstrip()
        static_inputs = [
            f"task_{task_number}_{name}_static = {static_value!r}"
            for name, static_value in obj.static_values.items()
        ]
    elif family == "gsm":
        obj = GSMSymbolicTask(
            task_ind=task_number,
            inp={f"task_{task_number}_input": {"chain_name": input_name, "value": value}},
            scalar_max_mag=SCALAR_MAX_MAG,
            list_len_max=LIST_LEN_MAX,
            template_index=task["template_index"],
        )
        output = obj.out
        prompt = obj.prompt.rstrip()
        static_inputs = []
    else:
        raise ValueError(f"Unknown family {family!r}. Use exactly 'math', 'python', or 'gsm'.")

    if bad_output(output):
        raise ValueError(f"{family}:{task['task_id']}:{task['task_name']} produced an unusable output.")

    return output, prompt, static_inputs
