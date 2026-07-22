"""Generate cross-family dependency-chain datasets.

Example:
  python generate_chain.py --task-types math python math --num_samples_per_task 1 --output /tmp/chain.json
"""

import argparse
import ast
import collections
import datetime
import json
import math
import os
import random
import re
import sys
from fractions import Fraction

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.dont_write_bytecode = True

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.gsm_task.gsm_symbolic_task import GSMSymbolicTask, get_default_templates_dir, load_templates
from tasks.math_task.math_task import MATH_TASK_SPECS, _build_math_input
from tasks.python_task.python_task import PythonTraceTask, load_candidate_algos


LIST_LEN_MAX = 10
SCALAR_MAX_MAG = 100
PYTHON_STATIC_ATTEMPTS_PER_FUNC = 20
FAMILIES = {"math", "python", "gsm"}
TARGET_TEMPLATES = (
    "0000.json", "0003.json", "0004.json", "0007.json", "0010.json", "0015.json",
    "0016.json", "0018.json", "0019.json", "0024.json", "0027.json", "0028.json",
    "0032.json", "0034.json", "0036.json", "0039.json", "0042.json", "0047.json",
    "0048.json", "0050.json", "0051.json", "0054.json", "0058.json", "0059.json",
    "0060.json", "0062.json", "0066.json", "0067.json", "0071.json", "0072.json",
    "0075.json", "0078.json", "0082.json", "0084.json", "0086.json", "0088.json",
    "0093.json", "0094.json", "0095.json", "0096.json", "0099.json",
)
PREAMBLE = (
    "Solve all tasks below. Put every answer in its own \\boxed{} block "
    "with the exact requested output name, for example \\boxed{task_1_out = ...}."
)


def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json_str = json.dumps(data, indent=2)
    json_str = re.sub(
        r"\[\s+([^\[\]\{\}]*?)\s+\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
        json_str,
    )
    json_str = re.sub(r"\[\s+\]", "[]", json_str)
    with open(path, "w") as f:
        f.write(json_str)


def parse_task_types(raw_values):
    if len(raw_values) == 1:
        text = raw_values[0].strip()
        if text.startswith("["):
            values = ast.literal_eval(text)
        elif "," in text:
            values = [part.strip() for part in text.split(",")]
        else:
            values = raw_values
    else:
        values = raw_values
    task_types = [str(value).strip().strip("\"'").lower() for value in values]
    invalid = [value for value in task_types if value not in FAMILIES]
    if invalid:
        raise ValueError(f"Unknown task type(s): {invalid}. Valid choices are {sorted(FAMILIES)}.")
    if len(task_types) < 1:
        raise ValueError("--task-types must contain at least one task type")
    return task_types


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, Fraction):
        return int(value) if value.denominator == 1 else float(value)
    if hasattr(value, "item"):
        try:
            return plain(value.item())
        except Exception:
            pass
    if type(value).__module__.startswith("sympy"):
        if getattr(value, "is_integer", False) is True:
            return int(value)
        if getattr(value, "is_real", False) is True:
            result = float(value)
            return int(result) if result.is_integer() else result
    return value


def unusable(value):
    if isinstance(value, list):
        if not value or len(value) == 1:
            return True
        if collections.Counter(value).most_common(1)[0][1] > len(value) / 2:
            return True
        return value == list(range(len(value))) or value == list(range(1, len(value) + 1))
    return isinstance(value, (int, float)) and (
        not math.isfinite(value) or value in (0, 1, -1) or abs(value) >= 10**10
    )


def random_value(kind):
    if kind == "list":
        return [random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG) for _ in range(LIST_LEN_MAX)]
    return random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)


def python_input_kinds(func):
    kinds = []
    if func["has_list_in"]:
        kinds.append("list")
    if func["has_scalar_in"]:
        kinds.append("scalar")
    return kinds


def load_pool(family):
    if family == "math":
        specs = sorted(MATH_TASK_SPECS, key=lambda spec: (spec.input_kind, spec.output_kind, spec.cls.__name__))
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
        funcs = load_candidate_algos(algos_file)
        return [
            {
                "family": "python",
                "task_id": task_id,
                "task_name": func["name"],
                "input_kinds": python_input_kinds(func),
                "output_kind": "list" if func["has_list_out"] else "scalar",
                "func": func,
                "algos_file": algos_file,
            }
            for task_id, func in enumerate(funcs)
        ]

    templates = load_templates(get_default_templates_dir())
    target_indices = [
        index for index, template in enumerate(templates)
        if template["_filename"] in TARGET_TEMPLATES
    ]
    return [
        {
            "family": "gsm",
            "task_id": task_id,
            "task_name": f"GSM_{templates[template_index]['_filename']}",
            "input_kinds": ["scalar"],
            "output_kind": "scalar",
            "template_index": template_index,
        }
        for task_id, template_index in enumerate(target_indices)
    ]


def identity(task):
    return task["family"], task["task_id"]


def shuffled(values, rng):
    values = list(values)
    rng.shuffle(values)
    return values


def assert_unique_stage_tasks(records):
    seen = {}
    for record in records:
        key = (record["family"], record["task_id"])
        if key in seen:
            raise ValueError(
                f"Duplicate task id {record['family']}:{record['task_id']} "
                f"at stages {seen[key]} and {record['task_number']}."
            )
        seen[key] = record["task_number"]


def sample_progress(targets, num_samples_per_task, desc):
    total = len(targets) * num_samples_per_task
    items = (
        (target, sample_idx)
        for target in targets
        for sample_idx in range(num_samples_per_task)
    )
    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc, unit="sample")
        return
    for index, item in enumerate(items, start=1):
        print(f"{desc}: {index}/{total}", flush=True)
        yield item


def choose_stages(task_types, target, pools, rng):
    used = {identity(target)}

    def choose_prefix(position, required_output_kind, used_identities):
        family = task_types[position]
        candidates = [
            task for task in pools[family]
            if task["output_kind"] == required_output_kind and identity(task) not in used_identities
        ]
        for task in shuffled(candidates, rng):
            next_used = used_identities | {identity(task)}
            for input_kind in shuffled(task["input_kinds"], rng):
                if position == 0:
                    return [(family, task, input_kind)]
                prefix = choose_prefix(position - 1, input_kind, next_used)
                if prefix is not None:
                    return prefix + [(family, task, input_kind)]
        return None

    for target_input_kind in shuffled(target["input_kinds"], rng):
        if len(task_types) == 1:
            return [(task_types[-1], target, target_input_kind)]
        prefix = choose_prefix(len(task_types) - 2, target_input_kind, used)
        if prefix is not None:
            return prefix + [(task_types[-1], target, target_input_kind)]

    raise ValueError(f"No compatible chain for target {target['family']}:{target['task_name']}.")


def make_stage(
    family,
    task,
    task_number,
    input_name,
    value,
    *,
    candidate_tasks=None,
    fixed_task=True,
):
    if fixed_task:
        candidates = [task]
    else:
        if not candidate_tasks:
            raise ValueError(f"No candidate {family} tasks for task {task_number}.")
        candidates = shuffled(candidate_tasks, random)

    last_error = None
    for actual_task in candidates:
        try:
            if family == "math":
                spec = actual_task["spec"]
                inp = _build_math_input(spec.adapter, input_name, value, LIST_LEN_MAX)
                obj = spec.cls(task_number, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
                output = plain(obj.out)
                prompt = obj.prompt.rstrip()
                initial_conditions = []
            elif family == "python":
                def rand_static_val(kind):
                    return random_value(kind)

                obj = PythonTraceTask(
                    task_ind=task_number,
                    inp={"chained": {"chain_name": input_name, "value": value}},
                    scalar_max_mag=SCALAR_MAX_MAG,
                    list_len_max=LIST_LEN_MAX,
                    rand_static_val=rand_static_val,
                    algos_file=actual_task["algos_file"],
                    force_output_kind=actual_task["output_kind"],
                    func_index=actual_task["task_id"],
                    static_attempts_per_func=PYTHON_STATIC_ATTEMPTS_PER_FUNC,
                )
                output = plain(obj.out)
                prompt = obj.prompt.rstrip()
                initial_conditions = [
                    f"task_{task_number}_{name}_static = {plain(static_value)!r}"
                    for name, static_value in obj.static_values.items()
                ]
            else:
                obj = GSMSymbolicTask(
                    task_ind=task_number,
                    inp={f"task_{task_number}_input": {"chain_name": input_name, "value": value}},
                    scalar_max_mag=SCALAR_MAX_MAG,
                    list_len_max=LIST_LEN_MAX,
                    template_index=actual_task["template_index"],
                )
                output = plain(obj.out)
                prompt = obj.prompt.rstrip()
                initial_conditions = []

            if unusable(output):
                raise ValueError(
                    f"{family}:{actual_task['task_name']} produced an unusable output."
                )

            return output, prompt, initial_conditions, actual_task
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        f"Could not instantiate task {task_number} from {len(candidates)} "
        f"{family} candidate(s): {last_error}"
    )


def make_sample(target, sample_idx, task_types, pools, rng):
    for _ in range(100):
        random.seed(rng.randrange(2**31))
        stages = choose_stages(task_types, target, pools, rng)
        try:
            value = random_value(stages[0][2])
            chain_input = plain(value)
            initial_conditions = [f"task_1_input = {chain_input!r}"]
            records = []
            used_actual = {identity(target)}

            for task_number, (family, task, input_kind) in enumerate(stages, start=1):
                input_name = "task_1_input" if task_number == 1 else f"task_{task_number - 1}_out"
                is_target_stage = task_number == len(stages)
                candidate_tasks = None
                fixed_task = is_target_stage
                if not is_target_stage:
                    candidate_tasks = [
                        candidate for candidate in pools[family]
                        if input_kind in candidate["input_kinds"]
                        and candidate["output_kind"] == task["output_kind"]
                        and identity(candidate) not in used_actual
                    ]
                output, prompt, extra_conditions, actual_task = make_stage(
                    family,
                    task,
                    task_number,
                    input_name,
                    value,
                    candidate_tasks=candidate_tasks,
                    fixed_task=fixed_task,
                )
                used_actual.add(identity(actual_task))
                initial_conditions.extend(extra_conditions)
                records.append({
                    "task_number": task_number,
                    "family": family,
                    "task_id": actual_task["task_id"],
                    "task_name": actual_task["task_name"],
                    "input_name": input_name,
                    "input_kind": input_kind,
                    "output_kind": actual_task["output_kind"],
                    "ground_truth": output,
                    "prompt": prompt,
                })
                value = output

            prompt = (
                f"{PREAMBLE}\n\nInitial Conditions:\n"
                + "\n".join(initial_conditions)
                + "\n\n"
                + "\n\n".join(record["prompt"] for record in records)
                + "\n"
            )
            assert_unique_stage_tasks(records)
            return {
                "task_id": target["task_id"],
                "task_name": target["task_name"],
                "target_family": target["family"],
                "sample_idx": sample_idx,
                "chain_input": chain_input,
                "prompt": prompt,
                "ground_truth": plain(value),
                "chain": {
                    "task_types": task_types,
                    "num_distractors": len(task_types) - 1,
                    "target_task_number": len(task_types),
                    "stages": [
                        {key: item for key, item in record.items() if key != "prompt"}
                        for record in records
                    ],
                },
            }
        except Exception:
            continue
    raise RuntimeError(f"Could not generate sample {sample_idx} for {target['family']}:{target['task_name']}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-types", "--task_types", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples_per_task", "--num-samples-per-task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_samples_per_task < 1:
        raise ValueError("--num_samples_per_task must be at least 1")

    task_types = parse_task_types(args.task_types)
    pools = {family: load_pool(family) for family in sorted(set(task_types))}
    targets = pools[task_types[-1]]
    rng = random.Random(args.seed)

    samples = [
        make_sample(target, sample_idx, task_types, pools, rng)
        for target, sample_idx in sample_progress(
            targets,
            args.num_samples_per_task,
            "Generating chain samples",
        )
    ]
    data = {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": "chain",
            "task_types": task_types,
            "target_family": task_types[-1],
            "num_tasks": len(targets),
            "num_samples_per_task": args.num_samples_per_task,
            "num_samples": len(samples),
            "seed": args.seed,
            "list_len_max": LIST_LEN_MAX,
            "scalar_max_mag": SCALAR_MAX_MAG,
            "target_position": "last",
        },
        "samples": samples,
    }
    save_json(data, args.output)
    print(f"Generated {len(samples)} chain samples at {args.output}")


if __name__ == "__main__":
    main()
