"""Generate multi-chain DAG datasets.

Example:
  python generate_multichain.py \
    --structure "['m','m','p',['m','m'],['p','m'],['p','p'],'m','p']" \
    --num_samples_per_task 1 \
    --output scratch/multichain_smoke/cross.json
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
FAMILY_ALIASES = {
    "m": "math",
    "math": "math",
    "p": "python",
    "py": "python",
    "python": "python",
    "g": "gsm",
    "gsm": "gsm",
}
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
    "Solve all tasks below. Some tasks depend on outputs from earlier tasks. "
    "Put every answer in its own \\boxed{} block with the exact requested output name, "
    "for example \\boxed{task_1_out = ...}."
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


def normalize_family(value):
    key = str(value).strip().strip("\"'").lower()
    if key not in FAMILY_ALIASES:
        raise ValueError(f"Unknown task family {value!r}. Use m/math, p/python, or g/gsm.")
    return FAMILY_ALIASES[key]


def parse_structure(text):
    raw = ast.literal_eval(text)
    if not isinstance(raw, list):
        raise ValueError("--structure must be a Python literal list")

    normalized = []
    for item in raw:
        if isinstance(item, list):
            if not item:
                raise ValueError("Branch lists must be non-empty.")
            if any(isinstance(child, list) for child in item):
                raise ValueError("Nested branch lists are not supported in v1.")
            normalized.append([normalize_family(child) for child in item])
        else:
            normalized.append(normalize_family(item))

    flat_segments = []
    branch_blocks = []
    current_flat = []
    index = 0
    while index < len(normalized):
        item = normalized[index]
        if isinstance(item, str):
            current_flat.append(item)
            index += 1
            continue

        if not current_flat:
            raise ValueError("Every branch block must have at least one flat task before it.")
        block = []
        while index < len(normalized) and isinstance(normalized[index], list):
            block.append(normalized[index])
            index += 1
        if len(block) < 2:
            raise ValueError("Every branch block must contain at least two branches.")
        if len(block) > LIST_LEN_MAX:
            raise ValueError(f"Branch block has {len(block)} branches, but LIST_LEN_MAX is {LIST_LEN_MAX}.")
        flat_segments.append(current_flat)
        branch_blocks.append(block)
        current_flat = []

    if not branch_blocks:
        raise ValueError("Multi-chain structure must contain at least one branch block.")
    if not current_flat:
        raise ValueError("Every branch block must have at least one flat task after it.")
    flat_segments.append(current_flat)
    return normalized, flat_segments, branch_blocks


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


def choose_linear_tasks(
    families,
    pools,
    rng,
    used,
    *,
    start_input_kind=None,
    end_output_kind=None,
    fixed_last_task=None,
):
    if not families:
        raise ValueError("Linear task segment must be non-empty.")
    if fixed_last_task is not None and fixed_last_task["family"] != families[-1]:
        raise ValueError(
            f"Fixed target {fixed_last_task['family']}:{fixed_last_task['task_name']} "
            f"does not match final family {families[-1]}."
        )

    fixed_identity = identity(fixed_last_task) if fixed_last_task is not None else None

    def rec(position, required_input_kind, used_here):
        family = families[position]
        is_last = position == len(families) - 1
        candidates = [fixed_last_task] if is_last and fixed_last_task is not None else pools[family]

        for task in shuffled(candidates, rng):
            task_identity = identity(task)
            if task_identity in used_here and not (is_last and task_identity == fixed_identity):
                continue
            if is_last and end_output_kind is not None and task["output_kind"] != end_output_kind:
                continue

            if required_input_kind is None:
                input_kinds = shuffled(task["input_kinds"], rng)
            elif required_input_kind in task["input_kinds"]:
                input_kinds = [required_input_kind]
            else:
                continue

            for input_kind in input_kinds:
                selected = (family, task, input_kind)
                if is_last:
                    return [selected]
                tail = rec(position + 1, task["output_kind"], used_here | {task_identity})
                if tail is not None:
                    return [selected] + tail
        return None

    result = rec(0, start_input_kind, set(used))
    if result is None:
        constraints = []
        if start_input_kind is not None:
            constraints.append(f"start_input={start_input_kind}")
        if end_output_kind is not None:
            constraints.append(f"end_output={end_output_kind}")
        if fixed_last_task is not None:
            constraints.append(f"fixed_last={fixed_last_task['family']}:{fixed_last_task['task_name']}")
        suffix = f" ({', '.join(constraints)})" if constraints else ""
        raise ValueError(f"Could not select compatible segment {families}{suffix}.")
    return result


def choose_layout(target, flat_segments, branch_blocks, pools, rng):
    used = {identity(target)}
    selected_flat_segments = []
    selected_branch_blocks = []

    for segment_index, families in enumerate(flat_segments):
        is_final_segment = segment_index == len(flat_segments) - 1
        selected_segment = choose_linear_tasks(
            families,
            pools,
            rng,
            used,
            start_input_kind=None if segment_index == 0 else "list",
            end_output_kind="list" if segment_index < len(branch_blocks) else None,
            fixed_last_task=target if is_final_segment else None,
        )
        selected_flat_segments.append(selected_segment)
        used.update(identity(task) for _family, task, _input_kind in selected_segment)

        if segment_index < len(branch_blocks):
            selected_block = []
            for branch_families in branch_blocks[segment_index]:
                selected_branch = choose_linear_tasks(
                    branch_families,
                    pools,
                    rng,
                    used,
                    start_input_kind="scalar",
                    end_output_kind="scalar",
                )
                selected_block.append(selected_branch)
                used.update(identity(task) for _family, task, _input_kind in selected_branch)
            selected_branch_blocks.append(selected_block)

    return selected_flat_segments, selected_branch_blocks


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


def expansion_adapter(expansion_output_name, branch_inputs):
    parts = [
        f"Use the first {len(branch_inputs)} elements of {expansion_output_name} "
        "as inputs to the following tasks."
    ]
    for index, input_name in enumerate(branch_inputs):
        parts.append(f"Define {input_name} = {expansion_output_name}[{index}].")
    return "\n".join(parts)


def expanded_next_input_values(branch_values, target_len=LIST_LEN_MAX):
    values = []
    multiplier = 1
    while len(values) < target_len:
        offset = multiplier - 1
        for value in branch_values:
            values.append(plain(multiplier * value + offset))
            if len(values) == target_len:
                break
        multiplier += 1
    return values


def expanded_next_input_expressions(branch_output_names, target_len=LIST_LEN_MAX):
    expressions = []
    multiplier = 1
    while len(expressions) < target_len:
        offset = multiplier - 1
        for name in branch_output_names:
            if multiplier == 1:
                expressions.append(name)
            else:
                expressions.append(f"{name} * {multiplier} + {offset}")
            if len(expressions) == target_len:
                break
        multiplier += 1
    return expressions


def next_input_prompt(next_input_name, branch_output_names):
    prelim_name = f"{next_input_name}_prelim"
    prelim = ", ".join(branch_output_names)
    expanded = ", ".join(expanded_next_input_expressions(branch_output_names))
    return (
        f"Define {prelim_name} = [{prelim}].\n"
        f"Define {next_input_name} by repeating transformed copies of "
        f"{prelim_name} until the list has at least {LIST_LEN_MAX} elements, "
        f"then keeping only the first {LIST_LEN_MAX} elements:\n"
        f"{next_input_name} = [{expanded}]."
    )


def stage_record(task_number, family, task, input_name, input_kind, output):
    return {
        "task_number": task_number,
        "family": family,
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "input_name": input_name,
        "input_kind": input_kind,
        "output_kind": task["output_kind"],
        "ground_truth": output,
    }


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


def generate_selected_layout(selected_flat_segments, selected_branch_blocks, pools, fixed_target_identity):
    initial_conditions = []
    prompt_parts = []
    records = []
    branch_block_records = []
    task_number = 1
    next_flat_input_value = None
    chain_input = None
    final_output = None
    used_actual = {fixed_target_identity} if fixed_target_identity is not None else set()
    final_task_number = (
        sum(len(segment) for segment in selected_flat_segments)
        + sum(len(branch) for block in selected_branch_blocks for branch in block)
    )

    for segment_index, selected_segment in enumerate(selected_flat_segments):
        segment_first = True
        previous_output = None

        for family, task, input_kind in selected_segment:
            if task_number == 1:
                input_name = "task_1_input"
                input_value = plain(random_value(input_kind))
                chain_input = input_value
                initial_conditions.append(f"{input_name} = {input_value!r}")
            elif segment_first and segment_index > 0:
                input_name = f"task_{task_number}_input"
                input_value = next_flat_input_value
            else:
                input_name = f"task_{task_number - 1}_out"
                input_value = previous_output

            is_fixed_target = (
                fixed_target_identity is not None
                and task_number == final_task_number
                and identity(task) == fixed_target_identity
            )
            candidate_tasks = None
            fixed_task = is_fixed_target
            if not is_fixed_target:
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
                input_value,
                candidate_tasks=candidate_tasks,
                fixed_task=fixed_task,
            )
            used_actual.add(identity(actual_task))
            initial_conditions.extend(extra_conditions)
            prompt_parts.append(prompt)
            records.append(stage_record(task_number, family, actual_task, input_name, input_kind, output))
            previous_output = output
            final_output = output
            segment_first = False
            task_number += 1

        if segment_index >= len(selected_branch_blocks):
            break

        expansion_record = records[-1]
        expansion_output = expansion_record["ground_truth"]
        branches = selected_branch_blocks[segment_index]
        if not isinstance(expansion_output, list) or len(expansion_output) < len(branches):
            raise ValueError("Expansion task did not produce enough list elements for the branch block.")

        branch_start_numbers = []
        future_task_number = task_number
        for branch in branches:
            branch_start_numbers.append(future_task_number)
            future_task_number += len(branch)

        branch_input_names = [f"task_{number}_input" for number in branch_start_numbers]
        prompt_parts.append(expansion_adapter(f"task_{expansion_record['task_number']}_out", branch_input_names))

        branch_records = []
        branch_final_values = []
        branch_final_output_names = []

        for branch_index, branch in enumerate(branches):
            branch_task_records = []
            branch_previous_output = None
            branch_input_value = plain(expansion_output[branch_index])
            if isinstance(branch_input_value, list):
                raise ValueError("Scalar branch lane received a list element.")

            for branch_task_index, (family, task, input_kind) in enumerate(branch):
                if branch_task_index == 0:
                    input_name = f"task_{task_number}_input"
                    input_value = branch_input_value
                else:
                    input_name = f"task_{task_number - 1}_out"
                    input_value = branch_previous_output

                is_fixed_target = (
                    fixed_target_identity is not None
                    and task_number == final_task_number
                    and identity(task) == fixed_target_identity
                )
                candidate_tasks = None
                fixed_task = is_fixed_target
                if not is_fixed_target:
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
                    input_value,
                    candidate_tasks=candidate_tasks,
                    fixed_task=fixed_task,
                )
                used_actual.add(identity(actual_task))
                initial_conditions.extend(extra_conditions)
                prompt_parts.append(prompt)
                record = stage_record(task_number, family, actual_task, input_name, input_kind, output)
                records.append(record)
                branch_task_records.append(record)
                branch_previous_output = output
                task_number += 1

            if isinstance(branch_previous_output, list):
                raise ValueError("Scalar branch lane ended with a list output.")
            branch_final_values.append(branch_previous_output)
            branch_final_output_names.append(f"task_{branch_task_records[-1]['task_number']}_out")
            branch_records.append({
                "branch_index": branch_index,
                "input_name": branch_task_records[0]["input_name"],
                "input_value": branch_input_value,
                "final_output_name": branch_final_output_names[-1],
                "final_output": branch_previous_output,
                "stages": branch_task_records,
            })

        next_input_name = f"task_{task_number}_input"
        next_input_prelim_value = plain(branch_final_values)
        next_flat_input_value = expanded_next_input_values(next_input_prelim_value)
        prompt_parts.append(next_input_prompt(next_input_name, branch_final_output_names))
        branch_block_records.append({
            "block_index": segment_index,
            "expansion_task_number": expansion_record["task_number"],
            "expansion_output_name": f"task_{expansion_record['task_number']}_out",
            "branch_inputs": branch_input_names,
            "next_input_name": next_input_name,
            "next_input_prelim_value": next_input_prelim_value,
            "next_input_value": next_flat_input_value,
            "branches": branch_records,
        })

    prompt = (
        f"{PREAMBLE}\n\nInitial Conditions:\n"
        + "\n".join(initial_conditions)
        + "\n\n"
        + "\n\n".join(prompt_parts)
        + "\n"
    )
    assert_unique_stage_tasks(records)
    return chain_input, prompt, plain(final_output), records, branch_block_records


def make_sample(target, sample_idx, flat_segments, branch_blocks, pools, rng):
    last_error = None
    for _ in range(100):
        random.seed(rng.randrange(2**31))
        try:
            selected_flat_segments, selected_branch_blocks = choose_layout(
                target,
                flat_segments,
                branch_blocks,
                pools,
                rng,
            )
            chain_input, prompt, ground_truth, records, branch_block_records = generate_selected_layout(
                selected_flat_segments,
                selected_branch_blocks,
                pools,
                identity(target),
            )
            return {
                "task_id": target["task_id"],
                "task_name": target["task_name"],
                "target_family": target["family"],
                "sample_idx": sample_idx,
                "chain_input": chain_input,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "multi_chain": {
                    "target_task_number": records[-1]["task_number"],
                    "stages": records,
                    "branch_blocks": branch_block_records,
                },
            }
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Could not generate sample {sample_idx} for {target['family']}:{target['task_name']}: {last_error}"
    )


def families_in_structure(normalized_structure):
    families = []
    for item in normalized_structure:
        if isinstance(item, list):
            families.extend(item)
        else:
            families.append(item)
    return families


def compatible_targets(targets, flat_segments, branch_blocks, pools, seed):
    result = []
    for target in targets:
        try:
            choose_layout(
                target,
                flat_segments,
                branch_blocks,
                pools,
                random.Random(f"{seed}:{target['family']}:{target['task_id']}"),
            )
            result.append(target)
        except ValueError:
            continue
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True, help="Python literal list, e.g. \"['m','p',['m'],['p'],'m']\"")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_samples_per_task < 1:
        raise ValueError("--num_samples_per_task must be at least 1")

    normalized_structure, flat_segments, branch_blocks = parse_structure(args.structure)
    final_family = flat_segments[-1][-1]
    pools = {family: load_pool(family) for family in sorted(set(families_in_structure(normalized_structure)))}
    all_targets = pools[final_family]
    targets = compatible_targets(all_targets, flat_segments, branch_blocks, pools, args.seed)
    if not targets:
        raise ValueError("No target tasks are compatible with the requested multi-chain structure.")
    rng = random.Random(args.seed)

    samples = [
        make_sample(target, sample_idx, flat_segments, branch_blocks, pools, rng)
        for target, sample_idx in sample_progress(
            targets,
            args.num_samples_per_task,
            "Generating multi-chain samples",
        )
    ]
    data = {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": "multi_chain",
            "structure": normalized_structure,
            "target_family": final_family,
            "num_tasks": len(targets),
            "num_tasks_total": len(all_targets),
            "num_samples_per_task": args.num_samples_per_task,
            "num_samples": len(samples),
            "seed": args.seed,
            "list_len_max": LIST_LEN_MAX,
            "scalar_max_mag": SCALAR_MAX_MAG,
            "target_position": "last",
            "branch_lane": "scalar",
        },
        "samples": samples,
    }
    save_json(data, args.output)
    print(f"Generated {len(samples)} multi-chain samples at {args.output}")


if __name__ == "__main__":
    main()
