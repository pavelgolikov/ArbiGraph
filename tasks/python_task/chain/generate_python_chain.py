"""Generate Python dependency chains directly from the candidate functions."""

import argparse
import collections
import datetime
import math
import os
import random
import sys


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TASK_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.forgetting_dataset import save_json
from tasks.python_task.python_task import PythonTraceTask, load_candidate_algos


LIST_LEN_MAX = 10
SCALAR_MAX_MAG = 100
PREAMBLE = (
    "Solve all tasks below. Put every answer in its own \\boxed{} block "
    "with the exact requested output name, for example \\boxed{task_1_out = ...}."
)

def input_kind(func):
    return "list" if func["has_list_in"] else "scalar"


def output_kind(func):
    return "list" if func["has_list_out"] else "scalar"


def random_value(kind):
    if kind == "list":
        return [random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG) for _ in range(LIST_LEN_MAX)]
    return random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)


def unusable(value):
    if isinstance(value, list):
        if not value or len(value) == 1:
            return True
        if collections.Counter(value).most_common(1)[0][1] > len(value) / 2:
            return True
        return value == list(range(len(value))) or value == list(range(1, len(value) + 1))
    return isinstance(value, (int, float)) and (not math.isfinite(value) or value in (0, 1, -1) or abs(value) >= 10**10)


def choose_stages(target_index, funcs, num_distractors, rng):
    stages = [target_index]
    used = {target_index}
    required_kind = input_kind(funcs[target_index])
    for _ in range(num_distractors):
        choices = [
            index for index, func in enumerate(funcs)
            if index not in used and output_kind(func) == required_kind
        ]
        if not choices:
            raise ValueError(f"No Python predecessor produces {required_kind}.")
        index = rng.choice(choices)
        stages.append(index)
        used.add(index)
        required_kind = input_kind(funcs[index])
    return list(reversed(stages))


def make_sample(task_id, funcs, algos_file, sample_idx, num_distractors, rng):
    for _ in range(100):
        random.seed(rng.randrange(2**31))
        try:
            stages = choose_stages(task_id, funcs, num_distractors, rng)
            value = random_value(input_kind(funcs[stages[0]]))
            chain_input = value
            initial_conditions = [f"task_1_input = {value!r}"]
            records = []

            def rand_static_val(kind):
                return random_value(kind)

            for task_number, func_index in enumerate(stages, start=1):
                input_name = "task_1_input" if task_number == 1 else f"task_{task_number - 1}_out"
                task = PythonTraceTask(
                    task_ind=task_number,
                    inp={"chained": {"chain_name": input_name, "value": value}},
                    scalar_max_mag=SCALAR_MAX_MAG,
                    list_len_max=LIST_LEN_MAX,
                    rand_static_val=rand_static_val,
                    algos_file=algos_file,
                    func_index=func_index,
                )
                output = list(task.out) if isinstance(task.out, tuple) else task.out
                if unusable(output):
                    raise ValueError(f"{funcs[func_index]['name']} produced an unusable output.")
                for name, static_value in task.static_values.items():
                    initial_conditions.append(f"task_{task_number}_{name}_static = {static_value!r}")
                records.append({
                    "task_number": task_number,
                    "task_id": func_index,
                    "task_name": funcs[func_index]["name"],
                    "input_name": input_name,
                    "ground_truth": output,
                    "prompt": task.prompt.rstrip(),
                })
                value = output

            prompt = (
                f"{PREAMBLE}\n\nInitial Conditions:\n"
                + "\n".join(initial_conditions)
                + "\n\n"
                + "\n\n".join(record["prompt"] for record in records)
                + "\n"
            )
            return {
                "task_id": task_id,
                "task_name": funcs[task_id]["name"],
                "sample_idx": sample_idx,
                "chain_input": chain_input,
                "prompt": prompt,
                "ground_truth": value,
                "chain": {
                    "num_distractors": num_distractors,
                    "target_task_number": num_distractors + 1,
                    "distractors": [
                        {key: value for key, value in record.items() if key != "prompt"}
                        for record in records[:-1]
                    ],
                    "target": {
                        key: value for key, value in records[-1].items() if key != "prompt"
                    },
                },
            }
        except Exception:
            continue
    raise RuntimeError(f"Could not generate sample {sample_idx} for {funcs[task_id]['name']}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(TASK_DIR, "python_chain.json"))
    parser.add_argument("--num_distractors", type=int, default=3)
    parser.add_argument("--num_samples_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_distractors < 1 or args.num_samples_per_task < 1:
        raise ValueError("--num_distractors and --num_samples_per_task must be at least 1")

    algos_file = os.path.join(REPO_ROOT, "tasks", "python_task", "leetcode_candidate_algos.txt")
    funcs = load_candidate_algos(algos_file)
    rng = random.Random(args.seed)
    samples = [
        make_sample(task_id, funcs, algos_file, sample_idx, args.num_distractors, rng)
        for task_id in range(len(funcs))
        for sample_idx in range(args.num_samples_per_task)
    ]
    data = {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": "python_chain",
            "num_tasks": len(funcs),
            "num_samples_per_task": args.num_samples_per_task,
            "num_samples": len(samples),
            "seed": args.seed,
            "num_distractors": args.num_distractors,
            "target_position": "last",
        },
        "samples": samples,
    }
    save_json(data, args.output)
    print(f"Generated {len(samples)} Python chain samples at {args.output}")


if __name__ == "__main__":
    main()
