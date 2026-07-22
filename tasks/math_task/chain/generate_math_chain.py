"""Generate math dependency chains directly from the math task registry."""

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
from tasks.math_task.math_task import MATH_TASK_SPECS, _build_math_input


LIST_LEN_MAX = 10
SCALAR_MAX_MAG = 100
PREAMBLE = (
    "Solve all tasks below. Put every answer in its own \\boxed{} block "
    "with the exact requested output name, for example \\boxed{task_1_out = ...}."
)


def random_value(kind):
    if kind == "list":
        return [random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG) for _ in range(LIST_LEN_MAX)]
    return random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, tuple):
        return [plain(item) for item in value]
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
    return isinstance(value, (int, float)) and (not math.isfinite(value) or value in (0, 1, -1) or abs(value) >= 10**10)


def choose_stages(target_spec, specs, num_distractors, rng):
    stages = [target_spec]
    used = {target_spec.cls}
    required_kind = target_spec.input_kind
    for _ in range(num_distractors):
        choices = [
            spec for spec in specs
            if spec.cls not in used and spec.output_kind == required_kind
        ]
        if not choices:
            raise ValueError(f"No math predecessor produces {required_kind}.")
        spec = rng.choice(choices)
        stages.append(spec)
        used.add(spec.cls)
        required_kind = spec.input_kind
    return list(reversed(stages))


def make_sample(task_id, target_spec, specs, sample_idx, num_distractors, rng):
    for _ in range(100):
        random.seed(rng.randrange(2**31))
        try:
            stages = choose_stages(target_spec, specs, num_distractors, rng)
            value = random_value(stages[0].input_kind)
            chain_input = value
            initial_conditions = [f"task_1_input = {value!r}"]
            records = []

            for task_number, spec in enumerate(stages, start=1):
                input_name = "task_1_input" if task_number == 1 else f"task_{task_number - 1}_out"
                inp = _build_math_input(spec.adapter, input_name, value, LIST_LEN_MAX)
                task = spec.cls(task_number, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
                output = plain(task.out)
                if unusable(output):
                    raise ValueError(f"{spec.cls.__name__} produced an unusable output.")
                records.append({
                    "task_number": task_number,
                    "task_name": spec.cls.__name__,
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
                "task_name": target_spec.cls.__name__,
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
    raise RuntimeError(f"Could not generate sample {sample_idx} for {target_spec.cls.__name__}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(TASK_DIR, "math_chain.json"))
    parser.add_argument("--num_distractors", type=int, default=3)
    parser.add_argument("--num_samples_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_distractors < 1 or args.num_samples_per_task < 1:
        raise ValueError("--num_distractors and --num_samples_per_task must be at least 1")

    specs = sorted(
        MATH_TASK_SPECS,
        key=lambda spec: (spec.input_kind, spec.output_kind, spec.cls.__name__),
    )
    rng = random.Random(args.seed)
    samples = [
        make_sample(task_id, target_spec, specs, sample_idx, args.num_distractors, rng)
        for task_id, target_spec in enumerate(specs)
        for sample_idx in range(args.num_samples_per_task)
    ]
    data = {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": "math_chain",
            "num_tasks": len(specs),
            "num_samples_per_task": args.num_samples_per_task,
            "num_samples": len(samples),
            "seed": args.seed,
            "num_distractors": args.num_distractors,
            "target_position": "last",
        },
        "samples": samples,
    }
    save_json(data, args.output)
    print(f"Generated {len(samples)} math chain samples at {args.output}")


if __name__ == "__main__":
    main()
