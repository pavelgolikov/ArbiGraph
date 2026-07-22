"""Generate GSM-Symbolic dependency chains directly from the template pool."""

import argparse
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
from tasks.gsm_task.gsm_symbolic_task import (
    GSMSymbolicTask,
    get_default_templates_dir,
    load_templates,
)


LIST_LEN_MAX = 10
SCALAR_MAX_MAG = 100
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
    "Solve all tasks below in order. Each task after the first uses the "
    "preceding task's output. Put every answer in its own \\boxed{} block "
    "with the exact requested output name, for example "
    "\\boxed{task_1_out = ...}."
)


def unusable(value):
    return value is None or (
        isinstance(value, (int, float))
        and (not math.isfinite(value) or value in (0, 1, -1) or abs(value) >= 10**10)
    )


def choose_stages(target_index, template_count, num_distractors, rng):
    stages = [target_index]
    used = {target_index}
    for _ in range(num_distractors):
        choices = [index for index in range(template_count) if index not in used]
        if not choices:
            raise ValueError("Not enough distinct GSM templates for this chain.")
        index = rng.choice(choices)
        stages.append(index)
        used.add(index)
    return list(reversed(stages))


def make_sample(task_id, target_index, target_name, template_count, sample_idx, num_distractors, rng):
    for _ in range(100):
        try:
            stages = choose_stages(target_index, template_count, num_distractors, rng)
            value = random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)
            chain_input = value
            initial_conditions = [f"task_1_input = {value!r}"]
            records = []

            for task_number, template_index in enumerate(stages, start=1):
                input_name = "task_1_input" if task_number == 1 else f"task_{task_number - 1}_out"
                task = GSMSymbolicTask(
                    task_ind=task_number,
                    inp={f"task_{task_number}_input": {"chain_name": input_name, "value": value}},
                    scalar_max_mag=SCALAR_MAX_MAG,
                    list_len_max=LIST_LEN_MAX,
                    template_index=template_index,
                )
                output = task.out
                if unusable(output):
                    raise ValueError("GSM task produced an unusable output.")
                records.append({
                    "task_number": task_number,
                    "template_index": template_index,
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
                "task_name": target_name,
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
    raise RuntimeError(f"Could not generate sample {sample_idx} for {target_name}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(TASK_DIR, "gsm_chain.json"))
    parser.add_argument("--num_distractors", type=int, default=3)
    parser.add_argument("--num_samples_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_distractors < 1 or args.num_samples_per_task < 1:
        raise ValueError("--num_distractors and --num_samples_per_task must be at least 1")

    templates = load_templates(get_default_templates_dir())
    target_indices = [
        index for index, template in enumerate(templates)
        if template["_filename"] in TARGET_TEMPLATES
    ]
    rng = random.Random(args.seed)
    samples = [
        make_sample(
            task_id,
            target_index,
            f"GSM_{templates[target_index]['_filename']}",
            len(templates),
            sample_idx,
            args.num_distractors,
            rng,
        )
        for task_id, target_index in enumerate(target_indices)
        for sample_idx in range(args.num_samples_per_task)
    ]
    data = {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": "gsm_chain",
            "num_tasks": len(target_indices),
            "num_samples_per_task": args.num_samples_per_task,
            "num_samples": len(samples),
            "seed": args.seed,
            "num_distractors": args.num_distractors,
            "target_position": "last",
        },
        "samples": samples,
    }
    save_json(data, args.output)
    print(f"Generated {len(samples)} GSM chain samples at {args.output}")


if __name__ == "__main__":
    main()
