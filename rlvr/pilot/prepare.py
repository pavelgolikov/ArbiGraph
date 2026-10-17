"""Build repeated, matched math prompts for the RLVR pilot."""

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "generate"))

from fill_dag import load_pool
from generate_exp2_datasets import (
    capture_interventions,
    make_edge_cut_dataset,
    make_isolated_target_dataset,
)


def subset(dataset, keys, condition):
    """Select the same target-task instances from one condition."""

    selected = [
        sample
        for sample in dataset["samples"]
        if (sample["target_native_task_id"], sample["sample_idx"]) in keys
    ]
    selected_keys = {
        (sample["target_native_task_id"], sample["sample_idx"])
        for sample in selected
    }
    if selected_keys != keys:
        raise ValueError(f"Condition {condition} does not contain every selected sample.")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "generate" / "generated" / "phase_3" / "math_pilot.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.samples_per_task <= 0:
        parser.error("--samples-per-task must be positive")
    if args.rollouts_per_prompt < 2:
        parser.error("--rollouts-per-prompt must be at least 2")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; pass --overwrite to replace it")

    exp1_root = REPO_ROOT / "generate" / "generated" / "exp1" / "math"
    full_path = exp1_root / "three_chain_target" / "three_chain_target_dataset.json"
    independent_path = (
        exp1_root
        / "three_independent_target"
        / "three_independent_target_dataset.json"
    )
    topology_path = REPO_ROOT / "generate" / "topologies" / "math" / "exp1" / "three_chain_target.json"
    baseline_path = (
        REPO_ROOT / "generate" / "topologies" / "math" / "exp1" / "single_target_baseline.json"
    )

    with full_path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    with independent_path.open(encoding="utf-8") as handle:
        independent = json.load(handle)
    with topology_path.open(encoding="utf-8") as handle:
        topology = json.load(handle)
    with baseline_path.open(encoding="utf-8") as handle:
        baseline = json.load(handle)

    available = Counter(sample["target_native_task_id"] for sample in full["samples"])
    too_small = {
        task_id: count
        for task_id, count in available.items()
        if count < args.samples_per_task
    }
    if too_small:
        raise ValueError(f"Not enough samples for target tasks: {too_small}")
    keys = {
        (sample["target_native_task_id"], sample["sample_idx"])
        for sample in full["samples"]
        if sample["sample_idx"] < args.samples_per_task
    }

    edge_cut = make_edge_cut_dataset(full, "three_chain_target")
    pool = load_pool("math")
    interventions = capture_interventions(full, topology, "math", pool)
    isolated = make_isolated_target_dataset(
        full,
        baseline,
        "math",
        "three_chain_target",
        pool,
        interventions,
    )

    conditions = [
        {
            "condition": "full",
            "samples": subset(full, keys, "full"),
        },
        {
            "condition": "edge_cut",
            "samples": subset(edge_cut, keys, "edge_cut"),
        },
        {
            "condition": "isolated_target",
            "samples": subset(isolated, keys, "isolated_target"),
        },
        {
            "condition": "independent",
            "samples": subset(independent, keys, "independent"),
        },
    ]

    samples = []
    for item in conditions:
        for sample in item["samples"]:
            for rollout_idx in range(args.rollouts_per_prompt):
                rollout = copy.deepcopy(sample)
                rollout.update(
                    {
                        "condition": item["condition"],
                        "rollout_idx": rollout_idx,
                    }
                )
                samples.append(rollout)

    pilot = {
        "summary": {
            "experiment": 3,
            "category": "math",
            "samples_per_task": args.samples_per_task,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "num_target_tasks": len(available),
            "num_prompts": len(samples) // args.rollouts_per_prompt,
            "num_samples": len(samples),
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {pilot['summary']['num_samples']} trajectories for "
        f"{pilot['summary']['num_prompts']} prompts to {args.output}"
    )


if __name__ == "__main__":
    main()
