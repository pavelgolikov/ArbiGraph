"""Prepare matched math datasets for full-parameter GRPO."""

import argparse
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2] / "source"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE.parent / "generate"))

from fill_dag import load_pool
from generate_dataset import generate_dataset
from generate_exp2_datasets import (
    capture_interventions,
    make_edge_cut_dataset,
    make_isolated_target_dataset,
)


SPLITS = {
    "train": (0, 1, 2, 6, 7, 8, 10, 11, 12, 14, 15, 19, 21, 23, 24, 25, 27, 29, 31, 32, 34, 36, 38, 39),
    "dev": (4, 5, 13, 18, 20, 28, 33, 37),
    "test": (3, 9, 16, 17, 22, 26, 30, 35),
}
TRAIN_TOPOLOGIES = ("two_chain_target", "three_chain_target")
EVAL_TOPOLOGIES = TRAIN_TOPOLOGIES + (
    "six_chain_target", "four_way_fan_in_target", "two_branch_recombine_target",
)
TWO_CHAIN = {
    "directed": True,
    "multigraph": False,
    "target": {"id": "task_2", "native_task_id": "all"},
    "nodes": [{"id": "task_1", "category": "math"}, {"id": "task_2", "category": "math"}],
    "edges": [{"source": "task_1", "destination": "task_2"}],
}


def select_split(dataset, split):
    samples = [s for s in dataset["samples"] if s["target_native_task_id"] in SPLITS[split]]
    count = dataset["summary"]["num_samples_per_task"]
    keys = [(s["target_native_task_id"], s["sample_idx"]) for s in samples]
    expected = {(task, i) for task in SPLITS[split] for i in range(count)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"Missing or duplicate examples in {split} split")
    return {"summary": {**dataset["summary"], "num_samples": len(samples)}, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SOURCE.parent / "generate" / "generated" / "rlvr_full_train")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    topology_dir = SOURCE.parent / "generate" / "topologies" / "math" / "exp1"
    baseline = json.loads((topology_dir / "single_target_baseline.json").read_text())
    pool = load_pool("math")
    for name in EVAL_TOPOLOGIES + ("six_independent_target",):
        topology = TWO_CHAIN if name == "two_chain_target" else json.loads(
            (topology_dir / f"{name}.json").read_text()
        )
        full = generate_dataset(topology, 8, seed=0, strict=True)
        conditions = {"full": full}
        if name in EVAL_TOPOLOGIES:
            conditions["edge_cut"] = make_edge_cut_dataset(full, name)
            interventions = capture_interventions(full, topology, "math", pool)
            conditions["isolated_target"] = make_isolated_target_dataset(
                full, baseline, "math", name, pool, interventions,
            )
        for split in SPLITS:
            for condition, dataset in conditions.items():
                if split == "train" and (name not in TRAIN_TOPOLOGIES or condition == "isolated_target"):
                    continue
                selected = select_split(dataset, split)
                path = args.output_dir / split / name / f"{condition}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
                print(f"Wrote {len(selected['samples'])} prompts to {path}", flush=True)


if __name__ == "__main__":
    main()
