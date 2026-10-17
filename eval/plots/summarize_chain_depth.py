"""Export node-by-node chain prediction gaps for the paper.

For every model/category pair, the point prediction at depth d is the mean
product of calibrated solo-task success rates over nodes 1..d in each sampled
chain.  Measured accuracy is the correctness of the value carried at node d.
Intervals jointly resample target-task templates and the observed calibration
rollouts, matching the prediction intervals used in the main gap figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it")
CATEGORIES = ("math", "python")
TOPOLOGY = "eighteen_chain_target"


def calibration_outcomes(model: str, category: str) -> dict[int, list[bool]]:
    outcomes: dict[int, list[bool]] = defaultdict(list)
    root = ROOT / "eval/results/gap" / model / category
    for path in sorted(root.glob("agent_single_target_baseline*.json")):
        for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            if not sample.get("grading"):
                raise ValueError(f"Ungraded calibration sample in {path}")
            outcomes[int(sample["target_native_task_id"])].append(
                bool(sample["grading"]["target_correct"])
            )
    if not outcomes:
        raise ValueError(f"No calibration outcomes for {model}/{category}")
    return dict(outcomes)


def summarize(model: str, category: str, *, draws: int, seed: int) -> list[dict]:
    outcomes = calibration_outcomes(model, category)
    tasks = sorted(outcomes)
    column = {task: index for index, task in enumerate(tasks)}
    successes = np.array([sum(outcomes[task]) for task in tasks], dtype=float)
    totals = np.array([len(outcomes[task]) for task in tasks], dtype=float)
    rates = successes / totals

    path = (ROOT / "eval/results/gap" / model / category /
            f"agent_{TOPOLOGY}_{model}.json")
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
    if any(not sample.get("grading") for sample in samples):
        raise ValueError(f"Ungraded chain sample in {path}")

    task_indices = np.array([
        [column[int(record["native_task_id"])] for record in sample["node_outputs"]]
        for sample in samples
    ], dtype=np.int16)
    correct = np.array([
        [bool(sample["grading"]["node_correct"][record["node_id"]])
         for record in sample["node_outputs"]]
        for sample in samples
    ], dtype=bool)
    predicted = np.cumprod(rates[task_indices], axis=1).mean(axis=0)
    measured = correct.mean(axis=0)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        clusters[int(sample["target_native_task_id"])].append(index)
    sizes = {len(indices) for indices in clusters.values()}
    if len(sizes) != 1:
        raise ValueError(f"Uneven target-template cluster sizes in {path}: {sorted(sizes)}")
    cluster_indices = np.array([clusters[key] for key in sorted(clusters)], dtype=np.int32)
    clustered_tasks = task_indices[cluster_indices]
    clustered_correct = correct[cluster_indices]

    rng = np.random.default_rng(seed)
    replicates = np.empty((draws, 18), dtype=float)
    batch_size = 250
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        count = stop - start
        probabilities = rng.binomial(totals.astype(int), rates,
                                     size=(count, len(tasks))) / totals
        choices = rng.integers(0, len(cluster_indices),
                               size=(count, len(cluster_indices)))
        selected_tasks = clustered_tasks[choices]
        selector = np.arange(count)[:, None, None, None]
        node_probabilities = probabilities[selector, selected_tasks]
        predicted_draws = np.cumprod(node_probabilities, axis=3).mean(axis=(1, 2))
        measured_draws = clustered_correct[choices].mean(axis=(1, 2))
        replicates[start:stop] = predicted_draws - measured_draws

    low, high = np.percentile(replicates, [2.5, 97.5], axis=0)
    return [
        {
            "model": model,
            "category": category,
            "depth": depth,
            "predicted": predicted[depth - 1],
            "measured": measured[depth - 1],
            "gap": predicted[depth - 1] - measured[depth - 1],
            "ci_low": low[depth - 1],
            "ci_high": high[depth - 1],
            "graphs": len(samples),
            "templates": len(clusters),
            "interval": "target-template + calibration bootstrap",
        }
        for depth in range(1, 19)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "eval/plots/gap/chain_depth_summary.csv",
    )
    args = parser.parse_args()
    rows = []
    for model_index, model in enumerate(MODELS):
        for category_index, category in enumerate(CATEGORIES):
            rows.extend(summarize(
                model, category, draws=args.draws,
                seed=20260925 + 100 * model_index + category_index,
            ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["depth"] in (1, 6, 12, 18):
            print(f"{row['model']:16s} {row['category']:6s} depth {row['depth']:2d}: "
                  f"gap {100 * row['gap']:+5.1f} pp "
                  f"[{100 * row['ci_low']:+5.1f}, {100 * row['ci_high']:+5.1f}]")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
