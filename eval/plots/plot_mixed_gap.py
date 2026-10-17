"""Plot absolute pure-math and mixed target accuracy in the gap plot's bar style.

Run from the repository root:

    python eval/plots/plot_mixed_gap.py

Each topology panel has two predicted/measured pairs, one for pure math and one
for mixed. The predictions use per-task solo rates and bootstrap their calibration
rollouts; measured bars use Wilson intervals. A separate CSV records paired
mixed-minus-pure differences. Direct-effect intervals use a target-template
cluster bootstrap; solo-adjusted intervals additionally propagate calibration
uncertainty with Jeffreys-binomial draws. Every requested model must have
complete graded results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_mixed_gap import (ROOT, TOPOLOGIES, analyze,
                               calibration_aware_interval, interval, load_samples)
from plot_prediction_gap import wilson_interval

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-arbigraph")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "eighteen_chain_target": "chain",
    "eighteen_braid_target": "braid",
    "eighteen_fan_out_fan_in": "fan-out/in",
}
METRICS = ("target", "target_adjusted_for_solo_accuracy")
CONDITIONS = (("pure math", "#4263a0"), ("mixed", "#ab1368"))
BAR_POSITIONS = (0.0, 0.72, 1.80, 2.52)
BAR_TICKS = (0.36, 2.16)
BOOTSTRAP_DRAWS = 2000
ADJUSTED_POOLED_DRAWS = 20_000


def complete(model: str) -> tuple[bool, str]:
    """Only use fully graded, uniquely keyed runs with matching pure pairs."""
    for topology in TOPOLOGIES:
        paths = (
            ROOT / "eval/results/gap" / model / "math" / f"agent_{topology}_{model}.json",
            ROOT / "eval/results/mixed_gap" / model / "mixed" / f"agent_{topology}_{model}.json",
        )
        keys = []
        for path in paths:
            if not path.exists():
                return False, f"missing {path}"
            samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
            if not samples or any(not sample.get("grading") for sample in samples):
                return False, f"ungraded samples in {path}"
            sample_keys = [(int(s["target_native_task_id"]), int(s["sample_idx"]))
                           for s in samples]
            if len(set(sample_keys)) != len(sample_keys):
                return False, f"duplicate target/sample keys in {path}"
            keys.append(set(sample_keys))
        if keys[0] != keys[1]:
            return False, f"pure/mixed keys differ for {model}/{topology}"
    return True, ""


def calibration_outcomes(model: str) -> dict[tuple[str, int], list[int]]:
    """Read exactly the solo runs used by the paired analysis."""
    outcomes: dict[tuple[str, int], list[int]] = defaultdict(list)
    for category in ("math", "python"):
        root = ROOT / "eval/results/gap" / model / category
        paths = sorted(root.glob("agent_single_target_baseline*.json"))
        if not paths:
            raise ValueError(f"Missing solo calibration in {root}")
        for path in paths:
            for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
                if not sample.get("grading"):
                    raise ValueError(f"Ungraded calibration sample in {path}")
                outcomes[category, int(sample["target_native_task_id"])].append(
                    int(bool(sample["grading"]["target_correct"])))
    for category in ("rag", "rlve"):
        path = (ROOT / "eval/results/mixed_gap/calibration" / model / category /
                f"agent_{category}_calibration_{model}.json")
        for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            if not sample.get("grading"):
                raise ValueError(f"Ungraded calibration sample in {path}")
            outcomes[category, int(sample["target_native_task_id"])].append(
                int(bool(sample["grading"]["target_correct"])))
    return dict(outcomes)


def task_key(record: dict, condition: str) -> tuple[str, int]:
    if condition == "pure math":
        category = "math"
    else:
        number = int(record["node_id"].removeprefix("task_"))
        category = ("math" if number in (1, 3, 5, 7, 9, 11, 13, 15, 18) else
                    "rag" if number == 17 else
                    "rlve" if number in (12, 14, 16) else "python")
    return category, int(record["native_task_id"])


def predicted_interval(samples: list[dict], condition: str,
                       outcomes: dict[tuple[str, int], list[int]],
                       rng: np.random.Generator) -> tuple[float, float, float]:
    """Per-task calibration bootstrap, holding graph assignments fixed."""
    keys = sorted(outcomes)
    columns = {key: index for index, key in enumerate(keys)}
    counts = np.array([len(outcomes[key]) for key in keys])
    rates = np.array([sum(outcomes[key]) for key in keys]) / counts
    node_indices = np.array([
        [columns[task_key(record, condition)] for record in sample["node_outputs"]]
        for sample in samples
    ])
    point = float(np.prod(rates[node_indices], axis=1).mean())
    draws = rng.binomial(counts, rates, size=(BOOTSTRAP_DRAWS, len(keys))) / counts
    replicates = np.prod(draws[:, node_indices], axis=2).mean(axis=1)
    low, high = np.percentile(replicates, [2.5, 97.5])
    return point, float(low), float(high)


def build_rows(models: list[str]) -> tuple[list[dict], list[dict]]:
    bars, differences = [], []
    rng = np.random.default_rng(20260924)
    for model in models:
        ready, reason = complete(model)
        if not ready:
            raise ValueError(f"Cannot plot {model}: {reason}")
        outcomes = calibration_outcomes(model)
        pooled: dict[str, list[tuple[int, float]]] = {metric: [] for metric in METRICS}
        pooled_calibration_records = []
        for topology in TOPOLOGIES:
            result, direct, _math, adjusted, calibration_records = analyze(model, topology)
            pooled_calibration_records.extend(calibration_records)
            name = LABELS[topology]
            for condition in ("pure math", "mixed"):
                path = ((ROOT / "eval/results/gap" / model / "math") if condition == "pure math"
                        else (ROOT / "eval/results/mixed_gap" / model / "mixed"))
                samples = list(load_samples(path / f"agent_{topology}_{model}.json").values())
                predicted, pred_low, pred_high = predicted_interval(samples, condition,
                                                                    outcomes, rng)
                expected = result["solo_predicted_pure" if condition == "pure math"
                                  else "solo_predicted_mixed"]
                if not math.isclose(predicted, expected, abs_tol=1e-10):
                    raise ValueError(f"Prediction disagrees with analysis for {model}/{name}/{condition}")
                correct = sum(bool(s["grading"]["target_correct"]) for s in samples)
                measured = correct / len(samples)
                meas_low, meas_high = wilson_interval(correct, len(samples))
                for bar, value, low, high, method in (
                    ("predicted", predicted, pred_low, pred_high, "calibration bootstrap"),
                    ("measured", measured, meas_low, meas_high, "Wilson"),
                ):
                    bars.append(dict(model=model, topology=name, condition=condition,
                                     bar=bar, value=value, ci_low=low, ci_high=high,
                                     interval=method, graphs=len(samples)))
            for metric, values in (("target", direct),
                                   ("target_adjusted_for_solo_accuracy", adjusted)):
                pooled[metric].extend(values)
                estimate = result[metric]["difference_mixed_minus_pure"]
                if metric == "target_adjusted_for_solo_accuracy":
                    low, high = result[metric]["calibration_aware_95"]
                    method = "cluster bootstrap + Jeffreys-binomial calibration"
                else:
                    low, high = result[metric]["cluster_bootstrap_95"]
                    method = "cluster bootstrap"
                differences.append(dict(model=model, topology=name, metric=metric,
                                        estimate=estimate, ci_low=low, ci_high=high,
                                        pairs=len(values), templates=len({task for task, _ in values}),
                                        interval=method))
        for metric, values in pooled.items():
            if metric == "target_adjusted_for_solo_accuracy":
                low, high = calibration_aware_interval(
                    pooled_calibration_records,
                    {key: [bool(value) for value in entries]
                     for key, entries in outcomes.items()},
                    draws=ADJUSTED_POOLED_DRAWS,
                    seed=20261017 + sum((index + 1) * ord(char)
                                        for index, char in enumerate(model)),
                )
                method = "cluster bootstrap + Jeffreys-binomial calibration"
            else:
                low, high = interval(values)
                method = "cluster bootstrap"
            differences.append(dict(model=model, topology="all three", metric=metric,
                                    estimate=statistics.mean(value for _, value in values),
                                    ci_low=low, ci_high=high, pairs=len(values),
                                    templates=len({task for task, _ in values}),
                                    interval=method))
    return bars, differences


def plot(rows: list[dict], path: Path) -> None:
    model = rows[0]["model"]
    lookup = {(row["topology"], row["condition"], row["bar"]): row for row in rows}
    fig, axes = plt.subplots(1, len(LABELS), figsize=(11.4, 3.5), sharey=True)
    for index, (ax, topology) in enumerate(zip(axes, LABELS.values())):
        selected = [lookup[topology, condition, bar]
                    for condition, _ in CONDITIONS for bar in ("predicted", "measured")]
        values = [100 * row["value"] for row in selected]
        lower = [100 * (row["value"] - row["ci_low"]) for row in selected]
        upper = [100 * (row["ci_high"] - row["value"]) for row in selected]
        for pair, (_condition, color) in enumerate(CONDITIONS):
            for offset, bar in enumerate(("predicted", "measured")):
                i = pair * 2 + offset
                ax.bar(BAR_POSITIONS[i], values[i], width=0.62,
                       color="white" if bar == "predicted" else color,
                       edgecolor=color, hatch="////" if bar == "predicted" else None,
                       linewidth=1.2, zorder=2)
        ax.errorbar(BAR_POSITIONS, values, yerr=[lower, upper], fmt="none",
                    ecolor="#222222", elinewidth=1.1, capsize=3.2, zorder=3)
        for position, value, top in zip(BAR_POSITIONS, values, upper):
            ax.text(position, value + top + 2, f"{value:.0f}", ha="center", va="bottom",
                    fontsize=8, color="#222222", zorder=4)
        for pair, (_condition, color) in enumerate(CONDITIONS):
            left, right = pair * 2, pair * 2 + 1
            height = max(values[left] + upper[left], values[right] + upper[right]) + 12
            ax.plot([BAR_POSITIONS[left], BAR_POSITIONS[left], BAR_POSITIONS[right],
                     BAR_POSITIONS[right]], [height - 3, height, height, height - 3],
                    color=color, linewidth=1, zorder=4)
            ax.text((BAR_POSITIONS[left] + BAR_POSITIONS[right]) / 2, height + 1,
                    f"Δ {values[left] - values[right]:.0f} pp", ha="center", va="bottom",
                    fontsize=8, color=color, zorder=4)
        ax.set_title(topology)
        ax.set_xticks(BAR_TICKS, ["pure math", "mixed"])
        ax.set_xlim(-0.58, 3.10)
        ax.set_ylim(0, 126)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("target accuracy (%)")
    from matplotlib.patches import Patch
    fig.suptitle(f"{model} — pure math and mixed 18-node graphs", fontsize=11)
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor="#555555", hatch="////",
              label="predicted from single-task rates"),
        Patch(facecolor="#555555", edgecolor="#555555", label="measured"),
    ], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it"])
    parser.add_argument("--output_dir", type=Path, default=Path("eval/plots/mixed_gap"))
    args = parser.parse_args()
    bars, differences = build_rows(args.models)
    if not bars:
        raise SystemExit("No models with complete graded mixed-gap results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in dict.fromkeys(row["model"] for row in bars):
        plot([row for row in bars if row["model"] == model],
             args.output_dir / f"mixed_gap_{model}.pdf")
    with (args.output_dir / "mixed_gap_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bars[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(bars)
    with (args.output_dir / "mixed_gap_paired_differences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(differences)
    for row in differences:
        print(f"{row['model']:16s} {row['topology']:10s} {row['metric']:34s} "
              f"{100 * row['estimate']:+5.1f} pp "
              f"[{100 * row['ci_low']:+5.1f}, {100 * row['ci_high']:+5.1f}] "
              f"({row['templates']} templates, {row['pairs']} pairs)")
    print(f"Wrote per-model figures and summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
