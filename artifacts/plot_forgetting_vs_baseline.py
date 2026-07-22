"""Plot forgetting accuracy with latest repair baseline overlays."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cmb")

from plot_forgetting_accuracy import (
    COLORS,
    DATASET_LABELS,
    DATASETS,
    MODELS,
    ResultSlot,
    collect_results,
)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def write_comparison_csv(
    path: Path,
    forgetting: dict[tuple[str, str], ResultSlot],
    baseline: dict[tuple[str, str], ResultSlot],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "dataset",
                "forgetting_accuracy",
                "baseline_repair_accuracy",
                "forgetting_note",
                "baseline_note",
            ]
        )
        for model in MODELS:
            for dataset in DATASETS:
                forget_slot = forgetting[(model, dataset)]
                base_slot = baseline[(model, dataset)]
                writer.writerow(
                    [
                        model,
                        dataset,
                        "NA" if forget_slot.accuracy is None else forget_slot.accuracy,
                        "NA" if base_slot.accuracy is None else base_slot.accuracy,
                        forget_slot.note,
                        base_slot.note,
                    ]
                )


def plot_comparison(
    forgetting: dict[tuple[str, str], ResultSlot],
    baseline: dict[tuple[str, str], ResultSlot],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    x = np.arange(len(MODELS))
    width = 0.23
    offsets = {"gsm": -width, "python": 0.0, "math": width}

    for dataset in DATASETS:
        positions = x + offsets[dataset]
        forget_values = [forgetting[(model, dataset)].accuracy for model in MODELS]
        base_values = [baseline[(model, dataset)].accuracy for model in MODELS]
        heights = [value * 100 if value is not None else 0 for value in forget_values]

        bars = ax.bar(
            positions,
            heights,
            width,
            color=COLORS[dataset],
            edgecolor="#222222",
            linewidth=0.65,
            label=DATASET_LABELS[dataset],
            zorder=2,
        )

        for position, bar, forget_value, base_value in zip(positions, bars, forget_values, base_values):
            if forget_value is None:
                bar.set_facecolor("#D8D8D8")
                bar.set_hatch("//")
                ax.text(
                    position,
                    2.0,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#555555",
                    zorder=4,
                )
                continue

            forget_percent = forget_value * 100
            if base_value is not None:
                base_percent = base_value * 100
                ax.bar(
                    position,
                    base_percent,
                    width,
                    color=COLORS[dataset],
                    alpha=0.24,
                    edgecolor="#111111",
                    linewidth=1.15,
                    zorder=3,
                )
                label = f"{forget_percent:.1f} / {base_percent:.1f}"
                y = max(forget_percent, base_percent) + 1.4
            else:
                label = f"{forget_percent:.1f} / NA"
                y = forget_percent + 1.4

            ax.text(position, y, label, ha="center", va="bottom", fontsize=8.5, color="#222222", zorder=4)

    ax.set_title("Forgetting vs. Baseline Repair Accuracy by Model", fontsize=15, pad=14)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 111)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    dataset_handles = [
        mpatches.Patch(facecolor=COLORS[dataset], edgecolor="#222222", label=DATASET_LABELS[dataset])
        for dataset in DATASETS
    ]
    baseline_handle = mpatches.Patch(
        facecolor="#777777",
        alpha=0.24,
        edgecolor="#111111",
        label="Baseline repair overlay",
    )
    ax.legend(
        handles=dataset_handles + [baseline_handle],
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
    )

    fig.text(
        0.5,
        0.018,
        "Solid bars are forgetting results; transparent overlays are latest complete baseline repair results. "
        "Labels are forgetting / baseline accuracy.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    fig.savefig(output_dir / "forgetting_vs_baseline_accuracy_by_model.png", dpi=200)
    fig.savefig(output_dir / "forgetting_vs_baseline_accuracy_by_model.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    forgetting = collect_results(repo_root, "forgetting")
    baseline = collect_results(repo_root, "baseline")
    plot_comparison(forgetting, baseline, output_dir)
    write_comparison_csv(output_dir / "forgetting_vs_baseline_accuracy_by_model.csv", forgetting, baseline)
    print(f"Wrote forgetting-vs-baseline plot and CSV to {output_dir}")


if __name__ == "__main__":
    main()
