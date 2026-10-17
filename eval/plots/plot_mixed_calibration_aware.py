"""Plot pooled paired effects with calibration-aware adjusted intervals.

This exploratory plot reads the summary produced by plot_mixed_gap.py and is
kept in the project repository; it does not modify the paper repository.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "eval/plots/mixed_gap/mixed_gap_paired_differences.csv"
OUTPUT = ROOT / "eval/plots/mixed_gap/mixed_paired_effects_calibration_aware"
MODELS = ["Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it"]
LABELS = {
    "Qwen3.8-27B": "Qwen3.8-27B",
    "GPT-OSS-120B": "GPT-OSS-120B",
    "gemma-4-31B-it": "Gemma-4-31B-it",
}
COLORS = ["#326b9c", "#b14b65", "#6b6099"]
INK = "#253047"


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["model"], row["topology"], row["metric"]): row for row in rows}

    fig, ax = plt.subplots(figsize=(7.1, 2.9), constrained_layout=True)
    for index, model in enumerate(MODELS):
        y = 2 - index
        for metric, dy, marker, fill in (
            ("target", 0.14, "o", "white"),
            ("target_adjusted_for_solo_accuracy", -0.14, "s", COLORS[index]),
        ):
            row = lookup[model, "all three", metric]
            value, low, high = (100 * float(row[field]) for field in
                                ("estimate", "ci_low", "ci_high"))
            ax.plot([low, high], [y + dy, y + dy], color=COLORS[index], linewidth=1.8)
            ax.plot([low, low], [y + dy - 0.055, y + dy + 0.055],
                    color=COLORS[index], linewidth=1.1)
            ax.plot([high, high], [y + dy - 0.055, y + dy + 0.055],
                    color=COLORS[index], linewidth=1.1)
            ax.scatter([value], [y + dy], marker=marker, s=55, facecolor=fill,
                       edgecolor=COLORS[index], linewidth=1.35, zorder=3)
        ax.axhline(y - 0.48, color="#e7eaf0", linewidth=0.7)

    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlim(-35, 25)
    ax.set_ylim(-0.48, 2.48)
    ax.set_yticks([2, 1, 0], [LABELS[name] for name in MODELS])
    ax.set_xticks([-30, -20, -10, 0, 10, 20])
    ax.set_xlabel("Mixed minus matched pure math (percentage points)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8, colors=INK, length=0)
    ax.grid(axis="x", color="#e5e9ef", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b8c1cd")
    ax.legend(handles=[
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK,
               linestyle="none", label="Direct paired effect"),
        Line2D([], [], marker="s", markerfacecolor=INK, markeredgecolor=INK,
               linestyle="none", label="Solo-adjusted; calibration-aware interval"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False,
       fontsize=8)

    plt.rcParams["svg.fonttype"] = "none"
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
