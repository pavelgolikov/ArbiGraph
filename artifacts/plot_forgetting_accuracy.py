"""Plot forgetting benchmark accuracy by model and dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cmb")

import matplotlib.pyplot as plt
import numpy as np


MODELS = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-122B-A10B"]
DATASETS = ["gsm", "python", "math"]
DATASET_LABELS = {"gsm": "GSM", "python": "Python", "math": "Math"}
COLORS = {"gsm": "#4C78A8", "python": "#F58518", "math": "#54A24B"}


@dataclass
class ResultSlot:
    accuracy: Optional[float]
    path: Optional[Path]
    note: str


def processed_sample_count(samples: list[dict]) -> int:
    return sum(("correct" in sample and sample.get("agent_status") is not None) for sample in samples)


def select_latest_complete_result(
    repo_root: Path,
    result_group: str,
    model: str,
    dataset: str,
) -> ResultSlot:
    dataset_dir = "py" if result_group == "baseline" and dataset == "python" else dataset
    result_dir = repo_root / "results" / result_group / model / dataset_dir / "repair"
    candidates = []
    incomplete = []

    for path in sorted(result_dir.glob("*.json")):
        data = json.loads(path.read_text())
        samples = data.get("samples", [])
        processed = processed_sample_count(samples)
        accuracy = data.get("summary", {}).get("overall_accuracy")

        if samples and processed == len(samples) and accuracy is not None:
            candidates.append((path.stat().st_mtime, path, float(accuracy)))
        elif samples:
            incomplete.append((path.stat().st_mtime, path, processed, len(samples)))

    if candidates:
        _, path, accuracy = max(candidates, key=lambda item: item[0])
        return ResultSlot(accuracy=accuracy, path=path, note=path.relative_to(repo_root).as_posix())

    if incomplete:
        _, path, processed, total = max(incomplete, key=lambda item: item[0])
        note = f"NA: incomplete {processed}/{total} ({path.relative_to(repo_root).as_posix()})"
        return ResultSlot(accuracy=None, path=path, note=note)

    return ResultSlot(accuracy=None, path=None, note="NA: no result file")


def collect_results(repo_root: Path, result_group: str) -> dict[tuple[str, str], ResultSlot]:
    return {
        (model, dataset): select_latest_complete_result(repo_root, result_group, model, dataset)
        for model in MODELS
        for dataset in DATASETS
    }


def write_csv(path: Path, results: dict[tuple[str, str], ResultSlot]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "accuracy", "note"])
        for model in MODELS:
            for dataset in DATASETS:
                slot = results[(model, dataset)]
                writer.writerow([model, dataset, "NA" if slot.accuracy is None else slot.accuracy, slot.note])


def plot_forgetting(results: dict[tuple[str, str], ResultSlot], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(MODELS))
    width = 0.23
    offsets = {"gsm": -width, "python": 0.0, "math": width}

    for dataset in DATASETS:
        values = [results[(model, dataset)].accuracy for model in MODELS]
        positions = x + offsets[dataset]
        heights = [value * 100 if value is not None else 0 for value in values]
        bars = ax.bar(
            positions,
            heights,
            width,
            label=DATASET_LABELS[dataset],
            color=COLORS[dataset],
            edgecolor="#222222",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            center = bar.get_x() + bar.get_width() / 2
            if value is None:
                bar.set_facecolor("#D8D8D8")
                bar.set_hatch("//")
                ax.text(center, 2.0, "NA", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#555555")
            else:
                percent = value * 100
                ax.text(center, percent + 1.2, f"{percent:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title("Forgetting Benchmark Accuracy by Model", fontsize=15, pad=14)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 106)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.text(
        0.5,
        0.015,
        "Latest complete result per model/dataset. Hatched bars indicate missing or incomplete result files.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(output_dir / "forgetting_accuracy_by_model.png", dpi=200)
    fig.savefig(output_dir / "forgetting_accuracy_by_model.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(repo_root, "forgetting")
    plot_forgetting(results, output_dir)
    write_csv(output_dir / "forgetting_accuracy_by_model.csv", results)
    print(f"Wrote forgetting plot and CSV to {output_dir}")


if __name__ == "__main__":
    main()
