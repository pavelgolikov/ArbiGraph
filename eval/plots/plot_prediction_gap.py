"""Plot solo accuracy and predicted/measured target accuracy for all gap topologies.

Each topology's prediction averages products of calibrated per-task accuracies
over the same sampled graphs used for its measured target accuracy. Run from the
repository root, for example:

    python eval/plots/plot_prediction_gap.py \
        --models Qwen3.8-27B GPT-OSS-120B gemma-4-31B-it \
        --categories math python --output_dir eval/plots/gap
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import collections
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# util and grade_results live in eval/, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util import sample_checkpoint_dir

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-arbigraph")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# One tick under each group; the hatch tells predicted from measured.
TOPOLOGY_NAMES = {
    "chain": "eighteen_chain_target",
    "braid": "eighteen_braid_target",
    "fan-out/in": "eighteen_fan_out_fan_in",
    "shuffled braid": "eighteen_braid_target_shuffled_labels",
}
GROUP_LABELS = ["single\ntarget", "chain", "braid", "fan-out/in", "shuffled\nbraid"]
LEGEND_LABELS = ["predicted from single-task rates", "measured"]
BAR_POSITIONS = np.array([0.0] + [1.35 + 1.75 * i + offset for i in range(4)
                                   for offset in (0.0, 0.72)])
GROUP_TICKS = [0.0] + [1.71 + 1.75 * i for i in range(4)]

# The paper's existing palette, from scripts/plot_qwen35_results.py.
SINGLE_COLOR = "#4a4a4a"
TOPOLOGY_COLORS = ("#00a189", "#ab1368", "#4263a0", "#c06c23")
BAR_COLORS = [SINGLE_COLOR] + [color for color in TOPOLOGY_COLORS for _ in range(2)]
PREDICTED_BARS = (1, 3, 5, 7)

BOOTSTRAP_DRAWS = 2000
Z = 1.96


@dataclass
class Calibration:
    """Per-task single-target outcomes: one list of 0/1 rollouts per task id."""

    outcomes: dict[int, list[int]]

    def rates(self) -> dict[int, float]:
        return {task: sum(runs) / len(runs) for task, runs in self.outcomes.items()}

    def pool_accuracy(self) -> tuple[int, int]:
        correct = sum(sum(runs) for runs in self.outcomes.values())
        total = sum(len(runs) for runs in self.outcomes.values())
        return correct, total

    def resample(self, rng: random.Random) -> dict[int, float]:
        """One bootstrap replicate of the per-task rates."""
        return {
            task: sum(rng.choice(runs) for _ in runs) / len(runs)
            for task, runs in self.outcomes.items()
        }


@dataclass
class Topology:
    """One topology's samples: the ancestor task ids and the target outcome."""

    name: str
    ancestor_tasks: list[list[int]]
    target_correct: list[int]
    node_count: int = 0

    def measured(self) -> tuple[int, int]:
        return sum(self.target_correct), len(self.target_correct)

    def predicted(self, rates: dict[int, float]) -> float:
        products = [
            math.prod(rates[task] for task in tasks)
            for tasks in self.ancestor_tasks
        ]
        return sum(products) / len(products)


@dataclass
class Panel:
    model: str
    category: str
    calibration: Calibration
    topologies: dict[str, Topology]
    bars: list[float] = field(default_factory=list)
    lows: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)


def wilson_interval(correct: int, total: int, z: float = Z) -> tuple[float, float]:
    """A 95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return math.nan, math.nan
    rate = correct / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def bootstrap_interval(panel: Panel, topology: Topology, seed: int) -> tuple[float, float]:
    """A 95% percentile interval for a predicted bar, over calibration noise.

    The graphs are held fixed and the calibration rollouts are resampled, so the
    interval covers what a finite number of single-target repeats leaves unknown
    about the per-task accuracies the product is built from.
    """
    rng = random.Random(seed)
    replicates = [
        topology.predicted(panel.calibration.resample(rng))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return float(np.percentile(replicates, 2.5)), float(np.percentile(replicates, 97.5))


def fill_panel(panel: Panel, seed: int) -> Panel:
    rates = panel.calibration.rates()
    single_correct, single_total = panel.calibration.pool_accuracy()
    panel.bars = [single_correct / single_total]
    intervals = [wilson_interval(single_correct, single_total)]
    for index, topology in enumerate(panel.topologies.values()):
        correct, total = topology.measured()
        panel.bars.extend((topology.predicted(rates), correct / total))
        intervals.extend((bootstrap_interval(panel, topology, seed + index),
                          wilson_interval(correct, total)))
    panel.lows = [low for low, _ in intervals]
    panel.highs = [high for _, high in intervals]
    return panel


def draw_panel(ax, panel: Panel, *, show_ylabel: bool, title: str) -> None:
    values = [value * 100 for value in panel.bars]
    lower = [(value - low) * 100 for value, low in zip(panel.bars, panel.lows)]
    upper = [(high - value) * 100 for value, high in zip(panel.bars, panel.highs)]
    positions = BAR_POSITIONS

    for index, position in enumerate(positions):
        predicted = index in PREDICTED_BARS
        ax.bar(
            position,
            values[index],
            width=0.62,
            color="white" if predicted else BAR_COLORS[index],
            edgecolor=BAR_COLORS[index],
            hatch="////" if predicted else None,
            linewidth=1.2,
            zorder=2,
        )
    ax.errorbar(
        positions,
        values,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.1,
        capsize=3.2,
        zorder=3,
    )

    for position, value, top in zip(positions, values, upper):
        ax.text(
            position, value + top + 2.0, f"{value:.0f}",
            ha="center", va="bottom", fontsize=8.0, color="#222222", zorder=4,
        )

    # One bracket per predicted/measured pair, labelled with the gap it spans.
    for index, color in enumerate(TOPOLOGY_COLORS):
        left, right = 1 + 2 * index, 2 + 2 * index
        gap = values[left] - values[right]
        height = max(values[left] + upper[left], values[right] + upper[right]) + 13.0
        ax.plot(
            [positions[left], positions[left], positions[right], positions[right]],
            [height - 3.0, height, height, height - 3.0],
            color=color, linewidth=1.0, zorder=4,
        )
        ax.text(
            (positions[left] + positions[right]) / 2, height + 1.0,
            f"Δ {gap:.0f} pp", ha="center", va="bottom",
            fontsize=8.5, color=color, zorder=4,
        )

    labels = list(GROUP_LABELS)
    for index, topology in enumerate(panel.topologies.values(), 1):
        labels[index] += f"\n({topology.node_count} nodes)"
    ax.set_xticks(GROUP_TICKS)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.65, BAR_POSITIONS[-1] + 0.65)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 124)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=9)
    if show_ylabel:
        ax.set_ylabel("target accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend_handles() -> list:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor="white", edgecolor="#555555", hatch="////", label=LEGEND_LABELS[0]),
        Patch(facecolor="#555555", edgecolor="#555555", label=LEGEND_LABELS[1]),
    ]


def write_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf",):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panels(panels: list[Panel], output_dir: Path) -> None:
    for panel in panels:
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        draw_panel(ax, panel, show_ylabel=True, title=f"{panel.model} — {panel.category}")
        # Below the axes, as in the combined figure: over bars near 100% the gap
        # brackets reach the top of the panel, where a legend inside would cover them.
        fig.tight_layout()
        fig.legend(handles=legend_handles(), fontsize=7.5, ncol=2, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.08), handlelength=1.8)
        write_figure(fig, output_dir, f"prediction_gap_{panel.model}_{panel.category}")

    by_model: dict[str, list[Panel]] = {}
    for panel in panels:
        by_model.setdefault(panel.model, []).append(panel)
    for model, model_panels in by_model.items():
        fig, axes = plt.subplots(
            1, len(model_panels), figsize=(7.6 * len(model_panels), 3.2), sharey=True,
        )
        axes = np.atleast_1d(axes)
        for index, (ax, panel) in enumerate(zip(axes, model_panels)):
            draw_panel(ax, panel, show_ylabel=index == 0, title=panel.category)
        fig.suptitle(model, fontsize=11)
        fig.legend(handles=legend_handles(), fontsize=9, ncol=2, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout()
        write_figure(fig, output_dir, f"prediction_gap_{model}")


def write_summary_csv(panels: list[Panel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rows of models not plotted in this run are kept, so each model can be
    # plotted on its own without erasing the others' numbers.
    plotted = {panel.model for panel in panels}
    kept = []
    if path.exists():
        with path.open(newline="") as handle:
            kept = [row for row in list(csv.reader(handle))[1:] if row and row[0] not in plotted]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "model", "category", "bar", "value", "ci_low", "ci_high",
            "interval", "denominator",
        ])
        writer.writerows(kept)
        for panel in panels:
            single_correct, single_total = panel.calibration.pool_accuracy()
            tasks = len(panel.calibration.outcomes)
            denominators = [f"{tasks} tasks x {single_total // tasks} rollouts"]
            kinds = ["wilson"]
            names = ["single_target"]
            for name, topology in panel.topologies.items():
                denominators.extend([f"{len(topology.target_correct)} graphs"] * 2)
                kinds.extend(("bootstrap", "wilson"))
                names.extend((f"{name}_predicted", f"{name}_measured"))
            for index, name in enumerate(names):
                writer.writerow([
                    panel.model, panel.category, name,
                    f"{panel.bars[index]:.4f}",
                    f"{panel.lows[index]:.4f}",
                    f"{panel.highs[index]:.4f}",
                    kinds[index], denominators[index],
                ])


# ---------------------------------------------------------------------------
# Real results
# ---------------------------------------------------------------------------

BASELINE_GLOB = "agent_single_target_baseline*.json"


def _target_correct(sample: dict) -> bool:
    return bool((sample.get("grading") or {})["target_correct"])


def load_calibration(results_dir: Path) -> Calibration:
    """Pool every single-target result file in one category into per-task rollouts.

    A category's calibration can arrive in several files: pinning makes sample_idx
    a label rather than an identity, so repeats of one task split across runs or
    stacks are all draws on the same problem and concatenate. Files are keyed by
    task, never by sample_idx, which repeats from zero in every file.
    """
    outcomes: dict[int, list[int]] = collections.defaultdict(list)
    paths = sorted(results_dir.glob(BASELINE_GLOB))
    if not paths:
        raise SystemExit(f"No {BASELINE_GLOB} under {results_dir}")
    for path in paths:
        for sample in json.loads(path.read_text())["samples"]:
            outcomes[int(sample["target_native_task_id"])].append(int(_target_correct(sample)))
    sizes = {len(runs) for runs in outcomes.values()}
    if len(sizes) != 1:
        print(f"  warning: uneven repeats per task in {results_dir.name}: {sorted(sizes)}")
    return Calibration(dict(outcomes))


def ancestor_closure(graph_path: Path) -> tuple[list[str], str]:
    """The target's ancestors plus the target. Nodes off that path do not enter."""
    graph = json.loads(graph_path.read_text())
    parents: dict[str, list[str]] = {node["id"]: [] for node in graph["nodes"]}
    for edge in graph.get("edges", []):
        parents[edge["destination"]].append(edge["source"])
    target = graph["target"]
    if isinstance(target, dict):
        target = target["id"]
    seen, stack = {target}, [target]
    while stack:
        for parent in parents[stack.pop()]:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return sorted(seen), target


def _grade_partial(results_path: Path, dataset_path: Path) -> list[dict]:
    """Grade an unfinished run from its per-sample checkpoints.

    Only episodes the agent actually finished are counted; one still in flight has
    no submission yet and would otherwise read as a failure.
    """
    from grade_results import grade_value

    checkpoints = sample_checkpoint_dir(str(results_path))
    dataset = json.loads(dataset_path.read_text())["samples"]
    graded = []
    for name in sorted(os.listdir(checkpoints)):
        match = re.fullmatch(r"sample_(\d+)\.json", name)
        if not match:
            continue
        state = json.loads((Path(checkpoints) / name).read_text())
        if (state.get("agent_status") or "in_progress") == "in_progress":
            continue
        sample = dataset[int(match.group(1))]
        node = sample["target_node_id"]
        truth = next(r["output_value"] for r in sample["node_outputs"] if r["node_id"] == node)
        target_number = next(
            position for position, record in enumerate(sample["node_outputs"], 1)
            if record["node_id"] == node
        )
        submitted = (state.get("submitted_answers") or {}).get(f"task_{target_number}_out")
        correct = bool(submitted) and grade_value(submitted.get("answer"), truth)
        graded.append({**sample, "grading": {"target_correct": correct}})
    return graded


def load_topology(name: str, results_path: Path, dataset_path: Path,
                  graph_path: Path) -> tuple[Topology, bool]:
    """One topology's samples. Falls back to checkpoints while a run is in flight."""
    nodes, _target = ancestor_closure(graph_path)
    provisional = False
    samples = json.loads(results_path.read_text())["samples"] if results_path.exists() else []
    if not samples or any("grading" not in s for s in samples):
        samples = _grade_partial(results_path, dataset_path)
        provisional = True

    ancestors, correct = [], []
    for sample in samples:
        by_node = {r["node_id"]: r["native_task_id"] for r in sample["node_outputs"]}
        ancestors.append([by_node[node] for node in nodes])
        correct.append(int(_target_correct(sample)))
    return Topology(name, ancestors, correct, node_count=len(nodes)), provisional


def real_panel(model: str, category: str, results_root: Path, data_root: Path) -> tuple[Panel, list[str]]:
    results_dir = results_root / model / category
    data_dir = data_root / model / category
    notes = []
    topologies = {}
    for key, topology in TOPOLOGY_NAMES.items():
        built, provisional = load_topology(
            topology,
            results_dir / f"agent_{topology}_{model}.json",
            data_dir / topology / f"{topology}_dataset.json",
            data_dir / topology / f"{topology}_input_graph.json",
        )
        topologies[key] = built
        if provisional:
            notes.append(f"{category}/{key} provisional: {len(built.target_correct)} finished samples")
    return Panel(
        model=model,
        category=category,
        calibration=load_calibration(results_dir),
        topologies=topologies,
    ), notes


# ---------------------------------------------------------------------------
# Placeholder data
# ---------------------------------------------------------------------------

# Per-task single-target accuracies observed in eval/results/exp1 at eight
# rollouts per task, used here only to give the placeholder figure the shape the
# real one will have.
PILOT_RATE_COUNTS = {
    "math": {0.75: 2, 0.875: 5, 1.0: 33},
    "python": {0.375: 2, 0.625: 3, 0.75: 4, 0.875: 14, 1.0: 57},
    "rlve": {0.125: 1, 0.375: 1, 0.5: 1, 0.875: 6, 1.0: 31},
}
# What one node costs beyond its own error rate, from the pilot eighteen-chain
# runs: measured target accuracy over predicted, taken to the power one over the
# number of ancestors.
PILOT_NODE_PENALTY = {"math": 0.972, "python": 0.968, "rlve": 0.985}


def fake_panel(
    model: str,
    category: str,
    repeats: int,
    graphs: int,
    chain_nodes: int,
    complex_nodes: int,
    rng: random.Random,
) -> Panel:
    """Build one panel from synthetic rollouts shaped like the pilot results."""
    truths = [
        rate
        for rate, count in PILOT_RATE_COUNTS[category].items()
        for _ in range(count)
    ]
    # A task observed perfect over eight rollouts is not perfect; place it just
    # below one so that the placeholder does not promise a noiseless product.
    truths = [min(rate, 0.99) for rate in truths]
    rng.shuffle(truths)
    truth = dict(enumerate(truths))

    calibration = Calibration({
        task: [1 if rng.random() < rate else 0 for _ in range(repeats)]
        for task, rate in truth.items()
    })

    def sample_topology(name: str, nodes: int) -> Topology:
        penalty = PILOT_NODE_PENALTY[category]
        ancestors, outcomes = [], []
        for _ in range(graphs):
            tasks = rng.sample(list(truth), nodes)
            success = math.prod(truth[task] * penalty for task in tasks)
            ancestors.append(tasks)
            outcomes.append(1 if rng.random() < success else 0)
        return Topology(name, ancestors, outcomes, node_count=nodes)

    return Panel(
        model=model,
        category=category,
        calibration=calibration,
        topologies={name: sample_topology(name, chain_nodes if name == "chain" else complex_nodes)
                    for name in TOPOLOGY_NAMES},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="draw placeholder numbers")
    parser.add_argument("--models", nargs="+",
                        default=["Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it"])
    parser.add_argument("--categories", nargs="+", default=["math", "python"])
    parser.add_argument("--repeats", type=int, default=32,
                        help="single-target rollouts per task in the placeholder")
    parser.add_argument("--graphs", type=int, default=96,
                        help="sampled graphs per topology in the placeholder")
    parser.add_argument("--chain_nodes", type=int, default=12)
    parser.add_argument("--complex_nodes", type=int, default=12)
    parser.add_argument("--results_root", type=Path, default=Path("eval/results/gap"))
    parser.add_argument("--data_root", type=Path, default=Path("generate/generated/gap"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval/plots/gap"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fake:
        panels, notes = [], []
        for model in args.models:
            for category in args.categories:
                if not (args.results_root / model / category).is_dir():
                    continue
                panel, panel_notes = real_panel(
                    model, category, args.results_root, args.data_root,
                )
                panels.append(fill_panel(panel, args.seed))
                notes.extend(panel_notes)
        if not panels:
            raise SystemExit(f"No results under {args.results_root}")
        for panel in panels:
            names = ["single"] + [f"{name} {kind}" for name in panel.topologies
                                   for kind in ("pred", "meas")]
            print(f"  {panel.model}/{panel.category}: " + "  ".join(
                f"{n} {v:.3f}" for n, v in zip(names, panel.bars)))
        for note in notes:
            print(f"  {note}")
        plot_panels(panels, args.output_dir)
        write_summary_csv(panels, args.output_dir / "prediction_gap_summary.csv")
        print(f"Wrote figures and prediction_gap_summary.csv to {args.output_dir}")
        return

    rng = random.Random(args.seed)
    panels = [
        fill_panel(
            fake_panel(
                model, category, args.repeats, args.graphs,
                args.chain_nodes, args.complex_nodes, rng,
            ),
            seed=args.seed,
        )
        for model in args.models
        for category in args.categories
    ]
    plot_panels(panels, args.output_dir)
    write_summary_csv(panels, args.output_dir / "prediction_gap_summary.csv")
    print(f"Wrote {len(panels)} panels and a summary to {args.output_dir}")


if __name__ == "__main__":
    main()
