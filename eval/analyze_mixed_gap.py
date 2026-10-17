"""Paired analysis of mixed 18-node graphs against pure math graphs.

Bootstrap units are target-task templates (each has three graph repeats), so the
18 node outcomes inside a graph are never treated as independent observations.
With --topology all, also compare each topology's mixed-minus-pure effect with
the chain's effect using the same matched target/repeat keys.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGIES = ("eighteen_chain_target", "eighteen_braid_target",
              "eighteen_fan_out_fan_in")
RETAINED_MATH = tuple(i for i in range(1, 19) if i not in (2, 4, 6, 8, 10, 12, 14, 16, 17))


def load_samples(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for sample in data["samples"]:
        if not sample.get("grading"):
            raise ValueError(f"Ungraded sample in {path}")
        key = (int(sample["target_native_task_id"]), int(sample["sample_idx"]))
        if key in rows:
            raise ValueError(f"Duplicate {key} in {path}")
        rows[key] = sample
    return rows


def interval(values: list[tuple[int, float]], *, draws: int = 5000) -> tuple[float, float]:
    """Cluster bootstrap, with each target native task as one resampling unit."""
    clusters = defaultdict(list)
    for task_id, value in values:
        clusters[task_id].append(value)
    ids = sorted(clusters)
    rng = random.Random(20260924)
    reps = []
    for _ in range(draws):
        sampled = [v for _i in range(len(ids)) for v in clusters[rng.choice(ids)]]
        reps.append(statistics.mean(sampled))
    reps.sort()
    return reps[int(0.025 * draws)], reps[int(0.975 * draws)]


def pooled_interval(by_model: dict[str, list[tuple[int, float]]], draws: int = 5000):
    """Equal-model pooled estimate, resampling target templates within each model."""
    grouped = {}
    for model, pairs in by_model.items():
        grouped[model] = defaultdict(list)
        for task_id, value in pairs:
            grouped[model][task_id].append(value)
    rng = random.Random(20260925)
    reps = []
    for _ in range(draws):
        model_means = []
        for clusters in grouped.values():
            ids = sorted(clusters)
            sampled = [value for _i in ids for value in clusters[rng.choice(ids)]]
            model_means.append(statistics.mean(sampled))
        reps.append(statistics.mean(model_means))
    reps.sort()
    return reps[int(0.025 * draws)], reps[int(0.975 * draws)]


def solo_outcomes(model: str) -> dict[tuple[str, int], list[bool]]:
    """Read the Bernoulli outcomes used to estimate each solo-task rate."""
    outcomes: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for category in ("math", "python"):
        root = ROOT / "eval" / "results" / "gap" / model / category
        for path in sorted(root.glob("agent_single_target_baseline*.json")):
            for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
                if sample.get("grading"):
                    outcomes[category, int(sample["target_native_task_id"])].append(
                        bool(sample["grading"]["target_correct"])
                    )
    for category in ("rag", "rlve"):
        path = (ROOT / "eval" / "results" / "mixed_gap" / "calibration" / model /
                category / f"agent_{category}_calibration_{model}.json")
        for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            if not sample.get("grading"):
                raise ValueError(f"Ungraded calibration sample in {path}")
            outcomes[category, int(sample["target_native_task_id"])].append(
                bool(sample["grading"]["target_correct"])
            )
        category_counts = [values for (name, _), values in outcomes.items()
                           if name == category]
        if any(len(values) != 30 for values in category_counts):
            raise ValueError(f"Expected 30 held-out repeats per task in {path}")
    return dict(outcomes)


def solo_rates(model: str):
    """Read the maximum-likelihood solo-task rates used for point estimates."""
    rates = defaultdict(dict)
    for (category, task_id), outcomes in solo_outcomes(model).items():
        rates[category][task_id] = statistics.mean(outcomes)
    return rates


def task_key(record: dict, *, mixed: bool) -> tuple[str, int]:
    """Return the calibration key corresponding to a graph-node record."""
    if not mixed:
        category = "math"
    else:
        number = int(record["node_id"].removeprefix("task_"))
        category = ("math" if number in RETAINED_MATH else
                    "rag" if number == 17 else
                    "rlve" if number in (12, 14, 16) else "python")
    return category, int(record["native_task_id"])


def calibration_aware_interval(
    records: list[tuple[int, float, tuple[tuple[str, int], ...],
                        tuple[tuple[str, int], ...]]],
    outcomes: dict[tuple[str, int], list[bool]],
    *,
    draws: int = 5000,
    seed: int = 20260926,
) -> tuple[float, float]:
    """Combine target-template resampling with binomial calibration uncertainty.

    Each draw resamples target-task clusters and draws every solo-task probability
    independently from its Jeffreys Beta(x+1/2, n-x+1/2) posterior.  The latter
    retains uncertainty when a calibrated task has zero observed errors.
    """
    clusters = defaultdict(list)
    required = set()
    for task_id, direct, pure_keys, mixed_keys in records:
        clusters[task_id].append((direct, pure_keys, mixed_keys))
        required.update(pure_keys)
        required.update(mixed_keys)
    missing = required.difference(outcomes)
    if missing:
        raise ValueError(f"Missing solo calibration for {sorted(missing)}")
    ids = sorted(clusters)
    keys = sorted(required)
    rng = random.Random(seed)
    reps = []
    for _ in range(draws):
        probabilities = {}
        for key in keys:
            values = outcomes[key]
            successes = sum(values)
            probabilities[key] = rng.betavariate(
                successes + 0.5, len(values) - successes + 0.5
            )
        sampled = [record for _i in ids for record in clusters[rng.choice(ids)]]
        adjusted = []
        for direct, pure_keys, mixed_keys in sampled:
            predicted_pure = math.prod(probabilities[key] for key in pure_keys)
            predicted_mixed = math.prod(probabilities[key] for key in mixed_keys)
            adjusted.append(direct - (predicted_mixed - predicted_pure))
        reps.append(statistics.mean(adjusted))
    reps.sort()
    return reps[int(0.025 * draws)], reps[int(0.975 * draws)]


def analyze(model: str, topology: str):
    pure_path = (ROOT / "eval" / "results" / "gap" / model / "math" /
                 f"agent_{topology}_{model}.json")
    mixed_path = (ROOT / "eval" / "results" / "mixed_gap" / model / "mixed" /
                  f"agent_{topology}_{model}.json")
    pure, mixed = load_samples(pure_path), load_samples(mixed_path)
    if set(pure) != set(mixed):
        raise ValueError(f"{model}: pure and mixed results have different target/sample keys")
    target_differences = []
    adjusted_differences = []
    calibration_records = []
    math_differences = []
    rag_correct, rlve_correct = [], []
    calibrated = solo_rates(model)
    pure_predictions, mixed_predictions = [], []
    for (task_id, sample_idx), p in sorted(pure.items()):
        m = mixed[(task_id, sample_idx)]
        pure_nodes = {record["node_id"]: record for record in p["node_outputs"]}
        mixed_nodes = {record["node_id"]: record for record in m["node_outputs"]}
        for number in RETAINED_MATH:
            if pure_nodes[f"task_{number}"] != mixed_nodes[f"task_{number}"]:
                raise ValueError(f"{model} {task_id}/{sample_idx}: math node {number} changed")
        target_differences.append((task_id,
            int(m["grading"]["target_correct"]) - int(p["grading"]["target_correct"])))
        pure_predictions.append(math.prod(
            calibrated["math"][int(record["native_task_id"])]
            for record in p["node_outputs"]
        ))
        mixed_predictions.append(math.prod(
            calibrated["math" if int(record["node_id"].removeprefix("task_")) in RETAINED_MATH else
                       "rag" if record["node_id"] == "task_17" else
                       "rlve" if record["node_id"] in ("task_12", "task_14", "task_16") else "python"]
                      [int(record["native_task_id"])]
            for record in m["node_outputs"]
        ))
        pure_keys = tuple(task_key(record, mixed=False) for record in p["node_outputs"])
        mixed_keys = tuple(task_key(record, mixed=True) for record in m["node_outputs"])
        calibration_records.append((task_id, target_differences[-1][1],
                                    pure_keys, mixed_keys))
        adjusted_differences.append((task_id,
            target_differences[-1][1] - (mixed_predictions[-1] - pure_predictions[-1])))
        math_differences.append((task_id, statistics.mean(
            int(m["grading"]["node_correct"][f"task_{i}"])
            - int(p["grading"]["node_correct"][f"task_{i}"])
            for i in RETAINED_MATH
        )))
        rag_correct.append(bool(m["grading"]["node_correct"]["task_17"]))
        rlve_correct.extend(bool(m["grading"]["node_correct"][f"task_{i}"])
                            for i in (12, 14, 16))
    def metric(differences):
        return {"difference_mixed_minus_pure": statistics.mean(v for _, v in differences),
                "cluster_bootstrap_95": interval(differences)}
    result = {
        "model": model,
        "topology": topology,
        "pairs": len(pure),
        "target": metric(target_differences),
        "target_adjusted_for_solo_accuracy": metric(adjusted_differences),
        "retained_math_node_mean": metric(math_differences),
        "pure_target_accuracy": statistics.mean(s["grading"]["target_correct"] for s in pure.values()),
        "mixed_target_accuracy": statistics.mean(s["grading"]["target_correct"] for s in mixed.values()),
        "solo_predicted_pure": statistics.mean(pure_predictions),
        "solo_predicted_mixed": statistics.mean(mixed_predictions),
        "solo_predicted_mixed_minus_pure": statistics.mean(mixed_predictions)
                                            - statistics.mean(pure_predictions),
        "heldout_selected_task_rates": {
            category: calibrated[category] for category in ("rag", "rlve")
        },
        "mixed_rag_node_accuracy": statistics.mean(rag_correct),
        "mixed_rlve_node_accuracy": statistics.mean(rlve_correct),
        "mixed_context_overflows": sum(s.get("agent_status") == "context_length_exceeded"
                                       for s in mixed.values()),
        "mean_generated_tokens_pure": statistics.mean(s.get("token_length") or 0 for s in pure.values()),
        "mean_generated_tokens_mixed": statistics.mean(s.get("token_length") or 0 for s in mixed.values()),
    }
    result["target_adjusted_for_solo_accuracy"]["calibration_aware_95"] = (
        calibration_aware_interval(
            calibration_records,
            solo_outcomes(model),
            seed=20260926 + sum((index + 1) * ord(char)
                                for index, char in enumerate(model + topology)),
        )
    )
    return (result, target_differences, math_differences, adjusted_differences,
            calibration_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it"])
    parser.add_argument("--topology", default="eighteen_chain_target",
                        choices=(*TOPOLOGIES, "all"))
    parser.add_argument("--output", type=Path,
                        default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = (ROOT / "eval" / "results" / "mixed_gap" /
                       f"analysis_{args.topology}.json")
    topologies = TOPOLOGIES if args.topology == "all" else (args.topology,)
    analyzed = {(model, topology): analyze(model, topology)
                for topology in topologies for model in args.models}
    results = [analyzed[model, topology][0]
               for topology in topologies for model in args.models]

    def pooled_metric(pairs):
        return {
            "difference_mixed_minus_pure": statistics.mean(
                statistics.mean(value for _, value in model_pairs)
                for model_pairs in pairs.values()
            ),
            "stratified_cluster_bootstrap_95": pooled_interval(pairs),
        }

    if len(args.models) > 1:
        for topology in topologies:
            pooled = {"models": args.models, "topology": topology,
                      "weighting": "equal model weight"}
            for label, index in (("target", 1), ("retained_math_node_mean", 2),
                                 ("target_adjusted_for_solo_accuracy", 3)):
                pooled[label] = pooled_metric({
                    model: analyzed[model, topology][index] for model in args.models
                })
            results.append({"pooled": pooled})
        if len(topologies) > 1:
            pooled = {"models": args.models, "topologies": topologies,
                      "weighting": "equal model and topology weight",
                      "cluster": "target task within model, including all topology repeats"}
            for label, index in (("target", 1), ("retained_math_node_mean", 2),
                                 ("target_adjusted_for_solo_accuracy", 3)):
                pooled[label] = pooled_metric({
                    model: [pair for topology in topologies
                            for pair in analyzed[model, topology][index]]
                    for model in args.models
                })
            results.append({"pooled_across_topologies": pooled})
    if len(topologies) > 1:
        reference = topologies[0]
        for model in args.models:
            reference_path = (ROOT / "eval/results/mixed_gap" / model / "mixed" /
                              f"agent_{reference}_{model}.json")
            reference_keys = set(load_samples(reference_path))
            for other in topologies[1:]:
                other_path = (ROOT / "eval/results/mixed_gap" / model / "mixed" /
                              f"agent_{other}_{model}.json")
                if set(load_samples(other_path)) != reference_keys:
                    raise ValueError(f"Unmatched topology sample keys for {model}: {other} vs {reference}")
                contrast = {"model": model, "reference": reference, "other": other,
                            "meaning": "mixed-minus-pure effect in other minus reference"}
                for label, index in (("target", 1),
                                     ("target_adjusted_for_solo_accuracy", 3)):
                    other_pairs = analyzed[model, other][index]
                    reference_pairs = analyzed[model, reference][index]
                    if len(other_pairs) != len(reference_pairs) or any(
                        other_task != reference_task
                        for (other_task, _), (reference_task, _) in zip(other_pairs, reference_pairs)
                    ):
                        raise ValueError(f"Unmatched topology samples for {model}: {other} vs {reference}")
                    paired_contrast = [
                        (task, other_value - reference_value)
                        for (task, other_value), (_, reference_value)
                        in zip(other_pairs, reference_pairs)
                    ]
                    contrast[label] = {
                        "difference_other_minus_reference": statistics.mean(
                            value for _, value in paired_contrast),
                        "cluster_bootstrap_95": interval(paired_contrast),
                    }
                results.append({"topology_contrast": contrast})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        if "pooled" in result:
            print("pooled", result["pooled"]["topology"],
                  result["pooled"]["target_adjusted_for_solo_accuracy"])
            continue
        if "pooled_across_topologies" in result:
            print("pooled across topologies",
                  result["pooled_across_topologies"]["target_adjusted_for_solo_accuracy"])
            continue
        if "topology_contrast" in result:
            contrast = result["topology_contrast"]
            print("topology contrast", contrast["model"], contrast["other"],
                  "minus", contrast["reference"], contrast["target"])
            continue
        low, high = result["target"]["cluster_bootstrap_95"]
        print(result["model"], result["topology"], "pairs", result["pairs"],
              "target mixed-pure", f"{result['target']['difference_mixed_minus_pure']:+.3f}",
              f"95% cluster CI [{low:+.3f}, {high:+.3f}]")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
