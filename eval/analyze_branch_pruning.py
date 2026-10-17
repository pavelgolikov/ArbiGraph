"""Analyze matched 18-task braid peninsula/island runs.

Primary outcome: the island rate of verified completion of the seven upstream
optional nodes. The prespecified meaningful-rate threshold is 10%. Math and
Python receive equal weight within each model. Resampling and paired sign flips
keep each target template's three graph assignments together. Gemma and GPT-OSS
are the two planned model tests; Holm adjustment controls their familywise
error rate. The peninsula-island contrast is secondary.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from parse_results import parse_tool_calls

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODELS = ("gemma-4-31B-it", "GPT-OSS-120B")
MODELS = (*PRIMARY_MODELS, "Qwen3.8-27B")
CATEGORIES = ("math", "python")
CONDITIONS = ("peninsula", "island")
OPTIONAL = {"task_3", "task_10", "task_11", "task_12", "task_13", "task_14", "task_15", "task_17"}
VERIFIABLE_OPTIONAL = OPTIONAL - {"task_17"}
PRIMARY = "all_seven_verified"
PRIMARY_THRESHOLD = 0.10
CONTRAST = "peninsula_minus_island"
BASELINE_CONTRAST = "full_braid_minus_island"
METRICS = ("task17_requested", "any_optional_requested", "all_seven_verified",
           "optional_nodes_requested", "target_correct", "agent_tool_calls")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(model: str, category: str, condition: str, results_root: Path) -> Path:
    name = f"eighteen_braid_{condition}_target"
    return results_root / model / category / f"agent_{name}_{model}.json"


def dataset_path(model: str, category: str, condition: str) -> Path:
    name = f"eighteen_braid_{condition}_target"
    return ROOT / "generate" / "generated" / "branch_pruning" / model / category / name / f"{name}_dataset.json"


def baseline_paths(model: str, category: str) -> tuple[Path, Path]:
    name = "eighteen_braid_target"
    dataset = (ROOT / "generate" / "generated" / "gap" / model / category / name
               / f"{name}_dataset.json")
    result = (ROOT / "eval" / "results" / "gap" / model / category
              / f"agent_{name}_{model}.json")
    return dataset, result


def serving_stack(path: Path) -> str:
    params = read_json(path)["summary"].get("generation_params", {})
    return "api" if params.get("api_model") else "local"


def key(sample: dict) -> tuple[int, int]:
    return int(sample["target_native_task_id"]), int(sample["sample_idx"])


def requests(sample: dict) -> set[str]:
    records = sample["node_outputs"]
    requested = set()
    for turn in sorted(sample.get("turn_outputs") or {}, key=int):
        for call in parse_tool_calls(sample["turn_outputs"][turn] or ""):
            if call.get("name") != "get_input":
                continue
            arguments = call.get("arguments") or {}
            index = arguments.get("task_index") if isinstance(arguments, dict) else None
            if type(index) is int and 1 <= index <= len(records):
                requested.add(records[index - 1]["node_id"])
    return requested


def measures(sample: dict) -> dict[str, float]:
    records = sample["node_outputs"]
    visible = {record["node_id"]: number for number, record in enumerate(records, 1)}
    asked = requests(sample)
    reported = sample.get("reported_correct") or {}
    return {
        "task17_requested": float("task_17" in asked),
        "any_optional_requested": float(bool(OPTIONAL & asked)),
        "all_seven_verified": float(all(reported.get(str(visible[node])) is True
                                          for node in VERIFIABLE_OPTIONAL)),
        "optional_nodes_requested": float(len(OPTIONAL & asked)),
        "target_correct": float(sample["grading"]["target_correct"]),
        "agent_tool_calls": float(sample["agent_tool_calls"]),
    }


def load_pairs(model: str, category: str, allow_partial: bool,
               results_root: Path) -> list[dict]:
    by_condition = {}
    expected = None
    for condition in CONDITIONS:
        dataset = read_json(dataset_path(model, category, condition))
        expected_keys = {key(sample) for sample in dataset["samples"]}
        if expected is None:
            expected = expected_keys
        elif expected != expected_keys:
            raise ValueError(f"Unmatched generated datasets for {model}/{category}")
        result = read_json(result_path(model, category, condition, results_root))
        if len(result["samples"]) != len({key(sample) for sample in result["samples"]}):
            raise ValueError(f"Duplicate result keys for {model}/{category}/{condition}")
        by_condition[condition] = {key(sample): sample for sample in result["samples"]
                                   if sample.get("grading") is not None}
        if not set(by_condition[condition]) <= expected:
            raise ValueError(f"Unexpected result keys for {model}/{category}/{condition}")
        if not allow_partial and set(by_condition[condition]) != expected:
            raise ValueError(f"Incomplete result for {model}/{category}/{condition}")
    baseline_dataset_path, baseline_result_path = baseline_paths(model, category)
    baseline_dataset = read_json(baseline_dataset_path)
    if {key(sample) for sample in baseline_dataset["samples"]} != expected:
        raise ValueError(f"Full-braid baseline dataset keys differ for {model}/{category}")
    baseline_result = read_json(baseline_result_path)
    baseline = {key(sample): sample for sample in baseline_result["samples"]
                if sample.get("grading") is not None}
    if set(baseline) != expected:
        raise ValueError(f"Full-braid baseline results are incomplete for {model}/{category}")
    paired_keys = sorted(set.intersection(*(set(rows) for rows in by_condition.values())))
    if not paired_keys:
        raise ValueError(f"No completed pairs for {model}/{category}")
    pairs = []
    for item_key in paired_keys:
        a, b = (by_condition[condition][item_key] for condition in CONDITIONS)
        full = baseline[item_key]
        if [(r["node_id"], r["native_task_id"], r["input_value"], r["output_value"])
            for r in a["node_outputs"]] != [
                (r["node_id"], r["native_task_id"], r["input_value"], r["output_value"])
                for r in b["node_outputs"]]:
            raise ValueError(f"Pinned tasks differ in {model}/{category}/{item_key}")
        if [(r["node_id"], r["native_task_id"], r["input_value"], r["output_value"])
            for r in a["node_outputs"]] != [
                (r["node_id"], r["native_task_id"], r["input_value"], r["output_value"])
                for r in full["node_outputs"]]:
            raise ValueError(f"Full-braid pinned tasks differ in {model}/{category}/{item_key}")
        pairs.append({"template": item_key[0], "sample_idx": item_key[1],
                      "peninsula": measures(a), "island": measures(b),
                      "full_braid": measures(full)})
    return pairs


def grouped_values(pairs: list[dict], metric: str, condition: str) -> dict[int, np.ndarray]:
    groups = defaultdict(list)
    for pair in pairs:
        if condition == "difference":
            value = pair["peninsula"][metric] - pair["island"][metric]
        elif condition == "full_braid_minus_island":
            value = pair["full_braid"][metric] - pair["island"][metric]
        else:
            value = pair[condition][metric]
        groups[pair["template"]].append(value)
    return {template: np.array(values, dtype=float) for template, values in groups.items()}


def estimate(groups_by_category: dict[str, dict[int, np.ndarray]], rng: np.random.Generator,
             draws: int) -> dict:
    category_means = {}
    category_draws = {}
    for category, groups in groups_by_category.items():
        cluster_means = np.array([values.mean() for values in groups.values()])
        category_means[category] = float(cluster_means.mean())
        selections = rng.integers(0, len(cluster_means), size=(draws, len(cluster_means)))
        category_draws[category] = cluster_means[selections].mean(axis=1)
    pooled_draws = np.mean(list(category_draws.values()), axis=0)
    pooled_mean = float(np.mean(list(category_means.values())))
    return {"mean": pooled_mean,
            "ci95": [float(np.quantile(pooled_draws, 0.025)),
                     float(np.quantile(pooled_draws, 0.975))],
            "category_means": category_means,
            "category_ci95": {category: [float(np.quantile(samples, 0.025)),
                                         float(np.quantile(samples, 0.975))]
                              for category, samples in category_draws.items()}}


def paired_cluster_pvalue(groups_by_category: dict[str, dict[int, np.ndarray]],
                          observed: float, seed: int, draws: int) -> float:
    rng = np.random.default_rng(seed)
    null_parts = []
    for groups in groups_by_category.values():
        cluster_means = np.array([values.mean() for values in groups.values()])
        signs = rng.choice((-1, 1), size=(draws, len(cluster_means)))
        null_parts.append((signs * cluster_means).mean(axis=1))
    null = np.mean(null_parts, axis=0)
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)) / (draws + 1))


def meaningful_rate_pvalue(groups_by_category: dict[str, dict[int, np.ndarray]],
                           observed: float, threshold: float) -> float:
    """One-sided cluster-level Welch test for an equal-category mean rate."""
    variances, dfs = [], []
    for groups in groups_by_category.values():
        cluster_means = np.array([values.mean() for values in groups.values()])
        n = len(cluster_means)
        if n < 2:
            raise ValueError("At least two target templates are needed per category")
        variances.append(float(cluster_means.var(ddof=1) / (4 * n)))
        dfs.append(n - 1)
    variance = sum(variances)
    if variance == 0:
        return 0.0 if observed > threshold else 1.0
    df = variance**2 / sum(v**2 / d for v, d in zip(variances, dfs) if v > 0)
    statistic = (observed - threshold) / variance**0.5
    return float(student_t.sf(statistic, df))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=PRIMARY_MODELS)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Monitor completed pairs; do not publish partial-run inference")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "eval" / "results" / "branch_pruning")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "eval" / "results" / "branch_pruning" / "analysis.json")
    args = parser.parse_args()
    results_root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    if args.draws < 1000:
        parser.error("--draws must be at least 1000")

    rng = np.random.default_rng(20270925)
    output = {"primary_metric": PRIMARY, "primary_condition": "island",
              "primary_rate_threshold": PRIMARY_THRESHOLD,
              "secondary_contrast": CONTRAST,
              "descriptive_baseline_contrast": BASELINE_CONTRAST,
              "category_weighting": "equal", "cluster": "target_native_task_id",
              "partial": args.allow_partial, "results_root": str(results_root), "models": {}}
    for model in args.models:
        pairs = {category: load_pairs(model, category, args.allow_partial, results_root)
                 for category in CATEGORIES}
        report = {"paired_episodes": {category: len(rows) for category, rows in pairs.items()},
                  "target_templates": {category: len({p["template"] for p in rows})
                                       for category, rows in pairs.items()},
                  "serving_stack": {
                      category: {
                          "full_braid": serving_stack(baseline_paths(model, category)[1]),
                          "peninsula": serving_stack(result_path(model, category, "peninsula", results_root)),
                          "island": serving_stack(result_path(model, category, "island", results_root)),
                      } for category in CATEGORIES
                  }, "metrics": {}}
        for metric in METRICS:
            report["metrics"][metric] = {}
            for condition in (*CONDITIONS, "full_braid", "difference", BASELINE_CONTRAST):
                groups = {category: grouped_values(rows, metric, condition)
                          for category, rows in pairs.items()}
                report["metrics"][metric][condition] = estimate(groups, rng, args.draws)
                if metric == PRIMARY and condition == "island" and not args.allow_partial:
                    report["primary_pvalue"] = meaningful_rate_pvalue(
                        groups, report["metrics"][metric][condition]["mean"], PRIMARY_THRESHOLD)
                if metric == PRIMARY and condition == "difference" and not args.allow_partial:
                    report["secondary_contrast_pvalue"] = paired_cluster_pvalue(
                        groups, report["metrics"][metric][condition]["mean"],
                        20270925 + len(output["models"]), args.draws * 2)
        output["models"][model] = report

    if not args.allow_partial and set(output["models"]) == set(PRIMARY_MODELS):
        ordered = sorted(output["models"], key=lambda model: output["models"][model]["primary_pvalue"])
        p_small = output["models"][ordered[0]]["primary_pvalue"]
        p_large = output["models"][ordered[1]]["primary_pvalue"]
        output["models"][ordered[0]]["primary_pvalue_holm"] = min(1.0, 2 * p_small)
        output["models"][ordered[1]]["primary_pvalue_holm"] = max(
            output["models"][ordered[0]]["primary_pvalue_holm"], p_large)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for model, report in output["models"].items():
        primary = report["metrics"][PRIMARY]
        print(model, "pairs", report["paired_episodes"],
              "full braid seven verified", f"{primary['full_braid']['mean']:.3f}",
              "island seven verified", f"{primary['island']['mean']:.3f}",
              "island CI", primary["island"]["ci95"],
              "peninsula seven verified", f"{primary['peninsula']['mean']:.3f}",
              "difference", f"{primary['difference']['mean']:.3f}",
              "difference CI", primary["difference"]["ci95"],
              "primary p", report.get("primary_pvalue"),
              "Holm p", report.get("primary_pvalue_holm"))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
