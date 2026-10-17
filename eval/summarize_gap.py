"""Print the gap statistics gap_analysis.md reports, from the finished prediction-gap runs.

Reads eval/results/gap/<model>/<category>/ with the loaders plot_prediction_gap.py
uses, so the statistics and the bars describe the same graphs. Per model, category
and topology:

1. The gap. The predicted bar carries its bootstrap standard error over calibration
   noise, the measured bar its binomial one, and their difference a z-score. A
   posterior interval backs the z-score up where calibration is near perfect: task
   rates are drawn from an empirical-Bayes beta prior, or from the pooled rate when
   the tasks show too little spread to fit one. The per-node penalty delta solves
   (single-target accuracy - delta) ** nodes = measured accuracy.
2. The topologies pooled, since all four share one predicted bar.
3. Shape and label effects: differences in measured accuracy between topologies,
   combined across categories by inverse variance.
4. The chain's root node alone and inside the graph, the chain's depth curve, and
   the per-node penalty early and late along it given every upstream value right.
5. Where each failed graph first went wrong, from replaying its get_input calls.

A node's value is what the model handed to its children in get_input requests,
falling back to its written answer, and for the target to its submission.

    python eval/summarize_gap.py --models Qwen3.8-27B GPT-OSS-120B gemma-4-31B-it
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

EVAL_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(EVAL_DIR / "plots"), str(EVAL_DIR.parent / "source")]
from parse_results import parse_tool_calls
from plot_prediction_gap import load_calibration, load_topology
from tasks.arbigraph_tools import InputTool

TOPOLOGIES = {
    "eighteen_chain_target": "chain",
    "eighteen_braid_target": "braid",
    "eighteen_fan_out_fan_in": "fan-out/in",
    "eighteen_braid_target_shuffled_labels": "braid shuffled",
}
CHAIN, BRAID, FAN, SHUFFLED = TOPOLOGIES
DRAWS = 20000


def fit_beta_prior(counts: dict[int, tuple[int, int]]) -> tuple[float, float]:
    """Maximum-likelihood beta-binomial prior over the per-task (correct, total) counts."""
    k = np.array([c for c, _ in counts.values()])
    n = np.array([t for _, t in counts.values()])

    def negative_log_likelihood(x):
        a, b = np.exp(x)
        return -np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                       + betaln(k + a, n - k + b) - betaln(a, b))

    result = minimize(negative_log_likelihood, x0=[1.0, 0.0], method="Nelder-Mead",
                      options=dict(xatol=1e-8, fatol=1e-10, maxiter=5000))
    return tuple(np.exp(result.x))


def rate_draws(counts: dict[int, tuple[int, int]], tasks: list[int], rng) -> np.ndarray:
    """Posterior draws of every task's rate, shape (DRAWS, tasks)."""
    a, b = fit_beta_prior(counts)
    if a + b < 1e5:
        return rng.beta([a + counts[t][0] for t in tasks],
                        [b + counts[t][1] - counts[t][0] for t in tasks], size=(DRAWS, len(tasks)))
    # The fit ran to a spike: the tasks show no spread, so draw one pooled rate.
    correct = sum(c for c, _ in counts.values())
    total = sum(n for _, n in counts.values())
    return np.repeat(rng.beta(correct + 0.5, total - correct + 0.5, size=DRAWS)[:, None], len(tasks), axis=1)


def node_value_right(sample: dict, number: int) -> bool:
    """Whether the value the model carried for task `number` was right."""
    if number == len(sample["node_outputs"]):
        return bool(sample["grading"]["target_correct"])
    reported = (sample.get("reported_correct") or {}).get(str(number))
    return bool(sample["grading"]["node_correct"][f"task_{number}"] if reported is None else reported)


def gaps(args, runs) -> dict:
    print("1. The gap per topology: predicted ± bootstrap SE, measured ± binomial SE, "
          "gap ± SE and z, posterior 95% interval, per-node penalty delta")
    rng = np.random.default_rng(args.seed)
    rows = {}
    for model, category in runs:
        calibration = load_calibration(args.results_root / model / category)
        rates = calibration.rates()
        counts = {t: (sum(v), len(v)) for t, v in calibration.outcomes.items()}
        tasks = sorted(counts)
        column = {t: i for i, t in enumerate(tasks)}
        correct, total = calibration.pool_accuracy()
        single = correct / total
        draws = rate_draws(counts, tasks, rng)
        repeats = sorted({n for _, n in counts.values()})
        print(f"  {model} {category}: {len(tasks)} tasks x {repeats} repeats, single target {single:.4f} "
              f"({correct}/{total}), lowest task rates {sorted(round(r, 2) for r in rates.values())[:3]}")
        for topology, short in TOPOLOGIES.items():
            data_dir = args.data_root / model / category / topology
            graphs, _ = load_topology(topology,
                                      args.results_root / model / category / f"agent_{topology}_{model}.json",
                                      data_dir / f"{topology}_dataset.json",
                                      data_dir / f"{topology}_input_graph.json")
            hits, n = graphs.measured()
            measured, predicted = hits / n, graphs.predicted(rates)
            resampler = random.Random(args.seed)
            se_predicted = float(np.std([graphs.predicted(calibration.resample(resampler)) for _ in range(2000)]))
            se_measured = math.sqrt(measured * (1 - measured) / n)
            se_gap = math.hypot(se_predicted, se_measured)
            index = np.array([[column[t] for t in tasks_] for tasks_ in graphs.ancestor_tasks])
            posterior = (np.prod(draws[:, index], axis=2).mean(axis=1)
                         - rng.beta(hits + 0.5, n - hits + 0.5, size=DRAWS))
            low, high = np.percentile(posterior, [2.5, 97.5])
            delta = single - measured ** (1 / graphs.node_count)
            rows[model, category, topology] = dict(predicted=predicted, measured=measured, hits=hits, n=n,
                                                   se_predicted=se_predicted, se_measured=se_measured,
                                                   single=single, nodes=graphs.node_count)
            print(f"    {short:15s} {n:3d} graphs  predicted {predicted:.3f} ± {se_predicted:.3f}  "
                  f"measured {measured:.3f} ± {se_measured:.3f}  gap {predicted - measured:+.3f} ± {se_gap:.3f}  "
                  f"z {(predicted - measured) / se_gap:4.1f}  [{low:+.3f}, {high:+.3f}]  "
                  f"P(gap <= 0) {np.mean(posterior <= 0):.4f}  delta {delta:.4f}")
    return rows


def pooled(runs, rows) -> None:
    print("\n2. Topologies pooled over one shared predicted bar")
    for model, category in runs:
        for name, topologies in (("chain + braid", (CHAIN, BRAID)), ("all four", tuple(TOPOLOGIES))):
            parts = [rows[model, category, t] for t in topologies]
            hits = sum(p["hits"] for p in parts)
            n = sum(p["n"] for p in parts)
            measured = hits / n
            predicted = np.mean([p["predicted"] for p in parts])
            se = math.hypot(parts[0]["se_predicted"], math.sqrt(measured * (1 - measured) / n))
            print(f"  {model:15s} {category:6s} {name:13s} predicted {predicted:.3f}  measured {measured:.3f} "
                  f"({n} graphs)  gap {predicted - measured:+.3f} ± {se:.3f}  z {(predicted - measured) / se:4.1f}  "
                  f"delta {parts[0]['single'] - measured ** (1 / parts[0]['nodes']):.4f}")


def effects(models, categories, rows) -> None:
    print("\n3. Shape and label effects: difference in measured accuracy, combined over categories")
    for model in models:
        for first, second in ((BRAID, CHAIN), (FAN, CHAIN), (SHUFFLED, BRAID)):
            parts = []
            for category in categories:
                a, b = rows[model, category, first], rows[model, category, second]
                parts.append((a["measured"] - b["measured"], math.hypot(a["se_measured"], b["se_measured"])))
            weights = [1 / se ** 2 for _, se in parts]
            combined = sum(w * d for w, (d, _) in zip(weights, parts)) / sum(weights)
            se = 1 / math.sqrt(sum(weights))
            print(f"  {model:15s} {TOPOLOGIES[first]} - {TOPOLOGIES[second]}: " + "  ".join(
                f"{category} {d:+.3f} ± {s:.3f} (z {d / s:+.2f})" for category, (d, s) in zip(categories, parts))
                + f"  | combined {combined:+.3f} ± {se:.3f} (z {combined / se:+.2f})")


def chain_depth(args, runs) -> None:
    print("\n4. The chain: root node alone and as node 1, depth curve (predicted / measured), "
          "and the per-node penalty given every upstream value right, depths 13-18 minus 1-6")
    rng = np.random.default_rng(args.seed)
    for model, category in runs:
        rates = load_calibration(args.results_root / model / category).rates()
        samples = json.loads((args.results_root / model / category
                              / f"agent_{CHAIN}_{model}.json").read_text())["samples"]
        right = np.array([[node_value_right(s, d) for d in range(1, 19)] for s in samples])
        calibrated = np.array([[rates[n["native_task_id"]] for n in s["node_outputs"]] for s in samples])
        alone, inside = calibrated[:, 0].mean(), right[:, 0].mean()
        curve = "  ".join(f"{d}: {np.cumprod(calibrated, axis=1)[:, d - 1].mean():.3f}/{right[:, d - 1].mean():.3f}"
                          for d in (1, 3, 6, 9, 12, 15, 18))

        def trend(rows_):
            penalties = []
            for depth in range(18):
                alive = right[rows_, :depth].all(axis=1)
                if alive.any():
                    penalties.append((depth + 1, calibrated[rows_, depth][alive].mean()
                                      - right[rows_, depth][alive].mean(), alive.sum()))

            def mean(low, high):
                chosen = [(p, w) for d, p, w in penalties if low <= d <= high]
                return np.average([p for p, _ in chosen], weights=[w for _, w in chosen])
            return mean(1, 6), mean(13, 18)

        early, late = trend(np.arange(len(samples)))
        boots = [np.subtract(*trend(rng.integers(0, len(samples), len(samples)))[::-1]) for _ in range(1000)]
        low, high = np.percentile(boots, [2.5, 97.5])
        z = (alone - inside) / math.sqrt(inside * (1 - inside) / len(samples))
        print(f"  {model:15s} {category:6s} root alone {alone:.4f}, as node 1 {inside:.4f}, "
              f"penalty {alone - inside:+.4f} (z {z:.1f}) | {curve}\n"
              f"  {'':22s} per-node penalty, depths 1-6 {early:.3f}, 13-18 {late:.3f}, "
              f"change {(late - early) * 100:+.1f} pp [{low * 100:+.1f}, {high * 100:+.1f}]")


def request_kind(tool: InputTool, parents: list[int], dependencies) -> str:
    if not parents:
        return "right"
    if not (isinstance(dependencies, list) and len(dependencies) == len(parents)
            and all(isinstance(v, dict) and set(v) == {"result"} for v in dependencies)):
        return "misshapen"
    return "right" if all(tool.tasks[p]["same_answer"](v["result"], tool.tasks[p]["output"])
                          for p, v in zip(parents, dependencies)) else "wrong value"


def failures(args, runs) -> None:
    print("\n5. Where failed graphs first went wrong, and the per-node penalty on the true input")
    for model, category in runs:
        rates = load_calibration(args.results_root / model / category).rates()
        for topology, short in TOPOLOGIES.items():
            samples = json.loads((args.results_root / model / category
                                  / f"agent_{topology}_{model}.json").read_text())["samples"]
            first = collections.Counter()
            on_true_input = collections.Counter()
            for sample in samples:
                records = sample["node_outputs"]
                tool = InputTool.from_records(records, attempts_per_task=5)
                count = len(records)
                target = next(i for i, r in enumerate(records, 1) if r["node_id"] == sample["target_node_id"])
                requests = collections.defaultdict(list)
                carried = collections.defaultdict(set)
                for turn in sorted(sample["turn_outputs"] or {}, key=int):
                    for call in parse_tool_calls(sample["turn_outputs"][turn] or ""):
                        arguments = call.get("arguments") or {}
                        index = arguments.get("task_index") if isinstance(arguments, dict) else None
                        if call.get("name") != "get_input" or type(index) is not int or index not in tool.tasks:
                            continue
                        parents = tool.tasks[index]["parents"]
                        dependencies = arguments.get("input_dependencies")
                        requests[index].append(request_kind(tool, parents, dependencies))
                        if isinstance(dependencies, list) and len(dependencies) == len(parents):
                            for parent, value in zip(parents, dependencies):
                                if isinstance(value, dict) and set(value) == {"result"}:
                                    carried[parent].add(bool(tool.tasks[parent]["same_answer"](
                                        value["result"], tool.tasks[parent]["output"])))
                text = {i: bool(sample["grading"]["node_correct"][r["node_id"]]) for i, r in enumerate(records, 1)}
                text[target] = bool(sample["grading"]["target_correct"])
                value = {i: (True in carried[i]) if carried[i] and i != target else text[i]
                         for i in range(1, count + 1)}
                for i in range(1, count + 1):
                    if all(value[p] for p in tool.tasks[i]["parents"]) and "right" in requests[i]:
                        on_true_input["nodes"] += 1
                        on_true_input["right"] += value[i]
                        on_true_input["calibrated"] += rates[records[i - 1]["native_task_id"]]
                if sample["grading"]["target_correct"]:
                    continue
                depth: dict[int, int] = {}

                def level(i):
                    if i not in depth:
                        depth[i] = 1 + max((level(p) for p in tool.tasks[i]["parents"]), default=0)
                    return depth[i]

                # The first wrong value in dependency order, which label order need not follow.
                wrong = min((i for i in range(1, count + 1)
                             if not value[i] and all(value[p] for p in tool.tasks[i]["parents"])), key=level)
                kinds = requests[wrong]
                first["never asked for its input" if not kinds else "computed wrong on the true input"
                      if "right" in kinds else "handed a wrong parent value"] += 1
            failed = sum(first.values())
            penalty = (on_true_input["calibrated"] - on_true_input["right"]) / on_true_input["nodes"]
            print(f"  {model:15s} {category:6s} {short:15s} {failed:3d} failed: " + ", ".join(
                f"{k} {v} ({v / failed:.0%})" for k, v in first.most_common())
                + f" | penalty on the true input {penalty:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it"])
    parser.add_argument("--categories", nargs="+", default=["math", "python"])
    parser.add_argument("--results_root", type=Path, default=Path("eval/results/gap"))
    parser.add_argument("--data_root", type=Path, default=Path("generate/generated/gap"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = [(model, category) for model in args.models for category in args.categories]
    rows = gaps(args, runs)
    pooled(runs, rows)
    effects(args.models, args.categories, rows)
    chain_depth(args, runs)
    failures(args, runs)


if __name__ == "__main__":
    main()
