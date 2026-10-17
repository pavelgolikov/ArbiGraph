"""Check that serving stacks agree before their calibration rollouts are pooled.

Give it two or more results files from the same pinned calibration dataset, one
per stack or run. Each arm repeats the same problems, so any systematic
difference between arms is the stack, not the tasks. Reports:

1. Accuracy per arm, a chi-square test of homogeneity across arms, and pairwise
   two-proportion z-scores. A shift in rollouts between runs of one stack sets
   the scale a between-stack shift has to beat.
2. The per-task paired difference, averaged over tasks: what one stack gains or
   loses on the same problem.
3. Reasoning length per arm and the per-task geometric-mean token ratio. A stack
   that silently lowers reasoning effort shows up here before it shows up in
   accuracy.
4. With --topology, the predicted bar each arm alone would give for that
   topology's graphs, and the difference between them in units of its standard
   error. The predicted bar is a product over eighteen nodes, so it magnifies a
   per-task shift eighteen-fold; this is the test that decides whether pooling
   is safe.

    python eval/compare_stacks.py \
        --results eval/results/gap/GPT-OSS-120B/math/agent_single_target_baseline_GPT-OSS-120B.json \
                  eval/results/gap/GPT-OSS-120B/math/agent_single_target_baseline_api_GPT-OSS-120B.json \
        --topology generate/generated/gap/GPT-OSS-120B/math/eighteen_chain_target
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path

from scipy.stats import chi2_contingency

# ancestor_closure lives with the plotting script in eval/plots.
sys.path.insert(0, str(Path(__file__).resolve().parent / "plots"))
from plot_prediction_gap import ancestor_closure


def load_arm(path: Path) -> dict:
    samples = json.loads(path.read_text())["samples"]
    outcomes, tokens = collections.defaultdict(list), collections.defaultdict(list)
    statuses = collections.Counter()
    for sample in samples:
        task = int(sample["target_native_task_id"])
        outcomes[task].append(int(bool((sample.get("grading") or {}).get("target_correct"))))
        tokens[task].append(sample.get("token_length") or 0)
        statuses[sample.get("agent_status")] += 1
    return {"name": path.stem, "outcomes": dict(outcomes), "tokens": dict(tokens),
            "statuses": statuses, "turns": [s.get("agent_turns") or 0 for s in samples]}


def accuracy(arm: dict) -> tuple[int, int]:
    return (sum(sum(runs) for runs in arm["outcomes"].values()),
            sum(len(runs) for runs in arm["outcomes"].values()))


def two_proportion_z(a: tuple[int, int], b: tuple[int, int]) -> float:
    pooled = (a[0] + b[0]) / (a[1] + b[1])
    se = math.sqrt(pooled * (1 - pooled) * (1 / a[1] + 1 / b[1]))
    return (a[0] / a[1] - b[0] / b[1]) / se if se else 0.0


def paired_task_difference(a: dict, b: dict) -> tuple[float, float]:
    """Mean over tasks of the accuracy difference on the same problem, with its SE."""
    diffs, variances = [], []
    for task in sorted(set(a["outcomes"]) & set(b["outcomes"])):
        ra, rb = a["outcomes"][task], b["outcomes"][task]
        pa, pb = sum(ra) / len(ra), sum(rb) / len(rb)
        diffs.append(pa - pb)
        variances.append(pa * (1 - pa) / len(ra) + pb * (1 - pb) / len(rb))
    return sum(diffs) / len(diffs), math.sqrt(sum(variances)) / len(diffs)


def token_ratio(a: dict, b: dict) -> float:
    """Geometric mean over tasks of the ratio of mean generated tokens, a over b."""
    logs = [math.log(max(1.0, sum(a["tokens"][t]) / len(a["tokens"][t]))
                     / max(1.0, sum(b["tokens"][t]) / len(b["tokens"][t])))
            for t in sorted(set(a["tokens"]) & set(b["tokens"]))]
    return math.exp(sum(logs) / len(logs))


def predicted_bar(arm: dict, graphs: list[list[int]]) -> tuple[float, float]:
    """The topology's predicted bar from this arm alone, with its delta-method SE."""
    rates = {task: sum(runs) / len(runs) for task, runs in arm["outcomes"].items()}
    products = [math.prod(rates[task] for task in graph) for graph in graphs]
    weight = collections.defaultdict(float)
    for graph, product in zip(graphs, products):
        for task in graph:
            if rates[task] > 0:
                weight[task] += product / rates[task] / len(graphs)
    variance = sum(weight[task] ** 2 * rates[task] * (1 - rates[task]) / len(arm["outcomes"][task])
                   for task in weight)
    return sum(products) / len(products), math.sqrt(variance)


def topology_graphs(topology_dir: Path) -> list[list[int]]:
    name = topology_dir.name
    nodes, _target = ancestor_closure(topology_dir / f"{name}_input_graph.json")
    samples = json.loads((topology_dir / f"{name}_dataset.json").read_text())["samples"]
    graphs = []
    for sample in samples:
        by_node = {record["node_id"]: record["native_task_id"] for record in sample["node_outputs"]}
        graphs.append([by_node[node] for node in nodes])
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, type=Path,
                        help="Two or more results files from the same calibration dataset.")
    parser.add_argument("--topology", type=Path, default=None,
                        help="A generated topology directory whose predicted bar each arm is compared on.")
    args = parser.parse_args()
    if len(args.results) < 2:
        raise SystemExit("Need at least two results files to compare.")

    arms = [load_arm(path) for path in args.results]
    tasks = set(arms[0]["outcomes"])
    for arm in arms[1:]:
        if set(arm["outcomes"]) != tasks:
            raise SystemExit(f"{arm['name']} covers different tasks from {arms[0]['name']}.")

    print("Per arm:")
    for arm in arms:
        correct, total = accuracy(arm)
        tokens = [t for runs in arm["tokens"].values() for t in runs]
        print(f"  {arm['name']}: {correct}/{total} = {correct / total:.4f}   "
              f"mean tokens {sum(tokens) / len(tokens):.0f}   mean turns {sum(arm['turns']) / len(arm['turns']):.2f}   "
              f"status {dict(arm['statuses'])}")

    table = [[c, t - c] for c, t in map(accuracy, arms)]
    chi2, p, dof, _expected = chi2_contingency(table, correction=False)
    print(f"\nHomogeneity of accuracy across {len(arms)} arms: chi2 = {chi2:.2f}, dof = {dof}, p = {p:.3f}")

    graphs = topology_graphs(args.topology) if args.topology else None
    for a, b in itertools.combinations(arms, 2):
        diff, se = paired_task_difference(a, b)
        print(f"\n{a['name']}  vs  {b['name']}")
        print(f"  pooled z = {two_proportion_z(accuracy(a), accuracy(b)):+.2f}")
        print(f"  paired per-task difference = {diff:+.4f} +/- {se:.4f}  (z = {diff / se if se else 0:+.2f})")
        print(f"  token ratio (geometric mean over tasks) = {token_ratio(a, b):.3f}")
        if graphs:
            (pa, sa), (pb, sb) = predicted_bar(a, graphs), predicted_bar(b, graphs)
            se_diff = math.sqrt(sa ** 2 + sb ** 2)
            z = (pa - pb) / se_diff if se_diff else 0.0
            print(f"  predicted bar on {args.topology.name}: {pa:.3f} +/- {sa:.3f}  vs  {pb:.3f} +/- {sb:.3f}"
                  f"   difference {pa - pb:+.3f} (z = {z:+.2f})")


if __name__ == "__main__":
    main()
