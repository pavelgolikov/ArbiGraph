# Prediction-gap experiment: results

What the experiment measured, what it found, and what the findings do and do not
license. Three models, Qwen3.8-27B, GPT-OSS-120B and gemma-4-31B-it, on math and
python, September 2026. Pools come from `generate/screen_task_pools.py`, datasets from
`generate/gen_gap_datasets.sh`, and the runs from `eval/slurm_scripts/run_gap.sh`
(local vLLM) and `run_gap_api.sh` (Vector API). `eval/summarize_gap.py` prints the gap
statistics below.

Two questions drove the experiment, and the three models add a third:

1. **Does context cost anything on top of each task's own error rate?** Yes, in every
   model, by an amount that depends on the model. In chains and braids, Qwen3.8-27B and
   gemma-4-31B-it lose 2.8-4.6 points of accuracy per node, which compounds across
   eighteen nodes into a 29-46 point gap. GPT-OSS-120B loses 0.3-1.3 points per node, a
   5-18 point gap.
2. **Does shape matter, or only the number of tasks?** Only the number. In every model,
   averaged over the two categories, a chain, a braid and a fan-out/fan-in are
   indistinguishable at the target.
3. **Does it matter whether task labels follow the dependency order?** For two of the
   three models. Shuffling the labels costs Gemma 16 points and GPT-OSS 7; Qwen is
   unaffected.

## What was run

| model | serving | category | pool | calibration | graphs per topology |
|---|---|---|---|---|---|
| Qwen3.8-27B | local vLLM and Vector API | math | 33 tasks | 50 repeats, 1,650 rollouts | 99 |
| | | python | 55 tasks | 45 repeats, 2,475 rollouts | 165 |
| GPT-OSS-120B | Vector API | math | 32 tasks | 50 repeats, 1,600 rollouts | 96 |
| | | python | 53 tasks | 30 repeats, 1,590 rollouts | 159 |
| gemma-4-31B-it | local vLLM | math | 37 tasks | 50 repeats, 1,850 rollouts | 111 |
| | | python | 52 tasks | 30 repeats, 1,560 rollouts | 156 |

Four eighteen-node topologies per category, each with three graphs per target task: a
chain; a braid with 28 edges; a fan-out/fan-in; and the braid with its task labels
shuffled, so that label order no longer follows dependency order. In every topology all
17 non-target nodes are ancestors of the target.

Every task carries one pinned instantiation — the same static data, input value and
bank wherever it appears, and for every model. The calibration re-runs that one problem
in isolation; the graphs place the same problem at a node. Verified for all three
models: every node's input value and ground truth is byte-identical to what the
calibration measured, each pinned problem is the one the model met as `sample_idx 0` in
`exp1`, and each results file holds exactly its dataset.

Each model's pool is its own 8/8 screen in `exp1`. GPT-OSS's 131k context also dropped
tasks averaging more than 12,000 generated tokens, and its python pool keeps only tasks
Qwen's holds. Gemma's python pool keeps only tasks GPT-OSS's holds, so the python pools
nest (52 ⊂ 53 ⊂ 55); the math pools share 24 tasks. Each model ran at its own `exp1`
settings, so its calibration repeats each pinned problem under the conditions that
screened it in: temperature 1.0, top-p 0.95, 16,384 tokens per turn, reasoning kept
across turns, and then

- Qwen3.8-27B: reasoning effort xhigh, top-k 20, 262k context.
- GPT-OSS-120B: reasoning effort high, top-k 20, 131k context, Harmony prompts rendered
  locally and checked token for token against Vector's on every turn.
- gemma-4-31B-it: thinking on, top-k 64 (Google's default), 262k context.

GPT-OSS had to be nudged a little, with prompt engineering, not to submit early. A
submission made before the graded task's input has been requested is turned back with a
short message naming the graded task, and the model carries on.

## The bars

One figure per model and category, `eval/plots/gap/prediction_gap_<model>_<category>.pdf`,
plus a two-category figure per model, `eval/plots/gap/prediction_gap_<model>.pdf`.
Each figure now shows the chain, braid, fan-out/fan-in, and shuffled-label braid.
Numbers for all four topologies are in
`eval/plots/gap/prediction_gap_summary.csv`.

| | single target | predicted | chain measured | chain gap | braid measured | braid gap |
|---|---|---|---|---|---|---|
| Qwen math | 0.9836 | 0.737 ± 0.043 | 0.313 | **0.424 ± 0.063**, 6.7σ | 0.343 | **0.393 ± 0.064**, 6.1σ |
| Qwen python | 0.9826 | 0.723 ± 0.034 | 0.436 | **0.286 ± 0.051**, 5.6σ | 0.376 | **0.347 ± 0.051**, 6.8σ |
| GPT-OSS math | 0.9944 | 0.900 ± 0.032 | 0.854 | **0.045 ± 0.048**, 0.9σ | 0.719 | **0.181 ± 0.056**, 3.2σ |
| GPT-OSS python | 0.9981 | 0.969 ± 0.017 | 0.868 | **0.101 ± 0.032**, 3.2σ | 0.887 | **0.082 ± 0.031**, 2.7σ |
| Gemma math | 0.9946 | 0.906 ± 0.029 | 0.468 | **0.437 ± 0.055**, 7.9σ | 0.486 | **0.419 ± 0.055**, 7.6σ |
| Gemma python | 0.9994 | 0.991 ± 0.009 | 0.564 | **0.427 ± 0.041**, 10.5σ | 0.532 | **0.459 ± 0.041**, 11.2σ |

The predicted bar is computed per sampled graph from the tasks that actually filled
it, then averaged over the same graphs the measured bar averages over, so each pair
describes one identical set of graphs. The predicted bar's standard error is a
bootstrap over calibration rollouts; the measured bar's is binomial; the gap's combines
the two. The figures draw bootstrap percentile intervals and Wilson intervals.

GPT-OSS and Gemma solve their pinned problems almost perfectly — GPT-OSS failed 3 of
1,590 python calibration rollouts, Gemma 1 of 1,560 — so most tasks score a perfect
30/30 and the bootstrap takes their rates as exactly 1. A posterior that does not
(task rates from an empirical-Bayes beta prior, or from the pooled rate where the tasks
show too little spread to fit one) gives these 95% intervals on the gap:

| | chain | braid |
|---|---|---|
| Qwen math | [0.307, 0.531] | [0.278, 0.506] |
| Qwen python | [0.190, 0.382] | [0.251, 0.442] |
| GPT-OSS math | [−0.029, 0.139] | [0.089, 0.288] |
| GPT-OSS python | [0.028, 0.162] | [0.013, 0.141] |
| Gemma math | [0.339, 0.538] | [0.322, 0.517] |
| Gemma python | [0.337, 0.500] | [0.368, 0.533] |

Of all 24 model, category and topology runs, only GPT-OSS's math chain has an interval
that reaches zero.

## Question 1: context costs, on top of per-task error

The independent-error null predicts 72-99% target accuracy; the models deliver 27-89%.
The gap is present in every model and every topology except GPT-OSS's math chain.

| gap, z | chain | braid | fan-out/in | braid, shuffled labels |
|---|---|---|---|---|
| Qwen math | 0.424, 6.7σ | 0.393, 6.1σ | 0.312, 4.8σ | 0.302, 4.6σ |
| Qwen python | 0.286, 5.6σ | 0.347, 6.8σ | 0.371, 7.4σ | 0.311, 6.1σ |
| GPT-OSS math | 0.045, 0.9σ | 0.181, 3.2σ | 0.160, 2.9σ | 0.285, 4.8σ |
| GPT-OSS python | 0.101, 3.2σ | 0.082, 2.7σ | 0.089, 2.9σ | 0.139, 4.0σ |
| Gemma math | 0.437, 7.9σ | 0.419, 7.6σ | 0.473, 8.6σ | 0.635, 12.5σ |
| Gemma python | 0.427, 10.5σ | 0.459, 11.2σ | 0.484, 11.8σ | 0.581, 14.4σ |

All four topologies share one predicted bar, so they pool:

| pooled gap | chain + braid | all four |
|---|---|---|
| Qwen math | 0.408 ± 0.054, 7.5σ | 0.358 ± 0.049, 7.3σ |
| Qwen python | 0.317 ± 0.043, 7.3σ | 0.329 ± 0.039, 8.4σ |
| GPT-OSS math | 0.113 ± 0.043, 2.6σ | 0.168 ± 0.039, 4.3σ |
| GPT-OSS python | 0.092 ± 0.025, 3.6σ | 0.103 ± 0.022, 4.7σ |
| Gemma math | 0.428 ± 0.044, 9.7σ | 0.491 ± 0.037, 13.3σ |
| Gemma python | 0.443 ± 0.030, 15.0σ | 0.488 ± 0.022, 22.3σ |

### It resolves to a per-node penalty

A single constant δ subtracted from every node's solo accuracy p̄ reproduces each
endpoint, (p̄ − δ)¹⁸ = measured:

| δ per node | chain | braid | fan-out/in | braid, shuffled labels |
|---|---|---|---|---|
| Qwen math | 0.046 | 0.041 | 0.030 | 0.029 |
| Qwen python | 0.028 | 0.036 | 0.039 | 0.031 |
| GPT-OSS math | 0.003 | 0.013 | 0.011 | 0.021 |
| GPT-OSS python | 0.006 | 0.005 | 0.005 | 0.008 |
| Gemma math | 0.036 | 0.034 | 0.040 | 0.065 |
| Gemma python | 0.031 | 0.034 | 0.037 | 0.048 |

So the gap is not exotic compounding. It is a few points lost per node, multiplied
eighteen times. Measuring each node inside the graph against the same pinned problem
solved alone, among nodes whose upstream values were all right and which received their
true input, gives the same figure directly: 0.026-0.059 for Qwen, 0.029-0.059 for
Gemma, 0.002-0.023 for GPT-OSS.

Qwen and Gemma pay about the same per node. GPT-OSS pays four to six times less, and so
nearly avoids the gap: the penalty is real in all three models, but its size is a
property of the model, not of composition alone. How much text the graph puts in context
does not explain the difference. An eighteen-node graph generates 54-67k tokens for
Qwen, 32-42k for GPT-OSS and 27-33k for Gemma, so Gemma carries the least and still
pays as much as Qwen.

### It is computation, not the hand-offs

Replaying each failed graph's `get_input` calls finds the first node whose value went
wrong. Outside the shuffled braid, that node received its true input and computed the
wrong answer in 88-100% of Qwen's failed graphs, 78-95% of GPT-OSS's and 86-98% of
Gemma's. Most of the rest are a node whose input the model never requested; a parent
value stated wrongly is at most 7%. Values are not being lost between nodes: the model
computes worse with more in its context.

### The root node

`task_1` in the chain has no parents. Nothing is passed into it, it makes no
dependency claim, and it is the identical pinned problem the calibration measured.

| | alone | as node 1 of 18 | penalty | |
|---|---|---|---|---|
| Qwen math | 0.9844 | 0.9293 | 0.055 | z = 2.1 |
| Qwen python | 0.9785 | 0.9212 | 0.057 | z = 2.7 |
| GPT-OSS math | 0.9956 | 0.9792 | 0.017 | z = 1.1 |
| GPT-OSS python | 0.9983 | 0.9811 | 0.017 | z = 1.6 |
| Gemma math | 0.9946 | 0.9820 | 0.013 | z = 1.0 |
| Gemma python | 0.9991 | 0.9872 | 0.012 | z = 1.3 |

For Qwen, a node that does no composition at all still loses about 5.5 points: the cost
of having eighteen task statements in the context at all, ~65k tokens instead of ~5k and
~40 agent turns instead of ~4. It is unconditional, over all 99 and 165 samples, so it
carries no survivorship concern.

This does not carry over. GPT-OSS and Gemma lose 1.2-1.7 points at the root, not
significant on their own. For Gemma, whose average per-node penalty matches Qwen's,
the cost lands later in the chain.

### Where along the chain the cost lands

Predicted against measured accuracy by position along the chain, predicted / measured:

| depth | 1 | 3 | 6 | 9 | 12 | 15 | 18 |
|---|---|---|---|---|---|---|---|
| Qwen math | .984 / .929 | .959 / .747 | .905 / .576 | .862 / .505 | .819 / .434 | .773 / .374 | .737 / .313 |
| Qwen python | .978 / .921 | .940 / .836 | .882 / .715 | .839 / .570 | .802 / .521 | .756 / .455 | .723 / .436 |
| GPT-OSS math | .996 / .979 | .983 / .969 | .965 / .948 | .948 / .917 | .930 / .906 | .916 / .875 | .900 / .854 |
| GPT-OSS python | .998 / .981 | .996 / .969 | .990 / .931 | .984 / .912 | .979 / .906 | .975 / .887 | .969 / .868 |
| Gemma math | .995 / .982 | .984 / .937 | .969 / .820 | .952 / .703 | .936 / .631 | .921 / .550 | .906 / .468 |
| Gemma python | .999 / .987 | .998 / .936 | .996 / .846 | .995 / .795 | .994 / .712 | .993 / .635 | .991 / .564 |

In every model the separation opens at the first nodes and widens along the chain: the
cost *accumulates*, which is the mechanism being claimed, and this argues it better than
two bars do. The per-node penalty given every upstream value right, averaged over the
first and the last six nodes:

| | depths 1-6 | depths 13-18 | change, 95% interval |
|---|---|---|---|
| Qwen math | 0.071 | 0.030 | −4.1 pp [−7.7, −0.6] |
| Qwen python | 0.034 | 0.016 | −1.8 pp [−3.9, +0.4] |
| GPT-OSS math | 0.003 | 0.005 | +0.2 pp [−1.1, +1.4] |
| GPT-OSS python | 0.010 | 0.005 | −0.5 pp [−1.4, +0.5] |
| Gemma math | 0.027 | 0.043 | +1.6 pp [−1.0, +4.1] |
| Gemma python | 0.027 | 0.038 | +1.2 pp [−0.6, +3.1] |

Qwen front-loads the cost; Gemma's does not fall and, in both categories, points
upward as the context fills. Conditioning on upstream nodes being right tends to pull a
penalty down with depth (see caveats), so Gemma's rise, though not significant in either
category alone, is if anything understated. Two models with the same total penalty pay
it in different places: Qwen for the presence of eighteen task statements, Gemma for
what accumulates as the episode proceeds.

## Question 2: shape does not matter, node count does

The braid has 18 nodes and 28 edges against the chain's 17, and the fan-out/fan-in
spreads the same 17 ancestors differently again; all three share one predicted bar and
differ only in wiring. Difference in measured accuracy, combined over categories by
inverse variance:

| braid − chain | math | python | combined |
|---|---|---|---|
| Qwen | +0.030 ± 0.067 | −0.061 ± 0.054 | **−0.025 ± 0.042**, z = −0.59 |
| GPT-OSS | −0.135 ± 0.058 | +0.019 ± 0.037 | **−0.025 ± 0.031**, z = −0.80 |
| Gemma | +0.018 ± 0.067 | −0.032 ± 0.056 | **−0.011 ± 0.043**, z = −0.26 |

| fan-out/in − chain | math | python | combined |
|---|---|---|---|
| Qwen | +0.111 ± 0.068 | −0.085 ± 0.054 | **−0.010 ± 0.042**, z = −0.24 |
| GPT-OSS | −0.115 ± 0.057 | +0.013 ± 0.037 | **−0.025 ± 0.031**, z = −0.80 |
| Gemma | −0.036 ± 0.067 | −0.058 ± 0.056 | **−0.049 ± 0.043**, z = −1.13 |

No model shows a shape effect once categories are combined. Single categories depart
by up to 13.5 points — GPT-OSS's math braid (z = −2.3) and fan-out (z = −2.0), Qwen's
math fan-out (z = +1.6) — but each vanishes or reverses in the other category, and with
this many comparisons one |z| near 2 is expected by chance. GPT-OSS's math chain is the
outlier there, not its braid: the chain's gap is its only one that is not significant.

State this as a bound, not as zero. Each combined difference is resolved to about ±6-8
points, and **the 95% intervals rule out a shape effect larger than 9-13 points in any
of the three models** (the widest, Gemma's fan-out/fan-in, reaches −13). A smaller real
effect would be invisible at this sample size.

This is predicted by the answer to question 1. If the cost is a per-node tax paid for
what is in the context, two topologies with the same node count should cost the same —
which is what was measured.

### The paths differ completely even though the endpoints match

Qwen's math chain against its math braid, per-node accuracy by position:

| depth | chain | braid |
|---|---|---|
| 1 | 0.929 | 0.889 |
| 6 | 0.576 | 0.747 |
| 12 | 0.434 | 0.606 |
| 17 | 0.323 | 0.515 |
| 18 | 0.313 | 0.343 |

The braid keeps its intermediate nodes far healthier, because they sit on shorter
paths from the root, and then falls 17 points in the single step from 17 to 18 where
the two branches converge. The chain drops 1 point over that same step and arrives at
the same place by bleeding out gradually. python shows the same pattern: 0.630 at
depth 17, 0.376 at the target.

Two different mechanisms, one destination. Worth a figure.

## Question 3: label order matters for some models

The shuffled braid is the braid with its task labels permuted, so `task_7` may depend
on `task_12`. Nothing else changes: the same problems, the same wiring, the same
predicted bar.

| braid, shuffled − braid | math | python | combined |
|---|---|---|---|
| Qwen | +0.091 ± 0.069 | +0.036 ± 0.054 | **+0.057 ± 0.042**, z = +1.34 |
| GPT-OSS | −0.104 ± 0.068 | −0.057 ± 0.039 | **−0.068 ± 0.034**, z = −2.03 |
| Gemma | −0.216 ± 0.063 | −0.122 ± 0.056 | **−0.163 ± 0.042**, z = −3.88 |

Gemma loses 16 points and GPT-OSS 7; Qwen, if anything, gains. Much of Gemma's loss is
nodes it computes without fetching their input: in its shuffled math braids, 36% of
failed graphs first go wrong at a node it never called `get_input` for, against 7% in the
unshuffled braid (python 17% against 8%). It does not work through the tasks in label
order; at such a node it works on a value it already holds instead of the node's own
input, in the episodes traced the parent's output. On nodes that did receive their
true input its per-node penalty also rises, from 0.034 to 0.047 in math and from 0.030
to 0.040 in python. GPT-OSS's shuffled failures stay mostly computation (78-81%), with
more of them starting at a wrongly stated parent value (8-15%, against 4-6% unshuffled).


## What the design rules out

The gap survives each of these because the design controls for them.

- **Task difficulty.** Both bars describe the same pinned instances; per-task accuracy
  is measured, not assumed.
- **Task selection.** Pool membership was decided by 8/8 single-target rollouts in
  `exp1`, before any graph ran, and both bars use that same pool.
- **Instance selection.** The instance is pinned by a rule fixed in advance
  (`sample_idx 0`), never because the model solved it. Selecting on outcome would
  inflate the calibrated rate and manufacture part of the gap.
- **Calibration noise.** K was sized so the predicted bar's standard error lands at or
  under the measured bar's. All 24 runs hit that, from Qwen math (0.043 against 0.047)
  to Gemma python (0.009 against 0.040). The posterior intervals above cover the
  near-perfect calibrations the bootstrap cannot.
- **The `get_input` protocol.** Failed graphs start at a node that received its true
  input and computed wrong in 78-100% of cases outside the shuffled braid.

## Caveats to carry into the paper

**Three models, and the size is model-dependent.** A per-node penalty now appears in
three model families. How large it is varies four- to sixfold: 2.8-4.6 points for Qwen
and Gemma in chains and braids, 0.3-1.3 for GPT-OSS. State the penalty as a property of the model
under composition, not a constant of composition.

**Pools differ between models.** Each model's pool is its own 8/8 screen, so the graphs
differ between models even where the tasks overlap. Compare models by per-node penalty,
not by raw measured bars, whose predicted bars differ (0.72-0.99).

**Two categories.** rlve was dropped on measurement, not on a hunch: its per-node
penalty is too small to resolve at feasible sample sizes.

**Depth-wise penalties are conditioned on upstream nodes being correct**, which selects
for runs where the model is doing well and may flatten the penalty with depth, and
per-depth counts fall (Qwen math from 99 to 32). Qwen's decline along the chain is
consistent with this survivorship rather than a real fall; Gemma's rise runs against
it. The root-node results do not depend on this conditioning.

**Several comparisons.** Each model has six shape and label comparisons. Gemma's label
effect (combined z = −3.9, both categories significant) survives this; GPT-OSS's label
effect (z = −2.0) and its math shape departures should be reported as suggestive.

**Context overflow.** 4 of Qwen's 1,056 graphs hit `context_length_exceeded` (1 math
chain, 1 python chain, 2 python braid); none of GPT-OSS's 1,020 at 131k or Gemma's
1,068. They grade as incorrect, which is conservative, but say so rather than leaving a
reader to find a silently dropped sample. Likewise 5 GPT-OSS and 4 Gemma graphs ended on
malformed tool calls and grade as incorrect.

**The braid states two values per `get_input` call** where the chain states one, since
most braid nodes have two parents. This was a worry in the design. It is now moot: the
braid is not worse than the chain in any model once categories are combined, and a
wrongly stated parent value starts at most 7% of failures outside the shuffled braid.

**Screening is on the task average, not the instance.** `transformArray` passed 8/8 in
Qwen's `exp1` but its pinned instance solves at 0.78, the weakest in the python pool. A
task can survive screening while carrying a hard instance. GPT-OSS's and Gemma's weakest
pinned instances solve at 0.94 in math and 0.97 in python.

## Reproducing

```bash
python eval/summarize_gap.py
python eval/plots/plot_prediction_gap.py \
    --models Qwen3.8-27B GPT-OSS-120B gemma-4-31B-it \
    --categories math python --output_dir eval/plots/gap
```

`summarize_gap.py` prints the gaps, per-node penalties, root-node and depth results,
shape and label effects, and failure attribution; the serving stack test is
`eval/compare_stacks.py`. The plot script reads
`eval/results/gap/<model>/<category>/`, pools every `agent_single_target_baseline*.json`
into the calibration, and computes the ancestor closure per topology from its
`_input_graph.json` rather than assuming all nodes are ancestors. It writes a figure per
model and category, a two-category figure per model, and
`eval/plots/gap/prediction_gap_summary.csv`, replacing only the rows of the models
it plots.
