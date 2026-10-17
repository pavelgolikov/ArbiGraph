# Mixed-category gap experiment: results

Three models, three 18-node topologies, and 918 graded mixed graphs paired with the
same number of existing pure-math graphs, September 2026. The setup and task
selection are in [mixed_gap_experiment.md](mixed_gap_experiment.md); this document
reports the results computed by [analyze_mixed_gap.py](eval/analyze_mixed_gap.py).

The main result is **model dependent**. GPT-OSS-120B loses target accuracy on mixed
graphs beyond the change predicted by the replacement tasks' solo accuracies.
Qwen3.8-27B is somewhat more accurate on mixed graphs, although its adjusted 95%
interval includes zero. Gemma-4-31B-it shows no resolved change. An average across
the three models therefore does not describe any one of them well.

## What was run

Each mixed graph starts from a pure-math gap graph with the same model, topology,
target math task, target-task instance, and sample index. Nine math nodes, including
the target, keep their pinned problems and graph IDs. Five other positions become
Python, three RLVE, and one RAG. The graph edges and target remain fixed. The
shuffled-label braid was not part of this experiment.

| Model | Serving | Target templates | Repeats per template | Pairs per topology | Mixed graphs, all three |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B | Vector API | 33 | 3 | 99 | 297 |
| GPT-OSS-120B | Vector API | 32 | 3 | 96 | 288 |
| gemma-4-31B-it | Local vLLM | 37 | 3 | 111 | 333 |

The topologies are a chain (17 edges), a braid (28), and a fan-out/fan-in (32).
All 18 nodes are ancestors of the math target. Pure and mixed result files have
the same target-task/sample-index keys, and the analysis verifies that the nine
retained math node records match exactly in every pair.

The RAG and RLVE problems were selected before seeing mixed graph outcomes.
Each selected problem was then calibrated with 30 fresh solo runs per model;
math and Python solo rates come from the original
[prediction-gap experiment](gap_analysis.md). The selected RAG tasks scored
88/90, 87/90, and 90/90 in Qwen, GPT-OSS, and Gemma solo calibration; RLVE scored
148/150, 146/150, and 150/150. These are the rates used below, not the earlier
8/8 prescreen scores.

## How to read the figures and intervals

The [mixed-gap figures](eval/plots/mixed_gap/) have the same bar convention as
the prediction-gap figures: hatched bars are the independent-task prediction,
the product of the 18 nodes' calibrated solo success rates averaged over the
sampled graphs; solid bars are measured target accuracy. Each topology has a
pure-math and a mixed pair of bars. The predicted bar has a 95% calibration
bootstrap interval with graph assignments fixed, and the measured bar has a
95% Wilson interval. These bar intervals describe each accuracy separately.

The comparison of interest uses the *paired* target outcomes. The direct effect
is `mixed target accuracy − pure-math target accuracy`. The adjusted effect
subtracts the predicted change from replacement-task difficulty:

`adjusted effect = (measured mixed − measured pure) − (predicted mixed − predicted pure)`.

Both effects have 95% percentile intervals from 5,000 cluster-bootstrap draws.
A target-task template is one resampling unit, keeping its three repeats together. A model's pooled
three-topology interval keeps all nine observations for a template together.
Negative values mean lower target accuracy on mixed graphs. These signed
effects are percentage-point *changes*, not negative accuracy scores.

## Target accuracy by model and topology

All accuracy columns are percentages; effects and their 95% intervals are in
percentage points. The predicted shift is identical across topologies within
each model because each matched target/repeat uses the same set of sampled
tasks, although their graph positions differ.

| Model | Topology | Predicted pure → mixed | Measured pure → mixed | Direct mixed − pure, 95% CI | Adjusted, 95% CI |
|---|---|---:|---:|---:|---:|
| Qwen | chain | 73.7 → 74.8 | 31.3 → 40.4 | +9.1 [−5.1, +23.2] | +8.0 [−6.1, +22.3] |
| Qwen | braid | 73.7 → 74.8 | 34.3 → 45.5 | +11.1 [−2.0, +24.2] | +10.0 [−3.0, +23.0] |
| Qwen | fan-out/in | 73.7 → 74.8 | 42.4 → 43.4 | +1.0 [−11.1, +12.1] | −0.1 [−11.7, +11.5] |
| GPT-OSS | chain | 90.0 → 84.1 | 85.4 → 57.3 | −28.1 [−40.6, −15.6] | −22.2 [−34.5, −9.9] |
| GPT-OSS | braid | 90.0 → 84.1 | 71.9 → 52.1 | −19.8 [−34.4, −5.2] | −13.9 [−28.3, +0.8] |
| GPT-OSS | fan-out/in | 90.0 → 84.1 | 74.0 → 56.2 | −17.7 [−30.2, −4.2] | −11.8 [−24.3, +1.4] |
| Gemma | chain | 90.6 → 95.0 | 46.8 → 55.0 | +8.1 [−5.4, +21.6] | +3.7 [−9.7, +17.4] |
| Gemma | braid | 90.6 → 95.0 | 48.6 → 47.7 | −0.9 [−15.3, +13.5] | −5.3 [−19.8, +8.6] |
| Gemma | fan-out/in | 90.6 → 95.0 | 43.2 → 42.3 | −0.9 [−12.6, +11.7] | −5.3 [−17.0, +7.0] |

The corresponding figures are
[Qwen](eval/plots/mixed_gap/mixed_gap_Qwen3.8-27B.pdf),
[GPT-OSS](eval/plots/mixed_gap/mixed_gap_GPT-OSS-120B.pdf), and
[Gemma](eval/plots/mixed_gap/mixed_gap_gemma-4-31B-it.pdf). Exact bar values and
intervals are in [mixed_gap_summary.csv](eval/plots/mixed_gap/mixed_gap_summary.csv).

### Pooling the three topologies within each model

| Model | Pure failures recovered by mixed | Pure successes lost by mixed | Direct effect, 95% CI | Adjusted effect, 95% CI |
|---|---:|---:|---:|---:|
| Qwen | 77 | 56 | +7.1 [+0.7, +13.5] | +6.0 [−0.8, +12.6] |
| GPT-OSS | 38 | 101 | −21.9 [−30.6, −13.2] | −16.0 [−24.7, −7.2] |
| Gemma | 85 | 78 | +2.1 [−5.7, +10.5] | −2.3 [−9.9, +5.9] |

The recovery and loss counts compare each mixed graph with its matched pure
graph. GPT-OSS has 63 more losses than recoveries across 288 pairs. Qwen has
21 more recoveries than losses across 297 pairs, but the adjusted pooled
interval includes zero. Gemma's seven-net-recovery difference across 333 pairs
is small relative to its interval.

The equal-model average over all topologies is −4.2 points directly
[−8.7, +0.3] and −4.1 points after adjustment [−8.6, +0.5]. These averages
hide the opposing Qwen and GPT-OSS responses and should not be used as a
universal mixing penalty.

### What each model shows

**GPT-OSS:** Mixed target accuracy is lower in all three topologies. Its solo-rate
prediction falls by only 5.9 points, while measured accuracy falls by 17.7–28.1
points. The chain's adjusted loss is resolved on its own; the braid and fan-out/in
adjusted intervals reach zero. Pooling the three paired topologies gives an
adjusted loss of 16.0 points [7.2, 24.7] in magnitude. Thus the change in the
selected tasks' solo success rates does not account for GPT-OSS's full decline.

**Qwen:** The predicted mixed accuracy is 1.1 points higher. Measured mixed
accuracy is 1.0–11.1 points higher, depending on topology. The pooled direct
increase has an interval above zero, but the pooled adjusted interval
[−0.8, +12.6] does not establish an improvement beyond solo difficulty. The
experiment gives no evidence of an extra mixing cost for Qwen at this precision.

**Gemma:** The predicted mixed accuracy is 4.4 points higher, but measured
changes range from −0.9 to +8.1 points. Every direct and adjusted topology
interval includes zero. The pooled adjusted estimate is −2.3 points
[−9.9, +5.9], leaving moderate effects in either direction possible.

The original gap is `predicted − measured` target accuracy. Rounded to whole
percentage points, its pure-math → mixed changes are:

| Model | Chain gap | Braid gap | Fan-out/in gap |
|---|---:|---:|---:|
| Qwen | 42 → 34 | 39 → 29 | 31 → 31 |
| GPT-OSS | 5 → 27 | 18 → 32 | 16 → 28 |
| Gemma | 44 → 40 | 42 → 47 | 47 → 53 |

Thus GPT-OSS's small-to-moderate pure-math gap becomes substantial on these
mixed graphs. Qwen and Gemma already had large pure-math gaps; mixing does not
create a common additional penalty in those models.

## Does the mixing effect depend on graph shape?

The point estimates vary by topology, but paired contrasts between topology
effects are imprecise. These compare each topology's `mixed − pure` change to
the chain's change, resampling the same target templates together:

| Model | Braid effect − chain effect, 95% CI | Fan-out/in effect − chain effect, 95% CI |
|---|---:|---:|
| Qwen | +2.0 [−18.2, +21.2] | −8.1 [−28.3, +11.1] |
| GPT-OSS | +8.3 [−6.2, +24.0] | +10.4 [−7.3, +28.1] |
| Gemma | −9.0 [−27.9, +9.9] | −9.0 [−27.0, +9.9] |

None establishes a topology interaction. The intervals are wide enough that
this is not an equivalence result. The direct and solo-adjusted topology
contrasts coincide because the predicted pure-to-mixed change is the same for
each topology within a model.

## Other outcomes and checks

The selected RAG and RLVE tasks are near ceiling alone, but their in-graph
node-correct rates are much lower. Across chain, braid, and fan-out/in, Qwen's
RAG rates are 40–91% and RLVE rates 45–87%; GPT-OSS's are 58–81% and 77–92%;
Gemma's are 56–74% and 64–92%. These are descriptive node outcomes at fixed
positions, not isolated estimates of a category-mixing effect. The mean of the
nine retained math nodes, which includes the target, changes by −5.7 to −1.9
points for GPT-OSS, −3.3 to +9.9 for Qwen, and −0.5 to +2.6 for Gemma across
topologies. Its inclusion of the target makes it unsuitable as an independent
mechanism check.

Mixed graph episodes generated about 47–54k tokens for Qwen, 33–35k for
GPT-OSS, and 27–28k for Gemma, compared with about 54–67k, 32–33k, and
28–29k in their paired pure graphs. None of the mixed episodes exceeded the
context limit. Five GPT-OSS mixed episodes and one Gemma mixed episode ended
with malformed tool calls; they remain in the accuracy denominators. The
corresponding pure graphs had four and three such episodes, respectively, so
this status difference alone is too small to account for GPT-OSS's target loss.

## What this result does and does not establish

Pairing controls the model, topology, target template, repeat index, and nine
retained math problems. The independent-task prediction accounts for measured
solo difficulty of the replacement tasks. Selection and calibration preceded
the mixed graph runs. These choices make the GPT-OSS result more specific than
a raw comparison of unrelated math and mixed graphs.

The intervention still changes more than category labels: it replaces nine
problems, their wording and output lengths, and their locations in the graph.
The RAG outputs in the selection set are shorter than the math/Python output
reference, and Qwen's mixed episodes are substantially shorter overall. The
experiment therefore identifies the effect of *this fixed task mixture under
this graph construction*, not a pure causal effect of category diversity or a
result for arbitrary mixtures and task pools. Model-specific pools also mean
the three models do not encounter identical graph problems.

The adjusted cluster-bootstrap intervals hold the estimated solo rates fixed.
They omit calibration uncertainty, especially relevant for tasks with 30/30
solo successes, which the ordinary bootstrap treats as exactly perfect. A
posterior or joint resampling analysis of calibration and graph outcomes would
be needed before presenting the adjusted intervals as fully calibrated. The
per-topology paired intervals are about ±12–15 points, so small mixing effects
remain unresolved; several topology comparisons add a multiple-comparison
concern. Finally, all conclusions are conditional on the screened, pinned
problems and these three models.

## Reproducing

From the repository root, using the existing graded results only:

```bash
python eval/analyze_mixed_gap.py --topology all
python eval/plots/plot_mixed_gap.py
```

The analysis writes [analysis_all.json](eval/results/mixed_gap/analysis_all.json).
The plotting script writes one PDF per model, the
[bar summary](eval/plots/mixed_gap/mixed_gap_summary.csv), and the
[paired-difference summary](eval/plots/mixed_gap/mixed_gap_paired_differences.csv).
No inference or server run is needed to reproduce these outputs.
