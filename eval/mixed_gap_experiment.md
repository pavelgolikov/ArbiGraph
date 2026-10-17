# Mixed-category gap experiment

## Question

Does placing different task categories in one graph reduce target accuracy beyond
what we would expect from the individual tasks' difficulty? The experiment compares
each mixed graph with an existing pure-math graph from the [prediction-gap
experiment](gap_analysis.md). Each pair has the same model, topology, math target,
target-task instance, sample index, and nine unchanged math nodes. Only the categories
and pinned problems at the other nine nodes differ. This also exercises ArbiGraph's
ability to place math, Python, RLVE, and RAG tasks in one topology without changing
its edges or input tool.

We use three existing 18-node topologies: `eighteen_chain_target` (17 edges),
`eighteen_braid_target` (28 edges), and `eighteen_fan_out_fan_in` (32 edges). The
shuffled-label braid is outside this experiment. All nodes are ancestors of the math
target at graph node `task_18`.

| Model | Serving | Math target templates | Repeats per template | Paired graphs per topology |
|---|---|---:|---:|---:|
| Qwen3.8-27B | Vector API | 33 | 3 | 99 |
| GPT-OSS-120B | Vector API | 32 | 3 | 96 |
| gemma-4-31B-it | Local vLLM | 37 | 3 | 111 |

Across the three topologies this is 918 mixed graph episodes, paired with already
collected pure-math episodes. The same target-task and sample-index keys occur in all
three topologies for a given model.

## Selecting RAG and RLVE problems

The candidate manifest contains 24 RAG questions and 12 RLVE tasks. We prescreened
each candidate as a pinned, single-node task for eight samples on each of the three
models. A candidate had to score 8/8 on **every** model to qualify. The selection
used only these prescreen results and existing math/Python solo output lengths, before
the subsequent calibration or mixed graph runs.

- **RAG:** select the three qualifying questions with the highest mean generated-token
  counts across the 24 prescreen samples per question. Selected native task IDs:
  `68, 25, 184`.
- **RLVE:** among qualifying tasks with at least 1,800 mean generated tokens, select
  the five closest to the 2,293.75-token reference median from the six model-by-category
  math/Python solo medians. Selected IDs: `13, 31, 38, 21, 11`.

The selected RAG questions generated about 720–965 tokens on average in prescreen;
the selected RLVE tasks generated about 2,040–3,694. RAG remains shorter in generated
output than the math/Python reference despite choosing the longest accurate RAG
questions. Its retrieved contexts are roughly 5,600–7,100 characters. We should
report output length alongside accuracy and avoid claiming exact length matching.

After selection, we ran **30 fresh single-node repeats per selected task** on each
model, using the same pinned problem and inference settings as the graph runs. The
prescreen and calibration use separate generation seeds (11 and 23). These held-out
calibration results estimate solo accuracy for the three RAG and five RLVE problems;
math and Python solo rates come from the existing prediction-gap calibration.

| Model | RAG calibration | RLVE calibration |
|---|---:|---:|
| Qwen3.8-27B | 88/90 | 148/150 |
| GPT-OSS-120B | 87/90 | 146/150 |
| gemma-4-31B-it | 90/90 | 150/150 |

The complete selection rule, candidate outcomes, and chosen IDs are recorded in
[`selection.json`](generate/generated/mixed_gap/selection.json). Prescreen and
calibration are complete; neither used mixed graph outcomes.

## Constructing paired mixed graphs

For each model and topology, [`build_mixed_gap_graph.py`](generate/build_mixed_gap_graph.py)
starts from that model's pure-math gap dataset. It copies the edges, math target, and
the nine math node records at graph IDs `task_1, 3, 5, 7, 9, 11, 13, 15, 18` exactly.
It replaces the other graph nodes as follows:

| Category | Graph node IDs | Problems used |
|---|---|---|
| Python | `task_2, 4, 6, 8, 10` | Five deterministic draws from that model's existing pinned Python gap pool per graph |
| RLVE | `task_12, 14, 16` | Three of the five selected pinned tasks, rotated across graphs |
| RAG | `task_17` | One of the three selected pinned questions, rotated across repeats |

Every replacement retains its solo-calibration input, true output, and wrong-input
bank. The builder verifies those fields against the single-node datasets. It keeps
the pure graph's parent lists and changes only the replaced task blocks in the
prompt. Graph node IDs identify replacement positions; in the braid, visible task
numbers have a different order, which the builder follows when writing tool-call
instructions. Task assignment is deterministic and does not use graph results.

The mixed datasets live at
`generate/generated/mixed_gap/<model>/mixed/<topology>/`. Each folder contains the
dataset JSON, input graph JSON, readable prompts, and topology SVG. The generator
entry point is [`gen_mixed_gap_datasets.sh`](generate/gen_mixed_gap_datasets.sh).

## Evaluation and analysis

The API [launcher](eval/slurm_scripts/launch_mixed_gap_api.sh) submits one job for
each Qwen/GPT-OSS topology pair; the [local runner](eval/slurm_scripts/run_mixed_gap_local.sh)
runs Gemma. Both use the same agent, tool, grading, and model settings as the
corresponding gap runs. Results are written to
`eval/results/mixed_gap/<model>/mixed/agent_<topology>_<model>.json`.

The direct comparison is the mean paired difference in target correctness,
`mixed − pure math`, within each model and topology. A negative value means the
mixed graphs had lower target accuracy. Because replacing math tasks also changes
solo difficulty, [`analyze_mixed_gap.py`](eval/analyze_mixed_gap.py) additionally
computes an independent-task prediction for each graph: multiply the calibrated
solo success rates of its 18 pinned tasks. The adjusted difference is

`(observed mixed − observed pure) − (predicted mixed − predicted pure)`.

This adjusted quantity asks whether the observed change exceeds the change
predicted from the replacement tasks' solo accuracies. Report it with the direct
paired difference, rather than interpreting the direct difference alone as a
category-mixing cost. The analysis also reports retained-math node accuracy,
in-graph RAG/RLVE accuracy, generated-token lengths, and context overflows.

The analysis uses 95% cluster-bootstrap intervals, resampling target-task templates
within each model. Each template's three repeats stay together. In the pooled
three-topology estimate, the same template's observations across topologies stay
together too; models and topologies receive equal weight. The adjusted intervals
currently hold the estimated solo rates fixed, so they **do not include calibration
uncertainty**. Near-ceiling calibration and selection on 8/8 prescreen results make
that qualification material. The RAG output-length difference and this one fixed
mix of categories also limit a broad causal claim about category mixing.

All three models' graph results are now graded. To reproduce the analysis, run:

```bash
python eval/analyze_mixed_gap.py --topology all
```

This writes `eval/results/mixed_gap/analysis_all.json` with per-model,
per-topology, pooled, and topology-contrast estimates. The results and
interpretation are in [mixed_gap_analysis.md](mixed_gap_analysis.md).

To draw absolute target-accuracy bar charts in the prediction-gap style, run
`python eval/plots/plot_mixed_gap.py` from the repository root. Each model's figure
in `eval/plots/mixed_gap/` compares pure math with mixed graphs for each topology:
hatched bars are solo-rate predictions and solid bars are measured accuracy, both
with 95% intervals. `mixed_gap_summary.csv` holds the bar values and intervals;
`mixed_gap_paired_differences.csv` holds the signed paired comparisons and their
target-template cluster bootstrap intervals. All three models are included.
The adjusted paired intervals hold solo calibration rates fixed.
