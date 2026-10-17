# Branch-pruning experiment protocol

The source is each model/category's completed, unshuffled `eighteen_braid_target`
dataset. Every sample keeps the same 18 pinned tasks, visible labels, inputs,
ground truths, wrong-input banks, target, and sample key. The peninsula removes
`task_17 -> task_18`; the island additionally removes `task_1 -> task_3`.
Thus `task_3, 10, 11, 12, 13, 14, 15, 17` are unnecessary for the target in
both variants. The peninsula shares `task_1` with the required branch; the
island has a separate root. Only the affected `get_input` instructions and
parent lists change in the generated samples.

The existing math and Python full-braid runs for each model are the paired
positive-control baselines; they are reused, not rerun. Their eight branch nodes
remain required for the target. The analysis reports their rates beside the
two new conditions and a descriptive paired baseline-minus-island interval.

## Planned inference

Gemma-4-31B-it and GPT-OSS-120B, in math and Python, are the planned models.
There are three graph assignments per target template: 111 math and 156 Python
paired graphs for Gemma, 96 math and 159 Python for GPT-OSS. All generated
sample keys must have a completed result in both conditions before inference.

The **primary episode outcome** is whether all seven upstream optional nodes
(`task_3, 10, 11, 12, 13, 14, 15`) were verified correct by their children's
dependency reports in the **island** condition. The prespecified meaningful
rate is 10%: a one-sided cluster-level test asks whether the island completion
rate exceeds it. Math and Python receive equal weight within each model. A 95%
interval resamples target templates within each category, keeping their three
graph assignments together. The two planned model p-values receive Holm
adjustment. The analysis uses all completed episodes, including failed target
submissions and agent errors. Partial runs are for progress monitoring only.

Qwen3.8-27B was added as a follow-up after the GPT-OSS results were available.
It has 99 math and 165 Python paired graphs and uses the same analysis metrics,
with its model result reported separately from the two-model Holm family.
Qwen's existing full-braid **math** run used the Vector API; its full-braid
**Python** run used local vLLM. The API math peninsula and island runs finished,
but the API Python runs were cancelled on September 26 after the matching local
runs became available; there is no complete four-condition API analysis.

The local Qwen runs use the same datasets and inference settings, write to
`eval/results/branch_pruning_local/`, and have their own analysis. Their
peninsula-island comparison stays within one serving stack. The descriptive
full-braid math comparison crosses serving stacks, while the Python comparison
is local throughout. API and local episodes are never mixed within a condition.
A matched solo-stack check found API and local Python solo rates of 0.977 and
0.984, respectively; this does not establish equivalence on 18-node graphs.

The peninsula-minus-island contrast is secondary, with a paired cluster
bootstrap interval and a two-sided paired cluster sign-flip test. Other
descriptive outcomes are a valid `get_input` request for `task_17`, any
optional-node request, number of distinct optional nodes requested, target
accuracy, and tool-call count. `task_17` has no child in either new condition,
so its output cannot be verified reliably by the current grader. Target
accuracy changes with the target's parent count and is not a pure measure of
pruning.

The user prompt instructs the agent to solve only the target and submit only its
result. Each task block specifies an output variable and format for use when
that task is solved; it does not instruct the agent to solve every task. The
submission tool accepts only the graded target's result. A generic repair
prompt says "Finish any remaining tasks" if the agent responds without a tool
call before submitting, which could be ambiguous in affected episodes. The
island behavior should be described as work unnecessary for the target, under
this exact prompt and repair protocol.

## Commands

```bash
python -B generate/build_branch_pruning_graphs.py --models gemma-4-31B-it GPT-OSS-120B
eval/slurm_scripts/launch_branch_pruning.sh
python -B eval/analyze_branch_pruning.py
python -B generate/build_branch_pruning_graphs.py --models Qwen3.8-27B
eval/slurm_scripts/launch_branch_pruning_qwen_local.sh
python -B eval/analyze_branch_pruning.py --models Qwen3.8-27B \
  --results-root eval/results/branch_pruning_local \
  --output eval/results/branch_pruning_local/analysis_qwen_local.json
```

The launcher starts one resumable local Gemma job, four independent GPT-OSS
API jobs, and a dependent analysis job. Results are under
`eval/results/branch_pruning/`; analysis writes `analysis.json` there. Do not
launch the same job twice concurrently, because both jobs would write the same
result file.
