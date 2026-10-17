# Branch pruning in 18-task braids: Gemma, GPT-OSS, and Qwen

Analysis of the completed Gemma-4-31B-it, GPT-OSS-120B, and Qwen3.8-27B math and Python runs. Gemma and GPT-OSS were the two planned models; Qwen was added as a follow-up and is analyzed separately. Percentages below weight math and Python equally within each model, rather than pooling their different numbers of episodes.

## Question and design

Does an agent perform work on a branch that is unnecessary for the graded target? Each episode asks the agent to **solve task 18 and submit only its result**. The 18 task descriptions specify each task's output format and dependencies, but the top-level instruction does not ask for all 18 results.

The full braid is the existing positive control. The **peninsula** deletes `task_17 -> task_18`; the **island** also deletes `task_1 -> task_3`. In both variants, `task_3, 10, 11, 12, 13, 14, 15, 17` are outside task 18's ancestry. The peninsula's optional branch still shares its root, `task_1`, with the required branch; the island's optional branch is disconnected. Each model's three conditions use matched target templates, graph assignments, task inputs, ground truths, wrong-input banks, and visible task labels. Only the specified edges, parent lists, and affected `get_input` instructions change.

Gemma has 111 matched math and 156 matched Python episodes (37 and 52 target templates). GPT-OSS has 96 math and 159 Python episodes (32 and 53 templates). Qwen has 99 math and 165 Python episodes (33 and 55 templates). There are three graph assignments per template. Every peninsula, island, and full-braid dataset used below has complete graded results. Gemma's runs, including its full-braid baseline, use local vLLM; GPT-OSS's runs use the API throughout. Qwen's new peninsula and island conditions use local vLLM. Its full-braid Python baseline is local, but its full-braid math baseline used the API.

## Primary result: complete the unnecessary upstream branch

The prespecified episode outcome is whether **all seven upstream optional tasks** (`task_3, 10, 11, 12, 13, 14, 15`) have correct outputs verified by their children's dependency reports. `task_17` is an optional leaf and cannot be verified this way. The prespecified one-sided test asks whether the island rate exceeds **10%**. Its p-values use target-template clusters and are Holm-adjusted across the two planned models. Qwen uses the same metric and threshold as a follow-up, without inclusion in that Holm family. The 95% intervals resample templates within each category, keeping their three graph assignments together (10,000 bootstrap draws).

| Model | Full braid | Peninsula | Island (95% CI) | Holm-adjusted p, island >10% | Peninsula − island (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma-4-31B-it | 74.7% | 27.4% | **18.3%** (13.6–23.3%) | 0.000749 | +9.1 pp (+1.9 to +16.3) |
| GPT-OSS-120B | 87.2% | 82.7% | **83.2%** (78.5–87.8%) | 1.10 × 10⁻³⁰ | −0.5 pp (−7.9 to +6.8) |

Gemma's full-braid-to-island decrease is **56.4 percentage points** (paired cluster 95% CI: 49.1–63.3 pp). GPT-OSS's decrease is **4.0 points** (95% CI: −2.1 to +10.2 pp); that interval includes zero. Thus both models exceed the meaningful 10% island rate, while their behavior under pruning differs sharply on their respective matched task sets. The direct difference between models is descriptive because their selected target-template pools differ.

The peninsula–island contrast is secondary. Gemma's +9.1-point contrast has an unadjusted paired cluster sign-flip p-value of **0.0192**; GPT-OSS's −0.5-point contrast has p = **0.911**. Gemma's contrast is concentrated in math:

| Model and category | Full braid | Peninsula | Island | Peninsula − island |
| --- | ---: | ---: | ---: | ---: |
| Gemma math | 75.7% | 29.7% | 13.5% | +16.2 pp |
| Gemma Python | 73.7% | 25.0% | 23.1% | +1.9 pp |
| GPT-OSS math | 78.1% | 72.9% | 74.0% | −1.0 pp |
| GPT-OSS Python | 96.2% | 92.5% | 92.5% | 0.0 pp |

The positive Gemma contrast is therefore suggestive of an effect of the shared root, especially in math; it is not a uniform result across both categories. GPT-OSS's interval does not establish equivalence between peninsula and island.

## Qwen follow-up: local peninsula and island runs

All four local Qwen variant datasets are complete and graded: 99 math and 165 Python episodes in each condition. The **island all-seven rate is 20.4%** (95% cluster bootstrap CI: 15.5–25.8%). Applying the same >10% test gives an exploratory, unadjusted p-value of **0.0000880**. The peninsula rate is **37.6%**, a **17.2-point** increase over island (95% paired cluster CI: 9.1–25.3 pp; two-sided paired sign-flip p = **0.000150**, unadjusted). This peninsula advantage appears in both categories:

| Qwen category | Full braid reference | Peninsula, local | Island, local | Peninsula − island |
| --- | ---: | ---: | ---: | ---: |
| Math | 52.5% (API) | 36.4% | 20.2% | +16.2 pp |
| Python | 66.1% (local) | 38.8% | 20.6% | +18.2 pp |
| Equal-category mean | 59.3% (mixed stacks) | 37.6% | 20.4% | +17.2 pp |

The peninsula–island contrast is entirely within the local serving stack. The existing full-braid **math** baseline used the API, so Qwen's pooled full-braid-to-island difference is only descriptive and should not be treated as a same-stack estimate of pruning. The completed API math variants are kept separate; the API Python variants were cancelled, so there is no complete Qwen API analysis. Qwen's follow-up p-values are not part of the planned Gemma/GPT-OSS multiplicity correction.

## Requests, effort, and target results

These descriptive measures also weight math and Python equally. A request means a parsed `get_input` call for the optional task. `task_17` is reported as a request only, since its output has no child to verify it.

| Model | Condition | Any optional task requested | Mean optional tasks requested, of 8 | `task_17` requested | Target correct | Mean tool calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gemma | Full braid | 99.5% | 7.87 | 98.5% | 50.9% | 21.57 |
| Gemma | Peninsula | 69.4% | 4.82 | 37.3% | 69.8% | 18.26 |
| Gemma | Island | **45.5%** | **3.24** | **24.7%** | 72.9% | 16.30 |
| GPT-OSS | Full braid | 100.0% | 7.89 | 99.0% | 80.3% | 19.07 |
| GPT-OSS | Peninsula | 98.6% | 7.60 | 91.6% | 81.8% | 18.87 |
| GPT-OSS | Island | **98.4%** | **7.72** | **92.2%** | 86.4% | 19.19 |
| Qwen | Full braid | 99.7% | 7.93 | 98.9% | 36.0% | 28.61 |
| Qwen | Peninsula, local | 83.9% | 5.82 | 59.0% | 59.6% | 26.27 |
| Qwen | Island, local | **71.9%** | **4.13** | **35.2%** | 58.2% | 22.54 |

Gemma often prunes the optional branch, reducing both optional requests and total tool use, but still completes all seven upstream optional tasks in 18.3% of island episodes. GPT-OSS requests nearly the entire optional branch even when it is disconnected and shows little reduction in tool use. Qwen lies between those patterns: it requests at least one optional task in 71.9% of island episodes and averages 4.13 of eight optional tasks requested. The models can also answer the target without the optional branch: island target accuracy is 72.9% for Gemma, 86.4% for GPT-OSS, and 58.2% for Qwen. Changes in target accuracy are descriptive because pruning also changes task 18's dependency count. Qwen's full-braid math row in this table has the serving-stack difference noted above.

## Interpretation and limitations

The result concerns work unnecessary **for the graded target under the actual prompt**. The top-level instruction requests only task 18, and `submit_answers` accepts only its result. Individual task blocks contain imperative output-format lines, which may nevertheless influence how an agent interprets the list of tasks. The harness also has a generic final-answer repair message saying “Finish any remaining tasks” when an agent responds without a tool call before submitting. A turn-trace audit found that the first optional-branch request preceded any response that could trigger this specific repair in all 51 Gemma island episodes that verified all seven tasks, in 201 of 218 corresponding GPT-OSS episodes, and in 53 of 54 Qwen episodes. This repair cannot explain most of the observed optional work.

Correctness of the seven upstream optional tasks is inferred from values supplied to their children in `get_input` calls; this tests whether the agent carried correct values through the branch. It does not verify the optional leaf `task_17`. All graded episodes, including failed target submissions and agent errors, remain in the denominators. The full braid is a matched-input positive control; its model generations are separate stochastic runs.

## Reproduction

- Protocol and generation: [`eval/branch_pruning_protocol.md`](eval/branch_pruning_protocol.md), [`generate/build_branch_pruning_graphs.py`](generate/build_branch_pruning_graphs.py).
- Analysis code and complete outputs: [`eval/analyze_branch_pruning.py`](eval/analyze_branch_pruning.py), [`eval/results/branch_pruning/analysis.json`](eval/results/branch_pruning/analysis.json), [`eval/results/branch_pruning_local/analysis_qwen_local.json`](eval/results/branch_pruning_local/analysis_qwen_local.json).
- Run the planned-model analysis from the repository root: `python -B eval/analyze_branch_pruning.py`.
- Run the Qwen follow-up: `python -B eval/analyze_branch_pruning.py --models Qwen3.8-27B --results-root eval/results/branch_pruning_local --output eval/results/branch_pruning_local/analysis_qwen_local.json`.
