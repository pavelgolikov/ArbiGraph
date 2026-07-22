# ArbiGraph

ArbiGraph is a benchmark generator for evaluating context management in
tool-assisted language agents. It builds verifiable task graphs whose nodes are
plain-English tasks with typed scalar or list-valued inputs and outputs. The
generator can instantiate independent distractor tasks, linear chains, and
branched multichain layouts, then score the final requested answer against
executable ground truth.

This branch is the release snapshot for the first public ArbiGraph release. It
contains the code, generated datasets, and Qwen3.5-27B result files used for the
reported initial evaluation.

## Contents

- `generate_baseline.py`, `generate_forgetting.py`, `generate_chain.py`,
  `generate_multichain.py`: dataset generators for the evaluated settings.
- `tasks/`: task implementations for math, Python tracing, GSM-style, custom,
  and prototype image tasks.
- `run_agent_calc.py`: calculator-assisted agent evaluation harness.
- `grader.py`: answer parsing and grading utilities.
- `results/`: result JSON files for the initial evaluation.

## Evaluated Snapshot

The release evaluation uses:

- Settings: baseline, forgetting, chain, and multichain.
- Task categories: math, Python tracing, and GSM-style arithmetic.
- Model: Qwen3.5-27B with calculator access.
- Samples: 16 generated samples per target task.
- Target tasks: 40 math tasks, 80 Python tasks, and 41 GSM-style tasks.

GSM-style multichain is not included in this release snapshot.

## Reproducing

See `REPRODUCING.md` for the commands used to regenerate datasets and
evaluation results.

GSM-style dataset generation uses the Apple GSM-Symbolic templates from
https://github.com/apple/ml-gsm-symbolic.

Files adapted from GSM-Symbolic are the corrected template overrides in
`tasks/gsm_task/corrected_templates/`: `0010.json`, `0016.json`,
`0024.json`, `0039.json`, `0048.json`, `0050.json`, `0060.json`,
`0062.json`, `0082.json`, and `0096.json`. The GSM datasets in this release
contain generated problem instances derived from GSM-Symbolic templates.

## Current Scope

This is an initial benchmark release. It is intended to make the graph
construction, adapters, generated datasets, evaluation harness, and result
artifacts inspectable and reproducible. Additional model evaluations, multimodal
task categories, and richer real-world workflow instantiations are in progress.
