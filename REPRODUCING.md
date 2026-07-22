# Reproducing the ArbiGraph Release Snapshot

This document records the commands and settings for the first ArbiGraph
release snapshot. Commands assume they are run from the repository root.

## Dataset Generation

The release uses 16 generated samples per target task.
The GSM-style generation commands require the Apple GSM-Symbolic templates:

```bash
git clone https://github.com/apple/ml-gsm-symbolic.git ml-gsm-symbolic
```

```bash
python generate_baseline.py --mode math --num_tasks 40 --num_samples_per_task 16 --output math_baseline.json
python generate_baseline.py --mode python --num_tasks 80 --num_samples_per_task 16 --output python_baseline.json
python generate_baseline.py --mode gsm --num_tasks 41 --num_samples_per_task 16 --output gsm_baseline.json

python tasks/math_task/forgetting/generate_math_forgetting.py --input math_baseline.json --output math_forgetting.json --num_distractors 3 --seed 0
python tasks/python_task/forgetting/generate_python_forgetting.py --input python_baseline.json --output python_forgetting.json --num_distractors 3 --seed 0
python tasks/gsm_task/forgetting/generate_gsm_forgetting.py --input gsm_baseline.json --output gsm_forgetting.json --num_distractors 3 --seed 0

python tasks/math_task/chain/generate_math_chain.py --output math_chain.json --num_distractors 3 --num_samples_per_task 16 --seed 0
python tasks/python_task/chain/generate_python_chain.py --output python_chain.json --num_distractors 3 --num_samples_per_task 16 --seed 0
python tasks/gsm_task/chain/generate_gsm_chain.py --output gsm_chain.json --num_distractors 3 --num_samples_per_task 16 --seed 0

python generate_multichain.py --structure "['math','math',['math','math'],['math','math'],'math','math']" --num_samples_per_task 16 --seed 0 --output math_multichain.json
python generate_multichain.py --structure "['python','python',['python','python'],['python','python'],'python','python']" --num_samples_per_task 16 --seed 0 --output python_multichain.json
```

## Agent Evaluation

The evaluation harness requires a local model runtime compatible with vLLM.
The release result files are already included under `results/`.

The repair budgets used in the included runs are:

- Baseline: `--initial_tool_repair_attempts 5`, `--final_answer_repair_attempts 3`,
  `--cutoff_repair_attempts 3`.
- Forgetting and chain: `5`, `12`, `12`.
- Multichain: `5`, `24`, `24`.

Example baseline command:

```bash
python run_agent_calc.py \
  --num_gpus 4 \
  --input math_baseline.json \
  --output-dir results/baseline/Qwen3.5-27B/math/repair \
  --model Qwen/Qwen3.5-27B \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts 3 \
  --cutoff_repair_attempts 3 \
  --require_initial_calculator_call
```

For the exact cluster job scripts used during the snapshot preparation, see
`run_job_baseline.sh`, `run_job_forgetting.sh`, `run_job_chain.sh`, and
`run_job_multichain.sh`. On a SLURM cluster, submit these scripts with
`sbatch`, for example `sbatch run_job_baseline.sh`.
