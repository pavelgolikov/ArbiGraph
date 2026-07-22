#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=base_eval_out.out
#SBATCH --error=base_eval_out.out
#SBATCH --account=aip-gpekhime


# Run Evaluation
echo "Starting Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
mkdir -p $HF_HOME
export NCCL_IGNORE_DISABLED_P2P=1
# NCCL Fixes

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0

# # Qwen3.5-27B ======================================================================
# # NO REPAIR ------------------------------------------------
# python run_agent_calc.py \
#     --num_gpus 4 \
#     --input gsm_baseline.json \
#     --output-dir results/baseline/Qwen3.5-27B/gsm/no_repair \
#     --model Qwen/Qwen3.5-27B \
#     --max_tokens 16384 \
#     --initial_tool_repair_attempts 0 \
#     --final_answer_repair_attempts 0 \
#     --cutoff_repair_attempts 0

# python run_agent_calc.py \
#     --num_gpus 4 \
#     --input python_baseline.json \
#     --output-dir results/baseline/Qwen3.5-27B/py/no_repair \
#     --model Qwen/Qwen3.5-27B \
#     --max_tokens 16384 \
#     --initial_tool_repair_attempts 0 \
#     --final_answer_repair_attempts 0 \
#     --cutoff_repair_attempts 0

# python run_agent_calc.py \
#     --num_gpus 4 \
#     --input math_baseline.json \
#     --output-dir results/baseline/Qwen3.5-27B/math/no_repair \
#     --model Qwen/Qwen3.5-27B \
#     --max_tokens 16384 \
#     --initial_tool_repair_attempts 0 \
#     --final_answer_repair_attempts 0 \
#     --cutoff_repair_attempts 0

# # # REPAIR -------------------------
# python run_agent_calc.py \
#     --num_gpus 4 \
#     --input gsm_baseline.json \
#     --output-dir results/baseline/Qwen3.5-27B/gsm/repair \
#     --model Qwen/Qwen3.5-27B \
#     --max_tokens 16384 \
#     --repair_turn_max_tokens 32768 \
#     --initial_tool_repair_attempts 5 \
#     --final_answer_repair_attempts 3 \
#     --cutoff_repair_attempts 3 \
#     --require_initial_calculator_call

python run_agent_calc.py \
    --num_gpus 4 \
    --input results/baseline/Qwen3.5-27B/py/repair/agent_python_baseline_Qwen3.5-27B_20260705_234750.json \
    --output results/baseline/Qwen3.5-27B/py/repair/agent_python_baseline_Qwen3.5-27B_20260705_234750.json \
    --model Qwen/Qwen3.5-27B \
    --max_tokens 16384 \
    --repair_turn_max_tokens 32768 \
    --initial_tool_repair_attempts 5 \
    --final_answer_repair_attempts 3 \
    --cutoff_repair_attempts 3 \
    --require_initial_calculator_call

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
