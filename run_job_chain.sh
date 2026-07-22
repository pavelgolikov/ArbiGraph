#!/bin/bash
#SBATCH --job-name=golikovp_chain
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=chain_eval_out.out
#SBATCH --error=chain_eval_out.out
#SBATCH --account=aip-gpekhime


# Run Chain Evaluation
echo "Starting Chain Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $HF_HOME
export NCCL_IGNORE_DISABLED_P2P=1
# NCCL Fixes

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0


# Qwen3.5-27B ======================================================================
python run_agent_calc.py \
    --num_gpus 4 \
    --input results/chain/Qwen3.5-27B/math/repair/agent_math_chain_Qwen3.5-27B_20260707_130726.json \
    --output results/chain/Qwen3.5-27B/math/repair/agent_math_chain_Qwen3.5-27B_20260707_130726.json \
    --model Qwen/Qwen3.5-27B \
    --max_tokens 16384 \
    --repair_turn_max_tokens 32768 \
    --initial_tool_repair_attempts 5 \
    --final_answer_repair_attempts 12 \
    --cutoff_repair_attempts 12 \
    --require_initial_calculator_call \
    --max_agent_turns 1200

python run_agent_calc.py \
    --num_gpus 4 \
    --input python_chain.json \
    --output-dir results/chain/Qwen3.5-27B/python/repair \
    --model Qwen/Qwen3.5-27B \
    --max_tokens 16384 \
    --repair_turn_max_tokens 32768 \
    --initial_tool_repair_attempts 5 \
    --final_answer_repair_attempts 12 \
    --cutoff_repair_attempts 12 \
    --require_initial_calculator_call \
    --max_agent_turns 1200

python run_agent_calc.py \
    --num_gpus 4 \
    --input gsm_chain.json \
    --output-dir results/chain/Qwen3.5-27B/gsm/repair \
    --model Qwen/Qwen3.5-27B \
    --max_tokens 16384 \
    --repair_turn_max_tokens 32768 \
    --initial_tool_repair_attempts 5 \
    --final_answer_repair_attempts 12 \
    --cutoff_repair_attempts 12 \
    --require_initial_calculator_call \
    --max_agent_turns 1200