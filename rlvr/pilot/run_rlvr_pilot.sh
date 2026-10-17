#!/bin/bash
#SBATCH --job-name=golikovp_rlvr_pilot
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_rlvr_pilot_%j.out
#SBATCH --error=logs/run_rlvr_pilot_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail

export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${USER}"
export VLLM_CONFIG_ROOT="/tmp/vllm_config_${USER}"
export NCCL_IGNORE_DISABLED_P2P=1

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0

mkdir -p "$HF_HOME" /home/golikovp/projects/aip-gpekhime/golikovp/ArbiGraph-private/eval/results/phase_3
cd /home/golikovp/projects/aip-gpekhime/golikovp/ArbiGraph-private || exit 1

python rlvr/pilot/prepare.py \
  --output generate/generated/phase_3/math_pilot.json \
  --rollouts-per-prompt 4 \
  --overwrite

python eval/run_agent.py \
  --input generate/generated/phase_3/math_pilot.json \
  --output eval/results/phase_3/Qwen3.5-9B.json \
  --model Qwen/Qwen3.5-9B \
  --num_gpus 1 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --presence_penalty 0.0 \
  --repetition_penalty 1.0 \
  --reasoning_effort xhigh \
  --preserve_thinking \
  --max_model_len 262144 \
  --max_tokens 16384 \
  --tool_repair_attempts 0 \
  --final_answer_repair_attempts 0 \
  --cutoff_repair_attempts 0 \
  --max_agent_turns 2000 \
  --max_turns_without_new_answer 200 \
  --max_concurrent_samples 128

python eval/grade_results.py \
  --input eval/results/phase_3/Qwen3.5-9B.json

python rlvr/pilot/analyze.py \
  --input eval/results/phase_3/Qwen3.5-9B.json \
  --output eval/results/phase_3/Qwen3.5-9B_analysis.json \
  --overwrite
