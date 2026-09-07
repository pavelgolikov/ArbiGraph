#!/bin/bash
#SBATCH --job-name=arbigraph_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --output=logs/run_eval_%j.out
#SBATCH --error=logs/run_eval_%j.out
#SBATCH --account=CHANGE_ME          # your cluster allocation

# Full HF benchmark evaluation on Qwen3.5-27B:
# 3 categories x 7 topologies = 21 datasets, run in order, one command each.
# Comment out the blocks you do not want in a given submission.
#
# Repair attempts are budgets per task in the graph, so --final_answer_repair_attempts_per_task 4
# and --cutoff_repair_attempts_per_task 4 give 12 of each on the 3-node topologies and 24 of each
# on the 6-node topologies. --initial_tool_repair_attempts 5 is a flat per-sample budget.
#
# Progress is checkpointed per sample into <results>_partial/, so re-running this script
# after a timeout resumes each dataset where it stopped instead of starting over.

echo "Starting ArbiGraph Evaluation..."

# Point caches somewhere with quota. Override ARBIGRAPH_CACHE_ROOT on a cluster
# where $HOME is small.
ARBIGRAPH_CACHE_ROOT="${ARBIGRAPH_CACHE_ROOT:-$HOME/.cache/arbigraph}"
export HF_HOME="$ARBIGRAPH_CACHE_ROOT"
export XDG_CACHE_HOME="$ARBIGRAPH_CACHE_ROOT"
export NLTK_DATA="$ARBIGRAPH_CACHE_ROOT/nltk_data"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${USER}"
export VLLM_CONFIG_ROOT="/tmp/vllm_config_${USER}"
mkdir -p $HF_HOME
export NCCL_IGNORE_DISABLED_P2P=1
# NCCL Fixes

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0

# Repository root. Defaults to the parent of this script; override to run a checkout
# from elsewhere.
REPO_ROOT="${ARBIGRAPH_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT/eval" || exit 1

# # ============================== gsm ==============================
# echo "=== gsm/single_target_baseline $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/gsm/single_target_baseline/single_target_baseline_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/gsm/agent_single_target_baseline_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# echo "=== gsm/three_independent_target $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/gsm/three_independent_target/three_independent_target_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/gsm/agent_three_independent_target_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# echo "=== gsm/three_chain_target $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/gsm/three_chain_target/three_chain_target_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/gsm/agent_three_chain_target_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# echo "=== gsm/six_independent_target $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/gsm/six_independent_target/six_independent_target_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/gsm/agent_six_independent_target_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# echo "=== gsm/six_chain_target $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/gsm/six_chain_target/six_chain_target_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/gsm/agent_six_chain_target_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# echo "=== gsm/four_way_fan_in_target $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/gsm/four_way_fan_in_target/four_way_fan_in_target_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/gsm/agent_four_way_fan_in_target_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# echo "=== gsm/two_branch_recombine_target $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/gsm/two_branch_recombine_target/two_branch_recombine_target_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/gsm/agent_two_branch_recombine_target_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# ============================== math ==============================
# echo "=== math/single_target_baseline $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/math/single_target_baseline/single_target_baseline_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/math/agent_single_target_baseline_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

# echo "=== math/three_independent_target $(date -Is) ==="
# python run_agent_calc.py \
#   --input "$REPO_ROOT"/source/generated/hf_datasets/math/three_independent_target/three_independent_target_dataset.json \
#   --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/math/agent_three_independent_target_Qwen3.5-27B.json \
#   --model Qwen/Qwen3.5-27B \
#   --num_gpus 4 \
#   --max_tokens 16384 \
#   --repair_turn_max_tokens 32768 \
#   --initial_tool_repair_attempts 5 \
#   --final_answer_repair_attempts_per_task 4 \
#   --cutoff_repair_attempts_per_task 4 \
#   --require_initial_calculator_call \
#   --max_agent_turns 1200 \
#   --max_concurrent_samples 128

echo "=== math/three_chain_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/math/three_chain_target/three_chain_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/math/agent_three_chain_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== math/six_independent_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/math/six_independent_target/six_independent_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/math/agent_six_independent_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== math/six_chain_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/math/six_chain_target/six_chain_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/math/agent_six_chain_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== math/four_way_fan_in_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/math/four_way_fan_in_target/four_way_fan_in_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/math/agent_four_way_fan_in_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== math/two_branch_recombine_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/math/two_branch_recombine_target/two_branch_recombine_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/math/agent_two_branch_recombine_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

# ============================== python ==============================
echo "=== python/single_target_baseline $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/python/single_target_baseline/single_target_baseline_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/python/agent_single_target_baseline_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== python/three_independent_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/python/three_independent_target/three_independent_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/python/agent_three_independent_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== python/three_chain_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/python/three_chain_target/three_chain_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/python/agent_three_chain_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== python/six_independent_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/python/six_independent_target/six_independent_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/python/agent_six_independent_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== python/six_chain_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/python/six_chain_target/six_chain_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/python/agent_six_chain_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== python/four_way_fan_in_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/python/four_way_fan_in_target/four_way_fan_in_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/python/agent_four_way_fan_in_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "=== python/two_branch_recombine_target $(date -Is) ==="
python run_agent_calc.py \
  --input "$REPO_ROOT"/source/generated/hf_datasets/python/two_branch_recombine_target/two_branch_recombine_target_dataset.json \
  --output "$REPO_ROOT"/eval/results/hf_datasets/Qwen3.5-27B/python/agent_two_branch_recombine_target_Qwen3.5-27B.json \
  --model Qwen/Qwen3.5-27B \
  --num_gpus 4 \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --initial_tool_repair_attempts 5 \
  --final_answer_repair_attempts_per_task 4 \
  --cutoff_repair_attempts_per_task 4 \
  --require_initial_calculator_call \
  --max_agent_turns 1200 \
  --max_concurrent_samples 128

echo "All datasets finished at $(date -Is)"
