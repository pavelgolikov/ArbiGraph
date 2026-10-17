#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_exp1
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/run_exp1_%j.out
#SBATCH --error=logs/run_exp1_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
echo "Starting ArbiGraph Experiment 1..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${USER}"
export VLLM_CONFIG_ROOT="/tmp/vllm_config_${USER}"
mkdir -p "$HF_HOME"
export NCCL_IGNORE_DISABLED_P2P=1
# NCCL Fixes

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0

cd /home/golikovp/projects/aip-gpekhime/golikovp/ArbiGraph-private/eval || exit 1

MODELS=(
  gemma-4-31B-it
  # Qwen3.8-27B
  # Nemotron-3-Nano-Omni-30B-A3B
  # GPT-OSS-120B
)

declare -A MODEL_PATHS=(
  [gemma-4-31B-it]="google/gemma-4-31B-it"
  [Qwen3.8-27B]="Qwen/Qwen3.8-27B"
  [Nemotron-3-Nano-Omni-30B-A3B]="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
  [GPT-OSS-120B]="openai/gpt-oss-120b"
)

declare -A MODEL_CONTEXT_LENGTHS=(
  [gemma-4-31B-it]=262144
  [Qwen3.8-27B]=262144
  [Nemotron-3-Nano-Omni-30B-A3B]=131072
  [GPT-OSS-120B]=131072
)

# Reasoning effort has no shared vocabulary across these three, so it cannot live in
# COMMON. Qwen takes xhigh (its default) and its template raises on "high"; gpt-oss
# takes low/medium/high and interpolates whatever it is given straight into the
# harmony system block, so "xhigh" there is a string the model never saw in training;
# Nemotron ignores the argument entirely and always reasons. Gemma 4's template has no
# effort levels and ignores it too: MODEL_THINKING below turns its reasoning on.
declare -A MODEL_REASONING_EFFORT=(
  [gemma-4-31B-it]=unused
  [Qwen3.8-27B]=xhigh
  [Nemotron-3-Nano-Omni-30B-A3B]=xhigh
  [GPT-OSS-120B]=high
)

# Gemma 4's template leaves thinking off unless asked.
declare -A MODEL_THINKING=(
  [gemma-4-31B-it]=--enable_thinking
)

# Temperature 1.0 and top_p 0.95 are every model's; top_k follows each vendor's
# recommendation, which is 64 for Gemma 4.
declare -A MODEL_TOP_K=(
  [gemma-4-31B-it]=64
  [Qwen3.8-27B]=20
  [Nemotron-3-Nano-Omni-30B-A3B]=20
  [GPT-OSS-120B]=20
)

# The current datasets use the target-specific opening instruction. Keep their
# results separate from completed runs generated with the earlier generic prompt.
RUN_TAG=target_only

# Comment out individual entries to exclude them from the sweep.
CATEGORIES=(
  # math
  python
  rag
  rlve
)

TOPOLOGIES=(
  single_target_baseline
  # three_independent_target
  # three_chain_target
  # six_independent_target
  # six_chain_target
  # four_way_fan_in_target
  # two_branch_recombine_target
)

COMMON=(
  --num_gpus 4
  --temperature 1.0
  --top_p 0.95
  --min_p 0.0
  --presence_penalty 0.0
  --repetition_penalty 1.0
  --preserve_thinking
  --max_tokens 16384
  --repair_turn_max_tokens 32768
  --tool_repair_attempts 5
  --final_answer_repair_attempts 20
  --cutoff_repair_attempts 20
  --max_agent_turns 2000
  --max_turns_without_new_answer 200
  --max_concurrent_samples 12
)

for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$MODEL]}"
  MAX_MODEL_LEN="${MODEL_CONTEXT_LENGTHS[$MODEL]}"
  REASONING_EFFORT="${MODEL_REASONING_EFFORT[$MODEL]}"
  THINKING="${MODEL_THINKING[$MODEL]:-}"
  TOP_K="${MODEL_TOP_K[$MODEL]}"
  RESULT_ROOT="results/exp1/$MODEL"
  echo "=== model $MODEL ($MODEL_PATH), max context $MAX_MODEL_LEN, reasoning $REASONING_EFFORT ${THINKING}, top_k $TOP_K $(date -Is) ==="

  for category in "${CATEGORIES[@]}"; do
    DATA_ROOT="../generate/generated/exp1/$category"
    GRAPH_ROOT="../generate/topologies/$category/exp1"

    for topology in "${TOPOLOGIES[@]}"; do
      echo "=== $MODEL/$category/$topology $(date -Is) ==="
      python run_agent.py \
        --input "$DATA_ROOT/$topology/${topology}_dataset.json" \
        --output "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
        --model "$MODEL_PATH" \
        --max_model_len "$MAX_MODEL_LEN" \
        --reasoning_effort "$REASONING_EFFORT" \
        ${THINKING:+"$THINKING"} \
        --top_k "$TOP_K" \
        "${COMMON[@]}"
      python -B grade_results.py \
        --input "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
        --graph "$GRAPH_ROOT/$topology.json"
    done
  done
done

echo "All datasets finished at $(date -Is)"
