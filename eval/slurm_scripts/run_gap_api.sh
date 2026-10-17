#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_gap_api
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/run_gap_api_%j.out
#SBATCH --error=logs/run_gap_api_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
echo "Starting the ArbiGraph prediction-gap runs via Vector API..."

# Use the same environment as run_exp1.sh; inference runs on Vector.
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${USER}"
export VLLM_CONFIG_ROOT="/tmp/vllm_config_${USER}"
mkdir -p "$HF_HOME"
export NCCL_IGNORE_DISABLED_P2P=1

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0

cd /home/golikovp/projects/aip-gpekhime/golikovp/ArbiGraph-private/eval || exit 1

# Qwen3.8-27B | GPT-OSS-120B
MODEL=GPT-OSS-120B
RESULT_ROOT="results/gap/$MODEL"
# Keeps two stacks running the same dataset from colliding. Empty for one stack.
# GPT-OSS-120B runs on this stack alone, so its files are untagged, which is where
# plots/plot_prediction_gap.py reads each topology. Its prompts are rendered with Harmony,
# and the API runner checks every turn's prompt tokens against Vector's, so a local
# run of the same datasets would give the model the same prompts.
RUN_TAG=

# Same per-model settings as run_exp1_api.sh.
declare -A MODEL_PATHS=(
  [Qwen3.8-27B]="Qwen/Qwen3.8-27B"
  [GPT-OSS-120B]="openai/gpt-oss-120b"
)
declare -A MODEL_API_IDS=(
  [Qwen3.8-27B]=Qwen3_8-27B
  [GPT-OSS-120B]=gpt-oss-120b
)
declare -A MODEL_CONTEXT_LENGTHS=(
  [Qwen3.8-27B]=262144
  [GPT-OSS-120B]=131072
)
declare -A MODEL_REASONING_EFFORT=(
  [Qwen3.8-27B]=xhigh
  [GPT-OSS-120B]=high
)

# Comment out individual entries to exclude them from the sweep.
CATEGORIES=(
  math
  python
)

# The calibration comes first: it is the cheap half and it fixes the predicted bar,
# whose error bar no number of graph samples can narrow.
TOPOLOGIES=(
  single_target_baseline
  eighteen_chain_target
  eighteen_braid_target
  eighteen_fan_out_fan_in
  eighteen_braid_target_shuffled_labels
)

# Arguments replace the lists above, so each category and topology can run as its
# own job alongside the others:
#   sbatch run_gap_api.sh math eighteen_chain_target
# Each category and topology has its own results file; never give two jobs the same pair.
if (( $# >= 1 )); then CATEGORIES=("$1"); fi
if (( $# >= 2 )); then TOPOLOGIES=("${@:2}"); fi

COMMON=(
  --model "${MODEL_PATHS[$MODEL]}"
  --api_model "${MODEL_API_IDS[$MODEL]}"
  --temperature 1.0
  --top_p 0.95
  --top_k 20
  --min_p 0.0
  --presence_penalty 0.0
  --repetition_penalty 1.0
  --reasoning_effort "${MODEL_REASONING_EFFORT[$MODEL]}"
  --preserve_thinking
  --max_model_len "${MODEL_CONTEXT_LENGTHS[$MODEL]}"
  --max_tokens 16384
  --repair_turn_max_tokens 32768
  --tool_repair_attempts 5
  --get_input_attempts_per_task 5
  --final_answer_repair_attempts 20
  --cutoff_repair_attempts 20
  --max_agent_turns 2000
  --max_turns_without_new_answer 200
  --max_concurrent_samples 32
)

for category in "${CATEGORIES[@]}"; do
  DATA_ROOT="../generate/generated/gap/$MODEL/$category"

  for topology in "${TOPOLOGIES[@]}"; do
    echo "=== $category/$topology $(date -Is) ==="
    python run_agent_api.py \
      --input "$DATA_ROOT/$topology/${topology}_dataset.json" \
      --output "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
      "${COMMON[@]}"
    python grade_results.py \
      --input "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
      --graph "$DATA_ROOT/$topology/${topology}_input_graph.json"
  done
done

echo "All gap runs finished at $(date -Is)"
