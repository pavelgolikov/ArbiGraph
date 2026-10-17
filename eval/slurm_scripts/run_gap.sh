#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_gap
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_gap_%j.out
#SBATCH --error=logs/run_gap_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
echo "Starting the ArbiGraph prediction-gap runs..."

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

# Qwen3.8-27B | GPT-OSS-120B | gemma-4-31B-it
MODEL=gemma-4-31B-it
RESULT_ROOT="results/gap/$MODEL"
# Keeps two stacks running the same dataset from colliding. Empty for one stack.
RUN_TAG=

# Same per-model settings as run_exp1.sh, so the calibration repeats each pinned
# problem under the conditions that screened it into the pool.
declare -A MODEL_PATHS=(
  [Qwen3.8-27B]="Qwen/Qwen3.8-27B"
  [GPT-OSS-120B]="openai/gpt-oss-120b"
  [gemma-4-31B-it]="google/gemma-4-31B-it"
)
declare -A MODEL_CONTEXT_LENGTHS=(
  [Qwen3.8-27B]=262144
  [GPT-OSS-120B]=131072
  [gemma-4-31B-it]=262144
)
declare -A MODEL_REASONING_EFFORT=(
  [Qwen3.8-27B]=xhigh
  [GPT-OSS-120B]=high
  [gemma-4-31B-it]=unused
)
declare -A MODEL_THINKING=(
  [gemma-4-31B-it]=--enable_thinking
)
declare -A MODEL_TOP_K=(
  [Qwen3.8-27B]=20
  [GPT-OSS-120B]=20
  [gemma-4-31B-it]=64
)
THINKING="${MODEL_THINKING[$MODEL]:-}"

# Comment out individual entries to exclude them from the sweep.
CATEGORIES=(
  math
  python
)

# Each topology runs for both categories before the next one starts, so the work that
# matters most finishes first. The calibration is the cheap half and it fixes the
# predicted bar, whose error bar no number of graph samples can narrow. The chains and
# braids carry the headline gap; the fan-out/fan-in and shuffled-label runs come last.
TOPOLOGIES=(
  # single_target_baseline
  # eighteen_chain_target
  # eighteen_braid_target
  eighteen_fan_out_fan_in
  eighteen_braid_target_shuffled_labels
)

# Arguments replace the lists above, to run one category or a few topologies:
#   sbatch run_gap.sh python eighteen_chain_target eighteen_braid_target
# A job stopped by its time limit resumes where it left off when submitted again, and
# runs that finished are skipped without loading the model. Queue several to run back
# to back, and stop them all by name:
#   for i in 1 2 3 4 5 6; do sbatch --dependency=singleton run_gap.sh; done
#   scancel -u "$USER" -n golikovp_arbigraph_gap
# Never let two jobs run the same category and topology: both would write its results
# file. GPT-OSS-120B's gap runs are on the API (run_gap_api.sh) and write these same
# untagged files, so never run it here while those jobs are queued or running.
if (( $# >= 1 )); then CATEGORIES=("$1"); fi
if (( $# >= 2 )); then TOPOLOGIES=("${@:2}"); fi

# Samples in flight set throughput only. At 32, eighteen-node graphs fill about half of
# gemma-4-31B-it's KV cache on four H100s.
COMMON=(
  --model "${MODEL_PATHS[$MODEL]}"
  --num_gpus 4
  --temperature 1.0
  --top_p 0.95
  --top_k "${MODEL_TOP_K[$MODEL]}"
  --min_p 0.0
  --presence_penalty 0.0
  --repetition_penalty 1.0
  --reasoning_effort "${MODEL_REASONING_EFFORT[$MODEL]}"
  ${THINKING:+"$THINKING"}
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

for topology in "${TOPOLOGIES[@]}"; do
  for category in "${CATEGORIES[@]}"; do
    # The generated directory carries a copy of each topology, so one root serves all
    # of them even though they come from different topology directories.
    DATA_ROOT="../generate/generated/gap/$MODEL/$category"

    echo "=== $category/$topology $(date -Is) ==="
    python run_agent.py \
      --input "$DATA_ROOT/$topology/${topology}_dataset.json" \
      --output "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
      "${COMMON[@]}"
    python -B grade_results.py \
      --input "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
      --graph "$DATA_ROOT/$topology/${topology}_input_graph.json"
  done
done

echo "All gap runs finished at $(date -Is)"
