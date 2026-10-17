#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_mixed_gap_local
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/run_mixed_gap_local_%j.out
#SBATCH --error=logs/run_mixed_gap_local_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
echo "Starting the ArbiGraph mixed-gap runs locally..."

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

MODELS=(
  gemma-4-31B-it
)

declare -A MODEL_PATHS=(
  [gemma-4-31B-it]="google/gemma-4-31B-it"
)
declare -A MODEL_CONTEXT_LENGTHS=(
  [gemma-4-31B-it]=262144
)
declare -A MODEL_REASONING_EFFORT=(
  [gemma-4-31B-it]=unused
)
declare -A MODEL_TOP_K=(
  [gemma-4-31B-it]=64
)

CATEGORIES=(
  # rag
  # rlve
  mixed
)

# Comment out entries to run one topology per job.
STAGES=(
  # prescreen
  # calibration
  eighteen_chain_target
  eighteen_braid_target
  eighteen_fan_out_fan_in
)

# Keep Gemma's settings identical to its existing gap runs.
COMMON=(
  --num_gpus 4
  --temperature 1.0
  --top_p 0.95
  --min_p 0.0
  --presence_penalty 0.0
  --repetition_penalty 1.0
  --enable_thinking
  --preserve_thinking
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

for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$MODEL]}"
  MAX_MODEL_LEN="${MODEL_CONTEXT_LENGTHS[$MODEL]}"
  REASONING_EFFORT="${MODEL_REASONING_EFFORT[$MODEL]}"
  TOP_K="${MODEL_TOP_K[$MODEL]}"
  echo "=== model $MODEL ($MODEL_PATH) $(date -Is) ==="

  for stage in "${STAGES[@]}"; do
    for category in "${CATEGORIES[@]}"; do
      if [[ "$category" == mixed && "$stage" == eighteen_* ]]; then
        topology="$stage"
        DATA_ROOT="../generate/generated/mixed_gap/$MODEL/$category"
        RESULT_ROOT="results/mixed_gap/$MODEL/$category"
      elif [[ "$category" != mixed && "$stage" != eighteen_* ]]; then
        topology="${category}_${stage}"
        DATA_ROOT="../generate/generated/mixed_gap"
        RESULT_ROOT="results/mixed_gap/$stage/$MODEL/$category"
      else
        continue
      fi
      dataset="$DATA_ROOT/$topology/${topology}_dataset.json"
      graph="$DATA_ROOT/$topology/${topology}_input_graph.json"
      output="$RESULT_ROOT/agent_${topology}_${MODEL}.json"
      if [[ ! -f "$dataset" || ! -f "$graph" ]]; then
        echo "Missing $dataset or $graph. Run generate/gen_mixed_gap_datasets.sh first." >&2
        exit 1
      fi

      echo "=== $MODEL/$category/$topology $(date -Is) ==="
      python run_agent.py \
        --input "$dataset" \
        --output "$output" \
        --model "$MODEL_PATH" \
        --max_model_len "$MAX_MODEL_LEN" \
        --reasoning_effort "$REASONING_EFFORT" \
        --top_k "$TOP_K" \
        "${COMMON[@]}"
      python -B grade_results.py \
        --input "$output" \
        --graph "$graph"
    done
  done
done

echo "All local mixed-gap runs finished at $(date -Is)"
