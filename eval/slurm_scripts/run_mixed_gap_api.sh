#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_mixed_gap_api
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/run_mixed_gap_api_%j.out
#SBATCH --error=logs/run_mixed_gap_api_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
echo "Starting the ArbiGraph mixed-gap runs via Vector API..."

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

# Comment out entries to exclude them from the default sweep.
MODELS=(
  Qwen3.8-27B
  GPT-OSS-120B
)

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

# The launcher passes one model, category and stage per job, as in run_gap_api.sh.
if (( $# >= 1 )); then MODELS=("$1"); fi
if (( $# >= 2 )); then CATEGORIES=("$2"); fi
if (( $# >= 3 )); then STAGES=("${@:3}"); fi

COMMON=(
  --temperature 1.0
  --top_p 0.95
  --top_k 20
  --min_p 0.0
  --presence_penalty 0.0
  --repetition_penalty 1.0
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
  API_MODEL="${MODEL_API_IDS[$MODEL]}"
  MAX_MODEL_LEN="${MODEL_CONTEXT_LENGTHS[$MODEL]}"
  REASONING_EFFORT="${MODEL_REASONING_EFFORT[$MODEL]}"
  echo "=== model $MODEL ($API_MODEL) $(date -Is) ==="

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
      python run_agent_api.py \
        --input "$dataset" \
        --output "$output" \
        --model "$MODEL_PATH" \
        --api_model "$API_MODEL" \
        --max_model_len "$MAX_MODEL_LEN" \
        --reasoning_effort "$REASONING_EFFORT" \
        "${COMMON[@]}"
      python -B grade_results.py \
        --input "$output" \
        --graph "$graph"
    done
  done
done

echo "All mixed-gap runs finished at $(date -Is)"
