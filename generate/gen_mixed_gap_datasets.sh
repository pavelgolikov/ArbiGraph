#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

# Prescreen uses 8 repeats; calibration uses 30 fresh repeats of the selected
# pinned questions. Comment out a stage to generate only the current dataset.
CATEGORIES=(
  rag
  rlve
)
STAGES=(
  # prescreen
  # calibration
)

GRAPH_MODELS=(
  Qwen3.8-27B
  GPT-OSS-120B
  gemma-4-31B-it
)

declare -A STAGE_SAMPLES=(
  [prescreen]=8
  [calibration]=30
)
declare -A STAGE_SEEDS=(
  [prescreen]=11
  [calibration]=23
)
declare -A STAGE_POOL_PREFIX=(
  [prescreen]=""
  [calibration]="selected_"
)

COMMON=(
  --pin_index 0
  --get_input_attempts_per_task 5
)

for stage in "${STAGES[@]}"; do
  for category in "${CATEGORIES[@]}"; do
    POOL="generated/mixed_gap/${STAGE_POOL_PREFIX[$stage]}${category}_ids.json"
    topology="${category}_${stage}"
    DATA_ROOT="generated/mixed_gap"
    echo "=== $category/$topology $(date -Is) ==="
    python -B ../source/generate_dataset.py \
      --graph "topologies/$category/exp1/single_target_baseline.json" \
      --output_dir "$DATA_ROOT/$topology" \
      --num_samples_per_task "${STAGE_SAMPLES[$stage]}" \
      --seed "${STAGE_SEEDS[$stage]}" \
      --pool_task_ids "$POOL" \
      --target_task_ids "$POOL" \
      "${COMMON[@]}"
    python -B dataset_to_text.py \
      --input "$DATA_ROOT/$topology/${topology}_dataset.json"
  done
done

# Build paired mixed chain, braid, and fan-out/fan-in datasets.
for MODEL in "${GRAPH_MODELS[@]}"; do
  python -B build_mixed_gap_graph.py --model "$MODEL"
done

echo "All mixed-gap datasets finished at $(date -Is)"
