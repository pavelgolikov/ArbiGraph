#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

# Datasets for the prediction-gap experiment. Every task carries one pinned
# instantiation, so a calibration dataset's samples all hold the same problem and
# differ only in how the model answers. Run screen_task_pools.py first.
#
#   ./gen_gap_datasets.sh GPT-OSS-120B
#
# The model names the pool to read and the directory to write, so datasets for one
# model never overwrite another's.

MODEL="${1:?usage: $0 <model>, e.g. GPT-OSS-120B}"
POOL_ROOT="pools/$MODEL"
DATA_ROOT="generated/gap/$MODEL"

CATEGORIES=(
  math
  python
)

# The single-target baseline is the calibration: its samples per task are the K
# repeats of each pinned problem, sized so the predicted bar's standard error lands
# at or under the measured bar's. Qwen3.8-27B ran its python dataset three times,
# to 45 repeats. GPT-OSS-120B and gemma-4-31B-it solve their python tasks so reliably
# that 30 suffice.
declare -A CALIBRATION_SAMPLES=(
  [Qwen3.8-27B/math]=50
  [Qwen3.8-27B/python]=15
  [GPT-OSS-120B/math]=50
  [GPT-OSS-120B/python]=30
  [gemma-4-31B-it/math]=50
  [gemma-4-31B-it/python]=30
)

# <topology under topologies/<category>/>:<samples per target task>
TOPOLOGIES=(
  "test/eighteen_chain_target:3"
  "test/eighteen_braid_target:3"
  "test/eighteen_fan_out_fan_in:3"
)

SHUFFLED_LABEL_TOPOLOGIES=(
  "test/eighteen_braid_target:3"
)

COMMON=(
  --seed 0
  --pin_index 0
  --distinct_tasks
  --get_input_attempts_per_task 5
)

for category in "${CATEGORIES[@]}"; do
  POOL="$POOL_ROOT/$category.json"
  if [[ ! -f "$POOL" ]]; then
    echo "Missing $POOL. Run screen_task_pools.py first." >&2
    exit 1
  fi
  calibration="${CALIBRATION_SAMPLES[$MODEL/$category]:?no calibration repeats set for $MODEL/$category}"

  for entry in "exp1/single_target_baseline:$calibration" "${TOPOLOGIES[@]}"; do
    graph="${entry%%:*}"
    samples="${entry##*:}"
    topology="$(basename "$graph")"
    echo "=== $category/$topology, $samples per target task $(date -Is) ==="
    python -B ../source/generate_dataset.py \
      --graph "topologies/$category/$graph.json" \
      --output_dir "$DATA_ROOT/$category/$topology" \
      --num_samples_per_task "$samples" \
      --pool_task_ids "$POOL" \
      "${COMMON[@]}"
    python -B dataset_to_text.py \
      --input "$DATA_ROOT/$category/$topology/${topology}_dataset.json"
  done

  for entry in "${SHUFFLED_LABEL_TOPOLOGIES[@]}"; do
    graph="${entry%%:*}"
    samples="${entry##*:}"
    topology="$(basename "$graph")_shuffled_labels"
    echo "=== $category/$topology, $samples per target task $(date -Is) ==="
    python -B ../source/generate_dataset.py \
      --graph "topologies/$category/$graph.json" \
      --output_dir "$DATA_ROOT/$category/$topology" \
      --num_samples_per_task "$samples" \
      --pool_task_ids "$POOL" \
      --shuffle_task_labels \
      "${COMMON[@]}"
    python -B dataset_to_text.py \
      --input "$DATA_ROOT/$category/$topology/${topology}_dataset.json"
  done
done

echo "All gap datasets finished at $(date -Is)"
