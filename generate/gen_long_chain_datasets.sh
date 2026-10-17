#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

CATEGORIES=(
  math
  python
  rlve
)

TOPOLOGIES=(
  eighteen_chain_target
)

COMMON=(
  --num_samples_per_task 8
  --seed 0
)

for category in "${CATEGORIES[@]}"; do
  DATA_ROOT="generated/long_chain/$category"

  for topology in "${TOPOLOGIES[@]}"; do
    python -B ../source/generate_dataset.py \
      --graph "topologies/$category/test/$topology.json" \
      --output_dir "$DATA_ROOT/$topology" \
      "${COMMON[@]}"
    python -B dataset_to_text.py \
      --input "$DATA_ROOT/$topology/${topology}_dataset.json"
  done
done
