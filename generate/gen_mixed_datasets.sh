#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

# One shared dataset for every local model. The target sweeps all 40 RLVE tasks,
# with four independently generated graph samples per target task (160 rows).
topology=six_mixed_recombine_target
output_dir="generated/mixed/$topology"

python -B ../source/generate_dataset.py \
  --graph "topologies/mixed/$topology.json" \
  --output_dir "$output_dir" \
  --num_samples_per_task 4 \
  --seed 0 \
  --get_input_attempts_per_task 5

python -B dataset_to_text.py \
  --input "$output_dir/${topology}_dataset.json"
