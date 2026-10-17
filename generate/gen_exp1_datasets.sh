#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

# Experiment 1 datasets: 8 instances per target template; one per selected RAG question.

# Comment out individual entries to exclude them from the sweep.
CATEGORIES=(
  math
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
  --seed 0
)

for category in "${CATEGORIES[@]}"; do
  GRAPH_ROOT="topologies/$category/exp1"
  DATA_ROOT="generated/exp1/$category"
  samples_per_task=8
  if [[ "$category" == "rag" ]]; then
    samples_per_task=1
  fi

  for topology in "${TOPOLOGIES[@]}"; do
    python -B ../source/generate_dataset.py \
      --graph "$GRAPH_ROOT/$topology.json" \
      --output_dir "$DATA_ROOT/$topology" \
      --num_samples_per_task "$samples_per_task" \
      "${COMMON[@]}"
    python -B dataset_to_text.py \
      --input "$DATA_ROOT/$topology/${topology}_dataset.json"
  done
done
