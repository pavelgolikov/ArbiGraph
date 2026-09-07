#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

python -B ../../generate_dataset.py --graph ../../topologies/python/test/easy_chain_four.json --output_dir tests/easy_chain_four --num_samples_per_task 1 --seed 10
python -B ../../dataset_to_text.py --input tests/easy_chain_four/easy_chain_four_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/python/test/easy_offset_join.json --output_dir tests/easy_offset_join --num_samples_per_task 1 --seed 11
python -B ../../dataset_to_text.py --input tests/easy_offset_join/easy_offset_join_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/python/test/medium_split_join.json --output_dir tests/medium_split_join --num_samples_per_task 1 --seed 12
python -B ../../dataset_to_text.py --input tests/medium_split_join/medium_split_join_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/python/test/medium_middle_branch.json --output_dir tests/medium_middle_branch --num_samples_per_task 1 --seed 13
python -B ../../dataset_to_text.py --input tests/medium_middle_branch/medium_middle_branch_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/python/test/complex_deep_ladder.json --output_dir tests/complex_deep_ladder --num_samples_per_task 1 --seed 14
python -B ../../dataset_to_text.py --input tests/complex_deep_ladder/complex_deep_ladder_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/python/test/complex_wide_layers.json --output_dir tests/complex_wide_layers --num_samples_per_task 1 --seed 15
python -B ../../dataset_to_text.py --input tests/complex_wide_layers/complex_wide_layers_dataset.json
