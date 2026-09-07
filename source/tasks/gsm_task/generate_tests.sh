#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

python -B ../../generate_dataset.py --graph ../../topologies/gsm/test/easy_chain.json --output_dir tests/easy_chain --num_samples_per_task 1 --seed 0
python -B ../../dataset_to_text.py --input tests/easy_chain/easy_chain_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/gsm/test/easy_join.json --output_dir tests/easy_join --num_samples_per_task 1 --seed 1
python -B ../../dataset_to_text.py --input tests/easy_join/easy_join_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/gsm/test/medium_diamond.json --output_dir tests/medium_diamond --num_samples_per_task 1 --seed 2
python -B ../../dataset_to_text.py --input tests/medium_diamond/medium_diamond_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/gsm/test/medium_target_middle.json --output_dir tests/medium_target_middle --num_samples_per_task 1 --seed 3
python -B ../../dataset_to_text.py --input tests/medium_target_middle/medium_target_middle_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/gsm/test/complex_deep_recombine.json --output_dir tests/complex_deep_recombine --num_samples_per_task 1 --seed 4
python -B ../../dataset_to_text.py --input tests/complex_deep_recombine/complex_deep_recombine_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/gsm/test/complex_wide_recombine.json --output_dir tests/complex_wide_recombine --num_samples_per_task 1 --seed 5
python -B ../../dataset_to_text.py --input tests/complex_wide_recombine/complex_wide_recombine_dataset.json
