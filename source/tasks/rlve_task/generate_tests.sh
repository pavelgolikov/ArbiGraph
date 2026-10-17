#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

python -B ../../generate_dataset.py --graph ../../../generate/topologies/rlve/exp1/single_target_baseline.json --output_dir tests/single_target_baseline --num_samples_per_task 1 --seed 30
python -B ../../../generate/dataset_to_text.py --input tests/single_target_baseline/single_target_baseline_dataset.json
python -B ../../generate_dataset.py --graph ../../../generate/topologies/rlve/test/easy_chain.json --output_dir tests/easy_chain --num_samples_per_task 1 --seed 31
python -B ../../../generate/dataset_to_text.py --input tests/easy_chain/easy_chain_dataset.json
python -B ../../generate_dataset.py --graph ../../../generate/topologies/rlve/exp1/three_chain_target.json --output_dir tests/three_chain_target --num_samples_per_task 1 --seed 32
python -B ../../../generate/dataset_to_text.py --input tests/three_chain_target/three_chain_target_dataset.json
python -B ../../generate_dataset.py --graph ../../../generate/topologies/rlve/exp1/four_way_fan_in_target.json --output_dir tests/four_way_fan_in_target --num_samples_per_task 1 --seed 33
python -B ../../../generate/dataset_to_text.py --input tests/four_way_fan_in_target/four_way_fan_in_target_dataset.json
python -B ../../generate_dataset.py --graph ../../../generate/topologies/rlve/exp1/two_branch_recombine_target.json --output_dir tests/two_branch_recombine_target --num_samples_per_task 1 --seed 34
python -B ../../../generate/dataset_to_text.py --input tests/two_branch_recombine_target/two_branch_recombine_target_dataset.json
python -B ../../generate_dataset.py --graph ../../../generate/topologies/rlve/exp1/six_chain_target.json --output_dir tests/six_chain_target --num_samples_per_task 1 --seed 35
python -B ../../../generate/dataset_to_text.py --input tests/six_chain_target/six_chain_target_dataset.json
