#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

python -B ../../generate_dataset.py --graph ../../topologies/math_task/easy_chain.json --output_dir tests/easy_chain --num_samples_per_task 1 --seed 0
python -B ../../dataset_to_text.py --input tests/easy_chain/easy_chain_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/easy_join.json --output_dir tests/easy_join --num_samples_per_task 1 --seed 1
python -B ../../dataset_to_text.py --input tests/easy_join/easy_join_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/medium_diamond.json --output_dir tests/medium_diamond --num_samples_per_task 1 --seed 2
python -B ../../dataset_to_text.py --input tests/medium_diamond/medium_diamond_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/medium_target_middle.json --output_dir tests/medium_target_middle --num_samples_per_task 1 --seed 3
python -B ../../dataset_to_text.py --input tests/medium_target_middle/medium_target_middle_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/complex_deep_recombine.json --output_dir tests/complex_deep_recombine --num_samples_per_task 1 --seed 4
python -B ../../dataset_to_text.py --input tests/complex_deep_recombine/complex_deep_recombine_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/complex_wide_recombine.json --output_dir tests/complex_wide_recombine --num_samples_per_task 1 --seed 5
python -B ../../dataset_to_text.py --input tests/complex_wide_recombine/complex_wide_recombine_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/target_join_two_parent_all_input_graph.json --output_dir tests/target_join_two_parent_all --num_samples_per_task 1 --seed 20
python -B ../../dataset_to_text.py --input tests/target_join_two_parent_all/target_join_two_parent_all_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/target_join_three_parent_all_input_graph.json --output_dir tests/target_join_three_parent_all --num_samples_per_task 1 --seed 21
python -B ../../dataset_to_text.py --input tests/target_join_three_parent_all/target_join_three_parent_all_dataset.json
python -B ../../generate_dataset.py --graph ../../topologies/math_task/target_join_diamond_all_input_graph.json --output_dir tests/target_join_diamond_all --num_samples_per_task 1 --seed 22
python -B ../../dataset_to_text.py --input tests/target_join_diamond_all/target_join_diamond_all_dataset.json
