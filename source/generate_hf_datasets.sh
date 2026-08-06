#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

python -B generate_dataset.py --graph topologies/math/hf/single_target_baseline.json --output_dir generated/hf_datasets/math/single_target_baseline --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/math/single_target_baseline/single_target_baseline_dataset.json
python -B generate_dataset.py --graph topologies/math/hf/three_independent_target.json --output_dir generated/hf_datasets/math/three_independent_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/math/three_independent_target/three_independent_target_dataset.json
python -B generate_dataset.py --graph topologies/math/hf/three_chain_target.json --output_dir generated/hf_datasets/math/three_chain_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/math/three_chain_target/three_chain_target_dataset.json
python -B generate_dataset.py --graph topologies/math/hf/six_independent_target.json --output_dir generated/hf_datasets/math/six_independent_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/math/six_independent_target/six_independent_target_dataset.json
python -B generate_dataset.py --graph topologies/math/hf/six_chain_target.json --output_dir generated/hf_datasets/math/six_chain_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/math/six_chain_target/six_chain_target_dataset.json
python -B generate_dataset.py --graph topologies/math/hf/four_way_fan_in_target.json --output_dir generated/hf_datasets/math/four_way_fan_in_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/math/four_way_fan_in_target/four_way_fan_in_target_dataset.json
python -B generate_dataset.py --graph topologies/math/hf/two_branch_recombine_target.json --output_dir generated/hf_datasets/math/two_branch_recombine_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/math/two_branch_recombine_target/two_branch_recombine_target_dataset.json

python -B generate_dataset.py --graph topologies/python/hf/single_target_baseline.json --output_dir generated/hf_datasets/python/single_target_baseline --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/python/single_target_baseline/single_target_baseline_dataset.json
python -B generate_dataset.py --graph topologies/python/hf/three_independent_target.json --output_dir generated/hf_datasets/python/three_independent_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/python/three_independent_target/three_independent_target_dataset.json
python -B generate_dataset.py --graph topologies/python/hf/three_chain_target.json --output_dir generated/hf_datasets/python/three_chain_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/python/three_chain_target/three_chain_target_dataset.json
python -B generate_dataset.py --graph topologies/python/hf/six_independent_target.json --output_dir generated/hf_datasets/python/six_independent_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/python/six_independent_target/six_independent_target_dataset.json
python -B generate_dataset.py --graph topologies/python/hf/six_chain_target.json --output_dir generated/hf_datasets/python/six_chain_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/python/six_chain_target/six_chain_target_dataset.json
python -B generate_dataset.py --graph topologies/python/hf/four_way_fan_in_target.json --output_dir generated/hf_datasets/python/four_way_fan_in_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/python/four_way_fan_in_target/four_way_fan_in_target_dataset.json
python -B generate_dataset.py --graph topologies/python/hf/two_branch_recombine_target.json --output_dir generated/hf_datasets/python/two_branch_recombine_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/python/two_branch_recombine_target/two_branch_recombine_target_dataset.json

python -B generate_dataset.py --graph topologies/gsm/hf/single_target_baseline.json --output_dir generated/hf_datasets/gsm/single_target_baseline --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/gsm/single_target_baseline/single_target_baseline_dataset.json
python -B generate_dataset.py --graph topologies/gsm/hf/three_independent_target.json --output_dir generated/hf_datasets/gsm/three_independent_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/gsm/three_independent_target/three_independent_target_dataset.json
python -B generate_dataset.py --graph topologies/gsm/hf/three_chain_target.json --output_dir generated/hf_datasets/gsm/three_chain_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/gsm/three_chain_target/three_chain_target_dataset.json
python -B generate_dataset.py --graph topologies/gsm/hf/six_independent_target.json --output_dir generated/hf_datasets/gsm/six_independent_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/gsm/six_independent_target/six_independent_target_dataset.json
python -B generate_dataset.py --graph topologies/gsm/hf/six_chain_target.json --output_dir generated/hf_datasets/gsm/six_chain_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/gsm/six_chain_target/six_chain_target_dataset.json
python -B generate_dataset.py --graph topologies/gsm/hf/four_way_fan_in_target.json --output_dir generated/hf_datasets/gsm/four_way_fan_in_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/gsm/four_way_fan_in_target/four_way_fan_in_target_dataset.json
python -B generate_dataset.py --graph topologies/gsm/hf/two_branch_recombine_target.json --output_dir generated/hf_datasets/gsm/two_branch_recombine_target --num_samples_per_task 16
python -B dataset_to_text.py --input generated/hf_datasets/gsm/two_branch_recombine_target/two_branch_recombine_target_dataset.json
