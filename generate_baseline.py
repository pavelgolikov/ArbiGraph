import argparse
import json
import os
import sys
import datetime
import re
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def save_json(data, path):
    json_str = json.dumps(data, indent=2)
    # Collapse arrays of simple types (like numbers/strings) horizontally
    json_str = re.sub(r'\[\s+([^\[\]\{\}]*?)\s+\]', lambda m: '[' + re.sub(r'\s+', ' ', m.group(1)) + ']', json_str)
    json_str = re.sub(r'\[\s+\]', '[]', json_str)
    with open(path, "w") as f:
        f.write(json_str)

import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    # Guard generated-chain verification from hanging tasks.
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

from tasks.math_task.math_task import MATH_TASK_SPECS, _build_math_input
from tasks.python_task.python_task import PythonTraceTask, load_candidate_algos, generate_random_value
from tasks.math_task.math_helpers import adapt_list_prompt, adapt_scalar_prompt
from tasks.gsm_task.gsm_symbolic_task import GSMSymbolicTask, load_templates, get_default_templates_dir

DIFFICULTIES = {
    "small": {"list_len_max": 10, "scalar_max_mag": 100},
    # "medium": {"list_len_max": 30, "scalar_max_mag": 1000},
    # "large": {"list_len_max": 100, "scalar_max_mag": 10000},
}

def has_large_digits(val):
    if isinstance(val, list):
        return any(has_large_digits(v) for v in val)
    if isinstance(val, (int, float)):
        try:
            return abs(val) >= 10**10
        except OverflowError:
            return True
    return False

def has_too_many_duplicates(val):
    if isinstance(val, list) and val:
        import collections
        counts = collections.Counter(val)
        return counts.most_common(1)[0][1] > len(val) / 2
    return False

def is_sequential(val):
    if isinstance(val, list) and val:
        if val == list(range(len(val))):
            return True
        if val == list(range(1, len(val) + 1)):
            return True
    return False

def generate_rand_val(type_hint, list_len_max, scalar_max_mag):
    return generate_random_value(
        type_hint,
        list_length=list_len_max,
        val_min=-scalar_max_mag,
        val_max=scalar_max_mag
    )

def convert_to_std(val):
    def _conv(x):
        t = type(x).__module__
        if t and t.startswith('sympy'):
            if getattr(x, 'is_integer', False) is True:
                return int(x)
            if getattr(x, 'is_real', False) is True:
                return float(x)
        return x
    if isinstance(val, list):
        return [_conv(v) for v in val]
    return _conv(val)

def generate_math_dataset(num_samples_per_task, num_tasks):
    sorted_specs = sorted(MATH_TASK_SPECS, key=lambda s: (s.input_kind, s.output_kind, s.cls.__name__))
    dataset = []
    
    for diff_name, bounds in DIFFICULTIES.items():
        list_len_max = bounds["list_len_max"]
        scalar_max_mag = bounds["scalar_max_mag"]
        
        for task_id, spec in enumerate(tqdm(sorted_specs, desc=f"Math Tasks ({diff_name})")):
            for sample_idx in range(num_samples_per_task):
                while True:
                    chained_value = generate_rand_val(spec.input_kind, list_len_max, scalar_max_mag)
                    inp = _build_math_input(spec.adapter, "task_1_input", chained_value, list_len_max)
                    try:
                        with time_limit(2):
                            task = spec.cls(1, inp, scalar_max_mag, list_len_max)
                            cand_out = convert_to_std(task.out)
                        if isinstance(cand_out, list) and (not cand_out or len(cand_out) == 1):
                            continue
                        if isinstance(cand_out, (int, float)) and cand_out in [0, 1, -1]:
                            continue
                        if cand_out == chained_value:
                            continue
                        if has_large_digits(cand_out):
                            continue
                        if has_too_many_duplicates(cand_out):
                            continue
                        if is_sequential(cand_out):
                            continue
                            
                        prompt = f"Define task_1_input = {chained_value}.\n" + task.prompt
                        dataset.append({
                            "difficulty": f"list_max_{list_len_max}_scalar_max_{scalar_max_mag}",
                            "task_id": task_id,
                            "task_name": spec.cls.__name__,
                            "sample_idx": sample_idx,
                            "chain_input": chained_value,
                            "prompt": prompt,
                            "ground_truth": cand_out
                        })
                        break
                    except BaseException as e:
                        pass
    return dataset

def generate_python_dataset(num_samples_per_task, num_tasks):
    algos_file = os.path.join(REPO_ROOT, "tasks", "python_task", "leetcode_candidate_algos.txt")
    all_funcs = load_candidate_algos(algos_file)
    all_funcs = all_funcs[:num_tasks] if num_tasks > 0 else all_funcs

    dataset = []
    
    for diff_name, bounds in DIFFICULTIES.items():
        list_len_max = bounds["list_len_max"]
        scalar_max_mag = bounds["scalar_max_mag"]
        
        for func_index, func_info in enumerate(tqdm(all_funcs, desc=f"Python Tasks ({diff_name})")):
            input_kind = "list" if func_info['has_list_in'] else "scalar"
            for sample_idx in range(num_samples_per_task):
                while True:
                    chained_value = generate_rand_val(input_kind, list_len_max, scalar_max_mag)
                    try:
                        with time_limit(2):
                            task = PythonTraceTask(
                                task_ind=1,
                                inp={"chained": {"chain_name": "task_1_input", "value": chained_value}},
                                scalar_max_mag=scalar_max_mag,
                                list_len_max=list_len_max,
                                rand_static_val=lambda t: generate_rand_val(t, list_len_max, scalar_max_mag),
                                algos_file=algos_file,
                                func_index=func_index
                            )
                        if isinstance(task.out, list) and (not task.out or len(task.out) == 1):
                            continue
                        if task.out == chained_value:
                            continue
                        if has_large_digits(task.out):
                            continue
                        if has_too_many_duplicates(task.out):
                            continue
                        if is_sequential(task.out):
                            continue
                            
                        prompt = f"Define task_1_input = {chained_value}.\n"
                        for pname, pval in task.static_values.items():
                            prompt += f"Define task_1_{pname}_static = {pval}.\n"
                        prompt += task.prompt
                        dataset.append({
                            "difficulty": f"list_max_{list_len_max}_scalar_max_{scalar_max_mag}",
                            "task_id": func_index,
                            "task_name": func_info['name'],
                            "sample_idx": sample_idx,
                            "chain_input": chained_value,
                            "extra_inputs": task.static_values,
                            "prompt": prompt,
                            "ground_truth": task.out
                        })
                        break
                    except BaseException:
                        pass
    return dataset

TARGET_TEMPLATES = [
    "0000.json", "0003.json", "0004.json", "0007.json", "0010.json", "0015.json",
    "0016.json", "0018.json", "0019.json", "0024.json", "0027.json", "0028.json",
    "0032.json", "0034.json", "0036.json", "0039.json", "0042.json", "0047.json",
    "0048.json", "0050.json", "0051.json", "0054.json", "0058.json", "0059.json",
    "0060.json", "0062.json", "0066.json", "0067.json", "0071.json", "0072.json",
    "0075.json", "0078.json", "0082.json", "0084.json", "0086.json",
    "0088.json", "0093.json", "0094.json", "0095.json",
    "0096.json", "0099.json"
]

def generate_gsm_dataset(num_samples_per_task, num_tasks):
    templates = load_templates(get_default_templates_dir())
    
    # Map the targeted templates to their original index in the loaded list
    target_indices = []
    for i, tpl in enumerate(templates):
        if tpl["_filename"] in TARGET_TEMPLATES:
            target_indices.append(i)
            
    if num_tasks > 0:
        target_indices = target_indices[:num_tasks]
    
    dataset = []
    
    for diff_name, bounds in DIFFICULTIES.items():
        list_len_max = bounds["list_len_max"]
        scalar_max_mag = bounds["scalar_max_mag"]
        
        for idx_in_targets, orig_idx in enumerate(tqdm(target_indices, desc=f"GSM Tasks ({diff_name})")):
            tpl = templates[orig_idx]
            for sample_idx in range(num_samples_per_task):
                while True:
                    chained_value = generate_rand_val("scalar", list_len_max, scalar_max_mag)
                    try:
                        with time_limit(2):
                            task = GSMSymbolicTask(
                                task_ind=1,
                                inp={"task_1_input": {"chain_name": "task_1_input", "value": chained_value}},
                                scalar_max_mag=scalar_max_mag,
                                list_len_max=list_len_max,
                                template_index=orig_idx
                            )
                            cand_out = convert_to_std(task.out)
                        
                        prompt = f"Define task_1_input = {chained_value}.\n" + task.prompt
                        dataset.append({
                            "difficulty": f"list_max_{list_len_max}_scalar_max_{scalar_max_mag}",
                            "task_id": idx_in_targets,
                            "task_name": f"GSM_{tpl['_filename']}",
                            "sample_idx": sample_idx,
                            "chain_input": chained_value,
                            "prompt": prompt,
                            "ground_truth": cand_out
                        })
                        break
                    except BaseException:
                        pass
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["math", "python", "gsm"], required=True)
    parser.add_argument("--num_samples_per_task", type=int, default=16, help="Number of samples to generate per task")
    parser.add_argument("--num_tasks", type=int, default=0, help="Number of tasks to evaluate")
    parser.add_argument("--output", type=str, default="", help="Output dataset json file")

    args = parser.parse_args()

    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output if args.output else os.path.join(REPO_ROOT, f"dataset_{args.mode}_{dt_str}.json")
    
    print(f"Generating {args.mode} dataset...")
    # dataset = generate_math_dataset(args.num_samples_per_task, args.num_tasks) if args.mode == "math" else generate_python_dataset(args.num_samples_per_task, args.num_tasks)
    dataset = None
    if args.mode == "math":
        dataset = generate_math_dataset(args.num_samples_per_task, args.num_tasks)
    elif args.mode == "python":
        dataset = generate_python_dataset(args.num_samples_per_task, args.num_tasks)
    elif args.mode == "gsm":
        dataset = generate_gsm_dataset(args.num_samples_per_task, args.num_tasks)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    data = {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": args.mode,
            "num_tasks": len(dataset) // args.num_samples_per_task,
            "num_samples_per_task": args.num_samples_per_task
        },
        "samples": dataset
    }
    save_json(data, json_path)
    print(f"Dataset generated and saved to {json_path}")

if __name__ == "__main__":
    main()
