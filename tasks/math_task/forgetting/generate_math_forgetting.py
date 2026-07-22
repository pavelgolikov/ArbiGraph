import argparse
import os
import sys


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TASK_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.forgetting_dataset import generate_forgetting_dataset, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--input", default=os.path.join(REPO_ROOT, "math_baseline.json"), help="Input baseline dataset JSON file",)
    parser.add_argument( "--output", default=os.path.join(TASK_DIR, "math_forgetting.json"), help="Output forgetting dataset JSON file",)
    parser.add_argument("--num_distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = generate_forgetting_dataset(
        input_path=args.input,
        mode="math",
        num_distractors=args.num_distractors,
        seed=args.seed,
        repo_root=REPO_ROOT,
    )
    save_json(data, args.output)
    print(f"Generated math forgetting dataset with {len(data['samples'])} samples at {args.output}")


if __name__ == "__main__":
    main()
