import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from tasks.image_task.traversal_image_generator import (
    DEFAULT_NUM_PATHS,
    build_traversal_task_payload,
    generate_traversal_metadata,
    render_traversal_image,
    save_json,
)


HEADER = (
    "Solve the task below. Put the answer in its own \\boxed{} block with the "
    "exact requested output name, for example \\boxed{task_3_out = ...}."
)


def _repo_relative(path: Path) -> str:
    return os.path.relpath(path, Path.cwd())


def scalar_for_start_index(image_index: int, start_index: int, num_paths: int) -> int:
    value = start_index + num_paths * (image_index + 1)
    if (image_index + start_index) % 2:
        value = -value
    return value


def build_full_prompt(input_name: str, input_value: int, task_prompt: str) -> str:
    return (
        f"{HEADER}\n\n"
        f"Initial Conditions:\n"
        f"{input_name} = {input_value}\n\n"
        f"{task_prompt}"
    )


def make_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((360, 252), Image.Resampling.LANCZOS)
        thumbs.append((image_path, image.copy()))

    columns = 5
    rows = (len(thumbs) + columns - 1) // columns
    cell_w, cell_h = 420, 310
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image_path, image) in enumerate(thumbs):
        x0 = (index % columns) * cell_w
        y0 = (index // columns) * cell_h
        x = x0 + (cell_w - image.width) // 2
        y = y0 + 30
        sheet.paste(image, (x, y))
        draw.text((x0 + 16, y0 + 10), image_path.stem, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def generate_dataset(
    output_dir: str,
    num_images: int,
    seed: int,
    num_paths: int,
    task_ind: int,
    input_name: str,
) -> dict[str, Any]:
    root = Path(output_dir)
    image_dir = root / "images"
    metadata_dir = root / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    image_paths = []
    for image_index in range(num_images):
        image_seed = seed + image_index
        metadata = generate_traversal_metadata(seed=image_seed, num_paths=num_paths)
        image_path = image_dir / f"traversal_image_{image_index:03d}_seed{image_seed}.png"
        metadata_path = metadata_dir / f"traversal_image_{image_index:03d}_seed{image_seed}.json"
        render_traversal_image(metadata, str(image_path))
        save_json(metadata, str(metadata_path))
        image_paths.append(image_path)

        for start_index in range(num_paths):
            input_value = scalar_for_start_index(image_index, start_index, num_paths)
            payload = build_traversal_task_payload(
                metadata,
                task_ind=task_ind,
                input_name=input_name,
                input_value=input_value,
            )
            sample_idx = len(samples)
            samples.append(
                {
                    "sample_idx": sample_idx,
                    "task_name": "traversal_image",
                    "image": _repo_relative(image_path),
                    "image_metadata": _repo_relative(metadata_path),
                    "prompt": build_full_prompt(input_name, input_value, payload["prompt"]),
                    "ground_truth": payload["out"],
                    "input_name": input_name,
                    "input_value": input_value,
                    "task_ind": task_ind,
                    "output_name": payload["output_name"],
                    "image_index": image_index,
                    "image_seed": image_seed,
                    "start_index": start_index,
                    "selected_start_index": payload["selected_start_index"],
                    "selected_start_number": payload["selected_start_number"],
                    "selected_path_id": payload["selected_path_id"],
                }
            )

    contact_sheet_path = root / "contact_sheet.png"
    make_contact_sheet(image_paths, contact_sheet_path)

    dataset = {
        "summary": {
            "mode": "traversal_image_baseline",
            "num_images": num_images,
            "num_paths": num_paths,
            "samples_per_image": num_paths,
            "num_samples": len(samples),
            "seed": seed,
            "task_ind": task_ind,
            "input_name": input_name,
            "image_dir": _repo_relative(image_dir),
            "metadata_dir": _repo_relative(metadata_dir),
            "contact_sheet": _repo_relative(contact_sheet_path),
        },
        "samples": samples,
    }
    dataset_path = root / f"traversal_baseline_{num_images}x{num_paths}.json"
    save_json(dataset, str(dataset_path))
    return {
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "image_dir": str(image_dir),
        "contact_sheet": str(contact_sheet_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate traversal-image baseline dataset.")
    parser.add_argument("--output-dir", default="scratch/traversal_baseline_40x8")
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)
    parser.add_argument("--task-ind", type=int, default=3)
    parser.add_argument("--input-name", default="task_2_out")
    args = parser.parse_args()

    result = generate_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
        num_paths=args.num_paths,
        task_ind=args.task_ind,
        input_name=args.input_name,
    )
    dataset = result["dataset"]
    print(f"Generated {dataset['summary']['num_samples']} samples")
    print(f"Dataset: {result['dataset_path']}")
    print(f"Images: {result['image_dir']}")
    print(f"Contact sheet: {result['contact_sheet']}")


if __name__ == "__main__":
    main()
