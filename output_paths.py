"""Small helpers for timestamped result filenames."""

import datetime
import os
import re


def filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    component = component.strip("._-")
    if not component:
        raise ValueError(f"Invalid empty filename component from {value!r}")
    return component


def timestamped_output_path(output_dir: str, input_path: str, model: str, prefix: str) -> str:
    input_stem = filename_component(os.path.splitext(os.path.basename(input_path))[0])
    model_stem = filename_component(model.split("/")[-1])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{input_stem}_{model_stem}_{timestamp}.json"
    return os.path.join(output_dir, filename)
