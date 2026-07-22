import os
from typing import Any, TypeAlias


NumericOutput: TypeAlias = int | float | list[int] | list[float]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_output(value: Any) -> NumericOutput:
    if _is_number(value):
        return value
    if isinstance(value, list):
        for item in value:
            if not _is_number(item):
                raise TypeError("ImageTask list outputs must contain only int or float values.")
        return value
    raise TypeError("ImageTask output must be an int, float, list[int], or list[float].")


def _infer_value_kind(value: Any) -> str:
    if _is_number(value):
        return "scalar"
    if isinstance(value, list):
        return "list"
    return "unknown"


def _infer_input_kind(inp: dict[str, Any]) -> str:
    values = []
    for entry in inp.values():
        if isinstance(entry, dict) and "value" in entry and entry["value"] is not None:
            values.append(entry["value"])

    if not values:
        return "unknown"

    kinds = {_infer_value_kind(value) for value in values}
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed"


class ImageTask:
    """Base container for image-grounded benchmark tasks.

    ImageTask intentionally does not define how an image is generated or how an
    input parameter selects visual content. Concrete tasks should compute their
    numeric output and final prompt first, then store task-specific structure in
    metadata.
    """

    def __init__(
        self,
        name: str,
        task_ind: int,
        inp: dict[str, Any],
        scalar_max_mag: int,
        list_len_max: int,
        image: str,
        prompt: str,
        out: NumericOutput,
        metadata: dict[str, Any] | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("ImageTask name must be a non-empty string.")
        if not isinstance(task_ind, int) or isinstance(task_ind, bool) or task_ind < 1:
            raise ValueError("ImageTask task_ind must be a positive integer.")
        if not isinstance(inp, dict):
            raise TypeError("ImageTask inp must be a dictionary.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("ImageTask prompt must be a non-empty string.")
        if not isinstance(image, str) or not image:
            raise ValueError("ImageTask image must be a non-empty local file path.")
        if not os.path.isfile(image):
            raise FileNotFoundError(f"ImageTask image does not exist: {image}")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("ImageTask metadata must be a dictionary when provided.")

        self.name = name
        self.task_ind = task_ind
        self.inp = inp
        self.scalar_max_mag = scalar_max_mag
        self.list_len_max = list_len_max
        self.image = image
        self.prompt = prompt
        self.out = _validate_output(out)
        self.metadata = {} if metadata is None else dict(metadata)
        self.input_kind = _infer_input_kind(inp)
        self.output_kind = _infer_value_kind(out)
