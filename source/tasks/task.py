"""Minimal abstract task interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Task(ABC):
    category: str
    task_id: int
    task_name: str
    input_type: str
    output_type: str

    # Describe the task in English using the named input and output variables.
    @abstractmethod
    def prompt_generator(self, task_number: int, input_names: list[str], output_name: str) -> str:
        raise NotImplementedError

    # Compute the concrete output for one concrete input instance.
    @abstractmethod
    def solution_generator(self, input_value: Any) -> Any:
        raise NotImplementedError
