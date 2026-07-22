from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


TaskInput = dict[str, dict[str, Any]]
PromptBuilder = Callable[["CustomTask"], str]
Solver = Callable[[Mapping[str, Any]], Any]
Adapter = Callable[[Any], Any]
Verifier = Callable[[Any, Any], bool]


def _infer_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "scalar"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, dict):
        return "mapping"
    return type(value).__name__


class CustomTask:
    """Generic container for user-defined verifiable task nodes.

    A CustomTask is chainable when the caller supplies the same pieces that the
    built-in task classes provide: a prompt, an executable solver, declared
    schemas, and optional adapters. The class is intentionally domain-agnostic;
    it does not assume scalar/list state, although the current dataset
    generators may still choose to restrict themselves to those schemas.
    """

    def __init__(
        self,
        name: str,
        task_ind: int,
        inp: TaskInput,
        prompt_builder: str | PromptBuilder,
        solver: Solver,
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Any | None = None,
        input_adapters: Mapping[str, Adapter] | None = None,
        output_adapter: Adapter | None = None,
        verifier: Verifier | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("CustomTask name must be a non-empty string.")
        if not isinstance(task_ind, int) or isinstance(task_ind, bool) or task_ind < 1:
            raise ValueError("CustomTask task_ind must be a positive integer.")
        if not isinstance(inp, dict):
            raise TypeError("CustomTask inp must be a dictionary.")
        if not isinstance(prompt_builder, str) and not callable(prompt_builder):
            raise TypeError("CustomTask prompt_builder must be a string or callable.")
        if isinstance(prompt_builder, str) and not prompt_builder.strip():
            raise ValueError("CustomTask prompt template must be non-empty.")
        if not callable(solver):
            raise TypeError("CustomTask solver must be callable.")
        if input_adapters is not None:
            for key, adapter in input_adapters.items():
                if not callable(adapter):
                    raise TypeError(f"Input adapter for {key!r} must be callable.")
        if output_adapter is not None and not callable(output_adapter):
            raise TypeError("CustomTask output_adapter must be callable.")
        if verifier is not None and not callable(verifier):
            raise TypeError("CustomTask verifier must be callable.")

        self.name = name
        self.task_ind = task_ind
        self.inp = inp
        self.prompt_builder = prompt_builder
        self.solver = solver
        self.input_schema = {} if input_schema is None else dict(input_schema)
        self.output_schema = output_schema
        self.input_adapters = {} if input_adapters is None else dict(input_adapters)
        self.output_adapter = output_adapter
        self.verifier = verifier
        self.metadata = {} if metadata is None else dict(metadata)
        self.output_name = f"task_{task_ind:d}_out"

        self.adapted_inp = self.adapt_inputs()
        self.out = self.get_output()
        if self.out is None:
            raise ValueError(f"CustomTask {self.name!r} returned None.")
        self.prompt = self.gen_prompt()
        self.input_kind = self._infer_input_kind()
        self.output_kind = _infer_kind(self.out)

    def adapt_inputs(self) -> dict[str, Any]:
        adapted = {}
        for name, entry in self.inp.items():
            if not isinstance(entry, dict) or "value" not in entry:
                raise TypeError(f"Input {name!r} must be a dictionary with a 'value' field.")
            value = entry["value"]
            adapter = self.input_adapters.get(name)
            adapted[name] = adapter(value) if adapter is not None else value
        return adapted

    def get_output(self) -> Any:
        output = self.solver(self.adapted_inp)
        if self.output_adapter is not None:
            output = self.output_adapter(output)
        return output

    def gen_prompt(self) -> str:
        if callable(self.prompt_builder):
            prompt = self.prompt_builder(self)
        else:
            prompt = self.prompt_builder.format(
                task_ind=self.task_ind,
                output_name=self.output_name,
                inputs=self.adapted_inp,
                metadata=self.metadata,
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("CustomTask prompt builder produced an empty prompt.")
        return prompt

    def verify(self, candidate: Any) -> bool:
        if self.verifier is not None:
            return bool(self.verifier(candidate, self.out))
        return candidate == self.out

    def _infer_input_kind(self) -> str:
        if self.input_schema:
            return "schema"
        if not self.adapted_inp:
            return "none"
        kinds = {_infer_kind(value) for value in self.adapted_inp.values()}
        return kinds.pop() if len(kinds) == 1 else "mixed"
