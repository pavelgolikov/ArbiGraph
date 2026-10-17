"""Trusted, per-rollout execution of the current ArbiGraph agent tools."""

from copy import deepcopy
from typing import Any

from tasks.arbigraph_tools import InputTool
from util import call_get_input_tool, validate_submission

from .schema import ScoreResult, VerifierState
from .scoring import score_completion


class EpisodeSession:
    """Serve inputs and accept one target answer without exposing verifier state."""

    def __init__(self, verifier_state: VerifierState) -> None:
        if not verifier_state.input_records:
            raise ValueError("Current-format node input records are required for tool sessions")
        self._verifier_state = verifier_state
        self._input_tool = InputTool.from_records(
            deepcopy(verifier_state.input_records),
            attempts_per_task=verifier_state.get_input_attempts_per_task,
        )
        self.submitted_answers: dict[str, object] | None = None

    @property
    def done(self) -> bool:
        return self.submitted_answers is not None

    def call_tool(self, name: str, arguments: Any) -> str:
        """Execute one model tool call and return its policy-visible response."""
        return self.call_tools([{"name": name, "arguments": arguments}])[0]

    def call_tools(self, calls: list[dict[str, Any]]) -> list[str]:
        """Execute a model turn; submissions see state before this turn's calls."""
        if self.done:
            raise RuntimeError("The episode has ended after submit_answers")
        verdicts = {
            index: validate_submission(
                call.get("arguments"), [self._verifier_state.target_output], self._input_tool
            )
            for index, call in enumerate(calls)
            if call.get("name") == "submit_answers"
        }
        accepted = [answer for answer in verdicts.values() if isinstance(answer, dict)]
        if accepted:
            self.submitted_answers = accepted[-1]
            responses = []
            for index in range(len(calls)):
                verdict = verdicts.get(index)
                responses.append(
                    "Submission accepted." if isinstance(verdict, dict)
                    else verdict if verdict is not None else "Episode ended."
                )
            return responses
        responses = []
        for index, call in enumerate(calls):
            name = call.get("name")
            if name == "get_input":
                responses.append(call_get_input_tool(self._input_tool, call.get("arguments")))
            elif name == "submit_answers":
                responses.append(verdicts[index])
            else:
                responses.append(f"Error: unknown tool {name!r}.")
        return responses

    def score(self, completion: str = "") -> ScoreResult:
        """Reward only the accepted target submission; text supplies diagnostics."""
        return score_completion(completion, self._verifier_state, self.submitted_answers)
