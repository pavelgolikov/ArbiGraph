"""Data passed between episode generation, policies, and scoring."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from tasks.arbigraph_tools import GET_INPUT_TOOL, SUBMIT_ANSWERS_TOOL


@dataclass
class EpisodeSpec:
    """Inputs that uniquely determine one generated episode."""

    topology_id: str
    topology: dict[str, Any]
    seed: int  # Randomness used while selecting and instantiating the surrounding graph.
    sample_idx: int  # Stable example number within the selected target native task.
    get_input_attempts_per_task: int = 5
    pool_task_ids: list[int] | None = None
    pin_index: int | None = None
    distinct_tasks: bool = False
    shuffle_task_labels: bool = False


@dataclass
class EpisodeObservation:
    """The complete policy-visible view of one episode."""

    episode_id: str
    prompt: str
    requested_outputs: tuple[str, ...]
    target_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible view with no verifier state."""
        record = {
            "episode_id": self.episode_id,
            "prompt": self.prompt,
            "requested_outputs": list(self.requested_outputs),
            "target_output": self.target_output,
            "tools": [deepcopy(GET_INPUT_TOOL), deepcopy(SUBMIT_ANSWERS_TOOL)],
        }
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return record


@dataclass
class VerifierState:
    """Private answers and inputs for one episode's tool sessions and scoring."""

    expected_outputs: dict[str, object] = field(repr=False)
    target_output: str
    node_ids_by_output: dict[str, str]
    # Comparator name per output; absent entries fall back to the numeric default.
    comparators_by_output: dict[str, str] = field(default_factory=dict)
    input_records: tuple[dict[str, Any], ...] = field(default_factory=tuple, repr=False)
    get_input_attempts_per_task: int = 5


@dataclass
class Episode:
    """Separated public observation and trusted verifier state."""

    observation: EpisodeObservation
    verifier_state: VerifierState = field(repr=False)

    def to_public_dict(self) -> dict[str, Any]:
        """Return only fields that are safe to serialize for a policy worker."""
        return self.observation.to_dict()

    def start(self) -> "EpisodeSession":
        """Start an independent tool session for one policy rollout."""
        from .session import EpisodeSession

        return EpisodeSession(self.verifier_state)


@dataclass
class ScoreResult:
    """Exact terminal reward plus per-node diagnostics."""

    reward: float
    target_correct: bool
    all_nodes_correct: bool
    parsed_answers: dict[str, object]
    node_correct: dict[str, bool]
    node_answered: dict[str, bool]
    first_incorrect_node: str | None
    format_valid: bool
    missing_outputs: tuple[str, ...]
