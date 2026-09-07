"""Score one-shot ArbiGraph completions."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from grade_results import grade_value
from parse_results import answers_from_response

from .schema import ScoreResult, VerifierState


def score_completion(completion: str, verifier_state: VerifierState) -> ScoreResult:
    """Score one completion with target-only terminal reward."""
    requested = tuple(verifier_state.expected_outputs)
    answers = dict(answers_from_response(completion, list(requested)))
    missing = tuple(name for name in requested if name not in answers)

    correct_by_output = {
        name: (
            name in answers
            and grade_value(answers[name], verifier_state.expected_outputs[name])
        )
        for name in requested
    }
    answered_by_output = {name: name in answers for name in requested}
    node_correct = {
        verifier_state.node_ids_by_output[name]: correct_by_output[name]
        for name in requested
    }
    node_answered = {
        verifier_state.node_ids_by_output[name]: answered_by_output[name]
        for name in requested
    }
    first_incorrect_output = next(
        (name for name in requested if not correct_by_output[name]),
        None,
    )
    first_incorrect_node = (
        None
        if first_incorrect_output is None
        else verifier_state.node_ids_by_output[first_incorrect_output]
    )
    target_correct = correct_by_output[verifier_state.target_output]

    return ScoreResult(
        reward=1.0 if target_correct else 0.0,
        target_correct=target_correct,
        all_nodes_correct=all(correct_by_output.values()),
        parsed_answers=answers,
        node_correct=node_correct,
        node_answered=node_answered,
        first_incorrect_node=first_incorrect_node,
        format_valid=not missing,
        missing_outputs=missing,
    )


def terminal_reward(completion: str, verifier_state: VerifierState) -> float:
    """Pure target-only terminal reward callable for RL trainers."""
    return score_completion(completion, verifier_state).reward


def trainer_reward_callback(
    completions: Sequence[str],
    verifier_states: Sequence[VerifierState],
) -> list[float]:
    """Score a batch of completions for a trainer."""
    if len(completions) != len(verifier_states):
        raise ValueError("completions and verifier_states must have equal length")
    return [
        terminal_reward(completion, verifier_state)
        for completion, verifier_state in zip(completions, verifier_states)
    ]
