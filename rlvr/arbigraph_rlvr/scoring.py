"""Score ArbiGraph tool submissions and text diagnostics."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

# The package __init__ puts source/ and eval/ on the path before this imports.
from grade_results import comparator, grade_value
from parse_results import answers_from_response

from .schema import ScoreResult, VerifierState

if TYPE_CHECKING:
    from .session import EpisodeSession


def score_completion(
    completion: str,
    verifier_state: VerifierState,
    submitted_answers: dict[str, object] | None = None,
) -> ScoreResult:
    """Score text diagnostics and an accepted target tool submission."""
    requested = tuple(verifier_state.expected_outputs)
    answers = dict(answers_from_response(completion, list(requested)))
    # A written target answer is not a submission. The evaluator grades this
    # node only from a successful submit_answers call.
    answers.pop(verifier_state.target_output, None)
    if submitted_answers is not None:
        answers.update(submitted_answers)
    missing = tuple(name for name in requested if name not in answers)

    correct_by_output = {
        name: (
            name in answers
            and comparator(verifier_state.comparators_by_output.get(name))(
                answers[name], verifier_state.expected_outputs[name]
            )
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


def terminal_reward(completion: str, session: "EpisodeSession") -> float:
    """Target-only terminal reward after the rollout's tool calls."""
    from .session import EpisodeSession

    if not isinstance(session, EpisodeSession):
        raise TypeError("terminal_reward requires an EpisodeSession, not a verifier state")
    return session.score(completion).reward


def trainer_reward_callback(
    completions: Sequence[str],
    sessions: Sequence["EpisodeSession"],
) -> list[float]:
    """Score a batch of completions for a trainer."""
    if len(completions) != len(sessions):
        raise ValueError("completions and sessions must have equal length")
    return [
        terminal_reward(completion, session)
        for completion, session in zip(completions, sessions)
    ]
