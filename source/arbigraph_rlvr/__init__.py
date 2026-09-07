"""RLVR interface for ArbiGraph."""

from .dataset import DatasetLoader
from .environment import ArbiGraphEnv
from .schema import (
    Episode,
    EpisodeObservation,
    EpisodeSpec,
    ScoreResult,
    VerifierState,
)
from .scoring import (
    grade_value,
    score_completion,
    terminal_reward,
    trainer_reward_callback,
)

__all__ = [
    "ArbiGraphEnv",
    "Episode",
    "EpisodeObservation",
    "EpisodeSpec",
    "DatasetLoader",
    "ScoreResult",
    "VerifierState",
    "grade_value",
    "score_completion",
    "terminal_reward",
    "trainer_reward_callback",
]
