"""RLVR interface for ArbiGraph."""

from pathlib import Path
import sys

# The task graph lives in source/ and the answer parser and grader in eval/;
# this package sits beside them at the repository root rather than inside either.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO_ROOT / "source"), str(_REPO_ROOT / "eval")]

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
    comparator,
    grade_value,
    score_completion,
    terminal_reward,
    trainer_reward_callback,
)
from .session import EpisodeSession

__all__ = [
    "ArbiGraphEnv",
    "Episode",
    "EpisodeObservation",
    "EpisodeSpec",
    "EpisodeSession",
    "DatasetLoader",
    "ScoreResult",
    "VerifierState",
    "comparator",
    "grade_value",
    "score_completion",
    "terminal_reward",
    "trainer_reward_callback",
]
