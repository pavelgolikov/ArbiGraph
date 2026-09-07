"""Generate and score ArbiGraph episodes on demand."""

from copy import deepcopy

from fill_dag import fill_dag, write_prompt
from parse_dag import parse_custom_dag

from .schema import (
    Episode,
    EpisodeObservation,
    EpisodeSpec,
    ScoreResult,
    VerifierState,
)
from .scoring import score_completion


class ArbiGraphEnv:
    """Generate an episode from a topology and score its completion."""

    def reset(self, spec: EpisodeSpec) -> Episode:
        """Generate one episode from an explicit specification."""
        self._validate_spec(spec)

        topology = deepcopy(dict(spec.topology))
        topology["target"] = deepcopy(topology["target"])
        target_task_id = topology["target"]["native_task_id"]
        if isinstance(target_task_id, bool) or target_task_id == "all":
            raise ValueError("topology must select one concrete target task")
        try:
            target_task_id = int(target_task_id)
        except (TypeError, ValueError) as error:
            raise ValueError("topology target task must be an integer") from error
        topology["target"]["native_task_id"] = str(target_task_id)
        filled = fill_dag(
            parse_custom_dag(topology),
            seed=spec.seed,
            sample_idx=spec.sample_idx,
        )

        nodes = filled["nodes"]
        expected_outputs = {
            node["output_name"]: node["output_value"] for node in nodes
        }
        node_ids_by_output = {
            node["output_name"]: str(node["node_id"]) for node in nodes
        }
        target_output = filled["target_output_name"]
        episode_id = (
            f"{spec.topology_id}/task-{target_task_id}/"
            f"sample-{spec.sample_idx}/seed-{spec.seed}"
        )

        observation = EpisodeObservation(
            episode_id=episode_id,
            prompt=write_prompt(filled),
            requested_outputs=tuple(expected_outputs),
            target_output=target_output,
        )
        verifier_state = VerifierState(
            expected_outputs=expected_outputs,
            target_output=target_output,
            node_ids_by_output=node_ids_by_output,
        )
        return Episode(observation, verifier_state)

    def score(
        self,
        completion: str,
        verifier_state: VerifierState,
    ) -> ScoreResult:
        """Score a completion against trusted state returned by reset()."""
        return score_completion(completion, verifier_state)

    @staticmethod
    def _validate_spec(spec: EpisodeSpec) -> None:
        if not isinstance(spec, EpisodeSpec):
            raise TypeError("spec must be an EpisodeSpec")
        if not spec.topology_id:
            raise ValueError("topology_id must be non-empty")
        if isinstance(spec.seed, bool) or not isinstance(spec.seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(spec.sample_idx, bool) or not isinstance(spec.sample_idx, int):
            raise TypeError("sample_idx must be an integer")
        if spec.sample_idx < 0:
            raise ValueError("sample_idx must be non-negative")
        if not isinstance(spec.topology, dict):
            raise TypeError("topology must be a dictionary")
