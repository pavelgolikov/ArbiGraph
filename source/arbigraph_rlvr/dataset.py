"""Load existing ArbiGraph dataset records for RLVR scoring."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import Episode, EpisodeObservation, ScoreResult, VerifierState
from .scoring import score_completion


REQUIRED_SAMPLE_FIELDS = (
    "target_native_task_id",
    "target_node_id",
    "sample_idx",
    "prompt",
    "node_outputs",
)


class DatasetLoader:
    """Load episodes from one dataset file or in-memory dataset."""

    def __init__(
        self,
        dataset: str | Path | dict[str, Any],
        topology_id: str,
        condition: str,
        category: str | None = None,
    ) -> None:
        if not topology_id:
            raise ValueError("topology_id must be non-empty")
        if not condition:
            raise ValueError("condition must be non-empty")
        self.topology_id = topology_id
        self.condition = condition
        self.category = category
        self.dataset = (
            dict(dataset) if isinstance(dataset, dict) else self._read_json(dataset)
        )

    @staticmethod
    def _read_json(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}.")
        return data

    @staticmethod
    def _validate_sample(sample: dict[str, Any]) -> None:
        missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"Dataset sample is missing required fields: {missing}.")
        if not isinstance(sample["prompt"], str):
            raise ValueError("Dataset sample prompt must be a string.")
        node_outputs = sample["node_outputs"]
        if not isinstance(node_outputs, list) or not node_outputs:
            raise ValueError("Dataset sample node_outputs must be a non-empty list.")
        node_ids = []
        for record in node_outputs:
            if not isinstance(record, dict) or not {"node_id", "output_value"} <= set(record):
                raise ValueError("Node output is missing node_id or output_value.")
            node_ids.append(record["node_id"])
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Dataset sample node IDs must be unique.")

    def load_sample(
        self,
        sample: dict[str, Any],
        category: str | None = None,
    ) -> Episode:
        """Convert one sample into separate policy and verifier views."""
        self._validate_sample(sample)
        resolved_category = category or self.category
        if not resolved_category:
            raise ValueError(
                "category is required when loading a sample outside a dataset summary"
            )

        target_task_id = int(sample["target_native_task_id"])
        sample_index = int(sample["sample_idx"])
        node_outputs = sample["node_outputs"]
        expected_outputs = {
            f"task_{position}_out": record["output_value"]
            for position, record in enumerate(node_outputs, start=1)
        }
        node_ids_by_output = {
            f"task_{position}_out": str(record["node_id"])
            for position, record in enumerate(node_outputs, start=1)
        }
        target_matches = [
            output_name
            for output_name, node_id in node_ids_by_output.items()
            if node_id == str(sample["target_node_id"])
        ]
        if len(target_matches) != 1:
            raise ValueError("Dataset target must occur exactly once in node_outputs.")
        target_output = target_matches[0]

        comparison_id = (
            f"{resolved_category}/{self.topology_id}/"
            f"task-{target_task_id}/sample-{sample_index}"
        )
        episode_id = f"{comparison_id}/{self.condition}"
        metadata = {
            "category": resolved_category,
            "topology_id": self.topology_id,
            "condition": self.condition,
            "comparison_id": comparison_id,
            "node_count": len(node_outputs),
            "target_node_id": sample["target_node_id"],
            "target_task_id": target_task_id,
            "sample_index": sample_index,
        }

        observation = EpisodeObservation(
            episode_id=episode_id,
            prompt=sample["prompt"],
            requested_outputs=tuple(expected_outputs),
            target_output=target_output,
            metadata=metadata,
        )
        verifier_state = VerifierState(
            expected_outputs=expected_outputs,
            target_output=target_output,
            node_ids_by_output=node_ids_by_output,
        )
        return Episode(observation=observation, verifier_state=verifier_state)

    def iter_episodes(self) -> Iterator[Episode]:
        """Stream every episode in the configured dataset."""
        samples = self.dataset.get("samples")
        if not isinstance(samples, list):
            raise ValueError("Dataset must contain a sample list.")
        summary = self.dataset.get("summary") or {}
        category = self.category or summary.get("target_category")
        if not category:
            raise ValueError("category is required or must appear in dataset summary")
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("Every dataset sample must be a JSON object.")
            yield self.load_sample(sample, str(category))

    def policy_batch(self) -> Iterator[dict[str, Any]]:
        """Stream records safe to send to a policy worker."""
        for episode in self.iter_episodes():
            yield episode.to_public_dict()

    @staticmethod
    def score(completion: str, verifier_state: VerifierState) -> ScoreResult:
        return score_completion(completion, verifier_state)
