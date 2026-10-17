import json
import tempfile
import unittest
from pathlib import Path

from arbigraph_rlvr import DatasetLoader, trainer_reward_callback


def sample(outputs=(7, 11, 13), target_node_id="node_c"):
    node_ids = ("node_a", "node_c", "node_b")
    return {
        "target_native_task_id": 4,
        "target_node_id": target_node_id,
        "sample_idx": 2,
        "prompt": "Solve all tasks.",
        "node_outputs": [
            {"node_id": node_id, "output_value": output}
            for node_id, output in zip(node_ids, outputs)
        ],
    }


def dataset(record=None):
    return {
        "summary": {"target_category": "math"},
        "samples": [record or sample()],
    }


def correct_completion(verifier_state):
    return "\n".join(
        f"{name} = {json.dumps({'result': value}, ensure_ascii=True)}"
        for name, value in verifier_state.expected_outputs.items()
    )


class DatasetLoaderTest(unittest.TestCase):
    def make_loader(self, condition="full", data=None):
        return DatasetLoader(
            data or dataset(),
            "example_topology",
            condition,
            None,
        )

    def test_loads_in_memory_generated_dataset(self):
        episode = next(self.make_loader().iter_episodes())

        self.assertEqual(episode.observation.prompt, "Solve all tasks.")
        self.assertEqual(episode.observation.metadata["category"], "math")
        self.assertEqual(episode.verifier_state.target_output, "task_2_out")

    def test_loads_dataset_from_arbitrary_path(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            dataset_path.write_text(json.dumps(dataset()), encoding="utf-8")
            loader = DatasetLoader(
                dataset_path,
                "example_topology",
                "full",
            )
            episode = next(loader.iter_episodes())

        self.assertEqual(episode.observation.metadata["category"], "math")
        self.assertEqual(episode.observation.metadata["condition"], "full")

    def test_output_names_are_reconstructed_by_position(self):
        episode = next(self.make_loader().iter_episodes())

        self.assertEqual(
            episode.verifier_state.node_ids_by_output,
            {
                "task_1_out": "node_a",
                "task_2_out": "node_c",
                "task_3_out": "node_b",
            },
        )
        self.assertEqual(episode.verifier_state.target_output, "task_2_out")

    def test_conditions_have_distinct_episode_ids_and_shared_comparison_id(self):
        episodes = [
            next(self.make_loader(condition).iter_episodes())
            for condition in ("full", "edge_cut", "isolated_target")
        ]

        self.assertEqual(len({item.observation.episode_id for item in episodes}), 3)
        self.assertEqual(
            len({item.observation.metadata["comparison_id"] for item in episodes}),
            1,
        )

    def test_one_node_sample_uses_its_own_answer_key(self):
        isolated_sample = {
            "target_native_task_id": 4,
            "target_node_id": "task_1",
            "sample_idx": 2,
            "prompt": "Solve the target.",
            "node_outputs": [{"node_id": "task_1", "output_value": 99}],
        }
        loader = DatasetLoader(
            dataset(isolated_sample),
            "example_topology",
            "isolated_target",
        )
        episode = next(loader.iter_episodes())

        self.assertEqual(episode.verifier_state.expected_outputs, {"task_1_out": 99})
        self.assertEqual(
            loader.score(
                'task_1_out = {"result": 99}',
                episode.verifier_state,
            ).reward,
            1.0,
        )

    def test_policy_batch_contains_no_hidden_answer_fields(self):
        record = next(self.make_loader().policy_batch())
        serialized = json.dumps(record, sort_keys=True)

        self.assertNotIn("node_outputs", serialized)
        self.assertNotIn("expected_outputs", serialized)
        self.assertNotIn("verifier_state", serialized)

    def test_trainer_callback_smoke(self):
        episodes = [
            next(self.make_loader(condition).iter_episodes())
            for condition in ("full", "edge_cut", "isolated_target")
        ]
        states = [episode.verifier_state for episode in episodes]
        completions = [correct_completion(state) for state in states]

        rewards = trainer_reward_callback(completions, states)
        self.assertEqual(rewards, [1.0, 1.0, 1.0])
        self.assertEqual(sum(rewards) / len(rewards), 1.0)
        self.assertEqual(trainer_reward_callback(["", "", ""], states), [0.0] * 3)


if __name__ == "__main__":
    unittest.main()
