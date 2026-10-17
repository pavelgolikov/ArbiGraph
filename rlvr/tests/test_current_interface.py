"""CPU checks of both current-format RLVR sample modes and their tool rules."""

import json
import unittest
from pathlib import Path

from arbigraph_rlvr import ArbiGraphEnv, DatasetLoader, EpisodeSpec, trainer_reward_callback
from generate_dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[2]
GRAPH = {
    "directed": True,
    "multigraph": False,
    "target": {"id": "child", "native_task_id": 0},
    "nodes": [
        {"id": "root", "category": "math"},
        {"id": "child", "category": "math"},
    ],
    "edges": [{"source": "root", "destination": "child"}],
}


class CurrentInterfaceTest(unittest.TestCase):
    def test_materialized_dataset_serves_input_and_requires_submission(self):
        path = ROOT / "generate/generated/exp1/math/single_target_baseline/single_target_baseline_dataset.json"
        episode = next(DatasetLoader(path, "single_target_baseline", "full").iter_episodes())
        public = json.dumps(episode.to_public_dict())
        self.assertIn('"get_input"', public)
        self.assertIn('"submit_answers"', public)
        self.assertNotIn("input_value", public)
        self.assertNotIn("wrong_inputs", public)

        target = episode.verifier_state.target_output
        truth = episode.verifier_state.expected_outputs[target]
        session = episode.start()
        self.assertEqual(session.score(f'{target} = {json.dumps({"result": truth})}').reward, 0.0)
        self.assertFalse(session.score(f'{target} = {json.dumps({"result": truth})}').node_correct[
            episode.observation.metadata["target_node_id"]
        ])
        self.assertIn("not been requested", session.call_tool(
            "submit_answers", {"answers": [{"result": truth}]}
        ))
        self.assertEqual(json.loads(session.call_tool(
            "get_input", {"task_index": 1, "input_dependencies": []}
        )), episode.verifier_state.input_records[0]["input_value"])
        self.assertEqual(session.call_tool(
            "submit_answers", {"answers": [{"result": truth}]}
        ), "Submission accepted.")
        self.assertEqual(session.score().reward, 1.0)
        self.assertFalse(episode.start().done)
        self.assertEqual(trainer_reward_callback([""], [session]), [1.0])
        self.assertEqual(trainer_reward_callback([""], [episode.start()]), [0.0])
        wrong = episode.start()
        wrong.call_tool("get_input", {"task_index": 1, "input_dependencies": []})
        wrong.call_tool("submit_answers", {"answers": [{"result": "wrong"}]})
        self.assertEqual(DatasetLoader.score("", wrong).reward, 0.0)

    def test_on_demand_banks_budget_and_same_turn_submission(self):
        episode = ArbiGraphEnv().reset(EpisodeSpec(
            "two_node", GRAPH, seed=23, sample_idx=0, get_input_attempts_per_task=2,
        ))
        records = episode.verifier_state.input_records
        root_truth = episode.verifier_state.expected_outputs["task_1_out"]
        target_truth = episode.verifier_state.expected_outputs["task_2_out"]
        session = episode.start()
        self.assertEqual(json.loads(session.call_tool(
            "get_input", {"task_index": 1, "input_dependencies": []}
        )), records[0]["input_value"])
        self.assertEqual(json.loads(session.call_tool(
            "get_input", {"task_index": 2, "input_dependencies": [{"result": "wrong"}]}
        )), records[1]["wrong_inputs"][0]["input"])
        self.assertEqual(json.loads(session.call_tool(
            "get_input", {"task_index": 2, "input_dependencies": [{"result": root_truth}]}
        )), records[1]["input_value"])
        self.assertEqual(session.call_tool(
            "submit_answers", {"answers": [{"result": target_truth}]}
        ), "Submission accepted.")
        self.assertTrue(session.done)
        self.assertEqual(session.score().reward, 1.0)
        self.assertEqual(ArbiGraphEnv().score("", session).reward, 1.0)
        limited = episode.start()
        for _ in range(2):
            limited.call_tool("get_input", {"task_index": 2, "input_dependencies": []})
        self.assertIn("limit reached", limited.call_tool(
            "get_input", {"task_index": 2, "input_dependencies": []}
        ))

    def test_materialized_dependent_task_uses_its_bank(self):
        data = generate_dataset(GRAPH, 1, seed=0, get_input_attempts_per_task=2)
        episode = next(DatasetLoader(
            data, "two_node", "full", get_input_attempts_per_task=2,
        ).iter_episodes())
        session = episode.start()
        child = episode.verifier_state.input_records[1]
        self.assertEqual(json.loads(session.call_tool(
            "get_input", {"task_index": 2, "input_dependencies": [{"result": "wrong"}]}
        )), child["wrong_inputs"][0]["input"])

    def test_same_turn_cannot_submit_before_target_input(self):
        episode = ArbiGraphEnv().reset(EpisodeSpec("two_node", GRAPH, 23, 0))
        session = episode.start()
        root_truth = episode.verifier_state.expected_outputs["task_1_out"]
        truth = episode.verifier_state.expected_outputs["task_2_out"]
        responses = session.call_tools([
            {"name": "get_input", "arguments": {
                "task_index": 2, "input_dependencies": [{"result": root_truth}]}},
            {"name": "submit_answers", "arguments": {"answers": [{"result": truth}]}},
        ])
        self.assertEqual(json.loads(responses[0]), episode.verifier_state.input_records[1]["input_value"])
        self.assertIn("not been requested", responses[1])
        self.assertFalse(session.done)
        self.assertEqual(session.score().reward, 0.0)
        self.assertEqual(session.call_tool("submit_answers", {"answers": [{"result": truth}]}),
                         "Submission accepted.")

    def test_shuffled_labels_keep_tool_lookup_aligned(self):
        episode = ArbiGraphEnv().reset(EpisodeSpec(
            "shuffled", GRAPH, 23, 0, shuffle_task_labels=True,
        ))
        records = episode.verifier_state.input_records
        root_number = next(record["task_number"] for record in records if record["node_id"] == "root")
        session = episode.start()
        self.assertEqual(json.loads(session.call_tool(
            "get_input", {"task_index": root_number, "input_dependencies": []}
        )), next(record["input_value"] for record in records if record["node_id"] == "root"))

    def test_fan_in_uses_parent_order(self):
        graph = {
            **GRAPH,
            "nodes": [
                {"id": "left", "category": "math"},
                {"id": "right", "category": "math"},
                {"id": "child", "category": "math"},
            ],
            "edges": [
                {"source": "right", "destination": "child"},
                {"source": "left", "destination": "child"},
            ],
        }
        episode = ArbiGraphEnv().reset(EpisodeSpec("fan_in", graph, 23, 0))
        state = episode.verifier_state
        records = state.input_records
        target_index = next(record["task_number"] for record in records if record["node_id"] == "child")
        parents = next(record["parent_ids"] for record in records if record["node_id"] == "child")
        dependencies = [
            {"result": next(record["output_value"] for record in records if record["node_id"] == parent)}
            for parent in parents
        ]
        session = episode.start()
        served = session.call_tool("get_input", {
            "task_index": target_index, "input_dependencies": dependencies,
        })
        self.assertEqual(json.loads(served), next(
            record["input_value"] for record in records if record["node_id"] == "child"
        ))


if __name__ == "__main__":
    unittest.main()
