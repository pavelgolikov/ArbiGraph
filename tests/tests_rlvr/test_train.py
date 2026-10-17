"""CPU tests of the dataset and reward adapter; no model download required."""

import json
from pathlib import Path
import tempfile
import unittest

from full_train.prepare import SPLITS, TRAIN_TOPOLOGIES, select_split, make_edge_cut_dataset
from full_train.train import SYSTEM_PROMPT, calculator, check_training_logs, load_training_rows, make_reward
from arbigraph_rlvr import VerifierState


class TrainingAdapterTest(unittest.TestCase):
    def test_fixed_splits(self):
        self.assertEqual([len(ids) for ids in SPLITS.values()], [24, 8, 8])
        self.assertEqual(sorted(sum((list(ids) for ids in SPLITS.values()), [])), list(range(40)))
        for split, per_signature in (("train", 6), ("dev", 2), ("test", 2)):
            self.assertEqual([sum(t // 10 == i for t in SPLITS[split]) for i in range(4)], [per_signature] * 4)

    def test_pairing_and_policy_rows(self):
        dataset = {
            "summary": {"target_category": "math", "num_samples_per_task": 8, "seed": 0},
            "samples": [
                {
                    "target_native_task_id": task,
                    "sample_idx": i,
                    "target_node_id": "task_2",
                    "prompt": 'Compute task_1_out["result"] + 1.',
                    "node_outputs": [
                        {"node_id": "task_1", "output_value": 987654321},
                        {"node_id": "task_2", "output_value": 987654322},
                    ],
                }
                for task in range(40) for i in range(8)
            ],
        }
        full = select_split(dataset, "train")
        edge = select_split(make_edge_cut_dataset(dataset, "two_chain_target"), "train")
        for original, cut in zip(full["samples"], edge["samples"], strict=True):
            self.assertEqual(original["node_outputs"], cut["node_outputs"])
            self.assertEqual(original["sample_idx"], cut["sample_idx"])
            self.assertEqual(original["target_native_task_id"], cut["target_native_task_id"])
            self.assertIn('task_1_out["result"]', original["prompt"])
            self.assertNotIn('task_1_out["result"]', cut["prompt"])
        with tempfile.TemporaryDirectory() as directory:
            for topology in TRAIN_TOPOLOGIES:
                path = Path(directory) / "train" / topology
                path.mkdir(parents=True)
                (path / "full.json").write_text(json.dumps(full))
                (path / "edge_cut.json").write_text(json.dumps(edge))
            rows, verifiers = load_training_rows(Path(directory), "full")
            cut_rows, _ = load_training_rows(Path(directory), "edge_cut")
        self.assertEqual(len(rows), 384)
        for row, cut_row in zip(rows, cut_rows, strict=True):
            self.assertEqual(set(row), {"prompt", "episode_id"})
            self.assertEqual(row["prompt"][0], {"role": "system", "content": SYSTEM_PROMPT})
            self.assertNotIn("987654322", json.dumps(row))
            self.assertIn(row["episode_id"], verifiers)
            self.assertEqual(row["episode_id"].rsplit("/", 1)[0], cut_row["episode_id"].rsplit("/", 1)[0])

    def test_reward_uses_assistant_text_in_order(self):
        state = VerifierState({"task_1_out": 2, "task_2_out": 3}, "task_2_out", {"task_1_out": "n1", "task_2_out": "n2"})
        reward = make_reward({"example": state})
        correct = 'task_2_out = {"result": 3}'
        incorrect = 'task_2_out = {"result": 4}'
        cases = [
            ([{"role": "assistant", "content": correct}], 1.0),
            ([{"role": "assistant", "content": incorrect}], 0.0),
            ([{"role": "assistant", "content": "No answer"}], 0.0),
            ([{"role": "tool", "content": correct}], 0.0),
            ([{"role": "assistant", "content": incorrect}, {"role": "tool", "content": correct}], 0.0),
            ([{"role": "assistant", "content": incorrect}, {"role": "assistant", "content": correct}], 1.0),
            ([{"role": "assistant", "reasoning_content": correct, "content": incorrect}], 0.0),
            ([{"role": "assistant", "reasoning_content": correct, "content": None}], 1.0),
            ([{"role": "assistant", "content": None, "tool_calls": [{"arguments": correct}]}], 0.0),
        ]
        for messages, expected in cases:
            with self.subTest(messages=messages):
                self.assertEqual(reward([messages], ["example"]), [expected])
        with self.assertRaises(ValueError):
            reward([cases[0][0]], [])
        with self.assertRaises(KeyError):
            reward([cases[0][0]], ["missing-id"])

    def test_calculator_adapter(self):
        from transformers.utils import get_json_schema

        schema = get_json_schema(calculator)["function"]
        self.assertEqual(schema["name"], "calculator")
        self.assertEqual(schema["parameters"]["required"], ["expressions"])
        self.assertEqual(schema["parameters"]["properties"]["expressions"]["items"], {"type": "string"})
        self.assertEqual(json.loads(calculator(["17 * 43 + 5", "100 // 7", "pow(3, 4, 5)"])), {"results": ["736", "14", "1"]})

    def test_execution_errors_are_fatal(self):
        check_training_logs({"loss": 0.0, "grad_norm": 0.1})
        for key in ("loss", "grad_norm"):
            for value in (float("nan"), float("inf")):
                with self.assertRaises(FloatingPointError):
                    check_training_logs({key: value})
        with self.assertRaises(RuntimeError):
            check_training_logs({"tools/failure_frequency": 0.25})


if __name__ == "__main__":
    unittest.main()
