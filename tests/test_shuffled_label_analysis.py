"""Tests for strict paired shuffled-label analysis and target indexing."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "eval", ROOT / "source"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_shuffled_labels import exact_mcnemar, summarize, validate_and_pair
from grade_results import grade_sample
from util import graded_output_name


def node(node_id, output, parents, task_number):
    return {
        "node_id": node_id,
        "task_number": task_number,
        "native_task_id": 7 if node_id == "root" else 11,
        "native_task_name": f"{node_id}_task",
        "output_value": output,
        "comparator": "numeric",
        "parent_ids": parents,
        "input_value": output - 1,
        "wrong_inputs": [],
    }


def sample(nodes, correct):
    return {
        "target_native_task_id": 11,
        "sample_idx": 0,
        "target_node_id": "target",
        "node_outputs": nodes,
        "grading": {"target_correct": correct},
        "agent_status": "success",
    }


class ShuffledLabelAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.control = sample([
            node("root", 4, [], 1),
            node("target", 9, ["root"], 2),
        ], True)
        self.shuffled = sample([
            node("target", 9, ["root"], 1),
            node("root", 4, [], 2),
        ], False)

    def write_inputs(self, directory):
        directory = Path(directory)
        ordinary = directory / "ordinary.json"
        shuffled = directory / "shuffled.json"
        graph = directory / "graph.json"
        ordinary.write_text(json.dumps({"samples": [self.control]}))
        shuffled.write_text(json.dumps({"samples": [self.shuffled]}))
        graph.write_text(json.dumps({
            "nodes": [{"id": "root"}, {"id": "target"}],
            "edges": [{"source": "root", "destination": "target"}],
        }))
        return ordinary, shuffled, graph

    def test_strict_pairing_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = validate_and_pair(*self.write_inputs(directory))

        self.assertEqual(rows, [{
            "key": (11, 0),
            "ordinary_correct": 1,
            "shuffled_correct": 0,
            "target_position": 1,
            "backward_edges": 1,
            "ordinary_status": "success",
            "shuffled_status": "success",
        }])
        result = summarize("math", rows, seed=3, draws=100)
        self.assertEqual(result["paired_delta"], -1)
        self.assertEqual(result["delta_ci"], (-1.0, -1.0))
        self.assertEqual((result["gains"], result["losses"]), (0, 1))

    def test_pairing_rejects_changed_underlying_values(self):
        self.shuffled["node_outputs"][0]["output_value"] = 10
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Underlying filled graph differs"):
                validate_and_pair(*self.write_inputs(directory))

    def test_shuffled_target_uses_visible_position(self):
        target_sample = dict(self.shuffled)
        target_sample["submitted_answers"] = {
            "task_1_out": {"answer": 9, "turn": 1},
        }
        self.assertEqual(graded_output_name(target_sample), "task_1_out")
        self.assertTrue(grade_sample(target_sample)["target_correct"])

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(0, 5), 0.0625)
        self.assertEqual(exact_mcnemar(2, 3), 1.0)


if __name__ == "__main__":
    unittest.main()
