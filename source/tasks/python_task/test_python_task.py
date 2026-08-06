import os
import random
import sys
import unittest


SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from fill_dag import fill_dag
from generate_dataset import generate_dataset
from parse_dag import parse_custom_dag
from tasks.python_task.python_task import load_pool, make_stage


def graph_data(nodes, edges, target_id, native_task_id):
    return {
        "directed": True,
        "multigraph": False,
        "target": {"id": target_id, "native_task_id": str(native_task_id)},
        "nodes": [{"id": node_id, "category": "python"} for node_id in nodes],
        "edges": [{"source": source, "destination": destination} for source, destination in edges],
    }


class PythonTaskIntegrationTest(unittest.TestCase):
    def test_python_pool_has_variant_ids(self):
        pool = load_pool()

        self.assertEqual([task["task_id"] for task in pool], list(range(len(pool))))
        self.assertEqual(len(pool), 94)
        self.assertFalse(any(task["input_type"].startswith("join-") for task in pool))

    def test_single_node_python_dataset_generates(self):
        dataset = generate_dataset(graph_data(["task_1"], [], "task_1", 0), 1, seed=0)
        sample = dataset["samples"][0]

        self.assertEqual(dataset["summary"]["num_samples"], 1)
        self.assertEqual(sample["target_native_task_id"], 0)
        self.assertIn("Trace the execution of task_1", sample["prompt"])
        self.assertEqual(sample["node_outputs"][0]["node_id"], "task_1")

    def test_static_inputs_are_defined_inline(self):
        random.seed(0)
        task = next(task for task in load_pool() if task["task_name"] == "lexicographicallySmallestArray:list")
        _output, prompt, _static, _input_value = make_stage(task, 1, ["task_1_input"], None)

        self.assertIn("Define task_1_limit_static =", prompt)
        self.assertIn("limit = task_1_limit_static", prompt)

    def test_python_scalar_join_fills_multi_parent_node(self):
        graph = parse_custom_dag(graph_data(
            ["task_1", "task_2", "task_3"],
            [("task_1", "task_3"), ("task_2", "task_3")],
            "task_3",
            3,
        ))
        filled = fill_dag(graph, seed=1)
        join = next(record for record in filled["nodes"] if record["node_id"] == "task_3")
        parent_values = [
            record["output_value"]
            for record in filled["nodes"]
            if record["node_id"] in {"task_1", "task_2"}
        ]

        self.assertEqual(join["input_type"], "scalar")
        self.assertEqual(join["input_names"], ["task_1_out", "task_2_out"])
        self.assertEqual(join["input_value"], sum(parent_values))
        self.assertIn('Let val_3_join be the sum of the following values:', join["prompt"])


if __name__ == "__main__":
    unittest.main()
