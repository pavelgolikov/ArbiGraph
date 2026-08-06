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
from tasks.gsm_task.gsm_symbolic_task import TARGET_TEMPLATE_FILES
from tasks.gsm_task.gsm_symbolic_task import load_pool, make_stage


def graph_data(nodes, edges, target_id, native_task_id):
    return {
        "directed": True,
        "multigraph": False,
        "target": {"id": target_id, "native_task_id": str(native_task_id)},
        "nodes": [{"id": node_id, "category": "gsm"} for node_id in nodes],
        "edges": [{"source": source, "destination": destination} for source, destination in edges],
    }


class GSMTaskIntegrationTest(unittest.TestCase):
    def test_gsm_pool_has_curated_templates(self):
        pool = load_pool()

        self.assertEqual(
            [task["task_id"] for task in pool],
            [int(filename.removesuffix(".json")) for filename in TARGET_TEMPLATE_FILES],
        )
        self.assertEqual(len(pool), len(TARGET_TEMPLATE_FILES))
        self.assertTrue(all(task["output_type"] == "scalar" for task in pool))
        self.assertFalse(any(task["input_type"].startswith("join-") for task in pool))

    def test_single_node_gsm_dataset_generates(self):
        dataset = generate_dataset(graph_data(["task_1"], [], "task_1", 10), 1, seed=0)
        sample = dataset["samples"][0]

        self.assertEqual(dataset["summary"]["target_category"], "gsm")
        self.assertEqual(dataset["summary"]["num_samples"], 1)
        self.assertEqual(sample["target_native_task_id"], 10)
        self.assertEqual(sample["target_node_id"], "task_1")
        self.assertIn("Compute the final numerical answer.", sample["prompt"])
        self.assertIn('Output the result as task_1_out = {"result": gsm_result_1}', sample["prompt"])
        self.assertEqual(sample["node_outputs"][0]["node_id"], "task_1")
        self.assertEqual(sample["node_outputs"][0]["native_task_name"], "GSM_0010")

    def test_generate_dataset_all_targets_uses_curated_templates(self):
        dataset = generate_dataset(graph_data(["task_1"], [], "task_1", "all"), 1, seed=1)

        self.assertEqual(
            [sample["target_native_task_id"] for sample in dataset["samples"]],
            [int(filename.removesuffix(".json")) for filename in TARGET_TEMPLATE_FILES],
        )
        self.assertEqual(dataset["summary"]["num_samples"], len(TARGET_TEMPLATE_FILES))

    def test_make_stage_renders_gsm_prompt_with_output_anchor(self):
        random.seed(0)
        task = next(task for task in load_pool() if task["task_id"] == 10)
        output, prompt, static_inputs, input_value = make_stage(task, 1, ["task_1_input"], None)

        self.assertIsInstance(output, int)
        self.assertIsInstance(input_value, int)
        self.assertEqual(static_inputs, [])
        self.assertIn("Define task_1_input =", prompt)
        self.assertIn("How much would a", prompt)
        self.assertIn('task_1_out = {"result": gsm_result_1}', prompt)

    def test_gsm_scalar_join_fills_multi_parent_node(self):
        graph = parse_custom_dag(graph_data(
            ["task_1", "task_2", "task_3", "task_4"],
            [("task_1", "task_3"), ("task_2", "task_3"), ("task_3", "task_4")],
            "task_4",
            10,
        ))
        filled = fill_dag(graph, seed=2)
        target = next(record for record in filled["nodes"] if record["node_id"] == "task_3")
        by_node = {record["node_id"]: record for record in filled["nodes"]}
        parent_values = [by_node["task_1"]["output_value"], by_node["task_2"]["output_value"]]

        self.assertEqual(target["input_type"], "scalar")
        self.assertEqual(target["input_names"], ["task_1_out", "task_2_out"])
        self.assertEqual(target["input_value"], sum(parent_values))
        self.assertIn('Let val_3_join be the sum of the following values:', target["prompt"])


if __name__ == "__main__":
    unittest.main()
