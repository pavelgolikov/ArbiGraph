import os
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GSM_TEMPLATES_DIR = os.path.join(
    REPO_ROOT,
    "ml-gsm-symbolic",
    "templates",
    "symbolic",
)
requires_gsm_templates = unittest.skipUnless(
    os.path.isdir(GSM_TEMPLATES_DIR),
    "requires GSM-Symbolic templates under ml-gsm-symbolic/templates/symbolic",
)

import generate_custom
from generate_custom import generate_dataset_from_graph_data, load_graph_data
from run_agent_calc import expected_task_outputs


def direct_graph():
    return {
        "directed": True,
        "multigraph": False,
        "graph": {"target_node": "task_2"},
        "nodes": [
            {
                "id": "task_1",
                "kind": "task",
                "family": "math",
                "task_id": 21,
                "input_kind": "scalar",
                "output_kind": "list",
            },
            {
                "id": "task_2",
                "kind": "task",
                "family": "math",
                "task_id": 10,
                "input_kind": "list",
                "output_kind": "scalar",
            },
        ],
        "edges": [
            {"source": "task_1", "destination": "task_2"},
        ],
    }


def branched_graph():
    return {
        "directed": True,
        "multigraph": False,
        "graph": {"target_node": "task_4"},
        "nodes": [
            {
                "id": "task_1",
                "kind": "task",
                "family": "math",
                "task_id": 21,
                "input_kind": "scalar",
                "output_kind": "list",
            },
            {"id": "adapter_1", "kind": "adapter", "op": "pick", "index": 0},
            {"id": "adapter_2", "kind": "adapter", "op": "pick", "index": 1},
            {
                "id": "task_2",
                "kind": "task",
                "family": "math",
                "task_id": 34,
                "input_kind": "scalar",
                "output_kind": "scalar",
            },
            {
                "id": "task_3",
                "kind": "task",
                "family": "gsm",
                "task_id": 0,
                "input_kind": "scalar",
                "output_kind": "scalar",
            },
            {"id": "adapter_3", "kind": "adapter", "op": "pack"},
            {
                "id": "task_4",
                "kind": "task",
                "family": "math",
                "task_id": 10,
                "input_kind": "list",
                "output_kind": "scalar",
            },
        ],
        "edges": [
            {"source": "task_1", "destination": "adapter_1"},
            {"source": "task_1", "destination": "adapter_2"},
            {"source": "adapter_1", "destination": "task_2"},
            {"source": "adapter_2", "destination": "task_3"},
            {"source": "task_2", "destination": "adapter_3", "slot": 0},
            {"source": "task_3", "destination": "adapter_3", "slot": 1},
            {"source": "adapter_3", "destination": "task_4"},
        ],
    }


def target_not_sink_graph():
    return {
        "directed": True,
        "multigraph": False,
        "graph": {"target_node": "task_2"},
        "nodes": [
            {
                "id": "task_1",
                "kind": "task",
                "family": "math",
                "task_id": 21,
                "input_kind": "scalar",
                "output_kind": "list",
            },
            {"id": "adapter_1", "kind": "adapter", "op": "pick", "index": 0},
            {
                "id": "task_2",
                "kind": "task",
                "family": "math",
                "task_id": 30,
                "input_kind": "scalar",
                "output_kind": "scalar",
            },
            {
                "id": "task_3",
                "kind": "task",
                "family": "gsm",
                "task_id": 0,
                "input_kind": "scalar",
                "output_kind": "scalar",
            },
        ],
        "edges": [
            {"source": "task_1", "destination": "adapter_1"},
            {"source": "adapter_1", "destination": "task_2"},
            {"source": "task_2", "destination": "task_3"},
        ],
    }


class CustomDagGeneratorTest(unittest.TestCase):
    def test_direct_edge_uses_upstream_task_output_without_adapter(self):
        data = generate_dataset_from_graph_data(
            direct_graph(),
            num_samples_per_task=1,
            seed=0,
        )
        sample = data["samples"][0]
        stages = sample["custom_dag"]["stages"]

        self.assertEqual(stages[0]["node_id"], "task_1")
        self.assertEqual(stages[0]["input_name"], "task_1_input")
        self.assertEqual(stages[1]["node_id"], "task_2")
        self.assertEqual(stages[1]["input_name"], "task_1_out")
        self.assertEqual(sample["target_node"], "task_2")
        self.assertEqual(sample["target_output_name"], "task_2_out")
        self.assertEqual(sample["custom_dag"]["adapters"], [])
        self.assertEqual(
            expected_task_outputs(sample["prompt"]),
            ["task_1_out", "task_2_out"],
        )

    @requires_gsm_templates
    def test_pick_and_pack_adapters_compile_real_transformations(self):
        data = generate_dataset_from_graph_data(
            branched_graph(),
            num_samples_per_task=1,
            seed=1,
        )
        sample = data["samples"][0]
        adapters = sample["custom_dag"]["adapters"]

        self.assertEqual([adapter["op"] for adapter in adapters], ["pick", "pick", "pack"])
        self.assertEqual(adapters[0]["output_name"], "task_2_input")
        self.assertEqual(adapters[1]["output_name"], "task_3_input")
        self.assertEqual(adapters[2]["output_name"], "task_4_input")
        self.assertIn("Define task_2_input = task_1_out[0].", sample["prompt"])
        self.assertIn("Define task_3_input = task_1_out[1].", sample["prompt"])
        self.assertNotIn("task_4_input_prelim", sample["prompt"])
        self.assertIn("Define task_4_input as the following list:", sample["prompt"])
        self.assertEqual(sample["custom_dag"]["stages"][-1]["input_name"], "task_4_input")
        self.assertEqual(
            expected_task_outputs(sample["prompt"]),
            ["task_1_out", "task_2_out", "task_3_out", "task_4_out"],
        )

    @requires_gsm_templates
    def test_target_node_can_be_non_sink(self):
        data = generate_dataset_from_graph_data(
            target_not_sink_graph(),
            num_samples_per_task=1,
            seed=2,
        )
        sample = data["samples"][0]

        self.assertEqual(sample["target_node"], "task_2")
        self.assertEqual(sample["target_output_name"], "task_2_out")
        self.assertEqual(
            expected_task_outputs(sample["prompt"]),
            ["task_1_out", "task_2_out", "task_3_out"],
        )

    def test_non_target_task_can_be_selected_from_family_pool(self):
        graph = direct_graph()
        del graph["nodes"][0]["task_id"]

        data = generate_dataset_from_graph_data(
            graph,
            num_samples_per_task=1,
            seed=3,
        )
        stage = data["samples"][0]["custom_dag"]["stages"][0]

        self.assertEqual(stage["node_id"], "task_1")
        self.assertEqual(stage["family"], "math")
        self.assertNotIn("input_kind", stage)
        self.assertNotIn("output_kind", stage)

    def test_same_task_id_can_appear_at_multiple_nodes(self):
        graph = {
            "directed": True,
            "multigraph": False,
            "graph": {"target_node": "task_2"},
            "nodes": [
                {
                    "id": "task_1",
                    "kind": "task",
                    "family": "math",
                    "task_id": 34,
                    "input_kind": "scalar",
                    "output_kind": "scalar",
                },
                {
                    "id": "task_2",
                    "kind": "task",
                    "family": "math",
                    "task_id": 34,
                    "input_kind": "scalar",
                    "output_kind": "scalar",
                },
            ],
            "edges": [{"source": "task_1", "destination": "task_2"}],
        }

        data = generate_dataset_from_graph_data(graph, num_samples_per_task=1, seed=5)
        stages = data["samples"][0]["custom_dag"]["stages"]

        self.assertEqual([stage["task_id"] for stage in stages], [34, 34])

    def test_candidate_selection_backtracks_to_prior_node(self):
        graph = {
            "directed": True,
            "multigraph": False,
            "graph": {"target_node": "task_2"},
            "nodes": [
                {
                    "id": "task_1",
                    "kind": "task",
                    "family": "math",
                    "input_kind": "scalar",
                    "output_kind": "scalar",
                },
                {
                    "id": "task_2",
                    "kind": "task",
                    "family": "math",
                    "task_id": 2,
                    "input_kind": "scalar",
                    "output_kind": "scalar",
                },
            ],
            "edges": [{"source": "task_1", "destination": "task_2"}],
        }
        pool = [
            {"family": "math", "task_id": 0, "task_name": "bad_parent", "input_kinds": ["scalar"], "output_kind": "scalar"},
            {"family": "math", "task_id": 1, "task_name": "good_parent", "input_kinds": ["scalar"], "output_kind": "scalar"},
            {"family": "math", "task_id": 2, "task_name": "target", "input_kinds": ["scalar"], "output_kind": "scalar"},
        ]
        calls = []

        def fake_make_stage(_family, task, task_number, _input_name, value):
            calls.append(task["task_id"])
            if task["task_id"] == 2 and value == 0:
                raise ValueError("target rejected first parent candidate")
            output = task["task_id"] if task_number == 1 else value + 10
            return output, f"Task {task_number}: Save task_{task_number}_out.", []

        with patch("generate_custom.load_pool", return_value=pool), patch("generate_custom.make_stage", side_effect=fake_make_stage):
            data = generate_dataset_from_graph_data(
                graph,
                num_samples_per_task=1,
                seed=5,
                max_sample_attempts=1,
            )

        stages = data["samples"][0]["custom_dag"]["stages"]
        self.assertEqual(calls, [0, 2, 1, 2])
        self.assertEqual([stage["task_id"] for stage in stages], [1, 2])

    def test_target_family_is_not_silently_filtered(self):
        graph = {
            "directed": True,
            "multigraph": False,
            "graph": {"target_node": "task_1"},
            "nodes": [
                {
                    "id": "task_1",
                    "kind": "task",
                    "family": "math",
                    "input_kind": "scalar",
                    "output_kind": "scalar",
                },
            ],
            "edges": [],
        }

        with self.assertRaisesRegex(ValueError, "does not accept declared input_kind"):
            generate_dataset_from_graph_data(graph, num_samples_per_task=1, seed=4)

    def test_identity_adapter_is_not_supported(self):
        graph = direct_graph()
        graph["nodes"].insert(1, {"id": "adapter_1", "kind": "adapter", "op": "identity"})
        graph["edges"] = [
            {"source": "task_1", "destination": "adapter_1"},
            {"source": "adapter_1", "destination": "task_2"},
        ]

        with self.assertRaisesRegex(ValueError, "Adapter node"):
            load_graph_data(graph)

    def test_pack_requires_explicit_slots(self):
        graph = branched_graph()
        for edge in graph["edges"]:
            edge.pop("slot", None)

        with self.assertRaises(KeyError):
            load_graph_data(graph)

    def test_custom_generator_does_not_import_other_generators(self):
        with open(generate_custom.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertNotIn("from generate_", source)
        self.assertNotIn("import generate_", source)


if __name__ == "__main__":
    unittest.main()
