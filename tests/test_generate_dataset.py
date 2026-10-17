"""Native input banks remain verifiable through ordinary DAG generation."""

from copy import deepcopy
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE_ROOT = str(Path(__file__).resolve().parents[1] / "source")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from fill_dag import fill_dag, load_pool, make_stage, write_prompt
from generate_dataset import generate_dataset, main
from parse_dag import _custom_dag_to_dot, parse_custom_dag
from tasks.arbigraph_tools import InputTool
from grade_results import comparator
from tasks.math_task.math_task import MatrixMulTask, PolyQuotientTask, plain_value

grade_value = comparator()


def topology(nodes, edges, target, native_id):
    return {
        "directed": True, "multigraph": False,
        "target": {"id": target, "native_task_id": native_id},
        "nodes": [{"id": name, "category": category} for name, category in nodes],
        "edges": [{"source": parent, "destination": child} for parent, child in edges],
    }


def _frozen_solver(category, instance):
    """Answer any input from one frozen draw of a task's parameters.

    Solving always happens on a copy. A math task reads its input off the
    instance and caches format adapters there as it goes, which is why its own
    bank generation copies rather than re-running gen_params.
    """
    if category == "math":
        def solve(value):
            attempt = deepcopy(instance)
            attempt.input_value = deepcopy(value)
            return plain_value(attempt.solution_generator())
        return solve
    return lambda value: instance.solution_generator(deepcopy(value))


class NativeGenerationTest(unittest.TestCase):
    def test_every_bank_pair_solves_with_the_same_frozen_parameters(self):
        cases = [
            ("math", "mat_mul"), ("math", "poly_quotient"),
            ("python", "lexicographicallySmallestArray:list"),
            ("rlve", "fractional_programming_bipartite_graph_matching"),
            ("rlve", "negative_base"),
        ]
        for category, name in cases:
            with self.subTest(category=category, task=name):
                task = next(item for item in load_pool(category) if item["task_name"] == name)
                implementation = task["task_implementation"]
                initialize = implementation.__init__
                instances = []

                def capture(instance, *args, **kwargs):
                    initialize(instance, *args, **kwargs)
                    instances.append(instance)

                random.seed(12)
                with patch.object(implementation, "__init__", capture):
                    output, prompt, _, value, bank = make_stage(category, task, 2, ["task_1_out"], 999999)
                self.assertEqual(len(instances), 1)
                instance = instances[0]
                solve = _frozen_solver(category, instance)
                self.assertTrue(grade_value(solve(value), output))
                answers = [output]
                for pair in bank:
                    self.assertTrue(grade_value(solve(pair["input"]), pair["output"]))
                    self.assertFalse(any(grade_value(pair["output"], answer) for answer in answers))
                    answers.append(pair["output"])
                self.assertEqual(len(bank), 5)
                self.assertIn("task_2_input", prompt)
                self.assertNotIn("999999", prompt)

    def test_native_matrix_and_polynomial_results(self):
        """A math input is a flat list; reshaping it is part of the prompt, not the input."""
        import sympy
        random.seed(0)

        # [1, 2, 3, 4] is read as the 2x2 matrix the prompt tells the model to build.
        task = MatrixMulTask(1, ["task_1_input"], [1, 2, 3, 4], 100, 10)
        task.matrix_2 = sympy.Matrix([[2, 0], [0, 3]])
        product = plain_value(task.solution_generator())
        self.assertEqual(product, [2, 6, 6, 12])

        # Coefficients go in and come back out as plain JSON numbers, not sympy objects.
        polynomial = PolyQuotientTask(1, ["task_1_input"], [2, 5, 9, 7, 4], 100, 10)
        quotient = plain_value(polynomial.solution_generator())
        self.assertEqual(quotient, [2, 3, 4])
        self.assertTrue(all(type(value) is int for value in product + quotient))

    def test_mixed_fan_in_and_middle_target_follow_topology_parent_order(self):
        graph = topology(
            [("a", "math"), ("b", "python"), ("target", "rlve"), ("child", "math")],
            [("b", "target"), ("a", "target"), ("target", "child")], "target", 0,
        )
        filled = fill_dag(parse_custom_dag(graph), seed=4)
        records = filled["nodes"]
        self.assertEqual(records[2]["parent_ids"], ["b", "a"])
        self.assertNotIn("input_tool", filled)
        data = json.loads(json.dumps(records))
        tool = InputTool.from_records(data)
        self.assertEqual(tool.tasks[3]["parents"], [2, 1])
        parent_outputs = [{"result": records[index]["output_value"]} for index in (1, 0)]
        self.assertEqual(tool.get_input(3, parent_outputs), records[2]["input_value"])
        self.assertEqual(filled["ground_truth"], records[2]["output_value"])
        prompt = write_prompt(filled)
        self.assertIn(
            "call the get_input tool with task_index=3 and "
            "input_dependencies=[task_2_out, task_1_out]",
            prompt,
        )
        for obsolete in ("preprocessing", "interleaving", "_fd", "wrong_inputs"):
            self.assertNotIn(obsolete, prompt)
        self.assertNotIn("Define task_3_input =", prompt)

    def test_type_mismatch_is_allowed(self):
        graph = topology([("a", "math"), ("b", "math")], [("a", "b")], "b", 0)
        pool = load_pool("math")
        scalar = next(task for task in pool if task["task_name"] == "divisor_sigma_square")
        target = next(task for task in pool if task["task_name"] == "convolution")
        graph["target"]["native_task_id"] = target["task_id"]
        from fill_dag import pick_task as original_pick

        def force_scalar(category, inputs, **kwargs):
            if kwargs["task_number"] == 1:
                kwargs["pinned_task"] = scalar
            return original_pick(category, inputs, **kwargs)

        with patch("fill_dag.pick_task", side_effect=force_scalar):
            filled = fill_dag(parse_custom_dag(graph), seed=1)
        self.assertEqual(filled["nodes"][0]["output_type"], "scalar")
        self.assertEqual(filled["nodes"][1]["input_type"], "list")
        self.assertEqual(filled["nodes"][0]["wrong_inputs"], [])
        self.assertEqual(len(filled["nodes"][1]["wrong_inputs"]), 5)

    def test_shuffled_task_labels_rewrite_references_without_changing_tasks(self):
        graph = topology(
            [("root", "math"), ("left", "math"), ("right", "math"), ("target", "math")],
            [("root", "left"), ("root", "right"), ("left", "target"), ("right", "target")],
            "target", 0,
        )
        ordinary = generate_dataset(graph, 2, seed=9)
        shuffled = generate_dataset(graph, 2, seed=9, shuffle_task_labels=True)
        self.assertEqual(shuffled, generate_dataset(graph, 2, seed=9, shuffle_task_labels=True))
        self.assertFalse(ordinary["summary"]["shuffle_task_labels"])
        self.assertTrue(shuffled["summary"]["shuffle_task_labels"])

        mappings = []
        for control, changed in zip(ordinary["samples"], shuffled["samples"]):
            control_by_node = {row["node_id"]: row for row in control["node_outputs"]}
            changed_by_node = {row["node_id"]: row for row in changed["node_outputs"]}
            mapping = {node_id: row["task_number"] for node_id, row in changed_by_node.items()}
            mappings.append(mapping)

            self.assertEqual(sorted(mapping.values()), [1, 2, 3, 4])
            self.assertEqual(
                [row["task_number"] for row in changed["node_outputs"]],
                [1, 2, 3, 4],
            )
            self.assertTrue(any(mapping[parent] > mapping[child] for parent, child in (
                ("root", "left"), ("root", "right"),
                ("left", "target"), ("right", "target"),
            )))
            self.assertEqual(
                re.findall(r"^Task (\d+):$", changed["prompt"], flags=re.MULTILINE),
                ["1", "2", "3", "4"],
            )
            target_number = mapping["target"]
            self.assertTrue(changed["prompt"].startswith(
                f"Solve task {target_number}. Once you have its result"
            ))

            for node_id, row in changed_by_node.items():
                control_row = dict(control_by_node[node_id])
                changed_row = dict(row)
                control_row.pop("task_number")
                changed_row.pop("task_number")
                self.assertEqual(changed_row, control_row)
                dependencies = ", ".join(
                    f"task_{mapping[parent]}_out" for parent in row["parent_ids"]
                )
                self.assertIn(
                    f"task_index={mapping[node_id]} and input_dependencies=[{dependencies}]",
                    changed["prompt"],
                )

            tool = InputTool.from_records(changed["node_outputs"])
            child = next(
                row for row in changed["node_outputs"]
                if row["parent_ids"] and any(
                    mapping[parent] > row["task_number"] for parent in row["parent_ids"]
                )
            )
            dependencies = [
                {"result": changed_by_node[parent]["output_value"]}
                for parent in child["parent_ids"]
            ]
            self.assertEqual(tool.get_input(child["task_number"], dependencies), child["input_value"])

        self.assertNotEqual(mappings[0], mappings[1])

    def test_serialized_dataset_and_cli_store_inputs_without_runtime_state(self):
        graph = topology([("a", "math"), ("b", "math")], [("a", "b")], "b", 0)
        with patch.object(InputTool, "__init__", side_effect=AssertionError("Runtime tool used during generation")):
            first = generate_dataset(graph, 1, seed=3)
            self.assertEqual(first, generate_dataset(graph, 1, seed=3))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "graph.json"
            source.write_text(json.dumps(graph))
            output = Path(directory) / "dataset"
            with patch("generate_dataset.render_custom_dag_svg"):
                main(["--graph", str(source), "--output_dir", str(output), "--seed", "3"])
            saved = json.loads((output / "dataset_dataset.json").read_text())
        self.assertEqual(saved, json.loads(json.dumps(first)))
        sample = saved["samples"][0]
        self.assertNotIn("input_tool", sample)
        self.assertEqual(set(sample["node_outputs"][0]), {
            "node_id", "task_number", "native_task_id", "native_task_name", "output_value",
            "comparator", "parent_ids", "input_value", "wrong_inputs",
        })
        # The task's declared comparator travels with its answer, so grading and
        # the input tool judge it the same way the task does.
        self.assertEqual([node["comparator"] for node in sample["node_outputs"]],
                         ["numeric", "numeric"])
        self.assertNotIn("get_input_attempts_per_task", saved["summary"])
        self.assertNotIn("valid submissions", sample["prompt"])
        self.assertNotIn("invalid requests", sample["prompt"])
        tool = InputTool.from_records(sample["node_outputs"], attempts_per_task=5)
        self.assertEqual(tool.get_input(2, [{"result": tool.tasks[1]["output"]}]), tool.tasks[2]["input"])
        self.assertEqual(tool.get_input(2, [{"result": None}]), tool.tasks[2]["wrong_inputs"][1]["input"])

    def test_cli_renders_first_shuffled_sample_labels_without_changing_edges(self):
        graph = topology(
            [("source", "math"), ("target", "math")],
            [("source", "target")], "target", 0,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "graph.json"
            source.write_text(json.dumps(graph))
            output = Path(directory) / "shuffled"
            with patch("generate_dataset.render_custom_dag_svg") as render:
                main([
                    "--graph", str(source), "--output_dir", str(output),
                    "--seed", "3", "--shuffle_task_labels",
                ])
            saved = json.loads((output / "shuffled_dataset.json").read_text())

        mapping = {
            row["node_id"]: row["task_number"]
            for row in saved["samples"][0]["node_outputs"]
        }
        self.assertEqual(render.call_args.kwargs, {
            "task_numbers": mapping,
            "annotation": "Visible task labels for Prompt 1 (sample_idx=0)",
        })
        dot = _custom_dag_to_dot(
            parse_custom_dag(graph), "LR", mapping,
            "Visible task labels for Prompt 1 (sample_idx=0)",
        )
        self.assertIn(f'task_{mapping["source"]}', dot)
        self.assertIn(f'task_{mapping["target"]}', dot)
        self.assertIn('\"source\" -> \"target\"', dot)
        self.assertIn("Visible task labels for Prompt 1", dot)

    def test_bank_failure_stops_generation(self):
        graph = topology([("root", "math")], [], "root", "all")
        with patch("generate_dataset.fill_dag", side_effect=ValueError("Cannot complete bank")) as fill:
            with self.assertRaisesRegex(RuntimeError, "Cannot complete bank"):
                generate_dataset(graph, 1)
        fill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
