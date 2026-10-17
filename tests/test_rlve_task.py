"""RLVE tasks answer a bank of native inputs served by the get_input tool."""

from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch


SOURCE_ROOT = str(Path(__file__).resolve().parents[1] / "source")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from fill_dag import fill_dag, write_prompt
from parse_dag import parse_custom_dag
from tasks.arbigraph_tools import InputTool
from tasks.rlve_task.rlve_task import (
    RLVE_TASK_CLASSES,
    same_answer as grade_value,
    FractionalProgramming_BipartiteGraphMatching,
    MinPairSumMultiplicationPermutation,
    SpiralMatrix,
    load_pool,
    make_stage,
)


def graph_data(nodes, edges, target_id, native_task_id):
    return {
        "directed": True,
        "multigraph": False,
        "target": {"id": target_id, "native_task_id": str(native_task_id)},
        "nodes": [{"id": node_id, "category": "rlve"} for node_id in nodes],
        "edges": [{"source": source, "destination": destination} for source, destination in edges],
    }


class RLVETaskTest(unittest.TestCase):
    def test_pool_covers_every_task_class(self):
        pool = load_pool()

        self.assertEqual(len(RLVE_TASK_CLASSES), 40)
        self.assertEqual(len(pool), len(RLVE_TASK_CLASSES))
        self.assertEqual([task["task_id"] for task in pool], list(range(len(pool))))
        for task in pool:
            self.assertEqual(task["category"], "rlve")
            self.assertIn(task["input_type"], {"list", "scalar"})
            self.assertIn(task["output_type"], {"list", "scalar"})

    def test_every_task_solves_deterministically_without_touching_its_parameters(self):
        for task_class in RLVE_TASK_CLASSES:
            with self.subTest(task=task_class.__name__):
                task = task_class(0)
                frozen = deepcopy(task.__dict__)
                rng = random.Random(123)
                for _ in range(3):
                    value = task.sample_input(rng)
                    first = task.solution_generator(deepcopy(value))
                    second = task.solution_generator(deepcopy(value))
                    self.assertEqual(first, second)
                self.assertEqual(task.__dict__, frozen)

    def test_every_task_builds_a_bank_of_distinct_answers(self):
        """A wrong input must look like the right one: same task, different answer."""
        for task in load_pool():
            with self.subTest(task=task["task_name"]):
                random.seed(0)
                output, prompt, static, value, bank = make_stage(task, 2, ["task_1_out"], 999999, 5)

                self.assertEqual(static, [])
                self.assertEqual(len(bank), 5)
                # The parent value is not the input; it only marks this as a non-root.
                self.assertNotIn("999999", prompt)
                self.assertIn("task_2_input", prompt)
                self.assertIn('Output the result as task_2_out = {"result": ', prompt)
                self.assertFalse(prompt.startswith("Task 2:"))

                answers = [output]
                for pair in bank:
                    self.assertFalse(any(grade_value(pair["output"], seen) for seen in answers))
                    answers.append(pair["output"])
                json.dumps({"inputs": [value] + [pair["input"] for pair in bank]}, allow_nan=False)

    def test_bank_answers_come_from_the_same_frozen_task(self):
        task = next(item for item in load_pool() if item["task_name"] == "knapsack")
        random.seed(7)
        instances = []
        initialize = task["task_implementation"].__init__

        def capture(instance, *args, **kwargs):
            initialize(instance, *args, **kwargs)
            instances.append(instance)

        with patch.object(task["task_implementation"], "__init__", capture):
            output, _prompt, _static, value, bank = make_stage(task, 2, ["task_1_out"], 1, 5)

        # The bank that survives comes from a single draw of parameters, or the
        # prompt would give away which input is the right one. Earlier draws are
        # discarded whole when their answers collide.
        solve = instances[-1].solution_generator
        self.assertTrue(grade_value(solve(deepcopy(value)), output))
        for pair in bank:
            self.assertTrue(grade_value(solve(deepcopy(pair["input"])), pair["output"]))

    def test_matrix_tasks_sample_native_rows(self):
        for task_class, shape in (
            (SpiralMatrix, (4, 2)),
            (FractionalProgramming_BipartiteGraphMatching, (6, 6)),
            (MinPairSumMultiplicationPermutation, (6, 6)),
        ):
            with self.subTest(task=task_class.__name__):
                task = task_class(0)
                value = task.sample_input(random.Random(123))

                self.assertEqual(len(value), shape[0])
                self.assertTrue(all(len(row) == shape[1] for row in value))
                self.assertTrue(all(type(item) is int for row in value for item in row))
                # The prompt indexes the input as a matrix, so the tool must serve one.
                self.assertIn(f"task_2_input[{shape[0] - 1}][{shape[1] - 1}]", task.prompt_generator(2))

    def test_spiral_uses_the_matrix_as_supplied(self):
        task = SpiralMatrix(0)

        self.assertEqual(task.solution_generator([[1, 2], [3, 4], [5, 6], [7, 8]]),
                         [1, 2, 4, 6, 8, 7, 5, 3])

    def test_root_input_is_printed_and_children_ask_for_theirs(self):
        graph = parse_custom_dag(graph_data(
            ["task_1", "task_2", "task_3"],
            [("task_1", "task_2"), ("task_2", "task_3")],
            "task_3",
            0,
        ))
        filled = fill_dag(graph, seed=0)
        records = filled["nodes"]
        prompt = write_prompt(filled)

        self.assertIn(f"Define task_1_input = {json.dumps(records[0]['input_value'])}.", prompt)
        self.assertIn(
            "To get task_2_input, call the get_input tool with task_index=2 and "
            "input_dependencies=[task_1_out]",
            prompt,
        )
        self.assertEqual(records[0]["wrong_inputs"], [])
        self.assertEqual(len(records[2]["wrong_inputs"]), 5)
        # Join adapters were named val_N_join and list_N_join; nothing merges now.
        self.assertNotIn("_join", prompt)
        self.assertNotIn('["result"]', prompt.split("Output the result as")[0])

    def test_fan_in_target_serves_its_input_on_the_listed_parent_order(self):
        graph = parse_custom_dag(graph_data(
            ["task_1", "task_2", "task_3"],
            [("task_2", "task_3"), ("task_1", "task_3")],
            "task_3",
            0,
        ))
        filled = fill_dag(graph, seed=1)
        records = json.loads(json.dumps(filled["nodes"]))
        target = records[2]
        tool = InputTool.from_records(records)

        self.assertEqual(target["parent_ids"], ["task_2", "task_1"])
        parents = [{"result": records[index]["output_value"]} for index in (1, 0)]
        self.assertEqual(tool.get_input(3, parents), target["input_value"])
        # A wrong dependency is answered with a bank input, never the right one.
        served = tool.get_input(3, [{"result": None}, {"result": None}])
        self.assertEqual(served, target["wrong_inputs"][1]["input"])
        self.assertNotEqual(served, target["input_value"])


if __name__ == "__main__":
    unittest.main()
