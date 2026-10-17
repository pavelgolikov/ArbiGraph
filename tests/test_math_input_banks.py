"""Math input banks use valid native inputs and preserve the displayed task."""

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

from generate_dataset import generate_dataset
from tasks.arbigraph_tools import InputTool
from tasks.math_task.math_task import same_answer as grade_value
from tasks.math_task.math_task import (
    MATH_TASKS, MathTask, bad_output, load_pool, plain_value,
)


# Resulting scalar ranges from the adapters in main, with modulus 100.
SCALAR_INPUT_RANGES = {
    "SummatoryTotientTask": (30, 129),
    "DivisorSumTask": (1, 100),
    "PrimeOmegaSumTask": (2, 101),
    "SumDivisorCountsTask": (1, 100),
    "PrimitiveRootCountTask": (1, 100),
    "LegendreResidueWeightTask": (25, 124),
    "ScaledStirlingFirstKindTask": (0, 99),
    "DivisorSigmaSquareTask": (1, 100),
    "MultiplicativeOrderTask": (1, 100),
    "UnitaryDivisorSumTask": (1, 100),
    "ModularMultiplicationSequenceTask": (1, 100),
    "WeightedDivisorCountPrefixSequenceTask": (1, 100),
    "GrayCodeSequenceTask": (1, 100),
    "TotientShiftSequenceTask": (1, 100),
    "SmallestFactorQuadraticTask": (1, 100),
    "TotientProductSequenceTask": (1, 100),
    "PolygonDiagonalsSequenceTask": (3, 102),
    "PrimesBetweenSquaresTask": (1, 100),
    "CentralBinomialPrimeValuationVectorTask": (30, 129),
    "QuadraticResiduesPrimeTask": (1, 100),
}
POWER_OF_TWO_TASKS = {"fwht", "ntt", "subset_zeta_transform"}


class RejectionTask(MathTask):
    task_name = "rejection_test"
    input_type = "scalar"

    def solution_generator(self):
        if self.input_value < 0:
            raise ValueError("Invalid input")
        return self.input_value // 2

    def prompt_generator(self):
        return "Compute floor(task_1_input / 2)."


class MathInputBankTest(unittest.TestCase):
    def test_scalar_sampling_bounds_match_main_including_both_endpoints(self):
        scalar_tasks = [task for task in MATH_TASKS if task.input_type == "scalar"]
        self.assertEqual({task.__name__ for task in scalar_tasks}, set(SCALAR_INPUT_RANGES))
        for implementation in scalar_tasks:
            minimum, maximum = SCALAR_INPUT_RANGES[implementation.__name__]
            for endpoint in (minimum, maximum):
                with self.subTest(task=implementation.task_name, endpoint=endpoint), \
                        patch.object(implementation, "solution_generator", return_value=2), \
                        patch("tasks.math_task.math_task.random.randint", return_value=endpoint) as draw:
                    bank = implementation.generate_inputs(1, ["task_1_input"], 100, 10, 1)
                    draw.assert_called_once_with(minimum, maximum)
                    self.assertEqual(bank[0].input_value, endpoint)

    def test_transform_inputs_generate_values_for_all_sixteen_elements(self):
        values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        for implementation in MATH_TASKS:
            if implementation.task_name not in POWER_OF_TWO_TASKS:
                continue
            with self.subTest(task=implementation.task_name), \
                    patch("tasks.math_task.math_task.random.choice", return_value=16), \
                    patch("tasks.math_task.math_task.random.randint", side_effect=values) as draw:
                bank = implementation.generate_inputs(1, ["task_1_input"], 100, 10, 1)
            self.assertEqual(bank[0].input_value, values)
            self.assertEqual(len(bank[0].out), 16)
            self.assertEqual(draw.call_count, 16)
            self.assertTrue(all(call.args == (-99, 99) for call in draw.call_args_list))

    def test_pairwise_gcd_inputs_use_absolute_value_domain(self):
        implementation = next(task for task in MATH_TASKS if task.task_name == "pairwise_gcd_sum")
        with patch("tasks.math_task.math_task.random.choice", return_value=2), \
                patch("tasks.math_task.math_task.random.randint", side_effect=[0, 99]) as draw:
            bank = implementation.generate_inputs(1, ["task_1_input"], 100, 10, 1)
        self.assertEqual(bank[0].input_value, [0, 99])
        self.assertTrue(all(call.args == (0, 99) for call in draw.call_args_list))

    def test_invalid_degenerate_and_duplicate_answers_are_rejected(self):
        with patch("tasks.math_task.math_task.random.randint", side_effect=[-2, 0, 2, 4, 5, 6]):
            bank = RejectionTask.generate_inputs(1, ["task_1_input"], 100, 10, 2)
        self.assertEqual([instance.input_value for instance in bank], [4, 6])
        self.assertEqual([instance.out for instance in bank], [2, 3])

    def test_sampling_stops_when_distinct_answers_cannot_be_found(self):
        with patch("tasks.math_task.math_task.MAX_BANK_DRAWS", 7), \
                patch("tasks.math_task.math_task.random.randint", return_value=4) as draw:
            with self.assertRaisesRegex(ValueError, "Could not generate 2 distinct"):
                RejectionTask.generate_inputs(1, ["task_1_input"], 100, 10, 2)
        self.assertEqual(draw.call_count, 8)

    def test_all_math_tasks_preserve_static_data_and_solve_every_bank_entry(self):
        lengths = set()
        for index, implementation in enumerate(MATH_TASKS):
            with self.subTest(task=implementation.task_name):
                random.seed(index)
                # A task instance can have too few distinct answers at its sampled length.
                for _ in range(200):
                    try:
                        bank = implementation.generate_inputs(1, ["task_1_input"], 100, 10, 6)
                        break
                    except ValueError:
                        continue
                else:
                    self.fail(f"Could not construct a bank for {implementation.task_name}")
                self.assertEqual(len(bank), 6)
                first = bank[0]
                original_first = deepcopy((first.input_value, first.out, first.prompt))
                answers = []
                for instance in bank:
                    value = instance.input_value
                    if implementation.input_type == "list":
                        lengths.add(len(value))
                        self.assertEqual(len(value), len(first.input_value))
                        minimum = 0 if implementation.task_name == "pairwise_gcd_sum" else -99
                        self.assertTrue(all(type(element) is int and minimum <= element <= 99 for element in value))
                        if implementation.task_name in POWER_OF_TWO_TASKS:
                            self.assertTrue(1 <= len(value) <= 16)
                            self.assertEqual(len(value) & (len(value) - 1), 0)
                            self.assertEqual(len(instance.out), len(value))
                        else:
                            self.assertTrue(1 <= len(value) <= 10)
                    else:
                        minimum, maximum = SCALAR_INPUT_RANGES[implementation.__name__]
                        self.assertIs(type(value), int)
                        self.assertTrue(minimum <= value <= maximum)
                    self.assertFalse(bad_output(instance.out))
                    self.assertFalse(any(grade_value(instance.out, answer) for answer in answers))
                    answers.append(instance.out)
                    self.assertEqual(instance.prompt, first.prompt)
                    self.assertEqual(instance.prompt_generator(), first.prompt_generator())
                    for name in ("matrix_2", "seq2", "eval_point", "prime"):
                        if hasattr(first, name):
                            self.assertEqual(getattr(instance, name), getattr(first, name))
                    replay = deepcopy(instance)
                    self.assertTrue(grade_value(plain_value(replay.solution_generator()), instance.out))
                    json.dumps({"input": value, "output": instance.out}, allow_nan=False)
                self.assertEqual((first.input_value, first.out, first.prompt), original_first)
        self.assertGreater(len(lengths), 1)

    def test_generated_banks_work_with_tool_arguments_and_configured_attempt_count(self):
        task = next(item for item in load_pool() if item["task_name"] == "divisor_sigma_square")
        graph = {
            "directed": True, "multigraph": False,
            "target": {"id": "join", "native_task_id": task["task_id"]},
            "nodes": [{"id": name, "category": "math"} for name in ("a", "b", "join", "child")],
            "edges": [
                {"source": "b", "destination": "join"},
                {"source": "a", "destination": "join"},
                {"source": "join", "destination": "child"},
            ],
        }
        with patch("fill_dag.load_math_pool", return_value=[task]), \
                patch.object(InputTool, "__init__", side_effect=AssertionError("Runtime tool used in generation")):
            dataset = generate_dataset(graph, 1, seed=6, get_input_attempts_per_task=2)
            self.assertEqual(dataset, generate_dataset(graph, 1, seed=6, get_input_attempts_per_task=2))
        sample = json.loads(json.dumps(dataset))["samples"][0]
        records = sample["node_outputs"]
        self.assertEqual([len(record["wrong_inputs"]) for record in records], [0, 0, 2, 2])
        self.assertIn("input_dependencies=[task_2_out, task_1_out]", sample["prompt"])
        self.assertIn("input_dependencies=[task_3_out]", sample["prompt"])
        self.assertNotIn("wrong_inputs", sample["prompt"])
        self.assertNotIn("get_input_attempts_per_task", dataset["summary"])
        tool = InputTool.from_records(records, attempts_per_task=2)
        self.assertEqual(
            tool.get_input(task_index=3, input_dependencies=[{"result": None}, {"result": None}]),
            records[2]["wrong_inputs"][0]["input"],
        )
        self.assertEqual(
            tool.get_input(task_index=4, input_dependencies=[{"result": records[2]["wrong_inputs"][0]["output"]}]),
            records[3]["wrong_inputs"][0]["input"],
        )
        self.assertEqual(
            tool.get_input(task_index=3, input_dependencies=[
                {"result": records[1]["output_value"]}, {"result": records[0]["output_value"]},
            ]),
            records[2]["input_value"],
        )
        self.assertEqual(
            tool.get_input(task_index=4, input_dependencies=[{"result": records[2]["output_value"]}]),
            records[3]["input_value"],
        )
        self.assertEqual(tool.get_input(task_index=4, input_dependencies=[{"result": 0}]),
                         "Error: input request limit reached for this task.")


if __name__ == "__main__":
    unittest.main()
