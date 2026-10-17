"""The input gate answers a misshapen request the same way it answers a wrong one."""

import json
from pathlib import Path
import sys
import unittest

SOURCE_ROOT = str(Path(__file__).resolve().parents[1] / "source")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from tasks.arbigraph_tools import InputTool


def records():
    return {
        1: {"parents": [], "input": 3, "output": 6, "wrong_inputs": []},
        2: {"parents": [1], "input": 10, "output": 20,
            "wrong_inputs": [{"input": n, "output": 2 * n} for n in range(11, 16)]},
        3: {"parents": [1, 2], "input": 30, "output": 60,
            "wrong_inputs": [{"input": n, "output": 2 * n} for n in range(31, 36)]},
    }


class InputToolTest(unittest.TestCase):
    def test_answers_alone_determine_input_and_corrections_can_recover(self):
        tool = InputTool(records())
        self.assertEqual(tool.get_input(3, [{"result": 6}, {"result": 20}]), 30)
        wrong = tool.get_input(2, [{"result": 7}])
        self.assertEqual(wrong, 11)
        self.assertEqual(tool.get_input(3, [{"result": 6}, {"result": 2 * wrong}]), 32)
        self.assertEqual(tool.get_input(2, [{"result": 6}]), 10)
        self.assertEqual(tool.get_input(3, [{"result": 6}, {"result": 20}]), 30)

    def test_order_and_single_parent_envelope(self):
        tool = InputTool(records())
        self.assertEqual(tool.get_input(3, [{"result": 20}, {"result": 6}]), 31)
        self.assertEqual(tool.get_input(2, {"result": 6}), 11)
        self.assertEqual(tool.counts[2], 1)

    def test_a_misshapen_request_is_wrong_rather_than_interpreted(self):
        """Every way of failing to state the value is answered from the bank."""
        misshapen = (
            None,                        # no dependencies given at all
            [],                          # empty list where one value is due
            [6],                         # the value, without its envelope
            [{"wrong_key": 6}],          # an envelope naming the wrong key
            [{"result": 6, "note": 1}],  # the right value, with something extra
            '[{"result": 6}]',           # the list left as a string
        )
        for values in misshapen:
            with self.subTest(values=values):
                tool = InputTool(records())
                self.assertEqual(tool.get_input(2, values), 11)
                self.assertEqual(tool.counts[2], 1)

    def test_a_root_has_nothing_to_state_so_its_shape_cannot_be_wrong(self):
        tool = InputTool(records())
        self.assertEqual(tool.get_input(1, []), 3)
        self.assertEqual(tool.get_input(1, None), 3)
        self.assertEqual(tool.counts[1], 2)

    def test_valid_requests_exhaust_separately_and_never_reuse_bank_entries(self):
        tool = InputTool(records())
        self.assertEqual(tool.get_input(2, [{"result": 6}]), 10)
        for n in range(12, 16):
            self.assertEqual(tool.get_input(2, [{"result": -1}]), n)
        self.assertEqual(tool.get_input(2, [{"result": 6}]),
                         "Error: input request limit reached for this task.")
        self.assertEqual(tool.counts[2], 5)
        self.assertEqual(tool.get_input(3, [{"result": 6}, {"result": 20}]), 30)

    def test_an_unidentifiable_task_is_the_only_free_mistake(self):
        """Nothing can be charged when the request names no task to charge."""
        tool = InputTool(records())
        self.assertEqual(tool.get_input(), "Error: unknown task index.")
        self.assertEqual(tool.get_input("2", [{"result": 6}]), "Error: unknown task index.")
        self.assertEqual(tool.counts, {1: 0, 2: 0, 3: 0})
        # A correctly named task is charged even when the rest of the call is not.
        self.assertEqual(tool.get_input(2, "not a list"), 11)
        self.assertEqual(tool.counts[2], 1)

    def test_json_round_trip_and_fresh_attempt(self):
        data = json.loads(json.dumps(records()))
        tool = InputTool(data)
        self.assertEqual(tool.get_input(2, [{"result": 0}]), 11)
        self.assertEqual(InputTool(data).get_input(2, [{"result": 0}]), 11)

    def test_native_matrices_and_defensive_copies(self):
        data = records()
        data[1]["output"] = [[1, 2], [3, 4]]
        data[2]["input"] = [[8, 9]]
        tool = InputTool(data)
        actual = tool.get_input(2, [{"result": [[1, 2], [3, 4]]}])
        actual[0][0] = 0
        data[2]["input"][0][0] = -1
        self.assertEqual(tool.get_input(2, [{"result": [[1, 2], [3, 4]]}]), [[8, 9]])
        self.assertEqual(tool.get_input(2, [{"result": [[1, 2], [3, 5]]}]), 13)

    def test_invalid_limits_and_banks_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                InputTool(records(), attempts_per_task=value)
        for output in (20, 22):
            data = records()
            data[2]["wrong_inputs"][1]["output"] = output
            with self.assertRaises(ValueError):
                InputTool(data)


if __name__ == "__main__":
    unittest.main()
