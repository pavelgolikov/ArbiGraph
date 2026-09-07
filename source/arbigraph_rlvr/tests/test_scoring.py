import unittest

from arbigraph_rlvr.schema import VerifierState
from arbigraph_rlvr.scoring import score_completion, terminal_reward


def verifier(expected, target="task_2_out"):
    outputs = tuple(expected)
    node_ids = {
        output: f"node_{position}"
        for position, output in enumerate(outputs, start=1)
    }
    return VerifierState(
        expected,
        target,
        node_ids,
    )


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.state = verifier({"task_1_out": 7, "task_2_out": [2, 3]})

    def test_correct_completion_receives_terminal_reward_one(self):
        completion = (
            'task_1_out = {"result": 7}\n'
            'task_2_out = {"result": [2, 3]}'
        )
        result = score_completion(completion, self.state)

        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.target_correct)
        self.assertTrue(result.all_nodes_correct)
        self.assertTrue(result.format_valid)
        self.assertEqual(result.node_correct, {"node_1": True, "node_2": True})

    def test_wrong_or_missing_target_receives_zero(self):
        wrong = score_completion(
            'task_1_out = {"result": 7}\ntask_2_out = {"result": [2, 4]}',
            self.state,
        )
        missing = score_completion('task_1_out = {"result": 7}', self.state)

        self.assertEqual(wrong.reward, 0.0)
        self.assertEqual(missing.reward, 0.0)
        self.assertEqual(missing.missing_outputs, ("task_2_out",))
        self.assertFalse(missing.format_valid)

    def test_latest_complete_assignment_wins(self):
        result = score_completion(
            'task_2_out = {"result": [0]}\n'
            'task_2_out = {"result": [2, 3]}\n'
            'task_1_out = {"result": 7}',
            self.state,
        )

        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.parsed_answers["task_2_out"], [2, 3])

    def test_markdown_is_accepted_and_tool_payload_is_masked(self):
        result = score_completion(
            '<tool_call>task_2_out = {"result": [2, 3]}</tool_call>\n'
            '```json\n'
            'task_1_out = {"result": 7}\n'
            'task_2_out = {"result": [2, 3]}\n'
            '```',
            self.state,
        )

        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.parsed_answers["task_2_out"], [2, 3])

    def test_integral_targets_are_not_rounded(self):
        state = verifier({"task_1_out": 1, "task_2_out": 3}, target="task_2_out")
        self.assertEqual(terminal_reward('task_2_out = {"result": 3.0}', state), 1.0)
        self.assertEqual(terminal_reward('task_2_out = {"result": 3.0001}', state), 0.0)
        self.assertEqual(terminal_reward('task_2_out = {"result": true}', state), 0.0)

    def test_per_node_diagnostics_identify_a_wrong_node(self):
        result = score_completion(
            'task_1_out = {"result": 8}\ntask_2_out = {"result": [2, 3]}',
            self.state,
        )

        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.first_incorrect_node, "node_1")
        self.assertEqual(result.node_correct, {"node_1": False, "node_2": True})
        self.assertEqual(result.node_answered, {"node_1": True, "node_2": True})

if __name__ == "__main__":
    unittest.main()
