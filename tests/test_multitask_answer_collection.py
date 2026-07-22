import unittest
from types import SimpleNamespace

import run_agent_calc
from grader import grade
from tasks.forgetting_dataset import renumber_prompt


class MultiTaskAnswerCollectionTest(unittest.TestCase):
    def test_expected_outputs_are_unique_and_ignore_internal_names(self):
        prompt = (
            "Task 1 produces task_1_out. "
            "Compute mat_task_2_out, then convert it to task_2_out. "
            "Mention task_1_out again."
        )
        self.assertEqual(
            run_agent_calc.expected_task_outputs(prompt),
            ["task_1_out", "task_2_out"],
        )

    def test_task_indices_are_unpadded_and_support_multiple_digits(self):
        prompt = (
            "Task 1 produces task_1_out. "
            "Task 12 produces task_12_out. "
            "The padded name task_02_out is invalid."
        )
        self.assertEqual(
            run_agent_calc.expected_task_outputs(prompt),
            ["task_1_out", "task_12_out"],
        )
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "\\boxed{task_02_out = 2}",
                ["task_2_out"],
            ),
            [],
        )

    def test_labeled_boxes_are_collected_and_nested_braces_are_preserved(self):
        response = (
            "\\boxed{task_1_out = 10}\n"
            "\\boxed{task_2_out = [1, {2, 3}]}"
        )
        self.assertEqual(
            run_agent_calc.answers_from_response(
                response,
                ["task_1_out", "task_2_out"],
            ),
            [
                ("task_1_out", "10"),
                ("task_2_out", "[1, {2, 3}]"),
            ],
        )

    def test_text_wrapped_label_inside_box_is_accepted(self):
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "\\boxed{\\text{task_1_out} = 42}",
                ["task_1_out"],
            ),
            [("task_1_out", "42")],
        )

    def test_label_outside_box_is_rejected(self):
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "task_1_out = \\boxed{42}",
                ["task_1_out"],
            ),
            [],
        )

    def test_unlabeled_boxes_are_rejected_even_in_complete_order(self):
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "\\boxed{10}\n\\boxed{20}",
                ["task_1_out", "task_2_out"],
            ),
            [],
        )

    def test_unboxed_assignment_is_rejected(self):
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "task_1_out = 10",
                ["task_1_out"],
            ),
            [],
        )

    def test_heading_outside_box_does_not_assign_box(self):
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "Task 1 answer: \\boxed{10}",
                ["task_1_out"],
            ),
            [],
        )

    def test_empty_and_placeholder_boxes_are_rejected(self):
        expected = ["task_1_out"]
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "\\boxed{task_1_out = }",
                expected,
            ),
            [],
        )
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "\\boxed{task_1_out = pending}",
                expected,
            ),
            [],
        )

    def test_unexpected_task_label_is_rejected(self):
        self.assertEqual(
            run_agent_calc.answers_from_response(
                "\\boxed{task_2_out = 20}",
                ["task_1_out"],
            ),
            [],
        )

    def test_tool_payload_is_not_mistaken_for_an_answer(self):
        response = (
            '<tool_call>{"name":"calculator","arguments":'
            '{"expressions":["\\\\boxed{task_1_out = 42}"]}}</tool_call>'
        )
        self.assertEqual(
            run_agent_calc.answers_from_response(response, ["task_1_out"]),
            [],
        )

    def test_multiple_labeled_boxes_on_one_line_are_collected(self):
        response = (
            "\\boxed{task_1_out = 10} "
            "\\boxed{task_2_out = [20, 21]}"
        )
        self.assertEqual(
            run_agent_calc.answers_from_response(
                response,
                ["task_1_out", "task_2_out"],
            ),
            [
                ("task_1_out", "10"),
                ("task_2_out", "[20, 21]"),
            ],
        )

    def test_latest_box_in_same_response_wins(self):
        answers = {}
        run_agent_calc.update_task_answers(
            answers,
            (
                "\\boxed{task_1_out = 10}\n"
                "Correction: \\boxed{task_1_out = 11}"
            ),
            ["task_1_out"],
            turn=0,
        )
        self.assertEqual(
            answers["task_1_out"],
            {"answer": "11", "turn": 0},
        )

    def test_latest_answer_replaces_previous_turn(self):
        answers = {}
        expected = ["task_1_out", "task_2_out"]
        run_agent_calc.update_task_answers(
            answers,
            "\\boxed{task_1_out = 10}",
            expected,
            turn=0,
        )
        run_agent_calc.update_task_answers(
            answers,
            "Correction: \\boxed{task_1_out = 11}",
            expected,
            turn=2,
        )
        self.assertEqual(
            answers["task_1_out"],
            {"answer": "11", "turn": 2},
        )

    def test_repair_prompt_lists_missing_and_requires_in_box_label(self):
        prompt = run_agent_calc.build_final_answer_repair_prompt(
            ["task_1_out", "task_2_out", "task_3_out"],
            {"task_1_out": {"answer": "10", "turn": 0}},
        )
        self.assertIn("task_2_out, task_3_out", prompt)
        self.assertIn("corrected version", prompt)
        self.assertIn("latest version", prompt)
        self.assertIn("\\boxed{task_1_out = ...}", prompt)
        self.assertIn("Labels outside boxes do not count", prompt)

    def test_grader_accepts_labeled_scalar_and_list_boxes(self):
        self.assertTrue(grade("\\boxed{task_1_out = 42}", 42))
        self.assertTrue(grade("\\boxed{task_1_out = 42.0}", 42))
        self.assertTrue(
            grade(
                "\\boxed{task_1_out = [1, 2.0, 3]}",
                [1, 2, 3],
            )
        )

    def test_grader_rejects_fractional_values_for_integer_ground_truths(self):
        self.assertFalse(grade("\\boxed{task_1_out = 42.9}", 42))
        self.assertFalse(grade("\\boxed{task_1_out = -3.8}", -3))
        self.assertFalse(
            grade(
                "\\boxed{task_1_out = [1.9, 2.0]}",
                [1, 2],
            )
        )

    def test_forgetting_renumbering_preserves_mathematical_subscripts(self):
        prompt = (
            "Task 1:\n"
            "Let list_1 = [H_0, H_1]. "
            "Compute P_1 and save task_1_out."
        )
        self.assertEqual(
            renumber_prompt(prompt, 12),
            (
                "Task 12:\n"
                "Let list_12 = [H_0, H_1]. "
                "Compute P_12 and save task_12_out."
            ),
        )


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append([dict(message) for message in messages])
        return "rendered prompt"

    def encode(self, text, add_special_tokens=False):
        return list(range(len((text or "").split())))


class _FakeEngine:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.max_tokens_seen = []

    async def generate(self, prompt, sampling_params, request_id):
        self.max_tokens_seen.append(sampling_params.max_tokens)
        text = next(self.responses)
        yield SimpleNamespace(outputs=[SimpleNamespace(text=text)])


class MultiTaskAgentLoopTest(unittest.IsolatedAsyncioTestCase):
    async def test_loop_repairs_cutoff_with_single_cutoff_prompt(self):
        engine = _FakeEngine([
            "reasoning stopped here",
            "\\boxed{task_1_out = 4}",
        ])
        tokenizer = _FakeTokenizer()

        result = await run_agent_calc.run_agent_loop(
            engine=engine,
            tokenizer=tokenizer,
            sampling_params=SimpleNamespace(max_tokens=3),
            system_prompt="system",
            task_prompt="Task 1: save task_1_out.",
            max_agent_turns=3,
            index=10,
            repair_sampling_params=SimpleNamespace(max_tokens=6),
            cutoff_repair_attempts=1,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(engine.max_tokens_seen, [3, 6])
        self.assertTrue(
            any(
                run_agent_calc.CUTOFF_REPAIR_PROMPT in message["content"]
                for call in tokenizer.calls
                for message in call
                if message["role"] == "user"
            )
        )

    async def test_loop_repairs_missing_outputs_and_keeps_latest_answers(self):
        task_prompt = (
            "Task 1: save task_1_out.\n"
            "Task 2: save task_2_out.\n"
            "Task 3: save task_3_out.\n"
        )
        engine = _FakeEngine([
            (
                "\\boxed{task_1_out = 10}\n"
                '<tool_call>{"name":"calculator","arguments":{"expressions":["1+1"]}}</tool_call>'
            ),
            "\\boxed{task_2_out = 20}",
            (
                "Correction: \\boxed{task_1_out = 11}\n"
                "\\boxed{task_3_out = 30}"
            ),
        ])
        tokenizer = _FakeTokenizer()

        result = await run_agent_calc.run_agent_loop(
            engine=engine,
            tokenizer=tokenizer,
            sampling_params=SimpleNamespace(max_tokens=1000),
            system_prompt="system",
            task_prompt=task_prompt,
            max_agent_turns=5,
            index=0,
            final_answer_repair_attempts=2,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["turns_taken"], 3)
        self.assertEqual(result["total_tool_calls"], 1)
        self.assertEqual(
            result["task_answers"],
            {
                "task_1_out": {"answer": "11", "turn": 2},
                "task_2_out": {"answer": "20", "turn": 1},
                "task_3_out": {"answer": "30", "turn": 2},
            },
        )
        target_text = run_agent_calc.target_output_text(
            result["expected_task_outputs"],
            result["task_answers"],
        )
        self.assertEqual(target_text, "\\boxed{task_3_out = 30}")
        self.assertTrue(grade(target_text, 30))

        repair_messages = [
            message["content"]
            for call in tokenizer.calls
            for message in call
            if message["role"] == "user"
            and "required" in message["content"]
        ]
        self.assertEqual(len(repair_messages), 1)
        self.assertIn("task_3_out", repair_messages[0])
        self.assertNotIn(
            "outputs still missing are: task_1_out",
            repair_messages[0],
        )

    async def test_outside_box_label_is_repaired(self):
        task_prompt = "Task 1: save task_1_out.\n"
        engine = _FakeEngine([
            "task_1_out = \\boxed{100}",
            "\\boxed{task_1_out = 100}",
        ])
        tokenizer = _FakeTokenizer()

        result = await run_agent_calc.run_agent_loop(
            engine=engine,
            tokenizer=tokenizer,
            sampling_params=SimpleNamespace(max_tokens=1000),
            system_prompt="system",
            task_prompt=task_prompt,
            max_agent_turns=3,
            index=1,
            final_answer_repair_attempts=1,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["turns_taken"], 2)
        self.assertEqual(
            result["task_answers"],
            {"task_1_out": {"answer": "100", "turn": 1}},
        )
        self.assertTrue(
            any(
                "\\boxed{task_1_out = ...}" in message["content"]
                for call in tokenizer.calls
                for message in call
                if message["role"] == "user"
            )
        )

    async def test_target_from_earlier_turn_survives_distractor_repair(self):
        task_prompt = (
            "Task 1: save task_1_out.\n"
            "Task 2: save task_2_out.\n"
        )
        engine = _FakeEngine([
            (
                "\\boxed{task_2_out = 200}\n"
                '<tool_call>{"name":"calculator","arguments":{"expressions":["100+100"]}}</tool_call>'
            ),
            "I am done.",
            "\\boxed{task_1_out = 100}",
        ])

        result = await run_agent_calc.run_agent_loop(
            engine=engine,
            tokenizer=_FakeTokenizer(),
            sampling_params=SimpleNamespace(max_tokens=1000),
            system_prompt="system",
            task_prompt=task_prompt,
            max_agent_turns=5,
            index=2,
            final_answer_repair_attempts=2,
        )

        self.assertEqual(result["status"], "success")
        target_text = run_agent_calc.target_output_text(
            result["expected_task_outputs"],
            result["task_answers"],
        )
        self.assertEqual(target_text, "\\boxed{task_2_out = 200}")
        self.assertTrue(grade(target_text, 200))


if __name__ == "__main__":
    unittest.main()
