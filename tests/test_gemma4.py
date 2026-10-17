"""Gemma 4 support: its tool-call format, its turn structure, and its chat template."""

from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import run_agent
import run_agent_api
import util
from parse_results import (
    GEMMA_OPEN_THOUGHT,
    answers_from_response,
    assistant_message_from_response,
    looks_like_tool_call,
    parse_tool_calls,
)
from tasks.arbigraph_tools import GET_INPUT_TOOL, SUBMIT_ANSWERS_TOOL, InputTool

Q = '<|"|>'
MODEL = "google/gemma-4-31B-it"


def gemma_call(name, arguments):
    return f"<|tool_call>call:{name}{{{arguments}}}<tool_call|>"


def records():
    return [
        {"node_id": "task_7", "native_task_name": "root", "parent_ids": [],
         "input_value": 3, "output_value": 6, "wrong_inputs": []},
        {"node_id": "task_2", "native_task_name": "single", "parent_ids": ["task_7"],
         "input_value": [10, 1], "output_value": 20,
         "wrong_inputs": [{"input": [n, 1], "output": 2 * n} for n in (11, 12)]},
        {"node_id": "task_20", "native_task_name": "join", "parent_ids": ["task_2", "task_7"],
         "input_value": [[30, 1]], "output_value": 60,
         "wrong_inputs": [{"input": [[n, 1]], "output": 2 * n} for n in (31, 32)]},
    ]


def gemma_tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(MODEL)
    except (OSError, ValueError):
        raise unittest.SkipTest("Gemma 4 tokenizer is not available")


def tool_schemas():
    return [{"type": "function", "function": {
        "name": tool["name"], "description": tool["description"], "parameters": tool["inputSchema"],
    }} for tool in (GET_INPUT_TOOL, SUBMIT_ANSWERS_TOOL)]


class GemmaToolCallTests(unittest.TestCase):
    def test_nested_arguments_keep_their_types(self):
        text = gemma_call("get_input", "input_dependencies:[{result:[3,-1,4.5,1e-3]},{result:true}],task_index:2")
        self.assertEqual(parse_tool_calls(text), [{"name": "get_input", "arguments": {
            "input_dependencies": [{"result": [3, -1, 4.5, 0.001]}, {"result": True}], "task_index": 2,
        }}])

    def test_strings_objects_and_null(self):
        text = gemma_call("submit_answers", f"answers:[{{result:{{answer:{Q}Paris, France{Q},sp:[[{Q}A{Q},0]],none:null}}}}]")
        self.assertEqual(parse_tool_calls(text)[0]["arguments"], {
            "answers": [{"result": {"answer": "Paris, France", "sp": [["A", 0]], "none": None}}]})

    def test_calls_keep_their_order_and_may_close_at_a_stop_token(self):
        text = ("<|channel>thought\nTwo calls.<channel|>"
                + gemma_call("calculator", f"expressions:[{Q}17 * 43{Q},{Q}pow(3, 5, 7){Q}]")
                + "<|tool_call>call:get_input{task_index:1,input_dependencies:[]}<|tool_response>")
        self.assertEqual([call["name"] for call in parse_tool_calls(text)], ["calculator", "get_input"])
        self.assertEqual(parse_tool_calls(text)[0]["arguments"], {"expressions": ["17 * 43", "pow(3, 5, 7)"]})

    def test_quoted_keys_are_read_like_bare_ones(self):
        text = gemma_call("get_input", f"{Q}task_index{Q}:1,input_dependencies:[]")
        self.assertEqual(parse_tool_calls(text)[0]["arguments"], {"task_index": 1, "input_dependencies": []})

    def test_a_malformed_call_is_not_run_but_is_recognised(self):
        for text in (gemma_call("get_input", f"task_index:2,input_dependencies:[{{result:{Q}x}}]"),
                     "<|tool_call>call:get_input{task_index:2,input_dep"):
            with self.subTest(text=text):
                self.assertEqual(parse_tool_calls(text), [])
                self.assertTrue(looks_like_tool_call(text))

    def test_answers_inside_a_call_are_not_read_as_text_answers(self):
        text = gemma_call("calculator", f"expressions:[{Q}task_1_out = {{\"result\": 5}}{Q}]") + "<turn|>"
        self.assertEqual(answers_from_response(text, ["task_1_out"]), [])

    def test_other_formats_are_unaffected(self):
        qwen = '<tool_call>{"name": "get_input", "arguments": {"task_index": 1}}</tool_call>'
        self.assertEqual(parse_tool_calls(qwen), [{"name": "get_input", "arguments": {"task_index": 1}}])
        self.assertEqual(assistant_message_from_response(qwen, parse_tool_calls(qwen), ["c"]),
                         {"role": "assistant", "content": qwen})


class GemmaTurnTests(unittest.TestCase):
    def test_a_call_turn_becomes_reasoning_and_structured_calls(self):
        text = ("<|channel>thought\nNeed task 1.<channel|>"
                + gemma_call("get_input", "input_dependencies:[],task_index:1") + "<|tool_response>")
        message = assistant_message_from_response(text, parse_tool_calls(text), ["call_0_0"])
        self.assertEqual(message, {
            "role": "assistant",
            "reasoning": "Need task 1.",
            "tool_calls": [{"id": "call_0_0", "type": "function", "function": {
                "name": "get_input", "arguments": json.dumps({"input_dependencies": [], "task_index": 1}),
            }}],
        })

    def test_a_turn_that_continues_an_opened_thought(self):
        # After a tool reply the prompt itself opens the thought channel.
        text = "Task 1 is 6.<channel|>" + gemma_call("get_input", "input_dependencies:[{result:6}],task_index:2")
        message = assistant_message_from_response(text, parse_tool_calls(text), ["call_1_0"])
        self.assertEqual(message["reasoning"], "Task 1 is 6.")
        self.assertNotIn("content", message)

    def test_a_final_answer_keeps_reasoning_and_visible_text(self):
        text = '<|channel>thought\nDone.<channel|>task_3_out = {"result": 60}<turn|>'
        self.assertEqual(assistant_message_from_response(text, []),
                         {"role": "assistant", "reasoning": "Done.", "content": 'task_3_out = {"result": 60}'})

    def test_a_turn_cut_off_mid_thought_stays_reasoning(self):
        text = GEMMA_OPEN_THOUGHT + "still working on the conv"
        self.assertEqual(assistant_message_from_response(text, []),
                         {"role": "assistant", "reasoning": "still working on the conv"})

    def test_a_malformed_call_stays_in_view_for_the_repair_prompt(self):
        broken = gemma_call("get_input", f"task_index:2,input_dependencies:[{{result:{Q}x}}]")
        self.assertEqual(assistant_message_from_response(broken, []), {"role": "assistant", "content": broken})

    def test_model_detection(self):
        self.assertTrue(util.is_gemma4_model(MODEL))
        self.assertTrue(util.keeps_special_tokens(MODEL))
        self.assertTrue(util.keeps_special_tokens("openai/gpt-oss-120b"))
        self.assertFalse(util.keeps_special_tokens("Qwen/Qwen3.8-27B"))
        self.assertFalse(util.is_gemma4_model("google/gemma-3-27b-it"))


class GemmaTemplateTests(unittest.IsolatedAsyncioTestCase):
    def test_structured_history_renders_tool_results_and_reasoning(self):
        tok = gemma_tokenizer()
        first = ("<|channel>thought\nNeed the input.<channel|>"
                 + gemma_call("get_input", "input_dependencies:[],task_index:1") + "<|tool_response>")
        messages = [
            {"role": "system", "content": util.SYSTEM_PROMPT},
            {"role": "user", "content": "Solve task 1."},
            assistant_message_from_response(first, parse_tool_calls(first), ["call_0_0"]),
            {"role": "tool", "name": "get_input", "tool_call_id": "call_0_0", "content": "3"},
        ]
        text, token_ids = util.render_prompt(tok, messages, tool_schemas(), {"enable_thinking": True})
        self.assertIsNone(token_ids)
        self.assertTrue(text.startswith("<bos><|turn>system\n<|think|>\n" + util.SYSTEM_PROMPT.strip()))
        self.assertIn("<|channel>thought\nNeed the input.\n<channel|>", text)
        self.assertIn("<|tool_call>call:get_input{input_dependencies:[],task_index:1}<tool_call|>", text)
        self.assertIn(f"<|tool_response>response:get_input{{value:{Q}3{Q}}}<tool_response|>", text)
        # The template opens the next thought itself, which the loop restores when reading the turn.
        self.assertTrue(text.endswith(GEMMA_OPEN_THOUGHT))

    async def test_an_episode_in_gemma_format_runs_end_to_end(self):
        tok = gemma_tokenizer()
        turns = [
            "<|channel>thought\nFetch task 1.<channel|>" + gemma_call("get_input", "input_dependencies:[],task_index:1"),
            "Task 1 gives 6.<channel|>" + gemma_call("get_input", "input_dependencies:[{result:6}],task_index:2"),
            "Task 2 gives 20.<channel|>"
            + gemma_call("get_input", "input_dependencies:[{result:20},{result:6}],task_index:3"),
            "Task 3 is 60.<channel|>" + gemma_call("submit_answers", "answers:[{result:60}]"),
        ]
        prompts = []

        async def generate(prompt, *args, messages, **kwargs):
            prompts.append(prompt)
            return turns[len(prompts) - 1] + "<|tool_response>", "stop"

        result = await util.run_agent_loop(
            generate, tok, SimpleNamespace(max_tokens=1024), tool_schemas(), None,
            util.SYSTEM_PROMPT, "Solve task 3.", 10, 0,
            expected_outputs=["task_1_out", "task_2_out", "task_3_out"], graded_outputs=["task_3_out"],
            input_tool=InputTool.from_records(records(), attempts_per_task=2),
            chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["submitted_answers"], {"task_3_out": {"answer": 60, "turn": 3}})
        self.assertEqual(result["total_tool_calls"], 3)
        # Each tool result reached the next prompt, and so did the reasoning that asked for it.
        self.assertIn(f"<|tool_response>response:get_input{{value:{Q}3{Q}}}<tool_response|>", prompts[1])
        self.assertIn(f"<|tool_response>response:get_input{{value:{Q}[10, 1]{Q}}}<tool_response|>", prompts[2])
        self.assertIn("<|channel>thought\nTask 1 gives 6.\n<channel|>", prompts[2])
        self.assertTrue(all(prompt.endswith(GEMMA_OPEN_THOUGHT) for prompt in prompts[1:]))


class GemmaSettingsTests(unittest.TestCase):
    def invoke(self, runner, argv):
        with patch.object(sys, "argv", [runner.__name__, *argv]), \
             patch.object(runner.asyncio, "run") as inference, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                runner.main()
            finally:
                inference.assert_not_called()

    def test_thinking_is_recorded_and_a_run_never_resumes_without_it(self):
        for runner in (run_agent, run_agent_api):
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "dataset.json"
                output = Path(directory) / "results.json"
                source.write_text(json.dumps({"summary": {}, "samples": []}))
                argv = ["--input", str(source), "--output", str(output), "--enable_thinking"]
                if runner is run_agent_api:
                    argv += ["--api_model", "test-model"]
                self.invoke(runner, argv)
                params = json.loads(output.read_text())["summary"]["generation_params"]
                self.assertIs(params["enable_thinking"], True)
                self.invoke(runner, argv)
                with self.assertRaises(SystemExit) as error:
                    self.invoke(runner, [arg for arg in argv if arg != "--enable_thinking"])
                self.assertEqual(error.exception.code, 1)


class GemmaThinkingTests(unittest.IsolatedAsyncioTestCase):
    async def test_thinking_reaches_the_template_only_when_set(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.list_tools.return_value = SimpleNamespace(tools=[SimpleNamespace(
            name="calculator", description="Calculate", inputSchema={"type": "object"},
        )])

        @asynccontextmanager
        async def stdio(*args):
            yield None, None

        seen = []

        async def loop(*args, chat_template_kwargs, **kwargs):
            seen.append(copy.deepcopy(chat_template_kwargs))
            return {"status": "success", "turns_taken": 0, "total_tool_calls": 0, "turn_outputs": {},
                    "task_answers": {}, "submitted_answers": None, "reported_correct": {},
                    "resume_state": None}

        for enable_thinking in (True, None):
            args = SimpleNamespace(
                model=MODEL, reasoning_effort="unused", preserve_thinking=True,
                enable_thinking=enable_thinking, example=False, max_agent_turns=10, max_model_len=None,
                tool_repair_attempts=0, final_answer_repair_attempts=0, cutoff_repair_attempts=0,
                max_turns_without_new_answer=20, max_concurrent_samples=1, get_input_attempts_per_task=2,
            )
            data = {"samples": [{"prompt": "Tasks", "target_node_id": "task_20", "node_outputs": records()}]}
            with tempfile.TemporaryDirectory() as directory, \
                 patch("mcp.ClientSession", return_value=client), \
                 patch("mcp.client.stdio.stdio_client", stdio), \
                 patch.object(util, "run_agent_loop", loop), redirect_stderr(io.StringIO()):
                await util.process_samples(
                    args, data, [0], directory, [], tokenizer=Mock(), generate_text=None,
                    sampling_params=SimpleNamespace(max_tokens=1024), repair_sampling_params=None,
                    system_prompt="System",
                )
        self.assertEqual(seen[0], {"reasoning_effort": "unused", "preserve_thinking": True, "enable_thinking": True})
        # Unset, the template keeps its own default, as it did before the flag existed.
        self.assertEqual(seen[1], {"reasoning_effort": "unused", "preserve_thinking": True})


if __name__ == "__main__":
    unittest.main()
