"""Exercise eval tool dispatch and resume without model or API calls."""

import asyncio
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
from parse_results import assistant_message_from_response, looks_like_tool_call, parse_tool_calls
from tasks.arbigraph_tools import GET_INPUT_TOOL, InputTool


def records(offset=0):
    return [
        {"node_id": "task_7", "native_task_name": "root", "parent_ids": [],
         "input_value": 3 + offset, "output_value": 6 + offset, "wrong_inputs": []},
        {"node_id": "task_2", "native_task_name": "single", "parent_ids": ["task_7"],
         "input_value": [10 + offset, 1], "output_value": 20 + offset,
         "wrong_inputs": [{"input": [n + offset, 1], "output": 2 * n + offset}
                          for n in (11, 12)]},
        {"node_id": "task_20", "native_task_name": "join", "parent_ids": ["task_2", "task_7"],
         "input_value": [[30 + offset, 1]], "output_value": 60 + offset,
         "wrong_inputs": [{"input": [[n + offset, 1]], "output": 2 * n + offset}
                          for n in (31, 32)]},
    ]


def call(name="get_input", arguments=None):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def request(index, *answers):
    return call(arguments={"task_index": index, "input_dependencies": [{"result": a} for a in answers]})


def submit(*results):
    return call("submit_answers", {"answers": [{"result": result} for result in results]})


# The rejection a submission gets when task 3 is graded and its input was never requested.
PREMATURE = ("Error: submission rejected, no answers were recorded. Task 3 is the graded task, "
             "and its input has not been requested yet. Solve task 3, then submit its result.")


def tokenizer():
    result = Mock()
    result.apply_chat_template.return_value = "Rendered prompt"
    result.encode.return_value = [1, 2]
    return result


class EvalInputToolTests(unittest.IsolatedAsyncioTestCase):
    async def run_loop(self, outputs, *, input_tool=None, **kwargs):
        generated = iter(outputs)
        messages_seen = []
        self.generate_kwargs = []

        async def generate(*args, messages, **rest):
            messages_seen.append(copy.deepcopy(messages))
            self.generate_kwargs.append(rest)
            return next(generated), "stop"

        result = await util.run_agent_loop(
            generate, tokenizer(), SimpleNamespace(max_tokens=1024),
            [{"function": {"name": "get_input"}}, {"function": {"name": "submit_answers"}}],
            None, "System", "Tasks", 30, 17,
            expected_outputs=["task_1_out", "task_2_out", "task_3_out"],
            input_tool=input_tool or InputTool.from_records(records(), attempts_per_task=2),
            tool_repair_attempts=0, **kwargs,
        )
        return result, messages_seen[-1]

    async def test_gpt_oss_harmony_call_and_reply_use_structured_history(self):
        first = (
            '<|channel|>analysis<|message|>Need the input.<|end|>'
            '<|start|>assistant<|channel|>commentary to=functions.get_input '
            '<|constrain|>json<|message|>'
            '{"task_index":1,"input_dependencies":[]}<|call|>'
        )
        second = (
            '<|channel|>commentary to=functions.submit_answers'
            '<|message|>{"answers":[{"result":6}]}'
            '<|call|>'
        )
        with patch.object(util, "render_harmony_prompt", return_value=[5, 6, 7]) as render:
            result, messages = await self.run_loop([first, second], harmony=True,
                                                   graded_outputs=["task_1_out"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_tool_calls"], 1)
        # The fields vLLM's Harmony renderer reads: reasoning, and a call id the
        # tool result names.
        self.assertEqual(messages[-2], {
            "role": "assistant",
            "reasoning": "Need the input.",
            "tool_calls": [{"id": "call_0_0", "type": "function", "function": {
                "name": "get_input",
                "arguments": json.dumps({"task_index": 1, "input_dependencies": []}),
            }}],
        })
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "call_0_0")
        self.assertEqual(json.loads(messages[-1]["content"]), 3)
        # Every turn goes to the model as the exact tokens rendered.
        self.assertEqual(render.call_count, 2)
        self.assertTrue(all(kw["prompt_token_ids"] == [5, 6, 7] for kw in self.generate_kwargs))

    async def test_chat_template_models_keep_their_history_unchanged(self):
        result, messages = await self.run_loop([request(1), submit(6)], graded_outputs=["task_1_out"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(messages[-2], {"role": "assistant", "content": request(1)})
        self.assertNotIn("tool_call_id", messages[-1])
        self.assertTrue(all("prompt_token_ids" not in kw for kw in self.generate_kwargs))

    def test_harmony_turn_without_answer_or_call_stays_reasoning(self):
        cut_off = "<|channel|>analysis<|message|>Still working on the conv"
        self.assertEqual(assistant_message_from_response(cut_off, []),
                         {"role": "assistant", "reasoning": "Still working on the conv"})
        answered = ("<|channel|>analysis<|message|>Done.<|end|>"
                    "<|start|>assistant<|channel|>final<|message|>task_1_out = 6<|return|>")
        self.assertEqual(assistant_message_from_response(answered, []),
                         {"role": "assistant", "content": "task_1_out = 6"})

    def test_harmony_render_matches_vllm_conventions(self):
        try:
            util.render_harmony_prompt([{"role": "user", "content": "x"}], None, "high")
        except ImportError:
            self.skipTest("vLLM is not installed")
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
        except OSError:
            self.skipTest("GPT-OSS tokenizer is not available")
        tools = [{"type": "function", "function": {
            "name": GET_INPUT_TOOL["name"], "description": GET_INPUT_TOOL["description"],
            "parameters": GET_INPUT_TOOL["inputSchema"],
        }}]
        first = ('<|channel|>analysis<|message|>Need the input.<|end|>'
                 '<|start|>assistant<|channel|>commentary to=functions.get_input '
                 '<|constrain|>json<|message|>{"task_index":1,"input_dependencies":[]}<|call|>')
        call = parse_tool_calls(first)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Tasks"},
            assistant_message_from_response(first, call, ["call_0_0"]),
            {"role": "tool", "name": "get_input", "tool_call_id": "call_0_0", "content": "3"},
            assistant_message_from_response("<|channel|>analysis<|message|>Cut", []),
        ]
        ids = util.render_harmony_prompt(messages, tools, "high", start_date="2000-01-02")
        text = tok.decode(ids, skip_special_tokens=False)
        self.assertIn("Current date: 2000-01-02", text)
        self.assertIn("Reasoning: high", text)
        self.assertIn("<|channel|>analysis<|message|>Need the input.<|end|>", text)
        self.assertIn("<|start|>assistant to=functions.get_input<|channel|>commentary json<|message|>"
                      '{"task_index": 1, "input_dependencies": []}<|call|>', text)
        self.assertIn("<|start|>functions.get_input to=assistant<|channel|>commentary<|message|>3<|end|>", text)
        self.assertIn("<|channel|>analysis<|message|>Cut<|end|>", text)
        self.assertTrue(text.endswith("<|start|>assistant"))
        self.assertEqual(util.harmony_start_date(tok, ids), "2000-01-02")

    def test_harmony_parser_handles_header_order_and_malformed_json(self):
        valid = ('<|channel|>commentary to=functions.get_input<|message|>'
                 '{"task_index":2,"input_dependencies":[{"result":6}]}<|call|>')
        self.assertEqual(parse_tool_calls(valid), [{
            "name": "get_input",
            "arguments": {"task_index": 2, "input_dependencies": [{"result": 6}]},
        }])
        broken = valid.replace('"task_index":2', '"task_index":')
        self.assertEqual(parse_tool_calls(broken), [])
        self.assertTrue(looks_like_tool_call(broken))
        self.assertEqual(assistant_message_from_response(broken, [])["role"], "assistant")

    async def test_gpt_oss_generation_decodes_special_tokens(self):
        class Engine:
            async def generate(self, *args):
                yield SimpleNamespace(outputs=[SimpleNamespace(
                    text="stripped", token_ids=[1, 2], finish_reason="stop",
                )])

        tok = Mock()
        tok.decode.return_value = "<|channel|>final<|message|>Done.<|return|>"
        text, reason = await run_agent._generate_text(
            Engine(), "prompt", None, "request", messages=[], tools=[],
            chat_template_kwargs={}, tokenizer=tok, keep_special_tokens=True,
        )
        self.assertEqual(text, tok.decode.return_value)
        self.assertEqual(reason, "stop")
        tok.decode.assert_called_once_with([1, 2], skip_special_tokens=False)

    async def test_harmony_prompt_reaches_vllm_as_token_ids(self):
        prompts = []

        class Engine:
            async def generate(self, prompt, *args):
                prompts.append(prompt)
                yield SimpleNamespace(outputs=[SimpleNamespace(
                    text="", token_ids=[1], finish_reason="stop",
                )])

        await run_agent._generate_text(
            Engine(), "decoded text", None, "request", messages=[], tools=[],
            chat_template_kwargs={}, tokenizer=Mock(), keep_special_tokens=True, prompt_token_ids=[4, 5],
        )
        await run_agent._generate_text(
            Engine(), "templated text", None, "request", messages=[], tools=[],
            chat_template_kwargs={},
        )
        self.assertEqual(prompts, [{"prompt_token_ids": [4, 5]}, "templated text"])

    async def test_shared_process_dispatches_per_sample_and_uses_global_indices(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.list_tools.return_value = SimpleNamespace(tools=[SimpleNamespace(
            name="calculator", description="Calculate", inputSchema={"type": "object"},
        )])
        client.call_tool.return_value = SimpleNamespace(content=[SimpleNamespace(text="6")], isError=False)

        @asynccontextmanager
        async def stdio(*args):
            yield None, None

        seen = {}

        async def generate(*args, request_id, messages, tools, **kwargs):
            _, index, _, turn = request_id.split("_")
            index, turn = int(index), int(turn)
            offset = 100 * index
            self.assertEqual([t["function"]["name"] for t in tools], ["calculator", "get_input", "submit_answers"])
            self.assertEqual(tools[1]["function"]["parameters"], GET_INPUT_TOOL["inputSchema"])
            self.assertEqual(tools[1]["function"]["description"], GET_INPUT_TOOL["description"])
            seen[index] = copy.deepcopy(messages)
            await asyncio.sleep(0)
            outputs = [
                call("calculator", {"expression": "3 + 3"}) + request(2, -1),
                request(2, 6 + offset) + request(3, 6 + offset, 20 + offset),
                request(2, 6 + offset) + request(3, 20 + offset, 6 + offset),
                # task_20 sits last, so only its result is submitted.
                call("submit_answers", {"answers": [{"result": 60 + offset}]}),
            ]
            return outputs[turn], "stop"

        args = SimpleNamespace(
            model="test-model", reasoning_effort="low", preserve_thinking=True, example=False,
            max_agent_turns=10, max_model_len=None, tool_repair_attempts=0,
            final_answer_repair_attempts=0, cutoff_repair_attempts=0,
            max_turns_without_new_answer=20, max_concurrent_samples=2,
            get_input_attempts_per_task=2,
        )
        data = {"samples": [
            {"prompt": "Tasks", "target_node_id": "task_20", "node_outputs": records(100 * i)}
            for i in range(2)
        ]}
        with tempfile.TemporaryDirectory() as directory, \
             patch("mcp.ClientSession", return_value=client), \
             patch("mcp.client.stdio.stdio_client", stdio), redirect_stderr(io.StringIO()):
            await util.process_samples(
                args, data, [0, 1], directory, [], tokenizer=tokenizer(),
                generate_text=generate, sampling_params=SimpleNamespace(max_tokens=1024),
                repair_sampling_params=None, system_prompt="System",
            )
        for index, sample in enumerate(data["samples"]):
            self.assertEqual(sample["agent_status"], "success")
            self.assertEqual(sample["agent_tool_calls"], 6)
            self.assertEqual(set(sample["submitted_answers"]), {"task_3_out"})
            replies = [m["content"] for m in seen[index] if m["role"] == "tool" and m["name"] == "get_input"]
            offset = 100 * index
            self.assertEqual([json.loads(r) for r in replies[:3]],
                             [[11 + offset, 1], [10 + offset, 1], [[31 + offset, 1]]])
            self.assertEqual(replies[3], "Error: input request limit reached for this task.")
            self.assertEqual(json.loads(replies[4]), [[30 + offset, 1]])
        self.assertEqual(client.call_tool.await_count, 2)
        for item in client.call_tool.await_args_list:
            self.assertEqual(item.args, ("calculator", {"expressions": ["3 + 3"]}))

    async def test_ill_shaped_calls_spend_the_bank_like_any_wrong_answer(self):
        """A call that reaches the tool without stating the value is simply wrong."""
        ill_shaped = [
            call(arguments={"task_index": 2, "input_dependencies": {"result": 6}}),
            call(arguments={"task_index": 2}),
            "<function=get_input><parameter=task_index>2</parameter>",
        ]
        for bad_call in ill_shaped:
            with self.subTest(call=bad_call):
                tool = InputTool.from_records(records(), attempts_per_task=2)
                result, messages = await self.run_loop(
                    [bad_call, bad_call, request(2, 6), submit(20)], input_tool=tool,
                    graded_outputs=["task_2_out"],
                )
                replies = [m["content"] for m in messages if m["role"] == "tool"]
                # Two bank entries, in order, then the budget is gone.
                self.assertEqual([json.loads(reply) for reply in replies[:2]], [[11, 1], [12, 1]])
                self.assertEqual(replies[2], "Error: input request limit reached for this task.")
                self.assertEqual(tool.counts[2], 2)
                self.assertEqual(result["status"], "success")

    async def test_unparsable_calls_take_the_repair_path_for_either_tool(self):
        """A call the parser cannot read reaches no tool, whichever tool it named."""
        unparsable = [
            '<tool_call>{"name":"get_input","arguments":{"task_index":2, BROKEN}}</tool_call>',
            '<tool_call>{"name":"get_input","arguments":{"task_index":2}}',
            '<tool_call>{"name":"calculator","arguments":{ BROKEN}}</tool_call>',
            "<function=calculator><parameter=expressions>[\'1+1\'",
        ]
        for bad_call in unparsable:
            with self.subTest(call=bad_call):
                tool = InputTool.from_records(records(), attempts_per_task=2)
                result, messages = await self.run_loop([bad_call], input_tool=tool)
                self.assertEqual(result["status"], "malformed_tool_calls")
                self.assertEqual([m for m in messages if m["role"] == "tool"], [])
                self.assertEqual(tool.counts[2], 0)

    async def test_a_complete_call_counts_even_with_stray_text_around_it(self):
        """Stray text does not make a finished submission free."""
        tool = InputTool.from_records(records(), attempts_per_task=2)
        wrapper = ("<function=get_input><parameter=task_index>2</parameter>"
                   '<parameter=input_dependencies>[{"result":6}]</parameter>junk</function>')
        result, messages = await self.run_loop([wrapper, submit(20)], input_tool=tool,
                                               graded_outputs=["task_2_out"])
        replies = [m["content"] for m in messages if m["role"] == "tool"]
        self.assertEqual(json.loads(replies[0]), [10, 1])
        self.assertEqual(tool.counts[2], 1)
        self.assertEqual(result["status"], "success")

    async def test_json_checkpoint_preserves_the_budget_and_wrong_bank_position(self):
        snapshots = []

        async def interrupt(result):
            snapshots.append(json.loads(json.dumps(result)))
            raise asyncio.CancelledError

        initial_tool = InputTool.from_records(records(), attempts_per_task=2)
        with self.assertRaises(asyncio.CancelledError):
            await self.run_loop(
                [request(2, -1) + call(arguments=[])],
                input_tool=initial_tool, checkpoint_callback=interrupt,
                graded_outputs=["task_3_out"],
            )
        saved = snapshots[0]
        state = saved["resume_state"]
        state.update(turn_outputs=saved["turn_outputs"], task_answers=saved["task_answers"])
        fresh_tool = InputTool.from_records(records(), attempts_per_task=2)
        result, messages = await self.run_loop(
            [request(2, -1) + call(arguments="not JSON"),
             request(2, 6) + request(3, 20, 6), submit(60)],
            input_tool=fresh_tool, resume_state=state, graded_outputs=["task_3_out"],
        )
        replies = [m["content"] for m in messages if m["role"] == "tool"]
        # The restored counter resumes at the second bank entry, not the first.
        self.assertEqual(json.loads(replies[2]), [12, 1])
        self.assertEqual(replies[3], "Error: arguments must be a JSON object.")
        self.assertEqual(replies[4], "Error: input request limit reached for this task.")
        self.assertEqual(json.loads(replies[5]), [[30, 1]])
        self.assertEqual(fresh_tool.counts, {1: 0, 2: 2, 3: 1})
        self.assertEqual(result["status"], "success")

    async def test_xml_arguments(self):
        xml = ('<tool_call><function=get_input><parameter=task_index>2</parameter>'
               '<parameter=input_dependencies>[{"result":6}]</parameter></function></tool_call>')
        result, messages = await self.run_loop([xml, request(3, 20, 6), submit(60)],
                                               graded_outputs=["task_3_out"])
        self.assertEqual([json.loads(m["content"]) for m in messages if m["role"] == "tool"],
                         [[10, 1], [[30, 1]]])
        self.assertEqual(result["status"], "success")

    async def test_mixed_call_formats_keep_submission_order(self):
        xml = ('<function=get_input><parameter=task_index>2</parameter>'
               '<parameter=input_dependencies>[{"result":6}]</parameter></function>')
        result, messages = await self.run_loop([xml + request(2, -1), submit(20)],
                                               graded_outputs=["task_2_out"])
        self.assertEqual([json.loads(m["content"]) for m in messages if m["role"] == "tool"],
                         [[10, 1], [12, 1]])
        self.assertEqual(result["status"], "success")

    async def test_a_submission_before_the_target_input_is_rejected_and_the_episode_goes_on(self):
        """Submitting an earlier task's result as soon as it is known must not end the episode."""
        tool = InputTool.from_records(records(), attempts_per_task=2)
        result, messages = await self.run_loop(
            [request(1), submit(6), request(2, 6), request(3, 20, 6), submit(60)],
            input_tool=tool, graded_outputs=["task_3_out"],
        )
        replies = [m["content"] for m in messages if m["role"] == "tool"]
        self.assertEqual(replies[1], PREMATURE)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["submitted_answers"], {"task_3_out": {"answer": 60, "turn": 4}})

    async def test_a_submission_cannot_lean_on_a_request_made_in_the_same_turn(self):
        """The model had not seen the input it requested beside its submission."""
        tool = InputTool.from_records(records(), attempts_per_task=2)
        result, messages = await self.run_loop(
            [request(1), request(2, 6), request(3, 20, 6) + submit(60), submit(60)],
            input_tool=tool, graded_outputs=["task_3_out"],
        )
        replies = [m["content"] for m in messages if m["role"] == "tool"]
        # The request is served, and the submission beside it is told it came too early,
        # the same verdict that kept the episode running.
        self.assertEqual(json.loads(replies[2]), [[30, 1]])
        self.assertEqual(replies[3], PREMATURE)
        self.assertEqual(result["submitted_answers"], {"task_3_out": {"answer": 60, "turn": 3}})

    def test_every_graded_input_must_be_requested(self):
        tool = InputTool.from_records(records(), attempts_per_task=2)
        submission = {"answers": [{"result": 20}, {"result": 60}]}
        graded = ["task_2_out", "task_3_out"]
        self.assertIn("inputs requested yet: 2, 3.", util.validate_submission(submission, graded, tool))
        tool.get_input(2, [{"result": 6}])
        self.assertIn("inputs requested yet: 3.", util.validate_submission(submission, graded, tool))
        tool.get_input(3, [{"result": 20}, {"result": 6}])
        self.assertEqual(util.validate_submission(submission, graded, tool), {"task_2_out": 20, "task_3_out": 60})
        # A wrong request still hands over an input, so it counts as requested.
        wrong = InputTool.from_records(records(), attempts_per_task=2)
        wrong.get_input(3, [{"result": -1}, {"result": -1}])
        guess = {"answers": [{"result": 1}]}
        self.assertEqual(util.validate_submission(guess, ["task_3_out"], wrong), {"task_3_out": 1})
        # Without the input tool nothing is known about requests, and nothing is checked.
        self.assertEqual(util.validate_submission(guess, ["task_3_out"]), {"task_3_out": 1})


class InputLimitCliTests(unittest.TestCase):
    def invoke(self, runner, argv):
        with patch.object(sys, "argv", [runner.__name__, *argv]), \
             patch.object(runner.asyncio, "run") as inference, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                runner.main()
            finally:
                inference.assert_not_called()

    def test_limits_are_saved_validated_and_checked_on_resume_by_both_runners(self):
        for runner in (run_agent, run_agent_api):
            for limit in (5, 2):
                with self.subTest(runner=runner.__name__, limit=limit), tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "dataset.json"
                    output = Path(directory) / "results.json"
                    source.write_text(json.dumps({"summary": {}, "samples": []}))
                    argv = ["--input", str(source), "--output", str(output)]
                    if runner is run_agent_api:
                        argv += ["--api_model", "test-model"]
                    name = "get_input_attempts_per_task"
                    if limit != 5:
                        argv += [f"--{name}", str(limit)]
                    self.invoke(runner, argv)
                    params = json.loads(output.read_text())["summary"]["generation_params"]
                    self.assertEqual(params[name], limit)
                    self.assertNotIn("get_input_syntax_retries_per_task", params)
                    self.invoke(runner, argv)
                    before = output.read_bytes()
                    for value, code in ((0, 2), (-1, 2), (7, 1)):
                        with self.assertRaises(SystemExit) as error:
                            self.invoke(runner, argv + [f"--{name}", str(value)])
                        self.assertEqual(error.exception.code, code)
                        self.assertEqual(output.read_bytes(), before)

    def test_a_gpt_oss_run_never_resumes_under_the_other_renderer(self):
        for runner in (run_agent, run_agent_api):
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "dataset.json"
                output = Path(directory) / "results.json"
                source.write_text(json.dumps({"summary": {}, "samples": []}))
                argv = ["--input", str(source), "--output", str(output),
                        "--model", "openai/gpt-oss-120b", "--reasoning_effort", "high"]
                if runner is run_agent_api:
                    argv += ["--api_model", "gpt-oss-120b"]
                self.invoke(runner, argv)
                results = json.loads(output.read_text())
                self.assertEqual(results["summary"]["generation_params"]["prompt_renderer"], "harmony")
                self.invoke(runner, argv)
                # A file written before Harmony rendering carries no renderer.
                del results["summary"]["generation_params"]["prompt_renderer"]
                output.write_text(json.dumps(results))
                with self.assertRaises(SystemExit) as error:
                    self.invoke(runner, argv)
                self.assertEqual(error.exception.code, 1)

    def test_a_run_never_resumes_under_another_submission_rule(self):
        for runner in (run_agent, run_agent_api):
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "dataset.json"
                output = Path(directory) / "results.json"
                source.write_text(json.dumps({"summary": {}, "samples": []}))
                argv = ["--input", str(source), "--output", str(output)]
                if runner is run_agent_api:
                    argv += ["--api_model", "test-model"]
                self.invoke(runner, argv)
                results = json.loads(output.read_text())
                self.assertEqual(results["summary"]["generation_params"]["submission_rule"], util.SUBMISSION_RULE)
                self.invoke(runner, argv)
                # A file written before the rule existed carries no rule.
                del results["summary"]["generation_params"]["submission_rule"]
                output.write_text(json.dumps(results))
                with self.assertRaises(SystemExit) as error:
                    self.invoke(runner, argv)
                self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
