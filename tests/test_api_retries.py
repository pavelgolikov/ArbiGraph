import asyncio
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from run_agent_api import VectorClient, process_api_samples
from util import run_agent_loop


class ApiRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.outcomes = []
        self.response_json = {
            "prompt_token_ids": [1, 2],
            "choices": [{"token_ids": [3], "finish_reason": "stop"}],
        }
        self.client = VectorClient.__new__(VectorClient)
        self.client.api_model = "test-model"
        self.client.tokenizer = Mock()
        self.client.tokenizer.encode.return_value = [1, 2]
        self.client.tokenizer.decode.return_value = 'task_1_out = {"result": 2}'
        self.client.tokenizer.apply_chat_template.return_value = "rendered prompt"
        self.client.client = httpx.AsyncClient(
            base_url="https://vector.test/v1/",
            transport=httpx.MockTransport(self.respond),
        )
        self.params = SimpleNamespace(
            max_tokens=16, temperature=1.0, top_p=0.95, top_k=20,
            min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
        )
        self.messages = [{"role": "user", "content": "Compute 1 + 1."}]
        self.template_kwargs = {"reasoning_effort": "xhigh", "preserve_thinking": True}
        sleep_patch = patch("run_agent_api.asyncio.sleep", new_callable=AsyncMock)
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        self.log = io.StringIO()
        output_redirect = redirect_stdout(self.log)
        output_redirect.__enter__()
        self.addCleanup(output_redirect.__exit__, None, None, None)

    async def asyncTearDown(self):
        await self.client.client.aclose()

    def respond(self, request):
        self.assertEqual(request.url.path, "/v1/chat/completions")
        self.requests.append(json.loads(request.content))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, httpx.Response):
            return outcome
        if outcome != 200:
            return httpx.Response(outcome, json=self.response_json)
        events = [
            {"prompt_token_ids": self.response_json["prompt_token_ids"], "choices": []},
            {"choices": self.response_json["choices"]},
        ]
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        content += "data: [DONE]\n\n"
        return httpx.Response(
            200, content=content.encode(), headers={"content-type": "text/event-stream"}
        )

    async def generate(self):
        return await self.client.generate_text(
            "rendered prompt", self.params, request_id="req_3_turn_1",
            messages=self.messages, tools=[], chat_template_kwargs=self.template_kwargs,
        )

    @staticmethod
    def stream_response(content):
        return httpx.Response(
            200, content=content.encode(), headers={"content-type": "text/event-stream"}
        )

    async def test_repeated_temporary_http_failures_preserve_request(self):
        self.outcomes = [502, 503, 504, 500, 429, 408] * 5 + [200]
        text, reason = await self.generate()
        self.assertEqual(text, 'task_1_out = {"result": 2}')
        self.assertEqual(reason, "stop")
        self.assertEqual(len(self.requests), 31)
        self.assertTrue(all(request == self.requests[0] for request in self.requests))
        self.assertEqual(self.requests[0], {
            "model": "test-model", "messages": self.messages, "tools": [],
            "tool_choice": "none", "chat_template_kwargs": self.template_kwargs,
            "return_token_ids": True, "stream": True, **vars(self.params),
        })
        self.assertEqual(self.sleep.await_count, 30)
        self.assertTrue(all(call.args == (30,) for call in self.sleep.await_args_list))
        self.assertIn("[req_3_turn_1] Vector API HTTP 502:", self.log.getvalue())
        self.assertIn("retrying in 30 seconds.", self.log.getvalue())

    async def test_gpt_oss_reasoning_effort_is_sent_top_level(self):
        self.client.api_model = "gpt-oss-120b"
        self.template_kwargs["reasoning_effort"] = "high"
        self.outcomes = [200]
        await self.generate()
        self.assertEqual(self.requests[0]["reasoning_effort"], "high")
        self.assertEqual(self.requests[0]["chat_template_kwargs"], self.template_kwargs)

    async def test_harmony_prompt_is_checked_against_the_rendered_tokens(self):
        self.client.api_model = "gpt-oss-120b"
        self.outcomes = [200]
        await self.client.generate_text(
            "decoded", self.params, request_id="req_3_turn_1", messages=self.messages,
            tools=[], chat_template_kwargs=self.template_kwargs, prompt_token_ids=[1, 2],
        )
        self.client.tokenizer.encode.assert_not_called()

    async def test_harmony_prompt_may_differ_only_in_the_servers_date(self):
        self.client.api_model = "gpt-oss-120b"
        self.client.tokenizer.decode.return_value = "<|start|>system<|message|>Current date: 2026-09-24\n"
        for server_render, accepted in (([1, 2], True), ([1, 9], False)):
            with self.subTest(accepted=accepted), \
                 patch("run_agent_api.render_harmony_prompt", return_value=server_render) as render:
                self.outcomes = [200]
                call = self.client.generate_text(
                    "decoded", self.params, request_id="req_3_turn_1", messages=self.messages,
                    tools=[], chat_template_kwargs=self.template_kwargs, prompt_token_ids=[7, 7],
                )
                if accepted:
                    await call
                else:
                    with self.assertRaises(ValueError):
                        await call
                render.assert_called_once_with(self.messages, [], "xhigh", start_date="2026-09-24")

    async def test_chat_template_prompt_mismatch_is_not_excused_by_a_date(self):
        self.client.tokenizer.encode.return_value = [7, 7]
        self.client.tokenizer.decode.return_value = "Current date: 2026-09-24"
        self.outcomes = [200]
        with patch("run_agent_api.render_harmony_prompt") as render, self.assertRaises(ValueError):
            await self.generate()
        render.assert_not_called()

    async def test_gpt_oss_output_keeps_harmony_tokens(self):
        # The Harmony parser finds tool calls by their special tokens.
        self.client.api_model = "gpt-oss-120b"
        self.outcomes = [200]
        await self.generate()
        self.client.tokenizer.decode.assert_called_once_with([3], skip_special_tokens=False)

    async def test_gemma4_output_keeps_its_special_tokens(self):
        # Gemma 4's tool calls, strings and reasoning are all delimited by special tokens.
        self.client.api_model = "gemma-4-31B-it"
        self.outcomes = [200]
        await self.generate()
        self.client.tokenizer.decode.assert_called_once_with([3], skip_special_tokens=False)

    async def test_network_failures_retry(self):
        self.outcomes = [
            httpx.ConnectError("connection failed"), httpx.ReadError("connection lost"),
            httpx.ReadTimeout("timed out"), httpx.RemoteProtocolError("server disconnected"),
            200,
        ]
        await self.generate()
        self.assertEqual(len(self.requests), 5)
        self.assertEqual(self.sleep.await_count, 4)
        self.assertTrue(all(request == self.requests[0] for request in self.requests))

    async def test_malformed_stream_retries_the_same_turn(self):
        malformed = (
            'data: {"prompt_token_ids": [1, 2], "choices": []}\n\n'
            'data: {"choices": [{"token_ids": [9], "finish_reason": "stop"}\n\n'
        )
        self.outcomes = [self.stream_response(malformed), 200]
        text, reason = await self.generate()
        self.assertEqual((text, reason), ('task_1_out = {"result": 2}', "stop"))
        self.assertEqual(self.requests[0], self.requests[1])
        self.assertEqual(self.sleep.await_count, 1)
        self.client.tokenizer.decode.assert_called_once_with([3], skip_special_tokens=True)

    async def test_truncated_stream_discards_partial_tokens_and_retries(self):
        truncated = (
            'data: {"prompt_token_ids": [1, 2], "choices": []}\n\n'
            'data: {"choices": [{"token_ids": [9], "finish_reason": "stop"}]}\n\n'
        )
        self.outcomes = [self.stream_response(truncated), 200]
        await self.generate()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.sleep.await_count, 1)
        self.client.tokenizer.decode.assert_called_once_with([3], skip_special_tokens=True)

    async def test_stream_error_event_retries(self):
        self.outcomes = [
            self.stream_response('data: {"error": {"message": "upstream failed"}}\n\n'),
            200,
        ]
        await self.generate()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.sleep.await_count, 1)

    async def test_multiline_sse_event(self):
        self.outcomes = [self.stream_response(
            'data: {"prompt_token_ids":\n'
            'data: [1, 2], "choices": []}\n\n'
            'data: {"choices": [{"token_ids": [3], "finish_reason": "stop"}]}\n\n'
            'data: [DONE]\n\n'
        )]
        text, reason = await self.generate()
        self.assertEqual((text, reason), ('task_1_out = {"result": 2}', "stop"))
        self.sleep.assert_not_awaited()

    async def test_permanent_http_errors_do_not_retry(self):
        for status in (400, 401, 403, 404, 422, 501):
            with self.subTest(status=status):
                self.requests.clear()
                self.outcomes = [status]
                with self.assertRaisesRegex(RuntimeError, f"Vector API HTTP {status}"):
                    await self.generate()
                self.assertEqual(len(self.requests), 1)
                self.sleep.assert_not_awaited()

    async def test_invalid_prompt_response_does_not_retry(self):
        self.response_json["prompt_token_ids"] = [99]
        self.outcomes = [200]
        with self.assertRaisesRegex(ValueError, "rendered prompt differs"):
            await self.generate()
        self.assertEqual(len(self.requests), 1)
        self.sleep.assert_not_awaited()

    async def test_cancellation_stops_retries(self):
        self.outcomes = [502]
        self.sleep.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.generate()
        self.assertEqual(len(self.requests), 1)

    async def test_retries_stay_within_the_same_agent_turn(self):
        self.outcomes = [502, 503, 200]
        self.client.tokenizer.decode.return_value = (
            '<tool_call>{"name": "submit_answers", "arguments": {"answers": [{"result": 2}]}}</tool_call>'
        )
        repair_params = SimpleNamespace(**{**vars(self.params), "max_tokens": 64})
        result = await run_agent_loop(
            generate_text=self.client.generate_text,
            tokenizer=self.client.tokenizer, sampling_params=self.params,
            repair_sampling_params=repair_params, tools=[], mcp_client=None,
            system_prompt="test", task_prompt="Compute 1 + 1.", max_agent_turns=2,
            index=3, expected_outputs=["task_1_out"],
            chat_template_kwargs=self.template_kwargs,
            resume_state={
                "messages": self.messages, "next_turn": 1,
                "turn_outputs": {"0": "Earlier response."},
                "total_tool_calls": 3, "next_sampling_mode": "repair",
                "final_answer_repair_attempts_used": 1,
            },
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["turns_taken"], 2)
        self.assertEqual(result["total_tool_calls"], 3)
        self.assertEqual(set(result["turn_outputs"]), {"0", "1"})
        self.assertEqual(len(self.requests), 3)
        self.assertTrue(all(request == self.requests[0] for request in self.requests))
        self.assertEqual(self.requests[0]["max_tokens"], 64)

    async def test_unexpected_api_error_aborts_instead_of_finishing_sample(self):
        with patch("run_agent_api.VectorClient", return_value=self.client), \
             patch("run_agent_api.make_sampling_params", return_value=(self.params, self.params)), \
             patch("run_agent_api.process_samples", new_callable=AsyncMock) as process:
            process.side_effect = RuntimeError("unexpected API failure")
            with self.assertRaisesRegex(RuntimeError, "unexpected API failure"):
                await process_api_samples(
                    SimpleNamespace(model="test-model", api_model="test-model"),
                    {}, [0], "checkpoints", [], "system",
                )
        self.assertTrue(process.await_args.kwargs["strict"])


if __name__ == "__main__":
    unittest.main()
