import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_agent
import run_agent_api
import util


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / "dataset.json"
        self.output = Path(self.directory.name) / "results.json"
        self.dataset = {
            "summary": {"mode": "dag", "seed": 42},
            "samples": [
                {
                    "prompt": f"Compute {index} + 1.",
                    "sample_idx": index,
                    "target_node_id": "task_1",
                    "node_outputs": [{
                        "node_id": "task_1", "native_task_name": "math",
                        "output_value": index + 1,
                    }],
                }
                for index in range(2)
            ],
        }
        self.input.write_text(json.dumps(self.dataset))

    def argv(self, runner):
        argv = [runner.__name__, "--input", str(self.input), "--output", str(self.output)]
        if runner is run_agent_api:
            argv += ["--api_model", "test-vector-model"]
        return argv

    def results(self, runner):
        class Captured(Exception):
            pass

        parsers = []

        def capture(parser):
            parsers.append(parser)
            raise Captured

        with patch.object(argparse.ArgumentParser, "parse_args", capture):
            with self.assertRaises(Captured):
                runner.main()
        args = parsers[0].parse_args(self.argv(runner)[1:])
        params = util.generation_params(args)
        if runner is run_agent_api:
            params["api_model"] = args.api_model
        else:
            params.update(language_model_only=args.language_model_only,
                          max_cudagraph_capture_size=args.max_cudagraph_capture_size,
                          hf_overrides=None, disable_custom_all_reduce=args.disable_custom_all_reduce)
        data = util.new_results(self.dataset, params, util.SYSTEM_PROMPT)
        for index, sample in enumerate(data["samples"]):
            sample.update(
                token_length=10, agent_turns=1, agent_tool_calls=0,
                turn_outputs={"0": f'task_1_out = {{"result": {index + 1}}}'},
                task_answers={"task_1_out": {"answer": index + 1, "turn": 0}},
                agent_status="success", agent_resume_state=None,
            )
        return data

    def invoke_without_inference(self, runner, argv=None):
        with patch.object(sys, "argv", argv or self.argv(runner)), \
             patch.object(runner.asyncio, "run") as generate, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                runner.main()
            finally:
                generate.assert_not_called()

    def test_fresh_run_and_same_command_resume(self):
        self.dataset["samples"] = []
        self.input.write_text(json.dumps(self.dataset))
        for runner in (run_agent, run_agent_api):
            with self.subTest(runner=runner.__name__):
                self.output.unlink(missing_ok=True)
                self.invoke_without_inference(runner)
                first = json.loads(self.output.read_text())
                self.assertEqual(first["summary"]["mode"], "agent")
                self.invoke_without_inference(runner)
                second = json.loads(self.output.read_text())
                self.assertEqual(len(second["summary"]["start_times"]), 2)
                first["summary"].pop("start_times")
                second["summary"].pop("start_times")
                self.assertEqual(first, second)

    def test_same_command_restores_sample_checkpoints(self):
        original_input = self.input.read_bytes()
        for runner in (run_agent, run_agent_api):
            with self.subTest(runner=runner.__name__):
                data = self.results(runner)
                completed = copy.deepcopy(data["samples"][0])
                for field in util.SAMPLE_STATE_FIELDS:
                    data["samples"][0][field] = None
                self.output.write_text(json.dumps(data))
                checkpoints = util.sample_checkpoint_dir(str(self.output))
                util.save_sample_state(checkpoints, 0, completed)
                self.invoke_without_inference(runner)
                resumed = json.loads(self.output.read_text())
                self.assertEqual(resumed["samples"][0], completed)
                self.assertEqual(resumed["samples"][1], data["samples"][1])
                self.assertEqual(resumed["summary"]["agent_status_counts"], {"success": 2})
                self.assertFalse(Path(checkpoints).exists())
                self.assertEqual(self.input.read_bytes(), original_input)
                self.invoke_without_inference(runner)
                self.assertEqual(json.loads(self.output.read_text())["samples"], resumed["samples"])

    def test_partial_conversation_is_restored(self):
        data = self.results(run_agent)
        self.output.write_text(json.dumps(data))
        partial = copy.deepcopy(data["samples"][0])
        partial.update(agent_status="in_progress", agent_resume_state={
            "next_turn": 1,
            "messages": [{"role": "assistant", "content": "Saved reasoning"}],
            "next_sampling_mode": "repair",
        })
        checkpoints = util.sample_checkpoint_dir(str(self.output))
        util.save_sample_state(checkpoints, 0, partial)
        loaded = util.load_run_data(str(self.input), str(self.output))
        with redirect_stdout(io.StringIO()):
            _, pending = util.prepare_samples(loaded, str(self.output))
        self.assertEqual(pending, [0])
        self.assertEqual(loaded["samples"][0], partial)

    def test_parameter_mismatch_preserves_results_and_checkpoints(self):
        changes = {
            "temperature": .5, "top_p": .5, "top_k": 3, "min_p": .1,
            "presence_penalty": .5, "repetition_penalty": 1.5,
            "reasoning_effort": "low", "preserve_thinking": False,
            "max_tokens": 1024, "repair_turn_max_tokens": 1024,
            "max_model_len": 4096, "model": "another-model", "max_agent_turns": 2,
            "tool_repair_attempts": 0, "final_answer_repair_attempts": 0,
            "cutoff_repair_attempts": 0, "max_turns_without_new_answer": 1,
            "hf_overrides": {"different": True},
        }
        for runner in (run_agent, run_agent_api):
            variants = dict(changes)
            if runner is run_agent_api:
                variants["api_model"] = "another-vector-model"
            else:
                variants.update(language_model_only=False, max_cudagraph_capture_size=8,
                                disable_custom_all_reduce=True)
            for name, value in variants.items():
                with self.subTest(runner=runner.__name__, parameter=name):
                    data = self.results(runner)
                    data["summary"]["generation_params"][name] = value
                    self.output.write_text(json.dumps(data))
                    checkpoints = util.sample_checkpoint_dir(str(self.output))
                    util.save_sample_state(checkpoints, 0, data["samples"][0])
                    checkpoint = Path(checkpoints) / "sample_000000.json"
                    before = self.output.read_bytes(), checkpoint.read_bytes()
                    with self.assertRaises(SystemExit) as error:
                        self.invoke_without_inference(runner)
                    self.assertEqual(error.exception.code, 1)
                    self.assertEqual((self.output.read_bytes(), checkpoint.read_bytes()), before)

    def test_dataset_mismatch_preserves_existing_output(self):
        changes = [
            lambda data: data["summary"].update(seed=43),
            lambda data: data["samples"][0].update(prompt="Different prompt"),
            lambda data: data["samples"][0]["node_outputs"][0].update(output_value=99),
            lambda data: data["samples"][0].update(sample_idx=99),
            lambda data: data["samples"].reverse(),
            lambda data: data["samples"].pop(),
        ]
        for runner in (run_agent, run_agent_api):
            for change in changes:
                data = self.results(runner)
                change(data)
                self.output.write_text(json.dumps(data))
                before = self.output.read_bytes()
                with self.subTest(runner=runner.__name__, change=change):
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        self.invoke_without_inference(runner)
                    self.assertEqual(self.output.read_bytes(), before)

    def test_grading_metadata_and_input_equal_to_output(self):
        for runner in (run_agent, run_agent_api):
            data = self.results(runner)
            data["summary"]["grading"] = {"target_accuracy": 1.0}
            data["samples"][0]["grading"] = {"target_correct": True}
            self.output.write_text(json.dumps(data))
            self.invoke_without_inference(runner)
            argv = self.argv(runner)
            argv[2] = str(self.output)
            self.invoke_without_inference(runner, argv)
            self.assertEqual(json.loads(self.output.read_text())["samples"], data["samples"])

    def test_unverifiable_output_or_checkpoints_raise(self):
        for contents, error in (("{broken", json.JSONDecodeError),
                                (json.dumps(self.dataset), ValueError)):
            self.output.write_text(contents)
            with self.assertRaises(error):
                util.load_run_data(str(self.input), str(self.output))
            self.assertEqual(self.output.read_text(), contents)
        self.output.unlink()
        checkpoints = Path(util.sample_checkpoint_dir(str(self.output)))
        checkpoints.mkdir()
        with self.assertRaisesRegex(ValueError, "Checkpoints exist without a results file"):
            util.load_run_data(str(self.input), str(self.output))
        self.assertFalse(self.output.exists())

    def test_one_required_output_argument(self):
        for runner in (run_agent, run_agent_api):
            for argv in (self.argv(runner) + ["--output-dir", self.directory.name],
                         [arg for arg in self.argv(runner) if arg not in ("--output", str(self.output))]):
                with self.subTest(runner=runner.__name__, argv=argv):
                    with self.assertRaises(SystemExit) as error:
                        self.invoke_without_inference(runner, argv)
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
