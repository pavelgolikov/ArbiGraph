"""Exercise original HotpotQA examples through DAG generation, tools, and grading."""

import asyncio
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "eval"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fill_dag import fill_dag, load_pool
from generate_dataset import generate_dataset
from parse_dag import parse_custom_dag
from tasks.arbigraph_tools import InputTool
from tasks.rag_task.rag_task import load_task
from test_rag_task import SampleFixture, extra_rows, fixture_rows
from grade_results import grade_sample
from parse_results import parse_tool_calls
from util import graded_output_name, mcp_server_args, run_agent_loop, save_json, validate_submission


SERVER = ROOT / "eval" / "mcp_server.py"


def topology(name):
    return json.loads((ROOT / "generate" / "topologies" / "rag" / "test" / f"{name}.json").read_text())


class FrameworkIntegrationTest(SampleFixture, unittest.TestCase):
    rows = staticmethod(lambda: fixture_rows() + extra_rows(8))

    def test_existing_topologies_materialize_complete_inputs_and_original_gold(self):
        for name in ("single_target", "mixed_chain", "fan_in"):
            with self.subTest(topology=name):
                graph = topology(name)
                filled = fill_dag(parse_custom_dag(graph), seed=0)
                records = {record["node_id"]: record for record in filled["nodes"]}
                rag = next(record for record in records.values() if record["category"] == "rag")
                self.assertEqual((rag["task_id"], rag["input_type"], rag["output_type"]), (0, "object", "object"))
                source = next(row for row in self.rows() if row["question"] == rag["input_value"]["question"])
                self.assertEqual(rag["input_value"], {"question": source["question"], "context": source["context"]})
                self.assertEqual(rag["output_value"], {"answer": source["answer"], "sp": source["supporting_facts"]})
                self.assertNotIn("get_question", rag["prompt"])
                parents = [records[parent] for parent in rag["parent_ids"]]
                tool = InputTool.from_records(json.loads(json.dumps(filled["nodes"])))
                if parents:
                    self.assertEqual(len(rag["wrong_inputs"]), 5)
                    dependencies = [{"result": parent["output_value"]} for parent in parents]
                    self.assertEqual(tool.get_input(rag["task_number"], dependencies), rag["input_value"])
                    wrong = tool.get_input(rag["task_number"], [{"result": None}] * len(parents))
                    self.assertEqual(wrong, rag["wrong_inputs"][1]["input"])
                    self.assertNotEqual(wrong, rag["input_value"])
                    self.assertNotIn(source["question"], rag["prompt"])
                else:
                    self.assertEqual(rag["wrong_inputs"], [])
                    self.assertIn(
                        f"task_index={rag['task_number']} and input_dependencies=[]",
                        rag["prompt"],
                    )
                    self.assertEqual(
                        tool.get_input(rag["task_number"], []),
                        rag["input_value"],
                    )
                sample = generate_dataset(graph, num_samples_per_task=1, seed=0)["samples"][0]
                completion = "<tool_call>" + json.dumps({"name": "submit_answers", "arguments": {
                    "answers": [{"result": record["output_value"]} for record in sample["node_outputs"]],
                }}) + "</tool_call>"
                expected = [f"task_{i}_out" for i in range(1, len(sample["node_outputs"]) + 1)]
                submitted = validate_submission(parse_tool_calls(completion)[0]["arguments"], expected)
                sample["submitted_answers"] = {name: {"answer": answer, "turn": 0} for name, answer in submitted.items()}
                self.assertTrue(grade_sample(sample)["all_nodes_correct"])
                rag_position = next(i for i, node in enumerate(sample["node_outputs"], 1) if node["comparator"] == "hotpotqa")
                sample["submitted_answers"][f"task_{rag_position}_out"]["answer"]["sp"] = []
                grading = grade_sample(sample)
                self.assertFalse(grading["target_correct"])
                self.assertEqual(grading["hotpotqa"][sample["target_node_id"]]["em"], 1)
                self.assertEqual(grading["hotpotqa"][sample["target_node_id"]]["joint_em"], 0)

    def test_single_target_samples_use_distinct_questions(self):
        graph = topology("single_target")
        graph["target"]["native_task_id"] = "all"
        dataset = generate_dataset(graph, 1, seed=7)
        questions = [
            sample["node_outputs"][0]["input_value"]["question"]
            for sample in dataset["samples"]
        ]
        self.assertEqual(len(questions), len(self.rows()))
        self.assertEqual(len(set(questions)), len(self.rows()))
        self.assertEqual(dataset, generate_dataset(graph, 1, seed=7))

    def test_generated_dataset_remains_usable_after_the_source_is_removed(self):
        dataset = generate_dataset(topology("mixed_chain"), 1, seed=7)
        self.assertEqual(dataset, generate_dataset(topology("mixed_chain"), 1, seed=7))
        saved = Path(self.temp.name) / "dataset.json"
        save_json(dataset, str(saved))
        self.source.unlink()
        load_task.cache_clear()
        dataset = json.loads(saved.read_text())
        self.assertEqual(mcp_server_args(dataset["samples"]), ["-B", str(SERVER)])
        sample = dataset["samples"][0]
        tool = InputTool.from_records(sample["node_outputs"])
        root, rag, child = sample["node_outputs"]
        self.assertEqual(tool.get_input(2, [{"result": root["output_value"]}]), rag["input_value"])
        equivalent = deepcopy(rag["output_value"])
        equivalent["answer"] = "THE " + equivalent["answer"].upper() + "!"
        equivalent["sp"].reverse()
        self.assertEqual(tool.get_input(3, [{"result": equivalent}]), child["input_value"])
        sample["submitted_answers"] = {f"task_{i}_out": {"answer": row["output_value"], "turn": 0}
                                       for i, row in enumerate(sample["node_outputs"], 1)}
        self.assertTrue(grade_sample(sample)["all_nodes_correct"])

    def test_only_generation_requires_the_hotpotqa_source(self):
        with patch("tasks.rag_task.rag_task.DATA_PATH", Path(self.temp.name) / "missing"):
            with self.assertRaises(FileNotFoundError):
                load_pool("rag")
            self.assertEqual(mcp_server_args([]), ["-B", str(SERVER)])
            graph = topology("single_target")
            graph["nodes"][0]["category"] = "math"
            self.assertEqual(generate_dataset(graph, 1)["summary"]["num_samples"], 1)


class AgentIntegrationTest(SampleFixture, unittest.IsolatedAsyncioTestCase):
    rows = staticmethod(lambda: fixture_rows() + extra_rows(8))

    async def test_agent_receives_complete_question_then_submits_answer_and_sentence_references(self):
        sample = generate_dataset(topology("mixed_chain"), 1, seed=3)["samples"][0]
        root, rag, child = sample["node_outputs"]
        target = next(r for r in sample["node_outputs"] if r["node_id"] == sample["target_node_id"])
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "Rendered prompt"
        tokenizer.encode.return_value = [1]
        seen = []

        def assignment(i, value):
            return f'task_{i}_out = ' + json.dumps({"result": value})

        def request(i, answer):
            return '<tool_call>' + json.dumps({"name": "get_input", "arguments": {
                "task_index": i, "input_dependencies": [{"result": answer}],
            }}) + '</tool_call>'

        def submit(*values):
            return '<tool_call>' + json.dumps({"name": "submit_answers", "arguments": {
                "answers": [{"result": value} for value in values],
            }}) + '</tool_call>'

        replies = iter([
            assignment(1, root["output_value"]) + request(2, root["output_value"]),
            assignment(2, rag["output_value"]) + request(3, rag["output_value"]),
            assignment(3, child["output_value"]) + submit(target["output_value"]),
        ])

        async def generate(*args, messages, **kwargs):
            seen.append(deepcopy(messages))
            return next(replies), "stop"

        result = await run_agent_loop(
            generate, tokenizer, SimpleNamespace(max_tokens=1024),
            [{"function": {"name": "get_input"}}], None, "System", sample["prompt"], 10, 0,
            expected_outputs=["task_1_out", "task_2_out", "task_3_out"], graded_outputs=[graded_output_name(sample)],
            input_tool=InputTool.from_records(sample["node_outputs"]),
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(json.loads(seen[1][-1]["content"]), rag["input_value"])
        self.assertEqual(json.loads(seen[2][-1]["content"]), child["input_value"])
        # Only the target is submitted; the other nodes are graded from what was written.
        sample["submitted_answers"] = result["submitted_answers"]
        sample["task_answers"] = result["task_answers"]
        self.assertTrue(grade_sample(sample)["all_nodes_correct"])


class MCPIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_needs_no_corpus_and_only_exposes_calculator(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["ARBIGRAPH_HOTPOTQA_PATH"] = "/missing/hotpotqa.json"
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode="w+") as errors:
            params = StdioServerParameters(command=sys.executable, args=mcp_server_args([]),
                                           env=environment, cwd=directory)
            async with asyncio.timeout(20):
                async with stdio_client(params, errlog=errors) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10)) as session:
                        await session.initialize()
                        self.assertEqual([tool.name for tool in (await session.list_tools()).tools], ["calculator"])
                        result = await session.call_tool("calculator", {"expressions": ["17 * 43 + 5", "pow(3, -1, 7)"]})
                        self.assertFalse(result.isError)
                        body = json.loads("".join(c.text for c in result.content if c.type == "text"))
                        self.assertEqual(body, {"results": ["736", "5"]})


if __name__ == "__main__":
    unittest.main()
