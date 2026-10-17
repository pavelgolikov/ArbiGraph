"""HotpotQA metric semantics and their use by grading and dependency gates."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "source"))

from grade_results import (
    HOTPOTQA_METRICS as METRIC_NAMES,
    _load_hotpotqa_evaluator,
    aggregate,
    comparator,
    grade_sample,
    save_json,
    score_hotpotqa as score,
)
from parse_results import answers_from_response, update_task_answers
from tasks.arbigraph_tools import InputTool


def gold():
    return {"answer": "red red blue", "sp": [["A", 0], ["B", 1], ["C", 2]]}


class HotpotQAMetricsTest(unittest.TestCase):
    def test_scoring_uses_official_functions_without_file_io(self):
        evaluator = _load_hotpotqa_evaluator()
        expected = gold()
        with (
            patch.object(evaluator, "update_answer", wraps=evaluator.update_answer) as answer,
            patch.object(evaluator, "update_sp", wraps=evaluator.update_sp) as support,
            patch("builtins.open", side_effect=AssertionError("Scoring must not open files")),
            patch("io.open", side_effect=AssertionError("Scoring must not open files")),
            patch("tempfile.mkdtemp", side_effect=AssertionError("Scoring must not create directories")),
        ):
            metrics = score(expected, expected)
        self.assertEqual(metrics, dict.fromkeys(METRIC_NAMES, 1.0))
        answer.assert_called_once_with(metrics, expected["answer"], expected["answer"])
        support.assert_called_once_with(metrics, expected["sp"], expected["sp"])

    def test_normalized_answers_and_unordered_support_sets(self):
        expected = {"answer": "New York", "sp": [["Title A", 0], ["Title B", 2]]}
        predicted = {"answer": "  THE New, YORK!  ", "sp": [["Title B", 2], ["Title A", 0], ["Title A", 0]]}
        self.assertEqual(score(predicted, expected), dict.fromkeys(METRIC_NAMES, 1.0))
        self.assertTrue(comparator("hotpotqa")(predicted, expected))
        predicted["sp"][0][0] = "title b"
        self.assertEqual(score(predicted, expected)["joint_em"], 0.0)

    def test_partial_scores_use_token_counts_and_joint_precision_recall(self):
        predicted = {"answer": "red green", "sp": [["A", 0], ["B", 1], ["D", 3], ["E", 4]]}
        actual = score(predicted, gold())
        expected = {"em": 0, "prec": 1 / 2, "recall": 1 / 3, "f1": 2 / 5,
                    "sp_em": 0, "sp_prec": 1 / 2, "sp_recall": 2 / 3, "sp_f1": 4 / 7,
                    "joint_em": 0, "joint_prec": 1 / 4, "joint_recall": 2 / 9, "joint_f1": 4 / 17}
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value)
        self.assertFalse(comparator("hotpotqa")(predicted, gold()))

    def test_special_answers_empty_answers_and_empty_support_match_official_rules(self):
        for word in ("yes", "no", "noanswer"):
            for prediction, answer in ((word + " indeed", word), (word, word + " indeed")):
                metrics = score({"answer": prediction, "sp": []}, {"answer": answer, "sp": []})
                self.assertEqual(metrics["f1"], 0)
                self.assertEqual(metrics["em"], 0)
            self.assertEqual(score({"answer": word.upper(), "sp": []}, {"answer": word, "sp": []})["em"], 1)
        metrics = score({"answer": "the", "sp": []}, {"answer": "", "sp": []})
        self.assertEqual(metrics["joint_em"], 1)
        self.assertEqual(metrics["joint_f1"], 0)

    def test_missing_and_malformed_components_do_not_crash_or_gain_joint_credit(self):
        for prediction in (None, [], True, "red red blue", {}):
            self.assertEqual(score(prediction, gold()), dict.fromkeys(METRIC_NAMES, 0.0))
        self.assertEqual(score({"answer": gold()["answer"]}, gold())["em"], 1)
        self.assertEqual(score({"sp": gold()["sp"]}, gold())["sp_em"], 1)
        for support in (None, {}, [["A"]], [["A", True]], [["A", 0.0]], [[[], 0]]):
            metrics = score({"answer": gold()["answer"], "sp": support}, gold())
            self.assertEqual(metrics["em"], 1)
            self.assertEqual(metrics["joint_em"], 0)

    def test_parser_collects_nested_outputs_and_latest_corrections(self):
        expected = gold()
        completion = 'task_2_out = ' + json.dumps({"result": expected})
        self.assertEqual(answers_from_response(completion, ["task_2_out"]), [("task_2_out", expected)])
        answers = {}
        update_task_answers(answers, completion, ["task_2_out"], 0)
        corrected = {**expected, "answer": "blue"}
        update_task_answers(answers, 'task_2_out = ' + json.dumps({"result": corrected}), ["task_2_out"], 1)
        self.assertEqual(answers["task_2_out"], {"answer": corrected, "turn": 1})

    def test_input_gate_uses_official_joint_em_and_recovers_after_partial_answers(self):
        records = [
            {"node_id": "rag", "parent_ids": [], "input_value": {}, "output_value": gold(),
             "wrong_inputs": [], "comparator": "hotpotqa"},
            {"node_id": "child", "parent_ids": ["rag"], "input_value": 5, "output_value": 10,
             "wrong_inputs": [{"input": n, "output": 2 * n} for n in (6, 7, 8)]},
        ]
        tool = InputTool.from_records(json.loads(json.dumps(records)), attempts_per_task=3)
        partial = {**gold(), "sp": gold()["sp"][:1]}
        self.assertEqual(tool.get_input(2, [{"result": partial}]), 6)
        equivalent = {"answer": "The RED red BLUE!", "sp": list(reversed(gold()["sp"]))}
        self.assertEqual(tool.get_input(2, [{"result": equivalent}]), 5)
        self.assertEqual(tool.get_input(2, [{"result": {"answer": "red green", "sp": gold()["sp"]}}]), 8)

    def test_grading_averages_per_node_and_target_metrics_including_missing_answers(self):
        sample = {
            "target_node_id": "target", "target_native_task_id": 0, "agent_status": "success",
            "node_outputs": [
                {"node_id": "other", "output_value": gold(), "comparator": "hotpotqa"},
                {"node_id": "target", "output_value": gold(), "comparator": "hotpotqa"},
            ],
            "submitted_answers": {"task_2_out": {"answer": gold(), "turn": 0}},
        }
        sample["grading"] = grade_sample(sample)
        missing = deepcopy(sample)
        missing["submitted_answers"] = {}
        missing["grading"] = grade_sample(missing)
        self.assertTrue(sample["grading"]["target_correct"])
        self.assertFalse(sample["grading"]["node_correct"]["other"])
        summary = aggregate([sample, missing], None)
        self.assertEqual(summary["hotpotqa"]["num_graded"], 4)
        self.assertEqual(summary["hotpotqa_target"]["num_graded"], 2)
        self.assertEqual(summary["hotpotqa"]["joint_em"], 0.25)
        self.assertEqual(summary["hotpotqa_target"]["joint_f1"], 0.5)
        self.assertEqual(summary["target_accuracy"], summary["hotpotqa_target"]["joint_em"])
        self.assertEqual(set(sample["grading"]["hotpotqa"]["target"]), set(METRIC_NAMES))

    def test_results_writers_preserve_original_text_whitespace(self):
        from util import save_json as save_run
        data = {"context": [["Title  A", ["A  sentence.\nAnother sentence."]]],
                "sp": [["Title  A", 0]], "answer": "A  string"}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "results.json")
            for save in (save_run, save_json):
                save(data, path)
                self.assertEqual(json.loads(Path(path).read_text()), data)


if __name__ == "__main__":
    unittest.main()
