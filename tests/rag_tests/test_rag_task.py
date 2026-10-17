"""Original HotpotQA rows remain intact through sampling and input-bank generation."""

from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

from tasks.rag_task.rag_task import RAGTask, load_pool, load_task, make_stage, same_answer


def fixture_rows():
    distractors = [[f"Distractor {i}", [f"Unrelated sentence {i}."]] for i in range(8)]
    first = {
        "_id": "q0", "question": "Where was the founder of Zephyr Observatory born?",
        "answer": "Larkspur", "supporting_facts": [["Zephyr Observatory", 1], ["Mara Vale", 0], ["Mara Vale", 1]],
        "context": [["Zephyr Observatory", ["An observatory.", "It was founded by Mara Vale."]],
                    ["Mara Vale", ["Mara Vale was born in Larkspur.", "She founded Zephyr Observatory."]],
                    *distractors],
        "type": "bridge", "level": "hard",
    }
    second = {
        "_id": "q1", "question": "Where was the founder of Delta Institute born?",
        "answer": "Novus", "supporting_facts": [["Delta Institute", 0], ["Ivo Pine", 0]],
        "context": [["Delta Institute", ["The institute was founded by Ivo Pine."]],
                    ["Ivo Pine", ["Ivo Pine was born in Novus."]], *distractors],
    }
    # The same evidence can support a different answer; joint EM keeps these distinct.
    third = {**deepcopy(first), "_id": "q2", "question": "Who founded Zephyr Observatory?", "answer": "Mara Vale"}
    return [first, second, third]


def extra_rows(count):
    return [{
        "_id": f"e{i}", "question": f"Where was the founder of Institute {i} born?",
        "answer": f"Town {i}", "supporting_facts": [[f"Institute {i}", 0], [f"Founder {i}", 0]],
        "context": [[f"Institute {i}", [f"Institute {i} was founded by Founder {i}."]],
                    [f"Founder {i}", [f"Founder {i} was born in Town {i}."]],
                    *[[f"Distractor {j}", ["An unrelated sentence."]] for j in range(8)]],
    } for i in range(count)]


class SampleFixture:
    rows = staticmethod(fixture_rows)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source.json"
        self.source.write_text(json.dumps(self.rows()))
        self.task = RAGTask(self.source)
        source_patch = patch("tasks.rag_task.rag_task.DATA_PATH", self.source)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        self.addCleanup(load_task.cache_clear)


class SourceTest(SampleFixture, unittest.TestCase):
    def test_all_original_rows_are_loaded_without_preparation(self):
        self.assertEqual(self.task.samples, self.rows())
        self.assertEqual(sorted(p.name for p in Path(self.temp.name).iterdir()), ["source.json"])
        # Original labels are retained verbatim, including source annotation quirks.
        rows = self.rows()
        rows[0]["supporting_facts"].append(["Original annotation", 99])
        self.source.write_text(json.dumps(rows))
        self.assertEqual(RAGTask(self.source).samples, rows)

    def test_empty_source_is_rejected(self):
        self.source.write_text("[]")
        with self.assertRaises(ValueError):
            RAGTask(self.source)


class TaskTest(SampleFixture, unittest.TestCase):
    def test_banks_carry_full_inputs_and_original_gold_without_mutation(self):
        random.seed(0)
        frozen = deepcopy(self.task.samples)
        bank = self.task.generate_inputs(3)
        self.assertEqual(self.task.samples, frozen)
        for entry in bank:
            self.assertEqual(set(entry["input"]), {"question", "context"})
            row = next(row for row in frozen if row["question"] == entry["input"]["question"])
            self.assertEqual(entry["input"]["context"], row["context"])
            self.assertEqual(entry["output"], {"answer": row["answer"], "sp": row["supporting_facts"]})
            self.assertEqual(len(entry["input"]["context"]), 10)
        bank[0]["input"]["context"][0][1][0] = "changed"
        bank[0]["output"]["sp"][0][0] = "changed"
        self.assertEqual(self.task.samples, frozen)

    def test_root_and_dependent_stages_share_instructions(self):
        task = load_pool()[0]
        self.assertEqual((task["task_id"], task["input_type"], task["output_type"]), (0, "object", "object"))
        self.assertEqual(task["comparator"], "hotpotqa")
        random.seed(4)
        output, prompt, static, value, bank = make_stage(task, 3, ["parent_out"], 999999, 2)
        self.assertEqual(len(bank), 2)
        self.assertEqual(static, [])
        self.assertIn("zero_based_sentence_index", prompt)
        self.assertNotIn("get_question", prompt)
        self.assertNotIn(value["question"], prompt)
        self.assertNotIn("999999", prompt)
        answers = [output] + [entry["output"] for entry in bank]
        self.assertFalse(any(same_answer(a, b) for i, a in enumerate(answers) for b in answers[i + 1:]))
        _, root_prompt, _, root_input, root_bank = make_stage(task, 3, [], None)
        self.assertEqual(root_prompt, prompt)
        self.assertEqual(root_bank, [])
        self.assertEqual(set(root_input), {"question", "context"})

    def test_pool_preselects_distinct_questions_deterministically(self):
        rows = fixture_rows() + extra_rows(205)
        self.source.write_text(json.dumps(rows))
        pool = load_pool()
        indices = [task["sample_index"] for task in pool]
        questions = [rows[index]["question"] for index in indices]
        self.assertEqual(len(pool), 200)
        self.assertEqual([task["task_id"] for task in pool], list(range(200)))
        self.assertEqual(len(set(indices)), 200)
        self.assertEqual(len(set(questions)), 200)
        self.assertEqual(
            indices,
            [task["sample_index"] for task in load_pool()],
        )

    def test_bank_uniqueness_uses_answer_normalization_and_support_sets(self):
        original = self.task.samples[0]
        equivalent = deepcopy(original)
        equivalent["_id"] = "equivalent"
        equivalent["answer"] = "  The LARKSPUR! "
        equivalent["supporting_facts"] = list(reversed(original["supporting_facts"])) + original["supporting_facts"][:1]
        self.task.samples = [original, equivalent]
        with self.assertRaisesRegex(ValueError, "distinct answers"):
            self.task.generate_inputs(2)
        # Identical short answers remain distinguishable by supporting sentences.
        different_support = deepcopy(equivalent)
        different_support["supporting_facts"] = [["Other title", 0]]
        self.task.samples = [original, different_support]
        self.assertEqual(len(self.task.generate_inputs(2)), 2)


if __name__ == "__main__":
    unittest.main()
