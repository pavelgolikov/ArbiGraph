import json
import sys
import unittest
from pathlib import Path

from arbigraph_rlvr import DatasetLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from grade_results import grade_sample


RESULT_FILES = (
    REPO_ROOT
    / "eval/results/exp_1/Qwen3.8-27B/math/"
    "agent_two_branch_recombine_target_Qwen3.8-27B.json",
    REPO_ROOT
    / "eval/results/exp_1/Qwen3.8-27B/gsm/"
    "agent_single_target_baseline_Qwen3.8-27B.json",
)


def saved_completion(sample):
    turns = sample.get("turn_outputs") or {}
    return "\n".join(
        turns[key] for key in sorted(turns, key=lambda value: int(value))
    )


RESULTS_AVAILABLE = all(path.exists() for path in RESULT_FILES)


@unittest.skipUnless(
    RESULTS_AVAILABLE,
    "Saved evaluation results are not distributed with this repository.",
)
class ExistingResultParityTest(unittest.TestCase):
    def test_saved_qwen_completions_match_current_grader(self):
        compared = 0
        for path in RESULT_FILES:
            data = json.loads(path.read_text(encoding="utf-8"))
            loader = DatasetLoader(data, path.stem, "full")
            for sample, reset in zip(data["samples"][:4], loader.iter_episodes()):
                current = grade_sample(sample)
                rlvr = loader.score(
                    saved_completion(sample),
                    reset.verifier_state,
                )

                self.assertEqual(rlvr.target_correct, current["target_correct"])
                self.assertEqual(rlvr.all_nodes_correct, current["all_nodes_correct"])
                self.assertEqual(rlvr.node_correct, current["node_correct"])
                self.assertEqual(rlvr.node_answered, current["node_answered"])
                self.assertEqual(
                    rlvr.first_incorrect_node,
                    current["first_incorrect_node"],
                )
                compared += 1

        self.assertEqual(compared, 8)


if __name__ == "__main__":
    unittest.main()
