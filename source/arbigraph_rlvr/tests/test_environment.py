import json
import os
import subprocess
import sys
import unittest

from arbigraph_rlvr import ArbiGraphEnv, EpisodeSpec


GRAPH = {
    "directed": True,
    "multigraph": False,
    "target": {"id": "target", "native_task_id": 0},
    "nodes": [
        {"id": "source", "category": "math"},
        {"id": "target", "category": "math"},
        {"id": "distractor", "category": "math"},
    ],
    "edges": [
        {"source": "source", "destination": "target"},
        {"source": "source", "destination": "distractor"},
    ],
}

ROOT_TARGET_GRAPH = {
    "directed": True,
    "multigraph": False,
    "target": {"id": "target", "native_task_id": 0},
    "nodes": [{"id": "target", "category": "math"}],
    "edges": [],
}


def correct_completion(verifier_state):
    return "\n".join(
        f"{name} = {json.dumps({'result': value})}"
        for name, value in verifier_state.expected_outputs.items()
    )


class EnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ArbiGraphEnv()
        cls.spec = EpisodeSpec("source_target", GRAPH, 23, 0)
        cls.reset_result = cls.env.reset(cls.spec)

    def test_reset_generates_an_episode(self):
        observation = self.reset_result.observation

        self.assertEqual(observation.metadata, {})
        self.assertNotIn("metadata", observation.to_dict())
        self.assertEqual(observation.requested_outputs, (
            "task_1_out",
            "task_2_out",
            "task_3_out",
        ))
        self.assertEqual(observation.target_output, "task_2_out")
        self.assertIn("Task 1:", observation.prompt)

    def test_public_observation_does_not_leak_hidden_answers(self):
        public = self.reset_result.to_public_dict()
        serialized = json.dumps(public, sort_keys=True)

        for hidden_name in (
            "expected_outputs",
            "node_outputs",
            "output_value",
            "ground_truth",
            "verifier_state",
        ):
            self.assertNotIn(hidden_name, serialized)
        self.assertNotIn("VerifierState", repr(self.reset_result))

    def test_correct_wrong_and_missing_target_rewards(self):
        state = self.reset_result.verifier_state
        correct = self.env.score(correct_completion(state), state)
        wrong = self.env.score(
            f'{state.target_output} = {{"result": "definitely wrong"}}',
            state,
        )
        missing = self.env.score("", state)

        self.assertEqual(correct.reward, 1.0)
        self.assertEqual(wrong.reward, 0.0)
        self.assertEqual(missing.reward, 0.0)

    def test_score_returns_per_node_diagnostics(self):
        state = self.reset_result.verifier_state
        answers = dict(state.expected_outputs)
        answers["task_1_out"] = "wrong upstream value"
        completion = "\n".join(
            f"{name} = {json.dumps({'result': value})}"
            for name, value in answers.items()
        )

        score = self.env.score(completion, state)

        self.assertEqual(
            score.node_correct,
            {"source": False, "target": True, "distractor": True},
        )
        self.assertEqual(score.first_incorrect_node, "source")

    def test_same_spec_is_reproducible_in_a_fresh_process(self):
        local = {
            "observation": self.reset_result.observation.to_dict(),
            "expected_outputs": dict(
                self.reset_result.verifier_state.expected_outputs
            ),
        }
        script = f"""
import json
from arbigraph_rlvr import ArbiGraphEnv, EpisodeSpec
graph = {GRAPH!r}
reset = ArbiGraphEnv().reset(EpisodeSpec('source_target', graph, 23, 0))
print(json.dumps({{
    'observation': reset.observation.to_dict(),
    'expected_outputs': dict(reset.verifier_state.expected_outputs),
}}, sort_keys=True))
"""
        environment = dict(os.environ)
        source_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        environment["PYTHONPATH"] = source_root
        fresh = subprocess.check_output(
            [sys.executable, "-c", script],
            text=True,
            env=environment,
        )

        self.assertEqual(json.loads(fresh), local)

    def test_sample_idx_selects_a_distinct_root_target_sample(self):
        first = self.env.reset(
            EpisodeSpec("root_target", ROOT_TARGET_GRAPH, 23, 1)
        )
        second = self.env.reset(
            EpisodeSpec("root_target", ROOT_TARGET_GRAPH, 23, 2)
        )

        self.assertNotEqual(first.observation.prompt, second.observation.prompt)
        self.assertNotEqual(
            first.verifier_state.expected_outputs,
            second.verifier_state.expected_outputs,
        )

    def test_seed_does_not_replace_root_target_sample_idx(self):
        first = self.env.reset(
            EpisodeSpec("root_target", ROOT_TARGET_GRAPH, 23, 1)
        )
        second = self.env.reset(
            EpisodeSpec("root_target", ROOT_TARGET_GRAPH, 24, 1)
        )

        self.assertEqual(first.observation.prompt, second.observation.prompt)
        self.assertEqual(
            first.verifier_state.expected_outputs,
            second.verifier_state.expected_outputs,
        )


if __name__ == "__main__":
    unittest.main()
