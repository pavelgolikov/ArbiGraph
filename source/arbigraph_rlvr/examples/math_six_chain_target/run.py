"""Use ArbiGraph's RLVR interface with the math six-chain topology."""

import json
from pathlib import Path
from pprint import pprint
import sys

# Let this file import arbigraph_rlvr when it is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arbigraph_rlvr import ArbiGraphEnv, EpisodeSpec


# Load the topology from the repository.
with open(
    Path(__file__).resolve().parents[3] / "topologies/math/hf/six_chain_target.json",
    encoding="utf-8",
) as topology_file:
    topology = json.load(topology_file)

# Select convolution as the target task.
topology["target"]["native_task_id"] = 0

# Choose a name, generation seed, and stable target sample index.
spec = EpisodeSpec(
    topology_id="math/hf/six_chain_target",
    topology=topology,
    seed=0,
    sample_idx=0,
)

# Generate the episode.
env = ArbiGraphEnv()
episode = env.reset(spec)

print("EPISODE SPEC")
pprint(spec)

print("\nPOLICY-VISIBLE OBSERVATION")
print(json.dumps(episode.observation.to_dict(), indent=2))

print("\nTRUSTED VERIFIER STATE")
pprint(episode.verifier_state)
print("Expected output names:", tuple(episode.verifier_state.expected_outputs))

# This is where a real run would call a model with episode.observation.prompt.
# An empty string stands in for the model response in this no-model example.
completion = ""

# Parse and grade the response with ArbiGraph's existing eval parser and grader,
# using expected answers that were never exposed to the policy.
score_result = env.score(completion, episode.verifier_state)

print("\nSCORE RESULT FOR THE EMPTY MOCK RESPONSE")
pprint(score_result)
