"""Generate one current-format RLVR episode without running a model."""

import json
from pathlib import Path
import sys

# Let this file import arbigraph_rlvr when it is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arbigraph_rlvr import ArbiGraphEnv, EpisodeSpec


# Load the topology from the repository.
with open(
    Path(__file__).resolve().parents[4] / "generate/topologies/math/exp1/six_chain_target.json",
    encoding="utf-8",
) as topology_file:
    topology = json.load(topology_file)

# Select convolution as the target task.
topology["target"]["native_task_id"] = 0

# Choose a name, generation seed, and stable target sample index.
spec = EpisodeSpec(
    topology_id="math/exp1/six_chain_target",
    topology=topology,
    seed=0,
    sample_idx=0,
)

# Generate the episode.
env = ArbiGraphEnv()
episode = env.reset(spec)

print("\nPOLICY-VISIBLE OBSERVATION")
print(json.dumps(episode.observation.to_dict(), indent=2))

# A policy worker receives only the observation. Its tool calls are served by
# this trusted per-rollout session; a fresh session resets all input budgets.
session = episode.start()
root_input = session.call_tool("get_input", {"task_index": 1, "input_dependencies": []})
print("\nROOT INPUT TOOL RESPONSE")
print(root_input)
print("\nREWARD WITHOUT A TARGET SUBMISSION")
print(session.score().reward)
