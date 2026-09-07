# ArbiGraph

A benchmark **generator** for evaluating how well language models and agents manage context —
whether they can carry forward what matters, discard what does not, and keep track of intermediate
results across a long, interdependent prompt.

You supply a DAG topology as JSON and pick a task category for each node. ArbiGraph fills every node
with a concrete, procedurally generated task, wires each task's output into its children's inputs,
and emits the whole graph as a single prompt. One node is designated the **target**, and accuracy is
measured on that node alone. Because the target can sit anywhere in the graph, you can vary how much
irrelevant material surrounds the answer, and how far the answer sits from the information it
depends on, while holding the underlying task difficulty fixed.

Every sample is verifiable by construction: the generator knows each node's ground-truth value
because it computed it while building the graph.

> **ArbiGraph: Arbitrarily Scalable Verifiable Task Graphs for Evaluating Context Management**
> Pavel Golikov, Evgenii Opryshko, Gennady Pekhimenko, Mark C. Jeffrey
> [arXiv:2607.20764](https://arxiv.org/abs/2607.20764)

## How it works

```
topology JSON  ──▶  parse_dag  ──▶  fill_dag  ──▶  generate_dataset  ──▶  dataset JSON
  nodes+edges       validate,        pick tasks,      N samples per         prompt +
  + target          build DiGraph    thread values    target task           ground truth
```

A topology names its nodes, gives each a category, lists the directed edges, and picks the target:

```json
{
  "directed": true,
  "multigraph": false,
  "target": {"id": "task_3", "native_task_id": "all"},
  "nodes": [
    {"id": "task_1", "category": "math"},
    {"id": "task_2", "category": "python"},
    {"id": "task_3", "category": "gsm"}
  ],
  "edges": [
    {"source": "task_1", "destination": "task_2"},
    {"source": "task_2", "destination": "task_3"}
  ]
}
```

`native_task_id` pins which concrete task occupies the target node, so you can hold the graded
problem fixed while varying the topology around it. Set it to `"all"` to sweep every task in that
category, generating one sample group per task.

Nodes are typed. Each task declares an input and output domain (`scalar` or `list`), and
[`choose_task_types`](source/fill_dag.py) searches for an assignment where every parent's output
type matches its child's input type, backtracking when an early choice makes a later edge
unsatisfiable. Where a node has several parents, a **join adapter** merges their outputs — summing
scalars, interleaving lists — and the merge is described in the prompt so the model can follow it.

Crucially, a child never sees its parent's value as a literal. The prompt refers to it by name
(`task_2_out["result"]`) and spells out any domain conversion in English, so the only way to answer
a downstream task is to have actually solved and retained the upstream one.

Task outputs use a JSON-style format:

```text
task_j_out = {"result": <result here>}
```

The paper version used an earlier output format based on `\boxed{...}` and different graph
topologies. That paper-era snapshot is preserved under the Git tag `paper-v1-boxed`.

## Task categories

| Category | Tasks | Domains | Source |
|---|---|---|---|
| `math` | 40 | scalar and list, all four combinations | Number theory, linear algebra, polynomials and combinatorics, built on sympy |
| `python` | 94 | scalar and list, all four combinations | LeetCode-style functions the model must trace by hand |
| `gsm` | 41 | scalar → scalar | GSM-Symbolic word-problem templates, rebound to the chained value |

Each category is a self-contained source under [source/tasks/](source/tasks/) exposing two
functions — `load_pool()`, returning its catalogue of tasks, and `make_stage()`, returning one
instantiated task ready to drop into the graph. Adding a category means writing those two functions
and adding a branch to the dispatch in [fill_dag.py](source/fill_dag.py).

Domain conversions shared across categories live in [source/tasks/adapters/](source/tasks/adapters/).

## Getting started

```bash
pip install -r requirements.txt

# Generate a dataset from a topology
cd source
python generate_dataset.py \
    --graph topologies/math/test/easy_chain.json \
    --output_dir generated/my_run \
    --num_samples_per_task 4 \
    --seed 0

# Read the prompts it produced
python dataset_to_text.py --input generated/my_run/my_run_dataset.json
```

The output directory receives the dataset JSON, a copy of the input topology, and an SVG rendering
of the graph with the target node highlighted (Graphviz `dot` must be on your PATH).

Ready-made topologies live in [source/topologies/](source/topologies/), one directory per category:
`test/` holds small graphs for development, `hf/` holds the seven release topologies — a single-node
baseline, three- and six-node chains, three and six independent nodes, a four-way fan-in, and a
two-branch recombination.

## RLVR environments

ArbiGraph exposes its verifiable task graphs as a suite of reinforcement learning with verifiable
rewards (RLVR) environments through [`source/arbigraph_rlvr/`](source/arbigraph_rlvr/). An episode
keeps the policy-visible observation separate from trusted verifier state. The policy produces one
text completion, and the scorer returns a terminal reward of `1` exactly when the designated target
output is correct, or `0` otherwise. Per-node correctness is also returned for diagnostics without
changing the terminal reward.

There are two ways to construct episodes:

- `DatasetLoader` turns materialized ArbiGraph dataset rows into episodes. The row's `sample_idx` is
  its stable identity within a target task; it is not used to randomly select a row.
- `ArbiGraphEnv` generates an episode on demand from a topology. `native_task_id` selects the target
  task, `sample_idx` selects a reproducible instance of that task, and `seed` independently controls
  graph-generation randomness.

From the `source/` directory, run the complete six-node math example with:

```bash
python arbigraph_rlvr/examples/math_six_chain_target/run.py
```

The example prints the policy-visible observation, the answer-hidden verifier state, and the score
for an empty mock completion. The same interface can be used directly:

```python
import json
from pathlib import Path

from arbigraph_rlvr import ArbiGraphEnv, EpisodeSpec

topology = json.loads(
    Path("topologies/math/hf/six_chain_target.json").read_text(encoding="utf-8")
)
topology["target"]["native_task_id"] = 0

env = ArbiGraphEnv()
episode = env.reset(
    EpisodeSpec(
        topology_id="math/hf/six_chain_target",
        topology=topology,
        seed=0,
        sample_idx=0,
    )
)

policy_input = episode.observation.to_dict()
model_completion = ""  # Replace with the policy's response to policy_input["prompt"].
score = env.score(model_completion, episode.verifier_state)
print(policy_input["episode_id"], score.reward)
```

Scoring reuses the same grader as the evaluation harness
([`eval/grade_results.py`](eval/grade_results.py) and
[`eval/parse_results.py`](eval/parse_results.py)), so RLVR rewards and reported benchmark accuracy
agree by construction.

## External dependencies

The GSM-Symbolic templates are a large third-party checkout and are kept **outside** this
repository:

```
~/
├── ArbiGraph/             this repo
└── ml-gsm-symbolic/       https://github.com/apple/ml-gsm-symbolic
```

By default ArbiGraph looks for it in the directory containing the repository. Set `ARBIGRAPH_DEPS`
to point somewhere else. Only the `gsm` category needs it; `math` and `python` have no external
checkout.

## Evaluation

[eval/](eval/) runs a generated dataset against a model with vLLM and grades the target node.

```bash
python eval/run_agent_calc.py --input source/generated/my_run/my_run_dataset.json \
    --model Qwen/Qwen3.5-27B --num_gpus 4 --output-dir eval/results
```

The harness runs the model as an agent with a calculator tool
([`calculator_mcp_server.py`](eval/calculator_mcp_server.py)), so arithmetic slips are not confused
with context-management failures. It checkpoints per sample, so an interrupted run resumes rather
than restarting. [`grade_results.py`](eval/grade_results.py) compares the parsed answer against
ground truth, requiring exact integer matches where the truth is integral and allowing a small
tolerance on floats. [`run_eval.sh`](eval/run_eval.sh) is a SLURM batch script covering the full
release sweep of 3 categories × 7 topologies; set `--account` and, if needed,
`ARBIGRAPH_REPO_ROOT` and `ARBIGRAPH_CACHE_ROOT` for your cluster before submitting it.

Evaluation results are not committed to this repository. Raw result JSON files from the paper are
available here:
https://drive.google.com/drive/folders/10ix3KcGRF02N3I1C1QGSYWwTV4n2QOLy?usp=sharing

## Datasets

Generated datasets are not committed to `main`. A set of example datasets is released on
[Hugging Face](https://huggingface.co/datasets/PavelGolikov/arbigraph). The repository contains 21
named configs: one for each combination of task category (`math`, `python`, or `gsm`) and graph
topology. It does not have a default config, so a config name is required when loading it.

```bash
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset(
    "PavelGolikov/arbigraph",
    "math_single_target_baseline",
)
test_dataset = dataset["test"]
```

Config names follow `<category>_<topology>`. All configs use the `test` split; see the Hugging Face
dataset card for the complete config list and schema.

To regenerate the Hugging Face release data from source:

```bash
cd source
bash generate_hf_datasets.sh
```

This writes generated files under `source/generated/hf_datasets/`.

## Releasing to Hugging Face

[hf_release/](hf_release/) packages generated datasets for publication. `generate_hf_datasets.sh`
produces all 21 release datasets, then:

```bash
pip install -r hf_release/requirements.txt
python hf_release/build_hf_release.py      # assemble configs from source/generated/hf_datasets
python hf_release/validate_hf_release.py   # check row counts and schema
HF_REPO_ID=<user>/<repo> ./hf_release/upload_hf_release.sh
```

## Ingesting external task sources

ArbiGraph can absorb tasks from any dataset whose instances are *generatable* — that is,
parameterized rather than fixed. Such a source must supply three things:

1. **Input and output domains** — the sets its task accepts and returns
2. **A solver** — mapping any value in the input domain to the single correct output
3. **A prompt generator** — invariant across input values, naming the input parameter so an upstream
   result reference can be substituted into it

Given those, an adapter converts between the source's domain and ArbiGraph's, and the task becomes a
node like any other. The `gsm` category is the worked example: GSM-Symbolic templates are
parameterized word problems, and [`gsm_symbolic_task.py`](source/tasks/gsm_task/gsm_symbolic_task.py)
binds the chained value to one template variable, re-renders the question with a variable reference
in place of the literal, and evaluates the template's answer expression.

## Repository layout

```
source/
  parse_dag.py           topology JSON → validated NetworkX DiGraph, plus SVG rendering
  fill_dag.py            task selection, type assignment, value threading, prompt assembly
  generate_dataset.py    CLI: topology → dataset of N samples per target task
  dataset_to_text.py     CLI: extract readable prompts from a dataset JSON
  generate_hf_datasets.sh  reproduce the Hugging Face dataset release
  arbigraph_rlvr/        RLVR episodes, dataset loading, exact rewards, examples, and tests
  tasks/                 one directory per category, each with load_pool + make_stage
  tasks/adapters/        domain conversions shared across categories
  topologies/            topology JSON by category, split into test/ and hf/
eval/                    vLLM agent harness, calculator tool, grader
hf_release/              packaging and upload for Hugging Face
```

## Tests

```bash
cd source
python -m pytest tasks/adapters/test_adapters.py tasks/gsm_task/test_gsm_task.py \
    tasks/python_task/test_python_task.py arbigraph_rlvr/tests
```

One RLVR test compares the environment's scoring against saved evaluation results and is skipped
automatically, since those result files are not distributed with the repository.

## Paper

The accompanying paper is available on arXiv: https://arxiv.org/abs/2607.20764.

If you need the exact code and datasets corresponding to the paper-era release, use:

```bash
git checkout paper-v1-boxed
```

If you use ArbiGraph, please cite:

```bibtex
@misc{golikov2026arbigraph,
      title={ArbiGraph: Arbitrarily Scalable Verifiable Task Graphs for Evaluating Context Management},
      author={Pavel Golikov and Evgenii Opryshko and Gennady Pekhimenko and Mark C. Jeffrey},
      year={2026},
      eprint={2607.20764},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.20764},
}
```
