# ArbiGraph

A benchmark **generator** for evaluating how well language models and agents manage context:
whether they can carry forward what matters, discard what does not, and keep track of intermediate
results across a long, interdependent task.

A benchmark is specified as a DAG topology in JSON, with a task category assigned to each node.
ArbiGraph fills every node with a procedurally generated task and emits the whole graph as one
prompt. One node is the **target**, and accuracy is measured on that node alone. Because the target
can sit anywhere in the graph, the amount of irrelevant material around it and its distance from the
information it depends on can be varied while task difficulty stays fixed. Every sample is
verifiable by construction, since the generator computes each node's ground truth while building
the graph.

The `get_input` mechanism is what allows arbitrary tasks to be chained. In a conventional task
chain, a child consumes its parent's output as its own input, so the parent's output type must
match the child's input type and only compatible tasks can be composed. ArbiGraph removes that
constraint. Each task is generated with its real input and a bank of alternative inputs, each
with a different answer. A non-root node never receives its parents' outputs; instead, the model
must call `get_input` with the values it believes its parents produced. If they are correct, the
tool returns the node's real input. If any is wrong, it returns an input from the bank, which is
a valid problem input whose answer is not the graded one. Dependencies are therefore enforced by
verification rather than by passing values. Any task can follow any other, a node can have any
number of parents, and a math task that returns a scalar can feed a Python task that expects a
list or a HotpotQA question without any input/output type matching or adaptation.

> **ArbiGraph: Arbitrarily Scalable Verifiable Task Graphs for Evaluating Context Management**
> Pavel Golikov, Evgenii Opryshko, Gennady Pekhimenko, Mark C. Jeffrey
> [arXiv:2607.20764](https://arxiv.org/abs/2607.20764)

## How it works

A topology names its nodes, gives each a category, lists the directed edges, and picks the target:

```json
{
  "directed": true,
  "multigraph": false,
  "target": {"id": "task_3", "native_task_id": "all"},
  "nodes": [
    {"id": "task_1", "category": "math"},
    {"id": "task_2", "category": "python"},
    {"id": "task_3", "category": "rlve"}
  ],
  "edges": [
    {"source": "task_1", "destination": "task_2"},
    {"source": "task_2", "destination": "task_3"}
  ]
}
```

`native_task_id` pins which task occupies the target node. Set it to `"all"` to generate one sample
group for every task in that category.

A root node's input is printed in its prompt, and every other node obtains its input through
`get_input`. The model solves the graph and submits the target's answer with a `submit_answers`
tool.

## Task categories

| Category | Tasks | Source |
|---|---|---|
| `math` | 40 | Number theory, linear algebra, polynomials and combinatorics, built on sympy |
| `python` | 80 | LeetCode-style functions the model must trace by hand |
| `rlve` | 40 | Reasoning Gym tasks, reimplemented in this repository |
| `rag` | 200 questions | HotpotQA distractor questions: short answer plus supporting sentences |

Each category lives under [source/tasks/](source/tasks/) and exposes `load_pool()` and
`make_stage()`. To add a category, implement those two functions and add a branch to the dispatch
in [fill_dag.py](source/fill_dag.py). A task needs a solver, a prompt that names its input, and
enough distinct answers across inputs to fill its bank of alternative inputs.

## Getting started

```bash
pip install -r requirements.txt

python source/generate_dataset.py \
    --graph generate/topologies/math/test/easy_chain.json \
    --output_dir generate/generated/my_run \
    --num_samples_per_task 4 \
    --seed 0

python generate/dataset_to_text.py --input generate/generated/my_run/my_run_dataset.json
```

The output directory receives the dataset JSON, a copy of the topology, and an SVG of the graph
with the target highlighted (this requires Graphviz `dot` on the PATH). Run
`python source/generate_dataset.py --help` for options that fix the task pool, pin task instances,
or shuffle the visible task labels.

Ready-made topologies are in [generate/topologies/](generate/topologies/), one directory per
category. `test/` holds small development graphs, and `exp1/` holds the seven experiment topologies.

### RAG data

The `rag` category needs the HotpotQA training set, which is not included in this repository.
Download `hotpot_train_v1.1.json` from the [HotpotQA website](https://hotpotqa.github.io/) and
place it at `source/tasks/rag_task/hotpot_train_v1.1.json`, or set `ARBIGRAPH_HOTPOTQA_PATH` to
its location. RAG grading also uses the official HotpotQA scorer, cloned next to this repository:

```bash
git clone https://github.com/hotpotqa/hotpot ../hotpotqa
```

## Evaluation

[eval/](eval/) runs a generated dataset against a model as a tool-using agent and grades the
target.

```bash
python eval/run_agent.py --input generate/generated/my_run/my_run_dataset.json \
    --model Qwen/Qwen3.5-27B --num_gpus 4 --output eval/results/my_run.json
python eval/grade_results.py --input eval/results/my_run.json
```

`run_agent.py` serves the model locally with vLLM, and `run_agent_api.py` runs the same agent
against an API endpoint. Both checkpoint per sample, so rerunning the same command resumes an
interrupted run. The agent also has a calculator tool, so arithmetic slips are not mistaken for
context-management failures. The SLURM scripts used for the experiments are in
[eval/slurm_scripts/](eval/slurm_scripts/).

### Results

The rollouts from the experiments are in [eval/results/](eval/results/), and the datasets they
were run on are in [generate/generated/](generate/generated/). Results from the old framework,
including the paper's raw results, are on
[Google Drive](https://drive.google.com/drive/folders/10ix3KcGRF02N3I1C1QGSYWwTV4n2QOLy?usp=sharing).

## RLVR environments

[rlvr/arbigraph_rlvr/](rlvr/arbigraph_rlvr/) exposes ArbiGraph graphs as reinforcement learning
environments with verifiable rewards. Each rollout gets a fresh session that serves the same tools
as evaluation, and the reward is `1` when the submitted target answer is correct and `0` otherwise.
`DatasetLoader` builds episodes from a generated dataset, and `ArbiGraphEnv` generates them on
demand from a topology. For a complete example, run:

```bash
python rlvr/arbigraph_rlvr/examples/math_six_chain_target/run.py
```

[rlvr/full_train/](rlvr/full_train/) contains a GRPO training setup built on this interface.

## Hugging Face dataset

An example dataset is published on
[Hugging Face](https://huggingface.co/datasets/PavelGolikov/arbigraph). It was generated with the
old framework, before the `get_input` tool, and can be regenerated from the Git tag `hf_release`.
[hf_release/](hf_release/) contains the packaging scripts.

## Repository layout

```
source/       generator: topology parsing, graph filling, dataset CLI, task categories
generate/     topologies, generation scripts, task pools, generated datasets
eval/         agent harnesses, calculator tool, grading, analysis, results
rlvr/         RLVR environments and training
tests/        test suite
hf_release/   Hugging Face packaging (old framework)
```

## Tests

```bash
python -m pytest tests
```

## Citation

The code and datasets used in the paper are preserved under the Git tag `paper-v1-boxed`.

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
