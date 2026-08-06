# ArbiGraph

ArbiGraph is a benchmark generator for evaluating context management in language
models and agents. It builds verifiable directed task graphs where each node is a
math, Python tracing, or GSM-style task, and each edge passes one task's output
into another task's input.

The current version uses JSON-style task outputs:

```text
task_j_out = {"result": <result here>}
```

The paper version used an earlier output format based on `\boxed{...}` and
different graph topologies. That paper-era snapshot is preserved in this
repository under the Git tag `paper-v1-boxed`. Raw paper result JSON files are
available here:
https://drive.google.com/drive/folders/10ix3KcGRF02N3I1C1QGSYWwTV4n2QOLy?usp=sharing

## Repository Layout

- `source/generate_dataset.py`: generate a dataset from an input graph topology.
- `source/generate_hf_datasets.sh`: reproduce the Hugging Face dataset release.
- `source/dataset_to_text.py`: export prompts from generated dataset JSON files.
- `source/topologies/`: graph topology JSON files grouped by task category.
- `source/tasks/`: task implementations for math, Python tracing, and GSM-style
  tasks.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Dataset generation also requires the Graphviz `dot` binary to render topology
SVG files.

## Datasets

Generated datasets are not committed to `main`. A set of example datasets is
released on [Hugging Face](https://huggingface.co/datasets/PavelGolikov/arbigraph).
The repository contains 21 named configs: one for each combination of task
category (`math`, `python`, or `gsm`) and graph topology. It does not have a
default config, so a config name is required when loading it.

Install the Hugging Face Datasets library:

```bash
pip install datasets
```

Then load a specific config:

```python
from datasets import load_dataset

dataset = load_dataset(
    "PavelGolikov/arbigraph",
    "math_single_target_baseline",
)
test_dataset = dataset["test"]
```

Config names follow `<category>_<topology>`. All configs use the `test` split;
see the Hugging Face dataset card for the complete config list and schema.

To regenerate the Hugging Face release data from source:

```bash
cd source
bash generate_hf_datasets.sh
```

This writes generated files under `source/generated/hf_datasets/`.

## Paper

The accompanying paper is available on arXiv:
https://arxiv.org/abs/2607.20764.

If you need the exact code and datasets corresponding to the paper-era release,
use:

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
