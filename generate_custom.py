"""Generate custom DAG datasets from strict NetworkX node-link JSON.

Required graph contract:
  - top-level ``directed`` is true and ``multigraph`` is false
  - top-level ``graph.target_node`` names the task node being evaluated
  - nodes use ``id``, ``kind``, and either task or adapter metadata
  - task nodes use family exactly ``math``, ``python``, or ``gsm``
  - non-target task nodes declare ``input_kind`` and ``output_kind``
  - edges use exactly ``source`` and ``destination``
  - pack-adapter input edges declare integer ``slot`` values 0..n-1

Example:
  python generate_custom.py \
    --graph scratch/custom_dag.json \
    --num_samples_per_task 1 \
    --output scratch/custom_dag_samples.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from tasks.dataset_utils import (
    LIST_LEN_MAX,
    PREAMBLE,
    SCALAR_MAX_MAG,
    load_pool,
    make_stage,
)


@dataclass(frozen=True)
class NodeOutput:
    name: str
    value: Any
    kind: str


@dataclass(frozen=True)
class CompiledTask:
    node_id: Any
    task_number: int
    task: dict[str, Any]
    predecessor_id: Any
    input_name: str
    input_value: Any
    output: Any


DEFAULT_MAX_SAMPLE_ATTEMPTS = 100


def load_graph_data(data: dict[str, Any]) -> nx.DiGraph:
    # Read the user-designated target node before NetworkX consumes the node-link JSON.
    target_node = data["graph"]["target_node"]
    graph = json_graph.node_link_graph(
        data,
        directed=True,
        multigraph=False,
        source="source",
        target="destination",
        edges="edges",
        nodes="nodes",
        name="id",
    )
    graph.graph["target_node"] = target_node

    # Enforce the graph-level contract before compiling any samples.
    if data["directed"] is not True:
        raise ValueError("Custom graph must set directed=true.")
    if data["multigraph"] is not False:
        raise ValueError("Custom graph must set multigraph=false.")
    if target_node not in graph:
        raise ValueError(f"target_node {target_node!r} is not a node id.")
    if graph.nodes[target_node]["kind"] != "task":
        raise ValueError("target_node must name a task node.")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Custom graph must be a DAG.")

    # Validate every node against the strict custom-DAG schema.
    for node_id, attrs in graph.nodes(data=True):
        # Task nodes either name the evaluated target position or a regular dependency task.
        if attrs["kind"] == "task":
            if attrs["family"] not in {"math", "python", "gsm"}:
                raise ValueError(f"Task node {node_id!r} has invalid family {attrs['family']!r}.")
            # Non-target task nodes have fixed input/output kinds in the graph.
            if node_id != target_node:
                if attrs["input_kind"] not in {"scalar", "list"}:
                    raise ValueError(f"Task node {node_id!r} has invalid input_kind.")
                if attrs["output_kind"] not in {"scalar", "list"}:
                    raise ValueError(f"Task node {node_id!r} has invalid output_kind.")
            else:
                # Target-node kinds may be omitted so the full family can be evaluated there.
                if "input_kind" in attrs and attrs["input_kind"] not in {"scalar", "list"}:
                    raise ValueError(f"Target node {node_id!r} has invalid input_kind.")
                if "output_kind" in attrs and attrs["output_kind"] not in {"scalar", "list"}:
                    raise ValueError(f"Target node {node_id!r} has invalid output_kind.")
            if graph.in_degree(node_id) > 1:
                raise ValueError(f"Task node {node_id!r} has more than one input edge.")
            continue

        # Adapter nodes reshape already-computed values between task nodes.
        if attrs["kind"] != "adapter":
            raise ValueError(f"Node {node_id!r} kind must be 'task' or 'adapter'.")
        if attrs["op"] not in {"pick", "pack"}:
            raise ValueError(f"Adapter node {node_id!r} op must be 'pick' or 'pack'.")

        # Each adapter defines the numbered input variable for exactly one downstream task.
        successors = list(graph.successors(node_id))
        if len(successors) != 1 or graph.nodes[successors[0]]["kind"] != "task":
            raise ValueError(
                f"Adapter node {node_id!r} must feed exactly one task node; "
                "that task's numbered input name is used for the adapter output."
            )
        if attrs["op"] == "pick" and graph.in_degree(node_id) != 1:
            raise ValueError(f"Pick adapter {node_id!r} must have exactly one input edge.")
        if attrs["op"] == "pick" and attrs["index"] < 0:
            raise ValueError(f"Pick adapter {node_id!r} index must be nonnegative.")
        if attrs["op"] == "pack":
            if graph.in_degree(node_id) == 0:
                raise ValueError(f"Pack adapter {node_id!r} must have at least one input edge.")
            # Pack input ordering is explicit: edge slots must be exactly 0, 1, ..., n-1.
            slots = [edge_attrs["slot"] for _src, _dst, edge_attrs in graph.in_edges(node_id, data=True)]
            if sorted(slots) != list(range(len(slots))):
                raise ValueError(
                    f"Pack adapter {node_id!r} slots must be 0..{len(slots) - 1}; got {sorted(slots)}."
                )

    return graph


def task_by_id(pool: list[dict[str, Any]], family: str, task_id: int) -> dict[str, Any]:
    # Find a user-pinned concrete task inside a family pool.
    for task in pool:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Unknown {family} task_id {task_id}.")


def node_output_kind(graph: nx.DiGraph, node_id: Any, target_task: dict[str, Any]) -> str:
    attrs = graph.nodes[node_id]
    # Adapter output kind is determined by the adapter operation.
    if attrs["kind"] == "adapter":
        return "scalar" if attrs["op"] == "pick" else "list"
    # The target node may vary across all tasks in its family.
    if node_id == graph.graph["target_node"]:
        return target_task["output_kind"]
    return attrs["output_kind"]


def node_input_kinds(graph: nx.DiGraph, node_id: Any, target_task: dict[str, Any]) -> set[str]:
    attrs = graph.nodes[node_id]
    # Pick consumes a list; pack consumes scalar inputs.
    if attrs["kind"] == "adapter":
        return {"list"} if attrs["op"] == "pick" else {"scalar"}
    # Non-target task nodes have a fixed declared input kind.
    if node_id != graph.graph["target_node"]:
        return {attrs["input_kind"]}
    # A target node may pin its input kind or inherit the allowed kinds from each target task.
    if "input_kind" in attrs:
        return {attrs["input_kind"]}
    return set(target_task["input_kinds"])


def target_tasks(graph: nx.DiGraph, pools: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    attrs = graph.nodes[graph.graph["target_node"]]
    # A target node with task_id evaluates one concrete task; otherwise evaluate the whole family.
    if "task_id" in attrs:
        return [task_by_id(pools[attrs["family"]], attrs["family"], attrs["task_id"])]
    return list(pools[attrs["family"]])


def validate_target_task_compatibility(
    graph: nx.DiGraph,
    tasks: list[dict[str, Any]],
) -> None:
    target_node = graph.graph["target_node"]
    target_attrs = graph.nodes[target_node]

    # Check every task that will occupy the target node; none are filtered out.
    for task in tasks:
        allowed_inputs = node_input_kinds(graph, target_node, task)
        if not allowed_inputs:
            raise ValueError(f"{task['task_id']}:{task['task_name']} has no allowed input kind")

        # Ensure each target-node predecessor produces a kind this target task can consume.
        for predecessor in graph.predecessors(target_node):
            predecessor_kind = node_output_kind(graph, predecessor, task)
            if predecessor_kind not in allowed_inputs:
                raise ValueError(
                    f"{task['task_id']}:{task['task_name']} cannot consume "
                    f"{predecessor_kind} from predecessor {predecessor!r}"
                )

        # If the graph pins the target input/output kind, every target task must match it.
        if "input_kind" in target_attrs and target_attrs["input_kind"] not in task["input_kinds"]:
            raise ValueError(
                f"{task['task_id']}:{task['task_name']} does not accept "
                f"declared input_kind {target_attrs['input_kind']!r}"
            )
        if "output_kind" in target_attrs and task["output_kind"] != target_attrs["output_kind"]:
            raise ValueError(
                f"{task['task_id']}:{task['task_name']} does not produce "
                f"declared output_kind {target_attrs['output_kind']!r}"
            )

        # Ensure this target task output can feed each downstream node.
        for successor in graph.successors(target_node):
            if task["output_kind"] not in node_input_kinds(graph, successor, task):
                raise ValueError(
                    f"{task['task_id']}:{task['task_name']} output_kind "
                    f"{task['output_kind']!r} cannot feed successor {successor!r}"
                )


def pack_values(values: list[Any]) -> list[Any]:
    result = []
    multiplier = 1
    # Repeat transformed copies of the scalar inputs until the framework list length is reached.
    while len(result) < LIST_LEN_MAX:
        for value in values:
            result.append(multiplier * value + multiplier - 1)
            if len(result) == LIST_LEN_MAX:
                break
        multiplier += 1
    return result


def pack_expression(input_names: list[str]) -> str:
    parts = []
    multiplier = 1
    # Build the prompt expression that mirrors pack_values for the model.
    while len(parts) < LIST_LEN_MAX:
        for name in input_names:
            parts.append(name if multiplier == 1 else f"{name} * {multiplier} + {multiplier - 1}")
            if len(parts) == LIST_LEN_MAX:
                break
        multiplier += 1
    return ", ".join(parts)


def compile_adapter(
    graph: nx.DiGraph,
    node_id: Any,
    task_numbers: dict[Any, int],
    outputs: dict[Any, NodeOutput],
    prompt_parts: list[str],
    adapters: list[dict[str, Any]],
) -> None:
    attrs = graph.nodes[node_id]
    successor = next(graph.successors(node_id))
    output_name = f"task_{task_numbers[successor]}_input"

    # Pick extracts one scalar element from an upstream list for the downstream task.
    if attrs["op"] == "pick":
        predecessor = next(graph.predecessors(node_id))
        parent = outputs[predecessor]
        output = parent.value[attrs["index"]]
        if parent.kind != "list" or isinstance(output, list):
            raise ValueError(f"Pick adapter {node_id!r} requires a list input and scalar output.")
        prompt_parts.append(f"Define {output_name} = {parent.name}[{attrs['index']}].")
        outputs[node_id] = NodeOutput(output_name, output, "scalar")
        adapters.append({
            "node_id": node_id,
            "op": "pick",
            "input_node_ids": [predecessor],
            "input_names": [parent.name],
            "output_name": output_name,
            "ground_truth": output,
        })
        return

    # Pack combines ordered scalar predecessors into one list input for the downstream task.
    ordered_edges = sorted(graph.in_edges(node_id, data=True), key=lambda item: item[2]["slot"])
    input_node_ids = [src for src, _dst, _attrs in ordered_edges]
    parents = [outputs[src] for src in input_node_ids]
    values = [parent.value for parent in parents]
    if any(parent.kind != "scalar" for parent in parents):
        raise ValueError(f"Pack adapter {node_id!r} requires scalar inputs.")

    output = pack_values(values)
    input_names = [parent.name for parent in parents]
    prompt_parts.append(
        f"Define {output_name} as the following list:\n"
        f"{output_name} = [{pack_expression(input_names)}]."
    )
    outputs[node_id] = NodeOutput(output_name, output, "list")
    adapters.append({
        "node_id": node_id,
        "op": "pack",
        "input_node_ids": input_node_ids,
        "input_names": input_names,
        "output_name": output_name,
        "ground_truth": output,
    })


def candidate_tasks(
    graph: nx.DiGraph,
    node_id: Any,
    input_kind: str,
    pools: dict[str, list[dict[str, Any]]],
    target_task: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    attrs = graph.nodes[node_id]
    # The caller passes the target task explicitly when this node is the target node.
    if node_id == graph.graph["target_node"]:
        return [target_task]

    # Non-target nodes can pin an exact task_id; otherwise all matching tasks are candidates.
    if "task_id" in attrs:
        task = task_by_id(pools[attrs["family"]], attrs["family"], attrs["task_id"])
        if input_kind not in task["input_kinds"] or task["output_kind"] != attrs["output_kind"]:
            raise ValueError(f"Fixed task at node {node_id!r} does not match the node type contract.")
        return [task]

    # Candidate selection respects the node family/kind contract. Task ids may repeat in a graph.
    candidates = [
        task for task in pools[attrs["family"]]
        if input_kind in task["input_kinds"]
        and task["output_kind"] == attrs["output_kind"]
    ]
    if not candidates:
        raise ValueError(f"No candidate task exists for node {node_id!r}.")
    rng.shuffle(candidates)
    return candidates


def compile_task(
    graph: nx.DiGraph,
    node_id: Any,
    task_number: int,
    task: dict[str, Any],
    input_name: str,
    input_value: Any,
) -> tuple[NodeOutput, str, list[str]]:
    # Instantiate one concrete task candidate.
    output, prompt, task_static_inputs = make_stage(
        graph.nodes[node_id]["family"],
        task,
        task_number,
        input_name,
        input_value,
    )
    return NodeOutput(f"task_{task_number}_out", output, task["output_kind"]), prompt, task_static_inputs


def build_sample_json(
    graph: nx.DiGraph,
    target_task: dict[str, Any],
    sample_idx: int,
    outputs: dict[Any, NodeOutput],
    static_input_lines: list[str],
    prompt_parts: list[str],
    compiled_tasks: list[CompiledTask],
    adapters: list[dict[str, Any]],
) -> dict[str, Any]:
    # The target node's output is the answer graded for this sample.
    target_node = graph.graph["target_node"]
    target_output = outputs[target_node]
    stages = []
    for compiled_task in compiled_tasks:
        task = compiled_task.task
        stages.append({
            "node_id": compiled_task.node_id,
            "task_number": compiled_task.task_number,
            "family": task["family"],
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "input_node_id": compiled_task.predecessor_id,
            "input_name": compiled_task.input_name,
            "input_value": compiled_task.input_value,
            "ground_truth": compiled_task.output,
        })
    prompt = (
        f"{PREAMBLE}\n\nStatic Inputs:\n"
        + "\n".join(static_input_lines)
        + "\n\n"
        + "\n\n".join(prompt_parts)
        + "\n"
    )
    return {
        "task_id": target_task["task_id"],
        "task_name": target_task["task_name"],
        "target_family": target_task["family"],
        "target_node": target_node,
        "target_output_name": target_output.name,
        "sample_idx": sample_idx,
        # Static inputs are the only values introduced from outside the DAG.
        "static_inputs": [
            {"node_id": stage["node_id"], "input_name": stage["input_name"], "value": stage["input_value"]}
            for stage in stages
            if stage["input_node_id"] is None
        ],
        "prompt": prompt,
        "ground_truth": target_output.value,
        "custom_dag": {
            "stages": stages,
            "adapters": adapters,
        },
    }


def compile_sample_attempt(
    graph: nx.DiGraph,
    pools: dict[str, list[dict[str, Any]]],
    target_task: dict[str, Any],
    sample_idx: int,
    rng: random.Random,
) -> dict[str, Any]:
    order = list(nx.topological_sort(graph))
    task_numbers = {}
    # Number only task nodes, in dependency order, to match the existing prompt naming scheme.
    for node_id in order:
        if graph.nodes[node_id]["kind"] == "task":
            task_numbers[node_id] = len(task_numbers) + 1

    static_inputs = {}
    # Fix root-node static inputs for this attempt so backtracking only changes task choices.
    for node_id in order:
        if graph.nodes[node_id]["kind"] == "task" and graph.in_degree(node_id) == 0:
            attrs = graph.nodes[node_id]
            input_kind = attrs["input_kind"] if "input_kind" in attrs else target_task["input_kinds"][0]
            input_name = f"task_{task_numbers[node_id]}_input"
            input_value = (
                [random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG) for _ in range(LIST_LEN_MAX)]
                if input_kind == "list"
                else random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)
            )
            static_inputs[node_id] = (input_name, input_value, input_kind, f"{input_name} = {input_value!r}")

    # Compile the graph starting at order[order_index] with the partial sample state
    # built so far. Each candidate attempt gets copied partial state, so failed task
    # candidates can be discarded cleanly when backtracking.
    def extend_dag_from_partial(
        order_index: int,
        outputs: dict[Any, NodeOutput],
        static_input_lines: list[str],
        prompt_parts: list[str],
        compiled_tasks: list[CompiledTask],
        adapters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if order_index == len(order):
            # If we've processed all nodes, build a complete sample JSON.
            return build_sample_json( graph, target_task, sample_idx, outputs, static_input_lines, prompt_parts, compiled_tasks, adapters,)

        node_id = order[order_index]
        if graph.nodes[node_id]["kind"] == "adapter":
            # Copy the partial sample state before adding this adapter output.
            partial_outputs = outputs.copy()
            partial_prompt_parts = prompt_parts.copy()
            partial_adapters = adapters.copy()
            compile_adapter(graph, node_id, task_numbers, partial_outputs, partial_prompt_parts, partial_adapters)
            return extend_dag_from_partial(order_index + 1, partial_outputs, static_input_lines, partial_prompt_parts, compiled_tasks, partial_adapters)

        predecessor_ids = list(graph.predecessors(node_id))
        # A task with a predecessor consumes the already-compiled predecessor output.
        if predecessor_ids:
            predecessor_id = predecessor_ids[0]
            predecessor_output = outputs[predecessor_id]
            input_name = predecessor_output.name
            input_value = predecessor_output.value
            input_kind = predecessor_output.kind
            static_input_line = None
        else:
            # Root task nodes use static inputs fixed for this sample attempt.
            predecessor_id = None
            input_name, input_value, input_kind, static_input_line = static_inputs[node_id]

        # Try each compatible task candidate. If a downstream node fails, backtrack here.
        for task in candidate_tasks(graph, node_id, input_kind, pools, target_task, rng):
            # Copy the partial sample state before mutating it for this candidate.
            # If this candidate fails later, these copies are discarded and the next
            # candidate starts from the original state passed into extend_dag_from_partial.
            partial_outputs = outputs.copy()
            partial_static_input_lines = static_input_lines.copy()
            partial_prompt_parts = prompt_parts.copy()
            partial_compiled_tasks = compiled_tasks.copy()
            partial_adapters = adapters.copy()
            if static_input_line is not None:
                partial_static_input_lines.append(static_input_line)
            try:
                node_output, task_prompt, task_static_inputs = compile_task(graph, node_id, task_numbers[node_id], task, input_name, input_value)
                partial_outputs[node_id] = node_output
                partial_static_input_lines.extend(task_static_inputs)
                partial_prompt_parts.append(task_prompt)
                partial_compiled_tasks.append(CompiledTask(node_id, task_numbers[node_id], task, predecessor_id, input_name, input_value, node_output.value))
                return extend_dag_from_partial(order_index + 1, partial_outputs, partial_static_input_lines, partial_prompt_parts, partial_compiled_tasks, partial_adapters)
            except Exception:
                pass

        raise ValueError(f"No task candidate worked for node {node_id!r}.")

    return extend_dag_from_partial(0, {}, [], [], [], [])


def compile_sample(
    graph: nx.DiGraph,
    pools: dict[str, list[dict[str, Any]]],
    target_task: dict[str, Any],
    sample_idx: int,
    rng: random.Random,
    max_sample_attempts: int,
) -> dict[str, Any]:
    # A sample attempt fixes static inputs, then backtracks over candidate task choices.
    for _attempt in range(max_sample_attempts):
        random.seed(rng.randrange(2**31))
        try:
            return compile_sample_attempt(graph, pools, target_task, sample_idx, rng)
        except Exception:
            pass

    raise ValueError(
        f"Could not compile sample {sample_idx} for target "
        f"{target_task['family']}:{target_task['task_id']} after {max_sample_attempts} attempts."
    )


def generate_dataset_from_graph_data(
    graph_data: dict[str, Any],
    *,
    num_samples_per_task: int = 16,
    seed: int = 0,
    max_sample_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS,
) -> dict[str, Any]:

    graph = load_graph_data(graph_data)
    # Load only the task families present in the graph.
    families = sorted({
        attrs["family"]
        for _node_id, attrs in graph.nodes(data=True)
        if attrs["kind"] == "task"
    })
    pools = {family: load_pool(family) for family in families}
    targets = target_tasks(graph, pools)
    validate_target_task_compatibility(graph, targets)

    rng = random.Random(seed)
    samples = []
    # Generate the requested number of samples for every target task.
    for target_task in targets:
        for sample_idx in range(num_samples_per_task):
            samples.append(compile_sample(graph, pools, target_task, sample_idx, rng, max_sample_attempts))

    return {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": "custom_dag",
            "target_node": graph.graph["target_node"],
            "target_family": graph.nodes[graph.graph["target_node"]]["family"],
            "num_target_tasks": len(targets),
            "num_target_tasks_total": len(pools[graph.nodes[graph.graph["target_node"]]["family"]]),
            "num_samples_per_task": num_samples_per_task,
            "num_samples": len(samples),
            "seed": seed,
            "list_len_max": LIST_LEN_MAX,
            "scalar_max_mag": SCALAR_MAX_MAG,
            "topological_order": list(nx.topological_sort(graph)),
        },
        "samples": samples,
    }


def dumps_json_compact_lists(value: Any, indent: int = 0) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            comma = "," if index + 1 < len(items) else ""
            lines.append(f"{' ' * (indent + 2)}{json.dumps(key)}: {dumps_json_compact_lists(item, indent + 2)}{comma}")
        lines.append(f"{' ' * indent}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        # Numeric/string lists are data values in these datasets; keep them horizontal.
        if all(not isinstance(item, dict) for item in value):
            return json.dumps(value)
        lines = ["["]
        for index, item in enumerate(value):
            comma = "," if index + 1 < len(value) else ""
            lines.append(f"{' ' * (indent + 2)}{dumps_json_compact_lists(item, indent + 2)}{comma}")
        lines.append(f"{' ' * indent}]")
        return "\n".join(lines)

    return json.dumps(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Strict NetworkX node-link JSON graph file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples_per_task", type=int, default=16)
    parser.add_argument(
        "--max_sample_attempts",
        type=int,
        default=DEFAULT_MAX_SAMPLE_ATTEMPTS,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Main handles file I/O; generation itself stays isolated for tests.
    with open(args.graph, "r", encoding="utf-8") as handle:
        data = generate_dataset_from_graph_data(
            json.load(handle),
            num_samples_per_task=args.num_samples_per_task,
            max_sample_attempts=args.max_sample_attempts,
            seed=args.seed,
        )
    data["summary"]["graph_path"] = args.graph
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(dumps_json_compact_lists(data))
        handle.write("\n")
    print(f"Generated {len(data['samples'])} custom DAG samples at {args.output}")


if __name__ == "__main__":
    main()
