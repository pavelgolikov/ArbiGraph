"""Fill a parsed custom DAG topology with concrete tasks and values."""

from __future__ import annotations

import random
from typing import Any

import networkx as nx

from tasks.gsm_task.gsm_symbolic_task import load_pool as load_gsm_pool
from tasks.gsm_task.gsm_symbolic_task import make_stage as make_gsm_stage
from tasks.math_task.math_task import load_pool as load_math_pool
from tasks.math_task.math_task import make_stage as make_math_stage
from tasks.python_task.python_task import load_pool as load_python_pool
from tasks.python_task.python_task import make_stage as make_python_stage


MAX_FILL_ATTEMPTS = 100


def load_pool(category: str) -> list[dict[str, Any]]:
    if category == "math":
        return load_math_pool()
    if category == "python":
        return load_python_pool()
    if category == "gsm":
        return load_gsm_pool()
    raise ValueError(f"Unknown category {category!r}.")


def make_stage(
    category: str,
    task: dict[str, Any],
    task_number: int,
    input_names: list[str],
    _input: Any,
) -> tuple[Any, str, list[str], Any]:
    if category == "math":
        return make_math_stage(task, task_number, input_names, _input)
    if category == "python":
        return make_python_stage(task, task_number, input_names, _input)
    if category == "gsm":
        return make_gsm_stage(task, task_number, input_names, _input)
    raise ValueError(f"Unknown category {category!r}.")


# Return the task with the requested native task id from a loaded pool.
def _task_by_id(pool: list[dict[str, Any]], task_id: int) -> dict[str, Any]:
    for task in pool:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Unknown target native_task_id {task_id}.")


# Build the candidate input/output type choices for a node category.
def _type_options(
    category: str,
    pool: list[dict[str, Any]],
    pinned_task: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = []
    seen = set()

    # Use only pinned_task for the target; use the full category pool otherwise.
    tasks = [pinned_task] if pinned_task is not None else pool
    store_pinned_task = pinned_task is not None

    for task in tasks:
        key = (task["input_type"], task["output_type"])
        if key in seen:
            continue
        seen.add(key)

        options.append({
            "category": category,
            "task": task if store_pinned_task else None,
            "input_type": task["input_type"],
            "output_type": task["output_type"],
        })
    return options


# Pick a concrete task matching the assigned types and instantiate it.
def pick_task(
    category: str,
    input_type: str,
    output_type: str,
    inputs: Any,
    *,
    input_names_for_stage: list[str],
    task_number: int,
    pools: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    pinned_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Start from the pinned target task or the full category pool.
    candidates = [pinned_task] if pinned_task is not None else list(pools[category])

    # Keep candidates that match the requested input/output types.
    candidates = [
        task for task in candidates
        if task["input_type"] == input_type
        and task["output_type"] == output_type
    ]
    rng.shuffle(candidates)

    last_error = None
    for task in candidates:
        try:
            # Try this task with the exact input value supplied by the DAG walk.
            output, prompt, static_inputs, input_value = make_stage(
                category,
                task,
                task_number,
                input_names_for_stage,
                inputs,
            )

            return {
                "task": task,
                "input": input_value,
                "output": output,
                "prompt": prompt,
                "static_inputs": static_inputs,
            }
        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"No {category} task worked for input_type={input_type!r}, "
        f"output_type={output_type!r}: {last_error}"
    )


# Assign compatible input/output types to every DAG node.
def choose_task_types(
    graph: nx.DiGraph,
    pools: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> dict[Any, dict[str, Any]]:
    order = list(nx.topological_sort(graph))

    # Resolve the pinned target task before assigning node types.
    target_id = graph.graph["target"]["id"]
    target_category = graph.nodes[target_id]["category"]
    target_task = _task_by_id(pools[target_category], int(graph.graph["target"]["native_task_id"]))
    options_by_node = {}

    for node_id in order:
        # Build independent type options; edge constraints are checked below.
        category = graph.nodes[node_id]["category"]
        pinned_task = target_task if node_id == target_id else None
        options = _type_options(category, pools[category], pinned_task)
        rng.shuffle(options)
        options_by_node[node_id] = options

    # Assign the target first so both ancestors and descendants see its types.
    node_order = [target_id] + [node_id for node_id in order if node_id != target_id]
    assigned = {}

    # Check whether one node type option is compatible with assigned neighbors.
    def compatible(node_id: Any, option: dict[str, Any]) -> bool:
        # Edges require parent output type to match the child's expected input type.
        for predecessor in graph.predecessors(node_id):
            if predecessor in assigned:
                if assigned[predecessor]["output_type"] != option["input_type"]:
                    return False

        # Already-assigned children must accept this output type.
        for successor in graph.successors(node_id):
            if successor in assigned:
                if option["output_type"] != assigned[successor]["input_type"]:
                    return False
        return True

    # Search type choices because early options can make later edges unsatisfiable.
    def solve(index: int) -> bool:
        if index == len(node_order):
            return True

        node_id = node_order[index]
        for option in options_by_node[node_id]:
            if compatible(node_id, option):
                assigned[node_id] = option
                if solve(index + 1):
                    return True
                del assigned[node_id]
        return False

    if not solve(0):
        raise ValueError("Could not assign compatible task types to DAG.")
    return dict(assigned)


# Instantiate all assigned node types into concrete task records and outputs.
def _instantiate(
    graph: nx.DiGraph,
    type_choices: dict[Any, dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    # Topological order guarantees parent outputs are ready before child tasks.
    order = list(nx.topological_sort(graph))
    task_numbers = {node_id: index + 1 for index, node_id in enumerate(order)}
    outputs = {}
    records = []

    for node_id in order:
        type_choice = type_choices[node_id]

        # Parent order follows the user's JSON edge order.
        parents = [source for source, destination in graph.graph["edge_order"] if destination == node_id]
        task_number = task_numbers[node_id]

        if not parents:
            # Root nodes defer first-input creation to pick_task.
            input_names = [f"task_{task_number}_input"]
            input_value = None
        elif len(parents) == 1:
            # Single-parent nodes receive the full parent output.
            parent = parents[0]
            input_names = [outputs[parent]["output_name"]]
            input_value = outputs[parent]["value"]
        else:
            # Category task factories adapt ordered parent outputs before the real task runs.
            input_names = [outputs[parent]["output_name"] for parent in parents]
            input_value = [outputs[parent]["value"] for parent in parents]

        picked = pick_task(
            type_choice["category"],
            type_choice["input_type"],
            type_choice["output_type"],
            input_value,
            input_names_for_stage=input_names,
            task_number=task_number,
            pools=pools,
            rng=rng,
            pinned_task=type_choice["task"],
        )
        task = picked["task"]
        input_value = picked["input"]

        output_name = f"task_{task_number}_out"
        outputs[node_id] = {"output_name": output_name, "value": picked["output"]}

        records.append({
            "node_id": node_id,
            "task_number": task_number,
            "category": type_choice["category"],
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "parent_ids": parents,
            "input_names": input_names,
            "input_type": type_choice["input_type"],
            "input_value": input_value,
            "output_name": output_name,
            "output_type": task["output_type"],
            "output_value": picked["output"],
            "prompt": picked["prompt"],
            "static_inputs": picked["static_inputs"],
        })

    # The target answer may come from the middle of the topological order.
    target_id = graph.graph["target"]["id"]
    target_output = outputs[target_id]

    return {
        "target": graph.graph["target"],
        "target_node": target_id,
        "target_output_name": target_output["output_name"],
        "ground_truth": target_output["value"],
        "ground_truth_type": type_choices[target_id]["output_type"],
        "nodes": records,
    }


# Load task pools and retry DAG filling until a valid instantiation is found.
def fill_dag(
    graph: nx.DiGraph,
    *,
    seed: int = 0,
    max_fill_attempts: int = MAX_FILL_ATTEMPTS,
) -> dict[str, Any]:
    rng = random.Random(seed)

    # Load one task pool per graph category.
    pools = {category: load_pool(category) for category in {attrs["category"] for _node_id, attrs in graph.nodes(data=True)}}
    last_error = None

    for _ in range(max_fill_attempts):
        random.seed(rng.randrange(2**31))

        # Retry from type choice after instantiation or output-vetting failure.
        type_choices = choose_task_types(graph, pools, rng)
        try:
            return _instantiate(graph, type_choices, pools, rng)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not fill DAG after {max_fill_attempts} attempts: {last_error}")


# Stitch together the task prompts in execution order.
def write_prompt(filled: dict[str, Any]) -> str:
    blocks = ["Solve the following tasks. Output the result for each."]
    blocks.extend(record["prompt"].strip() for record in filled["nodes"])
    return "\n\n".join(blocks) + "\n"
