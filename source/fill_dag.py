"""Fill a parsed custom DAG topology with concrete tasks and values."""

from __future__ import annotations

import random
from typing import Any

import networkx as nx

from tasks.math_task.math_task import load_pool as load_math_pool
from tasks.math_task.math_task import make_stage as make_math_stage
from tasks.arbigraph_tools import validate_limit
from tasks.python_task.python_task import load_pool as load_python_pool
from tasks.python_task.python_task import make_stage as make_python_stage
from tasks.rag_task.rag_task import load_pool as load_rag_pool
from tasks.rag_task.rag_task import make_stage as make_rag_stage
from tasks.rlve_task.rlve_task import load_pool as load_rlve_pool
from tasks.rlve_task.rlve_task import make_stage as make_rlve_stage


MAX_FILL_ATTEMPTS = 200

# How far past the requested pin index to look for an instantiation that fills any node.
MAX_PIN_SEARCH = 16

_PIN_CACHE: dict[tuple[str, int, int, int], int | None] = {}


# Name the instantiation a task takes, matching how a single-target baseline names its own.
def pin_key(category: str, task_id: int, pin_index: int) -> str:
    """Index N names the instantiation a single-target run generates at sample_idx N.

    The trailing zero is the retry counter that a lone target node has always carried,
    kept so that the key reproduces instances from datasets generated before pinning.
    """
    return f"{category}:{task_id}:{pin_index}:0"


def load_pool(category: str) -> list[dict[str, Any]]:
    if category == "math":
        return load_math_pool()
    if category == "python":
        return load_python_pool()
    if category == "rlve":
        return load_rlve_pool()
    if category == "rag":
        return load_rag_pool()
    raise ValueError(f"Unknown category {category!r}.")


def make_stage(
    category: str,
    task: dict[str, Any],
    task_number: int,
    input_names: list[str],
    _input: Any,
    get_input_attempts_per_task: int = 5,
    with_bank: bool = False,
) -> tuple[Any, str, list[str], Any, list[dict[str, Any]]]:
    if category == "math":
        return make_math_stage(task, task_number, input_names, _input, get_input_attempts_per_task, with_bank)
    if category == "python":
        return make_python_stage(task, task_number, input_names, _input, get_input_attempts_per_task, with_bank)
    if category == "rlve":
        return make_rlve_stage(task, task_number, input_names, _input, get_input_attempts_per_task, with_bank)
    if category == "rag":
        return make_rag_stage(task, task_number, input_names, _input, get_input_attempts_per_task, with_bank)
    raise ValueError(f"Unknown category {category!r}.")


# Choose which instantiation of a task is its pinned one.
def resolve_pin(
    category: str,
    task: dict[str, Any],
    pin_index: int,
    get_input_attempts_per_task: int = 5,
) -> int | None:
    """Return the first index at or above pin_index whose instantiation fills any node.

    An index selects the same instantiation a single-target baseline generates at that
    sample_idx, so pinning reuses a concrete problem the model has already been run on
    rather than inventing a new one.

    A node with a parent carries a bank of alternative inputs whose answers must all
    differ, and some instantiations cannot supply one: a four-element permutation has
    too few possible inversion counts, for instance. Such an instantiation builds as a
    lone root and nowhere else, so the search steps past it. The order searched is
    fixed before anything runs and reads no outcome, so which instantiation a task ends
    up with is independent of how the model performs. None means the search found none.
    """
    key = (category, task["task_id"], pin_index, get_input_attempts_per_task)
    if key not in _PIN_CACHE:
        _PIN_CACHE[key] = None
        for index in range(pin_index, pin_index + MAX_PIN_SEARCH):
            random.seed(pin_key(category, task["task_id"], index))
            try:
                make_stage(category, task, 2, ["task_1_out"], [0], get_input_attempts_per_task, True)
            except Exception:
                continue
            _PIN_CACHE[key] = index
            break
    return _PIN_CACHE[key]


# Report each task's pinned instantiation, leaving out the tasks that have none.
def resolve_pinned_pool(
    category: str,
    tasks: list[dict[str, Any]],
    pin_index: int,
    get_input_attempts_per_task: int = 5,
) -> dict[int, int]:
    resolved = {
        task["task_id"]: resolve_pin(category, task, pin_index, get_input_attempts_per_task)
        for task in tasks
    }
    return {task_id: index for task_id, index in resolved.items() if index is not None}


# Return the task with the requested native task id from a loaded pool.
def _task_by_id(pool: list[dict[str, Any]], task_id: int) -> dict[str, Any]:
    for task in pool:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Unknown target native_task_id {task_id}.")


# Pick a concrete task independently of parent output types.
def pick_task(
    category: str,
    inputs: Any,
    *,
    input_names_for_stage: list[str],
    task_number: int,
    pools: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    pinned_task: dict[str, Any] | None = None,
    get_input_attempts_per_task: int = 5,
    exclude: frozenset[int] | set[int] = frozenset(),
    pin_index: int | None = None,
) -> dict[str, Any]:
    # Start from the pinned target task or the full category pool.
    candidates = [pinned_task] if pinned_task is not None else [
        task for task in pools[category] if task["task_id"] not in exclude
    ]

    rng.shuffle(candidates)

    last_error = None
    for task in candidates:
        # With pin_index set, a task's instantiation is keyed by the task's own
        # identity, so the same task carries the same static data and the same input
        # value wherever it is placed. A node inside a graph is then the problem the
        # single-target run measured, and only its surroundings differ.
        if pin_index is not None:
            resolved = resolve_pin(category, task, pin_index, get_input_attempts_per_task)
            if resolved is None:
                continue
            random.seed(pin_key(category, task["task_id"], resolved))
        try:
            # Factories sample native inputs; parent values only identify non-roots.
            output, prompt, static_inputs, input_value, wrong_inputs = make_stage(
                category,
                task,
                task_number,
                input_names_for_stage,
                inputs,
                get_input_attempts_per_task,
                # A pinned root builds its bank too, so a task that cannot fill an
                # interior node is rejected here rather than calibrated and never used.
                with_bank=pin_index is not None,
            )

            return {
                "task": task,
                "input": input_value,
                "output": output,
                "prompt": prompt,
                "static_inputs": static_inputs,
                "wrong_inputs": wrong_inputs,
            }
        except Exception as exc:
            last_error = exc

    raise ValueError(f"No {category} task worked: {last_error}")


# Instantiate task records and store the inputs needed by get_input.
def _instantiate(
    graph: nx.DiGraph,
    pools: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    task_numbers: dict[str, int],
    sample_idx: int = 0,
    get_input_attempts_per_task: int = 5,
    pin_index: int | None = None,
    distinct_tasks: bool = False,
) -> dict[str, Any]:
    # Topological order guarantees parent outputs are ready before child tasks.
    order = list(nx.topological_sort(graph))
    outputs = {}
    records = []
    used_task_ids: set[int] = set()

    target_id = graph.graph["target"]["id"]
    target_category = graph.nodes[target_id]["category"]
    target_task = _task_by_id(pools[target_category], int(graph.graph["target"]["native_task_id"]))

    # The target's task is reserved, or an earlier node could take it and force a repeat.
    if distinct_tasks:
        used_task_ids.add(target_task["task_id"])

    for node_id in order:
        category = graph.nodes[node_id]["category"]

        # Parent order follows the user's JSON edge order.
        parents = [source for source, destination in graph.graph["edge_order"] if destination == node_id]
        task_number = task_numbers[node_id]

        if not parents:
            # Root nodes defer first-input creation to pick_task.
            input_names = [f"task_{task_number}_input"]
            input_value = None
        elif len(parents) == 1:
            # Preserve the parent reference for the get_input instruction.
            parent = parents[0]
            input_names = [outputs[parent]["output_name"]]
            input_value = outputs[parent]["value"]
        else:
            # Preserve the order in which get_input expects parent outputs.
            input_names = [outputs[parent]["output_name"] for parent in parents]
            input_value = [outputs[parent]["value"] for parent in parents]

        target_root = not parents and node_id == graph.graph["target"]["id"]
        target_attempt = 0
        while True:
            node_rng = rng
            # A pinned instantiation cannot be re-rolled, so its own key replaces this one.
            if target_root and pin_index is None:
                target_task_id = graph.graph["target"]["native_task_id"]
                key = f"{category}:{target_task_id}:{sample_idx}:{target_attempt}"
                random.seed(key)
                node_rng = random.Random(key)
            try:
                picked = pick_task(
                    category,
                    input_value,
                    input_names_for_stage=input_names,
                    task_number=task_number,
                    pools=pools,
                    rng=node_rng,
                    pinned_task=target_task if node_id == target_id else None,
                    get_input_attempts_per_task=get_input_attempts_per_task,
                    exclude=used_task_ids if distinct_tasks else frozenset(),
                    pin_index=pin_index,
                )
                break
            except Exception:
                target_attempt += 1
                if not target_root or pin_index is not None or target_attempt >= MAX_FILL_ATTEMPTS:
                    raise
        task = picked["task"]
        used_task_ids.add(task["task_id"])
        input_value = picked["input"]

        if parents:
            input_prompt = (
                f"To get task_{task_number}_input, call the get_input tool with "
                f"task_index={task_number} and input_dependencies=[{', '.join(input_names)}], substituting the actual values.\n"
            )
        else:
            input_prompt = (
                f"To get task_{task_number}_input, call the get_input tool with "
                f"task_index={task_number} and input_dependencies=[].\n"
            )

        output_name = f"task_{task_number}_out"
        outputs[node_id] = {"output_name": output_name, "value": picked["output"]}

        records.append({
            "node_id": node_id,
            "task_number": task_number,
            "category": category,
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "parent_ids": parents,
            "input_names": input_names,
            "input_type": task["input_type"],
            "input_value": input_value,
            "output_name": output_name,
            "output_type": task["output_type"],
            "output_value": picked["output"],
            "comparator": task.get("comparator", "numeric"),
            "prompt": f"Task {task_number}:\n" + input_prompt + picked["prompt"],
            "static_inputs": picked["static_inputs"],
            "wrong_inputs": picked["wrong_inputs"],
        })

    # Consumers use record position as the visible task index. Sorting here keeps
    # that invariant even when visible task labels are not topological.
    records.sort(key=lambda record: record["task_number"])

    # The target answer may carry any visible task number.
    target_id = graph.graph["target"]["id"]
    target_output = outputs[target_id]

    return {
        "target": graph.graph["target"],
        "target_node": target_id,
        "target_output_name": target_output["output_name"],
        "ground_truth": target_output["value"],
        "ground_truth_type": target_task["output_type"],
        "nodes": records,
    }


# Load task pools and retry DAG filling until a valid instantiation is found.
def fill_dag(
    graph: nx.DiGraph,
    *,
    seed: int = 0,
    sample_idx: int = 0,
    max_fill_attempts: int = MAX_FILL_ATTEMPTS,
    get_input_attempts_per_task: int = 5,
    pool_task_ids: list[int] | None = None,
    pin_index: int | None = None,
    distinct_tasks: bool = False,
    shuffle_task_labels: bool = False,
) -> dict[str, Any]:
    validate_limit(get_input_attempts_per_task, "get_input_attempts_per_task")
    rng = random.Random(seed)

    # Load one task pool per graph category.
    categories = {attrs["category"] for _node_id, attrs in graph.nodes(data=True)}
    pools = {category: load_pool(category) for category in categories}

    # Restricting the pool keeps nodes to a named subset of each category's tasks.
    if pool_task_ids is not None:
        allowed = set(pool_task_ids)
        pools = {
            category: [task for task in tasks if task["task_id"] in allowed]
            for category, tasks in pools.items()
        }

    for category, tasks in pools.items():
        needed = sum(1 for _node_id, attrs in graph.nodes(data=True) if attrs["category"] == category)
        if not tasks:
            raise ValueError(f"No {category} task is left after restricting the pool.")
        if distinct_tasks and len(tasks) < needed:
            raise ValueError(
                f"Filling {needed} {category} nodes without repeats needs at least {needed} "
                f"tasks in the pool, which holds {len(tasks)}."
            )

    last_error = None

    # Visible labels are presentation state, not task-generation state. Their own
    # namespaced RNG makes a shuffled sample exactly match its ordinary counterpart
    # in task selection, parameters, inputs, outputs, and alternative banks.
    order = list(nx.topological_sort(graph))
    labels = list(range(1, len(order) + 1))
    if shuffle_task_labels and len(labels) > 1:
        label_rng = random.Random(f"arbigraph-task-labels:{seed}")
        while True:
            label_rng.shuffle(labels)
            task_numbers = dict(zip(order, labels))
            # When dependencies exist, make the treatment genuine: at least one
            # child must appear before one of its parents in numerical prompt order.
            if not graph.edges or any(
                task_numbers[source] > task_numbers[destination]
                for source, destination in graph.edges
            ):
                break
    else:
        task_numbers = {
            node_id: index + 1 for index, node_id in enumerate(order)
        }

    for _ in range(max_fill_attempts):
        random.seed(rng.randrange(2**31))

        # Retry after task construction or wrong-input bank generation fails.
        try:
            return _instantiate(
                graph, pools, rng, task_numbers, sample_idx, get_input_attempts_per_task,
                pin_index, distinct_tasks,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not fill DAG after {max_fill_attempts} attempts: {last_error}")


# Stitch together task prompts in visible numerical order. With shuffled labels,
# this is deliberately not execution order.
def write_prompt(filled: dict[str, Any]) -> str:
    target = next(record for record in filled["nodes"] if record["node_id"] == filled["target_node"])
    target_number = target["task_number"]
    blocks = [
        f"Solve task {target_number}. Once you have its result, call the submit_answers "
        f"tool once with the result of task {target_number}."
    ]
    blocks.extend(record["prompt"].strip() for record in filled["nodes"])
    return "\n\n".join(blocks) + "\n"
