"""Parse and render new-format custom DAG topology JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


def format_json(data: dict[str, Any]) -> str:
    from fracturedjson import Encoder

    return json.dumps(data, cls=Encoder, indent=2) + "\n"


def parse_custom_dag(data: dict[str, Any]) -> nx.DiGraph:
    """Parse new-format custom DAG JSON into a validated NetworkX DiGraph."""
    if not isinstance(data, dict):
        raise ValueError("Custom DAG data must be a JSON object.")

    # The graph must be directed and non-multigraph.
    if "directed" not in data or data["directed"] is not True:
        raise ValueError("Custom DAG must declare directed=true.")
    if "multigraph" not in data or data["multigraph"] is not False:
        raise ValueError("Custom DAG must declare multigraph=false.")

    # The target identifies the evaluated task node.
    if "target" not in data or not isinstance(data["target"], dict):
        raise ValueError("Custom DAG must declare target as a JSON object.")
    target = data["target"]
    if "id" not in target or "native_task_id" not in target:
        raise ValueError("Custom DAG target must declare target 'id' and target 'native_task_id'.")
    target_id = target["id"]

    # Each node declares its task category.
    if "nodes" not in data or not isinstance(data["nodes"], list):
        raise ValueError("Custom DAG must declare nodes as a list.")
    nodes = data["nodes"]

    node_ids = []
    declared_node_ids = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or "id" not in node or "category" not in node:
            raise ValueError(f"Custom DAG node at index {index} must declare 'id' and 'category'.")
        node_id = node["id"]
        node_ids.append(node_id)
        declared_node_ids.add(node_id)

    # Node ids must uniquely identify topology nodes.
    if len(node_ids) != len(declared_node_ids):
        seen = set()
        duplicates = []
        for node_id in node_ids:
            if node_id in seen and node_id not in duplicates:
                duplicates.append(node_id)
            seen.add(node_id)
        raise ValueError(f"Custom DAG contains duplicate node id(s): {duplicates!r}.")

    # Target id must refer to a declared topology node.
    if target_id not in declared_node_ids:
        raise ValueError(f"Custom DAG target id {target_id!r} is not a declared node id.")

    # Edges declare directed dependencies.
    if "edges" not in data or not isinstance(data["edges"], list):
        raise ValueError("Custom DAG must declare edges as a list.")
    edges = data["edges"]

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict) or "source" not in edge or "destination" not in edge:
            raise ValueError(f"Custom DAG edge at index {index} must declare 'source' and 'destination'.")
        source = edge["source"]
        destination = edge["destination"]
        # Edge endpoints must refer to declared nodes.
        if source not in declared_node_ids:
            raise ValueError(f"Custom DAG edge at index {index} has unknown source {source!r}.")
        if destination not in declared_node_ids:
            raise ValueError(f"Custom DAG edge at index {index} has unknown destination {destination!r}.")

    graph = json_graph.node_link_graph(
        data,
        source="source",
        target="destination",
        name="id",
        edges="edges",
    )
    if type(graph) is not nx.DiGraph:
        raise ValueError("Custom DAG JSON did not produce a NetworkX DiGraph.")
    graph.graph["target"] = target
    graph.graph["edge_order"] = [(edge["source"], edge["destination"]) for edge in edges]

    # DAG topology must be acyclic.
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Custom DAG must not contain cycles.")

    return graph


def _custom_dag_to_dot(graph: nx.DiGraph, rankdir: str) -> str:
    if rankdir not in {"LR", "TB"}:
        raise ValueError("rankdir must be either 'LR' or 'TB'.")

    target = graph.graph["target"]
    target_id = target["id"]

    lines = [
        "digraph custom_dag {",
        f"  graph [rankdir={json.dumps(rankdir)}, bgcolor=\"white\", pad=\"0.2\", nodesep=\"0.45\", ranksep=\"0.65\"];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10, margin=\"0.08,0.06\"];",
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#555555\", arrowsize=0.7];",
    ]

    for node_id in nx.topological_sort(graph):
        attrs = graph.nodes[node_id]
        label = f"{node_id}\n{attrs['category']}"
        fill = "#fff2b8" if node_id == target_id else "#e8f1ff"
        border = "#c48a00" if node_id == target_id else "#5279b8"
        penwidth = "2.0" if node_id == target_id else "1.2"
        if node_id == target_id:
            label += f"\nnative_task_id={target['native_task_id']}"
        lines.append(
            f"  {json.dumps(str(node_id))} [label={json.dumps(label)}, fillcolor={json.dumps(fill)}, "
            f"color={json.dumps(border)}, penwidth={penwidth}];"
        )

    for source, destination in graph.edges:
        lines.append(f"  {json.dumps(str(source))} -> {json.dumps(str(destination))};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_custom_dag_svg(graph: nx.DiGraph, image_path: str, rankdir: str = "LR") -> None:
    """Render a parsed custom DAG graph to an SVG image with Graphviz."""
    image_dir = os.path.dirname(os.path.abspath(image_path))
    os.makedirs(image_dir, exist_ok=True)
    try:
        subprocess.run(
            ["dot", "-Tsvg", "-o", image_path],
            input=_custom_dag_to_dot(graph, rankdir),
            text=True,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"Graphviz failed to render custom DAG SVG: {details}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="New-format NetworkX node-link DAG JSON file.")
    parser.add_argument("--image", required=True, help="SVG image path to write.")
    parser.add_argument("--rankdir", choices=["LR", "TB"], default="LR")
    args = parser.parse_args()

    with open(args.graph, "r", encoding="utf-8") as handle:
        graph = parse_custom_dag(json.load(handle))

    render_custom_dag_svg(graph, args.image, args.rankdir)
    print(f"Wrote {args.image}")


if __name__ == "__main__":
    main()
