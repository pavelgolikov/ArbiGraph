"""Render a strict custom-DAG graph JSON to SVG with Graphviz dot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import networkx as nx

from generate_custom import load_graph_data


def dot_label(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def graph_to_dot(graph: nx.DiGraph, rankdir: str) -> str:
    lines = [
        "digraph custom_dag {",
        f"  graph [rankdir={json.dumps(rankdir)}, bgcolor=\"white\", pad=\"0.2\", nodesep=\"0.45\", ranksep=\"0.65\"];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10, margin=\"0.08,0.06\"];",
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#555555\", arrowsize=0.7];",
    ]

    target_node = graph.graph["target_node"]
    # Emit nodes in dependency order so the DOT source mirrors the generated prompt numbering.
    for node_id in nx.topological_sort(graph):
        attrs = graph.nodes[node_id]
        node_name = json.dumps(str(node_id))
        if attrs["kind"] == "task":
            task_id = f"\ntask_id={attrs['task_id']}" if "task_id" in attrs else ""
            label = (
                f"{node_id}\n"
                f"{attrs['family']} {attrs.get('input_kind', '?')}->{attrs.get('output_kind', '?')}"
                f"{task_id}"
            )
            fill = "#fff2b8" if node_id == target_node else "#e8f1ff"
            border = "#c48a00" if node_id == target_node else "#5279b8"
            penwidth = "2.0" if node_id == target_node else "1.2"
            lines.append(
                f"  {node_name} [label={dot_label(label)}, fillcolor={json.dumps(fill)}, "
                f"color={json.dumps(border)}, penwidth={penwidth}];"
            )
        else:
            label = f"{node_id}\n{attrs['op']}"
            if attrs["op"] == "pick":
                label += f" index={attrs['index']}"
            lines.append(
                f"  {node_name} [label={dot_label(label)}, shape=ellipse, "
                "fillcolor=\"#f1f1f1\", color=\"#777777\"];"
            )

    # Pack slots are the only edge metadata that affects custom-DAG semantics.
    for source, destination, attrs in graph.edges(data=True):
        label = f" [label={dot_label('slot ' + str(attrs['slot']))}]" if "slot" in attrs else ""
        lines.append(f"  {json.dumps(str(source))} -> {json.dumps(str(destination))}{label};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Strict NetworkX node-link JSON graph file.")
    parser.add_argument("--output", required=True, help="SVG output path.")
    parser.add_argument("--rankdir", choices=["LR", "TB"], default="LR")
    args = parser.parse_args()

    with open(args.graph, "r", encoding="utf-8") as handle:
        graph = load_graph_data(json.load(handle))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    subprocess.run(
        ["dot", "-Tsvg", "-o", args.output],
        input=graph_to_dot(graph, args.rankdir),
        text=True,
        check=True,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
