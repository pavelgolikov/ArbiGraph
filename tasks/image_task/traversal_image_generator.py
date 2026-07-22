import argparse
import json
import math
import os
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from tasks.math_task.math_helpers import adapt_scalar_compute, adapt_scalar_prompt


Point = tuple[float, float]
Segment = tuple[Point, Point]

DEFAULT_CANVAS_SIZE = (2000, 1400)
DEFAULT_MARGIN = 90
DEFAULT_NODE_RADIUS = 17
DEFAULT_LINE_WIDTH = 4
DEFAULT_NUM_PATHS = 8
DEFAULT_CROSSING_PARTNERS = 2
DEFAULT_MAX_CROSSING_INDEX_DISTANCE = 5
DEFAULT_PERTURBATION = 145.0
DEFAULT_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/opentype/stix-word/STIX-Bold.otf",
)


def save_json(data: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segment_vector(segment: Segment) -> Point:
    return _sub(segment[1], segment[0])


def _segment_length(segment: Segment) -> float:
    return _dist(segment[0], segment[1])


def point_segment_distance(point: Point, segment: Segment) -> float:
    a, b = segment
    ab = _sub(b, a)
    denom = _dot(ab, ab)
    if denom == 0:
        return _dist(point, a)
    t = max(0.0, min(1.0, _dot(_sub(point, a), ab) / denom))
    projection = (a[0] + t * ab[0], a[1] + t * ab[1])
    return _dist(point, projection)


def segment_distance(seg_a: Segment, seg_b: Segment) -> float:
    intersection = segment_intersection(seg_a, seg_b)
    if intersection is not None:
        return 0.0
    return min(
        point_segment_distance(seg_a[0], seg_b),
        point_segment_distance(seg_a[1], seg_b),
        point_segment_distance(seg_b[0], seg_a),
        point_segment_distance(seg_b[1], seg_a),
    )


def segment_intersection(seg_a: Segment, seg_b: Segment, eps: float = 1e-9) -> Point | None:
    """Return a proper segment intersection, excluding shared endpoints."""
    p, p2 = seg_a
    q, q2 = seg_b
    r = _sub(p2, p)
    s = _sub(q2, q)
    denom = _cross(r, s)
    if abs(denom) <= eps:
        return None

    qp = _sub(q, p)
    t = _cross(qp, s) / denom
    u = _cross(qp, r) / denom
    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        return (p[0] + t * r[0], p[1] + t * r[1])
    return None


def nearly_overlapping_segments(
    seg_a: Segment,
    seg_b: Segment,
    distance_threshold: float = 10.0,
    angle_threshold_degrees: float = 8.0,
) -> bool:
    va = _segment_vector(seg_a)
    vb = _segment_vector(seg_b)
    len_a = math.hypot(*va)
    len_b = math.hypot(*vb)
    if len_a == 0 or len_b == 0:
        return False

    sin_angle = abs(_cross(va, vb)) / (len_a * len_b)
    if sin_angle > math.sin(math.radians(angle_threshold_degrees)):
        return False
    return segment_distance(seg_a, seg_b) < distance_threshold


def path_segments(path: dict[str, Any]) -> list[Segment]:
    points = [tuple(point) for point in path["points"]]
    return list(zip(points[:-1], points[1:]))


def _path_display_tokens(path: dict[str, Any]) -> list[str]:
    if "display_tokens" in path:
        return [str(token) for token in path["display_tokens"]]
    return [str(label) for label in path["labels"]]


def _path_output_values(path: dict[str, Any]) -> list[int | float]:
    if "output_values" in path:
        return list(path["output_values"])
    return list(path["labels"])


def _has_duplicates(values: list[Any]) -> bool:
    return len(values) != len(set(values))


def all_nodes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for path in metadata["paths"]:
        path_nodes = path.get("nodes")
        if path_nodes is None:
            path_nodes = [
                {"point": point, "label": label}
                for point, label in zip(path["points"], path["labels"])
            ]
        for index, node in enumerate(path_nodes):
            label = node.get("label", node.get("output_value"))
            nodes.append(
                {
                    "path_id": path["path_id"],
                    "node_index": index,
                    "node_id": node.get("node_id"),
                    "point": tuple(node["point"]),
                    "label": label,
                    "display_token": node.get("display_token", str(label)),
                    "output_value": node.get("output_value", label),
                }
            )
    return nodes


def _segments_share_path_endpoint(seg_i: int, seg_j: int) -> bool:
    return abs(seg_i - seg_j) <= 1


def validate_traversal_metadata(
    metadata: dict[str, Any],
    min_node_distance: float = 34.0,
    min_crossing_node_distance: float = 38.0,
    min_crossing_distance: float = 24.0,
    min_segment_distance: float = 7.0,
    min_segment_length: float = 55.0,
) -> tuple[bool, dict[str, Any]]:
    paths = metadata.get("paths", [])
    if len(paths) < 2:
        return False, {"reason": "need_at_least_two_paths"}

    start_display_tokens = [_path_display_tokens(path)[0] for path in paths]
    if _has_duplicates(start_display_tokens):
        return False, {
            "reason": "duplicate_start_display_tokens",
            "display_tokens": start_display_tokens,
        }

    terminal_display_tokens = [_path_display_tokens(path)[-1] for path in paths]
    if _has_duplicates(terminal_display_tokens):
        return False, {
            "reason": "duplicate_terminal_display_tokens",
            "display_tokens": terminal_display_tokens,
        }

    start_values = [_path_output_values(path)[0] for path in paths]
    if _has_duplicates(start_values):
        return False, {"reason": "duplicate_start_values", "values": start_values}

    terminal_values = [_path_output_values(path)[-1] for path in paths]
    if _has_duplicates(terminal_values):
        return False, {"reason": "duplicate_terminal_values", "values": terminal_values}

    nodes = all_nodes(metadata)
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1:]:
            if _dist(node_a["point"], node_b["point"]) < min_node_distance:
                return False, {"reason": "nodes_too_close", "nodes": [node_a, node_b]}

    path_cross_counts = {path["path_id"]: 0 for path in paths}
    pair_cross_counts = {}
    crossings = []
    all_segments = []

    for path in paths:
        segments = path_segments(path)
        if len(segments) < 1:
            return False, {"reason": "path_too_short", "path_id": path["path_id"]}
        for segment_index, segment in enumerate(segments):
            if _segment_length(segment) < min_segment_length:
                return False, {
                    "reason": "segment_too_short",
                    "path_id": path["path_id"],
                    "segment_index": segment_index,
                }
            all_segments.append((path["path_id"], segment_index, segment))

        for i, seg_a in enumerate(segments):
            for j, seg_b in enumerate(segments[i + 1:], start=i + 1):
                if _segments_share_path_endpoint(i, j):
                    continue
                if segment_intersection(seg_a, seg_b) is not None:
                    return False, {"reason": "self_intersection", "path_id": path["path_id"]}
                if nearly_overlapping_segments(seg_a, seg_b):
                    return False, {"reason": "self_near_overlap", "path_id": path["path_id"]}

    for i, (path_a, seg_i, seg_a) in enumerate(all_segments):
        for path_b, seg_j, seg_b in all_segments[i + 1:]:
            if path_a == path_b:
                continue
            if nearly_overlapping_segments(seg_a, seg_b):
                return False, {
                    "reason": "inter_path_near_overlap",
                    "path_a": path_a,
                    "path_b": path_b,
                }

            crossing = segment_intersection(seg_a, seg_b)
            if crossing is not None:
                for node in nodes:
                    if _dist(crossing, node["point"]) < min_crossing_node_distance:
                        return False, {
                            "reason": "crossing_too_close_to_node",
                            "crossing": crossing,
                            "node": node,
                        }
                crossings.append(
                    {
                        "point": [round(crossing[0], 3), round(crossing[1], 3)],
                        "path_a": path_a,
                        "segment_a": seg_i,
                        "path_b": path_b,
                        "segment_b": seg_j,
                    }
                )
                path_cross_counts[path_a] += 1
                path_cross_counts[path_b] += 1
                pair_key = tuple(sorted((path_a, path_b)))
                pair_cross_counts[pair_key] = pair_cross_counts.get(pair_key, 0) + 1
            elif segment_distance(seg_a, seg_b) < min_segment_distance:
                return False, {
                    "reason": "segments_too_close_without_crossing",
                    "path_a": path_a,
                    "path_b": path_b,
                }

    for i, crossing_a in enumerate(crossings):
        point_a = tuple(crossing_a["point"])
        for crossing_b in crossings[i + 1:]:
            point_b = tuple(crossing_b["point"])
            if _dist(point_a, point_b) < min_crossing_distance:
                return False, {
                    "reason": "crossings_too_close",
                    "crossings": [crossing_a, crossing_b],
                }

    missing = [path_id for path_id, count in path_cross_counts.items() if count == 0]
    if missing:
        return False, {"reason": "path_without_intersection", "path_ids": missing}

    path_ids = [path["path_id"] for path in paths]
    required_pairs = metadata.get("layout", {}).get("required_crossing_pairs")
    if required_pairs is None:
        required_pairs = [
            [path_a, path_b]
            for i, path_a in enumerate(path_ids)
            for path_b in path_ids[i + 1:]
        ]

    missing_pairs = []
    for path_a, path_b in required_pairs:
        pair_key = tuple(sorted((path_a, path_b)))
        if pair_cross_counts.get(pair_key, 0) == 0:
            missing_pairs.append([path_a, path_b])
    if missing_pairs:
        return False, {"reason": "path_pairs_without_intersection", "path_pairs": missing_pairs}

    return True, {
        "reason": "valid",
        "crossings": crossings,
        "path_cross_counts": path_cross_counts,
        "pair_cross_counts": {
            f"{path_a}-{path_b}": count
            for (path_a, path_b), count in sorted(pair_cross_counts.items())
        },
    }


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _random_local_swap_schedule(
    num_paths: int,
    crossing_partners: int,
    max_crossing_index_distance: int,
    rng: random.Random,
    max_swaps: int | None = None,
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    """Return a bounded local braid schedule with controlled visual density."""
    target_degree = max(0, min(crossing_partners, num_paths - 1))
    max_distance = max(1, min(max_crossing_index_distance, num_paths - 1))
    if target_degree == 0:
        return [], set()
    if max_swaps is None:
        max_swaps = num_paths * num_paths

    best_degrees: list[int] | None = None
    start_parities = [0, 1]
    rng.shuffle(start_parities)

    for start_parity in start_parities:
        degrees = [0] * num_paths
        order = list(range(num_paths))
        schedule = []
        crossed_pairs: set[tuple[int, int]] = set()

        for wave in range(max(1, num_paths * target_degree * 2)):
            if len(schedule) >= max_swaps:
                break

            parity = (start_parity + wave) % 2
            positions = list(range(parity, num_paths - 1, 2))
            if rng.random() < 0.5:
                positions.reverse()

            for position in positions:
                if len(schedule) >= max_swaps:
                    break

                path_a = order[position]
                path_b = order[position + 1]
                pair = tuple(sorted((path_a, path_b)))
                if pair in crossed_pairs or abs(path_a - path_b) > max_distance:
                    continue

                schedule.append(
                    {
                        "start_parity": start_parity,
                        "wave": wave,
                        "position": position,
                        "path_a": path_a,
                        "path_b": path_b,
                        "required": True,
                        "duplicate_pair": False,
                    }
                )
                crossed_pairs.add(pair)
                degrees[path_a] += 1
                degrees[path_b] += 1
                order[position], order[position + 1] = order[position + 1], order[position]

            if min(degrees) >= target_degree:
                return schedule, crossed_pairs

        if best_degrees is None or min(degrees) > min(best_degrees):
            best_degrees = degrees

    degrees = best_degrees or [0] * num_paths
    missing = [path_id for path_id, degree in enumerate(degrees) if degree < target_degree]
    raise RuntimeError(
        "Could not realize enough local crossing partners: "
        f"path_ids={missing}, degrees={degrees}, target={target_degree}."
    )


def _sample_braid_paths(
    rng: random.Random,
    num_paths: int,
    canvas_size: tuple[int, int],
    margin: int,
    perturbation: float,
    crossing_partners: int,
    max_crossing_index_distance: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], set[tuple[int, int]]]:
    width, height = canvas_size
    swap_schedule, required_pairs = _random_local_swap_schedule(
        num_paths,
        crossing_partners,
        max_crossing_index_distance,
        rng,
    )
    num_intervals = len(swap_schedule) + 2
    x_positions = _linspace(margin, width - margin, num_intervals + 1)
    y_positions = _linspace(margin, height - margin, num_paths)

    order = list(range(num_paths))
    path_nodes = {path_id: [] for path_id in range(num_paths)}
    rendered_schedule = []

    def add_node(path_id: int, grid_x_index: int, lane: int, role: str) -> None:
        if path_nodes[path_id]:
            last = path_nodes[path_id][-1]["grid"]
            if last["x_index"] == grid_x_index and last["lane"] == lane:
                return

        grid_point = [round(x_positions[grid_x_index], 3), round(y_positions[lane], 3)]
        lane_spacing = (height - 2 * margin) / max(1, num_paths - 1)
        x_jitter_limit = min(float(perturbation), (x_positions[1] - x_positions[0]) * 0.34)
        y_jitter_limit = min(float(perturbation), lane_spacing * 0.40)

        if grid_x_index in (0, len(x_positions) - 1):
            dx = 0.0
            dy = rng.uniform(-y_jitter_limit, y_jitter_limit)
        else:
            dx = rng.uniform(-x_jitter_limit, x_jitter_limit)
            dy = rng.uniform(-y_jitter_limit, y_jitter_limit)

        point = [
            round(_clamp(grid_point[0] + dx, margin, width - margin), 3),
            round(_clamp(grid_point[1] + dy, margin, height - margin), 3),
        ]
        path_nodes[path_id].append(
            {
                "grid": {
                    "x_index": grid_x_index,
                    "lane": lane,
                    "point": grid_point,
                },
                "perturbation": [
                    round(point[0] - grid_point[0], 3),
                    round(point[1] - grid_point[1], 3),
                ],
                "point": point,
                "role": role,
            }
        )

    def add_all_nodes(grid_x_index: int, role: str) -> None:
        for lane, path_id in enumerate(order):
            add_node(path_id, grid_x_index, lane, role)

    for path_id, lane in enumerate(order):
        add_node(path_id, 0, lane, "start")

    for swap_index, swap in enumerate(swap_schedule, start=1):
        add_all_nodes(swap_index, "waypoint")
        left_position = swap["position"]
        path_a = order[left_position]
        path_b = order[left_position + 1]

        order[left_position], order[left_position + 1] = path_b, path_a
        add_all_nodes(swap_index + 1, "waypoint")
        rendered_schedule.append(
            {
                "interval_index": swap_index,
                "x_start": round(x_positions[swap_index], 3),
                "x_end": round(x_positions[swap_index + 1], 3),
                "position": left_position,
                "path_a": path_a,
                "path_b": path_b,
            }
        )

    # Right buffer interval after the last crossing.
    for lane, path_id in enumerate(order):
        add_node(path_id, len(x_positions) - 1, lane, "end")

    return [path_nodes[path_id] for path_id in range(num_paths)], rendered_schedule, required_pairs


def build_output_candidates(paths: list[dict[str, Any]]) -> dict[str, Any]:
    full_path_values = {
        str(path["path_id"]): _path_output_values(path)
        for path in paths
    }
    terminal_values = {
        str(path["path_id"]): _path_output_values(path)[-1]
        for path in paths
    }
    start_values = {
        str(path["path_id"]): _path_output_values(path)[0]
        for path in paths
    }
    full_path_display_tokens = {
        str(path["path_id"]): _path_display_tokens(path)
        for path in paths
    }
    full_path_node_ids = {
        str(path["path_id"]): list(path.get("node_ids", path["labels"]))
        for path in paths
    }
    return {
        "full_path_values": full_path_values,
        "terminal_values": terminal_values,
        "start_values": start_values,
        "full_path_display_tokens": full_path_display_tokens,
        "full_path_node_ids": full_path_node_ids,
        "full_path_labels": full_path_values,
        "terminal_labels": terminal_values,
        "start_labels": start_values,
    }


def normalize_traversal_selector(value: int | float, num_paths: int) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("Traversal selector input must be an int or float.")
    if num_paths < 1:
        raise ValueError("num_paths must be positive.")
    return int(
        adapt_scalar_compute(
            value,
            mod_value=num_paths,
            to_int=True,
            needs_abs=True,
        )
    )


def traversal_start_order(metadata: dict[str, Any]) -> list[int]:
    starts = []
    for path in metadata["paths"]:
        start_x, start_y = path["points"][0]
        starts.append((start_y, start_x, path["path_id"]))
    return [path_id for _y, _x, path_id in sorted(starts)]


def build_traversal_prompt(
    metadata: dict[str, Any],
    task_ind: int,
    input_name: str,
    output_name: str | None = None,
) -> str:
    num_paths = len(metadata["paths"])
    if output_name is None:
        output_name = f"task_{task_ind:d}_out"
    start_node_index = f"start_node_index_{task_ind:d}"
    preprocessing = adapt_scalar_prompt(
        input_name,
        start_node_index,
        mod_value=num_paths,
        to_int=True,
        needs_abs=True,
    )
    return (
        f"Task {task_ind:d}:\n"
        f"{preprocessing}"
        f"Follow the path whose starting node on the left has index {start_node_index} "
        "(top to bottom, starting at 0). "
        "Edges are straight line segments and don't change direction between nodes. "
        f"Return the numbers on that path from left to right as a list called {output_name}.\n"
    )


def build_traversal_task_payload(
    metadata: dict[str, Any],
    task_ind: int,
    input_name: str,
    input_value: int | float,
    output_name: str | None = None,
) -> dict[str, Any]:
    if output_name is None:
        output_name = f"task_{task_ind:d}_out"
    start_order = traversal_start_order(metadata)
    selected_start_index = normalize_traversal_selector(input_value, len(start_order))
    selected_path_id = start_order[selected_start_index]
    path_by_id = {path["path_id"]: path for path in metadata["paths"]}
    selected_path = path_by_id[selected_path_id]
    out = _path_output_values(selected_path)
    selected_start_number = _path_display_tokens(selected_path)[0]
    return {
        "prompt": build_traversal_prompt(
            metadata,
            task_ind,
            input_name,
            output_name,
        ),
        "out": out,
        "input_value": input_value,
        "input_name": input_name,
        "output_name": output_name,
        "selected_start_index": selected_start_index,
        "selected_start_number": selected_start_number,
        "selected_path_id": selected_path_id,
        "start_order": start_order,
    }


def generate_traversal_metadata(
    seed: int | None = None,
    num_paths: int = DEFAULT_NUM_PATHS,
    nodes_per_path: int = 0,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
    perturbation: float = DEFAULT_PERTURBATION,
    crossing_partners: int = DEFAULT_CROSSING_PARTNERS,
    max_crossing_index_distance: int = DEFAULT_MAX_CROSSING_INDEX_DISTANCE,
    max_attempts: int = 5000,
) -> dict[str, Any]:
    if perturbation < 0:
        raise ValueError("perturbation must be nonnegative.")
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2.")
    if crossing_partners < 1:
        raise ValueError("crossing_partners must be positive.")
    if max_crossing_index_distance < 1:
        raise ValueError("max_crossing_index_distance must be positive.")

    rng = random.Random(seed)
    last_validation = {"reason": "not_started"}

    for attempt in range(1, max_attempts + 1):
        path_nodes, swap_schedule, required_pairs = _sample_braid_paths(
            rng,
            num_paths,
            canvas_size,
            margin,
            perturbation,
            min(crossing_partners, num_paths - 1),
            min(max_crossing_index_distance, num_paths - 1),
        )
        label = 1
        paths = []
        for path_id, nodes in enumerate(path_nodes):
            node_ids = list(range(label, label + len(nodes)))
            output_values = list(node_ids)
            display_tokens = [str(value) for value in output_values]
            label += len(nodes)
            paths.append(
                {
                    "path_id": path_id,
                    "points": [node["point"] for node in nodes],
                    "nodes": [
                        {
                            **node,
                            "node_id": node_id,
                            "display_token": display_token,
                            "output_value": output_value,
                            "label": output_value,
                        }
                        for node, node_id, display_token, output_value
                        in zip(nodes, node_ids, display_tokens, output_values)
                    ],
                    "node_ids": node_ids,
                    "display_tokens": display_tokens,
                    "output_values": output_values,
                    "labels": output_values,
                }
            )

        metadata = {
            "canvas_size": list(canvas_size),
            "seed": seed,
            "attempt": attempt,
            "layout": {
                "type": "left_to_right_braid",
                "starts": "left",
                "ends": "right",
                "perturbation": perturbation,
                "crossing_partners": min(crossing_partners, num_paths - 1),
                "max_crossing_index_distance": min(max_crossing_index_distance, num_paths - 1),
                "required_crossing_pairs": [
                    [path_a, path_b]
                    for path_a, path_b in sorted(required_pairs)
                ],
                "swap_schedule": swap_schedule,
            },
            "visual_encoding": {
                "type": "numeric_text",
                "display_to_value": None,
            },
            "paths": paths,
            "output_candidates": build_output_candidates(paths),
        }
        valid, validation = validate_traversal_metadata(metadata)
        last_validation = validation
        if valid:
            metadata["validation"] = {
                "valid": True,
                "crossings": validation["crossings"],
                "path_cross_counts": validation["path_cross_counts"],
                "pair_cross_counts": validation["pair_cross_counts"],
            }
            return metadata

    raise RuntimeError(
        "Could not generate valid traversal metadata "
        f"after {max_attempts} attempts. Last validation: {last_validation}"
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in DEFAULT_FONT_PATHS:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_traversal_image(
    metadata: dict[str, Any],
    output_path: str,
    node_radius: int = DEFAULT_NODE_RADIUS,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> str:
    width, height = metadata["canvas_size"]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for path in metadata["paths"]:
        points = [tuple(point) for point in path["points"]]
        for start, end in zip(points[:-1], points[1:]):
            draw.line([start, end], fill="black", width=line_width)

    max_token_len = max(
        len(token)
        for path in metadata["paths"]
        for token in _path_display_tokens(path)
    )
    font_size = 18 if max_token_len <= 2 else 15
    font = _load_font(font_size)

    for path in metadata["paths"]:
        for point, token in zip(path["points"], _path_display_tokens(path)):
            x, y = point
            bbox = [
                x - node_radius,
                y - node_radius,
                x + node_radius,
                y + node_radius,
            ]
            draw.ellipse(bbox, fill="white", outline="black", width=3)
            text = str(token)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                (x - text_width / 2, y - text_height / 2 - 1),
                text,
                fill="black",
                font=font,
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path)
    return output_path


def generate_examples(
    output_dir: str,
    num_examples: int = 10,
    seed: int = 0,
    num_paths: int = DEFAULT_NUM_PATHS,
    nodes_per_path: int = 0,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    perturbation: float = DEFAULT_PERTURBATION,
    crossing_partners: int = DEFAULT_CROSSING_PARTNERS,
    max_crossing_index_distance: int = DEFAULT_MAX_CROSSING_INDEX_DISTANCE,
) -> list[dict[str, str]]:
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for index in range(num_examples):
        example_seed = seed + index
        metadata = generate_traversal_metadata(
            seed=example_seed,
            num_paths=num_paths,
            nodes_per_path=nodes_per_path,
            canvas_size=canvas_size,
            perturbation=perturbation,
            crossing_partners=crossing_partners,
            max_crossing_index_distance=max_crossing_index_distance,
        )
        stem = f"traversal_{index:03d}_seed{example_seed}"
        image_path = os.path.join(output_dir, f"{stem}.png")
        metadata_path = os.path.join(output_dir, f"{stem}.json")
        render_traversal_image(metadata, image_path)
        save_json(metadata, metadata_path)
        written.append({"image": image_path, "metadata": metadata_path})
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate traversal image examples.")
    parser.add_argument("--output-dir", default="scratch/traversal_image_examples")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)
    parser.add_argument("--perturbation", type=float, default=DEFAULT_PERTURBATION)
    parser.add_argument("--crossing-partners", type=int, default=DEFAULT_CROSSING_PARTNERS)
    parser.add_argument(
        "--max-crossing-index-distance",
        type=int,
        default=DEFAULT_MAX_CROSSING_INDEX_DISTANCE,
    )
    parser.add_argument(
        "--nodes-per-path",
        type=int,
        default=0,
        help="Deprecated for braid layouts; path length is determined by num_paths.",
    )
    parser.add_argument("--canvas-size", type=int, nargs=2, default=DEFAULT_CANVAS_SIZE)
    args = parser.parse_args()

    written = generate_examples(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        num_paths=args.num_paths,
        nodes_per_path=args.nodes_per_path,
        canvas_size=tuple(args.canvas_size),
        perturbation=args.perturbation,
        crossing_partners=args.crossing_partners,
        max_crossing_index_distance=args.max_crossing_index_distance,
    )
    for item in written:
        print(f"{item['image']} {item['metadata']}")


if __name__ == "__main__":
    main()
