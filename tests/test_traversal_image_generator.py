import os
import tempfile
import unittest

from tasks.math_task.math_helpers import adapt_scalar_prompt

from tasks.image_task.traversal_image_generator import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_CROSSING_PARTNERS,
    DEFAULT_MARGIN,
    DEFAULT_NUM_PATHS,
    DEFAULT_PERTURBATION,
    build_traversal_task_payload,
    generate_traversal_metadata,
    nearly_overlapping_segments,
    normalize_traversal_selector,
    render_traversal_image,
    segment_intersection,
    traversal_start_order,
    validate_traversal_metadata,
)


def fixed_metadata():
    paths = [
        {
            "path_id": 0,
            "points": [[50, 100], [250, 100]],
            "labels": [1, 2],
        },
        {
            "path_id": 1,
            "points": [[150, 40], [150, 260]],
            "labels": [3, 4],
        },
        {
            "path_id": 2,
            "points": [[70, 230], [250, 50]],
            "labels": [5, 6],
        },
    ]
    return {
        "canvas_size": [300, 300],
        "seed": 0,
        "attempt": 1,
        "paths": paths,
        "output_candidates": {
            "full_path_labels": {"0": [1, 2], "1": [3, 4], "2": [5, 6]},
            "terminal_labels": {"0": 2, "1": 4, "2": 6},
            "start_labels": {"0": 1, "1": 3, "2": 5},
        },
    }


class TraversalImageGeneratorTest(unittest.TestCase):
    def test_segment_intersection_excludes_endpoints(self):
        crossing = segment_intersection(((0, 0), (10, 10)), ((0, 10), (10, 0)))
        self.assertAlmostEqual(crossing[0], 5.0)
        self.assertAlmostEqual(crossing[1], 5.0)

        endpoint_touch = segment_intersection(((0, 0), (10, 0)), ((10, 0), (10, 10)))
        self.assertIsNone(endpoint_touch)

    def test_near_overlap_detection(self):
        self.assertTrue(
            nearly_overlapping_segments(
                ((0, 0), (100, 0)),
                ((10, 4), (90, 4)),
                distance_threshold=8,
            )
        )
        self.assertFalse(
            nearly_overlapping_segments(
                ((0, 0), (100, 0)),
                ((50, -50), (50, 50)),
                distance_threshold=8,
            )
        )

    def test_node_spacing_rejection(self):
        metadata = fixed_metadata()
        metadata["paths"][1]["points"][0] = [52, 102]

        valid, validation = validate_traversal_metadata(metadata)

        self.assertFalse(valid)
        self.assertEqual(validation["reason"], "nodes_too_close")

    def test_every_path_requires_inter_path_crossing(self):
        metadata = fixed_metadata()
        metadata["paths"][2]["points"] = [[40, 280], [260, 280]]

        valid, validation = validate_traversal_metadata(metadata)

        self.assertFalse(valid)
        self.assertEqual(validation["reason"], "path_without_intersection")
        self.assertIn(2, validation["path_ids"])

    def test_every_path_pair_requires_intersection(self):
        metadata = fixed_metadata()
        metadata["paths"][2]["points"] = [[40, 200], [260, 200]]

        valid, validation = validate_traversal_metadata(metadata)

        self.assertFalse(valid)
        self.assertEqual(validation["reason"], "path_pairs_without_intersection")
        self.assertIn([0, 2], validation["path_pairs"])

    def test_start_and_terminal_values_must_be_unique(self):
        metadata = fixed_metadata()
        metadata["paths"][1]["labels"] = [1, 4]

        valid, validation = validate_traversal_metadata(metadata)

        self.assertFalse(valid)
        self.assertEqual(validation["reason"], "duplicate_start_display_tokens")

        metadata = fixed_metadata()
        metadata["paths"][1]["labels"] = [3, 2]

        valid, validation = validate_traversal_metadata(metadata)

        self.assertFalse(valid)
        self.assertEqual(validation["reason"], "duplicate_terminal_display_tokens")

    def test_fixed_metadata_validates_and_renders(self):
        metadata = fixed_metadata()
        valid, validation = validate_traversal_metadata(metadata)
        metadata["validation"] = {
            "valid": valid,
            "crossings": validation.get("crossings", []),
            "path_cross_counts": validation.get("path_cross_counts", {}),
            "pair_cross_counts": validation.get("pair_cross_counts", {}),
        }

        self.assertTrue(valid)
        self.assertEqual(metadata["output_candidates"]["full_path_labels"]["0"], [1, 2])
        self.assertEqual(metadata["output_candidates"]["terminal_labels"]["2"], 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "fixed.png")
            render_traversal_image(metadata, image_path)
            self.assertTrue(os.path.exists(image_path))
            self.assertGreater(os.path.getsize(image_path), 0)

    def test_random_generation_produces_valid_metadata(self):
        metadata = generate_traversal_metadata(
            seed=17,
            num_paths=4,
            canvas_size=DEFAULT_CANVAS_SIZE,
            perturbation=34,
        )

        self.assertTrue(metadata["validation"]["valid"])
        self.assertEqual(len(metadata["paths"]), 4)
        labels = [label for path in metadata["paths"] for label in path["labels"]]
        self.assertEqual(len(labels), len(set(labels)))
        node_ids = [node["node_id"] for path in metadata["paths"] for node in path["nodes"]]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        start_values = list(metadata["output_candidates"]["start_values"].values())
        terminal_values = list(metadata["output_candidates"]["terminal_values"].values())
        self.assertEqual(len(start_values), len(set(start_values)))
        self.assertEqual(len(terminal_values), len(set(terminal_values)))
        self.assertTrue(all(count >= 1 for count in metadata["validation"]["path_cross_counts"].values()))
        required_pairs = {
            tuple(pair)
            for pair in metadata["layout"]["required_crossing_pairs"]
        }
        actual_pairs = {
            tuple(map(int, key.split("-")))
            for key in metadata["validation"]["pair_cross_counts"]
        }
        self.assertGreaterEqual(
            len(required_pairs),
            4 * min(DEFAULT_CROSSING_PARTNERS, 3) // 2,
        )
        self.assertTrue(required_pairs.issubset(actual_pairs))
        for path in metadata["paths"]:
            self.assertAlmostEqual(path["points"][0][0], DEFAULT_MARGIN)
            self.assertAlmostEqual(path["points"][-1][0], DEFAULT_CANVAS_SIZE[0] - DEFAULT_MARGIN)
            self.assertGreater(path["points"][-1][0], path["points"][0][0])
            self.assertEqual(len(path["nodes"]), len(path["points"]))
            for node, point, label in zip(path["nodes"], path["points"], path["labels"]):
                self.assertEqual(node["point"], point)
                self.assertEqual(node["label"], label)
                self.assertEqual(node["output_value"], label)
                self.assertEqual(node["display_token"], str(label))
                self.assertIn("node_id", node)
                self.assertIn("grid", node)
                self.assertIn("perturbation", node)
                self.assertIn("role", node)

        perturbations = [
            node["perturbation"]
            for path in metadata["paths"]
            for node in path["nodes"]
        ]
        self.assertTrue(any(dx != 0 or dy != 0 for dx, dy in perturbations))
        self.assertEqual(metadata["layout"]["type"], "left_to_right_braid")
        self.assertEqual(metadata["layout"]["perturbation"], 34)

    def test_perturbation_parameter_is_respected(self):
        metadata = generate_traversal_metadata(
            seed=19,
            num_paths=4,
            canvas_size=DEFAULT_CANVAS_SIZE,
            perturbation=12,
        )

        self.assertEqual(metadata["layout"]["perturbation"], 12)
        for path in metadata["paths"]:
            for node in path["nodes"]:
                dx, dy = node["perturbation"]
                self.assertLessEqual(abs(dx), 12)
                self.assertLessEqual(abs(dy), 12)

    def test_negative_perturbation_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_traversal_metadata(perturbation=-1)

    def test_zero_crossing_partners_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_traversal_metadata(crossing_partners=0)

    def test_selector_normalization_uses_round_abs_and_modulo(self):
        self.assertEqual(normalize_traversal_selector(-9, 8), 1)
        self.assertEqual(normalize_traversal_selector(2.5, 8), 3)
        self.assertEqual(normalize_traversal_selector(-2.5, 8), 3)

    def test_prompt_payload_selects_start_node_by_modded_scalar(self):
        metadata = generate_traversal_metadata(seed=23)
        payload = build_traversal_task_payload(
            metadata,
            task_ind=3,
            input_name="task_2_out",
            input_value=-17,
        )
        start_order = traversal_start_order(metadata)
        selected_path_id = start_order[1]
        expected = metadata["output_candidates"]["full_path_values"][str(selected_path_id)]

        self.assertEqual(payload["selected_start_index"], 1)
        self.assertEqual(payload["selected_path_id"], selected_path_id)
        self.assertEqual(payload["out"], expected)
        self.assertEqual(payload["selected_start_number"], str(expected[0]))
        self.assertIn(
            adapt_scalar_prompt(
                "task_2_out",
                "start_node_index_3",
                mod_value=8,
                to_int=True,
                needs_abs=True,
            ),
            payload["prompt"],
        )
        self.assertIn("starting node on the left has index start_node_index_3", payload["prompt"])
        self.assertNotIn(f"node numbered {expected[0]}", payload["prompt"])
        self.assertIn("straight line segments", payload["prompt"])
        self.assertIn("task_3_out", payload["prompt"])

    def test_default_generation_uses_eight_path_local_braid(self):
        metadata = generate_traversal_metadata(seed=23)

        self.assertEqual(len(metadata["paths"]), DEFAULT_NUM_PATHS)
        self.assertEqual(metadata["layout"]["crossing_partners"], DEFAULT_CROSSING_PARTNERS)
        self.assertEqual(metadata["layout"]["perturbation"], DEFAULT_PERTURBATION)
        required_pairs = {
            tuple(pair)
            for pair in metadata["layout"]["required_crossing_pairs"]
        }
        self.assertGreaterEqual(len(required_pairs), DEFAULT_NUM_PATHS * DEFAULT_CROSSING_PARTNERS // 2)
        self.assertTrue(
            required_pairs.issubset(
                {tuple(map(int, key.split("-"))) for key in metadata["validation"]["pair_cross_counts"]}
            )
        )


if __name__ == "__main__":
    unittest.main()
