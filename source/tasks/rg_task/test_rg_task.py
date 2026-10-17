import os
import sys
import unittest


SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from reasoning_gym import create_dataset

from tasks.rg_task.rg_task import RG_TASKS, _extract_root_input, bad_output, load_pool


class ReasoningGymIntegrationTest(unittest.TestCase):
    SCALARIZED_CLASSES = tuple(
        task
        for task in RG_TASKS
        if task.task_name
        in {
            "products",
            "aiw",
            "basic_arithmetic",
            "chain_sum",
            "dice",
            "fraction_simplification",
            "number_filtering",
            "number_sorting",
            "simple_equations",
            "binary_matrix",
            "largest_island",
            "game_of_life",
            "game_of_life_halting",
            "rotten_oranges",
            "spiral_matrix",
            "rotate_matrix",
            "manipulate_matrix",
            "pool_matrix",
            "course_schedule",
            "binary_alternation",
            "string_insertion",
            "string_manipulation",
            "isomorphic_strings",
            "ransom_note",
            "caesar_cipher",
            "bitwise_arithmetic",
            "bf",
            "emoji_mystery",
            "letter_counting",
            "maze",
            "leg_counting",
            "ab",
            "word_sorting",
            "word_sequence_reversal",
            "syllogism",
            "mahjong_puzzle",
        }
    )

    def test_pool_ids_and_names_are_unique(self):
        pool = load_pool()
        self.assertEqual([task["task_id"] for task in pool], list(range(53)))
        self.assertEqual(len({task["task_name"] for task in pool}), 53)

    def test_output_validation_uses_declared_type(self):
        self.assertFalse(bad_output(0, "scalar"))
        self.assertFalse(bad_output(False, "bool"))
        self.assertFalse(bad_output("", "text"))
        self.assertFalse(bad_output([1], "list"))
        self.assertTrue(bad_output(True, "scalar"))
        self.assertTrue(bad_output([], "list"))

    def test_path_star_maps_scalar_to_rg_path_length(self):
        task_class = next(task for task in RG_TASKS if task.task_name == "path_star")
        dataset = create_dataset("path_star", seed=0, size=1)
        entry = dataset[0]
        root_input = _extract_root_input("path_star", entry, dataset)
        task = task_class(
            1,
            ['task_1_input["result"]'],
            root_input,
            dataset,
            entry,
        )

        self.assertEqual(task_class.input_type, "scalar")
        self.assertEqual(task.adapted_inp["path_length"], entry["metadata"]["path_length"])
        self.assertEqual(task.out, [int(node) for node in task.rg_entry["answer"].split()])
        self.assertEqual(len(task.out), task.adapted_inp["path_length"] + 1)
        self.assertIn("Reasoning Gym's path_length parameter", task.prompt)

    def test_circuit_logic_maps_scalar_to_rg_num_terms(self):
        task_class = next(task for task in RG_TASKS if task.task_name == "circuit_logic")
        dataset = create_dataset("circuit_logic", seed=0, size=1)
        entry = dataset[0]
        root_input = _extract_root_input("circuit_logic", entry, dataset)
        task = task_class(
            1,
            ['task_1_input["result"]'],
            root_input,
            dataset,
            entry,
        )

        self.assertEqual(task_class.input_type, "scalar")
        self.assertEqual(task.adapted_inp["num_terms"], len(entry["metadata"]["term_strings"]))
        self.assertEqual(len(task.rg_entry["metadata"]["term_strings"]), task.adapted_inp["num_terms"])
        self.assertEqual(task.out, bool(int(task.rg_entry["answer"])))
        self.assertIn("Reasoning Gym's num_terms parameter", task.prompt)

    def test_sudoku_tasks_map_scalar_to_rg_num_empty(self):
        for task_name in ("sudoku", "mini_sudoku"):
            with self.subTest(task=task_name):
                task_class = next(task for task in RG_TASKS if task.task_name == task_name)
                dataset = create_dataset(task_name, seed=0, size=1)
                entry = dataset[0]
                root_input = _extract_root_input(task_name, entry, dataset)
                task = task_class(
                    1,
                    ['task_1_input["result"]'],
                    root_input,
                    dataset,
                    entry,
                )

                expected = [value for row in task.rg_entry["metadata"]["solution"] for value in row]
                self.assertEqual(task_class.input_type, "scalar")
                self.assertEqual(task.adapted_inp["num_empty"], entry["metadata"]["num_empty"])
                self.assertEqual(task.out, expected)
                self.assertIn("Reasoning Gym's num_empty parameter", task.prompt)

    def test_futoshiki_maps_scalar_to_rg_board_size(self):
        task_class = next(task for task in RG_TASKS if task.task_name == "futoshiki")
        dataset = create_dataset("futoshiki", seed=0, size=1)
        entry = dataset[0]
        root_input = _extract_root_input("futoshiki", entry, dataset)
        task = task_class(
            1,
            ['task_1_input["result"]'],
            root_input,
            dataset,
            entry,
        )

        expected = [value for row in task.rg_entry["metadata"]["solution"] for value in row]
        self.assertEqual(task_class.input_type, "scalar")
        self.assertEqual(task.adapted_inp["board_size"], entry["metadata"]["board_size"])
        self.assertEqual(task.rg_entry["metadata"]["board_size"], task.adapted_inp["board_size"])
        self.assertEqual(task.out, expected)
        self.assertIn("Reasoning Gym's board_size parameter", task.prompt)

    def test_scalarized_tasks_use_selected_rg_parameter_and_rg_answer(self):
        for task_class in self.SCALARIZED_CLASSES:
            with self.subTest(task=task_class.task_name):
                dataset = create_dataset(task_class.task_name, seed=0, size=1)
                entry = dataset[0]
                root_input = _extract_root_input(task_class.task_name, entry, dataset)
                task = task_class(
                    1,
                    ['task_1_input["result"]'],
                    root_input,
                    dataset,
                    entry,
                )

                metadata = task.rg_entry["metadata"]
                selected = next(iter(task.adapted_inp.values()))
                if task_class.task_name == "products":
                    self.assertEqual(metadata["num_digits"], selected)
                elif task_class.task_name == "aiw":
                    self.assertEqual(
                        metadata["task_type"],
                        dataset.config.task_types[selected].value,
                    )
                elif task_class.task_name in {"basic_arithmetic", "chain_sum"}:
                    self.assertEqual(metadata["num_terms"], selected)
                elif task_class.task_name == "dice":
                    self.assertEqual(metadata["difficulty"]["num_dice"], selected)
                elif task_class.task_name == "fraction_simplification":
                    self.assertEqual(metadata["factor"], selected)
                elif task_class.task_name in {"number_filtering", "number_sorting"}:
                    self.assertEqual(metadata["numbers"], selected)
                elif task_class.task_name == "simple_equations":
                    self.assertEqual(metadata["difficulty"]["min_terms"], selected)
                    self.assertEqual(metadata["difficulty"]["max_terms"], selected)
                elif task_class.task_name == "binary_matrix":
                    self.assertEqual(metadata["n"], selected)
                elif task_class.task_name == "largest_island":
                    self.assertEqual(len(metadata["grid"]), selected)
                    self.assertTrue(all(len(row) == selected for row in metadata["grid"]))
                elif task_class.task_name == "game_of_life":
                    self.assertEqual(metadata["simulation_steps"], selected)
                elif task_class.task_name == "game_of_life_halting":
                    self.assertEqual(metadata["simulation_steps"], selected)
                elif task_class.task_name in {
                    "rotten_oranges",
                    "spiral_matrix",
                    "binary_alternation",
                }:
                    self.assertEqual(metadata["n"], selected)
                elif task_class.task_name == "rotate_matrix":
                    self.assertEqual(metadata["num_rotations"], selected)
                elif task_class.task_name == "manipulate_matrix":
                    self.assertEqual(metadata["num_transforms"], selected)
                elif task_class.task_name == "pool_matrix":
                    self.assertEqual(metadata["pool_size"], selected)
                elif task_class.task_name == "course_schedule":
                    self.assertEqual(len(metadata["courses"]), selected)
                elif task_class.task_name in {"string_insertion", "isomorphic_strings"}:
                    self.assertEqual(metadata["string_length"], selected)
                elif task_class.task_name == "string_manipulation":
                    self.assertEqual(metadata["num_rules"], selected)
                elif task_class.task_name == "ransom_note":
                    self.assertEqual(metadata["note_length"], selected)
                elif task_class.task_name == "caesar_cipher":
                    self.assertEqual(metadata["rotation"], selected)
                elif task_class.task_name in {"bitwise_arithmetic", "bf"}:
                    self.assertEqual(metadata["difficulty"]["difficulty"], selected)
                elif task_class.task_name == "emoji_mystery":
                    self.assertEqual(metadata["num_words_in_sentence"], selected)
                elif task_class.task_name == "letter_counting":
                    self.assertEqual(metadata["span_length"], selected)
                elif task_class.task_name == "maze":
                    self.assertEqual(metadata["shortest_path_length"], selected)
                elif task_class.task_name == "leg_counting":
                    self.assertEqual(metadata["num_animals"], selected)
                elif task_class.task_name == "ab":
                    self.assertEqual(metadata["difficulty"]["length"], selected)
                elif task_class.task_name in {"word_sorting", "word_sequence_reversal"}:
                    self.assertEqual(metadata["num_words"], selected)
                elif task_class.task_name == "syllogism":
                    self.assertEqual(metadata["type"], "inversion" if selected else "syllogism")
                elif task_class.task_name == "mahjong_puzzle":
                    self.assertEqual(len(metadata["rounds"]), selected)

                self.assertEqual(task_class.input_type, "scalar")
                self.assertFalse(bad_output(task.out, task_class.output_type))
                self.assertIn("Reasoning Gym", task.prompt)


if __name__ == "__main__":
    unittest.main()
