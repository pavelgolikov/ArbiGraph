import os
import sys
import unittest


SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from tasks.adapters import (
    Adapter,
    FullListDependencyAdapter,
    ListToListAdapter,
    SumListsAdapter,
)


class FullListDependencyAdapterTest(unittest.TestCase):
    def test_compute_prepends_full_list_length_and_sum(self):
        adapter = FullListDependencyAdapter()

        self.assertEqual(adapter.compute([1, 2, 3]), [3, 6, 1, 2, 3])
        self.assertEqual(adapter.compute([]), [0, 0])

        prompt = adapter.prompt('task_1_out["result"]', "task_1_out_fd")
        self.assertIn(
            'Suppose task_1_out["result"] = [x_1, ..., x_n].',
            prompt,
        )
        self.assertIn(
            "Let task_1_out_fd = [n, s, x_1, ..., x_n], "
            "where s = x_1 + ... + x_n.",
            prompt,
        )

    def test_list_to_list_applies_full_dependency_before_truncation(self):
        adapter = ListToListAdapter(mod_value=100, list_len_max=2, to_int=True)

        self.assertEqual(adapter.compute([1, 2, 3, 4]), [4, 10])
        self.assertEqual(adapter.compute([1, 2, 3, 5]), [4, 11])

        prompt = adapter.prompt('task_1_out["result"]', "list_2")
        dependency = prompt.index('Suppose task_1_out["result"] = [x_1, ..., x_n]')
        preprocessing = prompt.index(
            "Let list_2 be the result of preprocessing task_1_out_fd"
        )
        truncation = prompt.index("keep only the first 2 elements")
        self.assertLess(dependency, preprocessing)
        self.assertLess(preprocessing, truncation)

    def test_list_to_list_skips_full_dependency_without_exactly_one_parent(self):
        for num_parents, input_name in ((0, "task_1_input"), (2, "list_2_join")):
            with self.subTest(num_parents=num_parents):
                adapter = ListToListAdapter(100, 2, num_parents, None, False, True)

                self.assertEqual(adapter.compute([1, 2, 3, 4]), [1, 2])

                prompt = adapter.prompt(input_name, "list_2")
                self.assertNotIn("[x_1, ..., x_n]", prompt)
                self.assertNotIn("_fd", prompt)
                self.assertIn(
                    f"Let list_2 be the result of preprocessing {input_name}",
                    prompt,
                )

    def test_sum_lists_groups_parent_dependency_prompt_once(self):
        adapter = SumListsAdapter()
        names = [
            'task_1_out["result"]',
            'task_5_out["result"]',
            'task_8_out["result"]',
        ]

        self.assertEqual(
            adapter.compute([[1, 2], [10, 20, 30], [100]]),
            [6, 163, 111, 22, 30],
        )

        prompt = adapter.prompt(names, "list_14_join")
        grouped = (
            'Pair the input lists task_1_out["result"], task_5_out["result"], and '
            'task_8_out["result"] with the output variables task_1_out_fd, '
            'task_5_out_fd, and task_8_out_fd, respectively.'
        )
        join = "Let list_14_join be the result of adding the following lists elementwise"
        self.assertEqual(prompt.count("For each pair independently"), 1)
        self.assertEqual(prompt.count("where s = x_1 + ... + x_n"), 1)
        self.assertEqual(prompt.count("output variable be [n, s, x_1, ..., x_n]"), 1)
        self.assertIn(grouped, prompt)
        self.assertIn(
            "adding the following lists elementwise: task_1_out_fd, task_5_out_fd, task_8_out_fd.",
            prompt,
        )
        self.assertLess(prompt.index(grouped), prompt.index(join))

        downstream = ListToListAdapter(0, 10, 3)
        self.assertEqual(
            downstream.compute(adapter.compute([[1, 2], [10, 20, 30], [100]])),
            [6, 163, 111, 22, 30],
        )
        combined_prompt = prompt + downstream.prompt("list_14_join", "list_14")
        self.assertEqual(combined_prompt.count("where s = x_1 + ... + x_n"), 1)
        self.assertNotIn("list_14_join_fd", combined_prompt)


if __name__ == "__main__":
    unittest.main()
