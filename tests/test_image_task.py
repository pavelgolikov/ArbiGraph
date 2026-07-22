import os
import tempfile
import unittest

from tasks.image_task import ImageTask


class ImageTaskTest(unittest.TestCase):
    def make_image(self):
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.write(b"not a real image, but a local task image path")
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def test_scalar_output_and_metadata_are_preserved(self):
        image = self.make_image()
        metadata = {
            "source": "traversalbench",
            "selection_rule": "selected_path = task_1_input % num_paths",
            "selection_mod": 4,
            "selected_path_index": 2,
        }

        task = ImageTask(
            name="traversal_path",
            task_ind=1,
            inp={"scalar": {"chain_name": "task_1_input", "value": 17}},
            scalar_max_mag=100,
            list_len_max=10,
            image=image,
            prompt="Task 1: follow the selected path and call the result task_1_out.",
            out=8,
            metadata=metadata,
        )

        self.assertEqual(task.name, "traversal_path")
        self.assertEqual(task.task_ind, 1)
        self.assertEqual(task.image, image)
        self.assertEqual(task.prompt, "Task 1: follow the selected path and call the result task_1_out.")
        self.assertEqual(task.out, 8)
        self.assertEqual(task.metadata, metadata)
        self.assertEqual(task.input_kind, "scalar")
        self.assertEqual(task.output_kind, "scalar")

    def test_list_output(self):
        image = self.make_image()

        task = ImageTask(
            name="traversal_path",
            task_ind=3,
            inp={"scalar": {"chain_name": "task_2_out", "value": -4}},
            scalar_max_mag=100,
            list_len_max=10,
            image=image,
            prompt="Task 3: output the full remapped path as task_3_out.",
            out=[3, 1, 4, 1, 5],
            metadata={"output_mode": "full_path"},
        )

        self.assertEqual(task.out, [3, 1, 4, 1, 5])
        self.assertEqual(task.input_kind, "scalar")
        self.assertEqual(task.output_kind, "list")

    def test_list_input_kind_is_inferred(self):
        image = self.make_image()

        task = ImageTask(
            name="visual_list_param",
            task_ind=2,
            inp={"values": {"chain_name": "task_1_out", "value": [1, 2, 3]}},
            scalar_max_mag=100,
            list_len_max=10,
            image=image,
            prompt="Task 2: use the list-valued input and call the result task_2_out.",
            out=2.5,
        )

        self.assertEqual(task.input_kind, "list")
        self.assertEqual(task.output_kind, "scalar")

    def test_missing_image_path_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            ImageTask(
                name="bad_image",
                task_ind=1,
                inp={"scalar": {"chain_name": "task_1_input", "value": 1}},
                scalar_max_mag=100,
                list_len_max=10,
                image="/tmp/does-not-exist-cmb-image-task.png",
                prompt="Task 1: answer task_1_out.",
                out=1,
            )

    def test_string_output_is_rejected(self):
        image = self.make_image()

        with self.assertRaises(TypeError):
            ImageTask(
                name="bad_output",
                task_ind=1,
                inp={"scalar": {"chain_name": "task_1_input", "value": 1}},
                scalar_max_mag=100,
                list_len_max=10,
                image=image,
                prompt="Task 1: answer task_1_out.",
                out="red square",
            )

    def test_nested_list_output_is_rejected(self):
        image = self.make_image()

        with self.assertRaises(TypeError):
            ImageTask(
                name="bad_output",
                task_ind=1,
                inp={"scalar": {"chain_name": "task_1_input", "value": 1}},
                scalar_max_mag=100,
                list_len_max=10,
                image=image,
                prompt="Task 1: answer task_1_out.",
                out=[[1, 2], [3, 4]],
            )

    def test_mixed_numeric_and_non_numeric_list_output_is_rejected(self):
        image = self.make_image()

        with self.assertRaises(TypeError):
            ImageTask(
                name="bad_output",
                task_ind=1,
                inp={"scalar": {"chain_name": "task_1_input", "value": 1}},
                scalar_max_mag=100,
                list_len_max=10,
                image=image,
                prompt="Task 1: answer task_1_out.",
                out=[1, "two", 3],
            )

    def test_bool_output_is_rejected(self):
        image = self.make_image()

        with self.assertRaises(TypeError):
            ImageTask(
                name="bad_output",
                task_ind=1,
                inp={"scalar": {"chain_name": "task_1_input", "value": 1}},
                scalar_max_mag=100,
                list_len_max=10,
                image=image,
                prompt="Task 1: answer task_1_out.",
                out=True,
            )


if __name__ == "__main__":
    unittest.main()
