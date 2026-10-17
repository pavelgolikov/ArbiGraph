"""Class-based implementations of all supported Reasoning Gym tasks."""

from __future__ import annotations

import json
import math
import random
import re
from copy import copy, deepcopy
from functools import reduce
from typing import Any

from tasks.adapters import (
    AddScalarsAdapter,
    ListToListAdapter,
    MatrixToListAdapter,
    ScalarToScalarAdapter,
    SumListsAdapter,
)


_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
_DICE_MIN_DICE = 4
_DICE_MAX_DICE = 10
_GAME_OF_LIFE_MIN_STEPS = 1
_GAME_OF_LIFE_MAX_STEPS = 10
_GAME_OF_LIFE_HALTING_MIN_STEPS = 20
_GAME_OF_LIFE_HALTING_MAX_STEPS = 200
_BITWISE_MIN_DIFFICULTY = 1
_BITWISE_MAX_DIFFICULTY = 10
_BF_MIN_DIFFICULTY = 1
_BF_MAX_DIFFICULTY = 3
_AB_MIN_LENGTH = 10
_AB_MAX_LENGTH = 100


# =============================================================================
# Base Class
# =============================================================================


class RGTask:
    def __init__(
        self,
        task_ind: int,
        input_names: list[str],
        input_value: Any,
        dataset: Any,
        entry: dict[str, Any],
        num_parents: int = 1,
    ):
        self.name = self.task_name
        self.task_ind = task_ind
        self.result_var_name = f"{self.name}_result_{task_ind:d}"
        self.input_names = input_names
        self.input_value = input_value
        self.dataset = dataset
        self.entry = entry
        self.num_parents = num_parents
        self.adapted_inp: dict[str, Any] = {}

        self.gen_params()

        self.out = self.solution_generator()
        if self.out is None:
            raise ValueError(f"{self.name} returned None.")

        prompt = self.prompt_generator().rstrip()
        result_placeholder = (
            f'"{self.result_var_name}"' if self.output_type == "text" else self.result_var_name
        )
        replacement = (
            f"replacing {self.result_var_name} with the JSON Boolean true or false"
            if self.output_type == "bool"
            else f"replacing {self.result_var_name} with its calculated value"
        )
        self.prompt = (
            f"{prompt}\n"
            f"Output the result as task_{self.task_ind:d}_out = "
            f'{{"result": {result_placeholder}}}, '
            f"{replacement}."
        )

    def gen_params(self):
        pass

    def solution_generator(self):
        raise NotImplementedError

    def prompt_generator(self):
        raise NotImplementedError

    def _regenerate_rg_entry(self, **config_updates: Any) -> dict[str, Any]:
        rg_config = deepcopy(self.dataset.config)
        for field, value in config_updates.items():
            setattr(rg_config, field, value)
        rg_config.validate()
        rg_dataset = type(self.dataset)(rg_config)
        return rg_dataset[self.entry["metadata"]["source_index"]]


class GCDTask(RGTask):
    task_name = "gcd"
    input_type = "list"
    output_type = "scalar"

    def gen_params(self):
        self.input_adapter = ListToListAdapter(
            mod_value=self.dataset.config.max_value + 1,
            list_len_max=len(self.entry["metadata"]["numbers"]),
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        self.adapted_inp["numbers"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        return reduce(math.gcd, self.adapted_inp["numbers"])

    def prompt_generator(self):
        numbers_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], numbers_name)
        prompt += (
            f"Find the greatest common divisor (GCD) of all the numbers in {numbers_name}. "
            f"Let {self.result_var_name} be that GCD.\n"
        )
        return prompt


class LCMTask(RGTask):
    task_name = "lcm"
    input_type = "list"
    output_type = "scalar"

    def gen_params(self):
        self.input_adapter = ListToListAdapter(
            mod_value=self.dataset.config.max_value + 1,
            list_len_max=len(self.entry["metadata"]["numbers"]),
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        self.adapted_inp["numbers"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        return reduce(math.lcm, self.adapted_inp["numbers"])

    def prompt_generator(self):
        numbers_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], numbers_name)
        prompt += (
            f"Find the least common multiple (LCM) of all the numbers in {numbers_name}. "
            f"Let {self.result_var_name} be that LCM.\n"
        )
        return prompt


class CountPrimesTask(RGTask):
    task_name = "count_primes"
    input_type = "list"
    output_type = "scalar"

    def gen_params(self):
        self.input_adapter = ListToListAdapter(
            mod_value=self.dataset.config.max_n + 1,
            list_len_max=2,
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        self.adapted_inp["bounds"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        from sympy import primepi

        start, end = self.adapted_inp["bounds"]
        if start > end or end < 2:
            return 0
        return int(primepi(end) - primepi(max(start, 2) - 1))

    def prompt_generator(self):
        bounds_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], bounds_name)
        prompt += (
            f"Count the prime numbers between {bounds_name}[0] and {bounds_name}[1], inclusive. "
            f"If {bounds_name}[0] is greater than {bounds_name}[1], the count is zero. "
            f"Let {self.result_var_name} be that count.\n"
        )
        return prompt


class CountBitsTask(RGTask):
    task_name = "count_bits"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=self.dataset.config.max_n + 1,
            to_int=True,
            needs_abs=True,
        )
        self.adapted_inp["number"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        return self.adapted_inp["number"].bit_count()

    def prompt_generator(self):
        number_name = f"value_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], number_name)
        prompt += (
            f"Count the 1 bits in the ordinary unsigned binary representation of {number_name}. "
            f"Let {self.result_var_name} be that count.\n"
        )
        return prompt


class PrimeFactorizationTask(RGTask):
    task_name = "prime_factorization"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=self.dataset.config.max_value + 1,
            to_int=True,
            needs_abs=True,
        )
        self.adapted_inp["number"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        from sympy import factorint

        return factorint(self.adapted_inp["number"], multiple=True)

    def prompt_generator(self):
        number_name = f"value_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], number_name)
        prompt += (
            f"Find the prime factorization of {number_name}. Let {self.result_var_name} be the list of "
            "prime factors in nondecreasing order, repeating factors according to their multiplicity.\n"
        )
        return prompt


class ProductsTask(RGTask):
    task_name = "products"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_digits - config.min_digits + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_digits,
        )
        self.adapted_inp["num_digits"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_digits=self.adapted_inp["num_digits"],
            max_digits=self.adapted_inp["num_digits"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        num_digits_name = f"products_num_digits_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_digits_name)
        prompt += (
            f"Use {num_digits_name} as Reasoning Gym's num_digits parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class AIWTask(RGTask):
    task_name = "aiw"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        task_types = self.dataset.config.task_types
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=len(task_types),
            to_int=True,
            needs_abs=True,
        )
        self.adapted_inp["task_type_index"] = self.input_adapter.compute(self.input_value)
        task_type_index = self.adapted_inp["task_type_index"]
        self.rg_entry = self._regenerate_rg_entry(
            task_type_weights=[
                1.0 if index == task_type_index else 0.0
                for index in range(len(task_types))
            ],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        task_type_name = f"aiw_task_type_index_{self.task_ind:d}"
        task_types = ", ".join(
            f"{index}={task_type.value}"
            for index, task_type in enumerate(self.dataset.config.task_types)
        )
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], task_type_name)
        prompt += (
            f"Use {task_type_name} as Reasoning Gym's task-type selector ({task_types}). Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class BaseConversionTask(RGTask):
    task_name = "base_conversion"
    input_type = "list"
    output_type = "text"

    def gen_params(self):
        metadata = self.entry["metadata"]
        self.input_adapter = ListToListAdapter(
            mod_value=metadata["source_base"],
            list_len_max=len(metadata["source_repr"]),
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        self.adapted_inp["digits"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        decimal_value = 0
        for digit in self.adapted_inp["digits"]:
            decimal_value = decimal_value * self.entry["metadata"]["source_base"] + digit
        return _int_to_base(decimal_value, self.entry["metadata"]["target_base"])

    def prompt_generator(self):
        metadata = self.entry["metadata"]
        digits_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], digits_name)
        prompt += (
            f"Treat {digits_name} as the digits of a base-{metadata['source_base']} numeral, from most "
            f"significant to least significant. Convert it to base {metadata['target_base']}. Use "
            "lowercase letters a-z for digits above 9, omit leading zeros, and represent zero as 0. "
            f"Let {self.result_var_name} be the resulting numeral as a string.\n"
        )
        return prompt


class BasicArithmeticTask(RGTask):
    task_name = "basic_arithmetic"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_terms - config.min_terms + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_terms,
        )
        self.adapted_inp["num_terms"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_terms=self.adapted_inp["num_terms"],
            max_terms=self.adapted_inp["num_terms"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        num_terms_name = f"basic_arithmetic_num_terms_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_terms_name)
        prompt += (
            f"Use {num_terms_name} as Reasoning Gym's num_terms parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class ChainSumTask(RGTask):
    task_name = "chain_sum"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_terms - config.min_terms + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_terms,
        )
        self.adapted_inp["num_terms"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_terms=self.adapted_inp["num_terms"],
            max_terms=self.adapted_inp["num_terms"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        num_terms_name = f"chain_sum_num_terms_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_terms_name)
        prompt += (
            f"Use {num_terms_name} as Reasoning Gym's num_terms parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class DiceTask(RGTask):
    task_name = "dice"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=_DICE_MAX_DICE - _DICE_MIN_DICE + 1,
            to_int=True,
            needs_abs=True,
            add_val=_DICE_MIN_DICE,
        )
        self.adapted_inp["num_dice"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(num_dice=self.adapted_inp["num_dice"])

    def solution_generator(self):
        puzzle = self.rg_entry["metadata"]["puzzle"]
        return [puzzle["num"], puzzle["den"]]

    def prompt_generator(self):
        num_dice_name = f"dice_num_dice_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_dice_name)
        prompt += (
            f"Use {num_dice_name} as Reasoning Gym's num_dice parameter. Reasoning Gym generated this "
            f"task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's reduced fraction converted to "
            "[numerator, denominator].\n"
        )
        return prompt


class FractionSimplificationTask(RGTask):
    task_name = "fraction_simplification"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_factor - config.min_factor + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_factor,
        )
        self.adapted_inp["factor"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_factor=self.adapted_inp["factor"],
            max_factor=self.adapted_inp["factor"],
        )

    def solution_generator(self):
        metadata = self.rg_entry["metadata"]
        return [metadata["simplified_numerator"], metadata["simplified_denominator"]]

    def prompt_generator(self):
        factor_name = f"fraction_factor_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], factor_name)
        prompt += (
            f"Use {factor_name} as Reasoning Gym's factor parameter. Reasoning Gym generated this "
            f"task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's simplified fraction converted to "
            "[numerator, denominator].\n"
        )
        return prompt


class NumberFilteringTask(RGTask):
    task_name = "number_filtering"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_numbers - config.min_numbers + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_numbers,
        )
        self.adapted_inp["num_numbers"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_numbers=self.adapted_inp["num_numbers"],
            max_numbers=self.adapted_inp["num_numbers"],
        )

    def solution_generator(self):
        return [float(number) for number in self.rg_entry["metadata"]["result"]]

    def prompt_generator(self):
        num_numbers_name = f"number_filtering_num_numbers_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_numbers_name)
        prompt += (
            f"Use {num_numbers_name} as Reasoning Gym's number-count parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's resulting list with each numeric string "
            "converted to a number.\n"
        )
        return prompt


class NumberSortingTask(RGTask):
    task_name = "number_sorting"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_numbers - config.min_numbers + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_numbers,
        )
        self.adapted_inp["num_numbers"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_numbers=self.adapted_inp["num_numbers"],
            max_numbers=self.adapted_inp["num_numbers"],
        )

    def solution_generator(self):
        return [float(number) for number in self.rg_entry["metadata"]["sorted_numbers"]]

    def prompt_generator(self):
        num_numbers_name = f"number_sorting_num_numbers_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_numbers_name)
        prompt += (
            f"Use {num_numbers_name} as Reasoning Gym's number-count parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's sorted list with each numeric string "
            "converted to a number.\n"
        )
        return prompt


class PathStarTask(RGTask):
    task_name = "path_star"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_path_length - config.min_path_length + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_path_length,
        )
        self.adapted_inp["path_length"] = self.input_adapter.compute(self.input_value)

        rg_dataset = copy(self.dataset)
        rg_dataset.config = deepcopy(self.dataset.config)
        rg_dataset.config.min_path_length = self.adapted_inp["path_length"]
        rg_dataset.config.max_path_length = self.adapted_inp["path_length"]
        self.rg_entry = rg_dataset[self.entry["metadata"]["source_index"]]

    def solution_generator(self):
        return [int(node) for node in self.rg_entry["answer"].split()]

    def prompt_generator(self):
        path_length_name = f"path_length_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], path_length_name)
        prompt += (
            f"Use {path_length_name} as Reasoning Gym's path_length parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be the requested space-separated sequence converted to a list "
            "of integer node labels.\n"
        )
        return prompt


class SelfReferenceTask(RGTask):
    task_name = "self_reference"
    input_type = "list"
    output_type = "scalar"

    def gen_params(self):
        self.input_adapter = ListToListAdapter(
            mod_value=8,
            list_len_max=4,
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        self.adapted_inp["parameters"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        from reasoning_gym.logic.self_reference import solve_puzzle_dynamic

        puzzle = {
            "n": 7,
            "parameters": dict(zip(("a", "b", "c", "d"), self.adapted_inp["parameters"])),
        }
        return len(solve_puzzle_dynamic(puzzle))

    def prompt_generator(self):
        parameters_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], parameters_name)
        prompt += (
            "Consider these seven statements:\n"
            f"1. At least {parameters_name}[0] of these 7 statements are true.\n"
            f"2. At most {parameters_name}[1] of these 7 statements are false.\n"
            f"3. Exactly {parameters_name}[2] of these 7 statements are true.\n"
            f"4. Exactly {parameters_name}[3] of these 7 statements are false.\n"
            "5. Either Statement 3 or Statement 4 is true, but not both.\n"
            "6. The number of true statements is a prime number.\n"
            "7. The number of false statements is a composite number.\n"
            f"Let {self.result_var_name} be the number of self-consistent truth assignments to all seven "
            "statements.\n"
        )
        return prompt


class SimpleEquationsTask(RGTask):
    task_name = "simple_equations"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_terms - config.min_terms + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_terms,
        )
        self.adapted_inp["num_terms"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_terms=self.adapted_inp["num_terms"],
            max_terms=self.adapted_inp["num_terms"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        num_terms_name = f"simple_equations_num_terms_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_terms_name)
        prompt += (
            f"Use {num_terms_name} as Reasoning Gym's num_terms parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class StringSplittingTask(RGTask):
    task_name = "string_splitting"
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.input_adapter = ListToListAdapter(
            mod_value=self.dataset.config.max_initial_machines + 1,
            list_len_max=3,
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        self.adapted_inp["counts"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        if len(self.adapted_inp["counts"]) != 3:
            raise ValueError("string_splitting requires three adapted counts.")
        return self.dataset._get_answer(*self.adapted_inp["counts"])[-1]

    def prompt_generator(self):
        from reasoning_gym.algorithmic.string_splitting import QUESTION_TEMPLATE

        counts_name = f"list_{self.task_ind:d}"
        question = QUESTION_TEMPLATE.format(
            A_machine=f"{counts_name}[0]",
            B_machine=f"{counts_name}[1]",
            C_machine=f"{counts_name}[2]",
            max_iterations=self.dataset.config.max_iterations,
        ).strip()
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], counts_name)
        prompt += (
            f"{question}\nLet {self.result_var_name} be the six counts as the list [A, B, C, X, Y, Z].\n"
        )
        return prompt


class StringSynthesisTask(RGTask):
    task_name = "string_synthesis"
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.input_adapter = ListToListAdapter(
            mod_value=self.dataset.config.max_initial_blocks + 1,
            list_len_max=3,
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        self.adapted_inp["counts"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        if len(self.adapted_inp["counts"]) != 3:
            raise ValueError("string_synthesis requires three adapted counts.")
        return self.dataset._get_answer(*self.adapted_inp["counts"])[-1]

    def prompt_generator(self):
        from reasoning_gym.algorithmic.string_synthesis import QUESTION_TEMPLATE

        counts_name = f"list_{self.task_ind:d}"
        question = QUESTION_TEMPLATE.format(
            A_square=f"{counts_name}[0]",
            B_square=f"{counts_name}[1]",
            C_square=f"{counts_name}[2]",
            max_iterations=self.dataset.config.max_iterations,
        ).strip()
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], counts_name)
        prompt += (
            f"{question}\nLet {self.result_var_name} be the nine counts as the list "
            "[[A], [B], [C], {A}, {B}, {C}, (A), (B), (C)].\n"
        )
        return prompt


class PowerFunctionTask(RGTask):
    task_name = "power_function"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        modulus = math.ceil(max(abs(config.min_base), abs(config.max_base))) + 1
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=modulus,
            to_int=False,
            needs_abs=False,
        )
        self.adapted_inp["base"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        return math.pow(self.adapted_inp["base"], self.entry["metadata"]["exponent"])

    def prompt_generator(self):
        base_name = f"value_{self.task_ind:d}"
        exponent = self.entry["metadata"]["exponent"]
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], base_name)
        prompt += (
            f"Compute {base_name}^{exponent}. Let {self.result_var_name} be the numeric value, correct "
            "to 3 significant figures. Scientific e notation is allowed.\n"
        )
        return prompt


class BinaryMatrixTask(RGTask):
    task_name = "binary_matrix"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_n - config.min_n + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_n,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["n"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_n=self.adapted_inp["n"],
            max_n=self.adapted_inp["n"],
        )

    def solution_generator(self):
        return self.output_adapter.compute(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        n_name = f"binary_matrix_n_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], n_name)
        prompt += (
            f"Use {n_name} as Reasoning Gym's n parameter. Reasoning Gym generated this task:\n"
            f"{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's answer matrix flattened in row-major order.\n"
        )
        return prompt


class LargestIslandTask(RGTask):
    task_name = "largest_island"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        min_n = max(config.min_rows, config.min_cols)
        max_n = min(config.max_rows, config.max_cols)
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=max_n - min_n + 1,
            to_int=True,
            needs_abs=True,
            add_val=min_n,
        )
        self.adapted_inp["n"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_rows=self.adapted_inp["n"],
            max_rows=self.adapted_inp["n"],
            min_cols=self.adapted_inp["n"],
            max_cols=self.adapted_inp["n"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        n_name = f"largest_island_n_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], n_name)
        prompt += (
            f"Use {n_name} as both Reasoning Gym's rows and cols parameters. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class GameOfLifeTask(RGTask):
    task_name = "game_of_life"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=_GAME_OF_LIFE_MAX_STEPS - _GAME_OF_LIFE_MIN_STEPS + 1,
            to_int=True,
            needs_abs=True,
            add_val=_GAME_OF_LIFE_MIN_STEPS,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["simulation_steps"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            simulation_steps=self.adapted_inp["simulation_steps"],
        )

    def solution_generator(self):
        return self.output_adapter.compute(json.loads(self.rg_entry["answer"]))

    def prompt_generator(self):
        simulation_steps_name = f"game_of_life_simulation_steps_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], simulation_steps_name)
        prompt += (
            f"Use {simulation_steps_name} as Reasoning Gym's simulation_steps parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's final board flattened in row-major order.\n"
        )
        return prompt


class GameOfLifeHaltingTask(RGTask):
    task_name = "game_of_life_halting"
    input_type = "scalar"
    output_type = "bool"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=_GAME_OF_LIFE_HALTING_MAX_STEPS - _GAME_OF_LIFE_HALTING_MIN_STEPS + 1,
            to_int=True,
            needs_abs=True,
            add_val=_GAME_OF_LIFE_HALTING_MIN_STEPS,
        )
        self.adapted_inp["max_simulation_steps"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            max_simulation_steps=self.adapted_inp["max_simulation_steps"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"] == "True"

    def prompt_generator(self):
        max_steps_name = f"game_of_life_halting_max_steps_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], max_steps_name)
        prompt += (
            f"Use {max_steps_name} as Reasoning Gym's max_simulation_steps parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's Boolean answer.\n"
        )
        return prompt


class KakurasuTask(RGTask):
    task_name = "kakurasu"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=4,
            to_int=True,
            needs_abs=True,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["selector"] = self.input_adapter.compute(self.input_value)

    def solution_generator(self):
        solution = deepcopy(self.entry["metadata"]["solution"])
        if self.adapted_inp["selector"] & 1:
            solution = [[1 - cell for cell in row] for row in solution]
        if self.adapted_inp["selector"] & 2:
            solution = _transpose_grid(solution)
        return self.output_adapter.compute(solution)

    def prompt_generator(self):
        metadata = self.entry["metadata"]
        selector_name = f"value_{self.task_ind:d}"
        rows = metadata["n_rows"]
        cols = metadata["n_cols"]
        row_total = cols * (cols + 1) // 2
        col_total = rows * (rows + 1) // 2
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], selector_name)
        prompt += (
            f"Start with this {rows} x {cols} Kakurasu puzzle:\n"
            f"Base row sums: {metadata['row_sums']}\n"
            f"Base column sums: {metadata['col_sums']}\n{_format_grid(metadata['puzzle'])}\n"
            f"If bit 0 of {selector_name} is set, complement the puzzle: replace every row target r with "
            f"{row_total}-r and every column target c with {col_total}-c. If bit 1 is set, transpose "
            "the puzzle after that operation, swapping its dimensions, row targets, and column targets.\n"
            "Fill the resulting puzzle with zeros and ones. In each row, a 1 in column j has weight j. "
            "In each column, a 1 in row i has weight i. Positions are 1-indexed and every weighted sum "
            f"must equal its transformed target. Let {self.result_var_name} be the unique completed grid "
            "flattened in row-major order.\n"
        )
        return prompt


class ModuloGridTask(RGTask):
    task_name = "modulo_grid"
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.input_adapter = ListToListAdapter(
            mod_value=0,
            list_len_max=5,
            to_int=True,
            needs_abs=True,
            num_parents=self.num_parents,
        )
        values = self.input_adapter.compute(self.input_value)
        self.adapted_inp["values"] = values + [0] * (5 - len(values))

    def solution_generator(self):
        from reasoning_gym.cognition.modulo_grid import generate_grid

        values = self.adapted_inp["values"]
        operations = ["sum", "diff", "prod", "pow"]
        operation = operations[values[0] % len(operations)]
        divisor = values[1] % self.dataset.config.max_divisor + 1
        target = values[2] % divisor
        grid = generate_grid(
            self.dataset.config.size_x,
            self.dataset.config.size_y,
            operation,
            (divisor, target),
        )
        return [1 if cell == "✅" else 0 for row in grid for cell in row]

    def prompt_generator(self):
        values_name = f"list_{self.task_ind:d}"
        size_x = self.dataset.config.size_x
        size_y = self.dataset.config.size_y
        max_divisor = self.dataset.config.max_divisor
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], values_name)
        prompt += f"If {values_name} has fewer than 5 elements, append zeros until it has 5 elements.\n"
        prompt += (
            "Let operations = [sum, diff, prod, pow].\n"
            f"Let operation_{self.task_ind:d} = operations[{values_name}[0] modulo 4].\n"
            f"Let divisor_{self.task_ind:d} = ({values_name}[1] modulo {max_divisor}) + 1.\n"
            f"Let target_{self.task_ind:d} = {values_name}[2] modulo divisor_{self.task_ind:d}.\n"
            f"Let hole_row_{self.task_ind:d} = {values_name}[3] modulo {size_y}.\n"
            f"Let hole_column_{self.task_ind:d} = {values_name}[4] modulo {size_x}.\n"
            f"Construct a {size_y} x {size_x} grid using zero-indexed row y and column x. For every cell, "
            f"calculate operation_{self.task_ind:d}(x, y), where sum means x+y, diff means abs(x-y), "
            "prod means x*y, and pow means x**y. Put 1 in the cell exactly when the result modulo "
            f"divisor_{self.task_ind:d} equals target_{self.task_ind:d}; otherwise put 0. Regard the cell "
            f"at [hole_row_{self.task_ind:d}][hole_column_{self.task_ind:d}] as the question-mark cell "
            f"that must be restored. Let {self.result_var_name} be the entire completed grid, including "
            "the restored cell, flattened in row-major order.\n"
        )
        return prompt


class RottenOrangesTask(RGTask):
    task_name = "rotten_oranges"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_n - config.min_n + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_n,
        )
        self.adapted_inp["n"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_n=self.adapted_inp["n"],
            max_n=self.adapted_inp["n"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        n_name = f"rotten_oranges_n_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], n_name)
        prompt += (
            f"Use {n_name} as Reasoning Gym's n parameter. Reasoning Gym generated this task:\n"
            f"{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class SpiralMatrixTask(RGTask):
    task_name = "spiral_matrix"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_n - config.min_n + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_n,
        )
        self.adapted_inp["n"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_n=self.adapted_inp["n"],
            max_n=self.adapted_inp["n"],
        )

    def solution_generator(self):
        return list(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        n_name = f"spiral_matrix_n_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], n_name)
        prompt += (
            f"Use {n_name} as Reasoning Gym's n parameter. Reasoning Gym generated this task:\n"
            f"{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's requested sequence converted to a list.\n"
        )
        return prompt


class RotateMatrixTask(RGTask):
    task_name = "rotate_matrix"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_rotations - config.min_rotations + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_rotations,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["num_rotations"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_rotations=self.adapted_inp["num_rotations"],
            max_rotations=self.adapted_inp["num_rotations"],
        )

    def solution_generator(self):
        return self.output_adapter.compute(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        num_rotations_name = f"rotate_matrix_num_rotations_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_rotations_name)
        prompt += (
            f"Use {num_rotations_name} as Reasoning Gym's num_rotations parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's answer matrix flattened in row-major order.\n"
        )
        return prompt


class ManipulateMatrixTask(RGTask):
    task_name = "manipulate_matrix"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_transforms - config.min_transforms + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_transforms,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["num_transforms"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_transforms=self.adapted_inp["num_transforms"],
            max_transforms=self.adapted_inp["num_transforms"],
        )

    def solution_generator(self):
        return self.output_adapter.compute(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        num_transforms_name = f"manipulate_matrix_num_transforms_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_transforms_name)
        prompt += (
            f"Use {num_transforms_name} as Reasoning Gym's num_transforms parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's answer matrix flattened in row-major order.\n"
        )
        return prompt


class PoolMatrixTask(RGTask):
    task_name = "pool_matrix"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_pool_size - config.min_pool_size + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_pool_size,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["pool_size"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_pool_size=self.adapted_inp["pool_size"],
            max_pool_size=self.adapted_inp["pool_size"],
        )

    def solution_generator(self):
        return self.output_adapter.compute(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        pool_size_name = f"pool_matrix_pool_size_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], pool_size_name)
        prompt += (
            f"Use {pool_size_name} as Reasoning Gym's pool_size parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's answer matrix flattened in row-major order.\n"
        )
        return prompt


class SudokuTask(RGTask):
    task_name = "sudoku"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_empty - config.min_empty + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_empty,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["num_empty"] = self.input_adapter.compute(self.input_value)

        rg_dataset = copy(self.dataset)
        rg_dataset.config = deepcopy(self.dataset.config)
        rg_dataset.config.min_empty = self.adapted_inp["num_empty"]
        rg_dataset.config.max_empty = self.adapted_inp["num_empty"]
        self.rg_entry = rg_dataset[self.entry["metadata"]["source_index"]]

    def solution_generator(self):
        return self.output_adapter.compute(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        num_empty_name = f"sudoku_num_empty_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_empty_name)
        prompt += (
            f"Use {num_empty_name} as Reasoning Gym's num_empty parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's completed grid flattened in row-major order.\n"
        )
        return prompt


class MiniSudokuTask(SudokuTask):
    task_name = "mini_sudoku"


class FutoshikiTask(RGTask):
    task_name = "futoshiki"
    input_type = "scalar"
    output_type = "list"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_board_size - config.min_board_size + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_board_size,
        )
        self.output_adapter = MatrixToListAdapter()
        self.adapted_inp["board_size"] = self.input_adapter.compute(self.input_value)

        rg_dataset = copy(self.dataset)
        rg_dataset.config = deepcopy(self.dataset.config)
        rg_dataset.config.min_board_size = self.adapted_inp["board_size"]
        rg_dataset.config.max_board_size = self.adapted_inp["board_size"]
        self.rg_entry = rg_dataset[self.entry["metadata"]["source_index"]]

    def solution_generator(self):
        return self.output_adapter.compute(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        board_size_name = f"futoshiki_board_size_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], board_size_name)
        prompt += (
            f"Use {board_size_name} as Reasoning Gym's board_size parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's completed grid flattened in row-major order.\n"
        )
        return prompt


class CourseScheduleTask(RGTask):
    task_name = "course_schedule"
    input_type = "scalar"
    output_type = "bool"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_num_courses - config.min_num_courses + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_num_courses,
        )
        self.adapted_inp["num_courses"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_num_courses=self.adapted_inp["num_courses"],
            max_num_courses=self.adapted_inp["num_courses"],
        )

    def solution_generator(self):
        return bool(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        num_courses_name = f"course_schedule_num_courses_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_courses_name)
        prompt += (
            f"Use {num_courses_name} as Reasoning Gym's num_courses parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's Boolean answer.\n"
        )
        return prompt


class BinaryAlternationTask(RGTask):
    task_name = "binary_alternation"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_n - config.min_n + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_n,
        )
        self.adapted_inp["n"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_n=self.adapted_inp["n"],
            max_n=self.adapted_inp["n"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        n_name = f"binary_alternation_n_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], n_name)
        prompt += (
            f"Use {n_name} as Reasoning Gym's n parameter. Reasoning Gym generated this task:\n"
            f"{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class StringInsertionTask(RGTask):
    task_name = "string_insertion"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_string_length - config.min_string_length + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_string_length,
        )
        self.adapted_inp["string_length"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_string_length=self.adapted_inp["string_length"],
            max_string_length=self.adapted_inp["string_length"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        string_length_name = f"string_insertion_length_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], string_length_name)
        prompt += (
            f"Use {string_length_name} as Reasoning Gym's string_length parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's text answer.\n"
        )
        return prompt


class StringManipulationTask(RGTask):
    task_name = "string_manipulation"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_num_rules - config.min_num_rules + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_num_rules,
        )
        self.adapted_inp["num_rules"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_num_rules=self.adapted_inp["num_rules"],
            max_num_rules=self.adapted_inp["num_rules"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        num_rules_name = f"string_manipulation_num_rules_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_rules_name)
        prompt += (
            f"Use {num_rules_name} as Reasoning Gym's num_rules parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's text answer.\n"
        )
        return prompt


class IsomorphicStringsTask(RGTask):
    task_name = "isomorphic_strings"
    input_type = "scalar"
    output_type = "bool"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_string_length - config.min_string_length + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_string_length,
        )
        self.adapted_inp["string_length"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_string_length=self.adapted_inp["string_length"],
            max_string_length=self.adapted_inp["string_length"],
        )

    def solution_generator(self):
        return bool(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        string_length_name = f"isomorphic_strings_length_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], string_length_name)
        prompt += (
            f"Use {string_length_name} as Reasoning Gym's string_length parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's Boolean answer.\n"
        )
        return prompt


class RansomNoteTask(RGTask):
    task_name = "ransom_note"
    input_type = "scalar"
    output_type = "bool"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_note_length - config.min_note_length + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_note_length,
        )
        self.adapted_inp["note_length"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_note_length=self.adapted_inp["note_length"],
            max_note_length=self.adapted_inp["note_length"],
        )

    def solution_generator(self):
        return bool(self.rg_entry["metadata"]["solution"])

    def prompt_generator(self):
        note_length_name = f"ransom_note_length_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], note_length_name)
        prompt += (
            f"Use {note_length_name} as Reasoning Gym's note_length parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's Boolean answer.\n"
        )
        return prompt


class CaesarCipherTask(RGTask):
    task_name = "caesar_cipher"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_rotation - config.min_rotation + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_rotation,
        )
        self.adapted_inp["rotation"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_rotation=self.adapted_inp["rotation"],
            max_rotation=self.adapted_inp["rotation"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        rotation_name = f"caesar_cipher_rotation_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], rotation_name)
        prompt += (
            f"Use {rotation_name} as Reasoning Gym's rotation parameter. Reasoning Gym generated this "
            f"task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's text answer.\n"
        )
        return prompt


class BitwiseArithmeticTask(RGTask):
    task_name = "bitwise_arithmetic"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=_BITWISE_MAX_DIFFICULTY - _BITWISE_MIN_DIFFICULTY + 1,
            to_int=True,
            needs_abs=True,
            add_val=_BITWISE_MIN_DIFFICULTY,
        )
        self.adapted_inp["difficulty"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(difficulty=self.adapted_inp["difficulty"])

    def solution_generator(self):
        return int(self.rg_entry["answer"], 0)

    def prompt_generator(self):
        difficulty_name = f"bitwise_arithmetic_difficulty_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], difficulty_name)
        prompt += (
            f"Use {difficulty_name} as Reasoning Gym's difficulty parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's hexadecimal answer converted to its numeric "
            "integer value.\n"
        )
        return prompt


class BFTask(RGTask):
    task_name = "bf"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=_BF_MAX_DIFFICULTY - _BF_MIN_DIFFICULTY + 1,
            to_int=True,
            needs_abs=True,
            add_val=_BF_MIN_DIFFICULTY,
        )
        self.adapted_inp["difficulty"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(difficulty=self.adapted_inp["difficulty"])

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        difficulty_name = f"bf_difficulty_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], difficulty_name)
        prompt += (
            f"Use {difficulty_name} as Reasoning Gym's difficulty parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's exact text answer.\n"
        )
        return prompt


class EmojiMysteryTask(RGTask):
    task_name = "emoji_mystery"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_words_in_sentence - config.min_words_in_sentence + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_words_in_sentence,
        )
        self.adapted_inp["num_words_in_sentence"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_words_in_sentence=self.adapted_inp["num_words_in_sentence"],
            max_words_in_sentence=self.adapted_inp["num_words_in_sentence"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        num_words_name = f"emoji_mystery_num_words_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_words_name)
        prompt += (
            f"Use {num_words_name} as Reasoning Gym's num_words_in_sentence parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's exact text answer.\n"
        )
        return prompt


class LetterCountingTask(RGTask):
    task_name = "letter_counting"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_words - config.min_words + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_words,
        )
        self.adapted_inp["span_length"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_words=self.adapted_inp["span_length"],
            max_words=self.adapted_inp["span_length"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        span_length_name = f"letter_counting_span_length_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], span_length_name)
        prompt += (
            f"Use {span_length_name} as Reasoning Gym's words parameter. Reasoning Gym generated this "
            f"task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class MazeTask(RGTask):
    task_name = "maze"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_dist - config.min_dist + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_dist,
        )
        self.adapted_inp["shortest_path_length"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_dist=self.adapted_inp["shortest_path_length"],
            max_dist=self.adapted_inp["shortest_path_length"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        distance_name = f"maze_shortest_path_length_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], distance_name)
        prompt += (
            f"Use {distance_name} as Reasoning Gym's shortest-path distance parameter. Reasoning Gym "
            f"generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class LegCountingTask(RGTask):
    task_name = "leg_counting"
    input_type = "scalar"
    output_type = "scalar"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_animals - config.min_animals + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_animals,
        )
        self.adapted_inp["num_animals"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_animals=self.adapted_inp["num_animals"],
            max_animals=self.adapted_inp["num_animals"],
        )

    def solution_generator(self):
        return int(self.rg_entry["answer"])

    def prompt_generator(self):
        num_animals_name = f"leg_counting_num_animals_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_animals_name)
        prompt += (
            f"Use {num_animals_name} as Reasoning Gym's num_animals parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's numeric answer.\n"
        )
        return prompt


class ABTask(RGTask):
    task_name = "ab"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=_AB_MAX_LENGTH - _AB_MIN_LENGTH + 1,
            to_int=True,
            needs_abs=True,
            add_val=_AB_MIN_LENGTH,
        )
        self.adapted_inp["length"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(length=self.adapted_inp["length"])

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        length_name = f"ab_length_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], length_name)
        prompt += (
            f"Use {length_name} as Reasoning Gym's length parameter. Reasoning Gym generated this "
            f"task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's exact text answer.\n"
        )
        return prompt


class WordSortingTask(RGTask):
    task_name = "word_sorting"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_words - config.min_words + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_words,
        )
        self.adapted_inp["num_words"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_words=self.adapted_inp["num_words"],
            max_words=self.adapted_inp["num_words"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        num_words_name = f"word_sorting_num_words_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_words_name)
        prompt += (
            f"Use {num_words_name} as Reasoning Gym's num_words parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's exact text answer.\n"
        )
        return prompt


class WordSequenceReversalTask(RGTask):
    task_name = "word_sequence_reversal"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_words - config.min_words + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_words,
        )
        self.adapted_inp["num_words"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_words=self.adapted_inp["num_words"],
            max_words=self.adapted_inp["num_words"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        num_words_name = f"word_sequence_reversal_num_words_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_words_name)
        prompt += (
            f"Use {num_words_name} as Reasoning Gym's num_words parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's exact text answer.\n"
        )
        return prompt


class SyllogismTask(RGTask):
    task_name = "syllogism"
    input_type = "scalar"
    output_type = "bool"

    def gen_params(self):
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=2,
            to_int=True,
            needs_abs=True,
        )
        self.adapted_inp["task_type_index"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            inversion_probability=float(self.adapted_inp["task_type_index"]),
        )

    def solution_generator(self):
        return self.rg_entry["answer"] == "Yes"

    def prompt_generator(self):
        task_type_name = f"syllogism_task_type_index_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], task_type_name)
        prompt += (
            f"Use {task_type_name} as Reasoning Gym's task-type selector (0=syllogism, 1=inversion). "
            f"Reasoning Gym generated this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be True if Reasoning Gym's answer is Yes and False if it is No.\n"
        )
        return prompt


class MahjongPuzzleTask(RGTask):
    task_name = "mahjong_puzzle"
    input_type = "scalar"
    output_type = "text"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_num_rounds - config.min_num_rounds + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_num_rounds,
        )
        self.adapted_inp["num_rounds"] = self.input_adapter.compute(self.input_value)
        self.rg_entry = self._regenerate_rg_entry(
            min_num_rounds=self.adapted_inp["num_rounds"],
            max_num_rounds=self.adapted_inp["num_rounds"],
        )

    def solution_generator(self):
        return self.rg_entry["answer"]

    def prompt_generator(self):
        num_rounds_name = f"mahjong_num_rounds_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_rounds_name)
        prompt += (
            f"Use {num_rounds_name} as Reasoning Gym's num_rounds parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be Reasoning Gym's exact text answer.\n"
        )
        return prompt


class CircuitLogicTask(RGTask):
    task_name = "circuit_logic"
    input_type = "scalar"
    output_type = "bool"

    def gen_params(self):
        config = self.dataset.config
        self.input_adapter = ScalarToScalarAdapter(
            mod_value=config.max_terms - config.min_terms + 1,
            to_int=True,
            needs_abs=True,
            add_val=config.min_terms,
        )
        self.adapted_inp["num_terms"] = self.input_adapter.compute(self.input_value)

        rg_dataset = copy(self.dataset)
        rg_dataset.config = deepcopy(self.dataset.config)
        rg_dataset.config.min_terms = self.adapted_inp["num_terms"]
        rg_dataset.config.max_terms = self.adapted_inp["num_terms"]
        self.rg_entry = rg_dataset[0]

    def solution_generator(self):
        return bool(int(self.rg_entry["answer"]))

    def prompt_generator(self):
        num_terms_name = f"circuit_num_terms_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += self.input_adapter.prompt(self.input_names[0], num_terms_name)
        prompt += (
            f"Use {num_terms_name} as Reasoning Gym's num_terms parameter. Reasoning Gym generated "
            f"this task:\n{self.rg_entry['question'].strip()}\n"
            f"Let {self.result_var_name} be True if Reasoning Gym's requested final output is 1 and "
            "False if it is 0.\n"
        )
        return prompt


# =============================================================================
# Task Pool and Stage Construction
# =============================================================================


RG_TASKS = [
    GCDTask,
    LCMTask,
    CountPrimesTask,
    CountBitsTask,
    PrimeFactorizationTask,
    ProductsTask,
    AIWTask,
    BaseConversionTask,
    BasicArithmeticTask,
    ChainSumTask,
    DiceTask,
    FractionSimplificationTask,
    NumberFilteringTask,
    NumberSortingTask,
    PathStarTask,
    SelfReferenceTask,
    SimpleEquationsTask,
    StringSplittingTask,
    StringSynthesisTask,
    PowerFunctionTask,
    BinaryMatrixTask,
    LargestIslandTask,
    GameOfLifeTask,
    GameOfLifeHaltingTask,
    KakurasuTask,
    ModuloGridTask,
    RottenOrangesTask,
    SpiralMatrixTask,
    RotateMatrixTask,
    ManipulateMatrixTask,
    PoolMatrixTask,
    SudokuTask,
    MiniSudokuTask,
    FutoshikiTask,
    CourseScheduleTask,
    BinaryAlternationTask,
    StringInsertionTask,
    StringManipulationTask,
    IsomorphicStringsTask,
    RansomNoteTask,
    CaesarCipherTask,
    BitwiseArithmeticTask,
    BFTask,
    EmojiMysteryTask,
    LetterCountingTask,
    MazeTask,
    LegCountingTask,
    ABTask,
    WordSortingTask,
    WordSequenceReversalTask,
    SyllogismTask,
    MahjongPuzzleTask,
    CircuitLogicTask,
]


def load_pool() -> list[dict[str, Any]]:
    return [
        {
            "category": "rg",
            "task_id": task_id,
            "task_name": task_implementation.task_name,
            "input_type": task_implementation.input_type,
            "output_type": task_implementation.output_type,
            "task_implementation": task_implementation,
        }
        for task_id, task_implementation in enumerate(RG_TASKS)
    ]


def _sample_rg(task: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    try:
        from reasoning_gym import create_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'rg' task category requires Reasoning Gym to be installed in the active environment."
        ) from exc

    dataset = create_dataset(task["task_name"], seed=random.randrange(2**32), size=1)
    return dataset, dataset[0]


def _extract_root_input(
    task_name: str,
    entry: dict[str, Any],
    dataset: Any,
) -> int | float | list[int] | list[float]:
    metadata = entry["metadata"]
    if task_name in {"gcd", "lcm"}:
        return list(metadata["numbers"])
    if task_name == "count_primes":
        return [metadata["start"], metadata["end"]]
    if task_name in {"count_bits", "prime_factorization"}:
        return int(metadata["number"])
    if task_name == "products":
        return metadata["num_digits"] - dataset.config.min_digits
    if task_name == "aiw":
        return next(
            index
            for index, task_type in enumerate(dataset.config.task_types)
            if task_type.value == metadata["task_type"]
        )
    if task_name == "base_conversion":
        return _base_digits(metadata["source_repr"])
    if task_name in {"basic_arithmetic", "chain_sum"}:
        return metadata["num_terms"] - dataset.config.min_terms
    if task_name == "dice":
        return dataset.config.num_dice - _DICE_MIN_DICE
    if task_name == "fraction_simplification":
        return metadata["factor"] - dataset.config.min_factor
    if task_name in {"number_filtering", "number_sorting"}:
        return metadata["numbers"] - dataset.config.min_numbers
    if task_name == "path_star":
        return metadata["path_length"] - dataset.config.min_path_length
    if task_name == "self_reference":
        return _self_reference_parameters(entry)
    if task_name == "simple_equations":
        return 0
    if task_name == "string_splitting":
        return list(metadata["initial_machines"])
    if task_name == "string_synthesis":
        return list(metadata["initial_blocks"])
    if task_name == "power_function":
        return float(metadata["base"])
    if task_name == "binary_matrix":
        return metadata["n"] - dataset.config.min_n
    if task_name == "largest_island":
        min_n = max(dataset.config.min_rows, dataset.config.min_cols)
        return len(metadata["grid"]) - min_n
    if task_name in {"rotten_oranges", "spiral_matrix"}:
        return metadata["n"] - dataset.config.min_n
    if task_name == "rotate_matrix":
        return metadata["num_rotations"] - dataset.config.min_rotations
    if task_name == "manipulate_matrix":
        return metadata["num_transforms"] - dataset.config.min_transforms
    if task_name == "pool_matrix":
        return metadata["pool_size"] - dataset.config.min_pool_size
    if task_name == "game_of_life":
        return metadata["simulation_steps"] - _GAME_OF_LIFE_MIN_STEPS
    if task_name == "game_of_life_halting":
        return metadata["simulation_steps"] - _GAME_OF_LIFE_HALTING_MIN_STEPS
    if task_name == "kakurasu":
        return 0
    if task_name == "modulo_grid":
        operation_index = ["sum", "diff", "prod", "pow"].index(metadata["operation"])
        hole_row, hole_col = _modulo_grid_hole(entry)
        return [operation_index, metadata["divisor"] - 1, metadata["target"], hole_row, hole_col]
    if task_name in {"sudoku", "mini_sudoku"}:
        return metadata["num_empty"] - dataset.config.min_empty
    if task_name == "futoshiki":
        return metadata["board_size"] - dataset.config.min_board_size
    if task_name == "course_schedule":
        return len(metadata["courses"]) - dataset.config.min_num_courses
    if task_name == "binary_alternation":
        return metadata["n"] - dataset.config.min_n
    if task_name == "string_insertion":
        return metadata["string_length"] - dataset.config.min_string_length
    if task_name == "string_manipulation":
        return metadata["num_rules"] - dataset.config.min_num_rules
    if task_name == "isomorphic_strings":
        return metadata["string_length"] - dataset.config.min_string_length
    if task_name == "ransom_note":
        return metadata["note_length"] - dataset.config.min_note_length
    if task_name == "caesar_cipher":
        return metadata["rotation"] - dataset.config.min_rotation
    if task_name == "bitwise_arithmetic":
        return metadata["difficulty"]["difficulty"] - _BITWISE_MIN_DIFFICULTY
    if task_name == "bf":
        return metadata["difficulty"]["difficulty"] - _BF_MIN_DIFFICULTY
    if task_name == "emoji_mystery":
        return metadata["num_words_in_sentence"] - dataset.config.min_words_in_sentence
    if task_name == "letter_counting":
        return metadata["span_length"] - dataset.config.min_words
    if task_name == "maze":
        return metadata["shortest_path_length"] - dataset.config.min_dist
    if task_name == "leg_counting":
        return metadata["num_animals"] - dataset.config.min_animals
    if task_name == "ab":
        return metadata["difficulty"]["length"] - _AB_MIN_LENGTH
    if task_name == "word_sorting":
        return metadata["num_words"] - dataset.config.min_words
    if task_name == "word_sequence_reversal":
        return metadata["num_words"] - dataset.config.min_words
    if task_name == "syllogism":
        return int(metadata["type"] == "inversion")
    if task_name == "mahjong_puzzle":
        return len(metadata["rounds"]) - dataset.config.min_num_rounds
    if task_name == "circuit_logic":
        return len(metadata["term_strings"]) - dataset.config.min_terms
    raise ValueError(f"No root input extractor for {task_name!r}.")


def make_stage(
    task: dict[str, Any],
    task_number: int,
    input_names: list[str],
    _input: Any,
) -> tuple[Any, str, list[str], Any]:
    dataset, entry = _sample_rg(task)
    task_input = _extract_root_input(task["task_name"], entry, dataset) if _input is None else _input
    num_parents = 0 if _input is None else len(input_names)

    references = [f'{input_name}["result"]' for input_name in input_names]
    adapter_prompt = ""
    if len(input_names) > 1:
        if task["input_type"] == "scalar":
            adapter_name = f"value_{task_number:d}_join"
            join_adapter = AddScalarsAdapter()
        elif task["input_type"] == "list":
            adapter_name = f"list_{task_number:d}_join"
            join_adapter = SumListsAdapter()
        else:
            raise ValueError(f"Cannot join inputs for input_type={task['input_type']!r}.")
        task_input = join_adapter.compute(task_input)
        adapter_prompt = join_adapter.prompt(references, adapter_name)
        references = [adapter_name]

    task_instance = task["task_implementation"](
        task_number,
        references,
        task_input,
        dataset,
        entry,
        num_parents,
    )
    prompt = task_instance.prompt.rstrip()
    if adapter_prompt:
        task_header, task_body = prompt.split("\n", 1)
        prompt = f"{task_header}\n{adapter_prompt.rstrip()}\n{task_body}"

    if bad_output(task_instance.out, task["output_type"]):
        raise ValueError(f"rg:{task['task_id']}:{task['task_name']} produced an unusable output.")

    if _input is None:
        input_json = json.dumps({"result": task_input}, ensure_ascii=True)
        prompt = f"Define {input_names[0]} = {input_json}.\n{prompt}"

    return task_instance.out, prompt, [], task_input


# =============================================================================
# Helpers
# =============================================================================

def bad_output(value: Any, output_type: str) -> bool:
    if output_type == "bool":
        return not isinstance(value, bool)
    if output_type == "text":
        return not isinstance(value, str)
    if output_type == "list":
        return not isinstance(value, list) or not value or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            for item in value
        )
    if output_type == "scalar":
        return (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        )
    raise ValueError(f"Unknown RG output type {output_type!r}.")


def _base_digits(representation: str) -> list[int]:
    return [_DIGITS.index(character.lower()) for character in representation]


def _int_to_base(number: int, base: int) -> str:
    if number == 0:
        return "0"
    digits = []
    while number:
        number, remainder = divmod(number, base)
        digits.append(_DIGITS[remainder])
    return "".join(reversed(digits))


def _self_reference_parameters(entry: dict[str, Any]) -> list[int]:
    patterns = (
        r"At least (\d+) of these 7 statements",
        r"At most (\d+) of these 7 statements",
        r"Exactly (\d+) of these 7 statements are true",
        r"Exactly (\d+) of these 7 statements are false",
    )
    parameters = []
    for pattern in patterns:
        match = re.search(pattern, entry["question"])
        if match is None:
            raise ValueError("Could not recover the self-reference parameters from the RG prompt.")
        parameters.append(int(match.group(1)))
    return parameters


def _transpose_grid(grid: list[list[Any]]) -> list[list[Any]]:
    return [list(row) for row in zip(*grid)]


def _format_grid(grid: list[list[Any]]) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in grid)


def _modulo_grid_hole(entry: dict[str, Any]) -> tuple[int, int]:
    grid_text = entry["question"].rsplit("\n\n", 1)[1]
    for row_index, row in enumerate(grid_text.splitlines()):
        column_index = row.find("❔")
        if column_index >= 0:
            return row_index, column_index
    raise ValueError("Could not recover the modulo-grid hole from the RG prompt.")
