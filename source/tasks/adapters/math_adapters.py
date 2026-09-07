"""Math adapters for ArbiGraph."""

import math

import sympy

from .adapter import Adapter


# =============================================================================
# In-domain adapters
# =============================================================================

class ScalarToScalarAdapter(Adapter):
    def __init__(self, mod_value=0, to_int=False, needs_abs=False, add_val=None, real_part=False):
        self.mod_value = mod_value
        self.to_int = to_int
        self.needs_abs = needs_abs
        self.add_val = add_val
        self.real_part = real_part

    def prompt(self, var_name_raw, var_name_adapted) -> str:
        ops = _adapt_element_prompt(
            self.mod_value,
            self.to_int,
            self.needs_abs,
            self.add_val,
            self.real_part,
        )
        prompt = ""
        prompt += f"Let {var_name_adapted} be the result of preprocessing {var_name_raw} "
        prompt += f"as follows - {ops}.\n"
        return prompt

    def compute(self, value):
        if self.real_part:
            value = sympy.re(value)
        if self.to_int:
            value = _round_away_from_zero(value)
        if self.mod_value != 0:
            if value < 0:
                value = -((-value) % self.mod_value)
            else:
                value = value % self.mod_value
        if self.needs_abs:
            value = abs(value)
        if self.add_val is not None:
            value += self.add_val
        return value


class ListToListAdapter(Adapter):
    def __init__(
        self,
        mod_value=0,
        list_len_max=0,
        num_parents=1,
        add_val=None,
        power_2=False,
        to_int=False,
        needs_abs=False,
        real_part=False,
    ):
        self.mod_value = mod_value
        self.list_len_max = list_len_max
        self.num_parents = num_parents
        self.add_val = add_val
        self.power_2 = power_2
        self.to_int = to_int
        self.needs_abs = needs_abs
        self.real_part = real_part
        self.element_adapter = ScalarToScalarAdapter(
            mod_value=mod_value,
            to_int=to_int,
            needs_abs=needs_abs,
            add_val=add_val,
            real_part=real_part,
        )
        self.full_list_dependency_adapter = FullListDependencyAdapter()

    def prompt(self, var_name_raw, var_name_adapted) -> str:
        ops = _adapt_element_prompt(
            self.mod_value,
            self.to_int,
            self.needs_abs,
            self.add_val,
            self.real_part,
        )
        prompt = ""
        if self.num_parents == 1:
            full_dependency_name = var_name_raw.replace('["result"]', "_fd")
            prompt = self.full_list_dependency_adapter.prompt(var_name_raw, full_dependency_name)
            var_name_raw = full_dependency_name
        prompt += f"Let {var_name_adapted} be the result of preprocessing {var_name_raw} as follows - "
        prompt += f"for every element, {ops}.\n"
        prompt += f"If {var_name_adapted} has more than {self.list_len_max} elements, "
        prompt += f"keep only the first {self.list_len_max} elements.\n"
        if self.power_2:
            prompt += f"If the length of {var_name_adapted} is not a power of 2, "
            prompt += f"update {var_name_adapted} by padding it with zeros to the next power of 2.\n"
        return prompt

    def compute(self, values):
        if self.num_parents == 1:
            values = self.full_list_dependency_adapter.compute(values)
        output = [self.element_adapter.compute(value) for value in values]
        if self.list_len_max > 0 and len(output) > self.list_len_max:
            output = output[:self.list_len_max]
        if self.power_2:
            output = _pad_to_power_of_2(output)
        return output


class FullListDependencyAdapter(Adapter):
    """Prepend the length and sum of a complete upstream list."""

    def prompt(self, input_list_name: str, out_var_name: str) -> str:
        prompt = ""
        prompt += f"Suppose {input_list_name} = [x_1, ..., x_n]. "
        prompt += f"Let {out_var_name} = [n, s, x_1, ..., x_n], where s = x_1 + ... + x_n.\n"
        return prompt

    def grouped_prompt(self, input_list_names: list[str], out_list_names: list[str]) -> str:
        if not input_list_names:
            raise ValueError("full_list_dependency requires at least one input list name.")
        joined_input_names = (
            " and ".join(input_list_names)
            if len(input_list_names) < 3
            else f"{', '.join(input_list_names[:-1])}, and {input_list_names[-1]}"
        )
        joined_out_names = (
            " and ".join(out_list_names)
            if len(out_list_names) < 3
            else f"{', '.join(out_list_names[:-1])}, and {out_list_names[-1]}"
        )
        prompt = f"Pair the input lists {joined_input_names} with the output variables {joined_out_names}, respectively.\n"
        prompt += "For each pair independently, suppose the input list is [x_1, ..., x_n] and let the paired "
        prompt += "output variable be [n, s, x_1, ..., x_n], where s = x_1 + ... + x_n for that input list.\n"
        return prompt

    def compute(self, values):
        if not isinstance(values, list):
            raise ValueError("full_list_dependency requires a list input.")

        _length = len(values)
        _sum = sum(values)
        ret = [_length, _sum] + list(values)
        return ret


class ListToMatrixAdapter(Adapter):
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def prompt(self, inp_list_name, matrix_name: str) -> str:
        prompt = ""
        prompt += f"Reshape list {inp_list_name} into matrix {matrix_name} of size "
        prompt += f"{self.num_rows} x {self.num_cols} using row-major order, "
        prompt += "filling the top row first, then the second row, and so on.\n"
        prompt += "To achieve the required dimension for the reshape, "
        prompt += f"you can append zeros to the end of list {inp_list_name} or truncate it.\n"
        return prompt

    def compute(self, values):
        total = self.num_rows * self.num_cols
        if len(values) < total:
            values = list(values) + [0] * (total - len(values))
        elif len(values) > total:
            values = list(values)[:total]

        matrix = []
        for row_index in range(self.num_rows):
            matrix.append(values[row_index * self.num_cols : (row_index + 1) * self.num_cols])
        return sympy.Matrix(matrix)


class MatrixToListAdapter(Adapter):
    def prompt(self, matrix_name, out_var_name: str) -> str:
        prompt = ""
        prompt += f"Convert matrix {matrix_name} into a list {out_var_name} "
        prompt += "by concatenating matrix rows starting from the top row, then second row, etc...\n"
        return prompt

    def compute(self, matrix):
        if isinstance(matrix, sympy.MatrixBase):
            return list(matrix)
        return [value for row in matrix for value in row]


class ListToPolynomialAdapter(Adapter):
    def __init__(self, symbol=None):
        self.symbol = sympy.Symbol("x") if symbol is None else sympy.sympify(symbol)

    def prompt(self, inp_list_name, poly_name: str) -> str:
        prompt = ""
        prompt += f"Let {poly_name}({self.symbol}) be a polynomial whose coefficients are the elements "
        prompt += f"of the list {inp_list_name} in order, starting from the highest degree.\n"
        return prompt

    def compute(self, coefficients):
        coefficients = list(coefficients)
        if not coefficients:
            coefficients = [0]
        return sympy.Poly(coefficients, self.symbol)


class PolynomialToListAdapter(Adapter):
    def __init__(self, symbol=None):
        self.symbol = sympy.Symbol("x") if symbol is None else sympy.sympify(symbol)

    def prompt(self, poly_name, out_var_name: str) -> str:
        prompt = ""
        prompt += f"Convert {poly_name} into a list {out_var_name} by extracting its coefficients "
        prompt += "starting from the highest degree to the constant term, "
        prompt += "including zero coefficients for missing intermediate powers. "
        prompt += "Omit leading zero coefficients. Represent the zero polynomial as [0].\n"
        return prompt

    def compute(self, polynomial):
        if not isinstance(polynomial, sympy.Poly):
            polynomial = sympy.Poly(polynomial, self.symbol)
        coefficients = list(polynomial.all_coeffs())
        return coefficients if coefficients else [0]


class InterleaveListsAdapter(Adapter):
    def __init__(self):
        self.full_list_dependency_adapter = FullListDependencyAdapter()

    def prompt(self, input_list_names: list[str], out_list_name: str) -> str:
        out_list_names = [input_list_name.replace('["result"]', "_fd") for input_list_name in input_list_names]
        joined_names = ", ".join(out_list_names)
        prompt = self.full_list_dependency_adapter.grouped_prompt(input_list_names, out_list_names)
        prompt += f"Let {out_list_name} be the result of interleaving the following lists "
        prompt += f"in the order listed: {joined_names}.\n"
        prompt += "Take the first element from each list, then the second element from each list, "
        prompt += "and so on. If one list is shorter, skip it after it runs out of elements.\n"
        prompt += "For example, interleaving [1, 2, 3], [10, 20], and [100, 200, 300, 400] gives "
        prompt += "[1, 10, 100, 2, 20, 200, 3, 300, 400].\n"
        return prompt

    def compute(self, input_lists):
        for input_list in input_lists:
            if not isinstance(input_list, list):
                raise ValueError("join_adaptar_interleave requires list inputs.")

        input_lists = [
            self.full_list_dependency_adapter.compute(input_list)
            for input_list in input_lists
        ]
        output = []
        max_len = max((len(input_list) for input_list in input_lists), default=0)
        for index in range(max_len):
            for input_list in input_lists:
                if index < len(input_list):
                    output.append(input_list[index])
        return output


class SumListsAdapter(Adapter):
    def __init__(self):
        self.full_list_dependency_adapter = FullListDependencyAdapter()

    def prompt(self, input_list_names: list[str], out_list_name: str) -> str:
        out_list_names = [input_list_name.replace('["result"]', "_fd") for input_list_name in input_list_names]
        joined_names = ", ".join(out_list_names)
        prompt = self.full_list_dependency_adapter.grouped_prompt(input_list_names, out_list_names)
        prompt += f"Let {out_list_name} be the result of adding the following lists elementwise: {joined_names}. "
        prompt += "At each position, add the elements from every list that has an element at that position.\n"
        prompt += "If a list is shorter than the others, omit it from the sums after it runs out of elements.\n"
        return prompt

    def compute(self, input_lists):
        for input_list in input_lists:
            if not isinstance(input_list, list):
                raise ValueError("sum_lists requires list inputs.")

        input_lists = [
            self.full_list_dependency_adapter.compute(input_list)
            for input_list in input_lists
        ]
        output = []
        max_len = max((len(input_list) for input_list in input_lists), default=0)
        for index in range(max_len):
            total = 0
            for input_list in input_lists:
                if index < len(input_list):
                    total += input_list[index]
            output.append(total)
        return output


# =============================================================================
# Cross-domain adapters
# =============================================================================

class AddScalarsAdapter(Adapter):
    def prompt(self, input_names: list[str], out_name: str) -> str:
        joined_names = ", ".join(input_names)
        prompt = ""
        prompt += f"Let {out_name} be the sum of the following values: {joined_names}.\n"
        return prompt

    def compute(self, input_values):
        total = 0
        for input_value in input_values:
            if isinstance(input_value, list):
                raise ValueError("join_adaptar_add requires scalar inputs.")
            total += input_value
        return total


# =============================================================================
# Shared numeric support primitives
# =============================================================================
def _round_away_from_zero(x):
    if x >= 0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


def _pad_to_power_of_2(values):
    if len(values) == 0:
        raise ValueError("Cannot pad empty sequence to power of 2.")
    length = len(values)
    if (length & (length - 1)) == 0:
        return values
    next_power = 2 ** math.ceil(math.log2(length))
    return list(values) + [0] * (next_power - length)


def _adapt_element_prompt(mod_value=0, to_int=False, needs_abs=False, add_val=None, real_part=False) -> str:
    prompt = ""
    if real_part:
        prompt += "take its real part; then "
    if to_int:
        prompt += "if it is a float, convert it to an integer by rounding ties away from zero "
        prompt += "(0.5 to 1, -0.5 to -1); then "
    if mod_value != 0:
        if needs_abs:
            prompt += f"replace it with the remainder of its absolute value modulo {mod_value}; then "
        else:
            prompt += f"replace it with the remainder of its absolute value modulo {mod_value} "
            prompt += "while preserving its original sign; then "
    if needs_abs and mod_value == 0:
        prompt += "take its absolute value; then "
    if add_val is not None and str(add_val) != "0":
        prompt += f"add {add_val} to it; then "

    return prompt.removesuffix("; then ") or "leave it unchanged"
