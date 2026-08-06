import collections
import json
import math
from fractions import Fraction
import sympy
import random
from typing import Any
from .math_helpers import *
from sympy.combinatorics.graycode import bin_to_gray
from sympy.combinatorics import Permutation


LIST_LEN_MAX = 10
SCALAR_MAX_MAG = 100


def bad_output(value: Any) -> bool:
    if isinstance(value, list):
        return (
            len(value) <= 1
            or collections.Counter(value).most_common(1)[0][1] > len(value) / 2
            or value == list(range(len(value)))
            or value == list(range(1, len(value) + 1))
        )
    return isinstance(value, (int, float)) and (
        not math.isfinite(value) or value in (0, 1, -1) or abs(value) >= 10**10
    )


def plain_value(value):
    if isinstance(value, list):
        return [plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [plain_value(item) for item in value]
    if isinstance(value, Fraction):
        return int(value) if value.denominator == 1 else float(value)
    if isinstance(value, sympy.MatrixBase):
        return [plain_value(item) for item in list(value)]
    if isinstance(value, sympy.Basic):
        if value.is_integer is True:
            return int(value)
        if value.is_real is True:
            result = float(value)
            return int(result) if result.is_integer() else result
    return value


# =============================================================================
# Base Class
# =============================================================================
class MathTask:
    def __init__(self,
        task_ind,
        input_names,
        input_value,
        scalar_max_mag,
        list_len_max):

        self.name = self.task_name
        self.task_ind = task_ind
        self.result_var_name = f"{self.name}_result_{task_ind:d}"
        self.input_names = input_names
        self.input_value = input_value
        self.adapted_inp = {}
        self.scalar_max_mag = scalar_max_mag
        self.list_len_max = list_len_max

        self.gen_params()

        self.out = plain_value(self.solution_generator())
        if self.out is None:
            raise ValueError(f"{self.name} returned None.")

        prompt = self._bind_result_variable(self.prompt_generator())
        self.prompt = (
            f"{prompt.rstrip()}\n"
            f"Output the result as task_{self.task_ind:d}_out = "
            f'{{"result": {self.result_var_name}}}, '
            f"replacing {self.result_var_name} with its calculated value."
        )

    def gen_params(self):
        pass

    def solution_generator(self) -> int | float | list[int] | list[float]:
        raise NotImplementedError

    def prompt_generator(self) -> str:
        raise NotImplementedError

    def _bind_result_variable(self, prompt: str) -> str:
        if self.result_var_name in prompt:
            return prompt.rstrip()
        result_kind = "list" if self.output_type == "list" else "value"
        return f"{prompt.rstrip()}\nLet {self.result_var_name} be the resulting {result_kind}."


class MatrixMulTask(MathTask):
    task_name = 'mat_mul'
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.input_value, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        length = len(self.adapted_inp['matrix_1'])
        num_rows = int(math.floor(math.sqrt(length)))
        num_cols = int(math.ceil(length / num_rows))
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.matrix_2 = _rand_matrix(num_cols, num_rows, -float(self.scalar_max_mag), float(self.scalar_max_mag))

    def solution_generator(self):
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.input_value, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        matrix_1 = list_to_matrix_compute(self.adapted_inp['matrix_1'], self.num_rows, self.num_cols)
        matrix_2 = self.matrix_2
        out = list(matrix_1 * matrix_2)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let L be max(1, the length of {adapted_inp_name}). Let r = max(1, floor(sqrt(L))). Let c = ceil(L / r).\n"
        prompt += list_to_matrix_prompt(adapted_inp_name, f"M1_{self.task_ind:d}", "r", "c")
        prompt += f"Define M2_{self.task_ind:d} = {self.matrix_2}.\n"
        prompt += f"Compute the matrix product of M1_{self.task_ind:d} and M2_{self.task_ind:d}. Let mat_{out_var_name} be this matrix product.\n"
        prompt += matrix_to_list_prompt(f'mat_{out_var_name}', out_var_name)
        return prompt


class FWHTTask(MathTask):
    task_name = 'fwht'
    input_type = "list"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, power_2=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max, to_int=True)
        padded = self.adapted_inp['seq']
        out = list(sympy.discrete.transforms.fwht(padded))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        seq_name = self.input_names[0]
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(seq_name, adapted_inp_name, power_2=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max, to_int=True)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}].\n"
        prompt += f"Compute the unnormalized Walsh-Hadamard transform in Sylvester order. That is, for each j = 0, 1, ..., N-1, define\n"
        prompt += "H_j = sum_{k=0}^{N-1} ((-1) ** popcount(j & k)) * a_k, where `&` is bitwise AND and popcount counts the number of 1-bits.\n"
        prompt += f"Return the sequence [H_0, H_1, ..., H_{{N-1}}].\n"
        return prompt


class NTTTask(MathTask):
    task_name = 'ntt'
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        N = len(self.adapted_inp['seq'])
        m = 1
        while True:
            p = m * N + 1
            if sympy.isprime(p):
                self.prime = int(p)
                break
            m += 1

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        as_ints = self.adapted_inp['seq']
        prime = int(self.prime)
        out = list(sympy.discrete.transforms.ntt(as_ints, prime))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        seq_name = self.input_names[0]
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(seq_name, adapted_inp_name, power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}]. Let p be the smallest prime number of the form m*N + 1 for a positive integer m.\n"
        prompt += f"All arithmetic in this transform is modulo p. Let g be the smallest positive primitive root modulo p, and define omega = g**((p - 1) // N) mod p. By construction, (N) divides (p-1), so (omega) has exact order (N) modulo (p).\n"
        prompt += f"Compute the number theoretic transform A = [A_0, A_1, ..., A_{{N-1}}], where for each j = 0, 1, ..., N-1, A_j = sum_{{k=0}}^{{N-1}} a_k * omega**(j*k) mod p.\n"
        prompt += f"Return each A_j as its least nonnegative residue in {{0, 1, ..., p-1}}.\n"
        return prompt


class SubsetZetaTransformTask(MathTask):
    task_name = 'subset_zeta_transform'
    input_type = "list"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        padded = self.adapted_inp['seq']
        out = list(sympy.discrete.transforms.mobius_transform(padded))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        seq_name = self.input_names[0]
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(seq_name, adapted_inp_name, power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} be a sequence of length 2^m. Index its entries by bitmasks S = 0, 1, ..., 2^m - 1.\n"
        prompt += f"Compute the subset-sum/zeta transform B, where for each bitmask S, B[S] = sum of {adapted_inp_name}[T] over all bitmasks T such that T is a subset of S.\n"
        prompt += f"Here, T is a subset of S means every 1-bit of T is also a 1-bit of S, equivalently (T & S) = T.\n"
        prompt += f"Return the values B[0], B[1], ..., B[2^m - 1] in increasing bitmask order.\n"
        return prompt


class KroneckerProductTask(MathTask):
    task_name = 'kronecker_product'
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        length = len(self.adapted_inp['matrix_1'])
        r = int(math.floor(math.sqrt(length)))
        c = int(math.ceil(length / r))
        self.num_rows = r
        self.num_cols = c
        self.matrix_2 = _rand_matrix(r, c, -float(self.scalar_max_mag), float(self.scalar_max_mag))

    def solution_generator(self):
        r = self.num_rows
        c = self.num_cols
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        inp_list = self.adapted_inp['matrix_1']
        m1 = list_to_matrix_compute(inp_list, r, c)
        m2 = self.matrix_2
        out = list(sympy.kronecker_product(m1, m2))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        m1_name = f"M1_{self.task_ind:d}"
        m2_name = f"M2_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let L be max(1, the length of {adapted_inp_name}). Let r = max(1, floor(sqrt(L))). Let c = ceil(L / r).\n"
        prompt += list_to_matrix_prompt(adapted_inp_name, m1_name, "r", "c")
        prompt += f"Define {m2_name} = {self.matrix_2}.\n"
        prompt += f"Compute the Kronecker product of {m1_name} and {m2_name}. Let mat_{out_var_name} be this matrix.\n"
        prompt += matrix_to_list_prompt(f"mat_{out_var_name}", out_var_name)
        return prompt


class InterpolateEvalTask(MathTask):
    task_name = 'interpolate_eval'
    input_type = "list"
    output_type = "scalar"

    def gen_params(self):
        self.eval_point = random.choice(range(-100, 100))

    def solution_generator(self):
        self.adapted_inp['points'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        points = self.adapted_inp['points']
        x = sympy.Symbol("x")
        points_dict = {i: val for i, val in enumerate(points)}
        poly = sympy.polys.polyfuncs.interpolate(points_dict, x)
        out = poly.subs(x, self.eval_point)
        return out


    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"points_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let N be the length of {adapted_inp_name}. Construct the unique interpolating polynomial through the points (i, {adapted_inp_name}[i]) for every integer i with 0 <= i < N.\n"
        prompt += f"Evaluate that polynomial at x = {self.eval_point}.\n"
        return prompt


class SummatoryTotientTask(MathTask):
    task_name = 'summatory_totient'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sum(int(sympy.totient(k)) for k in range(1, n + 1))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer k = 1, 2, ..., {adapted_inp_name}, compute Euler's totient function phi(k), the number of integers m with 1 <= m <= k such that gcd(m, k) = 1.\n"
        prompt += f"Compute the sum phi(1) + phi(2) + ... + phi({adapted_inp_name}).\n"
        return prompt


class DivisorSumTask(MathTask):
    task_name = 'divisor_sum'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        conditioned = self.adapted_inp['n']
        out = sympy.divisor_sigma(conditioned, 1)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the sum of all positive divisors of {adapted_inp_name}.\n"
        return prompt


class ConvolutionTask(MathTask):
    task_name = 'convolution'
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        self.seq2 = [random.randint(-5, 5) for _ in range(len(self.adapted_inp['seq']))]

    def solution_generator(self):
        seq = self.adapted_inp['seq']
        seq2 = self.seq2
        out = list(sympy.discrete.convolutions.convolution(seq, seq2))
        return out + [0] * (len(seq) + len(seq2) - 1 - len(out))

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Define seq2_{self.task_ind:d} = {self.seq2}.\n"
        prompt += f"Compute the linear discrete convolution of {adapted_inp_name} and seq2_{self.task_ind:d}.\n"
        return prompt


class PolyRemainderTask(MathTask):
    task_name = 'poly_remainder'
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        self.seq2 = [random.randint(-5, 5) for _ in range(len(self.adapted_inp['seq']))]

    def solution_generator(self):
        seq = self.adapted_inp['seq']
        x = sympy.Symbol('x')
        P = sympy.Poly(seq, x)
        Q = sympy.Poly([1] + self.seq2, x)
        out = sympy.rem(P, Q).all_coeffs()
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        x = sympy.Symbol('x')
        poly_2 = sympy.Poly([1] + self.seq2, x)
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f"poly_1_{self.task_ind:d}")
        prompt += f"Define poly_2_{self.task_ind:d} = {poly_2.as_expr()}.\n"
        prompt += f"Compute the polynomial remainder of poly_1_{self.task_ind:d} divided by poly_2_{self.task_ind:d}. Let poly_{out_var_name} be this polynomial.\n"
        prompt += poly_to_list_prompt(f'poly_{out_var_name}', out_var_name)
        return prompt


class PolyQuotientTask(MathTask):
    task_name = 'poly_quotient'
    input_type = "list"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        x = sympy.Symbol('x')
        P = sympy.Poly(seq, x)
        Q = sympy.Poly([1, 1, 1], x)
        q, r = sympy.div(P, Q)
        out = q.all_coeffs() if hasattr(q, 'all_coeffs') else []
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f"P_{self.task_ind:d}")
        prompt += f"Compute the polynomial quotient when P_{self.task_ind:d}(x) is divided by the monic polynomial x**2 + x + 1. Let poly_{out_var_name} be this polynomial.\n"
        prompt += poly_to_list_prompt(f'poly_{out_var_name}', out_var_name)
        return prompt


class MatrixPolynomialTask(MathTask):
    task_name = 'matrix_polynomial'
    input_type = "list"
    output_type = "list"

    def gen_params(self):
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        dim = int(math.ceil(math.sqrt(len(self.adapted_inp['matrix_1']))))
        self.num_rows = dim

    def solution_generator(self):
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['matrix_1']
        num_rows = self.num_rows
        m = list_to_matrix_compute(seq, num_rows, num_rows)
        out = m**3 + m**2 + m
        out = list(out)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let L be max(1, the length of {adapted_inp_name}). Let k = ceil(sqrt(L)).\n"
        prompt += list_to_matrix_prompt(adapted_inp_name, f"matrix_1_{self.task_ind:d}", "k", "k")
        prompt += f"Compute the matrix polynomial matrix_1_{self.task_ind:d}**3 + matrix_1_{self.task_ind:d}**2 + matrix_1_{self.task_ind:d}. Let mat_{out_var_name} be this matrix.\n"
        prompt += matrix_to_list_prompt(f"mat_{out_var_name}", out_var_name)
        return prompt


class SortByDivisorCountTask(MathTask):
    task_name = 'sort_by_divisor_count'
    input_type = "list"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        out = sorted(seq, key=lambda x: sympy.ntheory.factor_.divisor_count(abs(x)) if x != 0 else 0)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Sort {adapted_inp_name} based on the number of divisors of their absolute values in ascending order. If an element is 0, treat its divisor count as 0. Maintain stable sorting for ties.\n"
        return prompt


class MatrixFrobeniusSqTask(MathTask):
    task_name = 'matrix_frobenius_sq'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        k = math.isqrt(len(seq))
        mat = list_to_matrix_compute(seq, k, k)
        out = sum(mat[i, j]**2 for i in range(k) for j in range(k))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let L be the length of {adapted_inp_name}. Let k = floor(sqrt(L)).\n"
        prompt += list_to_matrix_prompt(adapted_inp_name, f'mat_1_{self.task_ind:d}', "k", "k")
        prompt += f"Compute the Frobenius norm squared (sum of squares of all elements) of mat_1_{self.task_ind:d}.\n"
        return prompt


class ConvexHullAreaTask(MathTask):
    task_name = 'convex_hull_area'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        pts = [sympy.geometry.Point(i, val) for i, val in enumerate(seq)]
        hull = sympy.geometry.convex_hull(*pts)
        out = 2 * hull.area if hasattr(hull, 'area') else 0
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Consider {adapted_inp_name} as a sequence of y-coordinates with x-coordinates being their indices (0, 1, 2, ...).\n"
        prompt += f"Compute the area of the convex hull of these 2D points and multiply the area by 2. If there are fewer than 3 points, or if all points are collinear, define the area to be 0.\n"
        return prompt


class MatrixNorm1Task(MathTask):
    task_name = 'matrix_norm1'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        k = math.isqrt(len(seq))
        mat = list_to_matrix_compute(seq, k, k)
        out = max(sum(abs(mat[r, c]) for r in range(k)) for c in range(k))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let L be the length of {adapted_inp_name}. Let k = floor(sqrt(L)).\n"
        prompt += list_to_matrix_prompt(adapted_inp_name, f'matrix_1_{self.task_ind:d}', "k", "k")
        prompt += f"Compute the 1-norm of matrix_1_{self.task_ind:d} (the maximum absolute column sum).\n"
        return prompt


class MatrixTraceCubeTask(MathTask):
    task_name = 'matrix_trace_cube'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        k = math.isqrt(len(seq))
        mat = list_to_matrix_compute(seq, k, k) 
        out = (mat**3).trace()
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let L be the length of {adapted_inp_name}. Let k = floor(sqrt(L)).\n"
        prompt += list_to_matrix_prompt(adapted_inp_name, f'mat_1_{self.task_ind:d}', "k", "k")
        prompt += f"Compute the trace of mat_1_{self.task_ind:d}**3.\n"
        return prompt


class PermutationInversionsTask(MathTask):
    task_name = 'permutation_inversions'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        indices = sorted(range(len(seq)), key=lambda x: seq[x])
        out = Permutation(indices).inversions()
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}].\nSort the indices 0, 1, ..., N-1 in ascending order by the pair (a_i, i).\n"
        prompt += f"In other words, sort by the corresponding sequence value, and break ties by the smaller original index.\nLet the resulting index list be pi = [pi_0, pi_1, ..., pi_{{N-1}}].\n"
        prompt += f"Compute the number of inversions in pi, meaning the number of pairs (r, s) such that 0 <= r < s < N and pi_r > pi_s.\n"
        return prompt


class PermutationOrderTask(MathTask):
    task_name = 'permutation_order'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        indices = sorted(range(len(seq)), key=lambda x: seq[x])
        out = Permutation(indices).order() if seq else 1
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}].\nSort the indices 0, 1, ..., N-1 in ascending order by the pair (a_i, i).\n"
        prompt += f"In other words, sort by the corresponding sequence value, and break ties by the smaller original index.\n"
        prompt += f"Let the resulting index list be pi = [pi_0, pi_1, ..., pi_{{N-1}}].\nTreat pi as the array form of a permutation of {{0, 1, ..., N-1}}, meaning pi maps r to pi_r.\n"
        prompt += f"Compute the order of this permutation: the smallest positive integer m such that applying pi exactly m times gives the identity permutation.\nIf N = 0, define the order to be 1.\n"
        return prompt


class PolySquareFreeEvalTask(MathTask):
    task_name = 'poly_square_free_eval'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        x = sympy.Symbol('x')
        P = sympy.Poly(seq, x, domain=sympy.QQ)
        if P.is_zero:
            return 0
        dP = P.diff()
        # Nonzero constant polynomial: prompt says gcd(P, P') = 1.
        if dP.is_zero:
            out = P.eval(2)
        else:
            G = sympy.gcd(P, dP).monic()
            S_expr = sympy.cancel(P.as_expr() / G.as_expr())
            out = sympy.simplify(S_expr.subs(x, 2))
        if getattr(out, "is_Integer", False):
            return int(out)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f'P_{self.task_ind:d}')
        prompt += f"If P_{self.task_ind:d}(x) is the zero polynomial, define {out_var_name} = 0. For a nonzero constant polynomial, use gcd(P_{self.task_ind:d}, P_{self.task_ind:d}') = 1.\n"
        prompt += f"Otherwise, let S_{self.task_ind:d}(x) = P_{self.task_ind:d}(x) / gcd(P_{self.task_ind:d}(x), P_{self.task_ind:d}'(x)), where the gcd is taken over Q[x] and made monic.\n"
        prompt += f"Do not further normalize S_{self.task_ind:d}(x): keep any scalar coefficient remaining after division by the monic gcd. "
        prompt += f"Evaluate S_{self.task_ind:d}(2), and let {out_var_name} be the resulting value.\n"
        return prompt


class PolyResultantFixedTask(MathTask):
    task_name = 'poly_resultant_fixed'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        x = sympy.Symbol('x')
        P = sympy.Poly(seq, x)
        Q = sympy.Poly([1, 1, 1], x)
        out = sympy.resultant(P, Q)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f'P_{self.task_ind:d}')
        prompt += f"Compute the resultant of P_{self.task_ind:d}(x) and the polynomial x**2 + x + 1.\n"
        return prompt


class PairwiseGcdSumTask(MathTask):
    task_name = 'pairwise_gcd_sum'
    input_type = "list"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.input_value, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        if len(seq) < 2:
            out = 0
        else:
            total = 0
            for i in range(len(seq)):
                for j in range(i+1, len(seq)):
                    total += sympy.gcd(seq[i], seq[j])
            out = int(total)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"For every pair of indices i, j with 0 <= i < j < len({adapted_inp_name}), compute gcd(|{adapted_inp_name}[i]|, |{adapted_inp_name}[j]|). Use the convention gcd(0, 0) = 0.\n"
        prompt += f"Compute the sum of all these GCD values.\n"
        return prompt


class PrimeOmegaSumTask(MathTask):
    task_name = 'prime_omega_sum'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, add_val=2, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sum(sympy.factorint(i).__len__() for i in range(2, n + 1))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=2, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 2 to {adapted_inp_name}, compute omega(i), the number of distinct prime factors of i.\n"
        prompt += f"Compute the sum of all these omega values.\n"
        return prompt


class SumDivisorCountsTask(MathTask):
    task_name = 'sum_divisor_counts'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, add_val=1, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sum(int(sympy.divisor_count(i)) for i in range(1, n + 1))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute the number of positive divisors of i.\n"
        prompt += f"Compute the sum of all these divisor counts.\n"
        return prompt



class PrimitiveRootCountTask(MathTask):
    task_name = 'primitive_root_count'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, add_val=1, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        p = sympy.prime(n)
        # The number of primitive roots modulo a prime p is phi(p - 1)
        out = sympy.totient(p - 1)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Find the {adapted_inp_name}-th prime number p.\n"
        prompt += f"Compute Euler's totient phi(p-1), which equals the count of primitive roots modulo p.\n"
        return prompt


class LegendreResidueWeightTask(MathTask):
    task_name = 'legendre_residue_weight'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, add_val=25, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        p = int(sympy.nextprime(n + 2))
        out = sum(a * (int(sympy.legendre_symbol(a, p)) + 1) for a in range(1, n + 1))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, add_val=25, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Let p be the smallest prime number strictly greater than {adapted_inp_name} + 2.\n"
        prompt += f"For each integer a = 1, 2, ..., {adapted_inp_name}, compute the Legendre symbol Legendre(a, p). This value is 1 if there exists an integer x such that x**2 is congruent to a modulo p, and -1 otherwise.\n"
        prompt += f"For each a, compute a * (Legendre(a, p) + 1). Then compute the sum of these values over all a = 1, 2, ..., {adapted_inp_name}.\n"
        return prompt


class ScaledStirlingFirstKindTask(MathTask):
    task_name = 'scaled_stirling_first_kind'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        m = math.isqrt(n) + 7
        out = 2 * sympy.binomial(m, 3) + 3 * sympy.binomial(m, 4)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Define m = floor(sqrt({adapted_inp_name})) + 7.\n"
        prompt += f"Compute the unsigned Stirling number of the first kind c(m, m - 2), which counts the number of permutations of m elements with exactly m - 2 disjoint cycles.\n"
        return prompt


class DivisorSigmaSquareTask(MathTask):
    task_name = 'divisor_sigma_square'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = int(sympy.divisor_sigma(n, 2))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the sum of the squares of all positive divisors of {adapted_inp_name}.\n"
        return prompt


class MultiplicativeOrderTask(MathTask):
    task_name = 'multiplicative_order'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sympy.ntheory.n_order(2, 2*n + 1)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the multiplicative order of 2 modulo (2*{adapted_inp_name} + 1).\n"
        return prompt


class UnitaryDivisorSumTask(MathTask):
    task_name = 'unitary_divisor_sum'
    input_type = "scalar"
    output_type = "scalar"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sympy.functions.combinatorial.numbers.udivisor_sigma(n, 1)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the sum of the unitary divisors of {adapted_inp_name} (divisors d where gcd(d, {adapted_inp_name}/d) == 1).\n"
        return prompt


class ModularMultiplicationSequenceTask(MathTask):
    task_name = 'modular_multiplication_sequence'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        p = sympy.prime(n)
        out = [int((i * n) % p) for i in range(1, n + 1)]
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        prompt = f"Task {self.task_ind:d}:\n"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute (i * {adapted_inp_name}) modulo the {adapted_inp_name}-th prime number.\n"
        prompt += f"Compute the sequence of these values.\n"
        return prompt


class WeightedDivisorCountPrefixSequenceTask(MathTask):
    task_name = 'weighted_divisor_count_prefix_sequence'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        total = 0
        out = []
        for i in range(1, n + 1):
            total += i * int(sympy.divisor_count(n + i))
            out.append(total)
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i = 1, 2, ..., {adapted_inp_name}, define A_i to be the sum over k = 1, 2, ..., i of k times the number of positive divisors of {adapted_inp_name} + k.\n"
        prompt += f"Return the list [A_1, A_2, ..., A_{{{adapted_inp_name}}}] in increasing order of i, so the first element corresponds to i = 1, the second element corresponds to i = 2, and so on.\n"
        return prompt


class GrayCodeSequenceTask(MathTask):
    task_name = 'gray_code_sequence'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(bin_to_gray(bin(i + n)[2:]), 2) for i in range(1, n + 1)]
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, let x = i + {adapted_inp_name}.\nCompute the standard reflected binary Gray code of x, defined as gray(x) = x XOR floor(x / 2), where XOR is bitwise exclusive OR.\n"
        prompt += f"Return these Gray code values as ordinary decimal integers, not as binary strings, in the order i = 1, 2, ..., {adapted_inp_name}.\n"
        return prompt


class TotientShiftSequenceTask(MathTask):
    task_name = 'totient_shift_sequence'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(sympy.totient(i + n)) for i in range(1, n + 1)]
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute Euler's totient function phi(i + {adapted_inp_name}).\n"
        prompt += f"Compute the sequence of these values.\n"
        return prompt


class SmallestFactorQuadraticTask(MathTask):
    task_name = 'smallest_factor_quadratic'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(min(sympy.factorint(i**2 + n).keys())) if (i**2 + n) > 1 else 1 for i in range(1, n + 1)]
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute the smallest prime factor of i**2 + {adapted_inp_name}.\n"
        prompt += f"Compute the sequence of these values.\n"
        return prompt


class TotientProductSequenceTask(MathTask):
    task_name = 'totient_product_sequence'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(sympy.totient(i * n)) for i in range(1, n + 1)]
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute Euler's totient function of i * {adapted_inp_name}.\n"
        prompt += f"Compute the sequence of these values.\n"
        return prompt


class PolygonDiagonalsSequenceTask(MathTask):
    task_name = 'polygon_diagonals_sequence'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=3, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int((i + n) * (i + n - 3) // 2) for i in range(1, n + 1)]
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=3, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute the number of diagonals in a convex polygon with i + {adapted_inp_name} vertices.\n"
        prompt += f"Compute the sequence of these values.\n"
        return prompt


class PrimesBetweenSquaresTask(MathTask):
    task_name = 'primes_between_squares'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = list(sympy.primerange(n**2 + 1, (n+1)**2))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"List all prime numbers strictly between {adapted_inp_name}**2 and ({adapted_inp_name}+1)**2 in increasing order.\n"
        return prompt


class CentralBinomialPrimeValuationVectorTask(MathTask):
    task_name = 'central_binomial_prime_valuation_vector'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = []
        for p in sympy.primerange(2, 2*n + 1):
            p = int(p)
            exponent = 0
            power = p
            while power <= 2*n:
                exponent += (2*n) // power - 2 * (n // power)
                power *= p
            out.append(int(p * (exponent + 1)))
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each prime number p with p <= 2*{adapted_inp_name}, define e_p as follows.\n"
        prompt += f"For every positive integer j such that p**j <= 2*{adapted_inp_name}, compute floor((2*{adapted_inp_name}) / p**j) - 2*floor({adapted_inp_name} / p**j), and let e_p be the sum of these values over all such j.\n"
        prompt += f"For each prime p with p <= 2*{adapted_inp_name}, compute p * (e_p + 1).\n"
        prompt += f"Return these values as a list in increasing order of p.\n"
        return prompt


class QuadraticResiduesPrimeTask(MathTask):
    task_name = 'quadratic_residues_prime'
    input_type = "scalar"
    output_type = "list"

    def solution_generator(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.input_value, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sympy.ntheory.quadratic_residues(sympy.prime(n)) if n > 0 else []
        return out

    def prompt_generator(self):
        out_var_name = self.result_var_name
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.input_names[0], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Let p be the {adapted_inp_name}-th prime number, using the convention that the 1st prime is 2.\nFind all distinct residues r in {{0, 1, ..., p-1}} for which there exists an integer x such that x^2 ≡ r mod p.\n"
        prompt += f"Include 0 as a quadratic residue. Return the residues in increasing order.\n"
        return prompt


# =============================================================================
# Internal Helpers
# =============================================================================

def _rand_matrix(num_rows: int, num_cols: int, low: float, high: float):
    return sympy.Matrix([
        [random.randint(int(low), int(high)) for _ in range(num_cols)]
        for _ in range(num_rows)
    ])


MATH_TASKS = [
    MatrixMulTask,
    FWHTTask,
    NTTTask,
    SubsetZetaTransformTask,
    KroneckerProductTask,
    ConvolutionTask,
    PolyRemainderTask,
    PolyQuotientTask,
    MatrixPolynomialTask,
    SortByDivisorCountTask,
    InterpolateEvalTask,
    MatrixFrobeniusSqTask,
    ConvexHullAreaTask,
    MatrixNorm1Task,
    MatrixTraceCubeTask,
    PermutationInversionsTask,
    PermutationOrderTask,
    PolySquareFreeEvalTask,
    PolyResultantFixedTask,
    PairwiseGcdSumTask,
    SummatoryTotientTask,
    DivisorSumTask,
    PrimeOmegaSumTask,
    SumDivisorCountsTask,
    PrimitiveRootCountTask,
    LegendreResidueWeightTask,
    ScaledStirlingFirstKindTask,
    DivisorSigmaSquareTask,
    MultiplicativeOrderTask,
    UnitaryDivisorSumTask,
    ModularMultiplicationSequenceTask,
    WeightedDivisorCountPrefixSequenceTask,
    GrayCodeSequenceTask,
    TotientShiftSequenceTask,
    SmallestFactorQuadraticTask,
    TotientProductSequenceTask,
    PolygonDiagonalsSequenceTask,
    PrimesBetweenSquaresTask,
    CentralBinomialPrimeValuationVectorTask,
    QuadraticResiduesPrimeTask,
]


def load_pool() -> list[dict[str, Any]]:
    task_implementations = sorted(
        MATH_TASKS,
        key=lambda task_implementation: (
            task_implementation.input_type,
            task_implementation.output_type,
            task_implementation.task_name,
        ),
    )
    return [
        {
            "category": "math",
            "task_id": task_id,
            "task_name": task_implementation.task_name,
            "input_type": task_implementation.input_type,
            "output_type": task_implementation.output_type,
            "task_implementation": task_implementation,
        }
        for task_id, task_implementation in enumerate(task_implementations)
    ]


def make_stage(
    task: dict[str, Any],
    task_number: int,
    input_names: list[str],
    _input: Any,
) -> tuple[Any, str, list[str], Any]:
    adapter_prompt = ""
    if _input is None:
        if task["input_type"] == "list":
            task_input = [random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG) for _ in range(LIST_LEN_MAX)]
        else:
            task_input = random.randint(-SCALAR_MAX_MAG, SCALAR_MAX_MAG)
    else:
        task_input = _input

    references = [f'{input_name}["result"]' for input_name in input_names]
    if len(input_names) > 1:
        if task["input_type"] == "scalar":
            adapter_name = f"val_{task_number:d}_join"
            task_input = join_adaptar_add_compute(task_input)
            adapter_prompt = join_adaptar_add_prompt(references, adapter_name)
        elif task["input_type"] == "list":
            adapter_name = f"list_{task_number:d}_join"
            task_input = join_adaptar_interleave_compute(task_input)
            adapter_prompt = join_adaptar_interleave_prompt(references, adapter_name)
        else:
            raise ValueError(f"Cannot join inputs for input_type={task['input_type']!r}.")
        references = [adapter_name]

    task_instance = task["task_implementation"](
        task_number,
        references,
        task_input,
        SCALAR_MAX_MAG,
        LIST_LEN_MAX,
    )
    prompt = task_instance.prompt.rstrip()
    if adapter_prompt:
        task_header, task_body = prompt.split("\n", 1)
        prompt = f"{task_header}\n{adapter_prompt.rstrip()}\n{task_body}"

    if bad_output(task_instance.out):
        raise ValueError(f"math:{task['task_id']}:{task['task_name']} produced an unusable output.")

    if _input is None:
        input_json = json.dumps({"result": task_input}, ensure_ascii=True)
        prompt = f"Define {input_names[0]} = {input_json}.\n{prompt}"

    return task_instance.out, prompt, [], task_input
