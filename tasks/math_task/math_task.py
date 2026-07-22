import math
from dataclasses import dataclass
from fractions import Fraction
import sympy
import random
from .math_helpers import *
from sympy.combinatorics.graycode import bin_to_gray
from sympy.combinatorics import Permutation


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
                 name,
                 task_ind,
                 inp,       
                 scalar_max_mag,
                 list_len_max):

        self.name = name
        self.task_ind = task_ind
        self.inp = inp
        self.adapted_inp = {}
        # dict of dicts: {"sympy_name" : {"chain_name" : str, "value" : int | float | list[int] | list[float]}}
        self.scalar_max_mag = scalar_max_mag
        self.list_len_max = list_len_max

        self.gen_params()   # generates whatever params needed, e.g. the other matrix in matrix mul, or dimension params

        self.out = plain_value(self.get_output())
        if self.out is None:
            raise ValueError(f"{self.name} returned None.")

        self.prompt = self.gen_prompt()

    def gen_params(self):
        raise NotImplementedError

    def get_output(self) -> int | float | list[int] | list[float]:
        raise NotImplementedError

    def gen_prompt(self) -> str:
        raise NotImplementedError


class MatrixMulTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if ('num_rows' not in inp.keys() or 'num_cols' not in inp.keys() or 
            inp['num_rows']['value'] is None or inp['num_cols']['value'] is None):
            raise ValueError("num_rows and num_cols are required for matrix multiplication task.")
        if 'matrix_1' not in inp.keys() or inp['matrix_1']['value'] is None:
            raise ValueError("matrix_1 is required for matrix multiplication task.")
        super().__init__('mat_mul', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        if 'matrix_2' in self.inp.keys() and isinstance(self.inp['matrix_2']['value'], sympy.MatrixBase):
            if self.inp['matrix_2']['value'].shape[0] != self.inp["num_cols"]["value"]:
                raise ValueError("matrix_2 shape is not compatible with matrix_1 shape.")
            return
        num_rows = self.inp["num_rows"]["value"]
        num_cols = self.inp["num_cols"]["value"]
        if 'matrix_2' not in self.inp.keys() or ('matrix_2' in self.inp.keys() and self.inp['matrix_2']['value'] is None):
            self.inp['matrix_2'] = {"chain_name": None, "value": _rand_matrix(num_cols, num_rows, -float(self.scalar_max_mag), float(self.scalar_max_mag))}

    def get_output(self):
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.inp["matrix_1"]["value"], mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        matrix_1 = list_to_matrix_compute(self.adapted_inp['matrix_1'], self.inp["num_rows"]["value"], self.inp["num_cols"]["value"])
        matrix_2 = self.inp["matrix_2"]["value"]
        out = list(matrix_1 * matrix_2)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp["matrix_1"]["chain_name"], adapted_inp_name, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_matrix_prompt(adapted_inp_name, f"M1_{self.task_ind:d}", self.inp["num_rows"]["value"], self.inp["num_cols"]["value"])
        prompt += f"Define M2_{self.task_ind:d} = {self.inp['matrix_2']['value']}.\n"
        prompt += f"Compute the matrix product of M1_{self.task_ind:d} and M2_{self.task_ind:d} and call the result mat_{out_var_name}.\n"
        prompt += matrix_to_list_prompt(f'mat_{out_var_name}', out_var_name)
        return prompt


class FWHTTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        super().__init__('fwht', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        if 'seq' not in self.inp.keys() or self.inp['seq']['value'] is None:
            raise ValueError("seq is required for FWHTTask.")

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], power_2=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max, to_int=True)
        padded = self.adapted_inp['seq']
        out = list(sympy.discrete.transforms.fwht(padded))
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        seq_name = self.inp['seq']['chain_name']
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(seq_name, adapted_inp_name, power_2=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max, to_int=True)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}].\n"
        prompt += f"Compute the unnormalized Walsh-Hadamard transform in Sylvester order. That is, for each j = 0, 1, ..., N-1, define\n"
        prompt += "H_j = sum_{k=0}^{N-1} ((-1) ** popcount(j & k)) * a_k, where `&` is bitwise AND and popcount counts the number of 1-bits.\n"
        prompt += f"Return the sequence [H_0, H_1, ..., H_{{N-1}}] and call it {out_var_name}.\n"
        return prompt


class NTTTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        super().__init__('ntt', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        if 'prime' in self.inp.keys() and self.inp['prime']['value'] is not None:
            seq = self.adapted_inp['seq']
            p = int(self.inp['prime']['value'])
            N = len(seq)
            if N <= 0:
                raise ValueError("NTT length N must be positive.")
            if not sympy.isprime(p):
                raise ValueError("NTT modulus must be prime.")
            if (p - 1) % N != 0:
                raise ValueError("N must divide p - 1 for the NTT root of unity.")
            return
        if 'seq' not in self.inp.keys() or self.inp['seq']['value'] is None:
            raise ValueError("seq is required for NTTTask.")
        if 'prime' not in self.inp.keys() or self.inp['prime']['value'] is None:
            seq = self.adapted_inp['seq']
            N = len(seq)
            m = 1
            while True:
                p = m * N + 1
                if sympy.isprime(p):
                    self.inp['prime'] = {"chain_name": None, "value": int(p)}
                    break
                m += 1

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        as_ints = self.adapted_inp['seq']
        prime = int(self.inp['prime']['value'])
        out = list(sympy.discrete.transforms.ntt(as_ints, prime))
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        seq_name = self.inp['seq']['chain_name']
        prime = self.inp['prime']['value']
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(seq_name, adapted_inp_name, power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}], and let p = {prime}.\n"
        prompt += f"All arithmetic in this transform is modulo p. Let g be the smallest positive primitive root modulo p, and define omega = g**((p - 1) // N) mod p. By construction, (N) divides (p-1), so (omega) has exact order (N) modulo (p).\n"
        prompt += f"Compute the number theoretic transform A = [A_0, A_1, ..., A_{{N-1}}], where for each j = 0, 1, ..., N-1, A_j = sum_{{k=0}}^{{N-1}} a_k * omega**(j*k) mod p.\n"
        prompt += f"Return each A_j as its least nonnegative residue in {{0, 1, ..., p-1}}. Call the resulting sequence {out_var_name}.\n"
        return prompt


class SubsetZetaTransformTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        super().__init__('subset_zeta_transform', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        if 'seq' not in self.inp.keys() or self.inp['seq']['value'] is None:
            raise ValueError("seq is required for SubsetZetaTransformTask.")

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        padded = self.adapted_inp['seq']
        out = list(sympy.discrete.transforms.mobius_transform(padded))
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        seq_name = self.inp['seq']['chain_name']
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(seq_name, adapted_inp_name, power_2=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} be a sequence of length 2^m. Index its entries by bitmasks S = 0, 1, ..., 2^m - 1.\n"
        prompt += f"Compute the subset-sum/zeta transform B, where for each bitmask S, B[S] = sum of {adapted_inp_name}[T] over all bitmasks T such that T is a subset of S.\n"
        prompt += f"Here, T is a subset of S means every 1-bit of T is also a 1-bit of S, equivalently (T & S) = T.\n"
        prompt += f"Return the values B[0], B[1], ..., B[2^m - 1] in increasing bitmask order and call the resulting sequence {out_var_name}.\n"
        return prompt


class KroneckerProductTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if ('num_rows' not in inp.keys() or 'num_cols' not in inp.keys() or
            inp['num_rows']['value'] is None or inp['num_cols']['value'] is None):
            raise ValueError("num_rows and num_cols are required for KroneckerProductTask.")
        super().__init__('kronecker_product', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        if 'matrix_1' not in self.inp.keys() or self.inp['matrix_1']['value'] is None:
            raise ValueError("matrix_1 is required for KroneckerProductTask.")

        r = self.inp['num_rows']['value']
        c = self.inp['num_cols']['value']

        if 'matrix_2' not in self.inp.keys() or self.inp['matrix_2']['value'] is None:
            self.inp['matrix_2'] = {"chain_name": None, "value": _rand_matrix(r, c, -float(self.scalar_max_mag), float(self.scalar_max_mag))}

    def get_output(self):
        r = self.inp['num_rows']['value']
        c = self.inp['num_cols']['value']
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.inp['matrix_1']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        inp_list = self.adapted_inp['matrix_1']
        m1 = list_to_matrix_compute(inp_list, r, c)
        m2 = self.inp['matrix_2']['value']
        out = list(sympy.kronecker_product(m1, m2))
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        r = self.inp['num_rows']['value']
        c = self.inp['num_cols']['value']
        m1_name = f"M1_{self.task_ind:d}"
        m2_name = f"M2_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['matrix_1']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_matrix_prompt(adapted_inp_name, m1_name, r, c)
        prompt += f"Define {m2_name} = {self.inp['matrix_2']['value']}.\n"
        prompt += f"Compute the Kronecker product of {m1_name} and {m2_name} and call the result mat_{out_var_name}.\n"
        prompt += matrix_to_list_prompt(f"mat_{out_var_name}", out_var_name)
        return prompt


class InterpolateEvalTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'points' not in inp.keys() or inp['points']['value'] is None:
            raise ValueError("points is required for InterpolateEvalTask.")
        super().__init__('interpolate_eval', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        if 'eval_point' not in self.inp.keys() or self.inp['eval_point']['value'] is None:
            self.inp['eval_point'] = {"chain_name": None, "value": random.choice(range(-100, 100))}

    def get_output(self):
        self.adapted_inp['points'] = adapt_list_compute(self.inp['points']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        points = self.adapted_inp['points']
        x = sympy.Symbol("x")
        points_dict = {i: val for i, val in enumerate(points)}
        poly = sympy.polys.polyfuncs.interpolate(points_dict, x)
        out = poly.subs(x, self.inp['eval_point']['value'])
        return out


    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"points_{self.task_ind:d}"
        last_index = len(self.adapted_inp['points']) - 1
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['points']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Construct the unique interpolating polynomial through the points "
        prompt += f"(0, {adapted_inp_name}[0]), (1, {adapted_inp_name}[1]), ..., ({last_index}, {adapted_inp_name}[{last_index}]).\n"
        prompt += f"Evaluate that polynomial at x = {self.inp['eval_point']['value']} and call the result {out_var_name}.\n"
        return prompt


class SummatoryTotientTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for SummatoryTotientTask.")
        super().__init__('summatory_totient', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sum(int(sympy.totient(k)) for k in range(1, n + 1))
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer k = 1, 2, ..., {adapted_inp_name}, compute Euler's totient function phi(k), the number of integers m with 1 <= m <= k such that gcd(m, k) = 1.\n"
        prompt += f"Compute the sum phi(1) + phi(2) + ... + phi({adapted_inp_name}) and call the result {out_var_name}.\n"
        return prompt


class DivisorSumTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for DivisorSumTask.")
        super().__init__('divisor_sum', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        conditioned = self.adapted_inp['n']
        out = sympy.divisor_sigma(conditioned, 1)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the sum of all positive divisors of {adapted_inp_name} and call the result {out_var_name}.\n"
        return prompt


class ConvolutionTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for ConvolutionTask.")
        super().__init__('convolution', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        if 'seq2' not in self.inp.keys() or self.inp['seq2']['value'] is None:
            self.inp['seq2'] = {"chain_name": None, "value": [random.randint(-5, 5) for _ in range(len(self.adapted_inp['seq']))]}

    def get_output(self):
        seq = self.adapted_inp['seq']
        seq2 = self.inp['seq2']['value']
        out = list(sympy.discrete.convolutions.convolution(seq, seq2))
        return out + [0] * (len(seq) + len(seq2) - 1 - len(out))

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Define seq2_{self.task_ind:d} = {self.inp['seq2']['value']}.\n"
        prompt += f"Compute the linear discrete convolution of {adapted_inp_name} and seq2_{self.task_ind:d}. Call the result {out_var_name}.\n"
        return prompt


class PolyRemainderTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for PolyRemainderTask.")
        super().__init__('poly_remainder', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        if 'seq2' not in self.inp.keys() or self.inp['seq2']['value'] is None:
            self.inp['seq2'] = {"chain_name": None, "value": [random.randint(-5, 5) for _ in range(len(self.adapted_inp['seq']))]}

    def get_output(self):
        seq = self.adapted_inp['seq']
        x = sympy.Symbol('x')
        P = sympy.Poly(seq, x)
        Q = sympy.Poly([1] + self.inp['seq2']['value'], x)
        out = sympy.rem(P, Q).all_coeffs()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        x = sympy.Symbol('x')
        poly_2 = sympy.Poly([1] + self.inp['seq2']['value'], x)
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f"poly_1_{self.task_ind:d}")
        prompt += f"Define poly_2_{self.task_ind:d} = {poly_2.as_expr()}.\n"
        prompt += f"Compute the polynomial remainder of poly_1_{self.task_ind:d} divided by poly_2_{self.task_ind:d}. Call the result poly_{out_var_name}.\n"
        prompt += poly_to_list_prompt(f'poly_{out_var_name}', out_var_name)
        return prompt


class PolyQuotientTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for PolyQuotientTask.")
        super().__init__('poly_quotient', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        x = sympy.Symbol('x')
        P = sympy.Poly(seq, x)
        Q = sympy.Poly([1, 1, 1], x)
        q, r = sympy.div(P, Q)
        out = q.all_coeffs() if hasattr(q, 'all_coeffs') else []
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f"P_{self.task_ind:d}")
        prompt += f"Compute the polynomial quotient when P_{self.task_ind:d}(x) is divided by the monic polynomial x**2 + x + 1. Call the result poly_{out_var_name}.\n"
        prompt += poly_to_list_prompt(f'poly_{out_var_name}', out_var_name)
        return prompt


class MatrixPolynomialTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'matrix_1' not in inp.keys() or inp['matrix_1']['value'] is None:
            raise ValueError("matrix_1 is required for MatrixPolynomialTask.")
        if 'num_rows' not in inp.keys() or inp['num_rows']['value'] is None:
            raise ValueError("num_rows is required for MatrixPolynomialTask.")
        super().__init__('matrix_polynomial', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['matrix_1'] = adapt_list_compute(self.inp['matrix_1']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['matrix_1']
        num_rows = self.inp['num_rows']['value']
        m = list_to_matrix_compute(seq, num_rows, num_rows)
        out = m**3 + m**2 + m
        out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp["matrix_1"]["chain_name"], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_matrix_prompt(adapted_inp_name, f"matrix_1_{self.task_ind:d}", self.inp["num_rows"]["value"], self.inp["num_rows"]["value"])
        prompt += f"Compute the matrix polynomial matrix_1_{self.task_ind:d}**3 + matrix_1_{self.task_ind:d}**2 + matrix_1_{self.task_ind:d}. Call the result mat_{out_var_name}.\n"
        prompt += matrix_to_list_prompt(f"mat_{out_var_name}", out_var_name)
        return prompt


class SortByDivisorCountTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for SortByDivisorCountTask.")
        super().__init__('sort_by_divisor_count', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        out = sorted(seq, key=lambda x: sympy.ntheory.factor_.divisor_count(abs(x)) if x != 0 else 0)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Sort {adapted_inp_name} based on the number of divisors of their absolute values in ascending order. If an element is 0, treat its divisor count as 0. Maintain stable sorting for ties. Call the result {out_var_name}.\n"
        return prompt


class MatrixFrobeniusSqTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for MatrixFrobeniusSqTask.")
        super().__init__('matrix_frobenius_sq', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        k = math.isqrt(len(seq))
        mat = list_to_matrix_compute(seq, k, k)
        out = sum(mat[i, j]**2 for i in range(k) for j in range(k))
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        k = math.isqrt(len(self.adapted_inp['seq']))
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_matrix_prompt(adapted_inp_name, f'mat_1_{self.task_ind:d}', k, k)
        prompt += f"Compute the Frobenius norm squared (sum of squares of all elements) of mat_1_{self.task_ind:d} and call the result {out_var_name}.\n"
        return prompt


class ConvexHullAreaTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for ConvexHullAreaTask.")
        super().__init__('convex_hull_area', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        pts = [sympy.geometry.Point(i, val) for i, val in enumerate(seq)]
        hull = sympy.geometry.convex_hull(*pts)
        out = 2 * hull.area if hasattr(hull, 'area') else 0
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Consider {adapted_inp_name} as a sequence of y-coordinates with x-coordinates being their indices (0, 1, 2, ...).\n"
        prompt += f"Compute the area of the convex hull of these 2D points, multiply the area by 2, and call the result {out_var_name}. If there are fewer than 3 points, or if all points are collinear, define the area to be 0.\n"
        return prompt


class MatrixNorm1Task(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for MatrixNorm1Task.")
        super().__init__('matrix_norm1', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        k = math.isqrt(len(seq))
        mat = list_to_matrix_compute(seq, k, k)
        out = max(sum(abs(mat[r, c]) for r in range(k)) for c in range(k))
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        k = math.isqrt(len(self.adapted_inp['seq']))
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_matrix_prompt(adapted_inp_name, f'matrix_1_{self.task_ind:d}', k, k)
        prompt += f"Compute the 1-norm of matrix_1_{self.task_ind:d} (the maximum absolute column sum) and call the result {out_var_name}.\n"
        return prompt


class MatrixTraceCubeTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for MatrixTraceCubeTask.")
        super().__init__('matrix_trace_cube', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        k = math.isqrt(len(seq))
        mat = list_to_matrix_compute(seq, k, k) 
        out = (mat**3).trace()
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        k = math.isqrt(len(self.adapted_inp['seq']))
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_matrix_prompt(adapted_inp_name, f'mat_1_{self.task_ind:d}', k, k)
        prompt += f"Compute the trace of mat_1_{self.task_ind:d}**3 and call the result {out_var_name}.\n"
        return prompt


class PermutationInversionsTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for PermutationInversionsTask.")
        super().__init__('permutation_inversions', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        indices = sorted(range(len(seq)), key=lambda x: seq[x])
        out = Permutation(indices).inversions()
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}].\nSort the indices 0, 1, ..., N-1 in ascending order by the pair (a_i, i).\n"
        prompt += f"In other words, sort by the corresponding sequence value, and break ties by the smaller original index.\nLet the resulting index list be pi = [pi_0, pi_1, ..., pi_{{N-1}}].\n"
        prompt += f"Compute the number of inversions in pi, meaning the number of pairs (r, s) such that 0 <= r < s < N and pi_r > pi_s.\nCall this result {out_var_name}.\n"
        return prompt


class PermutationOrderTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for PermutationOrderTask.")
        super().__init__('permutation_order', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        indices = sorted(range(len(seq)), key=lambda x: seq[x])
        out = Permutation(indices).order() if seq else 1
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"Let {adapted_inp_name} = [a_0, a_1, ..., a_{{N-1}}].\nSort the indices 0, 1, ..., N-1 in ascending order by the pair (a_i, i).\n"
        prompt += f"In other words, sort by the corresponding sequence value, and break ties by the smaller original index.\n"
        prompt += f"Let the resulting index list be pi = [pi_0, pi_1, ..., pi_{{N-1}}].\nTreat pi as the array form of a permutation of {{0, 1, ..., N-1}}, meaning pi maps r to pi_r.\n"
        prompt += f"Compute the order of this permutation: the smallest positive integer m such that applying pi exactly m times gives the identity permutation.\nIf N = 0, define the order to be 1.\n"
        prompt += f"Call the result {out_var_name}.\n"
        return prompt


class PolySquareFreeEvalTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for PolySquareFreeEvalTask.")
        super().__init__('poly_square_free_eval', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute( self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
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

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f'P_{self.task_ind:d}')
        prompt += f"If P_{self.task_ind:d}(x) is the zero polynomial, define {out_var_name} = 0. For a nonzero constant polynomial, use gcd(P_{self.task_ind:d}, P_{self.task_ind:d}') = 1.\n"
        prompt += f"Otherwise, let S_{self.task_ind:d}(x) = P_{self.task_ind:d}(x) / gcd(P_{self.task_ind:d}(x), P_{self.task_ind:d}'(x)), where the gcd is taken over Q[x] and made monic.\n"
        prompt += f"Do not further normalize S_{self.task_ind:d}(x): keep any scalar coefficient remaining after division by the monic gcd. "
        prompt += f"Evaluate S_{self.task_ind:d}(2) and call the result {out_var_name}.\n"
        return prompt


class PolyResultantFixedTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for PolyResultantFixedTask.")
        super().__init__('poly_resultant_fixed', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        x = sympy.Symbol('x')
        P = sympy.Poly(seq, x)
        Q = sympy.Poly([1, 1, 1], x)
        out = sympy.resultant(P, Q)
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += list_to_poly_prompt(adapted_inp_name, f'P_{self.task_ind:d}')
        prompt += f"Compute the resultant of P_{self.task_ind:d}(x) and the polynomial x**2 + x + 1 and call the result {out_var_name}.\n"
        return prompt


class PairwiseGcdSumTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'seq' not in inp.keys() or inp['seq']['value'] is None:
            raise ValueError("seq is required for PairwiseGcdSumTask.")
        super().__init__('pairwise_gcd_sum', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['seq'] = adapt_list_compute(self.inp['seq']['value'], needs_abs=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        seq = self.adapted_inp['seq']
        if len(seq) < 2:
            out = 0
        else:
            total = 0
            for i in range(len(seq)):
                for j in range(i+1, len(seq)):
                    total += sympy.gcd(seq[i], seq[j])
            out = int(total)
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"list_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_list_prompt(self.inp['seq']['chain_name'], adapted_inp_name, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag, list_len_max=self.list_len_max)
        prompt += f"For every pair of indices i, j with 0 <= i < j < len({adapted_inp_name}), compute gcd(|{adapted_inp_name}[i]|, |{adapted_inp_name}[j]|). Use the convention gcd(0, 0) = 0.\n"
        prompt += f"Compute the sum of all these GCD values and call the result {out_var_name}.\n"
        return prompt


class PrimeOmegaSumTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for PrimeOmegaSumTask.")
        super().__init__('prime_omega_sum', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], add_val=2, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sum(sympy.factorint(i).__len__() for i in range(2, n + 1))
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=2, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 2 to {adapted_inp_name}, compute omega(i), the number of distinct prime factors of i.\n"
        prompt += f"Compute the sum of all these omega values and call the result {out_var_name}.\n"
        return prompt


class SumDivisorCountsTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for SumDivisorCountsTask.")
        super().__init__('sum_divisor_counts', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], add_val=1, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sum(int(sympy.divisor_count(i)) for i in range(1, n + 1))
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute the number of positive divisors of i.\n"
        prompt += f"Compute the sum of all these divisor counts and call the result {out_var_name}.\n"
        return prompt



class PrimitiveRootCountTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for PrimitiveRootCountTask.")
        super().__init__('primitive_root_count', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], add_val=1, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        p = sympy.prime(n)
        # The number of primitive roots modulo a prime p is phi(p - 1)
        out = sympy.totient(p - 1)
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Find the {adapted_inp_name}-th prime number p.\n"
        prompt += f"Compute Euler's totient phi(p-1), which equals the count of primitive roots modulo p. Call the result {out_var_name}.\n"
        return prompt


class LegendreResidueWeightTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for LegendreResidueWeightTask.")
        super().__init__('legendre_residue_weight', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], add_val=25, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        p = int(sympy.nextprime(n + 2))
        out = sum(a * (int(sympy.legendre_symbol(a, p)) + 1) for a in range(1, n + 1))
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, add_val=25, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Let p be the smallest prime number strictly greater than {adapted_inp_name} + 2.\n"
        prompt += f"For each integer a = 1, 2, ..., {adapted_inp_name}, compute the Legendre symbol Legendre(a, p). This value is 1 if there exists an integer x such that x**2 is congruent to a modulo p, and -1 otherwise.\n"
        prompt += f"For each a, compute a * (Legendre(a, p) + 1). Then compute the sum of these values over all a = 1, 2, ..., {adapted_inp_name}.\n"
        prompt += f"Call the result {out_var_name}.\n"
        return prompt


class ScaledStirlingFirstKindTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for ScaledStirlingFirstKindTask.")
        super().__init__('scaled_stirling_first_kind', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        m = math.isqrt(n) + 7
        out = 2 * sympy.binomial(m, 3) + 3 * sympy.binomial(m, 4)
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Define m = floor(sqrt({adapted_inp_name})) + 7.\n"
        prompt += f"Compute the unsigned Stirling number of the first kind c(m, m - 2), which counts the number of permutations of m elements with exactly m - 2 disjoint cycles.\n"
        prompt += f"Call the result {out_var_name}.\n"
        return prompt


class DivisorSigmaSquareTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for DivisorSigmaSquareTask.")
        super().__init__('divisor_sigma_square', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = int(sympy.divisor_sigma(n, 2))
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the sum of the squares of all positive divisors of {adapted_inp_name} and call the result {out_var_name}.\n"
        return prompt


class MultiplicativeOrderTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for MultiplicativeOrderTask.")
        super().__init__('multiplicative_order', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sympy.ntheory.n_order(2, 2*n + 1)
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the multiplicative order of 2 modulo (2*{adapted_inp_name} + 1) and call the result {out_var_name}.\n"
        return prompt


class UnitaryDivisorSumTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for UnitaryDivisorSumTask.")
        super().__init__('unitary_divisor_sum', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sympy.functions.combinatorial.numbers.udivisor_sigma(n, 1)
        if hasattr(out, "as_expr"): out = out.as_expr()
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Compute the sum of the unitary divisors of {adapted_inp_name} (divisors d where gcd(d, {adapted_inp_name}/d) == 1) and call the result {out_var_name}.\n"
        return prompt


class ModularMultiplicationSequenceTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for ModularMultiplicationSequenceTask.")
        super().__init__('modular_multiplication_sequence', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        p = sympy.prime(n)
        out = [int((i * n) % p) for i in range(1, n + 1)]
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        prompt = f"Task {self.task_ind:d}:\n"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute (i * {adapted_inp_name}) modulo the {adapted_inp_name}-th prime number.\n"
        prompt += f"Compute the sequence of these values and call that sequence {out_var_name}.\n"
        return prompt


class WeightedDivisorCountPrefixSequenceTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for WeightedDivisorCountPrefixSequenceTask.")
        super().__init__('weighted_divisor_count_prefix_sequence', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        total = 0
        out = []
        for i in range(1, n + 1):
            total += i * int(sympy.divisor_count(n + i))
            out.append(total)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i = 1, 2, ..., {adapted_inp_name}, define A_i to be the sum over k = 1, 2, ..., i of k times the number of positive divisors of {adapted_inp_name} + k.\n"
        prompt += f"Return the list [A_1, A_2, ..., A_{{{adapted_inp_name}}}] in increasing order of i, so the first element corresponds to i = 1, the second element corresponds to i = 2, and so on.\n"
        prompt += f"Call the resulting list {out_var_name}.\n"
        return prompt


class GrayCodeSequenceTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for GrayCodeSequenceTask.")
        super().__init__('gray_code_sequence', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(bin_to_gray(bin(i + n)[2:]), 2) for i in range(1, n + 1)]
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, let x = i + {adapted_inp_name}.\nCompute the standard reflected binary Gray code of x, defined as gray(x) = x XOR floor(x / 2), where XOR is bitwise exclusive OR.\n"
        prompt += f"Return these Gray code values as ordinary decimal integers, not as binary strings, in the order i = 1, 2, ..., {adapted_inp_name}. Call the resulting sequence {out_var_name}.\n"
        return prompt


class TotientShiftSequenceTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for TotientShiftSequenceTask.")
        super().__init__('totient_shift_sequence', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(sympy.totient(i + n)) for i in range(1, n + 1)]
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute Euler's totient function phi(i + {adapted_inp_name}).\n"
        prompt += f"Compute the sequence of these values and call that sequence {out_var_name}.\n"
        return prompt


class SmallestFactorQuadraticTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for SmallestFactorQuadraticTask.")
        super().__init__('smallest_factor_quadratic', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(min(sympy.factorint(i**2 + n).keys())) if (i**2 + n) > 1 else 1 for i in range(1, n + 1)]
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute the smallest prime factor of i**2 + {adapted_inp_name}.\n"
        prompt += f"Compute the sequence of these values and call that sequence {out_var_name}.\n"
        return prompt


class TotientProductSequenceTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for TotientProductSequenceTask.")
        super().__init__('totient_product_sequence', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int(sympy.totient(i * n)) for i in range(1, n + 1)]
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute Euler's totient function of i * {adapted_inp_name}.\n"
        prompt += f"Compute the sequence of these values and call that sequence {out_var_name}.\n"
        return prompt


class PolygonDiagonalsSequenceTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for PolygonDiagonalsSequenceTask.")
        super().__init__('polygon_diagonals_sequence', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=3, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = [int((i + n) * (i + n - 3) // 2) for i in range(1, n + 1)]
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=3, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each integer i from 1 to {adapted_inp_name}, compute the number of diagonals in a convex polygon with i + {adapted_inp_name} vertices.\n"
        prompt += f"Compute the sequence of these values and call that sequence {out_var_name}.\n"
        return prompt


class PrimesBetweenSquaresTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for PrimesBetweenSquaresTask.")
        super().__init__('primes_between_squares', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = list(sympy.primerange(n**2 + 1, (n+1)**2))
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"List all prime numbers strictly between {adapted_inp_name}**2 and ({adapted_inp_name}+1)**2 in increasing order and call the result {out_var_name}.\n"
        return prompt


class CentralBinomialPrimeValuationVectorTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for CentralBinomialPrimeValuationVectorTask.")
        super().__init__('central_binomial_prime_valuation_vector', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
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

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=30, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"For each prime number p with p <= 2*{adapted_inp_name}, define e_p as follows.\n"
        prompt += f"For every positive integer j such that p**j <= 2*{adapted_inp_name}, compute floor((2*{adapted_inp_name}) / p**j) - 2*floor({adapted_inp_name} / p**j), and let e_p be the sum of these values over all such j.\n"
        prompt += f"For each prime p with p <= 2*{adapted_inp_name}, compute p * (e_p + 1).\n"
        prompt += f"Return these values as a list in increasing order of p and call the resulting list {out_var_name}.\n"
        return prompt


class QuadraticResiduesPrimeTask(MathTask):
    def __init__(self, task_ind, inp, scalar_max_mag, list_len_max):
        if 'n' not in inp.keys() or inp['n']['value'] is None:
            raise ValueError("n is required for QuadraticResiduesPrimeTask.")
        super().__init__('quadratic_residues_prime', task_ind, inp, scalar_max_mag, list_len_max)

    def gen_params(self):
        pass

    def get_output(self):
        self.adapted_inp['n'] = adapt_scalar_compute(self.inp['n']['value'], needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        n = self.adapted_inp['n']
        out = sympy.ntheory.quadratic_residues(sympy.prime(n)) if n > 0 else []
        if isinstance(out, sympy.MatrixBase): out = list(out)
        return out

    def gen_prompt(self):
        out_var_name = f"task_{self.task_ind:d}_out"
        adapted_inp_name = f"val_{self.task_ind:d}"
        prompt = f"Task {self.task_ind:d}:\n"
        prompt += adapt_scalar_prompt(self.inp['n']['chain_name'], adapted_inp_name, needs_abs=True, add_val=1, to_int=True, mod_value=self.scalar_max_mag)
        prompt += f"Let p be the {adapted_inp_name}-th prime number, using the convention that the 1st prime is 2.\nFind all distinct residues r in {{0, 1, ..., p-1}} for which there exists an integer x such that x^2 ≡ r mod p.\n"
        prompt += f"Include 0 as a quadratic residue. Return the residues in increasing order and call the resulting list {out_var_name}.\n"
        return prompt


# =============================================================================
# Internal Helpers
# =============================================================================

def _closest_factor_pair(n: int) -> tuple[int, int]:
    n = max(1, int(n))
    r = int(math.sqrt(n))
    while r > 1 and n % r != 0:
        r -= 1
    c = int(math.ceil(n / r))
    return r, c


def _square_dim_from_list_len(n: int) -> int:
    return max(1, int(math.ceil(math.sqrt(max(1, n)))))


def _rand_matrix(num_rows: int, num_cols: int, low: float, high: float):
    return sympy.Matrix([
        [random.randint(int(low), int(high)) for _ in range(num_cols)]
        for _ in range(num_rows)
    ])


def _rand_list(list_len, low, high):
    return [random.randint(int(low), int(high)) for _ in range(list_len)]


def _rand_poly_expr(degree: int, low: int = -20, high: int = 20):
    x = sympy.Symbol("x")
    coeffs = [random.randint(low, high) for _ in range(degree + 1)]
    return sympy.Poly(coeffs, x).as_expr()


def _ntt_prime_for_len(n: int) -> int:
    n = max(1, n)
    m = 1
    while True:
        p = m * n + 1
        if sympy.isprime(p):
            return int(p)
        m += 1

@dataclass(frozen=True)
class MathTaskSpec:
    # Describes how a math task fits into the hybrid chain.
    cls: type
    input_kind: str
    output_kind: str
    adapter: str

def _matrix_dim_ceiling(list_len_max):
    return max(1, int(math.ceil(math.sqrt(max(1, list_len_max)))))

def _square_dim(length, list_len_max):
    length = max(1, int(length))
    return max(1, min(_matrix_dim_ceiling(list_len_max), int(math.ceil(math.sqrt(length)))))

def _rect_dims(length, list_len_max):
    length = max(1, int(length))
    matrix_dim_max = _matrix_dim_ceiling(list_len_max)
    rows = max(1, min(matrix_dim_max, int(math.floor(math.sqrt(length))) or 1))
    cols = max(1, min(matrix_dim_max, int(math.ceil(length / rows))))
    return rows, cols

def _build_matrix_input(chain_name, value, rows, cols):
    return {
        "matrix_1": {"chain_name": chain_name, "value": list(value)},
        "num_rows": {"chain_name": None, "value": rows},
        "num_cols": {"chain_name": None, "value": cols},
    }

def _build_math_input(adapter, chain_name, value, list_len_max):
    # Adapt a generic chained value into the input shape expected by a math task.
    if adapter == "seq":
        return {"seq": {"chain_name": chain_name, "value": list(value)}}
    if adapter == "poly":
        return {"poly_1": {"chain_name": chain_name, "value": list(value)}}
    if adapter == "points":
        return {"points": {"chain_name": chain_name, "value": list(value)}}
    if adapter == "matrix_rect":
        rows, cols = _rect_dims(len(value), list_len_max)
        return _build_matrix_input(chain_name, value, rows, cols)
    if adapter == "matrix_square":
        dim = _square_dim(len(value), list_len_max)
        return _build_matrix_input(chain_name, value, dim, dim)
    if adapter == "scalar_n":
        return {"n": {"chain_name": chain_name, "value": value}}
    if adapter == "scalar_upper":
        return {"upper": {"chain_name": chain_name, "value": value}}
    if adapter == "gsm_input":
        return {"gsm_val": {"chain_name": chain_name, "value": value}}
    raise ValueError(f"Unknown adapter: {adapter}")

MATH_TASK_SPECS = [
    # list -> list
    MathTaskSpec(MatrixMulTask, "list", "list", "matrix_rect"),
    MathTaskSpec(FWHTTask, "list", "list", "seq"),
    MathTaskSpec(NTTTask, "list", "list", "seq"),
    MathTaskSpec(SubsetZetaTransformTask, "list", "list", "seq"),
    MathTaskSpec(KroneckerProductTask, "list", "list", "matrix_rect"),
    MathTaskSpec(ConvolutionTask, "list", "list", "seq"),
    MathTaskSpec(PolyRemainderTask, "list", "list", "seq"),
    MathTaskSpec(PolyQuotientTask, "list", "list", "seq"),
    MathTaskSpec(MatrixPolynomialTask, "list", "list", "matrix_square"),
    MathTaskSpec(SortByDivisorCountTask, "list", "list", "seq"),

    # list -> scalar
    MathTaskSpec(InterpolateEvalTask, "list", "scalar", "points"),
    MathTaskSpec(MatrixFrobeniusSqTask, "list", "scalar", "seq"),
    MathTaskSpec(ConvexHullAreaTask, "list", "scalar", "seq"),
    MathTaskSpec(MatrixNorm1Task, "list", "scalar", "seq"),
    MathTaskSpec(MatrixTraceCubeTask, "list", "scalar", "seq"),
    MathTaskSpec(PermutationInversionsTask, "list", "scalar", "seq"),
    MathTaskSpec(PermutationOrderTask, "list", "scalar", "seq"),
    MathTaskSpec(PolySquareFreeEvalTask, "list", "scalar", "seq"),
    MathTaskSpec(PolyResultantFixedTask, "list", "scalar", "seq"),
    MathTaskSpec(PairwiseGcdSumTask, "list", "scalar", "seq"),

    # scalar -> scalar
    MathTaskSpec(SummatoryTotientTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(DivisorSumTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(PrimeOmegaSumTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(SumDivisorCountsTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(PrimitiveRootCountTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(LegendreResidueWeightTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(ScaledStirlingFirstKindTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(DivisorSigmaSquareTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(MultiplicativeOrderTask, "scalar", "scalar", "scalar_n"),
    MathTaskSpec(UnitaryDivisorSumTask, "scalar", "scalar", "scalar_n"),

    # scalar -> list
    MathTaskSpec(ModularMultiplicationSequenceTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(WeightedDivisorCountPrefixSequenceTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(GrayCodeSequenceTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(TotientShiftSequenceTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(SmallestFactorQuadraticTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(TotientProductSequenceTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(PolygonDiagonalsSequenceTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(PrimesBetweenSquaresTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(CentralBinomialPrimeValuationVectorTask, "scalar", "list", "scalar_n"),
    MathTaskSpec(QuadraticResiduesPrimeTask, "scalar", "list", "scalar_n"),
]
