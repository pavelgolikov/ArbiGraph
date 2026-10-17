"""RLVE task implementations and their chain-facing prompts."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import defaultdict, deque
from copy import deepcopy
from fractions import Fraction
from functools import cmp_to_key
from itertools import combinations, permutations
from typing import Any
import heapq
import json
import math
import random
import re

from tasks.arbigraph_tools import comparator, validate_limit


MAX_BANK_DRAWS = 400
SETUP_ATTEMPTS = 20

# How this category's answers are compared, wherever two of them meet.
COMPARATOR = "numeric"
same_answer = comparator(COMPARATOR)


class _RLVETask:
    """One frozen draw of the task's parameters, answering every input in its bank.

    The parameters drawn in _setup are the task as the model is shown it. The
    input is not part of them: every input in a task's bank is answered by these
    same parameters, so a wrong input looks exactly like the right one from the
    prompt.
    """

    def __init__(self, seed=0):
        self.name = self.task_name
        self.seed = seed
        self._setup(random.Random(seed))

    def _setup(self, rng):
        pass

    @classmethod
    def generate_inputs(cls, num_inputs):
        """Build a frozen task and a bank of inputs with distinct accepted answers.

        A draw of parameters whose answers collide is thrown away and redrawn,
        since an unlucky one can leave a task answering nearly the same way
        whatever it is given.
        """
        validate_limit(num_inputs, "num_inputs")
        last_error = None
        for _ in range(SETUP_ATTEMPTS):
            try:
                instance = cls(random.randrange(2**32))
            except Exception as exc:
                last_error = exc
                continue
            bank = instance.draw_bank(num_inputs)
            if bank is not None:
                return instance, bank
        raise ValueError(
            f"Could not generate {num_inputs} distinct accepted inputs for "
            f"{cls.task_name}: {last_error}"
        )

    def draw_bank(self, num_inputs):
        """Draw inputs for these parameters until each one has its own answer."""
        bank = []
        for _ in range(num_inputs):
            for _ in range(MAX_BANK_DRAWS):
                value = self.sample_input(random)
                try:
                    output = self.solution_generator(deepcopy(value))
                    if not _valid_output(output):
                        continue
                    json.dumps(output, allow_nan=False)
                    # Two inputs sharing an answer would let a wrong one pass.
                    if any(same_answer(output, previous["output"]) for previous in bank):
                        continue
                except Exception:
                    continue
                bank.append({"input": value, "output": output})
                break
            else:
                return None
        return bank

    def _base_prompt_fields(self, result_name):
        N = getattr(self, "N", None)
        return {
            "result_name": result_name,
            "N": N,
            "N_minus_1": N - 1 if N is not None else None,
            "N_plus_1": N + 1 if N is not None else None,
        }

    def prompt_generator(self, task_number):
        """Describe the task against the input the model is given for this node."""
        result_name = f"{self.task_name}_result_{task_number:d}"
        body = self._format_prompt(f"task_{task_number:d}_input", result_name)
        return (
            f"{body.rstrip()}\n"
            f"Output the result as task_{task_number:d}_out = "
            f'{{"result": {result_name}}}, '
            f"replacing {result_name} with its calculated value."
        )


def _integer_list(rng, length, low, high):
    return [rng.randint(low, high) for _ in range(length)]


RATIO_MOD = 998244353


def _ratio_modulo_prime(numerator, denominator):
    """Encode a rational as one integer: p * q^-1 mod RATIO_MOD, for p/q in lowest terms.

    RATIO_MOD is prime and every denominator here is far smaller than it, so the inverse
    always exists and the map is injective: distinct ratios give distinct integers.
    """
    value = Fraction(numerator, denominator)
    inverse = pow(value.denominator, RATIO_MOD - 2, RATIO_MOD)
    return (value.numerator % RATIO_MOD) * inverse % RATIO_MOD


# class AndOr_Sequence_Counting(_RLVETask):
    # prompt_template = (
        # 'You are given an integer array `A` of length {N}:\n'
        # '{A}\n'
        # '\n'
        # 'Please count the number of valid integer arrays `B` of length {N} that satisfy the following '
        # 'conditions:\n'
        # '- For all indices 0 <= i <= {N_minus_1}, the value B[i] must be in the range: 0 <= B[i] < 2^{M} = '
        # '{power_2_M}\n'
        # '- For all indices 0 <= i < {N_minus_1}, the following bitwise conditions hold:\n'
        # '  - (A[i] & B[i]) <= (A[i + 1] & B[i + 1])\n'
        # '  - (A[i] | B[i]) >= (A[i + 1] | B[i + 1])\n'
        # '  - (Here, `&` is the bitwise AND operator and `|` is the bitwise OR operator.)\n'
        # '\n'
        # 'Let {result_name} be the number of valid arrays `B` that satisfy all the above conditions.'
    # )
#
    # def _format_prompt(self, input_reference, result_name):
        # N = getattr(self, "N", None)
        # fields = self._base_prompt_fields(result_name)
        # fields.update(M=self.M, power_2_M=2**self.M, A=_indexed("A", input_reference, N))
        # return self.prompt_template.format(**fields)
#    # N = 5
    # M = 20
#
    # def sample_input(self, rng):
        # return _integer_list(rng, self.N, 0, 2**self.M - 1)
#
    # def solution_generator(self, input_value):
        # A, N, M = input_value, self.N, self.M
#
        # def dp(required_bit):
            # previous = [[1 if left <= right else 0 for right in range(N)] for left in range(N)]
            # for bit in range(M):
                # current = [[0] * N for _ in range(N)]
                # prefix = [0]
                # for value in A:
                    # prefix.append(prefix[-1] + ((value >> bit) & 1))
                # for left in range(N):
                    # for right in range(left, N):
                        # for split in range(left - 1, right + 1):
                            # ones = prefix[right + 1] - prefix[split + 1]
                            # if (required_bit and ones != right - split) or (
                                # not required_bit and ones != 0
                            # ):
                                # continue
                            # left_count = previous[left][split] if split >= left else 1
                            # right_count = previous[split + 1][right] if split < right else 1
                            # current[left][right] += left_count * right_count
                # previous = current
            # return previous[0][N - 1]
#
        # return dp(True) * dp(False)


class BoundedSubarrayCounting(_RLVETask):
    prompt_template = (
        'Given an array A of length {N}:\n'
        '{A}\n'
        '\n'
        'Repeat array A {M} times to form a new array B of length {N} * {M} = {NM}. In the new array B, '
        'how many (nonempty) contiguous subarrays have a total sum less than or equal to {K}?\n'
        '\n'
        'Let {result_name} be the total number of (nonempty) subarrays in B whose sum is less than '
        'or equal to {K}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(M=self.M, NM=N * self.M, A=input_reference, K=self.K)
        return self.prompt_template.format(**fields)

    N = 6
    M = 5

    def _setup(self, rng):
        initial = _integer_list(rng, self.N, 1, self.N)
        self.K = rng.randint(max(initial), sum(initial) * self.M)

    def sample_input(self, rng):
        while True:
            values = _integer_list(rng, self.N, 1, self.N)
            if max(values) <= self.K <= sum(values) * self.M:
                return values

    def solution_generator(self, input_value):
        A, N, M, K = input_value, self.N, self.M, self.K
        prefix = [0]
        for value in A:
            prefix.append(prefix[-1] + value)
        total = prefix[-1]
        answer = 0
        all_full = M * (M - 1) // 2 * N
        for i in range(1, N + 1):
            if prefix[i] < K:
                full = (K - prefix[i]) // total
                if full < M - 1:
                    answer += i * (full + 1) + full * (full + 1) // 2 * N
                    extra = (K - prefix[i]) % total
                    j = bisect_left(prefix, total - extra)
                    answer += (i + full * N + N - j) * (M - 1 - full)
                else:
                    answer += i * M + all_full
            else:
                j = bisect_left(prefix, prefix[i] - K)
                answer += (i - j) * M
        return answer


class MaxDifferentGroupPairDivision(_RLVETask):
    prompt_template = (
        'You are given an array A of {N} integers: {A}\n'
        '\n'
        'Initially, the entire array is one single block. Let S = 0. You need to perform the following '
        'operation exactly {K} times:\n'
        '- Choose a position `i` such that A[i] and A[i + 1] are still in the same block.\n'
        '- Split the block into two parts: the first ends at A[i], the second starts at A[i + 1].\n'
        '- Let `sum1` and `sum2` be the sums of the two blocks. Then, update S += sum1 × sum2.\n'
        '\n'
        'After {K} operations, you will have {K} + 1 blocks. Try your best to **maximize** the final value '
        'of S.\n'
        '\n'
        'Let {result_name} be the maximum possible final value of S.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(A=_indexed("A", input_reference, N), K=self.K)
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        self.K = rng.randint(2, self.N - 2)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 0, self.N)

    def solution_generator(self, input_value):
        prefix = [0]
        for value in input_value:
            prefix.append(prefix[-1] + value)
        best = 0
        for cuts in combinations(range(1, self.N), self.K):
            bounds = (0,) + cuts + (self.N,)
            sums = [prefix[bounds[i + 1]] - prefix[bounds[i]] for i in range(self.K + 1)]
            best = max(best, sum(sums[i] * sums[j] for i in range(len(sums)) for j in range(i + 1, len(sums))))
        return best


class MoneyChargingGame(_RLVETask):
    prompt_template = (
        'There are {N} nodes, each associated with values A[i][1], A[i][2], and A[i][3]. For each node '
        '`i`, define: P[i][j] = A[i][j] / (A[i][1] + A[i][2] + A[i][3]) for j = 1, 2, 3. The values A are '
        'given as follows:\n'
        '{A}\n'
        '\n'
        'We define the following random process:\n'
        '1. For each node `i`, randomly assign W[i] = j with probability P[i][j] for j = 1, 2, 3.\n'
        '2. Starting from an empty set, repeatedly select a node `i` with probability proportional to '
        'W[i], and add it to the set (duplicates are allowed). Continue until all nodes are in the set.\n'
        '3. Let T[i] denote the first time node `i` is added to the set.\n'
        '\n'
        'You are also given a set of constraints (each of the form T[u] < T[v]) that correspond to the '
        'edges of an undirected tree:\n'
        '{T_inequalities}\n'
        'Please compute the total probability that all the above T[u] < T[v] conditions hold during the '
        'random process.\n'
        '\n'
        'Let {result_name} be that total probability modulo {MOD}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            A="\n".join(
                f"A[{i}][1, 2, 3] = [{input_reference}[{i}], {self.B[i]}, {self.C[i]}]"
                for i in range(N)
            ),
            T_inequalities="\n".join(f"T[{u}] < T[{v}]" for u, v in self.edges),
            MOD=self.MOD,
        )
        return self.prompt_template.format(**fields)

    N = 5
    MOD = 998244353

    def _setup(self, rng):
        N = self.N
        self.B = _integer_list(rng, N, 1, N)
        self.C = _integer_list(rng, N, 1, N)
        order = list(range(N))
        rng.shuffle(order)
        swap_probability = rng.random()
        self.edges = []
        for i in range(1, N):
            u, v = order[rng.randint(0, i - 1)], order[i]
            if rng.random() < swap_probability:
                u, v = v, u
            self.edges.append((u, v))
        rng.shuffle(self.edges)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        A, B, C, N, MOD = input_value, self.B, self.C, self.N, self.MOD
        inverses = [0] + [pow(i, MOD - 2, MOD) for i in range(1, 3 * N + 1)]
        scale = [pow(a + b + c, MOD - 2, MOD) for a, b, c in zip(A, B, C)]
        graph = [[] for _ in range(N)]
        for u, v in self.edges:
            graph[v].append((u, 1))
            graph[u].append((v, 0))
        values = [None] * N
        sizes = [0] * N

        def visit(node, parent):
            sizes[node] = 1
            current = [0] * 4
            current[1] = A[node] * scale[node] % MOD
            current[2] = B[node] * scale[node] * 2 % MOD
            current[3] = C[node] * scale[node] * 3 % MOD
            for child, subtract in graph[node]:
                if child == parent:
                    continue
                visit(child, node)
                merged = [0] * (3 * (sizes[node] + sizes[child]) + 1)
                for i in range(1, 3 * sizes[node] + 1):
                    for j in range(1, 3 * sizes[child] + 1):
                        product = current[i] * values[child][j] % MOD
                        if subtract:
                            merged[i + j] = (merged[i + j] - product) % MOD
                            merged[i] = (merged[i] + product) % MOD
                        else:
                            merged[i + j] = (merged[i + j] + product) % MOD
                sizes[node] += sizes[child]
                current = merged
            for i in range(1, 3 * sizes[node] + 1):
                current[i] = current[i] * inverses[i] % MOD
            values[node] = current

        visit(0, -1)
        return sum(values[0][1 : 3 * N + 1]) % MOD


class ProtectingFlowers(_RLVETask):
    prompt_template = (
        'You are given two arrays `T` and `D`, each containing {N} integers:\n'
        '{T_and_D}\n'
        '\n'
        'Consider a permutation of 1 to {N}, denoted as p[1], p[2], ..., p[{N}]:\n'
        '- Define S[i] as the sum of T[p[j]] for all 1 ≤ j < i (so S[1] = 0).\n'
        '- The objective is to minimize the total sum of S[i] * D[p[i]] for i from 1 to {N}.\n'
        '\n'
        'Let {result_name} be the minimum possible value of that total sum.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["T_and_D"] = "\n".join(
            f"T[{i}]={input_reference}[{i - 1}] D[{i}]={self.D[i - 1]}"
            for i in range(1, N + 1)
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.D = _integer_list(rng, self.N, 1, self.N)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        pairs = list(zip(input_value, self.D))

        def compare(left, right):
            return (left[0] * right[1] > left[1] * right[0]) - (
                left[0] * right[1] < left[1] * right[0]
            )

        pairs.sort(key=cmp_to_key(compare))
        remaining = sum(d for _, d in pairs)
        answer = 0
        for time, damage in pairs:
            remaining -= damage
            answer += time * remaining
        return answer


class RangeConstrained_IncreasingSequence_Counting(_RLVETask):
    prompt_template = (
        'Count the number of integer sequences A[0], A[1], ..., A[{N_minus_1}] of length {N} such that:\n'
        '- For each A[i], it is either 0 or an integer in [L[i], R[i]]\n'
        '- At least one A[i] is greater than 0\n'
        '- All non-zero A[i] form a strictly increasing sequence in order (i.e., if A[i] > 0 and A[j] > 0 '
        'with i < j, then A[i] < A[j])\n'
        '\n'
        'The bounds L[i] and R[i] for each position are given as:\n'
        '{L_and_R}\n'
        '\n'
        'Let {result_name} be the number of such sequences modulo {MOD}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            L_and_R="\n".join(
                f"L[{i}]={input_reference}[{i}] R[{i}]={self.R[i]}" for i in range(N)
            ),
            MOD=self.MOD,
        )
        return self.prompt_template.format(**fields)

    N = 5
    MOD = 1_000_000_007

    def _setup(self, rng):
        initial_l = _integer_list(rng, self.N, 1, self.N**2)
        self.R = [rng.randint(value, self.N**2) for value in initial_l]

    def sample_input(self, rng):
        return [rng.randint(1, right) for right in self.R]

    def solution_generator(self, input_value):
        N, MOD = self.N, self.MOD
        lower, upper = input_value.copy(), self.R.copy()
        coordinates = sorted(set(lower + [value + 1 for value in upper]))
        for i in range(N):
            lower[i] = bisect_left(coordinates, lower[i])
            upper[i] = bisect_left(coordinates, upper[i] + 1)
        inverse = [0, 1] + [0] * (N - 1)
        for i in range(2, N + 1):
            inverse[i] = (MOD - MOD // i) * inverse[MOD % i] % MOD
        ways = [1] + [0] * N
        coefficients = [0] * (N + 1)
        for segment in range(len(coordinates) - 1):
            length = coordinates[segment + 1] - coordinates[segment]
            coefficients[0] = 1
            for k in range(1, N + 1):
                coefficients[k] = coefficients[k - 1] * (length + k - 1) % MOD * inverse[k] % MOD
            for i in range(N, 0, -1):
                if lower[i - 1] <= segment < upper[i - 1]:
                    addition, count, coefficient = 0, 1, length
                    for previous in range(i - 1, -1, -1):
                        addition = (addition + coefficient * ways[previous]) % MOD
                        if previous and lower[previous - 1] <= segment < upper[previous - 1]:
                            count += 1
                            coefficient = coefficients[count]
                    ways[i] = (ways[i] + addition) % MOD
        return sum(ways[1:]) % MOD


class SalesmanFatigue(_RLVETask):
    prompt_template = (
        'You are given {N} pairs of integers `(S[i], A[i])` for `0 <= i < {N}`, provided as:\n'
        '{S_and_A}\n'
        '\n'
        '**Note:** The array `S` is sorted in non-decreasing order: `S[0] <= S[1] <= ... <= '
        'S[{N_minus_1}]`\n'
        '\n'
        'Please select k distinct pairs `i_1, i_2, ..., i_k` and maximize the following expression: '
        '`max(S[i_1], S[i_2], ..., S[i_k]) * 2 + A[i_1] + A[i_2] + ... + A[i_k]` (i.e., the sum of the '
        'selected A[i] values plus the maximum S[i] value multiplied by 2).\n'
        'Please compute the **maximum value of this expression** for each k = 1 to {N}.\n'
        '\n'
        'Let {result_name} be a single list with the maximum value of this expression for each k = '
        '1 to {N}, in order.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["S_and_A"] = "\n".join(
            f"S[{i}]={self.S[i]} A[{i}]={input_reference}[{i}]" for i in range(N)
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.S = sorted(_integer_list(rng, self.N, 1, self.N**2 // 2))

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        pairs = sorted(zip(self.S, input_value), key=lambda item: -item[1])
        prefix = [0]
        for _, value in pairs:
            prefix.append(prefix[-1] + value)
        left_max, right_max = [], [0] * self.N
        current = 0
        for position, _ in pairs:
            current = max(current, 2 * position)
            left_max.append(current)
        current = 0
        for i in range(self.N - 1, -1, -1):
            current = max(current, 2 * pairs[i][0] + pairs[i][1])
            right_max[i] = current
        return [
            max(prefix[x] + left_max[x - 1], prefix[x - 1] + right_max[x - 1])
            for x in range(1, self.N + 1)
        ]


class SpiralMatrix(_RLVETask):
    prompt_template = (
        'You are given a 2D integer matrix of size {M} x {N}:\n'
        '{matrix}\n'
        '\n'
        'Return all elements of the matrix in a clockwise spiral order, starting from the top-left corner. '
        'More precisely:\n'
        '- Start from the top-left corner and move right until you reach the right edge.\n'
        '- Then, move down until you reach the bottom-right corner.\n'
        '- Then, move left until you reach the bottom-left corner.\n'
        '- Then, move up until you reach the top-right corner.\n'
        '- Continue this inward spiral traversal until all elements have been visited exactly once.\n'
        '\n'
        '---\n'
        '\n'
        '**Example 1**\n'
        '\n'
        'You are given an integer matrix of size 3 x 3:\n'
        '1 2 3\n'
        '4 5 6\n'
        '7 8 9\n'
        '\n'
        'The spiral order is [1, 2, 3, 6, 9, 8, 7, 4, 5].\n'
        '\n'
        '**Example 2**\n'
        '\n'
        'You are given an integer matrix of size 3 x 4:\n'
        '1 2 3 4\n'
        '5 6 7 8\n'
        '9 10 11 12\n'
        '\n'
        'The spiral order is [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7].\n'
        '\n'
        '---\n'
        '\n'
        'Let {result_name} be a single list with all {MN} elements of the matrix in clockwise '
        'spiral order.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            M=self.rows,
            N=self.columns,
            MN=self.rows * self.columns,
            matrix="\n".join(
                " ".join(
                    f"{input_reference}[{row}][{column}]"
                    for column in range(self.columns)
                )
                for row in range(self.rows)
            ),
        )
        return self.prompt_template.format(**fields)

    def _setup(self, rng):
        self.rows = 4
        self.columns = 2

    def sample_input(self, rng):
        return [
            _integer_list(rng, self.columns, 1, self.rows * self.columns)
            for _ in range(self.rows)
        ]

    def solution_generator(self, input_value):
        matrix = input_value
        answer = []
        top, bottom, left, right = 0, self.rows - 1, 0, self.columns - 1
        while top <= bottom and left <= right:
            answer.extend(matrix[top][left : right + 1])
            top += 1
            for row in range(top, bottom + 1):
                answer.append(matrix[row][right])
            right -= 1
            if top <= bottom:
                answer.extend(reversed(matrix[bottom][left : right + 1]))
                bottom -= 1
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    answer.append(matrix[row][left])
                left += 1
        return answer


class TaskArrangement(_RLVETask):
    prompt_template = (
        'You are given {N} tasks, numbered from 1 to {N}. Each task i (1 <= i <= {N}) takes T[i] units of '
        'time to complete individually and has a cost coefficient F[i]. The values are given as:\n'
        '{T_and_F}\n'
        '\n'
        'You may divide these tasks (in order) into any number of **consecutive batches**. Let the total '
        'number of batches be k (k >= 1), and let end[1], end[2], ..., end[k] (1 <= end[1] < end[2] < ... '
        '< end[k] = {N}) denote the last task index in each batch.\n'
        '- This means:\n'
        '    + Batch 1 contains tasks 1 to end[1]\n'
        '    + Batch 2 contains tasks end[1] + 1 to end[2]\n'
        '    + ...\n'
        '    + Batch k contains tasks end[k - 1] + 1 to end[k] (with end[k] = {N})\n'
        '\n'
        '- Before starting each batch, the machine must spend an additional {S} units of startup time.\n'
        '- The time to **complete** a batch is the sum of T[i] for all tasks in that batch.\n'
        "- Therefore, the **total completion time** of each task in a batch is the sum of the batch's "
        'startup time ({S}) and the total time of all tasks in that batch.\n'
        '- All tasks in a batch are considered to finish **simultaneously**, at the end of that batch.\n'
        '\n'
        '- Tasks are completed in the order defined by the batch division.\n'
        '- The cost of each task is equal to **the time when its batch finishes (after all previous '
        'batches, if any, have completed and the current batch has been processed), multiplied by F[i]**.\n'
        '- The **total cost** is the sum of the costs of all tasks.\n'
        '\n'
        'Try your best to find a batch division (end[1], end[2], ..., end[k]) that **minimizes the total '
        'cost**.\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            S=self.S,
            T_and_F="\n".join(
                f"T[{i}]={input_reference}[{i - 1}] F[{i}]={self.F[i - 1]}"
                for i in range(1, N + 1)
            ),
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        self.S = rng.randint(0, self.N * 3)
        self.F = _integer_list(rng, self.N, 1, self.N)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        N, S = self.N, self.S
        T, F = [0] + input_value, [0] + self.F
        prefix_t, suffix_f = [0] * (N + 1), [0] * (N + 2)
        for i in range(1, N + 1):
            prefix_t[i] = prefix_t[i - 1] + T[i]
        for i in range(N, 0, -1):
            suffix_f[i] = suffix_f[i + 1] + F[i]
        best = [0] + [None] * N
        for i in range(1, N + 1):
            best[i] = min(
                best[j - 1] + (S + prefix_t[i] - prefix_t[j - 1]) * suffix_f[j]
                for j in range(1, i + 1)
            )
        return best[N]


class TreeCenter(_RLVETask):
    prompt_template = (
        'You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, '
        'labeled from `0` to `{N_minus_1}`.\n'
        '\n'
        'Each vertex has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex i:\n'
        '{C}\n'
        '\n'
        'The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented '
        'as a tuple `(u, v, w)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v` '
        'with weight `w`:\n'
        '{edges}\n'
        '\n'
        'Your task is to select a single vertex `r` (where `r` is in the range 0 to {N_minus_1}).\n'
        'Try your best to **minimize** dist(0, r) * C[0] + dist(1, r) * C[1] + ... + dist({N_minus_1}, r) '
        '* C[{N_minus_1}], where `dist(i, j)` is the distance between vertices i and j in the tree. The '
        'distance between two vertices is defined as the sum of the weights of the edges on the unique '
        'path connecting them (since the graph is a tree, there is exactly one unique path between any two '
        'vertices).\n'
        '\n'
        'Let {result_name} be the minimum possible value of that weighted sum.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            C="\n".join(f"C[{i}]={input_reference}[{i}]" for i in range(N)),
            edges="\n".join(f"({u}, {v}, {weight})" for u, v, weight in self.edges),
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        order = list(range(self.N))
        rng.shuffle(order)
        self.edges = []
        for index in range(1, self.N):
            u, v = order[index], rng.choice(order[:index])
            self.edges.append((u, v, rng.randint(1, self.N)))

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        graph = [[] for _ in range(self.N)]
        for u, v, weight in self.edges:
            graph[u].append((v, weight))
            graph[v].append((u, weight))
        return min(
            sum(input_value[node] * distance for node, distance in _tree_distances(graph, root))
            for root in range(self.N)
        )


def _tree_distances(graph, root):
    stack = [(root, -1, 0)]
    while stack:
        node, parent, distance = stack.pop()
        yield node, distance
        for child, weight in graph[node]:
            if child != parent:
                stack.append((child, node, distance + weight))


class WeightedBinaryTree(_RLVETask):
    prompt_template = (
        'You are given a binary tree with {N} nodes, labeled from 0 to {N_minus_1}.\n'
        'The **in-order traversal** of the tree is: `0, 1, ..., {N_minus_1}` — that is, the in-order '
        'sequence is fixed in increasing order of node labels.\n'
        '\n'
        'Each node `i` has an associated score `d_i` (where `0 ≤ i < {N}`), given as:\n'
        '{scores}\n'
        '\n'
        'The **score of a binary tree** is defined recursively as follows:\n'
        '- `score(tree) = score(left_subtree) × score(right_subtree) + d_i`, where `i` is the root of the '
        'current subtree.\n'
        '- If a subtree is **empty**, its score is defined to be `1`.\n'
        '- If a node is a **leaf**, its score is simply `d_i` (ignore its empty subtrees).\n'
        '\n'
        'Your task is to construct the binary tree that satisfies the above rules and has the **maximum '
        'possible score**.\n'
        '\n'
        'Let {result_name} be the maximum possible score of the whole tree.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            scores="\n".join(f"d_{i}={input_reference}[{i}]" for i in range(N)),
            all_node_sequence=" ".join(map(str, range(N))),
        )
        return self.prompt_template.format(**fields)

    N = 6

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, 5)

    def solution_generator(self, input_value):
        N = self.N
        best = [[0] * N for _ in range(N)]
        for i, score in enumerate(input_value):
            best[i][i] = score
        for length in range(2, N + 1):
            for left in range(N - length + 1):
                right = left + length - 1
                for root in range(left, right + 1):
                    left_value = best[left][root - 1] if root > left else 1
                    right_value = best[root + 1][right] if root < right else 1
                    best[left][right] = max(
                        best[left][right], left_value * right_value + input_value[root]
                    )
        return best[0][N - 1]


class BoxScheduling(_RLVETask):
    prompt_template = (
        'You are given a sequence C: {C}\n'
        '\n'
        'Now, please determine two non-negative integer sequences X[1], ..., X[{N_minus_1}] and Y[1], ..., '
        'Y[{N_minus_1}] such that:\n'
        '- For 1 ≤ i ≤ {N_minus_1}, define: Pos[i] = (C[i] + {D} × X[i] + Y[i]) mod {N}\n'
        '- The values Pos[1], ..., Pos[{N_minus_1}] must be all distinct.\n'
        '- No Pos[i] can be equal to {S}.\n'
        '- Among all valid solutions:\n'
        '  + First, minimize the lexicographical order of sequence Y.\n'
        '  + If multiple solutions have the same Y, then choose the one with the smallest lexicographical '
        'order of X.\n'
        '\n'
        'Let {result_name} be a single list with Pos[1], ..., Pos[{N_minus_1}] in order.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(C=_indexed("C", input_reference, N - 1, 1), D=self.D, S=self.S)
        return self.prompt_template.format(**fields)

    def _setup(self, rng):
        self.N = 6
        for _ in range(int(self.N**0.5)):
            self.D = rng.randint(1, self.N - 1)
            if math.gcd(self.D, self.N) > 1:
                break
        self.S = rng.randint(0, self.N - 1)

    def sample_input(self, rng):
        return _integer_list(rng, self.N - 1, 0, self.N - 1)

    def solution_generator(self, input_value):
        N, D, S = self.N, self.D, self.S
        values = [0] + input_value
        parent = list(range(N))

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        gcd = math.gcd(D, N)
        cycle_size = N // gcd
        target = S % gcd
        available = []
        positions = [0] * N
        for residue in range(gcd):
            if residue != target:
                insort(available, (residue, cycle_size))
            else:
                positions[0] = S
                parent[S] = find((S + D) % N)
                if cycle_size > 1:
                    insort(available, (residue, cycle_size - 1))
        for i in range(1, N):
            key = values[i] % gcd
            index = bisect_left(available, (key, -1))
            if index == len(available):
                index = 0
            residue, count = available.pop(index)
            if count > 1:
                insort(available, (residue, count - 1))
            if residue >= key:
                start = (values[i] + residue - key) % N
            else:
                start = (((values[i] // gcd) + 1) * gcd + residue) % N
            positions[i] = find(start)
            parent[positions[i]] = find((positions[i] + D) % N)
        return positions[1:]


class CantorExpansion(_RLVETask):
    prompt_template = (
        'Given a sequence of integers: {A}\n'
        '\n'
        'Please count the number of distinct permutations of this sequence that are **lexicographically '
        'smaller** than the original sequence.\n'
        'Note: Permutations that only differ by the positions of equal elements are considered the '
        '**same**.\n'
        '\n'
        'Let {result_name} be the number of such permutations modulo {MOD}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(A=input_reference, MOD=self.MOD)
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.MOD = rng.randint(2, 100_000)

    def sample_input(self, rng):
        maximum = rng.randint(2, self.N)
        return _integer_list(rng, self.N, 1, maximum)

    def solution_generator(self, input_value):
        A, N, MOD = input_value, self.N, self.MOD
        maximum = max(A)
        phi, remaining, primes = MOD, MOD, []
        candidate = 2
        while candidate * candidate <= remaining:
            if remaining % candidate == 0:
                primes.append(candidate)
                phi = phi // candidate * (candidate - 1)
                while remaining % candidate == 0:
                    remaining //= candidate
            candidate += 1
        if remaining > 1:
            primes.append(remaining)
            phi = phi // remaining * (remaining - 1)
        tree = [0] * (maximum + 1)

        def add(value):
            while value <= maximum:
                tree[value] += 1
                value += value & -value

        def prefix(value):
            result = 0
            while value:
                result += tree[value]
                value -= value & -value
            return result

        inverse = [0] * (N + 2)
        inverse[1] = 1
        exponents = [0] * len(primes)
        coprime_product = 1
        counts = [0] * (maximum + 1)
        answer = 0
        add(A[-1])
        counts[A[-1]] += 1
        for index in range(N - 2, -1, -1):
            smaller = prefix(A[index] - 1)
            factor = N - 1 - index
            for j, prime in enumerate(primes):
                while factor % prime == 0:
                    factor //= prime
                    exponents[j] += 1
            coprime_product = coprime_product * factor % MOD
            add(A[index])
            inverse[N - index] = pow(N - index, phi - 1, MOD)
            counts[A[index]] += 1
            factor = counts[A[index]]
            for j, prime in enumerate(primes):
                while factor % prime == 0:
                    factor //= prime
                    exponents[j] -= 1
            coprime_product = coprime_product * inverse[factor] % MOD
            if smaller:
                factor = smaller
                for j, prime in enumerate(primes):
                    while factor % prime == 0:
                        factor //= prime
                        exponents[j] += 1
                coprime_product = coprime_product * factor % MOD
                current = coprime_product
                for prime, exponent in zip(primes, exponents):
                    current = current * pow(prime, exponent, MOD) % MOD
                answer = (answer + current) % MOD
                factor = smaller
                for j, prime in enumerate(primes):
                    while factor % prime == 0:
                        factor //= prime
                        exponents[j] -= 1
                coprime_product = coprime_product * inverse[factor] % MOD
        return answer


class CoinSquareGame(_RLVETask):
    prompt_template = (
        'You are given {N} coins in a row (1-indexed from left to right). The i-th coin has value C[i]: '
        '{C}\n'
        'Alice and Bob play alternately, with Alice going first. On a turn, a player removes some '
        '**positive number** of **leftmost** coins and adds the sum of their values to their own score. '
        'The game ends when no coins remain.\n'
        '\n'
        'Rules:\n'
        '- On Alice’s **first** turn, she may take either 1 or 2 coins.\n'
        '- Thereafter, if the previous player took k coins, the current player may take any number of '
        'coins from 1 to min(k * 2, the number of remaining coins).\n'
        '\n'
        'Assuming both players play optimally, what is the **maximum total value** Alice can obtain?'
        '\n'
        '\n'
        'Let {result_name} be the maximum total value Alice can obtain.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["C"] = _indexed("C", input_reference, N, 1)
        return self.prompt_template.format(**fields)

    N = 8

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, 2 * self.N)

    def solution_generator(self, input_value):
        N = self.N
        prefix = [0]
        for value in reversed(input_value):
            prefix.append(prefix[-1] + value)
        rows = [[0]]
        for i in range(1, N + 1):
            row = [0] * ((i + 1) // 2 + 1)
            for j in range(1, len(row)):
                take = 2 * j - 1
                best = row[j - 1]
                if i >= take:
                    prior = rows[i - take]
                    best = max(best, prefix[i] - prior[min(take, len(prior) - 1)])
                if i >= take + 1:
                    prior = rows[i - take - 1]
                    best = max(best, prefix[i] - prior[min(take + 1, len(prior) - 1)])
                row[j] = best
            rows.append(row)
        return rows[N][1]


class Expression_AddingParenthese_Counting(_RLVETask):
    prompt_template = (
        'Given an expression {expression}, please count the number of **distinct values** that can be '
        'obtained by inserting parentheses in the expression (but rearranging terms is NOT allowed).'
        '\n'
        '\n'
        'Let {result_name} be the number of distinct values that can be obtained.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["expression"] = " ".join(
            input_reference + f"[{i // 2}]" if i % 2 == 0 else self.operations[i // 2]
            for i in range(2 * N - 1)
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.operations = [rng.choice(("+", "-", "*")) for _ in range(self.N - 1)]

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N**2)

    def solution_generator(self, input_value):
        memo = {}

        def values(left, right):
            if left == right:
                return {input_value[left]}
            if (left, right) not in memo:
                result = set()
                for split in range(left, right):
                    for a in values(left, split):
                        for b in values(split + 1, right):
                            operation = self.operations[split]
                            result.add(a + b if operation == "+" else a - b if operation == "-" else a * b)
                memo[left, right] = result
            return memo[left, right]

        return len(values(0, self.N - 1))


class FractionalProgramming(_RLVETask):
    prompt_template = (
        'You are given two arrays `A` and `B`, each containing {N} integers:\n'
        '{A_and_B}\n'
        '\n'
        'Please select {K} **distinct indices** `i_1, ..., i_{K}` to maximize the value of ratio '
        '`(A[i_1] + ... + A[i_{K}]) / (B[i_1] + ... + B[i_{K}])`\n'
        '\n'
        'The maximum possible value of this ratio is a rational number; write it in lowest terms '
        'as `p / q`. Let {result_name} be `p * q^-1 mod 998244353`, where `q^-1` is the modular '
        'multiplicative inverse of `q` modulo the prime 998244353 - that is, the unique integer '
        '`y` with `0 <= y < 998244353` and `q * y = 1 (mod 998244353)`.\n'
        'For example, if the ratio were `5 / 2`, the answer would be `499122179`, since '
        '`499122179 * 2 = 5 (mod 998244353)`.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            K=self.K,
            A_and_B="\n".join(
                f"A[{i}]={self.A[i]} B[{i}]={input_reference}[{i}]" for i in range(N)
            ),
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.K = rng.randint(2, self.N - 1)
        initial_b = _integer_list(rng, self.N, 1, self.N)
        self.A = [rng.randint(1, 2 * value) for value in initial_b]

    def sample_input(self, rng):
        return [rng.randint(max(1, (value + 1) // 2), self.N) for value in self.A]

    def solution_generator(self, input_value):
        low, high, selected = 0.0, max(a / b for a, b in zip(self.A, input_value)), None
        for _ in range(256):
            middle = (low + high) / 2
            indices = sorted(
                range(self.N),
                key=lambda index: self.A[index] - middle * input_value[index],
                reverse=True,
            )
            if sum(self.A[index] - middle * input_value[index] for index in indices[: self.K]) >= 0:
                low, selected = middle, indices[: self.K]
            else:
                high = middle
        return _ratio_modulo_prime(
            sum(self.A[index] for index in selected),
            sum(input_value[index] for index in selected),
        )


class FractionalProgramming_BipartiteGraphMatching(_RLVETask):
    prompt_template = (
        'You are given two matrices A and B of size {N} × {N} (0-indexed).\n'
        '\n'
        'Matrix A is (given in **row-major order**, with each row represented as a list of integers '
        'separated by spaces):\n'
        '{A}\n'
        '\n'
        'Matrix B is (given in **row-major order**, with each row represented as a list of integers '
        'separated by spaces):\n'
        '{B}\n'
        '\n'
        'Please find a permutation P of indices from 0 to {N_minus_1}, i.e., P[0], P[1], ..., '
        'P[{N_minus_1}], such that the following ratio is maximized: (A[0][P[0]] + A[1][P[1]] + '
        '... + A[{N_minus_1}][P[{N_minus_1}]]) / (B[0][P[0]] + B[1][P[1]] + ... + '
        'B[{N_minus_1}][P[{N_minus_1}]])\n'
        '\n'
        'The maximum possible value of this ratio is a rational number; write it in lowest terms '
        'as `p / q`. Let {result_name} be `p * q^-1 mod 998244353`, where `q^-1` is the modular '
        'multiplicative inverse of `q` modulo the prime 998244353 - that is, the unique integer '
        '`y` with `0 <= y < 998244353` and `q * y = 1 (mod 998244353)`.\n'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            A="\n".join(" ".join(map(str, row)) for row in self.A),
            B="\n".join(" ".join(f"{input_reference}[{i}][{j}]" for j in range(N)) for i in range(N)),
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        initial_b = [_integer_list(rng, self.N, 1, self.N) for _ in range(self.N)]
        self.A = [
            [rng.randint(1, 2 * value) for value in row]
            for row in initial_b
        ]

    def sample_input(self, rng):
        return [
            [rng.randint(max(1, (self.A[i][j] + 1) // 2), self.N) for j in range(self.N)]
            for i in range(self.N)
        ]

    def solution_generator(self, input_value):
        best_numerator, best_denominator = 0, 1
        for assignment in permutations(range(self.N)):
            numerator = sum(self.A[i][assignment[i]] for i in range(self.N))
            denominator = sum(input_value[i][assignment[i]] for i in range(self.N))
            if numerator * best_denominator > best_numerator * denominator:
                best_numerator, best_denominator = numerator, denominator
        return _ratio_modulo_prime(best_numerator, best_denominator)


class GradeRankingCounting(_RLVETask):
    prompt_template = (
        'Count the number of matrices A of size {N} × {M} (0-indexed) that satisfy the following '
        'conditions:\n'
        '1. Each element A[i][j] (0 ≤ i < {N}, 0 ≤ j < {M}) is an integer in the range [1, U[j]]. U is: '
        '{U}\n'
        '2. For each column j (0 ≤ j < {M}), there are exactly R[j] rows i (1 ≤ i < {N}) such that A[i][j] '
        '> A[0][j]. R is: {R}\n'
        '3. There are exactly {K} rows i (1 ≤ i < {N}) such that A[0][j] ≥ A[i][j] holds for **all** j (0 '
        '≤ j < {M}).\n'
        '\n'
        'Let {result_name} be the number of such matrices modulo {MOD}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            M=self.M,
            MOD=self.MOD,
            K=self.K,
            U=_indexed("U", input_reference, self.M),
            R=" ".join(f"R[{j}]={self.R[j]}" for j in range(self.M)),
        )
        return self.prompt_template.format(**fields)

    MODS = (1_000_000_007, 998_244_353)

    def _setup(self, rng):
        self.N, self.M = rng.randint(2, 5), rng.randint(2, 5)
        self.MOD = rng.choice(self.MODS)
        losers = set(rng.sample(range(1, self.N), rng.randint(0, self.N - 1)))
        initial_u = _integer_list(rng, self.M, 1, self.N)
        matrix = [[0] * self.M for _ in range(self.N)]
        self.R = [0] * self.M
        for column in range(self.M):
            matrix[0][column] = rng.randint(1, initial_u[column])
            for row in range(1, self.N):
                maximum = matrix[0][column] if row in losers else initial_u[column]
                matrix[row][column] = rng.randint(1, maximum)
                self.R[column] += matrix[row][column] > matrix[0][column]
        self.K = sum(
            all(matrix[0][column] >= matrix[row][column] for column in range(self.M))
            for row in range(1, self.N)
        )

    def sample_input(self, rng):
        return _integer_list(rng, self.M, 1, self.N)

    def solution_generator(self, input_value):
        N, K, MOD = self.N, self.K, self.MOD
        limit = 2 * N + 2
        factorial = [1] * (limit + 1)
        for i in range(1, limit + 1):
            factorial[i] = factorial[i - 1] * i % MOD
        inverse_factorial = [1] * (limit + 1)
        inverse_factorial[limit] = pow(factorial[limit], MOD - 2, MOD)
        for i in range(limit, 0, -1):
            inverse_factorial[i - 1] = inverse_factorial[i] * i % MOD

        def choose(n, k):
            if k < 0 or k > n:
                return 0
            return factorial[n] * inverse_factorial[k] % MOD * inverse_factorial[n - k] % MOD

        def power_sum(power, upper):
            return sum(pow(value, power, MOD) for value in range(1, upper + 1)) % MOD

        def course_contribution(upper, above):
            below = N - 1 - above
            result = 0
            for j in range(above + 1):
                coefficient = choose(above, j)
                if j & 1:
                    coefficient = -coefficient
                result += coefficient * pow(upper, above - j, MOD) * power_sum(below + j, upper)
            return result % MOD

        above_counts = self.R
        numeric = math.prod(
            course_contribution(upper, above)
            for upper, above in zip(input_value, above_counts)
        ) % MOD
        non_dominated = N - 1 - K
        patterns = 0
        for omitted in range(non_dominated + 1):
            product = math.prod(
                choose(non_dominated - omitted, above) for above in above_counts
            ) % MOD
            term = choose(non_dominated, omitted) * product
            patterns += -term if omitted & 1 else term
        patterns = patterns * choose(N - 1, K) % MOD
        return numeric * patterns % MOD


class GridColoringCounting(_RLVETask):
    prompt_template = (
        'You are given a grid of size {N} × {M}. You may color some cells (and leave others uncolored) '
        'using {C} colors labeled from 0 to {C_minus_1}, such that:\n'
        '1. No two different colors appear in the same row or the same column.\n'
        '2. Color `i` is used exactly X[i] times. The array X is given as: {Xs}\n'
        '\n'
        'Let {result_name} be the number of valid colorings modulo {MOD}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            M=self.M,
            C=self.C,
            C_minus_1=self.C - 1,
            Xs=" ".join(
                f"X[{i}]={input_reference} // {self.C} + "
                f"(1 if {i} < {input_reference} % {self.C} else 0)"
                for i in range(self.C)
            ),
            MOD=self.MOD,
        )
        return self.prompt_template.format(**fields)

    def _setup(self, rng):
        while True:
            self.N, self.M = rng.randint(2, 6), rng.randint(2, 6)
            initial_sum = rng.randint(1, self.N * self.M)
            self.C = rng.randint(1, min(self.N, self.M, initial_sum))
            self.MOD = rng.randint(2, 10_000)
            if any(
                self.solution_generator(value) > 0
                for value in range(self.C, self.N * self.M + 1)
            ):
                break

    def sample_input(self, rng):
        while True:
            value = rng.randint(self.C, self.N * self.M)
            if self.solution_generator(value) > 0:
                return value

    def solution_generator(self, input_value):
        N, M, C, MOD = self.N, self.M, self.C, self.MOD
        quotient, remainder = divmod(input_value, C)
        counts = [quotient + (color < remainder) for color in range(C)]
        total = N * M
        choose = [[0] * (total + 1) for _ in range(total + 1)]
        for i in range(total + 1):
            choose[i][0] = 1
            for j in range(1, i + 1):
                choose[i][j] = (choose[i - 1][j] + choose[i - 1][j - 1]) % MOD
        ways = [[[0] * (C + 1) for _ in range(M + 1)] for _ in range(N + 1)]
        ways[0][0][0] = 1
        for color in range(1, C + 1):
            amount = counts[color - 1]
            exact = [[0] * (M + 1) for _ in range(N + 1)]
            for rows in range(1, N + 1):
                for columns in range(1, M + 1):
                    if rows * columns < amount:
                        continue
                    value = choose[rows * columns][amount]
                    for smaller_rows in range(1, rows + 1):
                        for smaller_columns in range(1, columns + 1):
                            if smaller_rows < rows or smaller_columns < columns:
                                value -= (
                                    exact[smaller_rows][smaller_columns]
                                    * choose[rows][smaller_rows]
                                    * choose[columns][smaller_columns]
                                )
                    exact[rows][columns] = value % MOD
            for rows in range(1, N + 1):
                for columns in range(1, M + 1):
                    for prior_rows in range(rows):
                        for prior_columns in range(columns):
                            new_rows, new_columns = rows - prior_rows, columns - prior_columns
                            if new_rows * new_columns >= amount:
                                ways[rows][columns][color] += (
                                    ways[prior_rows][prior_columns][color - 1]
                                    * exact[new_rows][new_columns]
                                    * choose[N - prior_rows][new_rows]
                                    * choose[M - prior_columns][new_columns]
                                )
                                ways[rows][columns][color] %= MOD
        return sum(ways[rows][columns][C] for rows in range(1, N + 1) for columns in range(1, M + 1)) % MOD


class KingSorting(_RLVETask):
    prompt_template = (
        'You are given `{N} + 1 = {N_plus_1}` pairs of integers: `(A[0], B[0])`, `(a[1], b[1])`, `(a[2], '
        'b[2])`, ..., `(a[{N}], b[{N}])`\n'
        '{values}\n'
        '\n'
        'Your task is to **rearrange the {N} pairs** `(a[i], b[i])` for `1 ≤ i ≤ {N}` in some order (there '
        'are `{N}!` possible permutations). After rearrangement, define the new sequence of `{N_plus_1}` '
        'pairs as: `(A[0], B[0])`, `(A[1], B[1])`, ..., `(A[{N}], B[{N}])`, where `(A[i], B[i])` comes '
        'from the chosen permutation for `i ≥ 1`.\n'
        '\n'
        'Your goal is to **minimize** the following value: `max ( A[0] * A[1] * ... * A[i - 1] // B[i] | 1 '
        '≤ i ≤ {N} )` (Note: `//` means **integer division**, i.e., rounded down just like in Python).\n'
        'That is, for each `i` from `1` to `{N}`, compute the product of all previous `A` values (`A[0]` '
        'to `A[i - 1]`) divided by `B[i]`, take the maximum of these, and find a permutation that '
        'minimizes this maximum.\n'
        '\n'
        'Let {result_name} be the minimum possible value of that maximum.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            values="\n".join(
                [f"(A[0], B[0]) = ({input_reference}[0], {self.B[0]})"]
                + [
                    f"(a[{i}], b[{i}]) = ({input_reference}[{i}], {self.B[i]})"
                    for i in range(1, N + 1)
                ]
            ),
            REVERSE_INDICES=" ".join(map(str, range(N, 0, -1))),
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.B = _integer_list(rng, self.N + 1, 1, 10)

    def sample_input(self, rng):
        return _integer_list(rng, self.N + 1, 1, 10)

    def solution_generator(self, input_value):
        order = sorted(range(1, self.N + 1), key=lambda index: input_value[index] * self.B[index])
        product, answer = input_value[0], 0
        for index in order:
            answer = max(answer, product // self.B[index])
            product *= input_value[index]
        return answer


class Knapsack(_RLVETask):
    prompt_template = (
        'You are given {N} items labeled from `0` to `{N_minus_1}`. Each item has a **weight** W[i] and a '
        '**value** V[i]:\n'
        '{W_and_V}\n'
        '\n'
        'Please select a subset of **distinct items** i_1, i_2, ..., i_k such that:\n'
        '- The total weight W[i_1] + W[i_2] + ... + W[i_k] is **less than or equal to** {W_max}, and\n'
        '- Try your best to maximize the total value V[i_1] + V[i_2] + ... + V[i_k].\n'
        '\n'
        'Let {result_name} be the maximum possible total value.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            W_and_V="\n".join(
                f"W[{i}]={input_reference}[{i}] V[{i}]={self.V[i]}" for i in range(N)
            ),
            W_max=self.capacity,
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        initial_w = _integer_list(rng, self.N, 1, self.N)
        self.V = [rng.randint(1, 2 * weight) for weight in initial_w]
        self.capacity = rng.randint(min(initial_w), sum(initial_w))

    def sample_input(self, rng):
        while True:
            weights = [rng.randint(max(1, (value + 1) // 2), self.N) for value in self.V]
            if min(weights) <= self.capacity <= sum(weights):
                return weights

    def solution_generator(self, input_value):
        best = [0] * (self.capacity + 1)
        for weight, value in zip(input_value, self.V):
            for capacity in range(self.capacity, weight - 1, -1):
                best[capacity] = max(best[capacity], best[capacity - weight] + value)
        return best[self.capacity]


class LandAcquisition(_RLVETask):
    prompt_template = (
        'There are {N} items, and the i-th item has two attributes W[i] and L[i]. The arrays W and L are '
        'given as follows:\n'
        '{W_and_L}\n'
        '\n'
        'Partition all items into an arbitrary number of **disjoint non-empty sets**. For each set S, its '
        'cost is defined as: cost(S) = max(W[i] for i ∈ S) × max(L[i] for i ∈ S)\n'
        'Can you make the total cost, which is the sum of costs of all sets, as small as '
        'possible?\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["W_and_L"] = "\n".join(
            f"W[{i}]={input_reference}[{i - 1}] L[{i}]={self.L[i - 1]}"
            for i in range(1, N + 1)
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        while True:
            widths = _integer_list(rng, self.N, 1, self.N**2)
            self.L = _integer_list(rng, self.N, 1, self.N**2)
            if self._cost(widths) < self._baseline(widths):
                break

    def sample_input(self, rng):
        while True:
            widths = _integer_list(rng, self.N, 1, self.N**2)
            if self._cost(widths) < self._baseline(widths):
                return widths

    def _filtered(self, widths):
        stack = []
        for width, length in sorted(zip(widths, self.L)):
            while stack and length > stack[-1][1]:
                stack.pop()
            stack.append((width, length))
        return stack

    def _cost(self, widths):
        lands = self._filtered(widths)
        best = [0] + [math.inf] * len(lands)
        for i in range(1, len(lands) + 1):
            best[i] = min(best[j] + lands[i - 1][0] * lands[j][1] for j in range(i))
        return best[-1]

    def _baseline(self, widths):
        indices = sorted(range(self.N), key=lambda i: (widths[i], self.L[i]))
        answer = min(max(widths) * max(self.L), sum(w * length for w, length in zip(widths, self.L)))
        for split in range(1, self.N):
            left, right = indices[:split], indices[split:]
            answer = min(
                answer,
                max(widths[i] for i in left) * max(self.L[i] for i in left)
                + max(widths[i] for i in right) * max(self.L[i] for i in right),
            )
        return answer

    def solution_generator(self, input_value):
        return self._cost(input_value)


# class MaxMultSplit(_RLVETask):
#     prompt_template = (
#         'You are given a string of digits `S` of length {N}:\n'
#         '{string}\n'
#         '\n'
#         'Your task is to divide this string into exactly {K_plus_1} non-empty, non-overlapping parts (from '
#         'left to right, maintaining original order), such that the **product** of the resulting integer '
#         'values is **maximized**.\n'
#         '\n'
#         'Specifically, split the string into substrings: s_1, ..., s_{K_plus_1}, where:\n'
#         '- Each part s_i is a contiguous non-empty substring of `S`,\n'
#         '- The concatenation s_1 + ... + s_{K_plus_1} = S (here + means string concatenation),\n'
#         '- The value `int(s_1) * ... * int(s_{K_plus_1})` is as large as possible.\n'
#         '\n'
#         'Output Format:\n'
#         'Your final answer should be a single line containing the {K_plus_1} parts, separated by '
#         '**spaces**.\n'
#         'Example: `31 2` (do **NOT** include the backticks or quotes); this means the string "312" is '
#         'split into two parts: "31" and "2".\n'
#     )

#     def _format_prompt(self, input_reference, result_name):
#         N = getattr(self, "N", None)
#         fields = self._base_prompt_fields(result_name)
#         fields.update(K_plus_1=self.K + 1, string=input_reference)
#         return self.prompt_template.format(**fields)

#     N = 6
#     K = 5

#     def sample_input(self, rng):
#         return _integer_list(rng, self.N, 1, 9)

#     def solution_generator(self, input_value):
# #         digits = "".join(str(value) for value in input_value)
#         best = [[0] * self.N for _ in range(self.K + 1)]
#         for cuts in range(self.K + 1):
#             for end in range(self.N):
#                 if cuts == 0:
#                     best[0][end] = int(digits[: end + 1])
#                 else:
#                     for split in range(1, end + 1):
#                         best[cuts][end] = max(
#                             best[cuts][end],
#                             int(digits[split : end + 1]) * best[cuts - 1][split - 1],
#                         )
#         return best[self.K][self.N - 1]


class Max_TreeConstrainedPermutation_Weight(_RLVETask):
    prompt_template = (
        'You are given an array W of length {N}: {W}\n'
        '\n'
        'Please find a permutation P of 1 to {N} such that the following conditions are satisfied:\n'
        '{conditions}\n'
        '\n'
        'Try your best to **maximize** the sum of W[P[i]] × i for all i from 1 to {N}.\n'
        '\n'
        'Let {result_name} be the maximum possible value of the sum of W[P[i]] × i for all i from '
        '1 to {N}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            W=_indexed("W", input_reference, N, 1),
            conditions="\n".join(
                f"- The element {i} has no constraint."
                if parent == 0
                else f"- The element {parent} must come before element {i}."
                for i, parent in enumerate(self.parents, start=1)
            ),
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.parents = [rng.randint(0, index) for index in range(self.N)]

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        parent = [0] + self.parents
        weights = [0] + input_value
        representative = list(range(self.N + 1))
        sizes = [1] * (self.N + 1)

        class Entry:
            def __init__(self, node):
                self.node = node
                self.size = sizes[node]
                self.weight = weights[node]

            def __lt__(self, other):
                return self.weight * other.size < other.weight * self.size

        def find(node):
            while representative[node] != node:
                representative[node] = representative[representative[node]]
                node = representative[node]
            return node

        heap = [Entry(node) for node in range(1, self.N + 1)]
        heapq.heapify(heap)
        answer = 0
        while heap:
            entry = heapq.heappop(heap)
            node = find(entry.node)
            if sizes[node] != entry.size:
                continue
            target = find(parent[node])
            answer += weights[node] * sizes[target]
            weights[target] += weights[node]
            sizes[target] += sizes[node]
            representative[node] = target
            if target:
                heapq.heappush(heap, Entry(target))
        return answer


class MaximumLexicographicalOrderSubsequence(_RLVETask):
    prompt_template = (
        'Given an array A of length {N}: {A}\n'
        '\n'
        'Please find a (not necessarily contiguous) subsequence of length {K} (i.e., select {K} elements '
        'with increasing indices: 0 <= i1 < ... < i{K} < {N}) such that the resulting subsequence A[i1], '
        '..., A[i{K}] is **lexicographically maximal**.\n'
        '\n'
        'Let {result_name} be a single list with the selected subsequence A[i1], ..., A[i{K}] in '
        'order.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(K=self.K, A=_indexed("A", input_reference, N))
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.K = self.N - 1

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        candidates = deque()
        answer = []
        for index, value in enumerate(input_value):
            while candidates and candidates[-1] < value:
                candidates.pop()
            if len(candidates) < self.K:
                candidates.append(value)
            if index >= self.N - self.K:
                answer.append(candidates.popleft())
        return answer


class MinPairSumMultiplicationPermutation(_RLVETask):
    prompt_template = (
        'You are given two matrices `A` and `B`, each of size {N} × {N}:\n'
        '{matrix_A}\n'
        '{matrix_B}\n'
        '\n'
        'You need to find a permutation P of indices from 0 to {N_minus_1} such that the value (sum of '
        'A[0][P[0]], A[1][P[1]], ..., A[{N_minus_1}][P[{N_minus_1}]]) multiplied by (sum of B[0][P[0]], '
        'B[1][P[1]], ..., B[{N_minus_1}][P[{N_minus_1}]]) is minimized.\n'
        '\n'
        'Let {result_name} be the minimum possible value of that product.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            matrix_A=_matrix_rows("A", input_reference, N),
            matrix_B="\n".join(
                " ".join(f"B[{i}][{j}]={self.B[i][j]}" for j in range(N))
                for i in range(N)
            ),
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.B = [_integer_list(rng, self.N, 1, self.N) for _ in range(self.N)]

    def sample_input(self, rng):
        return [_integer_list(rng, self.N, 1, self.N) for _ in range(self.N)]

    def solution_generator(self, input_value):
        answer = None
        for assignment in permutations(range(self.N)):
            sum_a = sum(input_value[row][assignment[row]] for row in range(self.N))
            sum_b = sum(self.B[row][assignment[row]] for row in range(self.N))
            value = sum_a * sum_b
            answer = value if answer is None else min(answer, value)
        return answer


class MinSumChebyshevDistance(_RLVETask):
    prompt_template = (
        'You are given three arrays X, Y, and T, each of length {N}:\n'
        '{X_Y_T}\n'
        '\n'
        'Please find an integer point (x, y) such that the following sum is minimized: sum over 0 <= i < '
        '{N} of max(|x - X[i]|, |y - Y[i]|) * T[i].\n'
        '\n'
        'Let {result_name} be the minimum possible value of that sum.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["X_Y_T"] = "\n".join(
            f"X[{i}]={input_reference}[{i}] Y[{i}]={self.Y[i]} T[{i}]={self.T[i]}"
            for i in range(N)
        )
        return self.prompt_template.format(**fields)

    N = 5

    def _setup(self, rng):
        self.Y = _integer_list(rng, self.N, 1, 2 * self.N)
        self.T = _integer_list(rng, self.N, 1, self.N)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, 2 * self.N)

    def solution_generator(self, input_value):
        rotated_x = sorted([[x + y, weight] for x, y, weight in zip(input_value, self.Y, self.T)])
        rotated_y = sorted([[x - y, weight] for x, y, weight in zip(input_value, self.Y, self.T)])

        def weighted_median(values):
            left, right = 0, len(values) - 1
            while left < right:
                if values[left][1] < values[right][1]:
                    values[right][1] -= values[left][1]
                    left += 1
                elif values[left][1] > values[right][1]:
                    values[left][1] -= values[right][1]
                    right -= 1
                else:
                    left += 1
                    right -= 1
            return values[left][0]

        horizontal = weighted_median(rotated_x)
        vertical = weighted_median(rotated_y)
        x, y = int((horizontal + vertical) / 2), int((horizontal - vertical) / 2)
        return min(
            sum(
                max(abs(candidate_x - point_x), abs(candidate_y - point_y)) * weight
                for point_x, point_y, weight in zip(input_value, self.Y, self.T)
            )
            for candidate_x, candidate_y in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))
        )


# class MinSumPreXor(_RLVETask):
#     prompt_template = (
#         'You are given an array P of length {N}: {P}\n'
#         'Replace every entry P[i] that equals -1 (for 1 ≤ i ≤ {N}) with a **non-negative integer** (all '
#         'other entries are fixed non-negative integers), so as to **minimize** the sum: B[1] + B[2] + ... '
#         '+ B[{N}], where B[1] = P[1] and for i ≥ 2, B[i] = B[i−1] XOR P[i] (XOR is the bitwise exclusive '
#         'OR). Output the updated array P as {N} space-separated non-negative integers in one line.'
#     )

#     def _format_prompt(self, input_reference, result_name):
#         N = getattr(self, "N", None)
#         fields = self._base_prompt_fields(result_name)
#         fields["P"] = " ".join(
#             f"P[{i + 1}]={-1 if i in self.removed else input_reference + f'[{i}]'}"
#             for i in range(N)
#         )
#         return self.prompt_template.format(**fields)

#     N = 6

#     def _setup(self, rng):
#         self.removed = set(rng.sample(range(self.N), rng.randint(1, self.N - 1)))

#     def sample_input(self, rng):
#         return _integer_list(rng, self.N, 0, 2 * self.N)

#     def solution_generator(self, input_value):
# #         known = [(index + 1, value) for index, value in enumerate(input_value) if index not in self.removed]
#         bit_count = max(1, max(value for _, value in known).bit_length())
#         blocks, lengths, starts = [], [], []
#         running = 0
#         for index, (position, value) in enumerate(known):
#             if index == 0 or position != known[index - 1][0] + 1:
#                 blocks.append([0] * bit_count)
#                 lengths.append(0)
#                 starts.append(position)
#                 running = 0
#             running ^= value
#             for bit in range(bit_count):
#                 blocks[-1][bit] += (running >> bit) & 1
#             lengths[-1] += 1
#         answer = 0
#         for block_index, counts in enumerate(blocks):
#             starts_at_one = starts[block_index] == 1
#             for bit, ones in enumerate(counts):
#                 contribution = ones if starts_at_one else min(ones, lengths[block_index] - ones + 1)
#                 answer += contribution << bit
#         return answer


def _minimum_cost_flow(node_count, edges, demands):
    source, sink = node_count, node_count + 1
    graph = [[] for _ in range(node_count + 2)]

    def add_edge(start, end, capacity, cost):
        graph[start].append([end, capacity, cost, len(graph[end])])
        graph[end].append([start, 0, -cost, len(graph[start]) - 1])

    for start, end, capacity, cost in edges:
        add_edge(start, end, capacity, cost)
    required = 0
    for node, demand in enumerate(demands):
        if demand < 0:
            add_edge(source, node, -demand, 0)
        elif demand > 0:
            add_edge(node, sink, demand, 0)
            required += demand
    potential = [0] * len(graph)
    total_flow = total_cost = 0
    while total_flow < required:
        distance = [math.inf] * len(graph)
        parent = [None] * len(graph)
        distance[source] = 0
        queue = [(0, source)]
        while queue:
            current, node = heapq.heappop(queue)
            if current != distance[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                target, capacity, cost, _ = edge
                candidate = current + cost + potential[node] - potential[target]
                if capacity and candidate < distance[target]:
                    distance[target] = candidate
                    parent[target] = (node, edge_index)
                    heapq.heappush(queue, (candidate, target))
        if parent[sink] is None:
            raise ValueError("infeasible interval coverage instance")
        for node, value in enumerate(distance):
            if value < math.inf:
                potential[node] += value
        amount = required - total_flow
        node = sink
        while node != source:
            previous, edge_index = parent[node]
            amount = min(amount, graph[previous][edge_index][1])
            node = previous
        node = sink
        while node != source:
            previous, edge_index = parent[node]
            edge = graph[previous][edge_index]
            total_cost += amount * edge[2]
            edge[1] -= amount
            graph[node][edge[3]][1] += amount
            node = previous
        total_flow += amount
    return total_cost


class MinimumIntervalCoverage(_RLVETask):
    prompt_template = (
        'You are given {M} intervals within [1, {N}]. Each interval is defined as [L[i], R[i]] with an '
        'associated cost C[i]. The intervals are provided as:\n'
        '{intervals}\n'
        '\n'
        'You can select each interval any number of times (including 0). For each point i in [1, {N}], you '
        'must ensure it is covered by at least NEED[i] selected intervals, where the array NEED is given '
        'as:\n'
        '{NEED}\n'
        '\n'
        'Your goal is to minimize the **total cost** of the selected intervals while satisfying the above '
        'condition.\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            M=self.M,
            intervals="\n".join(
                f"L[{i}]={left} R[{i}]={right} C[{i}]={input_reference}[{i - 1}]"
                for i, (left, right) in enumerate(self.intervals, start=1)
            ),
            NEED=" ".join(f"NEED[{i}]={need}" for i, need in enumerate(self.needs, start=1)),
        )
        return self.prompt_template.format(**fields)

    N = 6
    M = 8

    def _setup(self, rng):
        self.intervals = []
        for _ in range(self.M):
            left, right = sorted((rng.randint(1, self.N), rng.randint(1, self.N)))
            rng.randint(1, 3 * (right - left + 1))
            self.intervals.append((left, right))
        self.needs = [
            rng.randint(0, self.N) if any(left <= point <= right for left, right in self.intervals) else 0
            for point in range(1, self.N + 1)
        ]

    def sample_input(self, rng):
        return [rng.randint(1, 3 * (right - left + 1)) for left, right in self.intervals]

    def solution_generator(self, input_value):
        padded = [0] + self.needs + [0]
        demands = [padded[i + 1] - padded[i] for i in range(self.N + 1)]
        capacity = sum(self.needs)
        edges = [(i, i + 1, capacity, 0) for i in range(self.N)]
        edges.extend(
            (right, left - 1, capacity, cost)
            for (left, right), cost in zip(self.intervals, input_value)
        )
        return _minimum_cost_flow(self.N + 1, edges, demands)


class MinimumTreeWeightedDominatingAncestor(_RLVETask):
    prompt_template = (
        'You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} + 1 = '
        '{N_plus_1} vertices, labeled from `0` to `{N}`.\n'
        '\n'
        '`0` is the root of the tree. Each non-root vertex has a parent, and the edge connecting it with '
        'its parent has a weight. The edges are given as follows:\n'
        '{edges}\n'
        '\n'
        'Each non-root vertex also has a cost, given as a list `C` of length {N}, where `C[i]` is the cost '
        'of vertex `i`:\n'
        '{C}\n'
        '\n'
        'The root (vertex `0`) is already selected. Your task is to select exactly {K} **additional** '
        'non-root vertices. The total cost of the selection is defined as follows:\n'
        '- For every vertex `u`, let `D[u]` be the distance from `u` to its **nearest selected ancestor**, '
        'where a selected ancestor includes `0` or the vertex itself (if selected). The **distance** '
        'between two vertices is the sum of weights along the unique path between them.\n'
        '- The cost contributed by vertex `u` is `C[u] × D[u]` for all non-root verticies `u`.\n'
        '- Try your best to **minimize** the total cost.\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            edges="\n".join(
                f"`{i}`'s parent is `{parent}` with weight `{weight}`"
                for i, (parent, weight) in enumerate(self.parents[1:], start=1)
            ),
            C=" ".join(f"C[{i}]={input_reference}[{i - 1}]" for i in range(1, N + 1)),
            K=self.K,
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        order = list(range(1, self.N + 1))
        rng.shuffle(order)
        order = [0] + order
        self.parents = [None] * (self.N + 1)
        for index in range(1, self.N + 1):
            self.parents[order[index]] = (rng.choice(order[:index]), rng.randint(1, 10))
        self.K = rng.randint(1, self.N - 1)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 0, 10)

    def solution_generator(self, input_value):
        costs = [0] + input_value
        answer = None
        for picked in combinations(range(1, self.N + 1), self.K):
            selected = {0, *picked}
            total = 0
            for node in range(1, self.N + 1):
                current, distance = node, 0
                while current not in selected:
                    current, weight = self.parents[current]
                    distance += weight
                total += costs[node] * distance
            answer = total if answer is None else min(answer, total)
        return answer


class Minimum_DominatingSet(_RLVETask):
    prompt_template = (
        'You are given an **undirected graph** with {N} vertices labeled from 0 to {N_minus_1}. The graph '
        'contains the following undirected edges:\n'
        '{edges}\n'
        '\n'
        'Each vertex has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex i:\n'
        '{C}\n'
        '\n'
        'Your task is to select a set of distinct vertices x_1, x_2, ..., x_k (you determine k), such that '
        'every vertex is either selected or has at least one selected neighbor.\n'
        'Try your best to minimize the total cost: C[x_1] + C[x_2] + ... + C[x_k].\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            edges="\n".join(f"({u}, {v})" for u, v in self.edges),
            C="\n".join(f"C[{i}]={input_reference}[{i}]" for i in range(N)),
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        self.edges = rng.sample(
            [(u, v) for u in range(self.N) for v in range(u + 1, self.N)], 18
        )
        self.covering = [1 << node for node in range(self.N)]
        for u, v in self.edges:
            self.covering[u] |= 1 << v
            self.covering[v] |= 1 << u

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, 10)

    def solution_generator(self, input_value):
        full = (1 << self.N) - 1
        answer = sum(input_value)
        for selected in range(1 << self.N):
            covered = 0
            cost = 0
            for node in range(self.N):
                if selected >> node & 1:
                    covered |= self.covering[node]
                    cost += input_value[node]
            if covered == full:
                answer = min(answer, cost)
        return answer


class Minimum_VertexCover(_RLVETask):
    prompt_template = (
        'You are given an **undirected graph** with {N} vertices labeled from 0 to {N_minus_1}. The graph '
        'contains the following undirected edges:\n'
        '{edges}\n'
        '\n'
        'Each vertex has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex i:\n'
        '{C}\n'
        '\n'
        'Your task is to select a set of distinct vertices x_1, x_2, ..., x_k (you determine k), such that '
        'every edge in the graph has at least one endpoint in the selected set; that is, for every edge '
        '(u, v), at least one of u or v must be included.\n'
        'Try your best to minimize the total cost: C[x_1] + C[x_2] + ... + C[x_k].\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            edges="\n".join(f"({u}, {v})" for u, v in self.edges),
            C="\n".join(f"C[{i}]={input_reference}[{i}]" for i in range(N)),
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        self.edges = rng.sample(
            [(u, v) for u in range(self.N) for v in range(u + 1, self.N)], 18
        )

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, 10)

    def solution_generator(self, input_value):
        answer = sum(input_value)
        for selected in range(1 << self.N):
            if all((selected >> u & 1) or (selected >> v & 1) for u, v in self.edges):
                answer = min(
                    answer,
                    sum(input_value[node] for node in range(self.N) if selected >> node & 1),
                )
        return answer


class NegativeBase(_RLVETask):
    prompt_template = (
        'We can represent integers using a **negative base** system with base `-R`, where `R` is a '
        'positive integer greater than 1. In this system, the digits used are from `0` to `R - 1` (in '
        'decimal).\n'
        'For example, the decimal number `-15` can be represented as `110001` in base `-2`, since:\n'
        '1×(-2)^5 + 1×(-2)^4 + 0×(-2)^3 + 0×(-2)^2 + 0×(-2)^1 + 1×(-2)^0 = (-15).\n'
        '\n'
        'Your task is to convert the decimal number `{N}` into base `-{R}`, and determine its digits (in '
        'decimal) from most significant to least significant.\n'
        '\n'
        'Let {result_name} be a single list with those digits (in decimal) from most significant '
        'to least significant.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(N=input_reference, R=self.R, R_minus_1=self.R - 1)
        return self.prompt_template.format(**fields)

    LOW = 183
    HIGH = 1640

    def _setup(self, rng):
        self.R = 3

    def sample_input(self, rng):
        return rng.randint(self.LOW, self.HIGH)

    def solution_generator(self, input_value):
        value, base = input_value, -self.R
        digits = []
        while value:
            value, remainder = divmod(value, base)
            if remainder < 0:
                value += 1
                remainder += self.R
            digits.append(remainder)
        return list(reversed(digits))


class RandomRangeMaxExpectation(_RLVETask):
    prompt_template = (
        'You are given an array of {N} integers: {array}\n'
        '\n'
        'You will perform {Q} operations in order. In each operation, you uniformly select a subarray (a '
        'contiguous segment of the array) at random from all {N} × ({N} + 1) / 2 possible subarrays. Then, '
        'all elements in that subarray are changed to the **maximum** value within it.\n'
        '\n'
        'Please compute the expected value of each position in the array after all {Q} operations. Since '
        'the expected value is a rational number with denominator ({N} × ({N} + 1) / 2)^{Q}, output the '
        '**numerator** (i.e., the expected value multiplied by ({N} × ({N} + 1) / 2)^{Q}), modulo {MOD}.\n'
        '\n'
        'Let {result_name} be a single list with the scaled expected values (modulo {MOD}) for '
        'each position, in order.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(array=input_reference, Q=self.Q, MOD=self.MOD)
        return self.prompt_template.format(**fields)

    N = 6
    MOD = 10_000

    def _setup(self, rng):
        self.Q = rng.randint(1, self.N)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 0, self.N)

    def solution_generator(self, input_value):
        A, N, MOD = input_value, self.N, self.MOD

        def triangular(value):
            return value * (value + 1) // 2 % MOD

        infinity = max(A) + 1
        previous = [[0] * N for _ in range(N)]
        weights = [[0] * N for _ in range(N)]
        for left in range(N):
            maximum = 0
            for right in range(left, N):
                length = right - left + 1
                weights[left][right] = (
                    triangular(length) + triangular(left) + triangular(N - 1 - right)
                ) % MOD
                maximum = max(maximum, A[right])
                if left == 0 and right == N - 1:
                    previous[left][right] = maximum % MOD
                else:
                    left_value = infinity if left == 0 else A[left - 1]
                    right_value = infinity if right == N - 1 else A[right + 1]
                    if left_value > maximum and right_value > maximum:
                        previous[left][right] = (maximum - min(left_value, right_value)) % MOD
        for _ in range(self.Q):
            left_sums = [[0] * N for _ in range(N)]
            right_sums = [[0] * N for _ in range(N)]
            for right in range(N):
                running = 0
                for left in range(right + 1):
                    running = (running + previous[left][right] * left) % MOD
                    left_sums[left][right] = running
            for left in range(N):
                running = 0
                for right in range(N - 1, left - 1, -1):
                    running = (running + previous[left][right] * (N - 1 - right)) % MOD
                    right_sums[left][right] = running
            current = [[0] * N for _ in range(N)]
            for left in range(N):
                for right in range(left, N):
                    current[left][right] = (
                        previous[left][right] * weights[left][right]
                        + (left_sums[left - 1][right] if left else 0)
                        + (right_sums[left][right + 1] if right + 1 < N else 0)
                    ) % MOD
            previous = current
        return [
            sum(previous[left][right] for left in range(index + 1) for right in range(index, N)) % MOD
            for index in range(N)
        ]


class RangeShrinkingSequenceCounting(_RLVETask):
    prompt_template = (
        'Count the number of sequences A[1], A[2], ..., A[{N}] such that:\n'
        '- For each i (1 ≤ i ≤ {N}), 1 ≤ A[i] ≤ R[i], where R is given as: {R}\n'
        '- For each i (3 ≤ i ≤ {N}):\n'
        '  - Let r = the minimum value among A[1], ..., A[i−2] that is ≥ A[i−1] (if none exists, r = +∞).\n'
        '  - Let l = the maximum value among A[1], ..., A[i−2] that is ≤ A[i−1] (if none exists, l = −∞).\n'
        '  - Then A[i] must satisfy l ≤ A[i] ≤ r.\n'
        '\n'
        'Let {result_name} be the number of valid sequences modulo {MOD}.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(R=_indexed("R", input_reference, N, 1, ", "), MOD=self.MOD)
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.MOD = rng.randint(2, 1_000_000)

    def sample_input(self, rng):
        sequence = [rng.randint(1, self.N), rng.randint(1, self.N)]
        lower, upper = 1, self.N
        if sequence[0] >= sequence[1]:
            upper = sequence[1]
        if sequence[0] <= sequence[1]:
            lower = sequence[1]
        for _ in range(2, self.N):
            sequence.append(rng.randint(lower, upper))
            if sequence[-2] >= sequence[-1]:
                upper = sequence[-1]
            if sequence[-2] <= sequence[-1]:
                lower = sequence[-1]
        return [rng.randint(value, self.N) for value in sequence]

    def solution_generator(self, input_value):
        sentinel = max(input_value) + 1
        states = {(0, sentinel, value): 1 for value in range(1, input_value[0] + 1)}
        for maximum in input_value[1:]:
            following = defaultdict(int)
            for (lower, upper, current), count in states.items():
                for value in range(lower + 1, min(current - 1, maximum) + 1):
                    following[lower, current, value] += count
                for value in range(current + 1, min(upper - 1, maximum) + 1):
                    following[current, upper, value] += count
                if lower and lower <= maximum:
                    following[lower, lower, lower] += count
                if upper <= maximum and upper <= max(input_value) and lower != upper:
                    following[upper, upper, upper] += count
                if current <= maximum and current not in (lower, upper):
                    following[current, current, current] += count
            states = {state: count % self.MOD for state, count in following.items()}
        return sum(states.values()) % self.MOD


class RepeatSequenceLNDS(_RLVETask):
    prompt_template = (
        'You are given an array that repeats every {n} elements. The initial pattern is: {a}. This pattern '
        'repeats {T} times, creating a total array length of {nT}.\n'
        '\n'
        'For example, if the initial pattern is [1, 3, 2] and it repeats 2 times, the full array would be '
        '[1, 3, 2, 1, 3, 2].\n'
        '\n'
        'Find the length of the longest non-decreasing subsequence (not necessarily contiguous) in this '
        'repeated array.\n'
        '\n'
        'Let {result_name} be the length of that longest non-decreasing subsequence.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(n=N, T=self.T, nT=N * self.T, a=input_reference)
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.T = rng.randint(2, 15)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        counts = [0] * (max(input_value) + 1)
        best = [0] * (max(input_value) + 1)
        for value in input_value:
            counts[value] += 1
        for value in input_value * min(self.T, 2 * self.N):
            best[value] = max(best[: value + 1]) + 1
        return max(best) + max((self.T - 2 * self.N) * max(counts), 0)


class SLOElephants(_RLVETask):
    prompt_template = (
        'There are {N} items labeled from 0 to {N_minus_1}. Each item labeled `i` has an associated cost '
        'C[i]. The array C is: {C}\n'
        'Initially, the items are arranged in the order A (this means the item at position 0 has label '
        'A[0], at position 1 has label A[1], etc): {A}\n'
        'You are required to rearrange the items into the target order B: {B}\n'
        '\n'
        'You may perform any number of swaps. Swapping the items labeled `i` and `j` incurs a cost of C[i] '
        '+ C[j]. Please minimize the total cost of all swaps.\n'
        '\n'
        'Let {result_name} be the minimum possible total cost of all swaps.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            C=_indexed("C", input_reference, N),
            A=" ".join(f"A[{i}]={value}" for i, value in enumerate(self.A)),
            B=" ".join(f"B[{i}]={value}" for i, value in enumerate(self.B)),
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.A, self.B = list(range(self.N)), list(range(self.N))
        while True:
            rng.shuffle(self.A)
            rng.shuffle(self.B)
            if self.A != self.B:
                break

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        destination = [0] * self.N
        for index, elephant in enumerate(self.B):
            destination[elephant] = index
        following = [self.A[destination[elephant]] for elephant in range(self.N)]
        visited = [False] * self.N
        global_minimum = min(input_value)
        answer = 0
        for elephant in range(self.N):
            if visited[elephant]:
                continue
            current, total, minimum, length = elephant, 0, math.inf, 0
            while not visited[current]:
                visited[current] = True
                total += input_value[current]
                minimum = min(minimum, input_value[current])
                length += 1
                current = following[current]
            if length > 1:
                answer += min(
                    total + minimum * (length - 2),
                    total + minimum + global_minimum * (length + 1),
                )
        return answer


# class SkaRockGarden(_RLVETask):
#     prompt_template = (
#         'There are {N} points in a 2D plane, where the i-th point is (X[i], Y[i]) for 0 ≤ i < {N}. Each '
#         'point has a cost M[i] to swap its coordinates (i.e., swapping (x, y) becomes (y, x)). Your goal '
#         'is as follows:\n'
#         '- First, minimize the total perimeter of the smallest axis-aligned rectangle that can enclose all '
#         'points after some of them are optionally swapped. The perimeter is obviously 2 × ((max_x - min_x) '
#         '+ (max_y - min_y)), where max_x and min_x are the maximum and minimum x-coordinates after your '
#         'swaps (similarly for y).\n'
#         '- If multiple swap strategies result in the same minimum perimeter, choose the one with the '
#         'smallest total swap cost (i.e., sum of M[i] for all swapped points).\n'
#         '\n'
#         'X, Y, and M are given as follows:\n'
#         '{X_Y_M}\n'
#         '\n'
#         'Let {result_name} be a single list [minimum perimeter, minimum total swap cost].'
#     )

#     def _format_prompt(self, input_reference, result_name):
#         N = getattr(self, "N", None)
#         fields = self._base_prompt_fields(result_name)
#         fields["X_Y_M"] = "\n".join(
#             f"X[{i}]={input_reference}[{i}] Y[{i}]={self.Y[i]} M[{i}]={self.M[i]}"
#             for i in range(N)
#         )
#         return self.prompt_template.format(**fields)

#     N = 6

#     def _setup(self, rng):
#         self.Y = _integer_list(rng, self.N, 0, 2 * self.N)
#         self.M = _integer_list(rng, self.N, 1, self.N)

#     def sample_input(self, rng):
#         return _integer_list(rng, self.N, 0, 2 * self.N)

#     def solution_generator(self, input_value):
# #         X, Y = input_value, self.Y
#         normalized = [(min(x, y), max(x, y)) for x, y in zip(X, Y)]
#         lower_x = min(x for x, _ in normalized)
#         upper_x = max(x for x, _ in normalized)
#         lower_y = min(y for _, y in normalized)
#         upper_y = max(y for _, y in normalized)
#         perimeter = 2 * ((upper_x - lower_x) + (upper_y - lower_y))
#         best_cost = sum(self.M)
#         for bounds in (
#             (lower_x, upper_x, lower_y, upper_y),
#             (lower_x, upper_y, lower_y, upper_x),
#             (lower_y, upper_x, lower_x, upper_y),
#             (lower_y, upper_y, lower_x, upper_x),
#         ):
#             lx, rx, ly, ry = bounds
#             cost = 0
#             valid = True
#             for x, y, swap_cost in zip(X, Y, self.M):
#                 if lx <= x <= rx and ly <= y <= ry:
#                     continue
#                 if lx <= y <= rx and ly <= x <= ry:
#                     cost += swap_cost
#                 else:
#                     valid = False
#                     break
#             if valid:
#                 best_cost = min(best_cost, cost)
#         return [perimeter, best_cost]


class SlidingWindow(_RLVETask):
    prompt_template = (
        'You are given the following list of {N} numbers: {A}\n'
        'Please find the minimum value in each contiguous subarray of size {K} (there are '
        '{N_minus_K_plus_1} such subarrays in total).\n'
        '\n'
        'Let {result_name} be a single list with those minimum values from the leftmost subarray '
        'to the rightmost, in order.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(K=self.K, N_minus_K_plus_1=N - self.K + 1, A=input_reference)
        return self.prompt_template.format(**fields)

    N = 8

    def _setup(self, rng):
        self.K = self.N // 2

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 0, self.N)

    def solution_generator(self, input_value):
        queue = deque()
        answer = []
        for index, value in enumerate(input_value):
            if queue and queue[0] <= index - self.K:
                queue.popleft()
            while queue and input_value[queue[-1]] > value:
                queue.pop()
            queue.append(index)
            if index >= self.K - 1:
                answer.append(input_value[queue[0]])
        return answer


class Sorting(_RLVETask):
    prompt_template = (
        'You are given the following list of numbers:\n'
        '{A}\n'
        'Please sort them in **ascending order**.\n'
        '\n'
        'Let {result_name} be a single list with the sorted numbers in ascending order.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["A"] = input_reference
        return self.prompt_template.format(**fields)

    N = 6

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 0, 30)

    def solution_generator(self, input_value):
        return sorted(input_value)


class SpyNetwork(_RLVETask):
    prompt_template = (
        'You are given a **directed graph** with {N} vertices, labeled from 0 to {N_minus_1}.\n'
        '\n'
        'The graph contains the following directed edges. Each edge is represented as a tuple (s, t), '
        'meaning there is a directed edge **from vertex s to vertex t**:\n'
        '{edges}\n'
        '\n'
        'Each vertex i has an associated cost c[i], given as follows:\n'
        '{costs}\n'
        '\n'
        'Your task is to select a subset of vertices s_1, s_2, ..., s_k such that:\n'
        '- Every vertex in the graph is reachable (i.e., there exists a path ending at that vertex) '
        'starting from at least one of the selected vertices.\n'
        '- Your goal is to **minimize** the total cost of the selected vertices: c[s_1] + c[s_2] + ... + '
        'c[s_k].\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            edges="\n".join(f"({start}, {end})" for start, end in self.edges),
            costs="\n".join(f"c[{i}]={input_reference}[{i}]" for i in range(N)),
            c_0=f"{input_reference}[0]",
            c_1=f"{input_reference}[1]",
            c_N_minus_1=f"{input_reference}[{N - 1}]",
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        dominated = [rng.random() < 0.7 for _ in range(self.N)]
        eligible = [
            (start, end)
            for start in range(self.N)
            for end in range(self.N)
            if start != end and (not dominated[start] or dominated[end])
        ]
        self.edges = rng.sample(eligible, 6)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, self.N)

    def solution_generator(self, input_value):
        graph = [[] for _ in range(self.N)]
        for start, end in self.edges:
            graph[start].append(end)
        clock = 0
        indices = [-1] * self.N
        low = [0] * self.N
        stack = []
        on_stack = [False] * self.N
        component = [-1] * self.N
        component_count = 0

        def visit(node):
            nonlocal clock, component_count
            indices[node] = low[node] = clock
            clock += 1
            stack.append(node)
            on_stack[node] = True
            for target in graph[node]:
                if indices[target] == -1:
                    visit(target)
                    low[node] = min(low[node], low[target])
                elif on_stack[target]:
                    low[node] = min(low[node], indices[target])
            if low[node] == indices[node]:
                while True:
                    target = stack.pop()
                    on_stack[target] = False
                    component[target] = component_count
                    if target == node:
                        break
                component_count += 1

        for node in range(self.N):
            if indices[node] == -1:
                visit(node)
        incoming = [False] * component_count
        for start, end in self.edges:
            if component[start] != component[end]:
                incoming[component[end]] = True
        minima = [math.inf] * component_count
        for node, cost in enumerate(input_value):
            minima[component[node]] = min(minima[component[node]], cost)
        return sum(minima[index] for index in range(component_count) if not incoming[index])


class StuntFlying(_RLVETask):
    prompt_template = (
        'There are {K} elements labeled from 0 to {K_minus_1}, and each element `x` has an associated '
        'value C[x]. C is: {C}\n'
        'You need to build an array A of length {N}, where each A[i] is one of these elements (i.e., 0 ≤ '
        'A[i] < {K} for all 1 ≤ i ≤ {N}). Each position i in A has a value defined as **C[A[i]] × T[i]**, '
        'where T[i] is determined as follows:\n'
        '- If there is no previous index j (0 ≤ j < i) such that A[j] = A[i], then T[i] = 0.\n'
        '- Otherwise, let j be the largest index (basically, closest to i) such that A[j] = A[i] (0 ≤ j < '
        'i), and set T[i] = i - j.\n'
        '\n'
        'Can you maximize the sum of all values **C[A[i]] × T[i]**?\n'
        '\n'
        'Let {result_name} be the maximum possible sum of all values C[A[i]] × T[i].'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            K=self.K,
            K_minus_1=self.K - 1,
            C="; ".join(f"C[{i}] = {input_reference}[{i}]" for i in range(self.K)),
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        self.K = rng.randint(2, self.N)

    def sample_input(self, rng):
        return _integer_list(rng, self.K, 1, self.K)

    def solution_generator(self, input_value):
        remaining = self.N - 1
        answer = 0
        for value in sorted(input_value, reverse=True):
            if remaining <= 0:
                break
            answer += remaining * value
            remaining -= 2
        return answer


class ValueDiminishingSelection(_RLVETask):
    prompt_template = (
        'You are given {N} items labeled from `0` to `{N_minus_1}`. Each item has a base value W[i] and a '
        'diminishing factor R[i]. The list of values and diminishing factors is given as:\n'
        '{W_and_R}\n'
        '\n'
        'You must select a sequence of **distinct items** (the order matters). When selecting the i-th '
        'item:\n'
        '- Its effective value is W[i] minus the total of R[j] for all previously selected items j.\n'
        '- In other words, each item selected **after** i will lose R[i] from their gain due to the '
        'diminishing effect.\n'
        '\n'
        'Your goal is to select a sequence of items to **maximize the total gain**.\n'
        '\n'
        'Let {result_name} be the maximum possible total gain.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields["W_and_R"] = "\n".join(
            f"W[{i}]={input_reference}[{i}] R[{i}]={self.R[i]}" for i in range(N)
        )
        return self.prompt_template.format(**fields)

    N = 6

    def _setup(self, rng):
        self.R = _integer_list(rng, self.N, 1, self.N)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 0, self.N**2 // 2)

    def solution_generator(self, input_value):
        pairs = sorted(zip(input_value, self.R), key=lambda item: item[1], reverse=True)
        best = [0] + [None] * self.N
        answer = 0
        for index, (value, reduction) in enumerate(pairs):
            following = best.copy()
            for count in range(1, index + 2):
                if best[count - 1] is not None:
                    candidate = best[count - 1] + value - reduction * (count - 1)
                    if following[count] is None or candidate > following[count]:
                        following[count] = candidate
                        answer = max(answer, candidate)
            best = following
        return answer


class WarehouseConstruction(_RLVETask):
    prompt_template = (
        'You are given {N} factories arranged from top to bottom along a mountain, indexed from 0 to '
        '{N_minus_1}. Factory 0 is at the top and factory {N_minus_1} is at the bottom.\n'
        '\n'
        'Each factory has\n'
        '- Distance from factory 0: {D}\n'
        '- Number of products: {P}\n'
        '- Cost to build a warehouse at that factory: {C}\n'
        '\n'
        'You can choose to build warehouses at any subset of factories.\n'
        '- A warehouse can store any number of products.\n'
        '- If a factory does not build a warehouse, all its products must be sent **downhill** to a '
        'factory with a warehouse (i.e., to a factory with a higher index). Transporting one product over '
        'one unit of distance costs 1.\n'
        '- The total cost is the sum of warehouse construction costs and product transportation costs. Try '
        'your best to minimize the total cost.\n'
        '\n'
        'Let {result_name} be the minimum possible total cost.'
    )

    def _format_prompt(self, input_reference, result_name):
        N = getattr(self, "N", None)
        fields = self._base_prompt_fields(result_name)
        fields.update(
            D=" ".join(f"D[{i}]={value}" for i, value in enumerate(self.D)),
            P=" ".join(f"P[{i}]={value}" for i, value in enumerate(self.P)),
            C=_indexed("C", input_reference, N),
        )
        return self.prompt_template.format(**fields)

    N = 7

    def _setup(self, rng):
        self.D = [0] + sorted(rng.sample(range(1, 2 * self.N + 1), self.N - 1))
        self.P = _integer_list(rng, self.N, 0, self.N)

    def sample_input(self, rng):
        return _integer_list(rng, self.N, 1, 2 * self.N)

    def solution_generator(self, input_value):
        prefix_products = [0] * (self.N + 1)
        prefix_weighted = [0] * (self.N + 1)
        for i in range(1, self.N + 1):
            prefix_products[i] = prefix_products[i - 1] + self.P[i - 1]
            prefix_weighted[i] = prefix_weighted[i - 1] + self.D[i - 1] * self.P[i - 1]
        best = [0] * (self.N + 1)
        queue = deque([0])

        def y(index):
            return best[index] + prefix_weighted[index]

        for i in range(1, self.N + 1):
            while len(queue) >= 2 and y(queue[1]) - y(queue[0]) <= self.D[i - 1] * (
                prefix_products[queue[1]] - prefix_products[queue[0]]
            ):
                queue.popleft()
            chosen = queue[0]
            best[i] = (
                best[chosen]
                + self.D[i - 1] * (prefix_products[i] - prefix_products[chosen])
                - (prefix_weighted[i] - prefix_weighted[chosen])
                + input_value[i - 1]
            )
            while len(queue) >= 2:
                newest, previous = queue[-1], queue[-2]
                if (y(newest) - y(previous)) * (
                    prefix_products[i] - prefix_products[newest]
                ) >= (y(i) - y(newest)) * (
                    prefix_products[newest] - prefix_products[previous]
                ):
                    queue.pop()
                else:
                    break
            queue.append(i)
        answer = best[self.N]
        index = self.N
        while index > 0 and self.P[index - 1] == 0:
            index -= 1
            answer = min(answer, best[index])
        return answer


def _indexed(name, reference, length, start=0, separator=" "):
    return separator.join(
        f"{name}[{index}]={reference}[{index - start}]"
        for index in range(start, start + length)
    )


def _matrix_rows(name, reference, size):
    return "\n".join(
        " ".join(
            f"{name}[{row}][{column}]={reference}[{row}][{column}]"
            for column in range(size)
        )
        for row in range(size)
    )


RLVE_TASK_CLASSES = [
    # AndOr_Sequence_Counting,
    BoundedSubarrayCounting,
    MaxDifferentGroupPairDivision,
    MoneyChargingGame,
    ProtectingFlowers,
    RangeConstrained_IncreasingSequence_Counting,
    SalesmanFatigue,
    SpiralMatrix,
    TaskArrangement,
    TreeCenter,
    WeightedBinaryTree,
    BoxScheduling,
    CantorExpansion,
    CoinSquareGame,
    Expression_AddingParenthese_Counting,
    FractionalProgramming,
    FractionalProgramming_BipartiteGraphMatching,
    GradeRankingCounting,
    GridColoringCounting,
    KingSorting,
    Knapsack,
    LandAcquisition,
    # MaxMultSplit,
    Max_TreeConstrainedPermutation_Weight,
    MaximumLexicographicalOrderSubsequence,
    MinPairSumMultiplicationPermutation,
    MinSumChebyshevDistance,
    # MinSumPreXor,
    MinimumIntervalCoverage,
    MinimumTreeWeightedDominatingAncestor,
    Minimum_DominatingSet,
    Minimum_VertexCover,
    NegativeBase,
    RandomRangeMaxExpectation,
    RangeShrinkingSequenceCounting,
    RepeatSequenceLNDS,
    SLOElephants,
    # SkaRockGarden,
    SlidingWindow,
    Sorting,
    SpyNetwork,
    StuntFlying,
    ValueDiminishingSelection,
    WarehouseConstruction,
]


_SCALAR_INPUT_TASKS = {GridColoringCounting, NegativeBase}
_LIST_OUTPUT_TASKS = {
    SalesmanFatigue,
    SpiralMatrix,
    BoxScheduling,
    MaximumLexicographicalOrderSubsequence,
    NegativeBase,
    RandomRangeMaxExpectation,
    # SkaRockGarden,
    SlidingWindow,
    Sorting,
}


def _snake_case(name):
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"_+", "_", name).lower()


for _task_class in RLVE_TASK_CLASSES:
    _task_class.category = "rlve"
    _task_class.task_name = _snake_case(_task_class.__name__)
    _task_class.input_type = "scalar" if _task_class in _SCALAR_INPUT_TASKS else "list"
    _task_class.output_type = "list" if _task_class in _LIST_OUTPUT_TASKS else "scalar"


def load_pool() -> list[dict[str, Any]]:
    task_implementations = sorted(
        RLVE_TASK_CLASSES,
        key=lambda task_implementation: (
            task_implementation.input_type,
            task_implementation.output_type,
            task_implementation.task_name,
        ),
    )
    return [
        {
            "category": "rlve",
            "task_id": task_id,
            "task_name": task_implementation.task_name,
            "input_type": task_implementation.input_type,
            "output_type": task_implementation.output_type,
            "comparator": COMPARATOR,
            "task_implementation": task_implementation,
        }
        for task_id, task_implementation in enumerate(task_implementations)
    ]


def _valid_output(output):
    if isinstance(output, bool):
        return False
    if isinstance(output, int):
        return True
    if isinstance(output, float):
        return math.isfinite(output)
    if isinstance(output, list):
        return all(not isinstance(value, list) and _valid_output(value) for value in output)
    return False


def make_stage(
    task: dict[str, Any],
    task_number: int,
    input_names: list[str],
    _input: Any,
    get_input_attempts_per_task: int = 5,
    with_bank: bool = False,
) -> tuple[Any, str, list[str], Any, list[dict[str, Any]]]:
    # Parent values never reach the task: a non-root gets a bank instead, and the
    # input tool picks from it by whether the model's dependencies were right.
    instance, bank = task["task_implementation"].generate_inputs(
        get_input_attempts_per_task + 1 if (_input is not None or with_bank) else 1,
    )
    correct, *wrong_inputs = bank
    return (
        correct["output"],
        instance.prompt_generator(task_number),
        [],
        correct["input"],
        wrong_inputs,
    )
