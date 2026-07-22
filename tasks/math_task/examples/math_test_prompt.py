"""
Test script: generates an example prompt for each math task class.
Instantiates each task with random inputs and prints the prompt the model sees.
"""
import sys
import os
import math
import random
import sympy

# Allow running as a standalone script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tasks.math_task.math_task import (
    MatrixMulTask, FWHTTask, NTTTask, SubsetZetaTransformTask,
    KroneckerProductTask, InterpolateEvalTask, SummatoryTotientTask, DivisorSumTask,
    ConvolutionTask, PolyRemainderTask, PolyQuotientTask,
    MatrixPolynomialTask, SortByDivisorCountTask,
    MatrixFrobeniusSqTask, ConvexHullAreaTask, MatrixNorm1Task,
    MatrixTraceCubeTask, PermutationInversionsTask, PermutationOrderTask,
    PolySquareFreeEvalTask, PolyResultantFixedTask, PairwiseGcdSumTask,
    PrimeOmegaSumTask, SumDivisorCountsTask, PrimitiveRootCountTask,
    LegendreResidueWeightTask, ScaledStirlingFirstKindTask, DivisorSigmaSquareTask,
    MultiplicativeOrderTask, UnitaryDivisorSumTask, ModularMultiplicationSequenceTask,
    WeightedDivisorCountPrefixSequenceTask, GrayCodeSequenceTask, TotientShiftSequenceTask,
    SmallestFactorQuadraticTask, TotientProductSequenceTask,
    PolygonDiagonalsSequenceTask, PrimesBetweenSquaresTask,
    CentralBinomialPrimeValuationVectorTask, QuadraticResiduesPrimeTask,
)

random.seed(42)

SCALAR_MAX_MAG = 100
LIST_LEN_MAX = 20
TASK_IND = 3  # example task index in a chain


def rand_int_list(n, low=-50, high=50):
    return [random.randint(low, high) for _ in range(n)]


def make_seq_inp(n=8, chain_name="task_2_out"):
    """Standard list input under key 'seq'."""
    return {"seq": {"chain_name": chain_name, "value": rand_int_list(n)}}


def make_n_inp(val=None, chain_name="task_2_out"):
    """Standard scalar input under key 'n'."""
    if val is None:
        val = random.randint(3, 15)
    return {"n": {"chain_name": chain_name, "value": val}}


def make_matrix_inp(rows=3, cols=3, chain_name="task_2_out"):
    """Input for matrix-based tasks: matrix_1 as a flat list + dimension params."""
    flat = rand_int_list(rows * cols)
    return {
        "matrix_1": {"chain_name": chain_name, "value": flat},
        "num_rows": {"chain_name": None, "value": rows},
        "num_cols": {"chain_name": None, "value": cols},
    }


def make_points_inp(n=8, chain_name="task_2_out"):
    """Input for InterpolateEvalTask."""
    pts = rand_int_list(n, -20, 20)
    return {"points": {"chain_name": chain_name, "value": pts}}


def print_task(label, task):
    """Print a task's prompt with header/footer."""
    sep = "-" * 70
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Output: {task.out}")
    print(f"{'=' * 70}")
    print(task.prompt)
    print(sep)


def test_all_prompts():
    print("=" * 70)
    print(f"  MATH TASK PROMPT TEST  |  scalar_max_mag={SCALAR_MAX_MAG}  list_len_max={LIST_LEN_MAX}")
    print("=" * 70)

    # ===================== LIST -> LIST =====================

    # MatrixMulTask
    inp = make_matrix_inp(3, 3)
    task = MatrixMulTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("MatrixMulTask (list->list)", task)

    # FWHTTask
    inp = make_seq_inp(8)
    task = FWHTTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("FWHTTask (list->list)", task)

    # NTTTask
    inp = make_seq_inp(8)
    task = NTTTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("NTTTask (list->list)", task)

    # SubsetZetaTransformTask
    inp = make_seq_inp(8)
    task = SubsetZetaTransformTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("SubsetZetaTransformTask (list->list)", task)

    # KroneckerProductTask
    inp = make_matrix_inp(2, 2)
    task = KroneckerProductTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("KroneckerProductTask (list->list)", task)

    # ConvolutionTask
    inp = make_seq_inp(6)
    task = ConvolutionTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("ConvolutionTask (list->list)", task)

    # PolyRemainderTask
    inp = make_seq_inp(6)
    task = PolyRemainderTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PolyRemainderTask (list->list)", task)

    # PolyQuotientTask
    inp = make_seq_inp(6)
    task = PolyQuotientTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PolyQuotientTask (list->list)", task)

    # MatrixPolynomialTask
    inp = make_matrix_inp(3, 3)
    task = MatrixPolynomialTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("MatrixPolynomialTask (list->list)", task)

    # SortByDivisorCountTask
    inp = make_seq_inp(8)
    task = SortByDivisorCountTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("SortByDivisorCountTask (list->list)", task)

    # ===================== LIST -> SCALAR =====================

    # MatrixFrobeniusSqTask  (needs perfect-square length seq)
    inp = make_seq_inp(9)
    task = MatrixFrobeniusSqTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("MatrixFrobeniusSqTask (list->scalar)", task)

    # ConvexHullAreaTask
    inp = make_seq_inp(8)
    task = ConvexHullAreaTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("ConvexHullAreaTask (list->scalar)", task)

    # MatrixNorm1Task  (needs perfect-square length seq)
    inp = make_seq_inp(9)
    task = MatrixNorm1Task(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("MatrixNorm1Task (list->scalar)", task)

    # MatrixTraceCubeTask  (needs perfect-square length seq)
    inp = make_seq_inp(9)
    task = MatrixTraceCubeTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("MatrixTraceCubeTask (list->scalar)", task)

    # PermutationInversionsTask
    inp = make_seq_inp(8)
    task = PermutationInversionsTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PermutationInversionsTask (list->scalar)", task)

    # PermutationOrderTask
    inp = make_seq_inp(8)
    task = PermutationOrderTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PermutationOrderTask (list->scalar)", task)

    # PolySquareFreeEvalTask
    inp = make_seq_inp(6)
    task = PolySquareFreeEvalTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PolySquareFreeEvalTask (list->scalar)", task)

    # PolyResultantFixedTask
    inp = make_seq_inp(6)
    task = PolyResultantFixedTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PolyResultantFixedTask (list->scalar)", task)

    # InterpolateEvalTask
    inp = make_points_inp(5)
    task = InterpolateEvalTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("InterpolateEvalTask (list->scalar)", task)

    # PairwiseGcdSumTask
    inp = make_seq_inp(6)
    task = PairwiseGcdSumTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PairwiseGcdSumTask (list->scalar)", task)

    # ===================== SCALAR -> SCALAR =====================

    # SummatoryTotientTask
    inp = make_n_inp(12)
    task = SummatoryTotientTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("SummatoryTotientTask (scalar->scalar)", task)

    # DivisorSumTask
    inp = make_n_inp(12)
    task = DivisorSumTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("DivisorSumTask (scalar->scalar)", task)

    # PrimeOmegaSumTask
    inp = make_n_inp(10)
    task = PrimeOmegaSumTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PrimeOmegaSumTask (scalar->scalar)", task)

    # SumDivisorCountsTask
    inp = make_n_inp(10)
    task = SumDivisorCountsTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("SumDivisorCountsTask (scalar->scalar)", task)

    # PrimitiveRootCountTask
    inp = make_n_inp(8)
    task = PrimitiveRootCountTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PrimitiveRootCountTask (scalar->scalar)", task)

    # LegendreResidueWeightTask
    inp = make_n_inp(5)
    task = LegendreResidueWeightTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("LegendreResidueWeightTask (scalar->scalar)", task)

    # ScaledStirlingFirstKindTask
    inp = make_n_inp(6)
    task = ScaledStirlingFirstKindTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("ScaledStirlingFirstKindTask (scalar->scalar)", task)

    # DivisorSigmaSquareTask
    inp = make_n_inp(12)
    task = DivisorSigmaSquareTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("DivisorSigmaSquareTask (scalar->scalar)", task)

    # MultiplicativeOrderTask
    inp = make_n_inp(7)
    task = MultiplicativeOrderTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("MultiplicativeOrderTask (scalar->scalar)", task)

    # UnitaryDivisorSumTask
    inp = make_n_inp(10)
    task = UnitaryDivisorSumTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("UnitaryDivisorSumTask (scalar->scalar)", task)

    # ===================== SCALAR -> LIST =====================

    # ModularMultiplicationSequenceTask
    inp = make_n_inp(8)
    task = ModularMultiplicationSequenceTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("ModularMultiplicationSequenceTask (scalar->list)", task)

    # WeightedDivisorCountPrefixSequenceTask
    inp = make_n_inp(4)
    task = WeightedDivisorCountPrefixSequenceTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("WeightedDivisorCountPrefixSequenceTask (scalar->list)", task)

    # GrayCodeSequenceTask
    inp = make_n_inp(8)
    task = GrayCodeSequenceTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("GrayCodeSequenceTask (scalar->list)", task)

    # TotientShiftSequenceTask
    inp = make_n_inp(8)
    task = TotientShiftSequenceTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("TotientShiftSequenceTask (scalar->list)", task)

    # SmallestFactorQuadraticTask
    inp = make_n_inp(8)
    task = SmallestFactorQuadraticTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("SmallestFactorQuadraticTask (scalar->list)", task)

    # TotientProductSequenceTask
    inp = make_n_inp(8)
    task = TotientProductSequenceTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("TotientProductSequenceTask (scalar->list)", task)

    # PolygonDiagonalsSequenceTask
    inp = make_n_inp(8)
    task = PolygonDiagonalsSequenceTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PolygonDiagonalsSequenceTask (scalar->list)", task)

    # PrimesBetweenSquaresTask
    inp = make_n_inp(5)
    task = PrimesBetweenSquaresTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("PrimesBetweenSquaresTask (scalar->list)", task)

    # CentralBinomialPrimeValuationVectorTask
    inp = make_n_inp(6)
    task = CentralBinomialPrimeValuationVectorTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("CentralBinomialPrimeValuationVectorTask (scalar->list)", task)

    # QuadraticResiduesPrimeTask
    inp = make_n_inp(5)
    task = QuadraticResiduesPrimeTask(TASK_IND, inp, SCALAR_MAX_MAG, LIST_LEN_MAX)
    print_task("QuadraticResiduesPrimeTask (scalar->list)", task)


if __name__ == '__main__':
    test_all_prompts()
