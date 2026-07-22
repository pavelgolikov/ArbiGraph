import sympy
import random
import math
from sympy.combinatorics import Permutation
from sympy.ntheory.continued_fraction import continued_fraction_reduce


def list_to_matrix_compute(seq, rows, cols):
    return sympy.Matrix(rows, cols, seq[:rows*cols])

def pad_to_power_of_2_compute(seq):
    return seq + [0] * ((1 << (len(seq) - 1).bit_length()) - len(seq)) if seq else [0]

# ================= LIST -> LIST =================
def matrix_mul_task(seq):
    k = math.isqrt(len(seq))
    if k == 0: return []
    mat = list_to_matrix_compute(seq, k, k)
    mat2 = sympy.Matrix(k, k, [random.randint(-5, 5) for _ in range(k*k)])
    return list(mat * mat2)

def convolution_task(seq):
    seq2 = [random.randint(-5, 5) for _ in range(len(seq))]
    return list(sympy.discrete.convolutions.convolution(seq, seq2))

def poly_remainder_task(seq):
    x = sympy.Symbol('x')
    seq2 = [random.randint(-5, 5) for _ in range(len(seq))]
    P = sympy.Poly(seq, x)
    Q = sympy.Poly([1] + seq2, x)
    return sympy.rem(P, Q).all_coeffs()

def fwht_task(seq):
    padded = pad_to_power_of_2_compute(seq)
    return list(sympy.discrete.transforms.fwht(padded))

def ntt_task(seq):
    padded = pad_to_power_of_2_compute(seq)
    as_ints = [int(x) for x in padded]
    prime = len(padded) + 1
    while not sympy.isprime(prime): prime += len(padded)
    return list(sympy.discrete.transforms.ntt(as_ints, prime))

def subset_zeta_transform_task(seq):
    padded = pad_to_power_of_2_compute(seq)
    return list(sympy.discrete.transforms.mobius_transform(padded))

def kronecker_product_task(seq):
    k = math.isqrt(len(seq))
    if k == 0: return []
    m1 = list_to_matrix_compute(seq, k, k)
    m2 = sympy.Matrix(2, 2, [1, 2, 3, 4])
    return list(sympy.matrices.kronecker_product(m1, m2))

def poly_quotient_task(seq):
    if not seq: return []
    x = sympy.Symbol('x')
    p1 = sympy.Poly(seq, x)
    p2 = sympy.Poly([1, 1, 1], x)
    q, r = sympy.div(p1, p2)
    return q.all_coeffs() if hasattr(q, 'all_coeffs') else []

def matrix_polynomial_task(seq):
    k = math.isqrt(len(seq))
    if k == 0: return []
    m = list_to_matrix_compute(seq, k, k)
    return list(m**3 + m**2 + m)

def sort_by_divisor_count_task(seq):
    return sorted(seq, key=lambda x: sympy.ntheory.factor_.divisor_count(abs(x)) if x != 0 else 0)

# ================= LIST -> SCALAR =================
def matrix_frobenius_sq_task(seq):
    """Frobenius norm squared = sum of all elements squared. Bounded, scales with size."""
    k = math.isqrt(len(seq))
    if k == 0: return 0
    mat = list_to_matrix_compute(seq, k, k)
    return sum(mat[i, j]**2 for i in range(k) for j in range(k))

def convex_hull_area_task(seq):
    pts = [sympy.geometry.Point(i, val) for i, val in enumerate(seq)]
    if len(pts) < 3: return 0
    hull = sympy.geometry.convex_hull(*pts)
    return 2 * hull.area if hasattr(hull, 'area') else 0

def matrix_norm_1_task(seq):
    k = math.isqrt(len(seq))
    if k == 0: return 0
    mat = sympy.Matrix(k, k, seq[:k*k])
    return max(sum(abs(mat[r, c]) for r in range(k)) for c in range(k))

def matrix_trace_cube_task(seq):
    k = int(len(seq)**0.5)
    if k == 0: return 0
    mat = sympy.Matrix(k, k, seq[:k*k])
    return (mat**3).trace()

def permutation_inversions_task(seq):
    from sympy.combinatorics import Permutation
    if not seq: return 0
    indices = sorted(range(len(seq)), key=lambda x: seq[x])
    return Permutation(indices).inversions()

def permutation_order_task(seq):
    indices = sorted(range(len(seq)), key=lambda x: seq[x])
    return Permutation(indices).order() if seq else 1

def poly_square_free_eval_task(seq):
    if not seq: return 0
    x = sympy.Symbol('x')
    poly = sympy.Poly(seq, x)
    return sympy.polys.polytools.sqf_part(poly).subs(x, 1)

def poly_resultant_fixed_task(seq):
    x = sympy.Symbol('x')
    P = sympy.Poly(seq, x)
    Q = sympy.Poly(x**2 + x + 1, x)
    return sympy.resultant(P, Q)

def interpolate_eval_task(seq, eval_point):
    k_len = len(seq) // 8 + 3
    sub_seq = seq[:k_len]
    x = sympy.Symbol('x')
    if not sub_seq: return 0
    points_coords = [(i, val) for i, val in enumerate(sub_seq)]
    poly = sympy.interpolate(points_coords, x)
    return poly.subs(x, k_len + eval_point)

def pairwise_gcd_sum_task(seq):
    """Sum of GCD(|seq[i]|, |seq[j]|) for all i < j. Bounded, scales with size."""
    if len(seq) < 2: return 0
    total = 0
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            total += sympy.gcd(abs(seq[i]), abs(seq[j]))
    return int(total)

# ================= SCALAR -> SCALAR =================
def prime_omega_sum_task(n):
    """Sum of number-of-distinct-prime-factors for all integers from 2 to n."""
    n = abs(n)
    if n < 2: return 0
    return sum(sympy.factorint(i).__len__() for i in range(2, n + 1))

def sum_divisor_counts_task(n):
    """Sum of divisor_count(i) for i in 1..n. Grows as n*ln(n), nicely bounded."""
    n = abs(n)
    if n < 1: return 0
    return sum(int(sympy.divisor_count(i)) for i in range(1, n + 1))

def summatory_totient_task(n):
    """Sum of phi(k) for k in 1..n, with an offset to avoid tiny outputs."""
    n = abs(n) + 30
    return sum(int(sympy.totient(k)) for k in range(1, n + 1))

def primitive_root_count_task(n):
    """Euler totient of the n-th prime (= count of primitive roots mod p)."""
    n = abs(n)
    if n < 1: return 0
    p = sympy.prime(n)
    return sympy.totient(p)

def legendre_residue_weight_task(n):
    """Twice the weighted sum of quadratic residues in 1..n modulo nextprime(n+2)."""
    n = abs(n) + 25
    p = int(sympy.nextprime(n + 2))
    return sum(a * (int(sympy.legendre_symbol(a, p)) + 1) for a in range(1, n + 1))

def scaled_stirling_first_task(n):
    """Unsigned Stirling c(m, m-2) for m = floor(sqrt(n)) + 7."""
    n = abs(n)
    m = math.isqrt(n) + 7
    return 2 * sympy.binomial(m, 3) + 3 * sympy.binomial(m, 4)

def divisor_sigma_square_task(n):
    """Sum of the squares of divisors of n."""
    n = abs(n)
    return int(sympy.divisor_sigma(n, 2))

def divisor_sum_task(n):
    n = abs(n)
    return sympy.divisor_sigma(n, 1)

def multiplicative_order_task(n):
    n = abs(n)
    if n <= 0: return 0
    try:
        return sympy.ntheory.n_order(2, 2*n + 1)
    except:
        return 0

# ================= SCALAR -> LIST =================
def modular_multiplication_sequence_task(n):
    """(i * n) modulo the n-th prime, for i from 1 to n."""
    n = abs(n)
    if n < 1: return [0]
    p = sympy.prime(n)
    return [int((i * n) % p) for i in range(1, n + 1)]

def weighted_divisor_count_prefix_sequence_task(n):
    """Prefix sums of i times the divisor count of n + i, for i from 1 to n."""
    n = abs(n)
    if n < 1: return [0]
    total = 0
    out = []
    for i in range(1, n + 1):
        total += i * int(sympy.divisor_count(n + i))
        out.append(total)
    return out

def gray_code_sequence_task(n):
    """Integer values of the first n Gray code codewords shifted by n."""
    from sympy.combinatorics.graycode import bin_to_gray
    n = abs(n)
    if n < 1: return [0]
    return [int(bin_to_gray(bin(i + n)[2:]), 2) for i in range(1, n + 1)]

def totient_shift_sequence_task(n):
    """totient(i + n) for i from 1 to n."""
    n = abs(n)
    if n < 1: return [0]
    return [int(sympy.totient(i + n)) for i in range(1, n + 1)]

def smallest_factor_quadratic_task(n):
    """Smallest prime factor of i^2 + n, for i from 1 to n."""
    n = abs(n)
    if n < 1: return [0]
    return [int(min(sympy.factorint(i**2 + n).keys())) if (i**2 + n) > 1 else 1 for i in range(1, n + 1)]

def totient_product_sequence_task(n):
    """Euler's totient of i * n for i from 1 to n."""
    n = abs(n)
    if n < 1: return [0]
    return [int(sympy.totient(i * n)) for i in range(1, n + 1)]

def polygon_diagonals_sequence_task(n):
    """Number of diagonals in a polygon with i + n vertices, for i from 1 to n."""
    n = abs(n)
    if n < 1: return [0]
    return [int((i + n) * (i + n - 3) // 2) for i in range(1, n + 1)]

def primes_between_squares_task(n):
    n = abs(n)
    return list(sympy.primerange(n**2 + 1, (n+1)**2)) if n > 0 else []

def central_binomial_prime_valuation_vector_task(n):
    """p * (e_p + 1) for each prime p <= 2*n, where e_p is the central binomial valuation."""
    n = abs(n) + 30
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

def quadratic_residues_prime_task(n):
    n = abs(n)
    return sympy.ntheory.quadratic_residues(sympy.prime(n)) if n > 0 else []


def unitary_divisor_sum_task(n):
    return sympy.functions.combinatorial.numbers.udivisor_sigma(n, 1)


def test_functions():
    sizes = [4, 6, 8, 12, 16, 24, 32, 50, 64]
    scalar_mags = [4, 16, 64, 250]
    
    functions_scalar_scalar = [
        ("PrimeOmegaSum", prime_omega_sum_task, False),
        ("SumDivisorCounts", sum_divisor_counts_task, False),
        ("SummatoryTotient", summatory_totient_task, False),
        ("PrimitiveRootCount", primitive_root_count_task, False),
        ("LegendreResidueWeight", legendre_residue_weight_task, False),
        ("ScaledStirlingFirstKind", scaled_stirling_first_task, False),
        ("DivisorSigmaSquare", divisor_sigma_square_task, False),
        ("DivisorSum", divisor_sum_task, False),
        ("MultiplicativeOrder", multiplicative_order_task, False),
        ("UnitaryDivisorSum", unitary_divisor_sum_task, False),
    ]

    functions_scalar_list = [
        ("ModularMultiplicationSequence", modular_multiplication_sequence_task),
        ("WeightedDivisorCountPrefixSequence", weighted_divisor_count_prefix_sequence_task),
        ("GrayCodeSequence", gray_code_sequence_task),
        ("TotientShiftSequence", totient_shift_sequence_task),
        ("SmallestFactorQuadratic", smallest_factor_quadratic_task),
        ("TotientProductSequence", totient_product_sequence_task),
        ("PolygonDiagonalsSequence", polygon_diagonals_sequence_task),
        ("PrimesBetweenSquares", primes_between_squares_task),
        ("CentralBinomialPrimeValuationVector", central_binomial_prime_valuation_vector_task),
        ("QuadraticResiduesPrime", quadratic_residues_prime_task),
    ]
    
    functions_list_scalar = [
        ("MatrixFrobeniusSq", matrix_frobenius_sq_task, False),
        ("ConvexHullArea", convex_hull_area_task, False),
        ("MatrixNorm1", matrix_norm_1_task, False),
        ("MatrixTraceCube", matrix_trace_cube_task, False),
        ("PermutationInversions", permutation_inversions_task, False),
        ("PermutationOrder", permutation_order_task, False),
        ("PolySquareFreeEval", poly_square_free_eval_task, False),
        ("PolyResultantFixed", poly_resultant_fixed_task, False),
        ("InterpolateEval", interpolate_eval_task, True),
        ("PairwiseGcdSum", pairwise_gcd_sum_task, False),
    ]

    functions_list_list = [
        ("MatrixMul", matrix_mul_task),
        ("Convolution", convolution_task),
        ("PolyRemainder", poly_remainder_task),
        ("FWHT", fwht_task),
        ("NTT", ntt_task),
        ("SubsetZetaTransform", subset_zeta_transform_task),
        ("KroneckerProduct", kronecker_product_task),
        ("PolyQuotient", poly_quotient_task),
        ("MatrixPolynomial", matrix_polynomial_task),
        ("SortByDivisorCount", sort_by_divisor_count_task),
    ]

    print("================ SCALAR -> SCALAR ================")
    for name, func, has_k in functions_scalar_scalar:
        print(f"\n--- {name} ---")
        for n in scalar_mags:
            k = random.randint(2, 5) if has_k else None
            try:
                out = func(n, k) if has_k else func(n)
                out_str = str(out)
                print(f"n={n:2}{', k='+str(k) if has_k else '    '} -> val: {out_str}")
            except Exception as e:
                print(f"n={n:2}{', k='+str(k) if has_k else '    '} -> ERROR: {e}")

    print("\n================ SCALAR -> LIST ================")
    for name, func in functions_scalar_list:
        print(f"\n--- {name} ---")
        for n in scalar_mags:
            try:
                out = func(n)
                print(f"n={n:2} -> Sequence: {[str(x) for x in out]}\n")
            except Exception as e:
                print(f"n={n:2} -> ERROR: {e}")

    print("\n================ LIST -> SCALAR ================")
    for name, func, has_k in functions_list_scalar:
        print(f"\n--- {name} ---")
        for n in sizes:
            seq = [random.randint(-100, 100) for _ in range(n)]
            k = random.randint(2, 4) if has_k else None
            try:
                out = func(seq, k) if has_k else func(seq)
                out_str = str(out)
                print(f"seq={seq}{', k='+str(k) if has_k else ''} -> val: {out_str}")
            except Exception as e:
                print(f"seq={seq}{', k='+str(k) if has_k else '    '} -> ERROR: {e}")

    print("\n================ LIST -> LIST ================")
    for name, func in functions_list_list:
        print(f"\n--- {name} ---")
        for n in sizes:
            seq = [random.randint(-100, 100) for _ in range(n)]
            try:
                out = func(seq)
                print(f"seq={seq} -> Sequence: {[str(x) for x in out]}\n")
            except Exception as e:
                print(f"seq={seq} -> ERROR: {e}")

if __name__ == '__main__':
    test_functions()
