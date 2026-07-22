# Math Task Functions Reference

This document provides a comprehensive specification for all math task candidate functions defined in `math_candidate_test_all.py`. Each section defines the inputs, outputs, conditions, complexity, output growth rate, core evaluation Python code, and LLM prompt.

---

## 0. MatrixMulTask (List → List)

- **Inputs**:
  - `matrix_1` (type: `list` of `int` or `float`): Flattened representation of the first matrix.
  - `matrix_2` (type: `sympy.Matrix`): The second matrix, generated randomly with dimensions matching the inner dimension of `matrix_1`.
  - `num_rows` (type: `int`): Number of rows of `matrix_1`.
  - `num_cols` (type: `int`): Number of columns of `matrix_1`.
- **Conditions**: `num_rows > 0`, `num_cols > 0`.
- **Output Type**: `list` of `int` or `float` (flattened representation of the product matrix).
- **Complexity**: $O(r \cdot c \cdot r) = O(N^{1.5})$ where $r$ is the row dimension and $N = r \cdot c$ is the input list size.
- **Python Evaluation Code**:
  ```python
  matrix_1 = list_to_matrix_compute(seq, num_rows, num_cols)
  out = matrix_1 * matrix_2
  ```
- **Prompt**:
  ```
  Compute the matrix product of matrix_1 and matrix_2.
  ```

---

## 1. ConvolutionTask (List → List)

- **Inputs**:
  - `seq` (type: `list` of `int`): The input sequence.
  - `seq2` (type: `list` of `int`): A randomly generated sequence of the same length as `seq`.
- **Conditions**: Elements are integers.
- **Output Type**: `list` of `int` (linear discrete convolution of the two sequences).
- **Complexity**: $O(N \log N)$ or $O(N^2)$ where $N$ is the sequence length.
- **Python Evaluation Code**:
  ```python
  out = sympy.discrete.convolutions.convolution(seq, seq2)
  ```
- **Prompt**:
  ```
  Compute the linear discrete convolution of seq and seq2.
  ```

---

## 2. PolyRemainderTask (List → List)

- **Inputs**:
  - `seq` (type: `list` of `int`): Polynomial coefficients representing $P(x)$.
  - `seq2` (type: `list` of `int`): Polynomial coefficients representing $Q(x)$ generated deterministically from the second half of `seq`.
- **Conditions**: All values are integers. $Q(x)$ is made monic by prepending 1.
- **Output Type**: `list` of `int` (coefficients of the polynomial remainder).
- **Complexity**: $O(N^2)$ polynomial division.
- **Python Evaluation Code**:
  ```python
  P = sympy.Poly(seq, x)
  Q = sympy.Poly([1] + seq2, x)
  out = sympy.rem(P, Q).all_coeffs()
  ```
- **Prompt**:
  ```
  Compute the polynomial remainder of poly_1 divided by poly_2.
  ```

---

## 3. FWHTTask (List → List)

- **Inputs**:
  - `seq` (type: `list` of `int` or `float`): The input sequence.
- **Conditions**: None.
- **Output Type**: `list` of `int` or `float` (Fast Walsh-Hadamard Transform coefficients).
- **Complexity**: $O(N \log N)$ where $N$ is padded to the next power of 2.
- **Python Evaluation Code**:
  ```python
  padded = pad_to_power_of_2_compute(seq)
  out = list(sympy.discrete.transforms.fwht(padded))
  ```
- **Prompt**:
  ```
  Compute the Fast Walsh-Hadamard Transform (FWHT) of seq.
  ```

---

## 4. NTTTask (List → List)

- **Inputs**:
  - `seq` (type: `list` of `int`): The input sequence.
  - `prime` (type: `int`): The prime modulus $p$ chosen such that $p \equiv 1 \pmod{M}$, where $M$ is the padded power-of-2 length of `seq`.
- **Conditions**: `seq` values rounded to integers towards zero.
- **Output Type**: `list` of `int` (Number Theoretic Transform coefficients).
- **Complexity**: $O(N \log N)$ where $N$ is the padded sequence length.
- **Python Evaluation Code**:
  ```python
  padded = pad_to_power_of_2_compute(seq)
  as_ints = [int(x) for x in padded]
  out = list(sympy.discrete.transforms.ntt(as_ints, prime))
  ```
- **Prompt**:
  ```
  Compute the Number Theoretic Transform (NTT) of seq modulo prime.
  ```

---

## 5. SubsetZetaTransformTask (List → List)

- **Inputs**:
  - `seq` (type: `list` of `int` or `float`): The input sequence.
- **Conditions**: None.
- **Output Type**: `list` of `int` or `float` (Möbius Transform coefficients).
- **Complexity**: $O(N \log N)$ where $N$ is padded to the next power of 2.
- **Python Evaluation Code**:
  ```python
  padded = pad_to_power_of_2_compute(seq)
  out = list(sympy.discrete.transforms.mobius_transform(padded))
  ```
- **Prompt**:
  ```
  Compute the Subset Zeta Transform (subset-sum variant) of seq.
  ```

---

## 6. KroneckerProductTask (List → List)

- **Inputs**:
  - `matrix_1` (type: `list` of `int` or `float`): Flattened representation of the first matrix.
  - `matrix_2` (type: `sympy.Matrix`): Second matrix generated randomly with dimensions matching the inner bounds.
  - `num_rows` (type: `int`): Number of rows of `matrix_1`.
  - `num_cols` (type: `int`): Number of columns of `matrix_1`.
- **Conditions**: `num_rows > 0`, `num_cols > 0`.
- **Output Type**: `list` of `int` or `float` (flattened Kronecker product matrix).
- **Complexity**: $O(r^2 \cdot c^2) = O(N^2)$ where $N = r \cdot c$ is the input list size.
- **Python Evaluation Code**:
  ```python
  m1 = list_to_matrix_compute(seq, num_rows, num_cols)
  out = sympy.matrices.kronecker_product(m1, matrix_2)
  ```
- **Prompt**:
  ```
  Compute the Kronecker product of matrix_1 and matrix_2.
  ```

---

## 7. PolyQuotientTask (List → List)

- **Inputs**:
  - `seq` (type: `list` of `int` or `float`): Coefficients of $P(x)$.
- **Conditions**: None.
- **Output Type**: `list` of `int` or `float` (coefficients of quotient).
- **Complexity**: $O(N)$ polynomial division.
- **Python Evaluation Code**:
  ```python
  import sympy
  if not seq:
      out = []
  else:
      x = sympy.Symbol('x')
      p1 = sympy.Poly(seq, x)
      p2 = sympy.Poly([1, 1, 1], x)
      q, r = sympy.div(p1, p2)
      out = q.all_coeffs() if hasattr(q, 'all_coeffs') else []
  ```
- **Prompt**:
  ```
  Treat the input list as the coefficients of a polynomial P(x) ordered from highest to lowest degree. Compute the polynomial quotient when P(x) is divided by the monic polynomial x^2 + x + 1, and return the coefficients of the quotient.
  ```

---

## 8. MatrixPolynomialTask (List → List)

- **Inputs**:
  - `matrix_1` (type: `list` of `int`): Flattened square matrix representation.
  - `num_rows` (type: `int`): Size of the square matrix.
- **Conditions**: Square matrix (`num_rows = num_cols > 0`), elements are integers.
- **Output Type**: `list` of `int` (flattened polynomial matrix $M^3 + M^2 + M$).
- **Complexity**: $O(k^3)$ matrix multiplications where $k = \text{num\_rows}$.
- **Python Evaluation Code**:
  ```python
  m = list_to_matrix_compute(seq, num_rows, num_rows)
  out = m**3 + m**2 + m
  ```
- **Prompt**:
  ```
  Compute the matrix polynomial matrix_1^3 + matrix_1^2 + matrix_1.
  ```

---

## 9. SortByDivisorCountTask (List → List)

- **Inputs**:
  - `seq` (type: `list` of `int`): Sequence of integers.
- **Conditions**: None.
- **Output Type**: `list` of `int` (sorted sequence).
- **Complexity**: $O(N \sqrt{\max(seq)} + N \log N)$ factorization and sorting.
- **Python Evaluation Code**:
  ```python
  import sympy
  out = sorted(seq, key=lambda x: sympy.ntheory.factor_.divisor_count(abs(x)) if x != 0 else 0)
  ```
- **Prompt**:
  ```
  Sort the input list of integers based on the number of divisors of their absolute values in ascending order. If an element is 0, treat its divisor count as 0. Maintain stable sorting for ties.
  ```

---

## 10. MatrixFrobeniusSqTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int` or `float`): Flattened square matrix representation.
- **Conditions**: Sequence length $N$ is a perfect square $k^2$ ($k \ge 1$).
- **Output Type**: `int` (Frobenius norm squared of the matrix).
- **Output Growth**: $O(k \cdot \max(|s_i|)^2)$ — linear in matrix size, quadratic in element magnitude. For inputs in $[-100, 100]$ and $k=8$: max ~$640{,}000$.
- **Complexity**: $O(k^2)$ element-wise squaring and summation.
- **Python Evaluation Code**:
  ```python
  k = math.isqrt(len(seq))
  mat = list_to_matrix_compute(seq, k, k)
  out = sum(mat[i, j]**2 for i in range(k) for j in range(k))
  ```
- **Prompt**:
  ```
  Compute the Frobenius norm squared (sum of squares of all elements) of the given matrix.
  ```

---

## 11. ConvexHullAreaTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Sequence of integers representing y-coordinates.
- **Conditions**: Length $N \ge 3$.
- **Output Type**: `int` (twice the area of the 2D convex hull).
- **Complexity**: $O(N \log N)$ using Graham scan algorithm.
- **Python Evaluation Code**:
  ```python
  import sympy
  pts = [sympy.geometry.Point(i, val) for i, val in enumerate(seq)]
  hull = sympy.geometry.convex_hull(*pts)
  out = 2 * hull.area if hasattr(hull, 'area') else 0
  ```
- **Prompt**:
  ```
  Consider the input list as a sequence of y-coordinates with x-coordinates being their indices (0, 1, 2, ...). Compute the area of the convex hull of these 2D points, multiply the area by 2, and return it as an integer.
  ```

---

## 12. MatrixNorm1Task (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Flattened square matrix representation.
- **Conditions**: Sequence length $N$ is a perfect square $k^2$ ($k \ge 1$).
- **Output Type**: `int` (maximum absolute column sum of the matrix).
- **Complexity**: $O(k^2)$ column operations where $k = \sqrt{N}$.
- **Python Evaluation Code**:
  ```python
  mat = sympy.Matrix(k, k, seq)
  out = max(sum(abs(mat[r, c]) for r in range(k)) for c in range(k))
  ```
- **Prompt**:
  ```
  Compute the 1-norm of matrix_1 (the maximum absolute column sum).
  ```

---

## 13. MatrixTraceCubeTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Input list of integers.
- **Conditions**: None.
- **Output Type**: `int` (trace of the cubed matrix).
- **Complexity**: $O(k^3)$ where $k = \lfloor \sqrt{N} \rfloor$.
- **Python Evaluation Code**:
  ```python
  import sympy
  k = int(len(seq)**0.5)
  if k == 0:
      out = 0
  else:
      mat = sympy.Matrix(k, k, seq[:k*k])
      out = (mat**3).trace()
  ```
- **Prompt**:
  ```
  Construct a square matrix M of size k x k from the first k^2 elements of the list, where k is the integer square root of the list length. Compute the trace of M^3.
  ```

---

## 14. PermutationInversionsTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Input list of integers.
- **Conditions**: None.
- **Output Type**: `int` (number of inversions in the permutation).
- **Complexity**: $O(N \log N)$ sorting and inversion counting.
- **Python Evaluation Code**:
  ```python
  from sympy.combinatorics import Permutation
  if not seq:
      out = 0
  else:
      indices = sorted(range(len(seq)), key=lambda x: seq[x])
      out = Permutation(indices).inversions()
  ```
- **Prompt**:
  ```
  Sort the indices (0 to len-1) of the input list based on their corresponding values to form a permutation. Return the exact number of inversions in this permutation.
  ```

---

## 15. PermutationOrderTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Sequence of integer values.
- **Conditions**: None.
- **Output Type**: `int` (the order of the resulting permutation).
- **Complexity**: $O(N \log N)$ to find the permutation order.
- **Python Evaluation Code**:
  ```python
  from sympy.combinatorics import Permutation
  indices = sorted(range(len(seq)), key=lambda x: seq[x])
  out = Permutation(indices).order() if seq else 1
  ```
- **Prompt**:
  ```
  Sort the indices (0 to N-1) of the input list based on their corresponding values to form a permutation. Return the order of this permutation (the smallest positive integer m such that applying the permutation m times yields the identity).
  ```

---

## 16. PolySquareFreeEvalTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Polynomial coefficients representing $P(x)$.
- **Conditions**: Non-empty sequence.
- **Output Type**: `int` (the square-free part evaluated at $x = 1$).
- **Complexity**: $O(N^2)$ polynomial GCD computations.
- **Python Evaluation Code**:
  ```python
  import sympy
  if not seq:
      out = 0
  else:
      x = sympy.Symbol('x')
      poly = sympy.Poly(seq, x)
      out = sympy.polys.polytools.sqf_part(poly).subs(x, 1)
  ```
- **Prompt**:
  ```
  Treat the input list as the coefficients of a polynomial P(x) ordered from highest to lowest degree. Compute the square-free part of P(x) and evaluate it at x=1.
  ```

---

## 17. PolyResultantFixedTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Polynomial coefficients representing $P(x)$.
- **Conditions**: Non-empty sequence.
- **Output Type**: `int` (resultant with the fixed quadratic $x^2 + x + 1$).
- **Complexity**: $O(N^2)$ Sylvester determinant reduction where $N$ is polynomial degree.
- **Python Evaluation Code**:
  ```python
  P = sympy.Poly(seq, x)
  Q = sympy.Poly(x**2 + x + 1, x)
  out = sympy.resultant(P, Q)
  ```
- **Prompt**:
  ```
  Compute the resultant of poly_1 and the polynomial x^2 + x + 1.
  ```

---

## 18. InterpolateEvalTask (List → Scalar)

- **Inputs**:
  - `points` (type: `list` of `int`): The y-coordinates corresponding to x-coordinates $[0, 1, \dots, L-1]$.
  - `eval_point` (type: `int`): Evaluation point base $x = k$.
- **Conditions**: Sequence must be non-empty. Unique interpolating polynomial always exists since x-coordinates are distinct.
- **Output Type**: `int` (interpolated polynomial evaluated at a point).
- **Complexity**: $O(N^2)$ Lagrange interpolation where $N$ is sequence length.
- **Python Evaluation Code**:
  ```python
  k_len = len(seq) // 8 + 3
  sub_seq = seq[:k_len]
  points_coords = [(i, val) for i, val in enumerate(sub_seq)]
  poly = sympy.interpolate(points_coords, x)
  out = poly.subs(x, k_len + eval_point)
  ```
- **Prompt**:
  ```
  Take the first L elements of points, where L is floor(len/8) + 3. Construct the unique interpolating polynomial through the coordinate points (0, sub_seq[0]), (1, sub_seq[1]), ..., and evaluate it at L + eval_point.
  ```

---

## 19. PairwiseGcdSumTask (List → Scalar)

- **Inputs**:
  - `seq` (type: `list` of `int`): Sequence of integers.
- **Conditions**: Length $N \ge 2$.
- **Output Type**: `int` (sum of GCDs of all pairs).
- **Output Growth**: $O\bigl(\binom{N}{2} \cdot \max|s_i|\bigr)$ in the worst case, but typically much smaller. For inputs in $[-100, 100]$ and $N=64$: max ~$6{,}000$.
- **Complexity**: $O(N^2 \log(\max|s_i|))$ pairwise GCD computations.
- **Python Evaluation Code**:
  ```python
  if len(seq) < 2:
      out = 0
  else:
      total = 0
      for i in range(len(seq)):
          for j in range(i+1, len(seq)):
              total += sympy.gcd(abs(seq[i]), abs(seq[j]))
      out = int(total)
  ```
- **Prompt**:
  ```
  For every pair of elements (i, j) where i < j in the input list, compute gcd(|seq[i]|, |seq[j]|). Return the sum of all these GCD values.
  ```

---

## 20. PrimeOmegaSumTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Upper bound integer.
- **Conditions**: $n \ge 2$.
- **Output Type**: `int` (cumulative count of distinct prime factors).
- **Output Growth**: $\Theta(n \log \log n)$ — sublinear per element on average. For $n=250$: output is $473$.
- **Complexity**: $O(n \sqrt{n})$ factorization of each integer $2$ through $n$.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 2:
      out = 0
  else:
      out = sum(sympy.factorint(i).__len__() for i in range(2, n + 1))
  ```
- **Prompt**:
  ```
  For each integer i from 2 to n, compute omega(i), the number of distinct prime factors of i. Return the sum of all these omega values.
  ```

---

## 21. SumDivisorCountsTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Upper bound integer.
- **Conditions**: $n \ge 1$.
- **Output Type**: `int` (cumulative divisor count sum).
- **Output Growth**: $\Theta(n \ln n)$ — the Dirichlet divisor sum. For $n=250$: output is $1{,}421$.
- **Complexity**: $O(n \sqrt{n})$ factorization of each integer.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 1:
      out = 0
  else:
      out = sum(int(sympy.divisor_count(i)) for i in range(1, n + 1))
  ```
- **Prompt**:
  ```
  For each integer i from 1 to n, compute the number of positive divisors of i. Return the sum of all these divisor counts.
  ```

---

## 22. SummatoryTotientTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Upper bound integer.
- **Conditions**: $n \ge 1$.
- **Output Type**: `int` (summatory Euler totient value).
- **Output Growth**: Approximately $\Theta(n^2)$, with moderate values under the current scalar cap.
- **Complexity**: $O(n \sqrt{n})$ totient computations.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  out = sum(int(sympy.totient(k)) for k in range(1, n + 1))
  ```
- **Prompt**:
  ```
  For each integer k = 1, 2, ..., n, compute Euler's totient function phi(k), the number of integers m with 1 <= m <= k such that gcd(m, k) = 1. Compute the sum phi(1) + phi(2) + ... + phi(n).
  ```

---

## 23. PrimitiveRootCountTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Index of prime.
- **Conditions**: $n \ge 1$.
- **Output Type**: `int` (number of primitive roots modulo the $n$-th prime).
- **Output Growth**: $\Theta(n \log n)$ — since $p_n \sim n \ln n$ and $\phi(p_n) = p_n - 1$. For $n=250$: output is $1{,}582$.
- **Complexity**: $O(n \log n)$ prime generation + $O(\sqrt{p_n})$ factorization.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 1:
      out = 0
  else:
      p = sympy.prime(n)
      out = sympy.totient(p)
  ```
- **Prompt**:
  ```
  Find the n-th prime number p. Compute Euler's totient phi(p), which equals the count of primitive roots modulo p.
  ```

---

## 24. LegendreResidueWeightTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Upper bound integer.
- **Conditions**: $n \ge 1$.
- **Output Type**: `int` (weighted quadratic-residue sum).
- **Output Growth**: $O(n^2)$, bounded by $n(n+1)$.
- **Complexity**: $O(n \log n)$ Legendre symbol evaluations.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  p = int(sympy.nextprime(n + 2))
  out = sum(a * (int(sympy.legendre_symbol(a, p)) + 1) for a in range(1, n + 1))
  ```
- **Prompt**:
  ```
  Let p be the smallest prime number strictly greater than n + 2. For each integer a = 1, 2, ..., n, compute the Legendre symbol Legendre(a, p). This value is 1 if there exists an integer x such that x**2 is congruent to a modulo p, and -1 otherwise. For each a, compute a * (Legendre(a, p) + 1). Return the sum of these values.
  ```

---

## 25. ScaledStirlingFirstKindTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Base sequence index.
- **Conditions**: $n \ge 0$.
- **Output Type**: `int` (scaled near-diagonal Stirling-style count).
- **Output Growth**: $O((\sqrt{n})^4) = O(n^2)$.
- **Complexity**: $O(1)$ arithmetic after computing the integer square root.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  m = math.isqrt(n) + 7
  out = 2 * sympy.binomial(m, 3) + 3 * sympy.binomial(m, 4)
  ```
- **Prompt**:
  ```
  Let m be the integer square root of n, plus 7. Compute 2*C(m, 3) + 3*C(m, 4), where C(a, b) denotes the binomial coefficient "a choose b".
  ```

---

## 26. DivisorSigmaSquareTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Modulus coordinate/value.
- **Conditions**: $n \ge 1$.
- **Output Type**: `int` (sum of the squares of all positive divisors).
- **Output Growth**: $O(n^2)$ — bounded strictly. For $n=250$: output is $81{,}380$.
- **Complexity**: $O(\sqrt{n})$ trial division divisor check.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  out = int(sympy.divisor_sigma(n, 2))
  ```
- **Prompt**:
  ```
  Compute the sum of the squares of all positive divisors of n.
  ```

---

## 28. DivisorSumTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Positive integer.
- **Conditions**: $n > 0$.
- **Output Type**: `int` (sum of all positive divisors $\sigma_1(n)$).
- **Complexity**: $O(\sqrt{n})$ factorization.
- **Python Evaluation Code**:
  ```python
  out = sympy.divisor_sigma(n, 1)
  ```
- **Prompt**:
  ```
  Compute the sum of all positive divisors of n.
  ```

---

## 29. MultiplicativeOrderTask (Scalar → Scalar)

- **Inputs**:
  - `n` (type: `int`): Modulus coordinate.
- **Conditions**: $n > 0$.
- **Output Type**: `int` (multiplicative order of 2 modulo $2n+1$).
- **Complexity**: $O(\sqrt{2n+1})$ order check.
- **Python Evaluation Code**:
  ```python
  out = sympy.ntheory.n_order(2, 2*n + 1)
  ```
- **Prompt**:
  ```
  Compute the multiplicative order of 2 modulo (2n + 1).
  ```

---

## 30. ModularMultiplicationSequenceTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Sequence parameter.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (modular product terms).
- **Output Length**: Exactly $n$ elements.
- **Output Growth**: Elements bounded by $p_n \sim n \ln n$. For $n=250$: max element is $1{,}512$.
- **Complexity**: $O(n \log n \log\log n)$ prime sieve + $O(n)$ modular multiplications.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 1:
      out = [0]
  else:
      p = sympy.prime(n)
      out = [int((i * n) % p) for i in range(1, n + 1)]
  ```
- **Prompt**:
  ```
  For each integer i from 1 to n, compute (i * n) modulo the n-th prime number. Return the sequence of these values.
  ```

---

## 31. PrimePiQuadraticSequenceTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Base upper bound.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (prime-counting sequence).
- **Output Length**: Exactly $n$ elements.
- **Output Growth**: Approximately $O(n^2 / \log n)$ per element.
- **Complexity**: Depends on SymPy's prime-counting implementation for each of the $n$ thresholds.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  out = [int(sympy.primepi(n**2 + i**2)) for i in range(1, n + 1)]
  ```
- **Prompt**:
  ```
  For each integer i = 1, 2, ..., n, define A_i to be the number of prime numbers q such that q <= n**2 + i**2. Return the list [A_1, A_2, ..., A_n] in increasing order of i, so the first element corresponds to i = 1, the second element corresponds to i = 2, and so on.
  ```

---

## 32. GrayCodeSequenceTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Sequence parameter.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (Gray code integer values).
- **Output Length**: Exactly $n$ elements.
- **Output Growth**: Elements grow linearly, bounded by $2n$. For $n=250$: max element is $511$.
- **Complexity**: $O(n)$ Gray code transformations.
- **Python Evaluation Code**:
  ```python
  from sympy.combinatorics.graycode import bin_to_gray
  n = abs(n)
  if n < 1:
      out = [0]
  else:
      out = [int(bin_to_gray(bin(i + n)[2:]), 2) for i in range(1, n + 1)]
  ```
- **Prompt**:
  ```
  For each integer i from 1 to n, compute the Gray code representation of i + n, and return the sequence of these values as integers.
  ```

---

## 33. TotientShiftSequenceTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Base shift parameter.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (shifted totient values).
- **Output Length**: Exactly $n$ elements.
- **Output Growth**: Elements bounded by $2n$. For $n=250$: max element is $498$.
- **Complexity**: $O(n \sqrt{n})$ totient computations.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 1:
      out = [0]
  else:
      out = [int(sympy.totient(i + n)) for i in range(1, n + 1)]
  ```
- **Prompt**:
  ```
  For each integer i from 1 to n, compute Euler's totient function phi(i + n). Return the sequence of these values.
  ```

---

## 34. SmallestFactorQuadraticTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Shift parameter.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (smallest prime factor of shifted squares).
- **Output Length**: Exactly $n$ elements.
- **Output Growth**: Individual elements are bounded by $n^2 + n$. For $n=250$: max element is $54{,}539$.
- **Complexity**: $O(n \sqrt{n^2 + n})$ factorization.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 1:
      out = [0]
  else:
      out = [int(min(sympy.factorint(i**2 + n).keys())) if (i**2 + n) > 1 else 1 for i in range(1, n + 1)]
  ```
- **Prompt**:
  ```
  For each integer i from 1 to n, compute the smallest prime factor of i^2 + n. Return the sequence of these values.
  ```

---

## 35. TotientProductSequenceTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Modulus/multiplier coordinate.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (Euler totient values).
- **Output Length**: Exactly $n$ elements.
- **Output Growth**: Elements scale quadratically, bounded by $n^2$. For $n=250$: max element is $62{,}500$.
- **Complexity**: $O(n \sqrt{n})$ totient evaluations.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 1:
      out = [0]
  else:
      out = [int(sympy.totient(i * n)) for i in range(1, n + 1)]
  ```
- **Prompt**:
  ```
  For each integer i from 1 to n, compute Euler's totient function of i * n. Return the sequence of these values.
  ```

---

## 36. PolygonDiagonalsSequenceTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Base vertex offset coordinate.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (diagonal counts).
- **Output Length**: Exactly $n$ elements.
- **Output Growth**: Elements scale quadratically, bounded by $2n^2$. For $n=250$: max element is $124{,}250$.
- **Complexity**: $O(n)$ arithmetic operations.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  if n < 1:
      out = [0]
  else:
      out = [int((i + n) * (i + n - 3) // 2) for i in range(1, n + 1)]
  ```
- **Prompt**:
  ```
  For each integer i from 1 to n, compute the number of diagonals in a convex polygon with i + n vertices. Return the sequence of these values.
  ```

---

## 37. PrimesBetweenSquaresTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Base square coordinate.
- **Conditions**: $n > 0$.
- **Output Type**: `list` of `int` (sequence of prime numbers).
- **Complexity**: $O(n \log n)$ prime sieve over interval.
- **Python Evaluation Code**:
  ```python
  out = list(sympy.primerange(n**2 + 1, (n+1)**2))
  ```
- **Prompt**:
  ```
  List all prime numbers strictly between n^2 and (n+1)^2 in increasing order.
  ```

---

## 38. CentralBinomialPrimeValuationVectorTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Base integer.
- **Conditions**: $n \ge 1$.
- **Output Type**: `list` of `int` (encoded prime valuations).
- **Output Length**: One element for each prime $p \le 2n$.
- **Output Growth**: $O(n)$ per element because each value is $p(e_p + 1)$ with $p \le 2n$.
- **Complexity**: $O(\pi(2n)\log n)$ valuation computation.
- **Python Evaluation Code**:
  ```python
  n = abs(n)
  out = []
  for p in sympy.primerange(2, 2*n + 1):
      p = int(p)
      exponent = 0
      power = p
      while power <= 2*n:
          exponent += (2*n) // power - 2 * (n // power)
          power *= p
      out.append(int(p * (exponent + 1)))
  ```
- **Prompt**:
  ```
  For each prime number p with p <= 2*n, define e_p as follows. For each positive integer j such that p**j <= 2*n, add floor(2*n / p**j) - 2*floor(n / p**j) to e_p. Then compute p * (e_p + 1). Return the list of these values in increasing order of p.
  ```

---

## 39. QuadraticResiduesPrimeTask (Scalar → List)

- **Inputs**:
  - `n` (type: `int`): Index of prime number.
- **Conditions**: $n > 0$.
- **Output Type**: `list` of `int` (quadratic residues modulo prime).
- **Complexity**: $O(p_n) = O(n \log n)$ residue evaluation.
- **Python Evaluation Code**:
  ```python
  import sympy
  out = sympy.ntheory.quadratic_residues(sympy.prime(n)) if n > 0 else []
  ```
- **Prompt**:
  ```
  Find all quadratic residues modulo the n-th prime number and return them as a sorted list.
  ```
