# Math Forgetting Dataset Naming Discrepancies and Conflicts Analysis

This report provides a detailed analysis of variable naming discrepancies and name conflicts within individual samples of `math_forgetting.txt`.

## 1. Unused Suffixed Variables (Discrepancies)
A major structural discrepancy exists where a task-specific variable is formally defined with the task suffix (e.g., `p_XX`), but the mathematical operations in the rest of the task description refer to a generic, suffixless variable (e.g., `p`).

**Total occurrences found:** 66

| Variable Pattern | Total Count | Example Location |
| --- | --- | --- |
| `p_XX` | 66 | Sample 7, Task 03 (`p_03`) |

### Detailed Analysis of NTTTask
In `NTTTask`, the prime modulus variable is defined as `p_XX` but used as `p`. For instance:
```
Define p_03 = 17.
...
Let list_03 = [a_0, a_1, ..., a_{N-1}], and let p = 17.
All arithmetic in this transform is modulo p. Let g be the smallest positive primitive root modulo p...
```
This means `p_03` is left unused, and the local name `p` is introduced, violating the expected convention of suffixing task variables to avoid global scope contamination.

## 2. Suffixless Variable Redeclaration and Scope Collisions (Conflicts)
Because temporary/local math variables (such as `p`, `m`, `pi`, `N`, `x`, `g`, `omega`) are not suffixed with the task number, they are reused across different tasks within the same sample. Since the model receives the entire chain of tasks in a single prompt context, this leads to namespace collisions.

**Total samples with suffixless variable collisions:** 130 / 640 (20.3%)

### Frequency of Suffixless Variable Conflicts
| Suffixless Variable | Number of Samples Affected | Conflict Description |
| --- | --- | --- |
| `x` | 71 | Reused as the polynomial variable in polynomial division, square-free evaluation, and Gray code calculation. |
| `N` | 26 | Reused as the size of lists/permutations across multiple tasks (e.g. Task 01 and Task 03). |
| `p` | 21 | Reused as prime numbers/moduli with different values (e.g. Legendre residues in Task 01 and NTT modulo in Task 03). |
| `m` | 18 | Reused as list log-length power-of-2 ($2^m$) and permutation order ($m$). |
| `pi` | 1 | Reused as a permutation variable in both Permutation Order and Permutation Inversion tasks. |
| `g` | 2 | Reused as the primitive root in different NTT tasks. |
| `omega` | 2 | Reused as roots of unity/omega functions in NTT and divisor tasks. |

### Example Collision: Variable `p` in Sample 7
- **Task 01**: `For each prime number p with p <= 2*val_01, define e_p as follows...` (Here `p` is a running variable over prime numbers)
- **Task 03**: `and let p = 17. All arithmetic in this transform is modulo p...` (Here `p` is a fixed scalar constant prime 17)

### Example Collision: Variable `m` in Sample 5
- **Task 01**: `Let list_01 be a sequence of length 2^m. Index its entries by bitmasks S = 0, 1, ..., 2^m - 1.` (Here `m` is the power of 2)
- **Task 03**: `Compute the order of this permutation: the smallest positive integer m such that...` (Here `m` is the permutation order)

## 3. Summary and Impact
1. **NTTTask `p_XX` Discrepancy:** This is a clear bug in the prompt generator template. `Define p_XX` is generated, but then `let p = {prime}` is used. This leaves `p_XX` defined but unused.
2. **Suffixless Variable Conflicts:** Due to the absence of task suffixes on local/expression variables, variables like `p`, `m`, `pi`, `N`, and `x` are redefined across tasks. While logically distinct within their respective task descriptions, their co-occurrence in a single prompt context creates namespace overlap, which could degrade model performance in context-tracking/forgetting tasks.
