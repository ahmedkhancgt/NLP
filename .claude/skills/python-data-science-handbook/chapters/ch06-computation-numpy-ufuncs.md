# Chapter 6: Computation on NumPy Arrays: Universal Functions

## Core Idea
Python loops are slow because of per-element dynamic type checking and dispatch; NumPy's universal functions (ufuncs) push the loop into compiled C code, making vectorized array operations orders of magnitude faster.

## Frameworks Introduced
- **Vectorization via ufuncs**: replace explicit Python loops over array elements with direct arithmetic/function calls on whole arrays.
  - When to use: any time you see a Python `for` loop touching array elements one at a time.
  - How: use operators (`+ - * / // ** %`) or `np.<func>` directly on arrays — NumPy dispatches to compiled, statically-typed code; scalar-array, array-array, and multidimensional operations are all supported.
- **ufunc.reduce / ufunc.accumulate / ufunc.outer**: generalized aggregation/accumulation/pairwise-combination built into every ufunc.
  - How: `np.add.reduce(x)` sums all elements; `np.add.accumulate(x)` returns running totals; `np.multiply.outer(x, x)` builds all-pairs product table.
- **`out=` argument**: write ufunc results directly into a preallocated array (or view) to avoid temporary-array memory overhead.

## Key Concepts
- **Ufunc**: a NumPy function that applies element-wise, implemented in compiled code, supporting broadcasting; unary (one input, e.g., `np.abs`) or binary (two inputs, e.g., `np.add`).
- **Vectorized operation**: an operation expressed over whole arrays rather than via explicit iteration.
- **Type dispatch overhead**: the per-iteration cost CPython pays to determine which implementation to call for a value's dynamic type — the root cause of Python loop slowness.
- **`scipy.special`**: module of additional specialized ufuncs (gamma, beta, error functions, etc.) for statistics/math.
- **Floor division (`//`), modulus (`%`)**: ufunc-backed operators, same as `np.floor_divide`, `np.mod`.
- **`np.absolute`/`np.abs`**: handles complex input by returning magnitude.

## Mental Models
- "Any time you see a loop operating element-by-element on a NumPy array, ask whether a vectorized expression can replace it" — this is the single most important NumPy performance heuristic.
- Think of a ufunc as a compiled "dispatch table" fixed at the array's dtype — the type-checking cost is paid once per array call, not once per element.

## Anti-patterns
- **Writing explicit Python `for` loops to transform array elements (e.g., manual reciprocal computation)**: ~1000x slower than the vectorized equivalent; replace with ufunc expressions.
- **Using `y[::2] = 2 ** x` instead of `np.power(2, x, out=y[::2])` for large arrays**: the former allocates a temporary array before copying; `out=` avoids that extra allocation.
- **Confusing `sum()` (Python builtin) with `np.sum()`**: their optional-argument semantics differ (addressed further in Ch 7), and the builtin is much slower on NumPy arrays.

## Code Examples
```python
import numpy as np
rng = np.random.default_rng(seed=1701)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = rng.integers(1, 10, size=5)
compute_reciprocals(values)      # slow Python-loop version

big_array = rng.integers(1, 100, size=1000000)
%timeit compute_reciprocals(big_array)   # ~2.61 s
%timeit (1.0 / big_array)                # ~2.54 ms  -- vectorized, ~1000x faster

np.arange(5) / np.arange(1, 6)           # array-array ufunc
x = np.arange(9).reshape((3, 3)); 2 ** x # ufunc on multidimensional array

x = np.arange(4)
x + 5; x - 5; x * 2; x / 2; x // 2
-x; x ** 2; x % 2
-(0.5*x + 1) ** 2                         # chained, respects order of operations
np.add(x, 2)                              # explicit ufunc form of `+`

x = np.array([-2, -1, 0, 1, 2]); abs(x); np.absolute(x); np.abs(x)
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j]); np.abs(x)  # magnitude of complex

theta = np.linspace(0, np.pi, 3)
np.sin(theta); np.cos(theta); np.tan(theta)
np.arcsin([-1, 0, 1]); np.arccos([-1, 0, 1]); np.arctan([-1, 0, 1])

x = [1, 2, 3]
np.exp(x); np.exp2(x); np.power(3., x)
np.log([1, 2, 4, 10]); np.log2([1, 2, 4, 10]); np.log10([1, 2, 4, 10])
np.expm1([0, 0.001, 0.01, 0.1]); np.log1p([0, 0.001, 0.01, 0.1])  # precision near 0

from scipy import special
special.gamma([1, 5, 10]); special.gammaln([1, 5, 10]); special.beta([1, 5, 10], 2)
special.erf(np.array([0, 0.3, 0.7, 1.0]))

x = np.arange(5); y = np.empty(5)
np.multiply(x, 10, out=y)

y = np.zeros(10)
np.power(2, x, out=y[::2])    # write into every-other element, no temp array

x = np.arange(1, 6)
np.add.reduce(x)              # 15 (sum)
np.multiply.reduce(x)         # 120 (product)
np.add.accumulate(x)          # [1,3,6,10,15]
np.multiply.accumulate(x)     # [1,2,6,24,120]
np.multiply.outer(x, x)       # 5x5 multiplication table
```
- **What it demonstrates**: the loop-vs-vectorized speed contrast, the breadth of arithmetic/trig/exp/log ufuncs, `scipy.special` extensions, and advanced ufunc methods (`out`, `reduce`, `accumulate`, `outer`).

## Reference Tables
| Operator | Equivalent ufunc | Description |
|---|---|---|
| `+` | `np.add` | Addition |
| `-` (binary) | `np.subtract` | Subtraction |
| `-` (unary) | `np.negative` | Negation |
| `*` | `np.multiply` | Multiplication |
| `/` | `np.divide` | Division |
| `//` | `np.floor_divide` | Floor division |
| `**` | `np.power` | Exponentiation |
| `%` | `np.mod` | Modulus |

## Worked Example
Benchmark: `compute_reciprocals` (manual Python loop) over 1,000,000 elements takes ~2.61s; `1.0 / big_array` (vectorized ufunc) takes ~2.54ms — about 1000x faster. The bottleneck is per-element type dispatch, not the arithmetic itself; vectorization pushes the loop into compiled code and eliminates that overhead.

## Key Takeaways
1. Python loops over array elements are slow mainly due to per-iteration type checking/dispatch, not the arithmetic itself.
2. Vectorized ufunc expressions are typically orders of magnitude faster and should replace explicit loops whenever possible.
3. Ufuncs support scalar-array, array-array, and multidimensional operations uniformly.
4. Standard arithmetic operators are convenience wrappers around named ufuncs (`+` → `np.add`, etc.).
5. `scipy.special` extends NumPy's ufunc library with gamma/beta/error functions and more.
6. `out=` avoids temporary-array allocation for large computations; `reduce`/`accumulate`/`outer` generalize aggregation/running-total/pairwise-combination to any ufunc.

## Connects To
- **Ch 3**: `%timeit` benchmarking technique used throughout.
- **Ch 7**: dedicated aggregation functions (`np.sum`, `np.prod`, etc.) are specializations of `ufunc.reduce`.
- **Ch 8**: broadcasting (operating between differently shaped arrays) is the next major ufunc capability.
- **Ch 16**: NaN-safe aggregate variants relate to missing-data handling.
