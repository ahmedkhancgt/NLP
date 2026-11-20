# Chapter 10: Fancy Indexing

## Core Idea
Fancy indexing — passing arrays of indices rather than scalars/slices — allows fast, flexible access and modification of complicated array subsets, with result shape determined by the (broadcast) shape of the index arrays, not the indexed array.

## Frameworks Introduced
- **Fancy indexing**: `x[index_array]` returns elements at those positions; shape of result follows the (broadcast) shape of the index array(s).
  - When to use: selecting/modifying multiple, non-contiguous elements; random sampling of rows; building custom histograms/binning.
  - How: pass a list/array of indices in place of a scalar; combine row/column index arrays for 2D selection (follows broadcasting rules from Ch 8); combine with slices and Boolean masks freely.
- **`ufunc.at(array, indices, value)`**: applies an operation in-place at given indices, correctly handling repeated indices (unlike `x[i] += 1`, which does NOT accumulate for repeats).

## Key Concepts
- **Index array shape propagation**: `x[ind]` result shape = shape of `ind`, not shape of `x`.
- **Combined indexing**: fancy + scalar, fancy + slice, fancy + mask can all be mixed in one indexing expression.
- **Repeated-index assignment pitfall**: `x[[0,0]] = [4,6]` keeps only the last write (`6`); `x[i] += 1` with repeated `i` does NOT add once per repetition (it's shorthand for `x[i] = x[i] + 1`, a single assignment).
- **`np.add.at`**: the correct way to accumulate repeated-index updates in-place.
- **`np.random.choice(n, k, replace=False)`**: select k random indices without repeats — used for random row sampling and train/test splitting.
- **`np.searchsorted`**: finds insertion/bin indices for values against a sorted array — basis of manual histogram binning.

## Mental Models
- Fancy indexing is "indexing by a mapping," not "indexing by a window" — the output's shape is determined entirely by the index arrays, decoupled from the source array's shape.
- `x[i] += 1` is sugar for `x[i] = x[i] + 1`: the right-hand side is computed once before any writes, so repeated indices in `i` "overwrite," they don't "accumulate." Use `ufunc.at` whenever you want true accumulation.

## Anti-patterns
- **Expecting `x[i] += 1` to accumulate for repeated indices in `i`**: it performs a single read-then-write per repeat occurrence of each index, so later repeats overwrite earlier (no accumulation) — use `np.add.at(x, i, 1)` instead.
- **Manually writing a custom histogram loop as the default choice**: `np.histogram` is more general/robust for large datasets even if a hand-rolled `np.add.at` version can be faster for small ones — benchmark before choosing.
- **Assuming result shape mirrors indexed array**: result shape mirrors the (broadcast) index array shape — easy source of confusion with multidimensional fancy indices.

## Code Examples
```python
import numpy as np
rng = np.random.default_rng(seed=1701)
x = rng.integers(100, size=10)

[x[3], x[7], x[2]]            # manual multi-element access
ind = [3, 7, 4]; x[ind]        # fancy indexing, same elements

ind = np.array([[3, 7], [4, 5]])
x[ind]                          # result shape (2,2), matches ind's shape

X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2]); col = np.array([2, 1, 3])
X[row, col]                     # [X[0,2], X[1,1], X[2,3]] = [2, 5, 11]
X[row[:, np.newaxis], col]      # broadcasting -> 2D result

X[2, [2, 0, 1]]                  # fancy + scalar
X[1:, [2, 0, 1]]                  # fancy + slice
mask = np.array([True, False, True, False])
X[row[:, np.newaxis], mask]       # fancy + mask

# random row selection
mean = [0, 0]; cov = [[1, 2], [2, 5]]
X = rng.multivariate_normal(mean, cov, 100)
indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices]            # (20, 2)

# modifying with fancy indexing
x = np.arange(10); i = np.array([2, 1, 8, 4])
x[i] = 99
x[i] -= 10

x = np.zeros(10); x[[0, 0]] = [4, 6]     # x[0] == 6, not 4
i = [2, 3, 3, 4, 4, 4]
x[i] += 1                                  # does NOT accumulate per repeat

x = np.zeros(10)
np.add.at(x, i, 1)                         # correctly accumulates: [0,0,1,2,3,0,...]

# manual histogram binning
x = rng.normal(size=100)
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)
i = np.searchsorted(bins, x)
np.add.at(counts, i, 1)
```
- **What it demonstrates**: multi-index access, broadcast-shaped fancy indexing, combined indexing modes, correct vs. incorrect repeated-index accumulation, and a manual histogram built from `searchsorted` + `ufunc.at`.

## Reference Tables
| Technique | Purpose |
|---|---|
| `x[ind]` (array `ind`) | Fancy indexing, result shape = `ind`'s shape |
| `x[i] = val` | Simple assignment (last write wins on repeats) |
| `x[i] += val` | NOT accumulating on repeated `i` (single read-then-write) |
| `np.add.at(x, i, val)` | True in-place accumulation at repeated indices |
| `np.random.choice(n, k, replace=False)` | k unique random indices from range(n) |
| `np.searchsorted(bins, x)` | Bin index for each value of `x` |

## Worked Example
Custom vs. built-in histogram benchmark: for 100 points, the hand-rolled `np.add.at(counts, np.searchsorted(bins, x), 1)` (~17.6 µs) beats `np.histogram` (~33.8 µs) because the built-in is more general. But at 1,000,000 points, `np.histogram` (~84.4 ms) beats the custom version (~128 ms) because NumPy's algorithm is optimized for scale — illustrating that algorithmic efficiency is context-dependent (small vs. large N), not absolute.

## Key Takeaways
1. Fancy indexing passes arrays of indices; result shape follows the (broadcast) shape of those index arrays.
2. Fancy indexing composes freely with scalar indices, slices, and Boolean masks.
3. `x[i] += 1` does not accumulate correctly for repeated indices — use `np.add.at` (or other `ufunc.at`) for true in-place accumulation.
4. `np.random.choice(..., replace=False)` is the standard way to draw a random unique subset of row indices (e.g., for train/test splits).
5. Algorithmic efficiency is scale-dependent: hand-rolled vectorized code can beat (or lose to) general-purpose library functions depending on dataset size.

## Connects To
- **Ch 8**: fancy-indexing pairing of row/column index arrays follows broadcasting rules.
- **Ch 9**: fancy indexing complements Boolean masking as another array-selection mechanism.
- **Ch 11**: `np.argsort`/`np.argpartition` produce index arrays consumed via fancy indexing (e.g., k-nearest-neighbors).
- **Ch 39**: `np.random.choice` for random sampling previews train/test split logic in model validation.
