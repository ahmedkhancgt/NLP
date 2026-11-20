# Chapter 7: Aggregations: min, max, and Everything in Between

## Core Idea
NumPy's compiled aggregation functions (sum, min, max, mean, std, percentile, etc.) summarize datasets far faster than Python builtins, and support axis-wise reduction over multidimensional arrays.

## Frameworks Introduced
- **Axis-based aggregation**: `arr.sum(axis=N)` / `np.min(arr, axis=N)` collapse the specified axis, returning a reduced-dimension result.
  - When to use: summarizing rows vs. columns in 2D+ data.
  - How: `axis=0` collapses rows (aggregates down each column); `axis=1` collapses columns (aggregates across each row) — axis specifies the dimension removed, not the one kept.
- **NaN-safe aggregates**: `np.nansum`, `np.nanmean`, etc. compute the same statistic while ignoring NaN values.

## Key Concepts
- **`np.sum` vs. Python `sum`**: NumPy's version runs in compiled code (much faster) and has different optional-argument semantics (`sum(x, 1)` initializes at 1; `np.sum(x, 1)` sums along axis 1).
- **Array methods as shorthand**: `arr.min()`, `arr.max()`, `arr.sum()` are equivalent to `np.min(arr)` etc.
- **Axis semantic**: "the axis keyword specifies the dimension that will be collapsed, rather than the dimension that will be returned."
- **Percentile/median**: rank-based summary statistics, `np.percentile(arr, q)`, `np.median(arr)`.
- **`np.any` / `np.all`**: boolean reduction — whether any/all elements are truthy (no NaN-safe counterpart).

## Mental Models
- Think of an aggregation as "squashing" an array along one axis like folding paper along a crease — what remains has one fewer effective dimension along that axis.
- Prefer NumPy aggregates on NumPy arrays categorically — "whenever possible, make sure that you are using the NumPy version of these aggregates."

## Anti-patterns
- **Using Python's built-in `sum`/`min`/`max` on large NumPy arrays**: dramatically slower (tens to hundreds of ms vs. sub-ms) than `np.sum`/`np.min`/`np.max`.
- **Confusing axis direction**: assuming `axis=0` means "aggregate rows" in the sense of keeping rows — it actually collapses axis 0 (rows), yielding one value per column.
- **Ignoring NaNs silently**: standard aggregates propagate NaN; use the `nan*` variants when missing data should be skipped.

## Code Examples
```python
import numpy as np
rng = np.random.default_rng()

L = rng.random(100)
sum(L)          # Python builtin
np.sum(L)       # NumPy version, same result, much faster at scale

big_array = rng.random(1000000)
%timeit sum(big_array)       # ~89.9 ms
%timeit np.sum(big_array)    # ~521 µs

min(big_array), max(big_array)
np.min(big_array), np.max(big_array)
big_array.min(), big_array.max(), big_array.sum()   # method shorthand

M = rng.integers(0, 10, (3, 4))
M.sum()            # aggregate over all elements
M.min(axis=0)      # per-column min (collapses axis 0 = rows)
M.max(axis=1)      # per-row max (collapses axis 1 = columns)

# President heights example
import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
heights.mean(), heights.std(), heights.min(), heights.max()
np.percentile(heights, 25); np.median(heights); np.percentile(heights, 75)
```
- **What it demonstrates**: speed contrast between Python and NumPy aggregates, axis-based multidimensional reduction, and a real summary-statistics workflow (reading data with Pandas, then aggregating with NumPy).

## Reference Tables
| Function | NaN-safe version | Description |
|---|---|---|
| `np.sum` | `np.nansum` | Sum of elements |
| `np.prod` | `np.nanprod` | Product of elements |
| `np.mean` | `np.nanmean` | Mean |
| `np.std` | `np.nanstd` | Standard deviation |
| `np.var` | `np.nanvar` | Variance |
| `np.min` | `np.nanmin` | Minimum |
| `np.max` | `np.nanmax` | Maximum |
| `np.argmin` | `np.nanargmin` | Index of minimum |
| `np.argmax` | `np.nanargmax` | Index of maximum |
| `np.median` | `np.nanmedian` | Median |
| `np.percentile` | `np.nanpercentile` | Rank-based statistics |
| `np.any` | N/A | Any element true |
| `np.all` | N/A | All elements true |

## Worked Example
US president heights analysis: load `president_heights.csv` with Pandas, extract `height(cm)` into a NumPy array of 44 values, then compute: mean ≈ 180.05 cm, std ≈ 6.98, min = 163, max = 193, 25th percentile ≈ 174.75, median = 182.0, 75th percentile ≈ 183.5. Demonstrates the typical "load → summarize → (optionally visualize with Matplotlib)" pattern for first-pass exploratory data analysis.

## Key Takeaways
1. Always prefer `np.sum`/`np.min`/`np.max`/array methods over Python builtins for NumPy arrays — large speed difference.
2. `axis=N` collapses dimension N; the result has one fewer effective dimension along that axis.
3. Python's `sum(x, 1)` and NumPy's `np.sum(x, 1)` mean different things — don't assume interchangeability.
4. NaN-safe variants (`nansum`, `nanmean`, ...) exist for most aggregates to handle missing data.
5. Combine NumPy aggregates with Pandas for data loading and Matplotlib for visualization — this is the standard EDA pipeline.

## Connects To
- **Ch 6**: aggregates like `np.sum` are specializations of `ufunc.reduce` (e.g., `np.add.reduce`).
- **Ch 16**: NaN-safe aggregates relate directly to handling missing data in Pandas.
- **Ch 20**: Pandas groupby aggregation generalizes these same statistics across groups.
