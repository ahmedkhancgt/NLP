# Chapter 18: Combining Datasets: concat and append

## Core Idea
`pd.concat` extends NumPy's `np.concatenate` to Series/DataFrames, preserving (and optionally handling duplicate) indices, with configurable join behavior (`outer`/`inner`) when inputs have mismatched columns.

## Frameworks Introduced
- **`pd.concat(objs, axis=, join=, ignore_index=, keys=, verify_integrity=...)`**: the general-purpose combiner for Series/DataFrames along rows (`axis=0`, default) or columns (`axis='columns'`).
  - When to use: stacking datasets with the same (or overlapping) structure — the simple case of combination, as opposed to key-based merging (Ch 19).
  - How: default preserves original indices (even creating duplicates); `verify_integrity=True` raises on overlap; `ignore_index=True` creates a fresh integer index; `keys=[...]` creates a hierarchical index labeling each source.
- **Join control for mismatched columns**: `join='outer'` (default, union of columns, fills missing with NaN) vs. `join='inner'` (intersection only); `reindex()` before concatenation gives finer manual control.

## Key Concepts
- **Index preservation (and duplication)**: unlike `np.concatenate`, `pd.concat` keeps original index labels as-is, even producing duplicate index values — handle via `verify_integrity`, `ignore_index`, or `keys`.
- **`axis` parameter**: `0`/`'index'` (default, stack rows) vs. `1`/`'columns'` (stack side-by-side).
- **`join` parameter**: `'outer'` (union, NaN-fill) or `'inner'` (intersection only) for mismatched column sets.
- **`.append()` method (deprecated-style convenience)**: `df1.append(df2)` ≈ `pd.concat([df1, df2])`, but creates a new object each call (does not mutate in place) and is inefficient for repeated use — prefer building a list and calling `pd.concat` once.

## Mental Models
- `pd.concat` is "stacking with label bookkeeping" — unlike raw NumPy concatenation, it must decide what to do about index collisions and column mismatches, which is why it exposes `verify_integrity`/`ignore_index`/`keys`/`join` as explicit controls rather than silently guessing.
- Repeated `.append()` calls are analogous to repeated Python list `+=` on immutable strings — each call reallocates; batch the inputs and call `concat` once for performance.

## Anti-patterns
- **Calling `.append()` in a loop to build up a combined DataFrame**: each call recreates the index/data buffer — inefficient; collect DataFrames in a list and call `pd.concat(list_of_dfs)` once.
- **Ignoring duplicate indices after concatenation**: silently valid in Pandas but often a sign of a logic error — use `verify_integrity=True` to catch unintended overlaps early.
- **Assuming `join='outer'` (the default) won't introduce NaNs**: when input DataFrames have different column sets, the default union-join fills non-overlapping columns with `NaN` — use `join='inner'` or explicit `reindex()` if that's undesired.

## Code Examples
```python
import pandas as pd
import numpy as np

x = [1, 2, 3]; y = [4, 5, 6]; z = [7, 8, 9]
np.concatenate([x, y, z])              # recall: NumPy concatenation

ser1 = pd.Series(['A','B','C'], index=[1,2,3])
ser2 = pd.Series(['D','E','F'], index=[4,5,6])
pd.concat([ser1, ser2])

df1 = make_df('AB', [1, 2]); df2 = make_df('AB', [3, 4])
pd.concat([df1, df2])                  # row-wise, default axis=0

df3 = make_df('AB', [0, 1]); df4 = make_df('CD', [0, 1])
pd.concat([df3, df4], axis='columns')   # column-wise

# duplicate indices
y.index = x.index
pd.concat([x, y])                        # duplicate index values preserved!
pd.concat([x, y], verify_integrity=True)   # raises ValueError on overlap
pd.concat([x, y], ignore_index=True)       # fresh integer index
pd.concat([x, y], keys=['x', 'y'])          # hierarchical index labeling source

# mismatched columns
df5 = make_df('ABC', [1, 2]); df6 = make_df('BCD', [3, 4])
pd.concat([df5, df6])                        # outer join (default): NaN-filled
pd.concat([df5, df6], join='inner')           # intersection only: cols B, C
pd.concat([df5, df6.reindex(df5.columns, axis=1)])   # explicit manual control

df1.append(df2)                                # convenience, but creates new object each time
```
- **What it demonstrates**: row-wise and column-wise concatenation, the three duplicate-index handling strategies (verify_integrity/ignore_index/keys), outer vs. inner join on mismatched columns, and the less-efficient `.append()` shortcut.

## Reference Tables
| Parameter | Effect |
|---|---|
| `axis=0` (default) | Concatenate rows |
| `axis='columns'` | Concatenate columns |
| `join='outer'` (default) | Union of columns, NaN-fill gaps |
| `join='inner'` | Intersection of columns only |
| `ignore_index=True` | Discard originals, assign fresh integer index |
| `verify_integrity=True` | Raise `ValueError` on index overlap |
| `keys=[...]` | Add hierarchical index labeling each source object |

## Worked Example
Concatenating `df5` (columns A, B, C) and `df6` (columns B, C, D): default `pd.concat([df5, df6])` produces the union of columns (A, B, C, D) with `NaN` wherever a column is absent from one input (e.g., `A` is NaN for df6's rows, `D` is NaN for df5's rows); `pd.concat([df5, df6], join='inner')` instead keeps only the shared columns `B, C`, discarding `A` and `D` entirely — showing the outer/inner trade-off directly.

## Key Takeaways
1. `pd.concat` generalizes `np.concatenate` to labeled Series/DataFrames, defaulting to row-wise stacking (`axis=0`).
2. Unlike NumPy, Pandas preserves original index labels through concatenation — even producing duplicates, which `verify_integrity`, `ignore_index`, or `keys` can variously guard against or resolve.
3. Mismatched columns default to an outer (union) join with NaN-fill; use `join='inner'` for intersection-only, or `reindex()` for precise manual control.
4. `.append()` is a convenience wrapper around `concat` but is inefficient for repeated calls — batch inputs into a list first.
5. The `keys=` option produces a hierarchical index, directly connecting to Chapter 17's MultiIndex tools for further reshaping.

## Connects To
- **Ch 5**: `pd.concat` is explicitly framed as the Pandas analog of `np.concatenate`.
- **Ch 17**: `keys=` concatenation results are naturally explored with MultiIndex tools (stack/unstack).
- **Ch 19**: the next chapter covers key-based (database-style) merging as the more powerful alternative to simple concatenation.
