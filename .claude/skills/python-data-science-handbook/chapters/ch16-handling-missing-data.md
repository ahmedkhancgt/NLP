# Chapter 16: Handling Missing Data

## Core Idea
Pandas represents missing data via sentinel values (`None`, `NaN`, and optionally `pd.NA` with nullable dtypes) rather than a separate mask array by default, and provides a consistent API (`isnull`/`notnull`/`dropna`/`fillna`) for detecting, removing, and imputing missing values.

## Frameworks Introduced
- **Masking vs. sentinel strategies**: two general approaches to representing missing data industry-wide — a separate Boolean mask array, or a special "sentinel" value embedded in the data itself. Pandas defaults to sentinels (`NaN`/`None`), with an opt-in nullable-dtype mask-based alternative.
- **Null-handling API**: `isnull()`/`notnull()` (detect), `dropna()` (remove), `fillna()` (impute) — the four core operations for working with missing data.
  - When to use `dropna`: when incomplete rows/columns should simply be excluded; control granularity via `axis`, `how` ('any'/'all'), and `thresh` (minimum non-null count to keep).
  - When to use `fillna`: when missing values should be imputed — with a constant, or via `method='ffill'`/`'bfill'` propagation, optionally along `axis=`.

## Key Concepts
- **`None`**: Python object sentinel; forces `dtype=object` arrays (slow, no native arithmetic — `TypeError` on aggregation).
- **`NaN`**: IEEE floating-point sentinel; keeps a native float dtype (fast), but "infects" any arithmetic it touches (`1 + np.nan == nan`); NaN-aware aggregates (`np.nansum`, etc.) exist to skip it.
- **Pandas' None/NaN interchangeability**: Pandas auto-converts between them and upcasts dtype as needed (e.g., setting an int Series entry to `None` upcasts the whole Series to float64 with `NaN`).
- **Upcasting table**: how storage dtype changes when NA is introduced, by original typeclass (floating: none; object: none; integer: → float64; boolean: → object).
- **Nullable dtypes (`pd.Int32`, etc.)**: capitalized dtype names that use an internal mask array and `pd.NA` sentinel, avoiding unwanted float upcasting — opt-in only, for backward compatibility.
- **`dropna` parameters**: `axis` (rows/columns), `how` ('any' vs 'all'), `thresh` (min non-null count to retain).
- **`fillna` parameters**: constant value, or `method='ffill'`/`'bfill'`, optionally with `axis=`.

## Mental Models
- "NaN is a data virus" — any arithmetic expression touching a NaN produces NaN; this is why NaN-aware aggregate functions exist as a deliberate opt-in antidote.
- Pandas' sentinel approach is a pragmatic trade-off, not a theoretically clean one: it avoids NumPy-wide bit-pattern reservations (which would be unwieldy across NumPy's many dtypes) at the cost of some implicit type-casting "magic" — accept this trade-off rather than expecting R-style unified NA handling.
- Think of `dropna`'s `how`/`thresh` as a dial from strict ("any" null drops the row/column) to lenient ("all" null, or below a minimum non-null count) — choose based on how much missingness is tolerable for the downstream analysis.

## Anti-patterns
- **Using `None` in NumPy-backed numeric arrays**: forces slow `dtype=object` storage and raises `TypeError` on aggregation (`sum`, etc.) — let Pandas convert to `NaN` instead.
- **Aggregating NaN-containing arrays with plain `np.sum`/`.min()`/`.max()`**: silently returns `NaN`, not an error — use `np.nansum`/`np.nanmin`/`np.nanmax` (or Pandas' own NaN-skipping aggregate methods) when you want to ignore missing values.
- **Assuming you can drop single missing cells from a DataFrame**: you can only drop entire rows or columns via `dropna`, not individual values — use `fillna` or manual masking if you want finer control.
- **Forward-filling without checking for leading NaNs**: `fillna(method='ffill')` leaves a `NaN` in place if there's no prior value to propagate (e.g., a NaN at the very start of a Series/row).

## Code Examples
```python
import numpy as np
import pandas as pd

vals1 = np.array([1, None, 2, 3])          # dtype=object, slow
vals1.sum()                                   # TypeError

vals2 = np.array([1, np.nan, 3, 4])          # native float dtype
1 + np.nan                                      # nan
vals2.sum(), vals2.min(), vals2.max()         # (nan, nan, nan)
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)  # (8.0, 1.0, 4.0)

pd.Series([1, np.nan, 2, None])               # None converted to NaN automatically

x = pd.Series(range(2), dtype=int)
x[0] = None                                     # upcasts to float64, None -> NaN

pd.Series([1, np.nan, 2, None, pd.NA], dtype='Int32')  # nullable dtype, pd.NA sentinel

data = pd.Series([1, np.nan, 'hello', None])
data.isnull()                                    # Boolean mask
data[data.notnull()]                              # filter via mask
data.dropna()                                      # same effect for a Series

df = pd.DataFrame([[1, np.nan, 2], [2, 3, 5], [np.nan, 4, 6]])
df.dropna()                                          # drops rows with ANY null (default how='any')
df.dropna(axis='columns')                             # drops columns with any null
df[3] = np.nan
df.dropna(axis='columns', how='all')                   # only drop all-null columns
df.dropna(axis='rows', thresh=3)                        # keep rows with >= 3 non-null values

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'), dtype='Int32')
data.fillna(0)
data.fillna(method='ffill')       # forward fill
data.fillna(method='bfill')       # backward fill
df.fillna(method='ffill', axis=1)  # fill across columns within each row
```
- **What it demonstrates**: None vs. NaN trade-offs and performance cost, Pandas' automatic conversion/upcasting, nullable dtypes with `pd.NA`, and the full detect/drop/fill workflow with fine-grained DataFrame control.

## Reference Tables
| Typeclass | Conversion when storing NAs | NA sentinel |
|---|---|---|
| floating | No change | `np.nan` |
| object | No change | `None` or `np.nan` |
| integer | Cast to `float64` | `np.nan` |
| boolean | Cast to `object` | `None` or `np.nan` |

| Method | Purpose |
|---|---|
| `isnull()` | Boolean mask of missing values |
| `notnull()` | Opposite of `isnull()` |
| `dropna(axis, how, thresh)` | Remove rows/columns with nulls |
| `fillna(value, method, axis)` | Impute missing values |

## Worked Example
DataFrame with a fully-null added column: `df[3] = np.nan` then `df.dropna(axis='columns', how='all')` correctly removes only the all-NaN column 3, preserving columns 0/1/2 that have partial data — contrasted with the default `df.dropna(axis='columns')` (`how='any'`), which would have also dropped columns 1 for containing even a single NaN, losing good data unnecessarily.

## Key Takeaways
1. Pandas defaults to sentinel-based missing data (`NaN` for floats, `None`/`NaN` interchangeably elsewhere), not a separate mask array, for compatibility with NumPy's limited dtype system.
2. `None` forces slow `object`-dtype storage and breaks arithmetic; `NaN` keeps native float performance but propagates through any arithmetic it touches.
3. Setting a value to missing in a non-float Series upcasts the dtype (commonly int→float) — this is automatic, "foolproof but implicit" behavior.
4. Nullable dtypes (`pd.Int32` etc., capitalized) with `pd.NA` avoid this upcasting at the cost of being opt-in only.
5. `isnull`/`notnull` detect; `dropna` removes (with `axis`/`how`/`thresh` controls); `fillna` imputes (constant, or `ffill`/`bfill`, optionally with `axis`).
6. You can only drop whole rows/columns via `dropna`, never individual cells.

## Connects To
- **Ch 7**: NaN-safe aggregate functions (`nansum`, etc.) directly address the "NaN infects everything" problem.
- **Ch 15**: index-alignment NaN-filling from binary ops is exactly the missing-data mechanism formalized here.
- **Ch 20**: groupby aggregation often needs explicit handling of NaN-containing groups, building on these tools.
