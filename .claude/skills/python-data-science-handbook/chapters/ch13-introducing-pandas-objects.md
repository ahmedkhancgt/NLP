# Chapter 13: Introducing Pandas Objects

## Core Idea
Pandas' `Series`, `DataFrame`, and `Index` enhance NumPy arrays with explicit labels, letting data be viewed simultaneously as generalized arrays and as (specialized, type-aware) dictionaries.

## Frameworks Introduced
- **Series as generalized array + specialized dictionary**: a `Series` combines a NumPy-array-like `.values` buffer with an explicit `.index` — item access works array-style (position) and dict-style (label/key), and arbitrary (non-integer, non-contiguous) index labels are allowed.
  - When to use: any labeled 1D data.
  - How: `pd.Series(data, index=index)` — `data` can be a list/array (index defaults to integers), a scalar (repeated across given index), or a dict (index defaults to sorted keys, or pass `index=` to subset/reorder).
- **DataFrame as generalized 2D array + specialized dictionary**: a `DataFrame` is an aligned collection of `Series` sharing an index (columns), or a 2D array with both row and column labels.
  - How: build from a dict of Series/arrays, a list of dicts, a 2D NumPy array (+ `columns=`/`index=`), or a NumPy structured array.
- **Index as immutable ordered (multi)set**: supports array-like access/slicing plus set operations (`union`, `intersection`, `symmetric_difference`), but cannot be mutated in place.

## Key Concepts
- **`.values`**: the underlying NumPy array backing a Series/DataFrame.
- **`.index`**: explicit label array (type `pd.Index`); for DataFrame, `.columns` is the column-label `Index`.
- **Alignment**: DataFrame columns are "aligned" Series sharing the same index.
- **Missing-key filling**: constructing a DataFrame from a list of dicts with inconsistent keys fills missing entries with `NaN`.
- **Immutability of Index**: `ind[1] = 0` raises `TypeError` — indices are safe to share across objects without risk of accidental mutation.
- **`data['col0']` vs. `data[0]`**: on a DataFrame, bracket indexing with a label gets a *column* (unlike a raw 2D NumPy array, where `data[0]` gets a *row*) — a key point of confusion to watch for.

## Mental Models
- Series ≈ "a dictionary with guaranteed, ordered, typed keys and values" — you get dict-style lookup (`'a' in data`, `data['b']`) plus array-style slicing/vectorized math that a plain dict lacks.
- DataFrame ≈ "an aligned collection of Series sharing one index" — think of each column as its own Series, all glued together by a common row-index.
- Think of `Index` as an immutable, set-like array: safe to share between many Series/DataFrames without fear of one object's edits leaking into another.

## Anti-patterns
- **Treating a DataFrame exactly like a 2D NumPy array for indexing**: `data[0]` does NOT get the first row (unlike `.values[0]`) — it is dictionary-style column-oriented bracket access, which errors/misleads if column 0 isn't labeled `0`.
- **Assuming explicit-index slices exclude the endpoint like implicit slices do**: `data['a':'c']` (label-based) *includes* `'c'`; `data[0:2]` (position-based) *excludes* index 2 — a Python-vs-Pandas slicing inconsistency to watch closely (fully resolved by `loc`/`iloc` in Ch 14).

## Code Examples
```python
import numpy as np
import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data.values; data.index          # array([0.25,...]); RangeIndex(0,4)
data[1]; data[1:3]

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data['b']
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])   # noncontiguous index
data[5]

population_dict = {'California': 39538223, 'Texas': 29145505, ...}
population = pd.Series(population_dict)
population['California']
population['California':'Florida']     # dict-style + array-style slicing

pd.Series([2, 4, 6])                     # from list, default integer index
pd.Series(5, index=[100, 200, 300])      # scalar repeated across index
pd.Series({2: 'a', 1: 'b', 3: 'c'})       # from dict
pd.Series({2: 'a', 1: 'b', 3: 'c'}, index=[1, 2])   # explicit subset/order

area = pd.Series({'California': 423967, 'Texas': 695662, ...})
states = pd.DataFrame({'population': population, 'area': area})
states.index; states.columns
states['area']                            # column access, returns a Series

pd.DataFrame(population, columns=['population'])          # from a single Series
pd.DataFrame([{'a': i, 'b': 2*i} for i in range(3)])        # from list of dicts
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])           # missing keys -> NaN
pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])

A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
pd.DataFrame(A)                           # from a structured array

ind = pd.Index([2, 3, 5, 7, 11])
ind[1]; ind[::2]; ind.size, ind.shape, ind.ndim, ind.dtype
ind[1] = 0        # TypeError: Index does not support mutable operations

indA = pd.Index([1, 3, 5, 7, 9]); indB = pd.Index([2, 3, 5, 7, 11])
indA.intersection(indB); indA.union(indB); indA.symmetric_difference(indB)
```
- **What it demonstrates**: constructing Series/DataFrame from every common source, the dual array/dict access model, and Index's array-like-but-immutable, set-like behavior.

## Reference Tables
| Construction source for `pd.Series(data, index=index)` | Index default |
|---|---|
| list/array | integer `RangeIndex` |
| scalar | repeats value at every given index |
| dict | dict's keys (order as given/sorted) |

| DataFrame construction source | Notes |
|---|---|
| Single Series | one-column DataFrame |
| List of dicts | missing keys → `NaN` |
| Dict of Series | columns aligned by shared index |
| 2D NumPy array | `columns=`/`index=` optional, else integer defaults |
| NumPy structured array | field names become column names |

## Worked Example
Building a US-states DataFrame: create `population` and `area` as separate `Series` keyed by state name, then `states = pd.DataFrame({'population': population, 'area': area})` auto-aligns both Series on their shared state-name index into one two-column table — illustrating the "DataFrame = aligned collection of Series" model directly.

## Key Takeaways
1. A Series pairs explicit, flexible-type index labels with a NumPy-backed values array — it's both array-like and dict-like.
2. A DataFrame is an aligned collection of Series sharing an index — accessible as a dict-of-columns or (with care) as a 2D array via `.values`.
3. DataFrames can be built from dicts of Series, lists of dicts (auto-filling `NaN` for missing keys), 2D arrays, or structured arrays.
4. `Index` objects behave like immutable arrays and like ordered (multi)sets, supporting `union`/`intersection`/`symmetric_difference`.
5. Bracket indexing on a bare DataFrame (`data[0]` or `data['col']`) is column/dict-oriented, not row-oriented like a NumPy array — a key early trap resolved properly with `loc`/`iloc` (Ch 14).

## Connects To
- **Ch 12**: Series/DataFrame are explicitly presented as an evolution of NumPy's structured arrays.
- **Ch 14**: resolves the indexing ambiguities (loc/iloc) introduced here.
- **Ch 16**: the "missing keys → NaN" behavior previews missing-data handling.
