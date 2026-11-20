# Chapter 15: Operating on Data in Pandas

## Core Idea
Pandas preserves index/column labels through NumPy ufuncs and automatically aligns indices (filling mismatches with NaN) during binary operations, making data-context preservation essentially automatic.

## Frameworks Introduced
- **Ufunc index preservation**: applying any NumPy ufunc to a Series/DataFrame returns the same type with the same index/columns intact.
  - How: `np.exp(ser)`, `np.sin(df * np.pi/4)` — works for any ufunc from Chapter 6.
- **Index alignment for binary ops**: `Series op Series` / `DataFrame op DataFrame` takes the *union* of indices/columns, filling missing combinations with `NaN`; use the explicit method form (`.add`, `.sub`, etc.) with `fill_value=` to control the fill instead of `NaN`.
- **DataFrame–Series broadcasting**: operations between a DataFrame and a Series follow NumPy's row-wise broadcasting by default (`df - df.iloc[0]`); pass `axis=0` to the method form (e.g., `df.subtract(series, axis=0)`) to instead operate column-wise.

## Key Concepts
- **Index union**: result of a binary op between misaligned Series/DataFrames includes every index label from both operands.
- **`fill_value=`**: parameter on arithmetic methods (`.add`, `.sub`, `.mul`, `.div`, etc.) to substitute a non-NaN value for missing alignment entries.
- **Arithmetic method mapping**: `+`→`.add`, `-`→`.sub`/`.subtract`, `*`→`.mul`/`.multiply`, `/`→`.truediv`/`.div`/`.divide`, `//`→`.floordiv`, `%`→`.mod`, `**`→`.pow`.
- **Row-wise default broadcasting**: `df - df.iloc[0]` subtracts the first row from every row (column labels align automatically).
- **Column-wise via `axis=`**: `df.subtract(df['R'], axis=0)` broadcasts a column Series down each row instead.

## Mental Models
- Think of every Pandas binary operation as "align first (via index union), then apply element-wise, filling absent pairs with NaN" — this is what makes combining heterogeneous/misaligned real-world data safe by default (no silent misalignment bugs as with raw NumPy arrays).
- DataFrame-Series arithmetic follows the same broadcasting intuition as NumPy 2D-1D array ops (Ch 8), just dressed up with label alignment on top.

## Anti-patterns
- **Assuming binary Series/DataFrame ops silently drop non-overlapping labels**: they don't — missing combinations become `NaN` in the union result, which can silently propagate into later computations if unnoticed.
- **Forgetting that NaN differs from "0" semantically in arithmetic fill**: default alignment fills gaps with `NaN`, not 0 — use `.add(other, fill_value=0)` explicitly when 0 is the intended default.
- **Using the raw operator when you need column-wise (not row-wise) broadcasting against a DataFrame**: `df - series` defaults to row-wise; use the `.subtract(series, axis=0)` method form for column-wise broadcasting.

## Code Examples
```python
import pandas as pd
import numpy as np
rng = np.random.default_rng(42)
ser = pd.Series(rng.integers(0, 10, 4))
df = pd.DataFrame(rng.integers(0, 10, (3, 4)), columns=['A','B','C','D'])

np.exp(ser)                    # index preserved
np.sin(df * np.pi / 4)          # index+columns preserved

area = pd.Series({'Alaska': 1723337, 'Texas': 695662, 'California': 423967})
population = pd.Series({'California': 39538223, 'Texas': 29145505, 'Florida': 21538187})
population / area               # union of indices; Alaska & Florida -> NaN
area.index.union(population.index)

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B                             # index 0 and 3 -> NaN
A.add(B, fill_value=0)            # explicit fill instead of NaN

A = pd.DataFrame(rng.integers(0, 20, (2, 2)), columns=['a', 'b'])
B = pd.DataFrame(rng.integers(0, 10, (3, 3)), columns=['b', 'a', 'c'])
A + B                              # aligned on union of both index and columns
A.add(B, fill_value=A.values.mean())

A = rng.integers(10, size=(3, 4))
A - A[0]                            # NumPy row-wise broadcasting
df = pd.DataFrame(A, columns=['Q','R','S','T'])
df - df.iloc[0]                      # Pandas: same row-wise default
df.subtract(df['R'], axis=0)          # column-wise broadcasting via axis=

halfrow = df.iloc[0, ::2]
df - halfrow                           # auto-aligns on column labels, NaN where absent
```
- **What it demonstrates**: ufunc index preservation, Series/DataFrame index-union alignment with NaN-filling, explicit fill_value control, and DataFrame-Series row-wise vs. column-wise broadcasting.

## Reference Tables
| Python operator | Pandas method(s) |
|---|---|
| `+` | `add` |
| `-` | `sub`, `subtract` |
| `*` | `mul`, `multiply` |
| `/` | `truediv`, `div`, `divide` |
| `//` | `floordiv` |
| `%` | `mod` |
| `**` | `pow` |

## Worked Example
Combining mismatched state-level `area` and `population` Series (different state sets): `population / area` returns a Series indexed by the *union* of both (`Alaska, California, Florida, Texas`), with `NaN` for Alaska (no population data) and Florida (no area data), and real density values for California/Texas — demonstrating safe, explicit handling of incomplete overlapping datasets without manual pre-alignment.

## Key Takeaways
1. NumPy ufuncs applied to Series/DataFrames preserve index/column labels automatically.
2. Binary operations between Series/DataFrames align on the union of indices/columns, filling non-overlapping entries with `NaN`.
3. Use the named arithmetic methods (`.add`, `.sub`, ...) with `fill_value=` to control the fill value instead of `NaN`.
4. DataFrame-Series operations broadcast row-wise by default; use `axis=0` on the method form for column-wise broadcasting.
5. This automatic alignment is what makes Pandas "essentially foolproof" for combining heterogeneous/misaligned data sources, compared to manual NumPy array alignment.

## Connects To
- **Ch 6**: all NumPy ufuncs work directly, inheriting their vectorized-computation benefits.
- **Ch 8**: DataFrame-Series broadcasting directly extends NumPy's array broadcasting rules.
- **Ch 16**: the `NaN` fill from misalignment is exactly the missing-data representation covered next.
