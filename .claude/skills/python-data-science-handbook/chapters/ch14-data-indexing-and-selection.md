# Chapter 14: Data Indexing and Selection

## Core Idea
Pandas resolves the ambiguity between label-based and position-based access via explicit `loc` (label) and `iloc` (position) indexers, while plain bracket indexing follows separate conventions for Series (array/dict-like) vs. DataFrame (column-oriented, with special-cased row slicing/masking).

## Frameworks Introduced
- **`loc` / `iloc` indexers**: explicit, unambiguous access.
  - `loc[...]`: always label-based (explicit index); slices are inclusive of the endpoint.
  - `iloc[...]`: always position-based (implicit integer index); slices exclude the endpoint (standard Python convention).
  - When to use: *always*, whenever the index could be integer-valued or ambiguity is possible — "explicit is better than implicit."
- **DataFrame bracket-indexing special cases**: `data['col']` selects a column; but `data['row_label':'row_label2']`, `data[1:3]` (slice), and `data[boolean_mask]` are interpreted row-wise — asymmetric but practical conventions.

## Key Concepts
- **Series as dict**: `'a' in data`, `data.keys()`, `list(data.items())`, `data['e'] = 1.25` (extends like a dict).
- **Series as array**: slicing/masking/fancy indexing analogous to NumPy (`data['a':'c']`, `data[(data>0.3)&(data<0.8)]`, `data[['a','e']]`).
- **Explicit vs. implicit slicing inconsistency (integer index)**: `data[1]` → explicit index lookup; `data[1:3]` → implicit positional slice — same object, different rule depending on indexing vs. slicing syntax, which is exactly what `loc`/`iloc` eliminate.
- **DataFrame as dict of Series**: `data['area']` or `data.area` (attribute access, only when the column name is a valid Python identifier and doesn't collide with a DataFrame method, e.g., `pop`).
- **DataFrame as 2D array**: `.values` gives the raw NumPy array; `.T` transposes; `.values[0]` gets a row (contrast with `data[0]` semantics).
- **Combining masking + fancy indexing in `loc`**: `data.loc[data.density > 120, ['pop','density']]`.

## Mental Models
- Series blends two metaphors (dict + array) that mostly agree, except for integer indices where indexing vs. slicing diverge — `loc`/`iloc` exist precisely to remove that ambiguity by picking one metaphor explicitly.
- For a DataFrame, bracket access is column-first (dict metaphor) by default, but row-first for slices and masks — memorize this split rather than expecting full consistency; `loc`/`iloc` are the consistent/array-like alternative for both rows and columns together.

## Anti-patterns
- **Relying on bare `data[...]` indexing with an integer-labeled Series/DataFrame**: integer index + plain brackets silently switches between label (`data[1]`) and position (`data[1:3]`) semantics depending on indexing vs. slicing — ambiguous and bug-prone; use `.loc`/`.iloc` explicitly instead.
- **Assigning to a column via attribute syntax (`data.pop = z`)**: dangerous/wrong when the column name shadows a DataFrame method (e.g., `pop`) — always use `data['pop'] = z`.
- **Assuming DataFrame slices/masks behave like column selection**: `data['Florida':'New York']`, `data[1:3]`, and `data[data.density > 120]` are all row-wise, unlike `data['area']` which is column-wise — know the exception, don't assume uniformity.

## Code Examples
```python
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data['b']
'a' in data
data.keys()
list(data.items())
data['e'] = 1.25                     # dict-like extension

data['a':'c']        # explicit-index slice, INCLUDES 'c'
data[0:2]             # implicit-index slice, EXCLUDES index 2
data[(data > 0.3) & (data < 0.8)]     # masking
data[['a', 'e']]       # fancy indexing

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data[1]        # explicit index -> 'a'
data[1:3]       # implicit index -> rows at positions 1,2 -> labels 3,5
data.loc[1]      # explicit, always -> 'a'
data.loc[1:3]     # explicit slice, includes label 3
data.iloc[1]      # implicit, always -> 'b'
data.iloc[1:3]     # implicit slice, positions 1,2

area = pd.Series({'California': 423967, ...})
pop = pd.Series({'California': 39538223, ...})
data = pd.DataFrame({'area': area, 'pop': pop})
data['area']; data.area                # column access (bracket or attribute)
data.pop is data['pop']                # False! 'pop' collides with DataFrame.pop method
data['density'] = data['pop'] / data['area']   # add a new column

data.values            # raw 2D NumPy array
data.T                   # transpose
data.values[0]           # first ROW (array semantics)
data['area']              # first COLUMN (dict semantics) -- note the asymmetry

data.iloc[:3, :2]          # array-style via implicit index, labels preserved
data.loc[:'Florida', :'pop']  # array-style via explicit index
data.loc[data.density > 120, ['pop', 'density']]   # mask + fancy indexing combined
data.iloc[0, 2] = 90        # set a value

data['Florida':'New York']   # row-wise slice by label (exception to column-first rule)
data[1:3]                      # row-wise slice by position
data[data.density > 120]       # row-wise mask
```
- **What it demonstrates**: the full Series dict/array duality, the integer-index indexing-vs-slicing trap, `loc`/`iloc` as the unambiguous fix, DataFrame's column-first dict metaphor contrasted with its row-wise slice/mask exceptions, and combining masking with fancy indexing via `loc`.

## Reference Tables
| Access pattern | Series | DataFrame |
|---|---|---|
| `obj[label]` | value at that label | column Series |
| `obj[int_slice]` | positional slice | row slice (positional) |
| `obj[label_slice]` | label slice (inclusive) | row slice (label, inclusive) |
| `obj[boolean_mask]` | filtered values | filtered rows |
| `obj.loc[...]` | always label-based | always label-based, 2D |
| `obj.iloc[...]` | always position-based | always position-based, 2D |

## Worked Example
US states DataFrame (`area`, `pop`, derived `density`): `data.loc[data.density > 120, ['pop', 'density']]` combines a Boolean mask (`density > 120`) with fancy column selection (`['pop','density']`) in one `loc` call, returning Florida and New York's `pop`/`density` — showing how `loc` unifies row filtering and column selection into a single explicit, label-based expression, the idiomatic Pandas pattern for conditional column subsets.

## Key Takeaways
1. `loc` is always label/explicit-index-based; `iloc` is always position/implicit-index-based — use them whenever ambiguity is possible (especially integer indices).
2. Explicit-index slices (`data['a':'c']`, `loc[...]`) include the endpoint; implicit/positional slices (`data[0:2]`, `iloc[...]`) exclude it, matching standard Python slicing.
3. On a bare DataFrame, `data['col']` selects a column, but `data[row_slice]` and `data[mask]` select rows — memorize this asymmetry.
4. Attribute-style column access (`data.col`) works only for valid-identifier, non-colliding column names — never assign via attribute syntax.
5. `.values` exposes the raw 2D NumPy array (`.values[0]` = first row); transpose with `.T`.
6. `loc`/`iloc` support combining masks and fancy indexing in one call, e.g. `data.loc[mask, ['col1','col2']]`.

## Connects To
- **Ch 9/Ch 10**: Pandas masking and fancy indexing directly mirror the NumPy mechanisms from these chapters.
- **Ch 13**: resolves the Series dict/array ambiguity introduced there.
- **Ch 17**: hierarchical (MultiIndex) indexing extends `loc`/`iloc` patterns to multi-level labels.
