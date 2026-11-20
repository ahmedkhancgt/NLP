# Chapter 17: Hierarchical Indexing

## Core Idea
Pandas' `MultiIndex` compactly represents higher-dimensional (3D+) data within the familiar 1D `Series`/2D `DataFrame` structures, by incorporating multiple index levels, with `stack`/`unstack` converting between a multiply indexed Series and an equivalent DataFrame.

## Frameworks Introduced
- **`MultiIndex`**: multiple levels of labels per data point, replacing a clumsy tuple-keyed Series.
  - When to use: data naturally keyed by 2+ categorical dimensions (e.g., state × year).
  - How: `pd.MultiIndex.from_tuples(...)`, `.from_arrays(...)`, `.from_product(...)` (Cartesian product), or direct `pd.MultiIndex(levels=..., codes=...)`; or simply pass a list of index arrays / dict with tuple keys to a Series/DataFrame constructor.
- **`stack()` / `unstack()`**: convert between a multiply indexed Series and a conventionally indexed DataFrame (moving the innermost/specified index level to/from columns).
- **Index rearrangement**: `reset_index()` turns index levels into columns; `set_index([...])` builds a MultiIndex from existing columns — the standard round-trip pattern for reshaping real data.
- **`pd.IndexSlice`**: enables slice syntax inside `loc` tuples when working with multi-level rows/columns (`health_data.loc[idx[:, 1], idx[:, 'HR']]`).

## Key Concepts
- **Partial indexing**: indexing a MultiIndex with just the outer level (`pop['California']`) returns a Series with remaining (inner) levels intact.
- **Partial slicing on a sorted index**: `pop[:, 2010]` selects across all outer-level values for a fixed inner-level value (requires sorted index, see below).
- **MultiIndex for columns**: DataFrame columns can also be hierarchical (e.g., subject × measurement type), since rows/columns are symmetric in a DataFrame.
- **`names` attribute**: label each MultiIndex level (`pop.index.names = ['state', 'year']`) for clarity.
- **Sorted vs. unsorted MultiIndex**: many slicing operations (e.g., `data['a':'b']`) require lexicographically sorted index levels — `sort_index()` fixes a `KeyError` caused by an unsorted index.
- **Blank entries in display**: a blank cell under a MultiIndex display means "same value as the line above it" (not missing data).

## Mental Models
- A MultiIndex level is "an extra dimension folded into a flat structure" — think of a Series with a 2-level MultiIndex as literally a 2D array, and `unstack`/`stack` as the explicit reshape operation between the flattened and "widened" representations.
- Partial indexing/slicing works by "peeling off" the outermost level(s) first, matching the hierarchical (nested) structure — you cannot slice an inner level without first either specifying the outer level or sorting the index.

## Anti-patterns
- **Using raw Python tuples as a Series index instead of a MultiIndex**: works for single-key lookup but partial selection (e.g., "all data from 2010") requires slow, messy list-comprehension filtering — use `pd.MultiIndex.from_tuples` instead.
- **Attempting partial slicing on an unsorted MultiIndex**: raises `KeyError: 'Key length (1) was greater than MultiIndex lexsort depth (0)'` — call `.sort_index()` first.
- **Writing a slice directly inside a `loc` indexing tuple**: `health_data.loc[(:, 1), (:, 'HR')]` is a `SyntaxError` — use `pd.IndexSlice` (`idx[:, 1]`) instead.

## Code Examples
```python
import pandas as pd
import numpy as np

# bad way: tuple keys
index = [('California', 2010), ('California', 2020), ('Texas', 2010), ('Texas', 2020)]
pop = pd.Series([37253956, 39538223, 25145561, 29145505], index=index)
pop[[i for i in pop.index if i[1] == 2010]]   # slow, messy

# better way: MultiIndex
index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
pop[:, 2020]                      # all states, year 2020 -- clean partial slice

pop_df = pop.unstack()             # multiply indexed Series -> DataFrame
pop_df.stack()                      # back to Series

pop_df = pd.DataFrame({'total': pop, 'under18': [...]})
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()

# construction methods
pd.MultiIndex.from_arrays([['a','a','b','b'], [1,2,1,2]])
pd.MultiIndex.from_tuples([('a',1), ('a',2), ('b',1), ('b',2)])
pd.MultiIndex.from_product([['a','b'], [1,2]])   # Cartesian product
pd.MultiIndex(levels=[['a','b'],[1,2]], codes=[[0,0,1,1],[0,1,0,1]])

pop.index.names = ['state', 'year']   # label the levels

# MultiIndex columns
index = pd.MultiIndex.from_product([[2013,2014],[1,2]], names=['year','visit'])
columns = pd.MultiIndex.from_product([['Bob','Guido','Sue'],['HR','Temp']], names=['subject','type'])
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data['Guido']                      # all of Guido's columns
health_data['Guido', 'HR']                  # specific column
health_data.iloc[:2, :2]
health_data.loc[:, ('Bob', 'HR')]

idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]     # slices inside loc tuples

# sorted vs unsorted
data = data.sort_index()            # required before partial slicing works
data['a':'b']

pop.unstack(level=0)   # explicit level choice
pop.unstack(level=1)

pop_flat = pop.reset_index(name='population')    # index -> columns
pop_flat.set_index(['state', 'year'])               # columns -> MultiIndex
```
- **What it demonstrates**: tuple-index pain points, full MultiIndex construction/slicing/partial-indexing vocabulary, multi-level columns, IndexSlice for compound slicing, sort-before-slice requirement, and the stack/unstack/reset_index/set_index reshape round-trip.

## Reference Tables
| Construction method | Input |
|---|---|
| `from_tuples` | list of tuples |
| `from_arrays` | list of per-level arrays |
| `from_product` | Cartesian product of per-level lists |
| `MultiIndex(levels=, codes=)` | explicit internal encoding |

| Reshape operation | Direction |
|---|---|
| `stack()` | columns → inner index level |
| `unstack(level=)` | index level → columns |
| `reset_index(name=)` | index levels → columns, flat integer index |
| `set_index([...])` | columns → MultiIndex |

## Worked Example
Medical mock dataset: rows MultiIndexed by `(year, visit)`, columns MultiIndexed by `(subject, type)`. `health_data['Guido']` slices to just Guido's `HR`/`Temp` columns (still indexed by year/visit); `health_data.loc[idx[:, 1], idx[:, 'HR']]` uses `IndexSlice` to select only visit==1 rows and only HR-type columns across all subjects in one expression — demonstrating four-dimensional data (subject × measurement × year × visit) manipulated entirely through 2-level row/column MultiIndexes.

## Key Takeaways
1. MultiIndex compactly represents N-dimensional data within Series (1D) / DataFrame (2D) structures via multiple index levels.
2. `stack`/`unstack` convert between the "flat multiply indexed Series" and "wide DataFrame" representations.
3. Partial indexing (outer level only) returns a reduced Series/DataFrame with remaining levels intact; partial slicing on inner levels requires a sorted index.
4. DataFrame columns can be hierarchical too — rows and columns are symmetric in this respect.
5. `pd.IndexSlice` is required to use slice syntax inside a `loc` tuple for multi-level indexing.
6. `reset_index`/`set_index` round-trip between index levels and plain columns — a very common real-world reshaping pattern.

## Connects To
- **Ch 14**: `loc`/`iloc` indexers extend directly to MultiIndex with tuple-based access.
- **Ch 18/Ch 19**: concat/merge operations frequently interact with hierarchical indices.
- **Ch 20/Ch 21**: groupby and pivot_table operations often produce (or consume) MultiIndex results.
