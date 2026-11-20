# Chapter 19: Combining Datasets: merge and join

## Core Idea
`pd.merge` implements database-style relational-algebra joins (one-to-one, many-to-one, many-to-many) that link datasets by shared key columns/indices, with explicit controls for key specification, set-arithmetic join type, and column-name collisions.

## Frameworks Introduced
- **`pd.merge(left, right, on=, left_on=/right_on=, left_index=/right_index=, how=, suffixes=)`**: the unified interface for all join types.
  - When to use: combining datasets by a shared key (not just stacking/concatenating) — the relational "lookup and link" operation.
  - How: default auto-detects shared column names as keys; `on=` specifies explicitly; `left_on=`/`right_on=` handle differently-named key columns (drop the redundant one after, if desired); `left_index=`/`right_index=` merge on the index (or use `.join()` shorthand); mix with `left_index=True, right_on=...` as needed.
- **Join type (`how=`)**: `'inner'` (default, intersection of keys), `'outer'` (union, NaN-fills), `'left'`, `'right'` (keep all rows from one side).
- **Three join cardinalities**: one-to-one (like column-wise concat), many-to-one (duplicates preserved from the "many" side, repeated to match), many-to-many (full cross-product per matching key group).

## Key Concepts
- **Key column auto-detection**: `pd.merge(df1, df2)` with no `on=` finds the single shared column name and joins on it.
- **Order is not necessarily preserved**: `pd.merge` correctly realigns rows by key even if row order differs between inputs.
- **Index discarded by default**: a merge-by-columns discards the original index unless merging by index (`left_index`/`right_index`).
- **`DataFrame.join()`**: convenience method for index-based merges without extra keywords.
- **`suffixes=`**: when both inputs have a non-key column with the same name, Pandas auto-appends `_x`/`_y` (customizable via `suffixes=[...]`) to disambiguate.
- **`.drop(col, axis=1)`**: common follow-up to remove a redundant column after a `left_on`/`right_on` merge.

## Mental Models
- `pd.merge` is "SQL JOIN for DataFrames" — the four `how=` options map directly to SQL's INNER/FULL OUTER/LEFT/RIGHT JOIN, and the many-to-many case is literally a key-grouped cross-product (think: for each shared key value, pair every matching row from the left with every matching row from the right).
- Real-world merges nearly always need a "merge → check for nulls → investigate/fix mismatches" loop: `merged.isnull().any()` after a merge is the standard sanity check for key mismatches (as opposed to genuinely missing source data).

## Anti-patterns
- **Assuming `pd.merge`'s default inner join keeps every row from both inputs**: inner join only keeps rows whose key appears in *both* — use `how='outer'`/`'left'`/`'right'` explicitly when you need to preserve unmatched rows.
- **Not checking for post-merge nulls**: silently proceeding after a merge can hide mismatched keys (e.g., abbreviation typos, unmapped categories) — always run `.isnull().any()` (or inspect `.loc[mask]`) right after a merge with `how='outer'`/`'left'`/`'right'`.
- **Ignoring column-name collisions**: unexpected `_x`/`_y` suffixes in merge output usually indicate overlapping non-key column names — rename or pass `suffixes=` deliberately rather than leaving ambiguous defaults.

## Code Examples
```python
import pandas as pd

# one-to-one
df1 = pd.DataFrame({'employee': ['Bob','Jake','Lisa','Sue'], 'group': [...]})
df2 = pd.DataFrame({'employee': ['Lisa','Bob','Jake','Sue'], 'hire_date': [...]})
df3 = pd.merge(df1, df2)          # auto-detects 'employee' as key

# many-to-one
df4 = pd.DataFrame({'group': [...], 'supervisor': [...]})
pd.merge(df3, df4)                 # supervisor repeated per matching group

# many-to-many
df5 = pd.DataFrame({'group': [...dup...], 'skills': [...]})
pd.merge(df1, df5)                  # full cross-product per group

# key specification
pd.merge(df1, df2, on='employee')
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)

df1a = df1.set_index('employee'); df2a = df2.set_index('employee')
pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)                      # shorthand for index-based merge
pd.merge(df1a, df3, left_index=True, right_on='name')   # mixed index/column

# set arithmetic
pd.merge(df6, df7, how='inner')     # default: intersection
pd.merge(df6, df7, how='outer')      # union, NaN-fill
pd.merge(df6, df7, how='left')        # keep all left rows
pd.merge(df6, df7, how='right')        # keep all right rows

# suffixes
pd.merge(df8, df9, on="name")                       # auto rank_x, rank_y
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])  # custom suffixes
```
- **What it demonstrates**: all three join cardinalities, every key-specification mechanism (on/left_on+right_on/left_index+right_index/.join()), all four how= set operations, and suffix disambiguation.

## Reference Tables
| `how=` value | Behavior |
|---|---|
| `'inner'` (default) | Intersection of keys |
| `'outer'` | Union of keys, NaN for unmatched side |
| `'left'` | All rows from left, matched right data or NaN |
| `'right'` | All rows from right, matched left data or NaN |

| Key specification | Use case |
|---|---|
| (none, auto) | Shared column name(s) |
| `on='col'` | Explicit shared column name |
| `left_on=`/`right_on=` | Differently named key columns |
| `left_index=`/`right_index=` | Merge using the index |
| `.join()` | Shorthand for index-based merge |

## Worked Example
US states population-density ranking: merge `pop` (state/region, ages, year, population) with `abbrevs` (state, abbreviation) via `how='outer', left_on='state/region', right_on='abbreviation'`, then check `merged.isnull().any()` → finds null `population` (Puerto Rico pre-2000, genuinely missing source data) and null `state` (because `'PR'`/`'USA'` codes aren't in `abbrevs`) — fix the latter with targeted `.loc[...] = ` assignments. Then merge with `areas` via `how='left', on='state'`, again checking nulls (finds "United States" has no area entry) and `dropna()` since it's irrelevant to the per-state density question. Finally: `data2010 = final.query("year == 2010 & ages == 'total'")`, `density = data2010['population'] / data2010['area (sq. mi)']`, sorted descending — top result: Washington DC (~8899/sq mi); bottom: Alaska (~1.09/sq mi).

## Key Takeaways
1. `pd.merge` implements relational-algebra joins (one-to-one, many-to-one, many-to-many), auto-detecting shared columns as keys by default.
2. Key specification is flexible: `on=`, `left_on=`/`right_on=` (differently named columns), `left_index=`/`right_index=` (or `.join()` shorthand), and combinations thereof.
3. `how=` controls set arithmetic: `'inner'` (default, intersection), `'outer'` (union+NaN), `'left'`, `'right'`.
4. Overlapping non-key column names get auto-suffixed (`_x`/`_y`), customizable via `suffixes=`.
5. Always check for nulls immediately after a real-world merge — they reveal either genuinely missing source data or key mismatches needing a fix.
6. Combine `query()` (Ch 24 dependency), `set_index`, sorting, and arithmetic on merged columns to answer analytical questions end-to-end.

## Connects To
- **Ch 17**: hierarchical indexing tools often needed to reshape merge results.
- **Ch 18**: merge is the key-based complement to simple positional concatenation.
- **Ch 20/Ch 21**: groupby/pivot_table frequently operate on merged datasets like this US-states example.
- **Ch 24**: `.query()` used here for filtering previews the `eval`/`query` performance chapter.
