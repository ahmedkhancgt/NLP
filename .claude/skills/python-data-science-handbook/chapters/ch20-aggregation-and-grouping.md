# Chapter 20: Aggregation and Grouping

## Core Idea
Pandas aggregation extends NumPy-style summarization to labeled data (per-column by default), and `groupby` implements the "split-apply-combine" pattern — splitting data by key, applying a function per group, and combining results — as a lazily evaluated, efficient single-pass operation.

## Frameworks Introduced
- **Split-apply-combine via `groupby`**: `df.groupby(key)` returns a lazy `GroupBy` object; no computation happens until an operation (aggregate/filter/transform/apply) is called.
  - When to use: any "summarize/transform/filter conditional on a category" task.
  - How: `df.groupby('col').agg_method()`; column-index the GroupBy (`.groupby('method')['orbital_period']`); iterate directly (`for name, group in gb:`); or call any DataFrame/Series method, which dispatches per-group.
- **Four GroupBy operation types**:
  - `aggregate(func_or_list_or_dict)`: reduces each group to summary value(s); accepts strings, functions, lists of either, or a dict mapping column→operation.
  - `filter(func)`: drops entire groups based on a Boolean group-level predicate.
  - `transform(func)`: returns a same-shape-as-input result (e.g., group-wise centering).
  - `apply(func)`: most general — function takes a group DataFrame, returns a DataFrame/Series/scalar; combine behavior adapts to output type.
- **Flexible split-key specification**: group by a column name, a list/array/Series of same length as the DataFrame, a dict/Series mapping index→group label, any Python function applied to the index, or a list combining any of these (producing a multi-index grouping).

## Key Concepts
- **`DataFrameGroupBy`/`SeriesGroupBy`**: lazy "view" objects; "split" is not materialized until an operation triggers computation (efficient single-pass).
- **`.describe()`**: convenience aggregate method returning count/mean/std/min/quartiles/max per column.
- **Dispatch methods**: any DataFrame/Series method not explicitly implemented by GroupBy is passed through and applied per-group, then combined.
- **Column indexing on GroupBy**: `df.groupby('method')['col']` narrows to a `SeriesGroupBy` before aggregating.
- **`axis='columns'`**: aggregate across each row instead of down each column.

## Mental Models
- "Split-apply-combine" is the conceptual decomposition of nearly all groupby operations — even though the actual implementation fuses split+apply+combine into a single pass for efficiency, thinking in these three explicit steps clarifies what any custom `apply`/`filter`/`transform` function needs to do.
- GroupBy's laziness means "declaring groups costs nothing" — you can freely chain column selection and inspect the object before deciding what aggregate to compute, since nothing runs until the final method call.

## Anti-patterns
- **Manually looping over unique key values with masking to replicate groupby**: works but reimplements (less efficiently) what `groupby` already does in a single optimized pass — use `groupby` directly.
- **Using `apply` when `aggregate`/`transform`/`filter` would suffice**: `apply` is the most general (and often slowest) option; prefer the more specific method when the task fits (aggregate for reduction, transform for same-shape output, filter for group-level inclusion/exclusion).
- **Iterating over GroupBy groups for bulk computation**: direct iteration (`for name, group in gb:`) is useful for debugging/inspection but far slower than built-in aggregate/apply dispatch for actual computation.

## Code Examples
```python
import numpy as np
import pandas as pd
import seaborn as sns
planets = sns.load_dataset('planets')

ser = pd.Series(rng.rand(5))
ser.sum(); ser.mean()

df = pd.DataFrame({'A': rng.rand(5), 'B': rng.rand(5)})
df.mean()                      # per-column (default axis=0)
df.mean(axis='columns')          # per-row

planets.dropna().describe()      # count/mean/std/min/25%/50%/75%/max per column

df = pd.DataFrame({'key': ['A','B','C','A','B','C'], 'data': range(6)})
df.groupby('key')                  # lazy DataFrameGroupBy object
df.groupby('key').sum()

planets.groupby('method')['orbital_period'].median()   # column-indexed groupby

for (method, group) in planets.groupby('method'):
    print(f"{method:30s} shape={group.shape}")           # direct iteration

planets.groupby('method')['year'].describe().unstack()   # dispatch method

df = pd.DataFrame({'key': ['A','B','C','A','B','C'], 'data1': range(6),
                    'data2': rng.randint(0, 10, 6)})
df.groupby('key').aggregate(['min', np.median, max])        # list of aggs
df.groupby('key').aggregate({'data1': 'min', 'data2': 'max'}) # dict of aggs

def filter_func(x):
    return x['data2'].std() > 4
df.groupby('key').filter(filter_func)                         # drops group A

def center(x):
    return x - x.mean()
df.groupby('key').transform(center)                           # same-shape output

def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x
df.groupby('key').apply(norm_by_data2)                         # most general

L = [0, 1, 0, 1, 2, 0]
df.groupby(L).sum()                     # group by external list/array

df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
df2.groupby(mapping).sum()               # group by dict mapping index -> group
df2.groupby(str.lower).mean()             # group by function of index
df2.groupby([str.lower, mapping]).mean()  # combined -> multi-index grouping

# full example: planets discovered by method and decade
decade = (10 * (planets['year'] // 10)).astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
```
- **What it demonstrates**: simple per-column/row aggregation, `describe()`, lazy GroupBy construction, column-indexed and iterated groupby, dispatch-method passthrough, all four core group operations (aggregate/filter/transform/apply), and the five ways to specify a split key (column name, list, dict, function, combination).

## Reference Tables
| Aggregation method | Returns |
|---|---|
| `count` | Count of items |
| `first`, `last` | First/last item |
| `mean`, `median` | Mean/median |
| `min`, `max` | Minimum/maximum |
| `std`, `var` | Standard deviation/variance |
| `mad` | Mean absolute deviation |
| `prod` | Product of items |
| `sum` | Sum of items |

| GroupBy operation | Output shape |
|---|---|
| `aggregate` | Reduced (one row per group) |
| `filter` | Subset of original rows (whole groups kept/dropped) |
| `transform` | Same shape as input |
| `apply` | Flexible (DataFrame, Series, or scalar) |

## Worked Example
Counting discovered exoplanets by detection method and decade: `decade = (10*(planets['year']//10)).astype(str)+'s'`, then `planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)` groups by both method and decade simultaneously, sums the `number` column within each group, pivots decade into columns via `unstack()`, and fills any missing method/decade combos with 0 — producing a clean method × decade count table showing Radial Velocity and Transit as the dominant methods, with Transit surging in the 2010s.

## Key Takeaways
1. `groupby` implements split-apply-combine lazily and efficiently in one pass — no explicit intermediate split is materialized.
2. The GroupBy object supports column indexing, direct iteration (for inspection), and dispatch of any DataFrame/Series method per-group.
3. `aggregate` (reduction), `filter` (group inclusion/exclusion), `transform` (same-shape output), and `apply` (most general) cover essentially all post-split needs.
4. Split keys can be a column name, an external list/array/Series, a dict/Series mapping, any Python function, or a combination thereof (producing multi-index grouping).
5. Combining `groupby` with `unstack()` is a common pattern for turning a multi-key group result into a readable pivoted table.

## Connects To
- **Ch 7**: Pandas aggregation builds directly on NumPy's aggregation functions.
- **Ch 17**: multi-key groupby results and `unstack()` connect to hierarchical indexing.
- **Ch 21**: Pivot Tables are explicitly framed as a convenient generalization of groupby+unstack patterns.
- **Ch 36**: the Planets dataset is loaded via Seaborn, previewing Part IV visualization tools.
