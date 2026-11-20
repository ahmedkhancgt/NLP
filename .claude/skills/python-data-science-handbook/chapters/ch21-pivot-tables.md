# Chapter 21: Pivot Tables

## Core Idea
`pivot_table` is a convenience wrapper for the common pattern of multidimensional groupby + unstack, taking column-wise data and summarizing it into a two-dimensional (or higher) grid, much like a spreadsheet pivot table.

## Frameworks Introduced
- **`DataFrame.pivot_table(values, index, columns, aggfunc='mean', fill_value=, margins=, dropna=, margins_name='All')`**: condenses `groupby([...]).aggregate(...).unstack()` chains into a single readable call.
  - When to use: summarizing one variable across the combination of two (or more) categorical dimensions.
  - How: `values` = column to aggregate, `index`/`columns` = grouping keys (can be lists for multilevel), `aggfunc` = string/function/dict of per-column aggregations.
- **Binning continuous variables for pivoting**: `pd.cut(series, bins)` creates fixed-width bins; `pd.qcut(series, q)` creates quantile-based bins — either can be used as an `index`/`columns` key to turn continuous data into a pivotable categorical dimension.
- **`margins=True`**: adds row/column totals (and grand total) automatically.

## Key Concepts
- **Pivot table as "2D groupby"**: "split-apply-combine, but both split and combine happen across a two-dimensional grid" rather than a single index.
- **Multilevel pivoting**: pass a list to `index` or `columns` for multiple grouping levels (produces hierarchical index/columns, per Ch 17).
- **`aggfunc` as dict**: `aggfunc={'col1': 'sum', 'col2': 'mean'}` auto-infers `values` and applies a different aggregation per column.
- **`margins_name`**: customizes the label for totals row/column (default `'All'`).

## Mental Models
- Pivot tables are the "spreadsheet mental model" applied to Pandas: think of `index` as the row grouping, `columns` as the column grouping, and the cell value as an aggregate of `values` for that row×column combination — directly analogous to Excel/Google Sheets pivot tables.
- Binning with `pd.cut`/`pd.qcut` turns "a continuous variable I want to summarize by range" into "a categorical variable I can pivot on," unlocking the same groupby machinery for continuous data.

## Anti-patterns
- **Writing out the full manual groupby chain (`groupby([...]).aggregate(...).unstack()`) for a simple two-way summary**: works, but is harder to read/maintain — `pivot_table` expresses the same intent far more directly for the common case.
- **Forgetting `aggfunc` defaults to `'mean'`**: silently computing averages when a sum or count was intended — always specify `aggfunc` explicitly when it's not the mean.
- **Not using `margins=True` when totals are needed**: manually computing row/column/grand totals separately is error-prone; `margins=True` computes them consistently in the same call.

## Code Examples
```python
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')

# manual groupby approach (equivalent, but verbose)
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()

# pivot_table equivalent
titanic.pivot_table('survived', index='sex', columns='class', aggfunc='mean')

# multilevel: binned age as extra row dimension
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')

# binned fare (quantiles) as extra column dimension
fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

# dict aggfunc, values inferred automatically
titanic.pivot_table(index='sex', columns='class',
                     aggfunc={'survived': sum, 'fare': 'mean'})

# totals via margins
titanic.pivot_table('survived', index='sex', columns='class', margins=True)

# birthrate example
births = pd.read_csv('data/births.csv')
births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')

births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()

# outlier removal via sigma-clipping + query (preview of Ch 24)
import numpy as np
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000*births.year + 100*births.month + births.day,
                               format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek

births.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean').plot()
```
- **What it demonstrates**: pivot_table as a cleaner alternative to manual groupby+unstack, binning continuous variables for multilevel pivoting, dict-based per-column aggregation, margin totals, and a realistic end-to-end exploratory workflow (birthrate data: clean → bin → pivot → plot).

## Reference Tables
| Parameter | Purpose |
|---|---|
| `values` | Column(s) to aggregate |
| `index` | Row grouping key(s) |
| `columns` | Column grouping key(s) |
| `aggfunc` | Aggregation function(s); string/function/dict; default `'mean'` |
| `fill_value` | Value to use for missing combinations |
| `margins` | Add row/column/grand totals |
| `margins_name` | Label for totals (default `'All'`) |
| `dropna` | Whether to drop columns with all-NaN entries |

## Worked Example
Titanic survival by sex × class: `titanic.pivot_table('survived', index='sex', columns='class', aggfunc='mean')` shows first-class females survived at ~97%, while third-class males survived at only ~13.5% — a direct, two-line illustration of the "class and gender gradient" in survival, compared to the equivalent multi-line manual `groupby(...).aggregate(...).unstack()` chain.

## Key Takeaways
1. `pivot_table` is a convenience wrapper for multidimensional groupby + unstack — functionally equivalent but far more readable for the common 2D (or N-D) summarization case.
2. `pd.cut`/`pd.qcut` convert continuous variables into categorical bins usable as pivot `index`/`columns` keys.
3. `aggfunc` accepts strings, functions, or a dict mapping columns to different aggregations (which also auto-infers `values`).
4. `margins=True` adds row/column/grand totals in the same call.
5. Combining `pivot_table` with Pandas' built-in `.plot()` (via Matplotlib) is a fast way to visualize categorical trends directly from a summarized table.

## Connects To
- **Ch 17**: multilevel pivot results use hierarchical (MultiIndex) rows/columns.
- **Ch 20**: pivot_table is explicitly presented as a convenience layer over groupby's split-apply-combine.
- **Ch 23/Ch 24**: the birthrate example previews datetime indexing (Ch 23) and `.query()` (Ch 24).
- **Ch 32**: the births-by-date plot is revisited there for annotation techniques.
