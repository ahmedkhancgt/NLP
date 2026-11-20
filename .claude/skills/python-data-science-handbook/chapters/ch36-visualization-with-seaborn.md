# Chapter 36: Visualization with Seaborn

## Core Idea
Seaborn is a high-level statistical visualization API built on Matplotlib that integrates natively with Pandas DataFrames, provides sane style/color defaults, and compresses common statistical plot patterns (joint/pair/faceted/categorical distributions) into single function calls.

## Frameworks Introduced
- **`sns.set()`**: applies Seaborn's default chart style globally.
- **Distribution plots**: `sns.kdeplot(data=, shade=)` (1D or 2D smooth density via KDE, Ch 28); `sns.displot(data, kde=)` (histogram/distribution).
- **Multivariate relationship plots**: `sns.pairplot(df, hue=)` (all pairwise scatter/histograms across columns — the N-dimensional generalization of a joint plot); `sns.jointplot(x=, y=, data=, kind='hex'/'reg'/...)` (joint 2D distribution + marginals, optionally with KDE or regression fit); `sns.PairGrid(df, vars=, hue=).map(plt.scatter)` (lower-level customizable pairwise grid).
- **Categorical/faceted plots**: `sns.FacetGrid(df, row=, col=).map(plt.hist, 'col')` (histograms split by categorical subgroups); `sns.catplot(x=, y=, hue=, data=, kind='box'/'count'/...)` (categorical distribution comparison — boxplots, counts, etc.); `sns.violinplot(x=, y=, hue=, data=, split=)` (distribution shape comparison, optionally split by a second category); `sns.lmplot(x=, y=, col=, data=)` (automatic linear regression fit per facet).

## Key Concepts
- **Why Seaborn over raw Matplotlib**: addresses three specific gaps — (historically) poor default styling, low-level/boilerplate-heavy API for statistical plots, and no native DataFrame-label awareness (pre-dating Pandas).
- **"Seaborn does this under the hood with Matplotlib"**: every Seaborn plot is ultimately composed of ordinary Matplotlib calls — Seaborn is a convenience layer, not a separate rendering engine, so Matplotlib customization (Ch 25-34) still applies to Seaborn figures.
- **`hue=`**: the standard Seaborn parameter for splitting/coloring a plot by a categorical column — used across `pairplot`, `catplot`, `PairGrid`, `lmplot`.
- **`sns.axes_style(...)` context manager**: temporarily applies a Seaborn style (parallel to Matplotlib's `plt.style.context`) scoped to a block.
- **DataFrame column references by name**: Seaborn functions take `data=df` plus column-name strings (`x="day"`, `y="total_bill"`) rather than requiring pre-extracted Series — the key Pandas-integration advantage over raw Matplotlib.

## Mental Models
- Seaborn plot functions correspond to standard EDA questions: "what's the distribution of one variable?" (kdeplot/displot), "how do two variables relate, plus their marginals?" (jointplot), "how do all variables relate pairwise?" (pairplot), "how does a distribution vary across a categorical split?" (FacetGrid/catplot/violinplot), "is there a relationship I should fit a regression to, possibly per subgroup?" (lmplot).
- Think of `pairplot` as "jointplot generalized to N dimensions" and `FacetGrid`/`catplot` as "histogram/boxplot generalized across categorical subgroups" — Seaborn's naming/API consistently extends simpler 1D/2D concepts to more dimensions or more groups.

## Anti-patterns
- **Manually extracting and concatenating Series from a DataFrame to feed raw Matplotlib functions**: exactly the tedious friction Seaborn eliminates — pass the DataFrame and column names directly to Seaborn functions instead.
- **Interpreting a small-sample subgroup result (e.g., 7 runners over age 80) as a robust finding**: always sanity-check group sizes before drawing conclusions from faceted/categorical comparisons — the marathon example explicitly flags this ("likely a small number effect").
- **Forgetting Seaborn plots are still Matplotlib underneath**: failing to use standard Matplotlib customization (labels, limits, annotations) on a Seaborn-generated figure when more control is needed.

## Code Examples
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set()

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

sns.kdeplot(data=data, shade=True)              # 1D smooth densities
sns.kdeplot(data=data, x='x', y='y')              # 2D joint KDE (conceptually)

iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species', height=2.5)      # all pairwise relationships

tips = sns.load_dataset('tips')
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))

with sns.axes_style(style='ticks'):
    g = sns.catplot(x="day", y="total_bill", hue="sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill")

with sns.axes_style('white'):
    sns.jointplot(x="total_bill", y="tip", data=tips, kind='hex')
sns.jointplot(x="total_bill", y="tip", data=tips, kind='reg')   # KDE + regression

planets = sns.load_dataset('planets')
with sns.axes_style('white'):
    g = sns.catplot(x="year", data=planets, aspect=2, kind="count", color='steelblue')
    g.set_xticklabels(step=5)

# Marathon example
data = pd.read_csv('data/marathon-data.csv',
                    converters={'split': convert_time, 'final': convert_time})
data['split_sec'] = data['split'].view(int) / 1E9
data['final_sec'] = data['final'].view(int) / 1E9
data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']

sns.displot(data['split_frac'], kde=False)

g = sns.PairGrid(data, vars=['age', 'split_sec', 'final_sec', 'split_frac'],
                  hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend()

sns.kdeplot(data.split_frac[data.gender == 'M'], label='men', shade=True)
sns.kdeplot(data.split_frac[data.gender == 'W'], label='women', shade=True)

sns.violinplot(x="gender", y="split_frac", data=data, palette=["lightblue", "lightpink"])

data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))
with sns.axes_style(style=None):
    sns.violinplot(x="age_dec", y="split_frac", hue="gender", data=data,
                    split=True, inner="quartile", palette=["lightblue", "lightpink"])

g = sns.lmplot(x='final_sec', y='split_frac', col='gender', data=data,
               markers=".", scatter_kws=dict(color='c'))
g.map(plt.axhline, y=0.0, color="k", ls=":")
```
- **What it demonstrates**: the full range of Seaborn's statistical plot types across three real datasets (synthetic multivariate normal, Iris, tips, planets, marathon), culminating in a complete exploratory analysis workflow on marathon split data.

## Reference Tables
| Seaborn function | Purpose |
|---|---|
| `sns.kdeplot` | Smooth 1D/2D density estimate |
| `sns.displot` | Histogram/distribution plot |
| `sns.pairplot` | All pairwise relationships across columns |
| `sns.jointplot` | 2D joint distribution + marginals (optionally KDE/regression) |
| `sns.FacetGrid` | Histograms/plots split by categorical row/col |
| `sns.catplot` | Categorical distribution comparison (box/count/etc.) |
| `sns.violinplot` | Distribution shape comparison, optionally split |
| `sns.PairGrid` | Customizable pairwise grid (lower-level than pairplot) |
| `sns.lmplot` | Automatic linear regression fit, faceted |

## Worked Example
Marathon split-fraction analysis: compute `split_frac = 1 - 2*split_sec/final_sec` (negative = ran the second half faster); `sns.violinplot(x="age_dec", y="split_frac", hue="gender", data=data, split=True, inner="quartile")` reveals men aged 20s-50s skew toward lower (faster-second-half) splits more than women of any age; `sns.lmplot(x='final_sec', y='split_frac', col='gender', data=data)` with regression lines shows that, for both genders, runners finishing under ~4 hours (15,000s) are far more likely to have negative splits than slower runners — demonstrating a complete exploratory workflow from raw CSV through feature engineering to multi-faceted statistical visualization.

## Key Takeaways
1. Seaborn is a convenience layer over Matplotlib with better defaults and native Pandas DataFrame support (pass `data=df` + column-name strings).
2. `kdeplot`/`displot` (distributions) → `jointplot` (2 variables + marginals) → `pairplot` (N variables pairwise) form a natural dimensional progression for exploring relationships.
3. `FacetGrid`/`catplot`/`violinplot` enable comparing distributions across categorical subgroups, with `hue=`/`split=`/`col=`/`row=` as the standard faceting parameters.
4. `lmplot` auto-fits linear regressions per facet — useful for spotting group-wise trend differences at a glance.
5. Every Seaborn plot is built from ordinary Matplotlib calls, so Matplotlib's customization tools (Ch 25-34) remain fully applicable to Seaborn figures.
6. Always check subgroup sample sizes before interpreting faceted/categorical comparison results.

## Connects To
- **Ch 20**: the Planets dataset is reused directly from the groupby chapter.
- **Ch 28**: `sns.kdeplot` is the Seaborn-level wrapper around kernel density estimation introduced there.
- **Ch 31**: `jointplot`'s main+marginal layout is the one-line equivalent of the manual GridSpec pattern from Ch 31.
- **Ch 42**: linear regression fitting here previews the dedicated regression chapter in Part V.
