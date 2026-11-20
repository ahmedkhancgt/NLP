# Chapter 3: Visualizing Data

## Core Idea
Data visualization serves two purposes — exploring data and communicating it — and `matplotlib.pyplot` is the book's chosen (if "showing its age") tool for both; the chapter is a practical tour of line charts, bar charts, histograms, and scatterplots plus the common ways each can mislead.

## Frameworks Introduced
- **matplotlib.pyplot stateful plotting**: pyplot maintains internal state you build up call by call (`plt.plot`, `plt.bar`, `plt.scatter`, labels/titles/ticks), then `savefig`/`show`.
  - When to use: quick exploratory and simple publication-grade static charts (not interactive web visualizations — use D3.js/Bokeh for those).
- **Chart-type selection rule**: bar chart for a discrete set of items' values or a histogram of bucketed values; line chart for trends over an ordered/continuous axis; scatterplot for the relationship between two paired variables.
  - How: match the data shape (discrete categories vs. ordered sequence vs. paired continuous values) to the chart type before worrying about styling.

## Key Concepts
- **Histogram-via-bar-chart**: bucket continuous values (e.g., by decile with `grade // 10 * 10`) with `Counter`, then render as a bar chart.
- **Misleading y-axis**: truncating the y-axis (not starting at 0) on a bar chart exaggerates differences — explicitly called "bad form."
- **Uncomparable axes**: letting matplotlib auto-scale two axes differently can misrepresent a scatterplot; fix with `plt.axis("equal")`.
- **plt.annotate**: labels individual scatter points at an offset from their coordinates.
- **plt.legend**: auto-generates a legend from each series' `label=` argument.

## Mental Models
- **Exploration vs. communication**: the same chart-building skill serves two different goals — quick exploratory sanity checks on your own data, versus a polished chart meant to convince or inform someone else; the latter demands far more care about what the axes imply.
- **Chart honesty checklist**: before publishing a chart, ask whether the y-axis starts at zero (for bar charts) and whether the axes are on comparable scales (for scatterplots) — both are easy, common ways to accidentally (or deliberately) mislead.

## Anti-patterns
- **Truncated y-axis on a bar chart**: `plt.axis([2016.5, 2018.5, 499, 506])` on data `[500, 505]` makes a trivial 1% change look like a "huge increase" — a textbook example of misleading data viz.
- **Letting matplotlib auto-scale unequal axes for comparable variables**: two test-score axes with different implicit scales make variance look similar when it isn't; use `plt.axis("equal")`.
- **Using a bar chart with a non-zero-based axis for any comparison a viewer could misread.**

## Code Examples
```python
from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
```
- **What it demonstrates**: a basic line chart with explicit styling.

```python
from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)
plt.bar([x + 5 for x in histogram.keys()], histogram.values(), 10, edgecolor=(0, 0, 0))
plt.axis([-5, 105, 0, 5])
plt.xticks([10 * i for i in range(11)])
```
- **What it demonstrates**: building a histogram from scratch via `Counter` bucketing, then rendering with `plt.bar`.

```python
friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
plt.scatter(friends, minutes)
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label, xy=(friend_count, minute_count), xytext=(5, -5), textcoords='offset points')
```
- **What it demonstrates**: labeled scatterplot via `zip` over three parallel lists and `plt.annotate`.

## Reference Tables
| Chart type | Best for | matplotlib call |
|---|---|---|
| Bar chart | discrete-item comparison, or histogram of bucketed numerics | `plt.bar` |
| Line chart | trend over an ordered axis, multiple series | `plt.plot` (+ `label=`, `plt.legend`) |
| Scatterplot | relationship between two paired continuous variables | `plt.scatter` (+ `plt.annotate`, `plt.axis("equal")`) |

## Worked Example
DataSciencester's `friends` vs. `minutes` data (9 users, friend counts 60-72 vs. minutes 105-220) is plotted as a labeled scatterplot to visually inspect the friends/time-on-site relationship that Chapter 1 touched on numerically — foreshadowing the regression work of Ch14. A parallel "exam grades" histogram example buckets 13 students' scores by decile (with 100 folded into the 90s bucket) to show the distribution shape.

## Key Takeaways
1. Match chart type to data shape: bar for discrete/bucketed values, line for trends, scatter for paired relationships.
2. Always start bar-chart y-axes at 0 unless you have a specific, disclosed reason not to.
3. Use `plt.axis("equal")` when comparing variables on the same underlying scale (e.g., two test scores).
4. `Counter` + `plt.bar` is the from-scratch recipe for histograms — no numpy/pandas binning needed.
5. matplotlib is adequate for this book's exploratory needs but not for interactive web visualization (seaborn, Altair, D3.js, Bokeh are better for that).

## Connects To
- **Ch1**: revisits the `friends`/`minutes` dataset used informally in the introduction, now actually plotted.
- **Ch14**: the scatterplot of friends vs. minutes previews the simple linear regression fit to this exact relationship.
- **matplotlib / seaborn / Altair / D3.js / Bokeh**: external visualization libraries the chapter explicitly points to for anything beyond simple static charts.
