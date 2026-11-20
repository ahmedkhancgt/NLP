# Chapter 8: Datavis: Choosing the Right Plot to Deliver a Message

## Core Idea
Every visualization choice (plot type, color, grid, font) should be judged by one question: does it help convey the intended message? If not, change it.

## Frameworks Introduced
- **Bars vs. Lines Decision Rule**: bars for categorical data (no inherent ordering, easy height comparison); lines for continuous/time-series data (sequential ordering, trend, mean level, curvature). Lines on categorical data falsely imply connection between categories; bars on long time series create clutter.
  - When to use: always ask what property you need to show (comparison vs. trend) before picking bar or line.
- **Slopegraph**: combines bars and lines — shows trend over two points for multiple categorical segments simultaneously, enabling cross-segment trend comparison.
  - When to use: need a trend message for categorical/segment data (e.g., how did each segment change between two periods).
- **Waterfall Chart**: decomposes a change in a metric into segment/category contributions (used for Ch 3's decompositions).
  - When to use: message is specifically about a decomposition; watch for one segment's very different scale distorting the chart.
- **Data-Ink Ratio** (Tufte): ratio of "data-ink" (non-erasable core information) to total ink in a graphic; treat as a North Star, not an absolute law — some redundant ink (e.g., annotations) can aid comprehension.
  - When to use: as a check against clutter; remove any element not conveying information, but allow deliberate exceptions for audience guidance.

## Key Concepts
- **Scatterplot smoother**: nonlinear fit line added to a scatterplot to reveal relationships obscured by density; always show the underlying data too (graphical integrity) so viewers can judge the smoother's validity.
- **Hexagonal bin plot / contour plot**: density-based alternatives to raw scatterplots for very large datasets (millions of points).
- **KDE (Kernel Density Estimate)**: smoothed distribution estimate depending on a kernel function and bandwidth; integrates to 1 so the y-axis scale is often meaningless — typically drop vertical axis labels.
- **Box plot vs. quantile line plot**: box plots carry more information than usually needed; a reduced quantile line plot (min, 25/50/75%, max) can convey distributional shift more cleanly.
- **Bubble plot**: adds a third dimension (Z) to a scatterplot via marker diameter.
- **Redundant color encoding**: using different colors per bar when the x-axis label already distinguishes categories — adds no information and risks implying a nonexistent extra variable.

## Mental Models
- "Is this plot helping me convey the message?" — the single governing question for every datavis decision.
- Data-ink ratio as guidance, not dogma: deliberate extra ink (highlighting, annotation) is acceptable when it directs attention to the message.
- Customization > semiautomation: invest in flexible tools (e.g., Matplotlib) over point-and-click defaults for long-run visualization quality.

## Anti-patterns
- **Using lines for categorical comparisons**: implies false continuity/connection between unordered categories.
- **Using bars for long time series**: creates unnecessary visual clutter versus a line.
- **Multi-colored bars with no third variable**: implies a nonexistent dimension to viewers.
- **Interactive plots for simple messages**: overshoot for most use cases; reserve for when the audience genuinely needs to self-inspect data.
- **Starting the explanation with "insights" before explaining the plot itself**: always first state what's on each axis and walk through one mark before interpreting.
- **Small font sizes**: a very common, easily avoidable DS presentation mistake.

## Code Examples
```python
# set plotting parameters from the beginning
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size' : 14}
axes = {'titlesize' : 22,
        'labelsize' : 20}
figure = {'figsize': (10,4),
          'autolayout': True}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)
matplotlib.rc('figure', **figure)
```
- **What it demonstrates**: setting global Matplotlib rcParams up front to guarantee readable font sizes and consistent figure sizing across a notebook, overridable per-plot as needed.

## Reference Tables
| Message type | Preferred plot |
|---|---|
| Compare categories | Bar plot (single color unless a 3rd variable exists) |
| Trend over time (continuous) | Line plot |
| Trend across categorical segments | Slopegraph |
| Decomposition of a change | Waterfall chart |
| Correlation, large N | Random sample scatter, hexbin, contour, or scatterplot smoother (with raw data shown) |
| Distribution (analyst use) | Histogram / KDE |
| Distribution shift (stakeholder use) | Box plot or reduced quantile line plot |
| 3-variable relationship | Bubble plot |

## Worked Example
Scatterplot for a simulated 10-million-row dataset with a quadratic generating process: plotting all points obscures the relationship; plotting a random sample reveals it cleanly; plotting the full data with a cubic scatterplot smoother overlay shows the fit while still letting viewers judge its validity against the raw data points underneath. Color-redundancy example: bar chart of a metric across 4 named segments — version with one bar per color (bad, implies extra variable), version with uniform color (clean), version that selectively highlights one underperforming segment in a different color (good — directs attention to the message).

## Key Takeaways
1. Every plot must serve a specific message; drop plots that don't.
2. Match plot type to data structure: categorical → bars, continuous/time → lines, segment trends → slopegraphs, decompositions → waterfalls.
3. For large-N scatterplots, show both the smoother and the underlying data (sampled if needed) to preserve graphical integrity.
4. Use color purposefully — never redundantly with axis labels, and use it to direct attention when there's a specific message.
5. Maximize the data-ink ratio as a guideline, with deliberate exceptions for attention-directing annotations.
6. Always set large enough font sizes and label axes/units; start any explanation by describing the plot, not jumping to the insight.
7. Prefer customizable tools (e.g., Matplotlib) over rigid semiautomated defaults for long-term visualization quality.
8. Avoid interactive plots unless the audience truly needs to self-inspect the data.

## Connects To
- **Ch 3**: Waterfall charts are the primary visualization for additive/multiplicative/mix-rate growth decompositions.
- **Ch 7**: Datavis clarity principles are a direct extension of the narrative-clarity principles from Chapter 7.
- **External concept**: Tufte's graphical integrity and data-ink ratio; Wilkinson's Grammar of Graphics (basis for ggplot).
