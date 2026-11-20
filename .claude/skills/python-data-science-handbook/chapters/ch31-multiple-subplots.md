# Chapter 31: Multiple Subplots

## Core Idea
Matplotlib offers four increasingly convenient tools for arranging multiple axes within one figure — manual `plt.axes`, single-subplot `plt.subplot`, whole-grid `plt.subplots`, and flexible-span `plt.GridSpec` — matching the complexity of the subplot-creation tool to the complexity of the desired layout.

## Frameworks Introduced
- **`plt.axes([left, bottom, width, height])` / `fig.add_axes(...)`**: manual axes placement in figure-fraction coordinates (0 to 1); used for insets or precisely touching stacked panels.
- **`plt.subplot(nrows, ncols, index)` / `fig.add_subplot(...)`**: creates one subplot at a time within an implied grid; `index` runs left-to-right, top-to-bottom, 1-based (MATLAB-style); `plt.subplots_adjust(hspace=, wspace=)` controls spacing.
- **`plt.subplots(nrows, ncols, sharex=, sharey=)`**: creates the entire grid in one call, returning a NumPy array of Axes (0-based indexing, `ax[i, j]`); `sharex='col'`/`sharey='row'` links axis scales and auto-removes redundant inner tick labels.
- **`plt.GridSpec(nrows, ncols, wspace=, hspace=)`**: a layout specification object (not a plot itself) that `plt.subplot`/`fig.add_subplot` can consume with slice syntax (`grid[0, 1:]`) to create subplots spanning multiple rows/columns — the tool for irregular/complex arrangements like the classic "scatter + marginal histograms" layout.

## Key Concepts
- **Figure-fraction coordinates**: `[left, bottom, width, height]`, each in [0, 1], relative to the whole figure.
- **1-based (`plt.subplot`) vs. 0-based (`plt.subplots`/`GridSpec`) indexing**: `plt.subplots` and `GridSpec` indexing is more "Pythonic"/consistent with standard array indexing than `plt.subplot`'s MATLAB-style numbering.
- **`sharex`/`sharey`**: links axis limits/scales across subplots in the same row/column and removes duplicate tick labels for a cleaner grid.
- **GridSpec slicing**: `grid[0, 0]`, `grid[0, 1:]`, `grid[1, :2]`, `grid[1, 2]` — standard Python slice syntax applied to the grid specification to span cells.
- **Marginal-histogram layout pattern**: a `GridSpec`-based main scatter axes plus two attached histogram axes (top/side), each `sharex`/`sharey` with the main axes and with inverted axis direction for proper visual orientation — the canonical multi-panel distribution visualization.

## Mental Models
- Choose the subplot tool by layout complexity: one inset → `plt.axes`; a single plot within an implied grid, one at a time → `plt.subplot`; a complete regular grid at once → `plt.subplots`; cells spanning multiple rows/columns (irregular layout) → `GridSpec`.
- `GridSpec` is a *specification*, not a drawable object — it must be consumed by `plt.subplot`/`fig.add_subplot` to actually produce axes; this separation is what allows flexible slicing-based spans.

## Anti-patterns
- **Manually creating and positioning many axes via `plt.axes` for a regular grid**: tedious and error-prone — use `plt.subplots` for regular grids instead.
- **Using `plt.subplot`'s one-based MATLAB-style indexing conventions when writing idiomatic, loopable Python code**: inconsistent with zero-based array indexing elsewhere — prefer `plt.subplots`' returned 0-indexed Axes array when iterating programmatically.
- **Manually hiding redundant tick labels on a shared-axis grid**: `sharex`/`sharey` in `plt.subplots` already does this automatically — don't duplicate the effort.

## Code Examples
```python
import matplotlib.pyplot as plt
import numpy as np

# plt.axes: manual placement
ax1 = plt.axes()                          # standard full-figure axes
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])      # inset axes

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x)); ax2.plot(np.cos(x))

# plt.subplot: one at a time, 1-based
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')

# plt.subplots: whole grid, 0-based, shared axes
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')

# GridSpec: irregular spans
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])

# marginal histogram layout
mean = [0, 0]; cov = [[1, 1], [1, 2]]
rng = np.random.default_rng(1701)
x, y = rng.multivariate_normal(mean, cov, 3000).T

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
y_hist.invert_xaxis()
```
- **What it demonstrates**: all four layout tools from simplest to most flexible, culminating in the GridSpec-based marginal-histogram pattern combining shared axes, inverted orientation, and span-based grid cells.

## Reference Tables
| Tool | Granularity | Indexing | Best for |
|---|---|---|---|
| `plt.axes` | One axes, manual coords | Figure fraction [0,1] | Insets, precise manual placement |
| `plt.subplot` | One axes in implied grid | 1-based | Simple loops over a regular grid |
| `plt.subplots` | Whole grid at once | 0-based, NumPy array | Regular grids, especially with shared axes |
| `plt.GridSpec` | Layout spec, slice-based | 0-based, slice syntax | Irregular spans (multi-row/col cells) |

## Worked Example
Scatter plot with marginal histograms: a `4×4 GridSpec` allocates the bottom-left 1×1 cell unused, the top-right 3×3 block to `main_ax` (scatter), the left column (top 3 rows) to `y_hist` (horizontal histogram, x-axis inverted), and the bottom row (right 3 columns) to `x_hist` (vertical histogram, y-axis inverted) — `sharey=main_ax`/`sharex=main_ax` keep axes aligned, producing the classic "distribution + its margins" visualization in ~15 lines, the same pattern later exposed as a one-line API in Seaborn (jointplot).

## Key Takeaways
1. Four subplot tools scale with layout complexity: `plt.axes` (manual) → `plt.subplot` (one grid cell at a time, 1-based) → `plt.subplots` (whole regular grid, 0-based array) → `plt.GridSpec` (irregular spans via slicing).
2. `plt.subplots`' `sharex`/`sharey` link axis scales across a grid and auto-hide redundant inner tick labels.
3. `GridSpec` is a layout specification consumed by `plt.subplot`/`fig.add_subplot`, not a plot object itself — it enables slice-based multi-cell spans.
4. The marginal-histogram layout (main scatter + attached top/side histograms via `GridSpec` + `sharex`/`sharey` + axis inversion) is a widely reusable multi-panel visualization pattern.

## Connects To
- **Ch 25**: builds on the Figure/Axes object-oriented interface established there.
- **Ch 26**: `ax.set()` property-bundling technique applies equally within any subplot grid.
- **Ch 36**: Seaborn's `jointplot` provides a one-line equivalent of the GridSpec marginal-histogram pattern.
