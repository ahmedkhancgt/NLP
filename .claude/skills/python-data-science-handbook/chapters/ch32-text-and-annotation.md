# Chapter 32: Text and Annotation

## Core Idea
Good visualizations often need textual cues (`ax.text`) and arrows (`ax.annotate`) to guide the reader, and understanding Matplotlib's three coordinate-transform systems (data, axes-fraction, figure-fraction) is key to placing that text correctly whether it should move with the data or stay fixed.

## Frameworks Introduced
- **`ax.text(x, y, string, **style)`**: places text at a specified position (default: data coordinates), with styling (`size`, `color`, `ha`/horizontal alignment, `va`, etc.).
- **Coordinate transforms**: `transform=` argument selects which coordinate system a position is interpreted in.
  - `ax.transData` (default): standard data coordinates — text moves if axis limits change.
  - `ax.transAxes`: fraction of the axes box (0 to 1 from bottom-left) — stays fixed relative to the axes regardless of data/limits.
  - `fig.transFigure`: fraction of the whole figure (0 to 1 from bottom-left) — stays fixed relative to the figure.
- **`ax.annotate(text, xy=, xycoords=, xytext=, textcoords=, arrowprops=, bbox=)`**: the preferred way to add an arrow-plus-text annotation (vs. the fragile `plt.arrow`), with highly configurable `arrowprops` (style, connection curve, color) and optional `bbox` (background box).

## Key Concepts
- **`ha`/`va`**: horizontal/vertical alignment keywords for text placement relative to the given coordinate.
- **Mixed coordinate systems**: `xycoords='data'` for the point being annotated plus `textcoords='offset points'` for the label's offset from that point — a common `annotate` pattern allowing a precise anchor plus a readable offset.
- **`arrowprops` dict**: keys like `arrowstyle` (`'->'`, `'fancy'`, `'wedge'`, `'|-|'`), `connectionstyle` (`'arc3,rad=...'`, `'angle3,...'`, `'angle,...'`) control arrow shape/curvature.
- **`plt.arrow` vs. `plt.annotate`**: `plt.arrow` creates SVG objects sensitive to the plot's varying aspect ratio (fragile/hard to get right); `plt.annotate` is the recommended, far more flexible alternative.
- **Interactive transform behavior**: changing axis limits only shifts `transData`-positioned elements; `transAxes`/`transFigure`-positioned elements stay stationary — visible directly when using `%matplotlib notebook`'s interactive pan/zoom.

## Mental Models
- Three coordinate systems answer three different questions about "where": "where on the data scale" (transData, moves with axis limits), "where on this specific axes box" (transAxes, fixed fraction of the plot area), "where on the entire figure" (transFigure, fixed fraction of the whole canvas including multiple subplots) — pick based on whether the text should track the data or stay visually anchored.
- `annotate`'s split between `xy`(the thing being pointed at) and `xytext`(where the label text sits) mirrors how a human would annotate a printed chart: point precisely at a feature, but place the label wherever it's legible, with an arrow connecting the two.

## Anti-patterns
- **Using `plt.arrow` for annotation arrows**: produces SVG shapes distorted by the plot's aspect ratio — use `plt.annotate` with `arrowprops` instead.
- **Positioning a figure-level caption or label with data coordinates (`transData`)**: it will shift whenever axis limits change — use `transAxes` or `transFigure` for elements meant to stay visually fixed regardless of the data view.
- **Treating heavily-tweaked `annotate` arrow styling as a template to copy wholesale**: the book explicitly notes this kind of intricate manual styling "is by no means best practice" and is shown only to demonstrate available options — favor simpler, consistent annotation choices in real work, reserving elaborate styling for cases that truly need it.

## Code Examples
```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# basic text annotation anchored to data coordinates
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.set(title='USA births by day of year (1969-1988)', ylabel='average daily births')
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

# coordinate transforms
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure)

# annotate with arrows
fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90"))

# richly styled annotations (birthrate example)
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
ax.annotate("New Year's Day", xy=('2012-1-1', 4100), xycoords='data',
            xytext=(50, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))
ax.annotate("Independence Day", xy=('2012-7-4', 4250), xycoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"),
            xytext=(10, -40), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
            xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2'})
ax.annotate('Halloween', xy=('2012-10-31', 4600), xycoords='data',
            xytext=(-80, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle="fancy", fc="0.6", ec="none",
                             connectionstyle="angle3,angleA=0,angleB=-90"))
```
- **What it demonstrates**: anchored-to-data text annotation with date-axis formatting, all three coordinate transforms contrasted, basic arrow annotation, and a richly styled multi-annotation real-world example (US births by holiday).

## Reference Tables
| Transform | Coordinate basis | Behavior on axis-limit change |
|---|---|---|
| `ax.transData` (default) | Data values | Moves with the data |
| `ax.transAxes` | Fraction of axes box [0,1] | Fixed relative to axes |
| `fig.transFigure` | Fraction of figure [0,1] | Fixed relative to figure |

| `annotate` key | Purpose |
|---|---|
| `xy`, `xycoords` | Point being annotated + its coordinate system |
| `xytext`, `textcoords` | Label text position + its coordinate system (often `'offset points'`) |
| `arrowprops` | Arrow style/curvature dict |
| `bbox` | Optional background box around the text |

## Worked Example
US births-by-holiday annotation: the fully annotated version of the births-by-date plot (first introduced in Ch 21) adds six holiday labels via a mix of `ax.text` (simple cases) and `ax.annotate` with varied `arrowprops` (curved arcs, angle connectors, fancy/wedge styles) and `bbox` boxes — illustrating the full range of styling knobs available, while the chapter explicitly cautions that this particular mix of styles is a feature demonstration, not a best-practice template.

## Key Takeaways
1. `ax.text(x, y, s, **style)` places text, defaulting to data coordinates (`transData`).
2. Three coordinate transforms — `transData`, `transAxes`, `transFigure` — determine whether a text/annotation position tracks the data, the axes box, or the whole figure.
3. `ax.annotate` (not `plt.arrow`) is the recommended way to add arrows with text, via a rich `arrowprops` dict controlling style and curvature.
4. `xycoords`/`textcoords` let the annotated point and the label text live in different coordinate systems (commonly `'data'` + `'offset points'`).
5. Elaborate manual annotation styling, while powerful, is time-consuming to get right and is not inherently "best practice" — match styling effort to the communication need.

## Connects To
- **Ch 21**: the births-by-date dataset and base plot are reused directly from the pivot-table example.
- **Ch 26**: builds on basic title/label conventions (`ax.set`).
- **Ch 33**: tick locators/formatters (`mpl.dates.MonthLocator`, etc.) used here for the date axis are covered in full in the next chapter.
