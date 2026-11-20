# Chapter 27: Simple Scatter Plots

## Core Idea
Scatter plots can be made via `plt.plot` (fast, uniform markers — efficient for large datasets) or `plt.scatter` (flexible, per-point size/color mapping — better for encoding extra data dimensions but slower at scale); error visualization uses `plt.errorbar` for discrete points and `plt.fill_between` for continuous uncertainty bands.

## Frameworks Introduced
- **`plt.plot(x, y, marker_code, ...)` for scatter**: the third positional argument is a marker-style code (`'o'`, `'.'`, `','`, `'x'`, `'+'`, `'v'`, `'^'`, `'<'`, `'>'`, `'s'`, `'d'`, etc.), combinable with line/color codes to plot connected points.
  - When to use: large datasets (identical markers rendered once, fast), or simple point visualization without per-point property variation.
- **`plt.scatter(x, y, c=, s=, alpha=, cmap=)` for data-mapped scatter**: `c=` maps color (often to a third variable, with `plt.colorbar()` for scale), `s=` maps marker size (pixels), `alpha=` sets transparency.
  - When to use: encoding additional data dimensions (color, size) into a 2D scatter — e.g., visualizing 4 dimensions at once (x, y, size, color).
- **Uncertainty visualization**: `plt.errorbar(x, y, yerr=, fmt=)` for discrete points with error bars; `plt.fill_between(x, y_lower, y_upper, alpha=)` for continuous uncertainty bands (e.g., from a Gaussian process regression fit).

## Key Concepts
- **Marker codes**: short string codes analogous to linestyle codes, usable alone or combined with color/linestyle in one shorthand string.
- **`plt.scatter`'s per-point property rendering cost**: must construct each point individually (vs. `plt.plot`'s single-pass identical-marker rendering) — the source of its performance disadvantage at scale.
- **`fmt=` in `errorbar`**: same shorthand syntax as `plt.plot` (color+marker+linestyle combined).
- **`fill_between(x, lower, upper)`**: fills the region between two y-boundaries at each x — the standard way to show continuous (non-discrete-point) uncertainty.
- **Four-dimensional encoding via scatter**: (x, y) position + size (`s=`) + color (`c=`) can jointly encode four variables in one 2D plot (e.g., the Iris dataset example: sepal length/width as position, petal width as size, species as color).

## Mental Models
- `plt.plot` vs. `plt.scatter`: think of `plt.plot` as "stamp the same marker N times" (cheap, uniform) and `plt.scatter` as "draw N custom, individually-configured markers" (expensive, flexible) — choose based on whether you need per-point variation.
- Uncertainty is not optional context — plotting a number without its error bar/band answers an incomplete question; always ask "are these two measurements consistent, given their uncertainties?" rather than just comparing point estimates.

## Anti-patterns
- **Using `plt.scatter` by default for large datasets (thousands+ points) with uniform markers**: unnecessarily slow — use `plt.plot` instead when no per-point size/color variation is needed.
- **Plotting measured values without their uncertainties**: "there is no way to know" whether two measurements are consistent without error bars/bands — always visualize uncertainty when it's available.
- **Rendering 1,000 individual error bars for a continuous fit (e.g., Gaussian process predictions)**: visually overwhelming and slow — use `plt.fill_between` for a continuous uncertainty band instead of `plt.errorbar` at high point density.

## Code Examples
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 30); y = np.sin(x)
plt.plot(x, y, 'o', color='black')              # scatter via plt.plot

rng = np.random.default_rng(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.random(2), rng.random(2), marker, color='black',
              label=f"marker='{marker}'")
plt.legend(numpoints=1, fontsize=13)

plt.scatter(x, y, marker='o')                     # scatter via plt.scatter

x = rng.normal(size=100); y = rng.normal(size=100)
colors = rng.random(100); sizes = 1000 * rng.random(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3)
plt.colorbar()                                        # color scale

from sklearn.datasets import load_iris
iris = load_iris(); features = iris.data.T
plt.scatter(features[0], features[1], alpha=0.4,
            s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0]); plt.ylabel(iris.feature_names[1])

# Errorbars
x = np.linspace(0, 10, 50); dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

# Continuous errors via Gaussian process + fill_between
from sklearn.gaussian_process import GaussianProcessRegressor
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8]); ydata = model(xdata)
gp = GaussianProcessRegressor()
gp.fit(xdata[:, np.newaxis], ydata)
xfit = np.linspace(0, 10, 1000)
yfit, dyfit = gp.predict(xfit[:, np.newaxis], return_std=True)

plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2)
```
- **What it demonstrates**: both scatter methods, marker code variety, data-driven size/color encoding (including a real 4-dimension Iris visualization), and both discrete and continuous uncertainty visualization approaches.

## Reference Tables
| Method | Rendering cost | Per-point customization | Best for |
|---|---|---|---|
| `plt.plot` | Low (one pass) | No (identical markers) | Large datasets, simple scatter |
| `plt.scatter` | Higher (per-point) | Yes (size, color, etc.) | Small/medium datasets needing encoded dimensions |

| Uncertainty type | Tool |
|---|---|
| Discrete points | `plt.errorbar(x, y, yerr=)` |
| Continuous function | `plt.plot` + `plt.fill_between(x, lower, upper)` |

## Worked Example
Iris dataset 4D scatter: `plt.scatter(features[0], features[1], alpha=0.4, s=100*features[3], c=iris.target, cmap='viridis')` encodes sepal length (x), sepal width (y), petal width (point size), and species (color) simultaneously — demonstrating how `plt.scatter`'s `s=`/`c=` mapping extends a 2D plot to represent four data dimensions at once for exploratory visualization.

## Key Takeaways
1. `plt.plot` with a marker-code third argument is the efficient choice for large, uniform scatter plots; `plt.scatter` trades speed for per-point size/color/alpha customization.
2. `plt.scatter`'s `c=`/`s=` arguments let a 2D scatter encode two additional data dimensions (color + size), useful for multidimensional exploratory visualization.
3. For discrete measurements, `plt.errorbar(x, y, yerr=dy, fmt=...)` visualizes uncertainty with the same shorthand format codes as `plt.plot`.
4. For continuous model predictions with uncertainty (e.g., Gaussian process fits), `plt.fill_between` is far more appropriate and legible than plotting thousands of individual error bars.
5. Reporting a measurement without its uncertainty makes comparison claims (e.g., "are these two values consistent?") impossible to answer rigorously.

## Connects To
- **Ch 26**: directly extends line-plot styling conventions (marker/color/linestyle codes) to scatter plots.
- **Ch 30**: colorbar customization is covered in depth there.
- **Ch 38**: Gaussian process regression via scikit-learn previews Part V's modeling API.
- **Ch 36**: Seaborn offers a more compact API for this chapter's continuous-error visualizations.
