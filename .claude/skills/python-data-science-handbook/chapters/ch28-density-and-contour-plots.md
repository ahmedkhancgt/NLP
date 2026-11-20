# Chapter 28: Density and Contour Plots

## Core Idea
Three-dimensional (z = f(x,y)) and density data can be visualized in 2D via contour lines (`plt.contour`), filled contour regions (`plt.contourf`), smooth images (`plt.imshow`), and histogram/KDE-based density estimates (1D and 2D), each with distinct trade-offs in smoothness, performance, and interpretability.

## Frameworks Introduced
- **Contour family for 3D-in-2D visualization**: `plt.contour(X, Y, Z, levels, cmap=)` (lines; dashed=negative/solid=positive by default), `plt.contourf(...)` (filled regions, pairs well with `plt.colorbar()`), `plt.imshow(Z, extent=, origin=, interpolation=, aspect=)` (smooth image rendering, needs manual extent/origin care), combinable (transparent `imshow` + labeled `contour` via `plt.clabel`).
  - When to use `np.meshgrid`: build 2D coordinate grids from 1D x/y arrays as required input for contour/image functions.
- **Density estimation family**: `plt.hist`/`np.histogram` (1D), `plt.hist2d`/`np.histogram2d` (2D square bins), `plt.hexbin` (2D hexagonal bins), `scipy.stats.gaussian_kde` (kernel density estimation — smooth density from scattered points).
  - When to use KDE over histograms: when a smooth density estimate (rather than discretely binned counts) is desired, at the cost of needing to choose a smoothing bandwidth.

## Key Concepts
- **`np.meshgrid(x, y)`**: builds 2D `X`, `Y` coordinate grids from 1D arrays, for evaluating `Z = f(X, Y)` across a plotting grid.
- **Divergent colormaps (e.g., `'RdGy'`)**: appropriate for data with meaningful positive/negative variation around zero.
- **`plt.imshow` gotchas**: (1) doesn't accept x/y grids — must specify `extent=[xmin,xmax,ymin,ymax]` manually; (2) default origin is upper-left (image convention), not lower-left (plot convention) — must set `origin='lower'` for gridded data; (3) auto-adjusts aspect ratio — override with `aspect=`.
- **`interpolation='gaussian'`** (or other): smooths `imshow` output, avoiding the "splotchy" discrete-step look of high-contour-count `contourf`.
- **`plt.clabel`**: adds inline numeric labels directly on contour lines.
- **1D vs 2D histogram counterparts**: `plt.hist`↔`np.histogram`; `plt.hist2d`↔`np.histogram2d`; `np.histogramdd` generalizes to N dimensions.
- **KDE smoothing length**: a bias-variance trade-off knob — `gaussian_kde` uses a rule-of-thumb default; alternative implementations exist in scikit-learn (`KernelDensity`) and statsmodels.

## Mental Models
- Contour/image functions are "ways of flattening a 3rd dimension (z) into 2D visual encoding" — lines (contour), filled regions (contourf), or continuous color (imshow) are three points on a discreteness-to-smoothness spectrum, each trading rendering cost against visual continuity.
- Think of `plt.hist`/`hist2d`/`hexbin` vs. KDE as "binned/discrete density estimate" vs. "smoothed/continuous density estimate" — binning is simple and fast but blocky; KDE is smoother but requires a bandwidth choice and more computation.

## Anti-patterns
- **Cranking up contour levels to get a smoother-looking `contourf` plot**: inefficient — Matplotlib renders a new polygon per level, and discreteness persists at finer granularity; use `plt.imshow` with `interpolation=` for true smoothness instead.
- **Using `plt.imshow` on gridded x/y data without setting `origin='lower'`**: default image convention (origin upper-left) will display the data vertically flipped relative to standard plot conventions.
- **Plotting 1,000+ individual hexbin/histogram bins when a smooth KDE would communicate the density better**: know when binned vs. smoothed density representation better suits the audience/purpose.

## Code Examples
```python
import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50); y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, colors='black')          # line contours (dashed=neg, solid=pos)
plt.contour(X, Y, Z, 20, cmap='RdGy')           # 20 levels, colormap

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy',
           interpolation='gaussian', aspect='equal')
plt.colorbar()

contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)
plt.colorbar()

# 1D histogram
rng = np.random.default_rng(1701)
data = rng.normal(size=1000)
plt.hist(data)
plt.hist(data, bins=30, density=True, alpha=0.5, histtype='stepfilled', color='steelblue')

counts, bin_edges = np.histogram(data, bins=5)    # compute without plotting

x1 = rng.normal(0, 0.8, 1000); x2 = rng.normal(-2, 1, 1000); x3 = rng.normal(3, 2, 1000)
kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
plt.hist(x1, **kwargs); plt.hist(x2, **kwargs); plt.hist(x3, **kwargs)

# 2D histogram
mean = [0, 0]; cov = [[1, 1], [1, 2]]
x, y = rng.multivariate_normal(mean, cov, 10000).T
plt.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar()
counts, xedges, yedges = np.histogram2d(x, y, bins=30)   # compute-only

plt.hexbin(x, y, gridsize=30, cmap='Blues')
plt.colorbar()

# Kernel density estimation
from scipy.stats import gaussian_kde
data = np.vstack([x, y])
kde = gaussian_kde(data)
xgrid = np.linspace(-3.5, 3.5, 40); ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
```
- **What it demonstrates**: the full contour/image spectrum for 3D-in-2D plotting (lines → filled → smooth image → combined), the 1D/2D/hexbin histogram family with compute-only counterparts, and kernel density estimation as a smooth alternative.

## Reference Tables
| Function | Output | Note |
|---|---|---|
| `plt.contour` | Lines | Dashed=negative, solid=positive (single color) |
| `plt.contourf` | Filled regions | Discrete color steps; pairs with `colorbar` |
| `plt.imshow` | Smooth image | Needs `extent`, `origin='lower'`, `aspect` care |
| `plt.hist`/`np.histogram` | 1D binned counts | — |
| `plt.hist2d`/`np.histogram2d` | 2D square-binned counts | — |
| `plt.hexbin` | 2D hexagon-binned counts | Alternative tesselation shape |
| `scipy.stats.gaussian_kde` | Smooth density estimate | Requires bandwidth choice |

## Worked Example
Visualizing `z = sin(x)^10 + cos(10+yx)*cos(x)` across a grid: `np.meshgrid` builds `X`,`Y` from 1D `linspace` arrays, `Z = f(X,Y)` evaluates the function, then `plt.contour(X,Y,Z,20,cmap='RdGy')` (lines), `plt.contourf(...)` (filled, splotchy at discrete levels), and `plt.imshow(Z, extent=[0,5,0,5], origin='lower', interpolation='gaussian')` (smooth) each render the same underlying data with progressively smoother/different visual character — showing the trade-offs explicitly on one running example.

## Key Takeaways
1. `plt.contour`/`plt.contourf`/`plt.imshow` offer a spectrum from discrete lines to filled regions to smooth images for visualizing z=f(x,y) data; they can be combined (transparent imshow + labeled contour lines).
2. `np.meshgrid` is the standard way to build the 2D coordinate grids these functions require.
3. `plt.imshow` has three specific gotchas: manual `extent`, `origin='lower'` for standard plot orientation, and `aspect=` control.
4. 1D histograms (`plt.hist`) generalize to 2D via `plt.hist2d` (square bins) or `plt.hexbin` (hexagonal bins); both have `np.histogram`/`np.histogram2d` compute-only counterparts.
5. Kernel density estimation (`scipy.stats.gaussian_kde`) provides a smooth density alternative to binned histograms, at the cost of a bandwidth/smoothing-length choice.

## Connects To
- **Ch 8**: the example function `f(x,y)` directly reuses the broadcasting motivating example.
- **Ch 30**: colorbar customization is covered in depth there.
- **Ch 35**: 3D visualization is the natural extension for this type of data.
- **Ch 49**: kernel density estimation is covered in full statistical depth there.
- **Ch 36**: Seaborn offers more compact APIs for KDE-based visualization.
