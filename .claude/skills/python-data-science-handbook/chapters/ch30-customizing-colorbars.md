# Chapter 30: Customizing Colorbars

## Core Idea
Colorbars provide a continuous-data analog to discrete legends; choosing the right colormap category (sequential/divergent/qualitative) and customizing color limits/extensions/discretization meaningfully affects how accurately and fairly a visualization communicates data.

## Frameworks Introduced
- **`plt.colorbar()`**: creates a separate axes acting as a key for color-encoded data; `cmap=` on the originating plot call sets the colormap; `plt.cm.<TAB>` lists available colormaps.
- **Three colormap categories** — choose deliberately based on data structure:
  - Sequential (e.g., `'binary'`, `'viridis'`): one continuous color sequence — for data without a meaningful zero/center.
  - Divergent (e.g., `'RdBu'`, `'PuOr'`): two-color scheme diverging from a center — for data with meaningful positive/negative deviation from a mean.
  - Qualitative (e.g., `'rainbow'`, `'jet'`): colors with no inherent order — poor for quantitative data since brightness doesn't vary uniformly with value.
- **Color limit control**: `plt.clim(vmin, vmax)` narrows displayed color range; `extend='both'`/`'min'`/`'max'` on `colorbar()` adds triangular out-of-bounds indicators — essential when noisy outliers would otherwise wash out the color scale.
- **Discrete colorbars**: `plt.cm.get_cmap(name, n_bins)` quantizes a continuous colormap into `n_bins` discrete colors, usable exactly like any other colormap.

## Key Concepts
- **Uneven luminance problem (qualitative colormaps like `jet`)**: when converted to grayscale, brightness doesn't vary monotonically with value — creates misleading visual emphasis on arbitrary regions of the data range; this is why `viridis` (even luminance, perceptually uniform) replaced `jet` as Matplotlib's default in v2.0.
- **Grayscale conversion test**: a quick way to audit a colormap's perceptual uniformity — convert to grayscale (perceived luminance formula) and check for unexpected bright/dark "stripes."
- **Divergent colormap grayscale loss**: positive/negative sign information from a divergent colormap (e.g., `RdBu`) is lost when printed/viewed in grayscale — a practical accessibility/reproducibility consideration.
- **Colorbar as an `Axes` instance**: all standard axes/tick customization tools apply to a colorbar itself.

## Mental Models
- A colorbar is to continuous color-encoded data what a legend is to discrete labeled elements — "a key explaining what the color means," not a decorative addition.
- Choosing a colormap is a data-structure decision, not an aesthetic one: ask "does my data have a meaningful zero/center (divergent), one continuous range (sequential), or no inherent order (qualitative, rarely appropriate for quantitative data)?" before picking colors.
- Always imagine your plot in grayscale (or to a colorblind viewer) as a sanity check — qualitative/jet-like colormaps that look fine in full color can completely mislead or lose information once color perception is impaired.

## Anti-patterns
- **Using a qualitative colormap (e.g., `jet`, `rainbow`) for ordered/quantitative data**: brightness doesn't vary uniformly with value, visually distorting which parts of the data draw attention — prefer `viridis` or another perceptually uniform sequential/divergent map.
- **Letting a few noisy/outlier pixels set the full color range**: washes out the main pattern of interest — manually set `plt.clim()` and use `extend='both'` to flag out-of-range values instead of letting them dominate the scale.
- **Assuming a divergent colormap's meaning survives grayscale/colorblind-safe rendering**: positive/negative sign information is lost — check an alternate rendering if accessibility matters.

## Code Examples
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])
plt.imshow(I)
plt.colorbar()

plt.imshow(I, cmap='Blues')     # specify colormap via cmap=

# noisy data washing out color range -- fix with clim + extend
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))
plt.imshow(I, cmap='RdBu')
plt.colorbar()                        # default limits distorted by noise
# vs.
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1)                        # manually narrowed range + out-of-bounds arrows

# discrete colorbar
plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))   # 6-bin discretized colormap
plt.colorbar(extend='both')
plt.clim(-1, 1)

# handwritten digits manifold example
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])

from sklearn.manifold import Isomap
iso = Isomap(n_components=2, n_neighbors=15)
projection = iso.fit_transform(digits.data)

plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=digits.target, cmap=plt.cm.get_cmap('plasma', 6))
plt.colorbar(ticks=range(6), label='digit value')
```
- **What it demonstrates**: basic colorbar creation, colormap selection and its perceptual implications (sequential vs. divergent vs. qualitative), color-limit/extend handling for noisy data, discretized colormaps, and a full real-world example (handwritten digits manifold projection) using a discrete colormap to label categorical digit values on a continuous scatter.

## Reference Tables
| Colormap category | Example | Use when |
|---|---|---|
| Sequential | `viridis`, `binary` | Data with one continuous range, no natural center |
| Divergent | `RdBu`, `PuOr` | Data with meaningful positive/negative deviation from a mean |
| Qualitative | `rainbow`, `jet` | Rarely appropriate for quantitative data — uneven luminance |

| Tool | Purpose |
|---|---|
| `plt.clim(vmin, vmax)` | Narrow displayed color range |
| `colorbar(extend='both'/'min'/'max')` | Flag out-of-range values with arrows |
| `plt.cm.get_cmap(name, n)` | Discretize a colormap into n bins |

## Worked Example
Handwritten digits (0–5) manifold visualization: project 64-dimensional pixel data to 2D via `sklearn.manifold.Isomap`, then `plt.scatter(..., c=digits.target, cmap=plt.cm.get_cmap('plasma', 6))` with `plt.colorbar(ticks=range(6), label='digit value')` uses a 6-bin discretized colormap to cleanly label each of the 6 digit classes on a continuous 2D scatter — revealing that digits 2 and 3 project to overlapping regions (likely confusable) while 0 and 1 are well-separated (unlikely to be confused) by a classifier.

## Key Takeaways
1. A colorbar is the continuous-data counterpart to a legend — created via `plt.colorbar()`, customizable like any `Axes`.
2. Colormap choice should match data structure: sequential (one range), divergent (deviation from center), or qualitative (rarely right for quantitative data) — `viridis` is the modern perceptually-uniform default, replacing the uneven-luminance `jet`.
3. `plt.clim()` plus `extend='both'`/`'min'`/`'max'` prevents outliers/noise from washing out the meaningful color range.
4. `plt.cm.get_cmap(name, n_bins)` discretizes any colormap for representing categorical/discrete values on a color scale.
5. Always sanity-check a colormap's grayscale/colorblind-safe rendering, especially for divergent maps where sign information can be lost.

## Connects To
- **Ch 28**: contour/image plots from the previous chapter are where colorbars are most commonly attached.
- **Ch 29**: colorbars and legends are explicitly framed as parallel discrete/continuous tools.
- **Ch 46**: Isomap manifold learning is covered in depth in Part V.
- **Ch 36**: Seaborn is recommended for more principled color-choice tooling.
