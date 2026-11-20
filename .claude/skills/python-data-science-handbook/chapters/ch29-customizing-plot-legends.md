# Chapter 29: Customizing Plot Legends

## Core Idea
`plt.legend`/`ax.legend` auto-generates legends from labeled plot elements, with extensive placement/styling options, plus techniques for selective element inclusion, size-based legends (via empty-list proxy plots), and multiple legends on one axes (via manual `Legend` artist creation).

## Frameworks Introduced
- **Basic legend**: `ax.legend()` auto-includes all elements with a `label=` set; customize via `loc=`, `frameon=`, `ncol=`, `fancybox=`, `framealpha=`, `shadow=`, `borderpad=`.
- **Selective legend elements**: either pass explicit line-object lists (`plt.legend(lines[:2], ['first','second'])`) or (preferred) only set `label=` on the elements you want shown — unlabeled elements are automatically excluded.
- **Legend for encoded point size**: plot invisible/empty proxy points (`plt.scatter([], [], c='k', s=area, label=...)`) purely to register legend entries representing a size scale that isn't itself a real data series.
- **Multiple legends on one axes**: create additional `matplotlib.legend.Legend` artist objects manually and attach via `ax.add_artist(leg)` — the standard `ax.legend()` call only supports one legend per axes and overwrites any prior call.

## Key Concepts
- **`label=` attribute**: the only thing that determines whether a plot element appears in an auto-generated legend.
- **`loc=` placement codes**: e.g. `'upper left'`, `'lower right'`, etc.
- **Legend styling knobs**: `ncol` (columns), `fancybox` (rounded corners), `framealpha` (frame transparency), `shadow`, `borderpad` (text padding).
- **Proxy/"fake" plot elements**: plotting empty lists (`[]`,`[]`) to create labeled, invisible artists purely so the legend references them — the legend can only reference objects actually plotted.
- **`Artist` base class**: Matplotlib's base class for all visual plot components; `Legend` is one such artist, normally managed implicitly by `ax.legend()` but can be instantiated and added directly via `ax.add_artist()`.

## Mental Models
- A legend is fundamentally "a list of labeled artists" — it can't display anything that isn't plotted somewhere (even invisibly), which explains the empty-list proxy-plot trick for showing a size/color scale with no corresponding visible series.
- Matplotlib's `ax.legend()` convenience function is "sugar" over manually constructing a `Legend` artist and attaching it to the axes — seeing this (via `ax.legend??`) explains why only one such convenience call is supported, and how to bypass that limit manually.

## Anti-patterns
- **Calling `ax.legend()` twice expecting two separate legends**: the second call simply overwrites the first — use a manually constructed `Legend` artist with `ax.add_artist()` for a second legend.
- **Trying to add a legend entry for a size/color scale without plotting anything for it**: the legend can only show entries for actual (even if invisible/empty) plotted artists — use the empty-list proxy-plot technique.
- **Passing explicit line-lists to `plt.legend()` as the default approach**: works, but the chapter notes it's "clearer... to use the first method, applying labels to the plot elements you'd like to show" — prefer setting `label=` selectively over post-hoc line-list filtering.

## Code Examples
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend()

ax.legend(loc='upper left', frameon=True)
ax.legend(ncol=2)
ax.legend(frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)

# selecting which elements appear
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)                        # list of Line2D instances
plt.legend(lines[:2], ['first', 'second'], frameon=True)

# preferred: label only the elements you want shown
plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])                           # no label -> excluded from legend
plt.legend(frameon=True)

# legend for point size (California cities example)
import pandas as pd
cities = pd.read_csv('data/california_cities.csv')
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']
plt.scatter(lon, lat, label=None, c=np.log10(population), cmap='viridis',
            s=area, linewidth=0, alpha=0.5)
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

# multiple legends
fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']
for i in range(4):
    lines += ax.plot(x, np.sin(x - i*np.pi/2), styles[i], color='black')
ax.axis('equal')
ax.legend(lines[:2], ['line A', 'line B'], loc='upper right')

from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'], loc='lower right')
ax.add_artist(leg)
```
- **What it demonstrates**: basic legend creation/styling, two methods of selective legend-element inclusion, the empty-list proxy trick for a size legend, and manual Legend artist construction for a second legend on the same axes.

## Reference Tables
| Legend customization | Keyword |
|---|---|
| Location | `loc` |
| Frame on/off | `frameon` |
| Number of columns | `ncol` |
| Rounded box | `fancybox` |
| Frame transparency | `framealpha` |
| Shadow | `shadow` |
| Padding around text | `borderpad` |

## Worked Example
California cities visualization: a scatter plot (`label=None`) encodes population via color and area via point size, so the legend can't auto-generate a size key from the real data; the fix plots three invisible reference points (`plt.scatter([], [], c='k', s=area, label=f'{area} km²')` for `area` in `[100, 300, 500]`) purely to populate a "City Area" legend showing what each point size represents — a widely reusable trick whenever size/color encodes a continuous variable that needs a discrete legend key.

## Key Takeaways
1. `ax.legend()` includes only elements with `label=` set; this is the primary lever for controlling legend content.
2. Legend placement/styling is controlled via `loc`, `frameon`, `ncol`, `fancybox`, `framealpha`, `shadow`, `borderpad`.
3. To show a legend entry for a size/color scale with no real corresponding series, plot invisible proxy elements (empty-list scatter/plot calls) with the desired `label=`.
4. A standard axes supports only one `ax.legend()` call; for multiple legends, manually construct additional `matplotlib.legend.Legend` artists and attach them with `ax.add_artist()`.
5. Reading `ax.legend??`'s source reveals that `ax.legend()` is just a convenience wrapper constructing and attaching a `Legend` artist — understanding this demystifies the manual multi-legend workaround.

## Connects To
- **Ch 26**: builds directly on the basic `label=`/`plt.legend()` introduced there.
- **Ch 27**: the California cities example combines scatter size/color encoding techniques from the scatter-plot chapter.
- **Ch 30**: colorbars are the continuous-data analog to legends, covered next.
