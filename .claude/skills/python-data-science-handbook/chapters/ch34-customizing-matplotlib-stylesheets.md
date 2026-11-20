# Chapter 34: Customizing Matplotlib: Configurations and Stylesheets

## Core Idea
Instead of manually re-styling every plot element by hand each time, Matplotlib's runtime configuration (`plt.rc`/`rcParams`) and built-in stylesheets (`plt.style.use`/`plt.style.context`) let you set a global aesthetic once, applied consistently to every subsequent plot.

## Frameworks Introduced
- **Manual per-element customization**: adjusting facecolor, spines, gridlines, tick colors, and histogram colors one call at a time — illustrates the tedium that global config solves.
- **`plt.rc(group, **kwargs)`**: sets default rcParams for a category (`'figure'`, `'axes'`, `'grid'`, `'xtick'`/`'ytick'`, `'patch'`, `'lines'`, etc.) — persists for the whole session; can be saved permanently in a `.matplotlibrc` file.
  - `cycler('color', [...])` passed as `prop_cycle` sets a custom default color cycle for multi-line/series plots.
- **Stylesheets (`plt.style`)**: pre-packaged bundles of rc settings.
  - `plt.style.available` lists built-in stylesheet names.
  - `plt.style.use('name')`: switches style for the rest of the session (persistent).
  - `with plt.style.context('name'): ...`: applies a style temporarily, scoped to the `with` block.
  - Custom styles: `.mplstyle` files, structured like `.matplotlibrc`.

## Key Concepts
- **rcParams categories relevant here**: `figure` (facecolor), `axes` (facecolor, edgecolor, axisbelow, grid, prop_cycle), `grid` (color, linestyle), `xtick`/`ytick` (direction, color), `patch` (edgecolor), `lines` (linewidth).
- **`ax.set_axisbelow(True)`**: draws gridlines behind plot elements rather than on top.
- **Spines**: the box lines around an axes (`ax.spines.values()`); `spine.set_visible(False)` hides them for a cleaner look.
- **Built-in stylesheet gallery**: `'classic'`/default (Matplotlib 2.0+ default), `'fivethirtyeight'` (bold colors, thick lines, transparent axes), `'ggplot'` (mimics R's ggplot2), `'bmh'` (from *Bayesian Methods for Hackers*), `'dark_background'` (for presentations), `'grayscale'` (for non-color print), `'seaborn-whitegrid'`/other seaborn-* styles (clean, popular for data exploration).

## Mental Models
- Think of per-plot manual styling as "writing the same boilerplate every time" and rcParams/stylesheets as "writing it once, as a default" — the exact same visual result, but amortized across every future plot in the session (rc) or scoped cleanly to specific plots (`style.context`).
- Stylesheets are essentially curated, named bundles of rcParams — switching styles is equivalent to loading a different preset configuration, not a fundamentally different mechanism from `plt.rc`.

## Anti-patterns
- **Manually re-applying the same dozen styling tweaks (spine visibility, tick color, grid color, etc.) to every new plot**: tedious and error-prone — set it once via `plt.rc` or a stylesheet instead.
- **Using `plt.style.use(...)` when only one or two plots need a different look**: this changes the style for the rest of the session — use the `with plt.style.context(...)` context manager instead for scoped, temporary styling.
- **Assuming a stylesheet choice is purely aesthetic preference with no practical trade-offs**: context matters — e.g., `grayscale` for non-color print publications, `dark_background` for presentations, default/`seaborn-*` for on-screen exploration.

## Code Examples
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(1000)
plt.hist(x)                           # drab default

# manual per-plot customization
fig = plt.figure(facecolor='white')
ax = plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)
plt.grid(color='w', linestyle='solid')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.xaxis.tick_bottom(); ax.yaxis.tick_left()
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels(): tick.set_color('gray')
for tick in ax.get_yticklabels(): tick.set_color('gray')
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666')

# rcParams: set once, applies to all future plots this session
from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('figure', facecolor='white')
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

plt.hist(x)                            # now styled by default

# stylesheets
plt.style.available[:5]                 # list available styles

plt.style.use('stylename')               # persistent for the session

with plt.style.context('stylename'):     # temporary, scoped to block
    make_a_plot()

def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')

with plt.style.context('fivethirtyeight'):
    hist_and_lines()
with plt.style.context('ggplot'):
    hist_and_lines()
with plt.style.context('bmh'):
    hist_and_lines()
with plt.style.context('dark_background'):
    hist_and_lines()
with plt.style.context('grayscale'):
    hist_and_lines()
with plt.style.context('seaborn-whitegrid'):
    hist_and_lines()
```
- **What it demonstrates**: the manual-styling-vs-rcParams equivalence (same visual result, radically less repeated code), and a tour of the major built-in stylesheets via a reusable comparison function.

## Reference Tables
| Stylesheet | Character |
|---|---|
| default (2.0+) | Matplotlib's modern baseline |
| `fivethirtyeight` | Bold colors, thick lines, transparent axes |
| `ggplot` | Mimics R's ggplot2 |
| `bmh` | From *Bayesian Methods for Hackers* |
| `dark_background` | Dark theme, good for presentations |
| `grayscale` | For non-color print publications |
| `seaborn-whitegrid` (and other `seaborn-*`) | Clean, popular for data exploration |

## Worked Example
Replicating a ggplot-inspired look without manual per-plot tweaking: instead of the ~15-line manual customization block (gray facecolor, white solid gridlines, hidden spines, gray ticks, custom histogram colors) repeated for every plot, `plt.rc(...)` calls across `'figure'`, `'axes'`, `'grid'`, `'xtick'`/`'ytick'`, `'patch'`, and `'lines'` groups set the equivalent defaults once — every subsequent `plt.hist(x)` or line plot automatically inherits the styled look for the rest of the session.

## Key Takeaways
1. `plt.rc(group, **kwargs)` sets session-wide default styling per rcParams category, replacing repeated manual per-plot customization; settings can be persisted in a `.matplotlibrc` file.
2. `plt.style.available` lists built-in stylesheets; `plt.style.use('name')` switches persistently, `with plt.style.context('name'):` switches temporarily/scoped.
3. Stylesheets are essentially named, curated bundles of rcParams — not a separate mechanism from `plt.rc`, just packaged presets.
4. Built-in styles serve different practical purposes: `grayscale` for print, `dark_background` for presentations, `seaborn-*`/default for on-screen exploration, `ggplot`/`fivethirtyeight`/`bmh` for specific aesthetic inspirations.
5. Custom styles can be packaged as `.mplstyle` files for reuse across projects.

## Connects To
- **Ch 25**: `plt.style.use('classic')` was introduced there as a basic setup step; this chapter explains the full mechanism behind it.
- **Ch 36**: several built-in stylesheets are explicitly inspired by (and can be paired with) the Seaborn library covered next.
