# Chapter 33: Customizing Ticks

## Core Idea
Every axes' tickmarks are governed by a Locator (where ticks go) and a Formatter (how tick labels look), independently settable for major/minor ticks on each axis, giving fine-grained control beyond Matplotlib's reasonable-but-not-always-optimal defaults.

## Frameworks Introduced
- **Locator/Formatter pattern**: `ax.xaxis`/`ax.yaxis` each expose `set_major_locator()`/`set_minor_locator()` (where ticks appear) and `set_major_formatter()`/`set_minor_formatter()` (how labels render), independent of each other.
  - When to use: hiding clutter, reducing/increasing tick density, aligning ticks to meaningful units (e.g., multiples of π), or custom label text.
- **Key locator classes**: `NullLocator` (no ticks), `MultipleLocator(base)` (ticks at multiples of `base`), `MaxNLocator(n)` (up to n ticks at "nice" locations), `AutoLocator` (default), `LogLocator`, `FixedLocator`, `IndexLocator`, `LinearLocator`, `AutoMinorLocator`.
- **Key formatter classes**: `NullFormatter` (no labels), `FuncFormatter(func)` (fully custom label text via a user function), `FixedFormatter`, `FormatStrFormatter`, `ScalarFormatter` (default), `LogFormatter`, `IndexFormatter`.

## Key Concepts
- **Object hierarchy**: Figure contains Axes; each Axes has `.xaxis`/`.yaxis`, which hold the tick/label/line properties — ticks are just more objects in Matplotlib's "everything is an object" design.
- **Major vs. minor ticks**: major ticks are typically larger/labeled with gridlines; minor ticks smaller, often unlabeled by default (e.g., log-scale plots show minor ticks via `LogLocator` + `NullFormatter` — present but unlabeled).
- **`plt.NullLocator()`/`plt.NullFormatter()`**: the standard way to hide ticks (locator) or just their labels (formatter) while potentially keeping the other.
- **`plt.MaxNLocator(n)`**: solves "too many crowded tick labels" by capping the tick count and letting Matplotlib choose nice positions.
- **`plt.FuncFormatter(func)`**: `func(value, tick_number)` returns the label string — enables fully custom formatting (e.g., labeling ticks as fractions of π using LaTeX via `$...$` syntax).

## Mental Models
- Think of tick customization as answering two independent questions per axis: "where should ticks go?" (Locator) and "what should they say?" (Formatter) — these can be mixed and matched freely (e.g., `MultipleLocator` for placement + `FuncFormatter` for custom text).
- Hiding all ticks (`NullLocator` on both axes) is itself a meaningful choice, not just an omission — appropriate when tick values (e.g., raw pixel indices in an image grid) carry no useful information for the viewer.

## Anti-patterns
- **Leaving default tick density on small multi-panel grids**: labels crowd/overlap and become illegible — use `MaxNLocator(n)` to cap tick count per subplot.
- **Manually hardcoding decimal tick labels for naturally periodic data (e.g., multiples of π)**: numeric decimals obscure the underlying structure — use `MultipleLocator` for placement plus a `FuncFormatter` for meaningful symbolic labels.
- **Assuming hiding ticks removes gridlines too (or vice versa)**: locator and formatter control different aspects — removing the locator (`NullLocator`) removes ticks/gridlines; removing just the formatter (`NullFormatter`) removes only labels while ticks/gridlines can remain.

## Code Examples
```python
import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes(xscale='log', yscale='log')
print(ax.xaxis.get_major_locator())    # LogLocator
print(ax.xaxis.get_minor_formatter())  # LogFormatterSciNotation (minor often Null in practice)

# hiding ticks/labels
ax = plt.axes()
rng = np.random.default_rng(1701)
ax.plot(rng.random(50))
ax.grid()
ax.yaxis.set_major_locator(plt.NullLocator())      # remove y ticks entirely
ax.xaxis.set_major_formatter(plt.NullFormatter())   # keep x ticks, hide labels

# fully hidden ticks for image grids
fig, ax = plt.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(hspace=0, wspace=0)
for i in range(5):
    for j in range(5):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())

# reducing tick count on a crowded grid
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))

# multiples of pi with custom formatter
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')
ax.grid(True); ax.legend(frameon=False); ax.axis('equal'); ax.set_xlim(0, 3*np.pi)

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

def format_func(value, tick_number):
    N = int(np.round(2 * value / np.pi))
    if N == 0: return "0"
    elif N == 1: return r"$\pi/2$"
    elif N == 2: return r"$\pi$"
    elif N % 2 > 0: return rf"${N}\pi/2$"
    else: return rf"${N // 2}\pi$"

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
```
- **What it demonstrates**: inspecting default locators/formatters, hiding ticks/labels for cleaner image-grid displays, reducing crowded tick counts, and combining `MultipleLocator` + `FuncFormatter` for meaningful symbolic (π-based) tick labels with LaTeX rendering.

## Reference Tables
| Locator | Description |
|---|---|
| `NullLocator` | No ticks |
| `FixedLocator` | Fixed tick locations |
| `IndexLocator` | For index plots (x = range(len(y))) |
| `LinearLocator` | Evenly spaced from min to max |
| `LogLocator` | Log-spaced from min to max |
| `MultipleLocator` | Ticks at multiples of a base value |
| `MaxNLocator` | Up to N ticks at "nice" locations |
| `AutoLocator` | Default (MaxNLocator with simple defaults) |
| `AutoMinorLocator` | For minor ticks |

| Formatter | Description |
|---|---|
| `NullFormatter` | No labels |
| `IndexFormatter` | Strings from a label list |
| `FixedFormatter` | Manually set label strings |
| `FuncFormatter` | User-defined function sets labels |
| `FormatStrFormatter` | Format string per value |
| `ScalarFormatter` | Default for scalar values |
| `LogFormatter` | Default for log axes |

## Worked Example
Sine/cosine plot with π-based ticks: setting `MultipleLocator(np.pi/2)` (major) and `MultipleLocator(np.pi/4)` (minor) places ticks at meaningful periodic intervals instead of arbitrary decimals; a `FuncFormatter` then converts each tick's numeric value into a LaTeX-rendered fraction of π (`$\pi/2$`, `$\pi$`, `$3\pi/2$`, etc.) — the combination makes the plot's periodic structure immediately legible, which plain decimal tick labels would obscure.

## Key Takeaways
1. Every axis has independently settable major/minor Locators (tick placement) and Formatters (tick label text).
2. `NullLocator`/`NullFormatter` hide ticks/labels respectively — useful for image grids where tick values are meaningless.
3. `MaxNLocator(n)` fixes crowded tick labels on small/multi-panel plots by capping tick count.
4. `MultipleLocator(base)` aligns ticks to meaningful units (e.g., multiples of π); `FuncFormatter` provides fully custom label text, including LaTeX (`$...$`) rendering.
5. Matplotlib's "everything is an object" philosophy extends down to ticks — `ax.xaxis`/`ax.yaxis` hold all tick-related sub-objects, inspectable via `get_major_locator()`/`get_major_formatter()`.

## Connects To
- **Ch 23/Ch 32**: date-axis formatting (`mpl.dates.MonthLocator`, `DateFormatter`) is a specialized application of this same Locator/Formatter pattern.
- **Ch 25**: builds on the Figure/Axes object hierarchy established there.
- **Ch 43**: the Olivetti faces image-grid example previews supervised learning / face recognition use cases.
