# Chapter 25: General Matplotlib Tips

## Core Idea
Matplotlib display/save mechanics depend on context (script vs. IPython shell vs. Jupyter notebook), and the library offers two parallel APIs — a stateful MATLAB-style `plt` interface and a more powerful, explicit object-oriented `Figure`/`Axes` interface.

## Frameworks Introduced
- **Context-dependent display**: how to actually see a plot differs by environment.
  - Script: call `plt.show()` once, usually at the very end — starts an event loop and opens interactive windows for all active figures.
  - IPython shell: run `%matplotlib` magic once to enable interactive mode; subsequent `plt` commands auto-display; use `plt.draw()` to force-refresh after modifying existing elements.
  - Jupyter notebook: `%matplotlib inline` embeds static images; `%matplotlib notebook` embeds interactive plots.
- **Two parallel interfaces**:
  - MATLAB-style (`plt.plot`, `plt.subplot`, etc.): stateful, tracks a "current" figure/axes (`plt.gcf()`, `plt.gca()`); fast for simple plots, clunky for revisiting earlier panels.
  - Object-oriented (`fig, ax = plt.subplots(...)`; `ax.plot(...)`): explicit Figure/Axes objects as method targets; necessary for more complex, multi-panel control.

## Key Concepts
- **`plt.style.use('classic')`**: sets a global aesthetic style (see Ch 34 for stylesheets in depth).
- **`Figure` vs. `Axes`**: a `Figure` is the single container for everything (axes, graphics, text, labels); an `Axes` is a bounding box with ticks/grids/labels that holds the actual plot elements.
- **`fig.savefig(filename)`**: saves a figure; output format inferred from file extension; `fig.canvas.get_supported_filetypes()` lists available formats (png, pdf, svg, eps, jpg, etc.); `plt.show()` is NOT needed when only saving.
- **One `plt.show()` per session**: calling it multiple times can cause unpredictable backend-dependent behavior.

## Mental Models
- Think of the "display context" question (script/IPython/notebook) as orthogonal to the "which API" question (stateful vs. object-oriented) — you choose independently how you *view* a plot and how you *build* it.
- The stateful `plt` interface is "always operating on whatever is currently active" — convenient until you need to go back and modify an earlier panel, at which point the object-oriented interface's explicit object references become necessary.

## Anti-patterns
- **Calling `plt.show()` multiple times in one script**: leads to unpredictable, backend-dependent behavior — call it once, typically at the end.
- **Mixing up `axes` (with an e) and `axis` (with an i)**: `plt.axis()` is a distinct function (for qualitative axis limit control, e.g., `'tight'`, `'equal'`) from the `Axes` object — easy source of confusion (addressed further in Ch 26).
- **Calling `plt.show()` before `savefig()`, assuming it's required**: unnecessary — `savefig` works independently of whether the figure is displayed.

## Code Examples
```python
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('classic')

# script context
# file: myplot.py
import numpy as np
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()                      # once, at the end

# IPython shell
# %matplotlib            (magic, enables interactive mode)
# plt.draw()              # force refresh after modifying existing elements

# Jupyter notebook
# %matplotlib inline       (static images)
# %matplotlib notebook      (interactive)
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')

fig.savefig('my_figure.png')         # format from extension
fig.canvas.get_supported_filetypes()    # list of supported formats
from IPython.display import Image
Image('my_figure.png')                   # display saved file in notebook

# MATLAB-style interface (stateful)
plt.figure()
plt.subplot(2, 1, 1)         # (rows, cols, panel number)
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))

# Object-oriented interface
fig, ax = plt.subplots(2)       # ax is an array of Axes objects
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
```
- **What it demonstrates**: environment-specific display setup, figure saving and format discovery, and the two parallel plotting APIs producing equivalent output.

## Reference Tables
| Context | How to display |
|---|---|
| Script | `plt.show()` once, usually at the end |
| IPython shell | `%matplotlib` magic, then plain `plt` commands auto-display; `plt.draw()` to refresh |
| Jupyter notebook | `%matplotlib inline` (static) or `%matplotlib notebook` (interactive) |

| Interface | Style | Use when |
|---|---|---|
| MATLAB-style (`plt.*`) | Stateful (current figure/axes) | Simple, single-pass plots |
| Object-oriented (`fig`, `ax`) | Explicit objects | Multi-panel / complex plots, revisiting earlier panels |

## Worked Example
Saving and re-displaying a figure: `fig.savefig('my_figure.png')` writes a 26KB PNG inferred from the `.png` extension; `fig.canvas.get_supported_filetypes()` reveals Matplotlib also supports pdf, svg, eps, jpg, tiff, and more — so changing only the extension string in `savefig` switches output format with no other code changes.

## Key Takeaways
1. How you *see* a Matplotlib plot depends on your environment: `plt.show()` for scripts, `%matplotlib` for IPython, `%matplotlib inline`/`notebook` for Jupyter.
2. Call `plt.show()` only once per session, typically at the end.
3. `fig.savefig(filename)` infers format from the extension and works independently of display; `fig.canvas.get_supported_filetypes()` lists available formats.
4. Matplotlib offers two interfaces: the stateful MATLAB-style `plt` API (fast, simple) and the object-oriented `Figure`/`Axes` API (explicit, necessary for complex multi-panel control).
5. Most `plt.*` calls map directly to `ax.*` methods, but a few key exceptions (labels/limits/titles) require `ax.set_*` — covered fully in Chapter 26.

## Connects To
- **Ch 1**: `%matplotlib` is an IPython magic command, building on Ch 1/Ch 2's magic-command foundations.
- **Ch 26**: immediately continues into line-plot specifics and the `plt.*` → `ax.set_*` naming gotchas.
- **Ch 34**: stylesheet customization (`plt.style.use`) is covered in depth there.
