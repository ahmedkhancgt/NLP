# Chapter 26: Simple Line Plots

## Core Idea
Every Matplotlib plot starts from a `Figure` + `Axes` pair; line appearance (color, style) and axis framing (limits, labels, legend) are controlled via a small, consistent set of keyword arguments and methods, with specific naming differences between the stateful and object-oriented interfaces.

## Frameworks Introduced
- **Figure/Axes construction**: `fig = plt.figure(); ax = plt.axes()` (or the shortcut `fig, ax = plt.subplots()`) creates the container (`Figure`) and the plotting region (`Axes`) that all subsequent calls target.
- **Line styling keywords**: `color=` (name, short code, grayscale string, hex, RGB tuple, or HTML name) and `linestyle=` (`'solid'`/`'-'`, `'dashed'`/`'--'`, `'dashdot'`/`'-.'`, `'dotted'`/`':'`) — combinable into a single shorthand string argument (e.g., `'-g'`).
- **Axis limit control**: `plt.xlim()`/`plt.ylim()` (explicit bounds, reversible by swapping argument order) vs. `plt.axis()` (qualitative: `'tight'`, `'equal'`, `'on'`/`'off'`, `'square'`, `'image'`).
- **Labeling**: `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()` (driven by each `plot()` call's `label=` keyword).
- **MATLAB-style ↔ object-oriented naming gotcha**: most `plt.X` calls map directly to `ax.X`, but axis-limit/label/title functions become `ax.set_X` (`plt.xlabel`→`ax.set_xlabel`, etc.); `ax.set(...)` sets several properties in one call.

## Key Concepts
- **`fig.add_axes`/`plt.axes()`**: creates the Axes bounding box with ticks/grid/labels.
- **Trailing semicolon convention**: suppresses the textual (`Out[]:`) representation of the last plotting call in a notebook.
- **Multiple `plot()` calls accumulate on the same axes** by default (overplotting).
- **Automatic color cycling**: if no `color=` is given, Matplotlib cycles through a default palette across successive `plot()` calls.
- **`ax.set()`**: convenience method bundling multiple `set_*` calls (title, xlabel, ylabel, xlim, ylim, etc.) into one.

## Mental Models
- A plot call sequence is literally additive: each `plt.plot(...)` (or `ax.plot(...)`) draws onto the current (or specified) Axes, so building a multi-line plot is just repeated calls, not a special "multi-series" API.
- Axis-limit "reversal" (swapping `xlim`/`ylim` argument order) is a deliberate trick, not an error — Matplotlib draws whatever range you specify, direction included.

## Anti-patterns
- **Assuming every `plt.X` function has a same-named `ax.X` equivalent**: labels/limits/titles specifically become `ax.set_X` in the object-oriented interface — a frequent source of `AttributeError` when porting code between interfaces.
- **Confusing `axes` (the object, with an "e") and `axis` (the function `plt.axis()`, with an "i")**: they are different things — `plt.axis('equal')` is a convenience function for qualitative limit-setting, not a reference to the Axes object.
- **Manually setting color/linestyle redundantly when the default cycle already distinguishes lines**: unnecessary verbosity for simple multi-line plots; reserve explicit styling for when distinction or specific aesthetics matter.

## Code Examples
```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));              # trailing ; suppresses Out[] text

plt.plot(x, np.sin(x))               # MATLAB-style equivalent

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x));               # overplotting multiple lines

plt.plot(x, np.sin(x - 0), color='blue')
plt.plot(x, np.sin(x - 1), color='g')         # short code
plt.plot(x, np.sin(x - 2), color='0.75')       # grayscale
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # hex
plt.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3))  # RGB tuple
plt.plot(x, np.sin(x - 5), color='chartreuse')    # HTML name

plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')
plt.plot(x, x + 4, linestyle='-')
plt.plot(x, x + 5, linestyle='--')
plt.plot(x, x + 6, linestyle='-.')
plt.plot(x, x + 7, linestyle=':')

plt.plot(x, x + 0, '-g')          # shorthand: color + linestyle combined

plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

plt.plot(x, np.sin(x))
plt.xlim(10, 0)                       # reversed x-axis
plt.ylim(1.2, -1.2)                    # reversed y-axis

plt.plot(x, np.sin(x))
plt.axis('tight')                       # tighten bounds to content
plt.axis('equal')                        # equal unit ratio

plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")

plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend();

# object-oriented naming differences
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set_xlim(-1, 11)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')
ax.set_title('A Sine Curve')
# equivalently, in one call:
ax.set(xlim=(0, 10), ylim=(-2, 2), xlabel='x', ylabel='sin(x)', title='A Simple Plot')
```
- **What it demonstrates**: Figure/Axes creation, line overplotting, every color/linestyle specification method (and their shorthand combination), explicit vs. qualitative axis-limit control, labeling/legend, and the `plt.X` → `ax.set_X` naming conversion including the bundled `ax.set()` form.

## Reference Tables
| MATLAB-style (`plt.`) | Object-oriented (`ax.`) |
|---|---|
| `plt.plot` | `ax.plot` |
| `plt.legend` | `ax.legend` |
| `plt.xlabel` | `ax.set_xlabel` |
| `plt.ylabel` | `ax.set_ylabel` |
| `plt.xlim` | `ax.set_xlim` |
| `plt.ylim` | `ax.set_ylim` |
| `plt.title` | `ax.set_title` |

| Linestyle name | Shorthand |
|---|---|
| `'solid'` | `'-'` |
| `'dashed'` | `'--'` |
| `'dashdot'` | `'-.'` |
| `'dotted'` | `':'` |

## Worked Example
Plotting sin(x) and cos(x) with styled lines and a legend: `plt.plot(x, np.sin(x), '-g', label='sin(x)'); plt.plot(x, np.cos(x), ':b', label='cos(x)'); plt.axis('equal'); plt.legend()` — in four lines, produces a labeled, equal-aspect-ratio, color/style-differentiated two-line plot with an auto-generated legend matching each line's style and label.

## Key Takeaways
1. Every plot starts from a `Figure` + `Axes` pair (`plt.figure()`/`plt.axes()`, or `plt.subplots()`).
2. Color accepts many formats (name, short code, grayscale string, hex, RGB tuple, HTML name); linestyle has both verbose and shorthand forms; both combine into one shorthand string argument.
3. `plt.xlim`/`plt.ylim` set explicit bounds (reversible by argument order); `plt.axis()` sets qualitative bounds (`'tight'`, `'equal'`, etc.) — don't confuse `axis` (function) with `Axes` (object).
4. `plt.legend()` uses each `plot()` call's `label=` keyword to build the legend automatically.
5. Labels/limits/titles require the `ax.set_*` naming pattern in the object-oriented interface (not just `ax.xlabel`, etc.); `ax.set(...)` bundles several into one call.

## Connects To
- **Ch 25**: directly continues the two-interface distinction and extends it with the specific naming gotchas.
- **Ch 27**: Scatter plots use many of the same color/style keyword conventions.
- **Ch 29**: legend customization is covered in much greater depth there.
