# Chapter 2: Enhanced Interactive Features

## Core Idea
IPython/Jupyter add "magic commands," persistent input/output history objects, and direct shell access on top of standard Python, reducing context-switching during interactive data analysis.

## Frameworks Introduced
- **Magic commands (`%`/`%%`)**: line magics (`%`, single line) and cell magics (`%%`, multiple lines) that solve common interactive-analysis problems.
  - When to use: running external scripts, timing code, inspecting history, shell interaction.
  - How: `%run file.py` executes a script and imports its definitions into the session; `%timeit`/`%%timeit` time a statement/block; `%history`, `%rerun`, `%save` manage command history.
- **In/Out history objects**: `In` (list of input strings) and `Out` (dict mapping line number → output) let you reference prior results programmatically (`Out[2] ** 2 + Out[3] ** 2`).
- **Shell bridge (`!`)**: run any system shell command from inside IPython, with two-way variable passing (`contents = !ls`, `!echo {message}`).

## Key Concepts
- **Line magic vs. cell magic**: `%` operates on one line; `%%` operates on the whole cell.
- **Automagic**: shell-like magics (`%cd`, `%cat`, `%cp`, `%env`, `%ls`, `%man`, `%mkdir`, `%more`, `%mv`, `%pwd`, `%rm`, `%rmdir`) can be called without the `%` prefix when automagic is on.
- **Underscore shortcuts**: `_` = last output, `__` = second-to-last, `___` = third-to-last (IPython only goes 3 deep); `_N` is shorthand for `Out[N]`.
- **SList**: the special return type (`IPython.utils.text.SList`) of `!shell_command` assignments — acts like a list but adds `.grep`, `.fields`, and `.s`/`.n`/`.p` properties.
- **Output suppression**: trailing `;` on a statement suppresses display and prevents storage in `Out`.
- **Subshell statelessness**: each `!` command runs in its own temporary subshell, so `!cd ..` does not persist — use `%cd` for a durable directory change.

## Mental Models
- IPython's shell integration treats Python and the OS shell as two adjacent namespaces connected by `!` and `{varname}` interpolation, not one unified language — state doesn't automatically cross that boundary except through explicit assignment/interpolation.
- `Out` only stores results of statements that return non-`None`; `print()` and `import` never populate it, so a forgotten unassigned expensive computation can still often be recovered from `Out`.

## Anti-patterns
- **Using `!cd` to change directories persistently**: subshells are stateless; use `%cd` (or automagic `cd`) instead.
- **Timing a single destructive/stateful operation with `%timeit`**: repetition skews results (e.g., sorting an already-sorted list after the first `%timeit` iteration) — use `%time` for one-shot/order-sensitive operations.
- **Relying on `%time` for very fast snippets**: system noise dominates; prefer `%timeit`, which also disables garbage collection during timing for cleaner results.

## Code Examples
```python
In [1]: %run myscript.py
1 squared is 1
2 squared is 4
3 squared is 9

In [3]: %timeit L = [n ** 2 for n in range(1000)]
430 µs ± 3.21 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [4]: %%timeit
   ...: L = []
   ...: for n in range(1000):
   ...:     L.append(n ** 2)

In [4]: In
Out[4]: ['', 'import math', 'math.sin(2)', 'math.cos(2)', 'In']

In [8]: Out[2] ** 2 + Out[3] ** 2
Out[8]: 1.0

In [14]: math.sin(2) + math.cos(2);   # suppressed, not stored in Out

In [16]: %history -n 1-3

In [4]: contents = !ls
In [9]: message = "hello from Python"
In [10]: !echo {message}
hello from Python

In [14]: %cd ..
In [15]: cd myproject   # automagic, no % needed
```
- **What it demonstrates**: script execution with namespace injection, line/cell timing, history introspection, output suppression, and bidirectional shell/Python variable passing.

## Reference Tables
| Magic | Purpose |
|---|---|
| `%run file.py` | Execute external script, import its names |
| `%timeit` / `%%timeit` | Time repeated execution (robust, auto-repeats) |
| `%time` / `%%time` | Time single execution (good for stateful/slow ops) |
| `%history -n a-b` | Show input history lines a through b |
| `%rerun` | Re-execute a portion of history |
| `%save` | Save history to a file |
| `%magic` | General description of magic functions |
| `%lsmagic` | List all available magic functions |
| `%cd`, `%cat`, `%cp`, `%env`, `%ls`, `%man`, `%mkdir`, `%more`, `%mv`, `%pwd`, `%rm`, `%rmdir` | Shell-equivalent magics (automagic-eligible) |

## Worked Example
Comparing list-comprehension vs. for-loop construction speed: `%timeit L = [n**2 for n in range(1000)]` → ~430 µs; the cell-magic equivalent with an explicit `for` loop and `.append` → ~484 µs. Conclusion: the list comprehension is ~10% faster for this case — illustrating how `%timeit` supports quick micro-benchmarking decisions during development.

## Key Takeaways
1. `%run` executes a script and merges its top-level names into the current session.
2. `%timeit`/`%%timeit` auto-repeat for statistically robust timing; `%time` is better for one-off or order-sensitive operations.
3. `In` (list) and `Out` (dict) let you recover and reuse prior inputs/outputs even if you forgot to assign them to variables.
4. `_`, `__`, `___`, and `_N` are shorthand accessors for recent outputs.
5. `!` runs shell commands; `{var}` interpolates Python variables into them; assignment captures shell output as an `SList`.
6. Shell state doesn't persist across `!` calls — use magic equivalents like `%cd` for durable changes.

## Connects To
- **Ch 1**: builds directly on the `?`/Tab discovery tools introduced there.
- **Ch 3**: `%timeit`/`%time` are the simple cases; Ch 3 covers full profiling (`%prun`, `%lprun`, `%memit`, `%mprun`) and debugging (`%debug`, `%xmode`).
