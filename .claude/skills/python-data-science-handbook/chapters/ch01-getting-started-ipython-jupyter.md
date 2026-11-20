# Chapter 1: Getting Started in IPython and Jupyter

## Core Idea
IPython and Jupyter turn documentation lookup, source inspection, and namespace exploration into a few keystrokes, making "knowing how to find the answer" more productive than memorizing APIs.

## Frameworks Introduced
- **The `?` / `??` / Tab discovery loop**: a workflow for exploring unfamiliar code.
  - When to use: whenever you encounter an unfamiliar function, object, or module.
  - How: `obj?` shows the docstring/signature; `obj??` shows source (when available, i.e. pure Python, not C-extension); `obj.<TAB>` lists attributes/methods; `import <TAB>` lists importable modules; `*Warning?` / `str.*find*?` wildcard-matches names anywhere in the string.

## Key Concepts
- **Docstring**: a string literal as the first statement in a function/class/module body, surfaced by `help()` and `?`.
- **IPython shell**: enhanced interactive Python interpreter (`ipython` on the command line).
- **Jupyter Notebook/Lab**: browser-based interface to a running Python "kernel", launched via `jupyter lab`, supporting rich/mixed text+code+visualization documents.
- **Kernel**: the running Python process a notebook connects to over a local web server.
- **Tab completion**: context-aware autocomplete for object attributes, module contents, and importable names.
- **Wildcard matching (`*`)**: `*` matches any substring (including empty) when searching names with `?`.
- **Reverse-search (Ctrl-r)**: incremental search backward through persistent command history (stored in a SQLite DB in the IPython profile directory).

## Mental Models
- Treat IPython as a "fast local search engine" for code — cheaper than a web search for questions about objects you already have in hand.
- Underscore-prefixed names are hidden from default tab-completion by convention (not enforcement) — explicit `_` reveals dunder/private methods.
- `??` falls back to `?` output silently when source isn't inspectable (compiled/C-extension objects) — absence of source is itself informative (tells you the implementation language).

## Anti-patterns
- **Memorizing exact API signatures instead of learning discovery tools**: wastes effort; `?` and tab-completion make recall unnecessary for most day-to-day work.
- **Assuming `??` always works**: built-ins and C-extension code (e.g., `len`) have no retrievable Python source — don't rely on `??` universally.

## Code Examples
```python
In [1]: help(len)
Help on built-in function len in module builtins:
len(obj, /)
    Return the number of items in a container.

In [2]: len?
Signature: len(obj, /)
Docstring: Return the number of items in a container.
Type: builtin_function_or_method

In [6]: def square(a):
   ....:     """Return the square of a."""
   ....:     return a ** 2

In [7]: square?
Signature: square(a)
Docstring: Return the square of a.

In [8]: square??
Source:
def square(a):
    """Return the square of a."""
    return a ** 2

In [10]: L.<TAB>        # lists append, count, insert, reverse, clear, extend, pop, sort, ...
In [10]: from itertools import co<TAB>   # combinations, combinations_with_replacement, compress, count
In [10]: *Warning?       # lists BytesWarning, RuntimeWarning, DeprecationWarning, ...
In [11]: str.*find*?     # lists str.find, str.rfind
```
- **What it demonstrates**: `?`/`??` for docs/source; `<TAB>` for completion of objects, imports, and top-level namespace; `*pattern*?` for substring search across names.

## Reference Tables
| Keystroke | Action (Navigation) |
|---|---|
| Ctrl-a | Move cursor to beginning of line |
| Ctrl-e | Move cursor to end of line |
| Ctrl-b / Left arrow | Move cursor back one character |
| Ctrl-f / Right arrow | Move cursor forward one character |

| Keystroke | Action (History) |
|---|---|
| Ctrl-p / Up arrow | Previous command (prefix match) |
| Ctrl-n / Down arrow | Next command (prefix match) |
| Ctrl-r | Reverse incremental search through history |

| Keystroke | Action (Misc) |
|---|---|
| Ctrl-l | Clear terminal screen |
| Ctrl-c | Interrupt current command |
| Ctrl-d | Exit IPython session |

## Worked Example
Starting a new IPython session and recovering a previously defined `square` function via reverse-search: press Ctrl-r, type `sqa`, IPython autofills the most recent match (`square??`); press Ctrl-r again to cycle to an older match (`def square(a): ...`); press Return to accept and re-run `square(2)` → `4`.

## Key Takeaways
1. `?` gets docstring+signature; `??` additionally attempts source code (fails gracefully to `?` output for compiled objects).
2. Tab completion works on object attributes, module imports, and the global `import` namespace.
3. `*` wildcard search finds names by substring anywhere, not just prefix.
4. Command history is persistent across sessions (SQLite-backed) and searchable with Ctrl-r.
5. Launch the shell with `ipython`; launch notebooks with `jupyter lab`, which starts a local server and a kernel.

## Connects To
- **Ch 2**: Enhanced Interactive Features extends IPython with magic commands and input/output history built on this same shell.
- **Ch 3**: Debugging and Profiling uses the same `?`-style magic command philosophy (`%debug`, `%timeit`).
