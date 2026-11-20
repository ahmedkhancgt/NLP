# Chapter 3: Debugging and Profiling

## Core Idea
IPython provides layered tools — exception-verbosity control, an interactive debugger, and multiple profilers (time, function-level, line-level, memory) — to progressively localize bugs and performance bottlenecks.

## Frameworks Introduced
- **Exception-verbosity control (`%xmode`)**: switch traceback detail between `Plain`, `Context` (default), and `Verbose` (adds local variable values at each frame).
  - When to use: `Plain` for quick glance at complex code; `Verbose` when you need to see argument/variable state at each failing frame.
- **Interactive debugging (`%debug`, `%pdb`, ipdb commands)**: drop into a post-mortem debugger at the exception point; step through the call stack with `up`/`down`, inspect variables with `print`, continue with `c`, quit with `q`.
  - When to use: when the traceback alone doesn't explain the bug.
  - How: run `%debug` right after an exception (post-mortem), or `%pdb on` to auto-launch on every exception, or `%run -d script.py` + `next` to step from the start.
- **Profiling hierarchy**: `%time`/`%timeit` (statement-level) → `%prun` (function-level profiler) → `%lprun` (line-level, needs `line_profiler`) → `%memit`/`%mprun` (memory, needs `memory_profiler`).
  - When to use: narrow down from "is this slow" to "which exact line/function is slow" to "which line uses the most memory."

## Key Concepts
- **Traceback**: the chain of frames showing where/how an exception propagated.
- **pdb / ipdb**: Python's standard debugger and IPython's enhanced version.
- **Post-mortem debugging**: entering the debugger after a crash (via `%debug`) rather than stepping through from the start.
- **`%prun`**: Python's cProfile exposed as a magic — reports `ncalls`, `tottime`, `percall`, `cumtime` per function.
- **`%lprun`**: line-by-line time profiling of a *named* function (requires `%load_ext line_profiler`, `pip install line_profiler`).
- **`%memit` / `%mprun`**: memory-profiling equivalents of `%timeit`/`%lprun` (requires `memory_profiler`); `%mprun` only works on functions defined in an importable module, not in the notebook itself.
- **Premature optimization**: Knuth's warning — don't profile/optimize before the code works and the real bottleneck is known.

## Mental Models
- Debugging/profiling is a funnel: broad signal first (did it crash? how long did it take overall?), then progressively finer-grained tools (function → line → memory) only once a broad signal justifies the investigation cost.
- The debugger's stack is navigable like a Rolodex (`up`/`down`) — you aren't stuck at the exception frame; you can inspect caller state too.

## Anti-patterns
- **Always using Verbose `%xmode`**: tracebacks balloon in size for complex call chains; use Plain/Context by default, Verbose only when needed.
- **Profiling/optimizing before correctness**: violates the "premature optimization" principle — get it working, then measure, then optimize only the proven bottleneck.
- **Trying `%mprun` on a function defined directly in the notebook**: it requires the function to live in an importable `.py` module (use `%%file` to write one).

## Code Examples
```python
In [1]: def func1(a, b):
    return a / b
def func2(x):
    a = x
    b = x - 1
    return func1(a, b)

In [2]: func2(1)   # raises ZeroDivisionError

In [3]: %xmode Plain
In [5]: %xmode Verbose

In [7]: %debug
ipdb> print(a)
1
ipdb> print(b)
0
ipdb> up
ipdb> print(x)
1
ipdb> down
ipdb> quit

In [9]: %xmode Plain
        %pdb on
        func2(1)   # auto-launches ipdb on exception

In [7]: def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total

In [8]: %prun sum_of_lists(1000000)

In [9]: %load_ext line_profiler
In [10]: %lprun -f sum_of_lists sum_of_lists(5000)

In [11]: %load_ext memory_profiler
In [12]: %memit sum_of_lists(1000000)
peak memory: 141.70 MiB, increment: 75.65 MiB

In [13]: %%file mprun_demo.py
def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
        del L
    return total

In [14]: from mprun_demo import sum_of_lists
         %mprun -f sum_of_lists sum_of_lists(1000000)
```
- **What it demonstrates**: full pipeline from a raised exception through traceback modes, stack-navigating debugger, and progressively finer profilers.

## Reference Tables
| Debugger command | Action |
|---|---|
| `l(ist)` | Show current location in file |
| `h(elp)` | List commands / help on one |
| `q(uit)` | Quit debugger and program |
| `c(ontinue)` | Quit debugger, resume program |
| `n(ext)` | Next line |
| `<enter>` | Repeat previous command |
| `p(rint)` | Print a variable |
| `s(tep)` | Step into subroutine |
| `r(eturn)` | Return out of subroutine |

| Magic | Granularity | Needs install |
|---|---|---|
| `%time` / `%%time` | Statement, single run | built-in |
| `%timeit` / `%%timeit` | Statement, many runs (robust) | built-in |
| `%prun` | Per-function | built-in |
| `%lprun` | Per-line (time) | `line_profiler` |
| `%memit` | Statement (memory) | `memory_profiler` |
| `%mprun` | Per-line (memory), module-only | `memory_profiler` |

## Worked Example
`sum_of_lists(1000000)` profiled with `%prun` shows ~0.93s total, with 0.808s (87%) spent in the list comprehension inside the function — immediately pinpointing where optimization effort should go, rather than guessing. `%lprun` on a smaller input then confirms the list-comprehension line consumes 96.2% of line-level time.

## Key Takeaways
1. `%xmode {Plain|Context|Verbose}` controls traceback detail; default is Context.
2. `%debug` opens a post-mortem ipdb session at the exception; `%pdb on` automates this for every exception; `up`/`down` navigate stack frames.
3. Profile in order of granularity: `%time`/`%timeit` → `%prun` → `%lprun` → `%memit`/`%mprun`.
4. `%timeit` disables garbage collection and auto-repeats for accuracy; `%time` is better for stateful operations (e.g., sorting) where repetition changes the input.
5. Line/memory profilers are separate installable extensions (`line_profiler`, `memory_profiler`), loaded via `%load_ext`.
6. Don't optimize before profiling — "premature optimization is the root of all evil" (Knuth).

## Connects To
- **Ch 2**: extends `%timeit`/`%time` introduced there into a full profiling toolkit.
- **Ch 6 / Ch 24**: performance concerns here (vectorization vs. loops) recur as NumPy ufuncs (Ch 6) and Pandas `eval`/`query` (Ch 24) as the actual fixes for profiled bottlenecks.
