# Chapter 2: Analyzing Code Performance

## Core Idea
Measure before you optimize: find out where your code is actually slow or memory-hungry (via timing, profiling, memory profiling), and only then improve it — guided by an understanding of how runtime scales with input size (time complexity / Big O).

## Frameworks Introduced
- **Measure-first optimization workflow**: (1) confirm the code is correct, (2) confirm there's an actual performance requirement, (3) measure (time/profile/memory), (4) only then optimize the identified bottleneck.
  - When to use: any time you're tempted to "make code faster."
  - How: time with `time`/`timeit`; profile with `cProfile`/`line_profiler`; profile memory with `Memray`; then apply a targeted fix (better algorithm, better data structure, built-ins, compiling, async, parallel/distributed).
- **Big O / time-complexity analysis**: describes how runtime grows as input size n grows, independent of hardware; written O(f(n)) as an upper bound, dropping constants/lower-order terms.
  - When to use: when you need to predict how code will behave as data scales, or can't feasibly measure runtime directly (e.g., choosing between algorithms before implementing).
  - How: count the "steps" as a function of n (e.g., single loop → n+2 steps ≈ O(n); nested loop over both inputs → O(n²)); compare using the growth-rate class, not exact constants.
- **Methods to improve performance (menu)**: algorithm choice, data structure choice, built-in functions, compiling (Cython/Numba/PyPy), asynchronous code, parallel/distributed computing.
  - When to use: once a bottleneck is identified via measurement.
  - How: pick the lever: for CPU-bound Python loops, better algorithm/data structure or built-ins (often C-implemented, faster); for I/O-bound waits, async; for CPU-parallelizable work, `multiprocessing` or distributed computing; for numeric-heavy code, a compiler (Numba = compiles a Python subset; Cython = Python superset with C typing; PyPy = JIT reimplementation of Python).

## Key Concepts
- **Premature optimization**: optimizing before knowing time/memory requirements or where bottlenecks are (Knuth: "premature optimization is the root of all evil"); avoid it — optimize only once a real need and real bottleneck are established.
- **Performance**: minimizing computer resource usage (time and/or memory), typically with a trade-off between the two (faster code often uses more memory).
- **Profiling**: a measurement technique that reports which parts of a function/script consume the most time (or memory), more detailed than basic timing.
- **cProfile**: Python's built-in profiler; function-call-level breakdown (`tottime`, `cumtime`, `ncalls`); output is comprehensive but hard to map to source lines. SnakeViz gives a graphical view of cProfile output.
- **line_profiler**: third-party profiler giving a per-line time breakdown within a target function (`%lprun -f func func(...)`); much more readable than cProfile for finding the exact slow line.
- **Memray**: Bloomberg's memory profiler; runs a script (`memray run script.py`) and generates reports (e.g., `memray flamegraph`) showing per-line memory allocation.
- **Time complexity**: a description of how an algorithm's runtime scales with input size n, independent of hardware — an asymptotic trend, not a wall-clock measurement.
- **Big O notation O(f(n))**: an upper bound on the number of operations as a function of n; constants and lower-order terms are dropped since they become insignificant as n grows large.
- **O(1) / constant time**: runtime independent of dataset size (e.g., indexing a list element).
- **O(n) / linear time**: runtime grows proportionally with n (e.g., a single pass/loop).
- **O(n²) / quadratic time**: runtime grows with the square of n (e.g., nested loops over the same data) — gets slow very quickly; a prime target for algorithmic improvement.
- **O(2ⁿ) / exponential, O(log n) / logarithmic, O(n log n)**: other common classes; exponential often arises from recursive algorithms; logarithmic often arises in search algorithms.

## Mental Models
- **"Correct first, then timely, then necessary"**: performance work only makes sense after correctness is established and only when there's an actual requirement driving it (e.g., a 100ms SLA for a user-facing prediction). No requirement, no need to optimize.
- **Big O as a hardware-independent trend line**: rather than asking "how many seconds," ask "how does runtime change if I double n?" — this predicts behavior across machines and future data growth, where raw timing cannot.
- **"What you measure matters"**: a data-scientist instinct directly transferable to code — don't guess where the bottleneck is, collect profiling data and let it tell you.
- **Realistic range of n**: Big O classification matters only in proportion to how large n will realistically get; O(n²) is irrelevant if n is always ≤ 5.

## Anti-patterns
- **Optimizing without measuring**: guessing which part of the code is slow wastes effort and may miss the real bottleneck (this is what profiling is for).
- **Optimizing before correctness or before there's a real requirement**: fixing performance on code that doesn't work yet, or that already meets its speed requirement, is wasted effort (premature optimization).
- **Nested loops over the same or related data**: a common source of avoidable O(n²) behavior (illustrated by the slow covariance function) when a single-pass O(n) algorithm would do.
- **Reinventing built-ins**: hand-rolled implementations of things like frequency counting are typically slower than Python's built-in/C-implemented equivalents (e.g., `collections.Counter`).

## Code Examples
```python
# Timing with the time module
import time

start = time.time()
slow_way_to_calculate_mode(random_integers)
end = time.time()
print(end - start)
```
- **What it demonstrates**: simplest manual timing of a function call.

```python
# Timing with timeit in a standalone script
import numpy as np
import timeit

random_integers = np.random.randint(1, 100_000, 1000)

def slow_way_to_calculate_mode(list_of_numbers):
    result_dict = {}
    for i in list_of_numbers:
        if i not in result_dict:
            result_dict[i] = 1
        else:
            result_dict[i] += 1
    mode_vals = []
    max_frequency = max(result_dict.values())
    for key, value in result_dict.items():
        if value == max_frequency:
            mode_vals.append(key)
    return mode_vals

mode_timer = timeit.Timer(stmt="slow_way_to_calculate_mode(random_integers)",
                          setup="from __main__ import"\
                          "slow_way_to_calculate_mode,random_integers")
time_taken = mode_timer.timeit(number=10)
print(f"Execution time: {time_taken} seconds")
```
- **What it demonstrates**: robust repeated-measurement timing outside Jupyter (vs. `%%timeit` cell magic inside Jupyter).

```python
# Faster rewrite using a built-in
from collections import Counter

def mode_using_counter(list_of_numbers):
    c = Counter(list_of_numbers)
    return c.most_common(1)[0][0]
```
- **What it demonstrates**: hand-written dict-counting (267ms) replaced by `Counter` (23ms) — built-ins implemented in C beat hand-rolled Python loops.

```python
# O(n): weighted mean — single loop
def weighted_mean(list_of_numbers, weights):
    running_total = 0
    for i in range(len(list_of_numbers)):
        running_total += (list_of_numbers[i] * weights[i])
    return (running_total / sum(weights))
```
- **What it demonstrates**: steps = n + 2 → classified as O(n).

```python
# O(n^2): covariance with nested loops
def covariance(X, Y):
    cov_sum = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            cov_sum += 0.5 * (X[i] - X[j]) * (Y[i] - Y[j])
    return cov_sum / (len(X) ** 2)

# O(n): rewritten with a single loop
def covariance_fast(X, Y):
    avg_X = sum(X) / len(X)
    avg_Y = sum(Y) / len(Y)
    result = 0
    for i in range(len(X)):
        result += (X[i] - avg_X) * (Y[i] - avg_Y)
    return result / len(X)
```
- **What it demonstrates**: nested loops (O(n²)) rewritten as a single pass (O(n)) by precomputing the means — algorithmic choice directly changes complexity class.

## Reference Tables
| Big O class | Name | Example |
|---|---|---|
| O(1) | Constant time | Indexing an element in a list |
| O(log n) | Logarithmic time | Many search algorithms |
| O(n) | Linear time | Single loop (weighted mean) |
| O(n log n) | — | Efficient sorting algorithms |
| O(n²) | Quadratic time | Nested loop over same data (naive covariance) |
| O(2ⁿ) | Exponential time | Many recursive algorithms |

| Tool | Granularity | Use case |
|---|---|---|
| `time` module | Whole block, single run | Quick manual check |
| `timeit` / `%%timeit` | Whole statement, repeated runs | Reliable mean ± std dev timing |
| `cProfile` (+ SnakeViz) | Per function call | Find which function in a script is slow |
| `line_profiler` | Per line within a function | Pinpoint the exact slow line |
| `Memray` | Per line, memory | Find high memory-allocation lines |

## Worked Example
Profiling `mode_using_counter(10_000_000)` with `line_profiler` shows: generating the random integer array takes 4.8% of total time, while `Counter(some_list)` takes 95.0% of total time. This tells you precisely where to focus optimization effort (the `Counter` call) rather than guessing — and in a longer script, `cProfile`/SnakeViz would be used first to find which *function* is the bottleneck, then `line_profiler` to drill into which *line* within it.

## Key Takeaways
1. Don't optimize until correctness is established and a real performance requirement exists ("premature optimization is the root of all evil").
2. Always measure first — use `time`/`timeit` for quick checks, `cProfile`/`line_profiler` for detailed breakdowns, `Memray` for memory.
3. `line_profiler` and SnakeViz-visualized `cProfile` are much easier to interpret than raw `cProfile` output.
4. Built-in functions (e.g., `collections.Counter`) are usually faster than hand-rolled equivalents because they're implemented in C.
5. Big O notation describes hardware-independent growth trends (O(1), O(n), O(n²), O(2ⁿ), O(log n), O(n log n)) — use it to predict how code will scale, especially nested loops (O(n²)) which should usually be avoided when a single-pass alternative exists.
6. Consider the realistic range of n for your use case — complexity class matters only if n will actually grow large.

## Connects To
- **Ch 1**: Performance is one of the five pillars of good code.
- **Ch 3**: Choosing the right data structure (e.g., dict/set O(1) lookup vs. list O(n) search) is the next lever after algorithm choice; Big O is used throughout.
- **External**: *High Performance Python* by Micha Gorelick and Ian Osvald (O'Reilly) for deeper performance work; FastAPI docs for an async-code introduction; `snakeviz`, `line_profiler`, `memray` packages (pip-installable).
