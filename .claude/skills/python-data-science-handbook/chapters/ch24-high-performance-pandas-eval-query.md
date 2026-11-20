# Chapter 24: High-Performance Pandas: eval and query

## Core Idea
`pd.eval`/`DataFrame.eval`/`DataFrame.query` use the NumExpr library to evaluate compound expressions element-by-element without allocating full intermediate temporary arrays, trading a small/no speed difference for large memory savings on big datasets.

## Frameworks Introduced
- **String-expression evaluation (NumExpr-backed)**: evaluate NumPy/Pandas-style expressions from a string, avoiding intermediate array allocation.
  - When to use: compound expressions (multiple chained operations) on DataFrames/arrays large enough that temporary allocations matter (roughly, array size approaching your CPU's L1/L2 cache or system memory).
  - How: `pd.eval('expr_string')` (top-level, arithmetic/comparison/bitwise operators, attribute/index access); `df.eval('expr')` (column names usable as bare variables, supports assignment with `inplace=True`); `df.query('boolean_expr')` (for filtering — where masking syntax can't be expressed via `eval`).
- **`@` local-variable marker**: inside `DataFrame.eval`/`DataFrame.query` strings, `@varname` refers to a Python-namespace variable (as opposed to a bare name, which refers to a DataFrame column) — only supported by the DataFrame methods, not top-level `pd.eval`.

## Key Concepts
- **Compound expression overhead**: `(x > 0.5) & (y < 0.5)` is internally equivalent to `tmp1 = x>0.5; tmp2 = y<0.5; mask = tmp1 & tmp2` — each step allocates a full temporary array.
- **NumExpr**: underlying library that evaluates such expressions without materializing every intermediate array.
- **`pd.eval` supported operations**: arithmetic, comparison (including chained, e.g. `'df1 < df2 <= df3 != df4'`), bitwise (`&`, `|`, and literal `and`/`or`), attribute/index access (`obj.attr`, `obj[index]`). NOT supported: function calls, conditionals, loops (use raw NumExpr for those).
- **`DataFrame.eval` column-namespace**: bare names in the expression string resolve to DataFrame columns (`df.eval('(A + B) / (C - 1)')` ≡ `(df.A + df.B) / (df.C - 1)`).
- **Assignment via `df.eval('D = ...', inplace=True)`**: creates or overwrites a column directly from a string expression.
- **`DataFrame.query`**: needed for filtering/masking expressions (`df.query('A < 0.5 and B < 0.5')`) that `DataFrame.eval` cannot express (eval doesn't support the bracket-indexing-as-filter idiom cleanly as a readable string).

## Mental Models
- `eval`/`query` are "memory-first, not speed-first" optimizations — "the benefit... is mainly in the saved memory, and the sometimes cleaner syntax they offer," not raw speed (traditional methods can even be faster for small arrays).
- Decide to use `eval`/`query` based on *temporary array size relative to L1/L2 cache or system RAM*, not based on an assumption that string-expression evaluation is always faster — check `df.values.nbytes` as a quick proxy.

## Anti-patterns
- **Reaching for `eval`/`query` by default on small DataFrames**: traditional vectorized syntax is often just as fast or faster for small data — reserve these tools for large compound-expression workloads where temporary-array memory matters.
- **Expecting `pd.eval`/`df.eval` to support function calls, conditionals, or loops**: these are explicitly NOT implemented — use NumExpr directly for such cases.
- **Forgetting `@` is DataFrame-method-only**: `pd.eval()` (top-level function) has no Python-variable namespace access — `@varname` only works inside `df.eval()`/`df.query()`.

## Code Examples
```python
import numpy as np
rng = np.random.default_rng(42)
x = rng.random(1000000); y = rng.random(1000000)
%timeit x + y                              # fast vectorized op
%timeit np.fromiter((xi+yi for xi, yi in zip(x,y)), dtype=x.dtype, count=len(x))  # slow loop

mask = (x > 0.5) & (y < 0.5)    # implicit temp arrays under the hood

import numexpr
mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')  # no temp arrays
np.all(mask == mask_numexpr)

import pandas as pd
df1, df2, df3, df4 = (pd.DataFrame(rng.random((100000, 100))) for _ in range(4))
%timeit df1 + df2 + df3 + df4              # ~73 ms
%timeit pd.eval('df1 + df2 + df3 + df4')    # ~34 ms, ~50% faster + less memory

pd.eval('-df1 * df2 / (df3 + df4) - df5')       # arithmetic
pd.eval('df1 < df2 <= df3 != df4')               # chained comparison
pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')  # bitwise
pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')  # literal and/or
pd.eval('df2.T[0] + df3.iloc[1]')                  # attribute/index access

df = pd.DataFrame(rng.random((1000, 3)), columns=['A', 'B', 'C'])
df.eval('(A + B) / (C - 1)')                 # column names as bare vars

df.eval('D = (A + B) / C', inplace=True)      # create new column
df.eval('D = (A - B) / C', inplace=True)       # overwrite existing column

column_mean = df.mean(1)
df.eval('A + @column_mean')                     # @ marks local Python variable

df.query('A < 0.5 and B < 0.5')                  # filtering, cleaner than masking
Cmean = df['C'].mean()
df.query('A < @Cmean and B < @Cmean')             # @ works in query too

df.values.nbytes                                   # check array size for eval/query decision
```
- **What it demonstrates**: motivation for compound-expression temp-array overhead, NumExpr's raw speed, `pd.eval`'s supported operation categories, `DataFrame.eval`'s column-namespace shorthand and assignment capability, `@` local-variable access, `DataFrame.query` for filtering, and a memory-size sanity check.

## Reference Tables
| Tool | Namespace(s) | Use case |
|---|---|---|
| `pd.eval('...')` | Python globals/locals only | Top-level compound expressions on existing objects |
| `df.eval('...')` | DataFrame columns + `@`-prefixed Python vars | Column-wise arithmetic, including assignment |
| `df.query('...')` | DataFrame columns + `@`-prefixed Python vars | Boolean filtering/masking |

| Supported in eval | Not supported |
|---|---|
| Arithmetic, comparison (chained), bitwise, `and`/`or`, attribute/index access | Function calls, conditionals, loops |

## Worked Example
Comparing memory/speed trade-offs: for `df1+df2+df3+df4` (4 DataFrames of shape 100000×100), `pd.eval(...)` runs in ~34ms vs. ~73ms for the direct Pandas expression — about 50% faster, while also avoiding the allocation of intermediate temporary DataFrames at each `+`. The decisive factor for choosing `eval`/`query` is whether `df.values.nbytes` approaches the scale of the CPU's L1/L2 cache or system RAM, not an assumption that eval is universally faster.

## Key Takeaways
1. Compound expressions implicitly allocate a full temporary array/DataFrame at every intermediate step — this is the core inefficiency `eval`/`query` address.
2. `pd.eval`, `DataFrame.eval`, and `DataFrame.query` are all NumExpr-backed wrappers; they support arithmetic, comparison, bitwise, and attribute/index operations, but not function calls/conditionals/loops.
3. `DataFrame.eval` lets you reference columns as bare names and supports in-place column assignment (`df.eval('D = ...', inplace=True)`).
4. `DataFrame.query` is the right tool for boolean filtering expressions that `eval` can't cleanly express.
5. `@varname` accesses the Python-object namespace inside `df.eval`/`df.query` strings (not available in top-level `pd.eval`).
6. The benefit is primarily memory savings (and sometimes cleaner syntax), not guaranteed speed — for small arrays, traditional syntax can be just as fast or faster; check `df.values.nbytes` to judge when these tools are worth using.

## Connects To
- **Ch 6**: motivated by the same vectorization-vs-loop performance story, extended to compound-expression temporary-array overhead.
- **Ch 9**: `eval`/`query`'s bitwise/comparison support directly parallels NumPy Boolean masking syntax.
- **Ch 21**: `.query()` was previewed in the pivot-table/birthrate example for outlier filtering.
- **Ch 3**: `%timeit` benchmarking technique used throughout to justify eval/query decisions.
