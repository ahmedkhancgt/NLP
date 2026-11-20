# Chapter 9: Comparisons, Masks, and Boolean Logic

## Core Idea
NumPy implements comparison operators as element-wise ufuncs producing Boolean arrays, which can then be aggregated (counted) or used directly as masks to select/filter array data — the standard efficient alternative to manual looping.

## Frameworks Introduced
- **Comparison-ufunc + aggregation pattern**: `np.sum(condition)` / `np.count_nonzero(condition)` / `np.any(condition)` / `np.all(condition)` answer "how many / any / all" questions without explicit loops.
  - When to use: counting values meeting a criterion, optionally per-axis (`axis=` parameter works same as other aggregates).
- **Boolean masking**: `x[boolean_array]` returns a flattened 1D array of exactly the elements where the mask is `True`.
  - When to use: extracting/filtering subsets of data by condition (e.g., `rainfall_mm[rainy]`).
- **Bitwise Boolean combination**: `&`, `|`, `^`, `~` combine Boolean arrays element-wise (vs. Python's `and`/`or`/`not`, which operate on whole objects).

## Key Concepts
- **Comparison ufuncs**: `<`,`>`,`<=`,`>=`,`!=`,`==` map to `np.less`, `np.greater`, `np.less_equal`, `np.greater_equal`, `np.not_equal`, `np.equal`.
- **`np.count_nonzero`**: counts `True`/nonzero entries in an array.
- **`np.sum` on Boolean arrays**: `True`→1, `False`→0, so summing counts matches; supports `axis=`.
- **Bitwise ufuncs**: `&`→`np.bitwise_and`, `|`→`np.bitwise_or`, `^`→`np.bitwise_xor`, `~`→`np.bitwise_not`; applied to Boolean arrays they act element-wise.
- **Masking**: using a Boolean array as an index selects corresponding `True`-positioned elements, returning a 1D array.
- **`and`/`or` vs. `&`/`|`**: keywords operate on the truthiness of the whole object (ambiguous/error for multi-element arrays); operators operate bitwise/element-wise — for NumPy Boolean arrays, always use `&`/`|`/`~`.

## Mental Models
- A Boolean array is "a string of bits" (1=True, 0=False) — `&`/`|` act on it the same way they'd act on any binary integer representation, bit by bit.
- Masking is filtering: build the condition once as a Boolean array, then reuse it both to aggregate (count/sum) and to index (extract values) — same mask, two different operations.

## Anti-patterns
- **Using `and`/`or` on NumPy Boolean arrays with more than one element**: raises `ValueError: The truth value of an array with more than one element is ambiguous` — use `&`/`|` instead.
- **Omitting parentheses in compound comparisons**: `rainfall_mm > 10 & rainfall_mm < 20` is parsed wrong due to operator precedence (`&` binds tighter than comparisons) — always parenthesize each comparison: `(rainfall_mm > 10) & (rainfall_mm < 20)`.
- **Using Python's builtin `sum`/`any`/`all` on multidimensional Boolean arrays**: different semantics/failure modes than `np.sum`/`np.any`/`np.all` — always use the NumPy versions on arrays.

## Code Examples
```python
import numpy as np
x = np.array([1, 2, 3, 4, 5])
x < 3; x > 3; x <= 3; x >= 3; x != 3; x == 3
(2 * x) == (x ** 2)

rng = np.random.default_rng(seed=1701)
x = rng.integers(10, size=(3, 4))
x < 6                            # 2D Boolean array

np.count_nonzero(x < 6)          # 8
np.sum(x < 6)                    # 8 (True=1, False=0)
np.sum(x < 6, axis=1)             # per-row counts

np.any(x > 8); np.any(x < 0); np.all(x < 10); np.all(x == 6)
np.all(x < 8, axis=1)             # per-row all()

# compound Boolean conditions (rainfall example)
np.sum((rainfall_mm > 10) & (rainfall_mm < 20))     # 16
np.sum(~((rainfall_mm <= 10) | (rainfall_mm >= 20)))  # De Morgan's equivalent, also 16

print("Number days without rain: ", np.sum(rainfall_mm == 0))
print("Rainy days with < 5 mm: ", np.sum((rainfall_mm > 0) & (rainfall_mm < 5)))

# masking
x[x < 5]                          # 1D array of matching values

rainy = (rainfall_mm > 0)
days = np.arange(365)
summer = (days > 172) & (days < 262)
np.median(rainfall_mm[rainy])
np.median(rainfall_mm[summer])
np.median(rainfall_mm[rainy & ~summer])

# keyword vs operator distinction
bool(42 and 0)      # False (whole-object truthiness)
bool(42 or 0)        # True
bin(42 & 59)         # bitwise AND of integer bit patterns
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B                 # element-wise OR
A or B                # ValueError: ambiguous truth value
(x > 4) & (x < 8)     # correct
(x > 4) and (x < 8)   # ValueError
```
- **What it demonstrates**: comparison ufuncs producing Boolean arrays; counting/aggregating with `np.sum`/`np.count_nonzero`/`np.any`/`np.all`; masking to extract subsets; bitwise vs. logical-keyword distinction; a full real-data workflow (Seattle rainfall).

## Reference Tables
| Operator | Equivalent ufunc |
|---|---|
| `==` | `np.equal` |
| `!=` | `np.not_equal` |
| `<` | `np.less` |
| `<=` | `np.less_equal` |
| `>` | `np.greater` |
| `>=` | `np.greater_equal` |

| Bitwise operator | Equivalent ufunc |
|---|---|
| `&` | `np.bitwise_and` |
| `\|` | `np.bitwise_or` |
| `^` | `np.bitwise_xor` |
| `~` | `np.bitwise_not` |

## Worked Example
Seattle 2015 rainfall analysis (365 daily values in `rainfall_mm`): `np.sum(rainfall_mm == 0)` → 221 days without rain; `np.sum(rainfall_mm != 0)` → 144 days with rain; `np.sum(rainfall_mm > 10)` → 34 days with >10mm; `np.sum((rainfall_mm > 0) & (rainfall_mm < 5))` → 83 light-rain days. Then using masks: `rainy = rainfall_mm > 0`, `summer = (days > 172) & (days < 262)`, computing `np.median(rainfall_mm[rainy])` = 3.8mm, `np.median(rainfall_mm[summer])` = 0.0mm, `np.median(rainfall_mm[rainy & ~summer])` = 4.1mm — answering compound questions entirely via vectorized Boolean operations.

## Key Takeaways
1. Comparison operators on arrays are ufuncs returning Boolean arrays, element-wise.
2. `np.count_nonzero`, `np.sum`, `np.any`, `np.all` aggregate Boolean arrays (with `axis=` support).
3. Boolean arrays index arrays directly as masks (`x[mask]`), returning only `True`-positioned elements as a flat array.
4. Always parenthesize individual comparisons when combining with `&`/`|` due to operator precedence.
5. Use `&`/`|`/`~`/`^` (element-wise) rather than `and`/`or`/`not` (whole-object) on NumPy arrays — the latter raise `ValueError` for multi-element arrays.
6. The same mask can drive both aggregation (counts/statistics) and extraction (filtered subarrays) — a key reusable pattern.

## Connects To
- **Ch 5**: masking is an extension of the basic indexing/slicing vocabulary.
- **Ch 10**: Fancy Indexing covers the complementary technique of indexing with integer arrays rather than Boolean masks.
- **Ch 16**: Boolean masks are the core mechanism for handling missing/invalid data (`isna()`-style masks in Pandas).
