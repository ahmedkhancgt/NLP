# Chapter 3: Using Data Structures Effectively

## Core Idea
Choosing the right data structure for the question you're asking of your data is one of the biggest levers on performance: each structure (list, tuple, dict, set, NumPy array, pandas DataFrame) has characteristic time complexities and memory trade-offs, and picking correctly makes code both faster and easier to use.

## Frameworks Introduced
- **"Pick a data structure that can answer your questions quickly"** (Gorelick & Osvald): a large part of performant programming is identifying the actual query pattern (index lookup? key lookup? membership test? ordered iteration?) and matching it to the structure with the matching O() for that operation.
  - When to use: at design time, before writing processing code, and whenever profiling (Ch 2) points to a structure-related bottleneck.
  - How: ask "do I need order?", "do I need key-based lookup?", "do I need uniqueness/fast membership tests?", "is my data uniform-typed and multidimensional?", "is it naturally tabular?" — then choose list/tuple, dict, set, NumPy array, or pandas DataFrame respectively.
- **Vectorization**: replacing explicit per-element Python loops/type-checks with array-wide operations that drop straight into compiled C code.
  - When to use: any time a NumPy or pandas built-in vectorized method exists for the computation.
  - How: use `np.sum()`/`df['col'].str.lower()` etc. instead of Python loops, `.apply()`, or `.iterrows()`; apply is a fallback when no vectorized method exists, and `.iterrows()` should be avoided.

## Key Concepts
- **List**: Python's dynamic array; ordered, resizable, mixed-type; contiguous memory. Index lookup O(1); append O(1) (amortized, with occasional copy overhead); insert/delete in the middle O(n); search (`in`) O(n).
- **Tuple**: immutable, static-size array; cached by the runtime, so lookup is O(1) and even faster than a list; best for fixed, small, unchanging collections.
- **Dictionary**: key-value store backed by a hash table; a hash function maps keys to list indices. Lookup, insert, update, delete are all O(1); keys must be hashable (str, int, float — not a list) and unique.
- **Set**: hash-table-based collection of unique keys (no values); lookup/insert/delete/update all O(1); ideal for fast membership testing and counting unique elements, but converting a list to a set has an upfront O(n) cost that only pays off with repeated lookups.
- **NumPy ndarray (n-dimensional array)**: single-typed, contiguous array; enables vectorized operations that skip per-element type-checking and go straight to compiled C code — can be ~100x faster than Python loops for applicable operations.
- **View vs. copy**: slicing a NumPy array returns a view (shares memory with the original — changes propagate both ways) whereas slicing/selecting from a list creates a new copy; views are faster and more memory-efficient.
- **dtype**: the single element type stored with a NumPy array (e.g., `int64`, `int32`, `int16`); choosing a smaller dtype that still covers your data's range cuts memory footprint (e.g., `int64`→`int32` halves bytes).
- **Dask**: a library that parallelizes/distributes NumPy- and pandas-like operations by splitting arrays/DataFrames into chunks, processing chunks concurrently (across cores or machines) and combining results; also enables working with data larger than memory. Adds complexity/overhead, so use only when needed.
- **pandas Series / DataFrame**: a Series is a 1D NumPy-like array plus an index (label-based lookup); a DataFrame is a 2D collection of Series (shared row index, per-column dtype, column index). Supports vectorized operations (including vectorized string ops) and mixed column types (via an "object" dtype).
- **Sparse matrix**: a memory-efficient structure for matrices that are mostly zeros (e.g., text-vectorization/bag-of-words features); used under the hood by scikit-learn's `CountVectorizer` via SciPy.
- **Polars**: a newer, Rust-based, Apache-Arrow-backed DataFrame library with a pandas-like interface, offering better speed/memory efficiency (less mature than pandas).

## Mental Models
- **Contiguous memory → O(1) indexed lookup, O(n) middle insert/delete**: because lists/tuples/arrays are laid out as one continuous memory block, direct-offset lookup is instant, but any operation that shifts everything after an insertion point costs O(n).
- **Hashing trades memory for speed**: dictionaries/sets convert key lookup from "search everything" (O(n)) to "compute an address" (O(1)) via a hash function — at the cost of a larger memory footprint than a plain list.
- **Single-typedness as a performance unlock, not a limitation**: NumPy's "only one dtype per array" constraint is precisely what lets it skip Python's per-element type dispatch and vectorize into C — the restriction *is* the speedup.
- **Measure, don't assume, the crossover point**: converting a list to a set is only worth it if you'll do many repeated lookups afterward — the conversion itself has O(n) cost that a single lookup doesn't amortize; same logic applies to Dask/Polars adoption (added complexity must be justified by actual need).

## Anti-patterns
- **Searching a list for membership repeatedly**: `x in my_list` is O(n) per call; if checking membership many times, convert to a set first (one O(n) conversion, then O(1) per lookup) rather than repeatedly scanning the list.
- **Using `.iterrows()` to process a DataFrame row-by-row**: measured as the slowest of the three approaches (plain loop/apply/vectorized) in the chapter's own benchmark — avoid it in favor of vectorized operations or, failing that, `.apply()`.
- **Appending to a NumPy array in a loop**: unlike a Python list, NumPy allocates no extra headroom, so each append is a full O(n) reallocation/copy; preallocate with `np.zeros(n)` and fill in values instead.
- **Using a NumPy array for mixed-type data**: NumPy silently coerces all elements to one type (e.g., everything becomes a string) rather than raising an error — a correctness trap, not just a performance one.
- **Reaching for Dask/Polars/GPU tensors before measuring need**: added complexity is only worth it when native structures (list/dict/set/NumPy/pandas) have already been tried and shown insufficient for the data size or parallelism required.

## Code Examples
```python
# List: O(1) lookup regardless of size (indexing), O(n) search
small_list = list(range(10))
large_list = list(range(10_000))
# small_list[-1] and large_list[-1]: ~same time (O(1))
# 4200 in small_list vs 4200 in large_list: large ~1000x slower (O(n))
```
- **What it demonstrates**: list indexing is constant time; list membership search scales linearly with size.

```python
# Set conversion amortizes over repeated lookups
>>> %%timeit
>>> large_set = set(large_list)
>>> 4200 in large_set
# first lookup (incl. conversion): ~76.7 µs — slower than one list lookup (23.8 µs)
>>> %%timeit
>>> 4592 in large_set
# subsequent lookup: ~22 ns — ~1000x faster than list lookup
```
- **What it demonstrates**: the one-time O(n) cost of building a set pays off only with repeated O(1) lookups afterward.

```python
# NumPy: multidimensional slicing vs. nested-list workaround
python_2d_list = [[1, 3, 5], [2, 4, 6], [7, 9, 11]]
first_column = [python_2d_list[i][0] for i in range(len(python_2d_list))]

import numpy as np
np_2d_array = np.array([[1, 3, 5], [2, 4, 6], [7, 9, 11]])
first_column = np_2d_array[:, 0]   # a VIEW, not a copy
```
- **What it demonstrates**: NumPy slicing syntax replaces manual iteration and returns a memory-sharing view instead of a copy.

```python
# Vectorization speedup: built-in sum vs. np.sum
random_int_array = np.random.randint(1, 100_000, 100_000)
random_int_list = list(random_int_array)
# sum(random_int_list): ~1.96 ms
# np.sum(random_int_array): ~13.9 µs  (~100x faster)
```
- **What it demonstrates**: vectorized NumPy operations vastly outperform Python-level loops/built-ins on arrays.

```python
# Reducing memory footprint via dtype
random_int_array = np.random.randint(1, 100_000, 100_000)
random_int_array.nbytes            # 800000 (int64)
random_int_array_32 = random_int_array.astype(np.int32)
random_int_array_32.nbytes         # 400000 (half the memory)
small_array = np.array([1, 3, 5], dtype=np.int16)
```
- **What it demonstrates**: choosing a smaller dtype that still fits your data's range halves (or further reduces) memory usage.

```python
# Dask: parallel max() vs. NumPy max()
large_np_array = np.random.randint(1, 100000, 1_000_000_000)
# np.max(large_np_array): ~30.7 s

import dask.array as da
large_dask_array = da.from_array(large_np_array)
array_max = large_dask_array.max()
array_max.compute()
# ~1.51 s  (~20x faster)
```
- **What it demonstrates**: Dask splits the array into chunks and parallelizes the reduction, dramatically cutting runtime on very large arrays.

```python
# pandas: vectorized op vs. apply vs. iterrows
df["India_fraction"] = df["India"] / 100                                 # ~65.7 µs — vectorized, fastest
df["India_fraction"] = df["India"].apply(lambda x: x / 100)              # ~87.7 µs — apply, some overhead
df["India_fraction"] = [row['India']/100 for index, row in df.iterrows()] # ~348 µs — iterrows, slowest
```
- **What it demonstrates**: a clear performance ordering — vectorized > apply > iterrows — for the same calculation.

## Reference Tables
| Structure | Ordered? | Mutable? | Lookup | Insert/Delete | Best for |
|---|---|---|---|---|---|
| List | Yes | Yes | O(1) by index, O(n) search | O(1) append (amortized), O(n) middle | Ordered, index-based access |
| Tuple | Yes | No | O(1) | N/A (immutable) | Small fixed collections |
| Dict | No (insertion order preserved in practice) | Yes | O(1) by key | O(1) | Key→value lookups |
| Set | No | Yes | O(1) | O(1) | Fast membership tests, uniqueness |
| NumPy array | Yes (n-dim) | Yes | O(1) | O(n) append (no headroom) | Multidimensional, single-type numeric data; vectorized math |
| pandas DataFrame | Yes (index-based) | Yes | O(1)-ish by label | Varies | Tabular, mixed-column-type, time series data |

| Tool for scale | Purpose |
|---|---|
| Dask | Parallelize array/DataFrame ops across cores/machines; handle data > memory |
| Polars | Faster, more memory-efficient pandas-like DataFrame library (Rust + Arrow) |
| TensorFlow / PyTorch tensors | GPU-optimized structures for ML training (easy NumPy conversion) |
| SciPy sparse matrix | Memory-efficient storage for mostly-zero matrices (e.g., text features) |

## Worked Example
Measuring dictionary lookup time with the `Faker` library: build a 10-entry dict and a 10,000-entry dict of fake names→addresses, then time `small_dict['Erin Hogan']` (~21.7 ns) vs. `large_dict['Nicole Morgan']` (~23.9 ns). Despite the 1,000x size difference, lookup time is essentially unchanged — empirically confirming dict lookup is O(1), a direct consequence of hash-table indexing (hash(key) → list index) rather than linear search.

## Key Takeaways
1. Match the data structure to the query pattern: lists for index-based access, dicts for key lookups, sets for membership tests, NumPy arrays for multidimensional single-type numeric data, pandas DataFrames for tabular/mixed-type data.
2. Dict and set operations (lookup, insert, update, delete) are O(1) thanks to hash tables; list search is O(n) — convert to a set if you'll do many repeated membership checks.
3. NumPy's single-dtype constraint enables vectorization, which can be ~100x faster than native Python loops; always prefer a vectorized method when one exists.
4. Preallocate NumPy arrays (`np.zeros`) rather than appending in a loop, since NumPy arrays have no spare capacity and each append triggers a full copy (O(n)).
5. Choosing a smaller NumPy dtype (e.g., int32 vs int64) that still covers your data's range cuts memory usage substantially.
6. In pandas, prefer vectorized operations, then `.apply()`, and avoid `.iterrows()` — benchmarks in this chapter show a clear performance ordering.
7. Reach for Dask (parallel/out-of-memory) or Polars (faster pandas alternative) only once native structures and vectorization have been tried and are insufficient.

## Connects To
- **Ch 2**: Big O notation and profiling are the tools used throughout this chapter to justify structure choices.
- **Ch 4**: `.apply()` with custom functions connects to functional-programming style transformations on DataFrames.
- **External**: *A Common-Sense Guide to Data Structures and Algorithms in Python* (Jay Wengrow) for structures not covered here (linked lists, heaps, trees); *Python for Data Analysis* (Wes McKinney) for full NumPy/pandas reference; *High Performance Python* (Gorelick & Osvald); *Scaling Python with Dask* (Karau & Kimmins).
