# Chapter 5: The Basics of NumPy Arrays

## Core Idea
NumPy array manipulation — attributes, indexing, slicing, reshaping, concatenation/splitting — forms the foundational vocabulary used throughout the rest of the book (and Pandas on top of it).

## Frameworks Introduced
- **Array attribute inspection**: `.ndim`, `.shape`, `.size`, `.dtype` describe an array's structure without touching its data.
- **Slice notation `x[start:stop:step]`**: generalizes Python list slicing to N dimensions via comma-separated slices; negative `step` reverses and swaps start/stop defaults.
- **View-vs-copy semantics**: slices return *views* (shared memory) by default; `.copy()` forces an independent copy.
- **Reshaping (`reshape`, `np.newaxis`)**: changes an array's shape without changing its data, as long as total size matches; `np.newaxis` inserts a new length-1 axis for quick row/column-vector conversion.
- **Concatenation/splitting**: `np.concatenate`/`vstack`/`hstack`/`dstack` join arrays; `np.split`/`vsplit`/`hsplit`/`dsplit` divide them at given index points.

## Key Concepts
- **`ndim`, `shape`, `size`, `dtype`**: dimensionality, per-axis lengths, total element count, element type.
- **Fancy comma indexing**: `x2[row, col]` for multidimensional element access.
- **Silent truncation**: assigning a float into an int-typed array truncates without warning (fixed dtype enforced).
- **View**: a slice that shares the same underlying data buffer as the original array — mutating one mutates the other.
- **`axis` parameter**: determines which dimension an operation (e.g., `concatenate`) acts along (axis 0 = rows/vertical, axis 1 = columns/horizontal).
- **N split points → N+1 subarrays**: the rule for `np.split`.

## Mental Models
- Think of slicing as a "window" onto existing memory, not a new allocation — this is why views propagate edits back to the original, and why it's efficient for huge datasets (no copy needed for inspection).
- `np.newaxis` is a notational trick for reshape: inserting it is equivalent to `.reshape()` with an extra `1` in that position.

## Anti-patterns
- **Mutating a slice expecting it not to affect the original array**: NumPy slices are views, not copies — use `.copy()` explicitly when independence is required.
- **Assigning floats into int arrays without checking dtype**: silent truncation can introduce subtle bugs.
- **Forgetting that `vstack`/`hstack` are for mismatched dimensions**: use `np.concatenate(..., axis=...)` when dimensions already match for clarity.

## Code Examples
```python
import numpy as np
rng = np.random.default_rng(seed=1701)
x1 = rng.integers(10, size=6)
x2 = rng.integers(10, size=(3, 4))
x3 = rng.integers(10, size=(3, 4, 5))

x3.ndim, x3.shape, x3.size, x3.dtype   # 3, (3,4,5), 60, int64

x1[0]; x1[-1]                 # single-element indexing, from front/back
x2[0, 0]; x2[2, -1]            # multidimensional indexing
x2[0, 0] = 12                  # in-place modification
x1[0] = 3.14159                # silently truncated to 3 (int array)

x1[:3]; x1[3:]; x1[1:4]
x1[::2]; x1[1::2]
x1[::-1]; x1[4::-2]            # reversed slices

x2[:2, :3]                     # first two rows, three columns
x2[:3, ::2]                    # every second column
x2[::-1, ::-1]                 # reverse both axes
x2[:, 0]                       # first column
x2[0, :]; x2[0]                # first row (equivalent forms)

x2_sub = x2[:2, :2]            # VIEW, not copy
x2_sub[0, 0] = 99              # mutates x2 too!
x2_sub_copy = x2[:2, :2].copy()  # independent copy

grid = np.arange(1, 10).reshape(3, 3)
x = np.array([1, 2, 3])
x.reshape((1, 3))              # row vector
x.reshape((3, 1))              # column vector
x[np.newaxis, :]               # row vector via newaxis
x[:, np.newaxis]               # column vector via newaxis

np.concatenate([x, y, z])
np.concatenate([grid, grid], axis=1)
np.vstack([x, grid])
np.hstack([grid, y])

x1_, x2_, x3_ = np.split(x, [3, 5])     # N+1 = 3 subarrays from 2 split points
upper, lower = np.vsplit(grid, [2])
left, right = np.hsplit(grid, [2])
```
- **What it demonstrates**: the full basic-manipulation vocabulary: attributes, indexing, slicing (1D/ND), view semantics, reshape/newaxis, and join/split operations.

## Reference Tables
| Operation category | Key functions |
|---|---|
| Attributes | `.ndim`, `.shape`, `.size`, `.dtype` |
| Indexing | `x[i]`, `x[i, j]`, negative indices |
| Slicing | `x[start:stop:step]`, multi-axis with commas |
| Copy | `.copy()` |
| Reshape | `.reshape(shape)`, `np.newaxis` |
| Concatenate | `np.concatenate`, `np.vstack`, `np.hstack`, `np.dstack` |
| Split | `np.split`, `np.vsplit`, `np.hsplit`, `np.dsplit` |

## Worked Example
Extracting and accidentally mutating a view: `x2_sub = x2[:2, :2]` then `x2_sub[0,0] = 99` changes `x2[0,0]` too, since both share the underlying buffer. To avoid this when independence is needed: `x2_sub_copy = x2[:2, :2].copy()`; modifying `x2_sub_copy` leaves `x2` untouched.

## Key Takeaways
1. `.ndim`/`.shape`/`.size`/`.dtype` are the four basic descriptive attributes of any ndarray.
2. Slices are views sharing memory with the original array — use `.copy()` for independence.
3. Negative step reverses an array and swaps start/stop defaults.
4. `reshape()` requires matching total size and usually returns a view; `np.newaxis` is shorthand for adding a length-1 axis.
5. `concatenate`/`vstack`/`hstack` join; `split`/`vsplit`/`hsplit` divide — N split points yield N+1 pieces.
6. Assigning a value of the "wrong" type into a fixed-dtype array truncates silently.

## Connects To
- **Ch 4**: builds on the fixed-type array model and construction routines.
- **Ch 9/Ch 10**: boolean masking and fancy indexing extend this chapter's basic indexing/slicing into more powerful selection mechanisms.
