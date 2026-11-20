# Chapter 11: Sorting Arrays

## Core Idea
NumPy provides fast, vectorized sorting (`np.sort`, `np.argsort`) and partial-sorting (`np.partition`, `np.argpartition`) routines, and combining `argsort`/`argpartition` with broadcasting enables efficient vectorized algorithms like k-nearest-neighbors without explicit loops.

## Frameworks Introduced
- **`np.sort` / `np.argsort`**: full sort returning values or index positions (indices usable via fancy indexing to reconstruct sorted order); `axis=` sorts independently along rows or columns.
  - When to use: need a fully ordered array or its ranking, possibly per-row/column.
- **`np.partition` / `np.argpartition`**: partial sort — puts the k smallest elements (unordered among themselves) to the left of position k, remainder to the right; returns either values or indices.
  - When to use: need only the k smallest/largest elements, not a full ordering — cheaper than a full sort.
- **Vectorized k-NN pattern**: pairwise squared distances via broadcasting (`X[:, np.newaxis] - X[np.newaxis, :]`) + `.sum(-1)`, then `np.argsort`/`np.argpartition` along `axis=1` to rank/select nearest neighbors per point — all without explicit loops.

## Key Concepts
- **`sorted()` / `list.sort()`**: Python builtins — `sorted` returns a new sorted list, `.sort()` sorts in place; flexible but slow for large numeric arrays vs. NumPy.
- **`np.sort(x)`**: returns a sorted copy; `x.sort()` sorts in-place.
- **`np.argsort(x)`**: returns indices that would sort `x`; `x[np.argsort(x)]` reconstructs the sorted array.
- **Row/column-independent sorting**: `axis=0` sorts each column independently, `axis=1` sorts each row independently — any cross-row/column relationships between values are destroyed.
- **Partitioning**: order within each partition (the k-smallest side and the remainder side) is arbitrary/unspecified.
- **Pairwise distance matrix via broadcasting**: `differences = X[:, np.newaxis] - X[np.newaxis, :]` produces shape `(N, N, D)`; squaring and `.sum(-1)` gives `(N, N)` squared-distance matrix.

## Mental Models
- Sorting along an axis treats each row/column as a fully independent 1D array to be sorted — think of applying `np.sort` separately N times, once per row (or column), not as a global 2D ordering.
- Partitioning is "just enough sorting" — a cheaper structural guarantee (k smallest on one side) when full order isn't needed, trading determinism of internal order for speed.
- The pairwise-distance-matrix trick is the signature "do it with broadcasting, not loops" NumPy idiom: insert `np.newaxis` on each side to create a full "outer" comparison grid (N × N × D), then reduce along the last axis.

## Anti-patterns
- **Using a full `np.argsort` when only the k nearest/smallest are needed**: wastes time ordering elements you'll discard; use `np.argpartition` instead for a cheaper partial result.
- **Sorting a 2D array thinking it preserves row-to-row/column-to-column relationships**: `axis=0`/`axis=1` sorts each row/column independently, discarding any alignment between them.
- **Manually looping over points to compute pairwise distances**: the broadcasting-based vectorized formulation scales to the same code regardless of N or dimensionality, and is far faster than an explicit double loop.

## Code Examples
```python
L = [3, 1, 4, 1, 5, 9, 2, 6]
sorted(L)       # new sorted list
L.sort()        # in-place, returns None

import numpy as np
x = np.array([2, 1, 4, 3, 5])
np.sort(x)       # sorted copy
x.sort()         # in-place

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)         # [1, 0, 3, 2, 4]
x[i]                        # reconstructs sorted array via fancy indexing

rng = np.random.default_rng(seed=42)
X = rng.integers(0, 10, (4, 6))
np.sort(X, axis=0)   # sort each column independently
np.sort(X, axis=1)   # sort each row independently

x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)        # [2, 1, 3, 4, 6, 5, 7] -- first 3 are smallest, unordered
np.partition(X, 2, axis=1)  # per-row partial sort

# k-nearest-neighbors example
X = rng.random((10, 2))
differences = X[:, np.newaxis] - X[np.newaxis, :]   # shape (10, 10, 2)
sq_differences = differences ** 2
dist_sq = sq_differences.sum(-1)                      # shape (10, 10)
dist_sq.diagonal()                                      # all zeros, sanity check

nearest = np.argsort(dist_sq, axis=1)   # full ranking of neighbors per point
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)  # cheaper: just K nearest
```
- **What it demonstrates**: contrasting Python vs. NumPy sort performance characteristics, axis-wise sorting semantics, partial sort/partition, and a complete vectorized nearest-neighbor computation combining broadcasting + argsort/argpartition.

## Reference Tables
| Function | Returns | Use case |
|---|---|---|
| `sorted(list)` | new sorted list | small/generic Python iterables |
| `list.sort()` | None (in-place) | small/generic Python lists |
| `np.sort(x)` | sorted copy | numeric NumPy arrays, fast |
| `x.sort()` | None (in-place) | numeric NumPy arrays, fast |
| `np.argsort(x)` | indices of sorted order | ranking, reordering other arrays in parallel |
| `np.partition(x, k)` | array, k smallest unordered on left | need only k smallest, not full order |
| `np.argpartition(x, k)` | indices, k smallest on left | same, but indices for parallel reordering |

## Worked Example
k-nearest-neighbors on 10 random 2D points: compute all pairwise squared distances via broadcasting into a (10,10) matrix in one expression; `np.argsort(dist_sq, axis=1)` gives, for each point, all other points ranked by distance (first column is always the point itself, distance 0); for just the K=2 nearest neighbors, `np.argpartition(dist_sq, K+1, axis=1)` is cheaper since it doesn't need a full ordering — then plot lines from each point to its `K` nearest per `nearest_partition`. Note: the "nearest neighbor" relation is not necessarily symmetric — some points may have more than K lines pointing to them.

## Key Takeaways
1. `np.sort`/`np.argsort` are fast vectorized alternatives to Python's `sorted`/`.sort()` for numeric arrays.
2. `argsort` returns index order; apply with fancy indexing to reorder related arrays in parallel.
3. `axis=` sorts each row or column independently — no cross-axis relationship is preserved.
4. `np.partition`/`np.argpartition` give a cheaper "k smallest, unordered" guarantee when full sort isn't needed.
5. The broadcasting pattern `X[:, np.newaxis] - X[np.newaxis, :]` computes all pairwise differences in one vectorized expression — a reusable building block for distance-matrix-style problems.
6. For very large nearest-neighbor searches, tree-based/approximate algorithms (e.g., KD-Tree in scikit-learn) scale better than the O(N²) brute-force approach shown here.

## Connects To
- **Ch 8**: the k-NN example directly reuses the broadcasting rules.
- **Ch 10**: `argsort`/`argpartition` outputs are consumed via fancy indexing.
- **Ch 44/Ch 47**: distance-based reasoning recurs in Random Forests/k-Means in Part V.
