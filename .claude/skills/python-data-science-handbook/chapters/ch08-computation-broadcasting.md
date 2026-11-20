# Chapter 8: Computation on Arrays: Broadcasting

## Core Idea
Broadcasting is NumPy's rule-based mechanism for applying binary ufuncs to arrays of different (but compatible) shapes, by conceptually "stretching" smaller arrays without actually copying memory.

## Frameworks Introduced
- **The three broadcasting rules**: determine whether/how two arrays of different shapes can be combined element-wise.
  - Rule 1: if arrays differ in number of dimensions, pad the smaller shape with 1s on the left.
  - Rule 2: if shapes disagree in any dimension where one is 1, stretch that dimension to match.
  - Rule 3: if sizes disagree in any dimension and neither is 1, raise an error (incompatible shapes).
  - When to use: combining a scalar with an array, a 1D array with a 2D array, or two arrays with complementary shapes (e.g., row vector + column vector → outer-product-like grid).

## Key Concepts
- **Broadcasting**: conceptual stretching/duplication of smaller arrays to match shapes for element-wise ops, without actual memory duplication.
- **Shape padding**: prepending `1`s to the shape tuple of the lower-dimensional array.
- **Shape incompatibility**: `ValueError: operands could not be broadcast together with shapes ...` when rule 3 is hit.
- **Left- vs. right-side padding**: broadcasting always pads on the left; to pad on the right, use `np.newaxis` explicitly.
- **`logaddexp`**: example of a binary ufunc (beyond `+`) that also obeys broadcasting rules.

## Mental Models
- Visualize broadcasting as overlaying a "light box" (the stretched/duplicated values) onto the array grid — useful for shape reasoning, even though NumPy never actually materializes the duplicated data in memory.
- Broadcasting compatibility is a purely shape-based negotiation performed dimension-by-dimension, independent of what the actual values are.

## Anti-patterns
- **Assuming shapes can be padded on the right automatically**: broadcasting only pads on the left; use `a[:, np.newaxis]` to achieve right-side alignment explicitly.
- **Not checking shapes before combining arrays**: leads to confusing `ValueError`s; reason through the three rules dimension-by-dimension first.
- **Worrying broadcasting wastes memory**: it does not copy data for the stretched dimensions — safe to use even on large arrays.

## Code Examples
```python
import numpy as np
a = np.array([0, 1, 2]); b = np.array([5, 5, 5])
a + b                       # [5, 6, 7] -- same-shape element-wise

a + 5                        # scalar broadcast: [5, 6, 7]

M = np.ones((3, 3))
M + a                         # 1D array broadcast across 2D array's rows

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
a + b                         # both broadcast -> 2D result (outer-sum)

# Example 1: M.shape (2,3), a.shape (3,) -> compatible, result (2,3)
M = np.ones((2, 3)); a = np.arange(3)
M + a

# Example 2: a.shape (3,1), b.shape (3,) -> both stretch to (3,3)
a = np.arange(3).reshape((3, 1)); b = np.arange(3)
a + b

# Example 3: M.shape (3,2), a.shape (3,) -> INCOMPATIBLE
M = np.ones((3, 2)); a = np.arange(3)
M + a   # ValueError: operands could not be broadcast together with shapes (3,2) (3,)

# Fix with explicit right-side padding via newaxis
a[:, np.newaxis].shape       # (3, 1)
M + a[:, np.newaxis]          # now compatible

np.logaddexp(M, a[:, np.newaxis])   # broadcasting works for any binary ufunc

# Centering data (row-wise mean subtraction)
rng = np.random.default_rng(seed=1701)
X = rng.random((10, 3))
Xmean = X.mean(0)
X_centered = X - Xmean
X_centered.mean(0)            # ~0, confirms centering

# Plotting a 2D function via broadcasting
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```
- **What it demonstrates**: scalar/1D/2D broadcasting combinations, the three-rule shape-resolution procedure, an incompatible-shape error, explicit axis control with `np.newaxis`, and two practical use cases (mean-centering, 2D function evaluation for plotting).

## Reference Tables
| Rule | Statement |
|---|---|
| Rule 1 | Pad the smaller-ndim array's shape with 1s on the left |
| Rule 2 | Stretch any dimension of size 1 to match the other array |
| Rule 3 | Mismatched sizes where neither is 1 → error |

| Example | Shape A | Shape B | Result |
|---|---|---|---|
| 1 | (2,3) | (3,) → (1,3) → (2,3) | (2,3) |
| 2 | (3,1) → (3,3) | (3,) → (1,3) → (3,3) | (3,3) |
| 3 | (3,2) | (3,) → (1,3) → (3,3) | ERROR (size mismatch vs (3,2)) |

## Worked Example
Centering a dataset: given `X` of shape (10, 3) representing 10 observations of 3 features, `Xmean = X.mean(0)` gives shape (3,). `X - Xmean` broadcasts `Xmean` (padded to (1,3), stretched to (10,3)) across all 10 rows, subtracting the per-feature mean from every observation in one vectorized operation. Verifying `X_centered.mean(0)` yields ~0 (to machine precision) confirms correctness.

## Key Takeaways
1. Broadcasting lets ufuncs operate on differently shaped arrays by virtually stretching dimensions of size 1 — no memory is actually copied.
2. Three rules fully determine compatibility: pad left with 1s, stretch size-1 dims, error if mismatched and neither is 1.
3. Padding is always on the left; use `np.newaxis` for explicit right-side alignment.
4. Broadcasting applies to any binary ufunc, not just `+`.
5. Two canonical practical uses: centering data by subtracting broadcast means, and evaluating 2D functions over a grid for visualization.

## Connects To
- **Ch 6**: broadcasting is the generalization of ufunc element-wise operations to mismatched shapes.
- **Ch 5**: reuses `np.newaxis` for explicit reshaping/axis insertion.
- **Ch 28**: 2D function plotting example previews density/contour plotting techniques.
- **Ch 38**: the "samples × features" 2D array convention referenced for `X` is the standard scikit-learn data layout.
