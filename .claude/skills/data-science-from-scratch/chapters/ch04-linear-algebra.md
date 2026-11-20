# Chapter 4: Linear Algebra

## Core Idea
Vectors (lists of floats) and matrices (lists of lists) are the book's from-scratch numeric foundation; every later algorithm (regression, gradient descent, neural nets, clustering) is built on `add`, `subtract`, `dot`, `magnitude`, `distance`, and matrix shape/row/column operations defined here in pure Python.

## Frameworks Introduced
- **Vector as `List[float]` type alias**: `Vector = List[float]`; treat any fixed-length numeric record (height/weight/age, exam scores) as a vector.
  - When to use: representing any row of numeric features.
  - How: componentwise `add`/`subtract` via `zip` + list comprehension; `assert len(v) == len(w)` guards mismatched lengths.
- **Dot product as the core primitive**: `dot(v, w) = sum(v_i * w_i)`; `sum_of_squares(v) = dot(v, v)`; `magnitude(v) = sqrt(sum_of_squares(v))`; `distance(v, w) = magnitude(subtract(v, w))`.
  - When to use: computing vector length, projection, or Euclidean distance between two points — used constantly in k-NN, clustering, gradient descent.
- **Matrix as `List[List[float]]` type alias**: `Matrix = List[List[float]]`; `shape(A)` returns `(num_rows, num_cols)`; `get_row`/`get_column` extract a row/column as a `Vector`.
  - When to use: representing a dataset (rows = observations), a linear map (k-dim -> n-dim), or a binary relationship/adjacency matrix.
  - How: `make_matrix(num_rows, num_cols, entry_fn)` builds any matrix from a generator function via nested comprehension; `identity_matrix(n)` is `entry_fn = lambda i, j: 1 if i == j else 0`.

## Key Concepts
- **Vector**: a list of floats representing a point in finite-dimensional space; supports componentwise addition and scalar multiplication.
- **Dot product**: sum of componentwise products; measures how far `v` extends in `w`'s direction when `w` has magnitude 1 (projection).
- **Magnitude (length)**: `sqrt(dot(v, v))`, the Euclidean norm of a vector.
- **Euclidean distance**: `magnitude(subtract(v, w))`, the straight-line distance between two vectors/points.
- **Matrix shape**: `(len(A), len(A[0]))` — an n x k matrix has n rows and k columns, zero-indexed.
- **Adjacency matrix**: an alternative to an edge-pair list for representing a graph; `A[i][j] == 1` iff nodes i and j are connected — O(1) lookup but O(n^2) storage, wasteful for sparse graphs.
- **vector_mean**: componentwise average of a list of same-size vectors, `scalar_multiply(1/n, vector_sum(vectors))`.

## Mental Models
- **Lists-as-vectors trade-off**: representing vectors/matrices as plain Python lists is great for exposition (transparent, no dependencies) but terrible for performance — the book explicitly flags that production code should use NumPy, which provides the same operations with much better speed.
- **Matrices as three things at once**: a dataset (rows = observations), a linear function (maps k-dim to n-dim vectors), or a binary relation (adjacency matrix) — recognizing which role a given matrix plays clarifies what operations make sense on it.
- **Dense vs. sparse representation trade-off**: adjacency-matrix lookups are O(1) but wasteful in storage for sparse graphs; adjacency lists (Ch1/Ch22) are the opposite trade-off.

## Anti-patterns
- **Adding/combining vectors of different lengths**: guarded everywhere with `assert len(v) == len(w)`, since there's no sensible componentwise operation otherwise.
- **Using Python lists as vectors in production/performance-sensitive code**: fine for teaching, bad for real workloads — use NumPy arrays instead.
- **Using a dense adjacency matrix for a very sparse, large, evolving graph**: wastes memory on mostly-zero entries; an adjacency list or per-node connection list is usually better at scale.

## Code Examples
```python
Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def magnitude(v: Vector) -> float:
    return math.sqrt(dot(v, v))

def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))
```
- **What it demonstrates**: the full from-scratch vector arithmetic toolkit used throughout the rest of the book.

```python
Matrix = List[List[float]]

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)
```
- **What it demonstrates**: generic matrix construction via a generator function, specialized to build an identity matrix.

```python
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                 # ... 10x10 adjacency matrix for DataSciencester
                ]
friends_of_five = [i for i, is_friend in enumerate(friend_matrix[5]) if is_friend]
```
- **What it demonstrates**: representing a graph as an adjacency matrix instead of an edge-pair list, and using it for O(1) connection lookups.

## Reference Tables
| Operation | Signature | Formula |
|---|---|---|
| `add(v, w)` | Vector, Vector -> Vector | `v_i + w_i` componentwise |
| `subtract(v, w)` | Vector, Vector -> Vector | `v_i - w_i` componentwise |
| `vector_sum(vectors)` | List[Vector] -> Vector | componentwise sum |
| `scalar_multiply(c, v)` | float, Vector -> Vector | `c * v_i` |
| `vector_mean(vectors)` | List[Vector] -> Vector | `scalar_multiply(1/n, vector_sum(vectors))` |
| `dot(v, w)` | Vector, Vector -> float | `sum(v_i * w_i)` |
| `sum_of_squares(v)` | Vector -> float | `dot(v, v)` |
| `magnitude(v)` | Vector -> float | `sqrt(sum_of_squares(v))` |
| `distance(v, w)` | Vector, Vector -> float | `magnitude(subtract(v, w))` |
| `shape(A)` | Matrix -> (int, int) | `(len(A), len(A[0]))` |

## Worked Example
DataSciencester's 10-user friendship graph from Ch1 (`friendship_pairs`) is re-expressed as a 10x10 `friend_matrix` of 0s and 1s. `friend_matrix[0][2] == 1` confirms users 0 and 2 are friends; `friend_matrix[0][8] == 0` confirms 0 and 8 are not. Finding all of user 5's friends becomes a one-line list comprehension over row 5, contrasted with the O(edges) scan needed on the pair-list representation from Ch1.

## Key Takeaways
1. `Vector = List[float]` and `Matrix = List[List[float]]` type aliases, plus `add/subtract/dot/magnitude/distance`, are the atomic operations reused in nearly every later chapter.
2. The dot product underlies projection, length, and distance — understand it once, reuse it everywhere (regression coefficients, gradient descent steps, k-NN distances).
3. Choose adjacency matrix vs. adjacency list by sparsity: dense/small graphs favor matrices (O(1) lookup); large sparse graphs favor lists (less memory).
4. This from-scratch vector/matrix code trades performance for clarity; swap in NumPy for production.

## Connects To
- **Ch1**: re-represents the DataSciencester friendship graph as an adjacency matrix instead of edge pairs.
- **Ch12 (k-Nearest Neighbors)**: uses `distance()` directly to find nearest neighbors.
- **Ch8 (Gradient Descent)**, **Ch15 (Multiple Regression)**, **Ch18-19 (Neural Networks/Deep Learning)**: all build on vectors/matrices and `dot` for weighted sums and parameter updates.
- **NumPy**: the library the book says "gives you all of this for free," with far better performance, recommended for any real (non-teaching) use.
