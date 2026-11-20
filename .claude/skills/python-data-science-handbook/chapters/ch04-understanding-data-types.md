# Chapter 4: Understanding Data Types in Python

## Core Idea
Python's dynamic typing makes every value a full object carrying type/refcount metadata, which is flexible but costly at scale; NumPy trades that flexibility for fixed-type, contiguous storage that is far more efficient for numerical arrays.

## Frameworks Introduced
- **Fixed-type array model**: NumPy's `ndarray` stores one pointer to a contiguous block of uniformly typed data, instead of a list's "pointer to a block of pointers to full Python objects."
  - When to use: any time you have homogeneous numerical data and care about memory/speed.
  - How: build with `np.array(list)`, `np.zeros`, `np.ones`, `np.full`, `np.arange`, `np.linspace`, `np.random.*`, `np.eye`, `np.empty`; constrain/inspect type via `dtype=`.

## Key Concepts
- **Dynamic typing**: Python variables carry type info at runtime rather than being declared statically (contrast with C).
- **PyObject overhead**: a Python `int` is a C struct with `ob_refcnt`, `ob_type`, `ob_size`, `ob_digit` — far more than a raw machine integer.
- **Python list of objects**: each element is an independent full Python object (its own type+refcount); flexible (heterogeneous) but memory/CPU-inefficient for uniform numeric data.
- **`array.array`**: built-in fixed-type dense array (type code like `'i'`) — storage-efficient but lacks NumPy's vectorized operations.
- **`ndarray`**: NumPy's efficient, fixed-type, operation-rich array type.
- **Type promotion/upcasting**: NumPy silently upcasts mixed-type input (e.g., ints + float → all float) when building an array.
- **`dtype`**: explicit type specification, as string (`'int16'`) or NumPy object (`np.int16`).

## Mental Models
- A Python list is "a box of boxes" (pointers to objects); a NumPy array is "one box of raw values" — the latter sacrifices per-element flexibility for bulk efficiency.
- Storing numeric data always has type overhead *somewhere*; the question is whether that overhead is paid once per array (NumPy) or once per element (Python list).

## Anti-patterns
- **Storing large homogeneous numeric collections as plain Python lists**: incurs per-element object overhead; use NumPy arrays instead.
- **Assuming array dtype is irrelevant**: mixed-type input causes implicit (sometimes surprising) upcasting; always check `.dtype` when precision/memory matters.

## Code Examples
```python
L = list(range(10))
type(L[0])            # int
L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]   # [bool, str, float, int] -- heterogeneous, costly

import array
A = array.array('i', L)       # fixed-type, no vectorized ops

import numpy as np
np.array([1, 4, 2, 5, 3])                  # array([1, 4, 2, 5, 3])
np.array([3.14, 4, 2, 3])                  # upcast to float: array([3.14, 4., 2., 3.])
np.array([1, 2, 3, 4], dtype=np.float32)
np.array([range(i, i + 3) for i in [2, 4, 6]])   # nested list -> 2D array

np.zeros(10, dtype=int)
np.ones((3, 5), dtype=float)
np.full((3, 5), 3.14)
np.arange(0, 20, 2)
np.linspace(0, 1, 5)
np.random.random((3, 3))
np.random.normal(0, 1, (3, 3))
np.random.randint(0, 10, (3, 3))
np.eye(3)
np.empty(3)   # uninitialized -- contents are whatever garbage is in memory
```
- **What it demonstrates**: contrast between Python list typing cost and NumPy's uniform-type storage, plus the main array-construction routines.

## Reference Tables
| Data type | Description |
|---|---|
| `bool_` | Boolean, stored as a byte |
| `int_` | Default integer (C long; int64/int32) |
| `intc` | C `int` equivalent |
| `intp` | Indexing integer (C `ssize_t`) |
| `int8`/`int16`/`int32`/`int64` | Signed integers of given bit width |
| `uint8`/`uint16`/`uint32`/`uint64` | Unsigned integers of given bit width |
| `float_` | Shorthand for `float64` |
| `float16`/`float32`/`float64` | Floats of given precision |
| `complex_` | Shorthand for `complex128` |
| `complex64`/`complex128` | Complex numbers (two 32-bit or 64-bit floats) |

## Worked Example
Building a 3×3 grid via nested-list input: `np.array([range(i, i+3) for i in [2,4,6]])` produces `[[2,3,4],[4,5,6],[6,7,8]]` — each inner `range` becomes a row, demonstrating how nested sequences map to 2D shape, and setting up the indexing/slicing operations covered in Chapter 5.

## Key Takeaways
1. Every Python object (even an `int`) carries type and refcount metadata — this is the source of Python's flexibility and its per-element overhead.
2. Python lists store pointers to independent objects; NumPy arrays store one contiguous, uniformly typed buffer.
3. `np.array()` infers dtype from input and upcasts mixed types; `dtype=` pins it explicitly.
4. NumPy provides many "from scratch" constructors (`zeros`, `ones`, `full`, `arange`, `linspace`, `random.*`, `eye`, `empty`) that are more efficient than building from Python lists for large arrays.
5. `np.empty` does not initialize memory — do not assume zeros.
6. Compound/structured dtypes (heterogeneous named fields within one array) are covered later, in Chapter 12.

## Connects To
- **Ch 5**: uses these array-construction routines as the basis for indexing/slicing/reshaping.
- **Ch 12**: extends fixed dtypes to compound/structured arrays with named heterogeneous fields.
