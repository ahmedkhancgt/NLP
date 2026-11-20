# Chapter 12: Structured Data: NumPy's Structured Arrays

## Core Idea
NumPy structured arrays let heterogeneous, named fields (like rows in a table) be stored in one compound-dtype array rather than several separate parallel arrays, foreshadowing Pandas' DataFrame which handles this more conveniently.

## Frameworks Introduced
- **Compound dtype specification**: define a single array holding multiple named, differently-typed fields.
  - When to use: compact storage of tabular/record-like data, or interop with C/Fortran structs; otherwise prefer Pandas.
  - How: `np.dtype({'names': (...), 'formats': (...)})`, a list of tuples `[('name','S10'), ...]`, or a comma-separated type string `'S10,i4,f8'`.
- **Record arrays (`np.recarray`)**: structured arrays where fields are also accessible as attributes (`data_rec.age`) instead of only dict-style keys (`data['age']`), at a performance cost.

## Key Concepts
- **Structured array**: single `ndarray` with a compound dtype, where each element is a "row" with multiple named fields of (possibly) different types.
- **Type code syntax**: e.g., `'U10'` = Unicode string max length 10, `'i4'` = 4-byte int, `'f8'` = 8-byte float; optional leading `<`/`>` denotes little-/big-endian.
- **Field access**: `data['name']` gets a column (array across all rows); `data[0]` gets a row (tuple-like record); `data[-1]['name']` combines row+field access.
- **Boolean masking on structured arrays**: works normally, e.g. `data[data['age'] < 30]['name']`.
- **Nested/array-valued fields**: a field can itself be a fixed-shape array, e.g. `('mat', 'f8', (3,3))` — a 3×3 matrix per record; useful for direct interop with C structs.
- **`np.recarray`**: `.view(np.recarray)` converts a structured array for attribute-style field access; slower than dict-style access on the base structured array.

## Mental Models
- A structured array is "a NumPy-native, bare-metal table": memory-compact and C-struct-compatible, but without the rich indexing/joining/grouping machinery of Pandas — think of it as the low-level building block Pandas DataFrames are constructed on top of.
- Field access (`data['name']`) is like selecting a column; row access (`data[0]`) is like selecting a row — same mental model as a 2D table, implemented as a 1D array of compound-typed "rows."

## Anti-patterns
- **Using structured arrays for everyday "data munging" tasks (joins, groupby, missing data)**: Pandas DataFrames provide much richer functionality for this — structured arrays are best reserved for low-level/C-interop use cases.
- **Defaulting to `np.recarray` for convenience without checking cost**: attribute access (`data_rec.age`) is markedly slower (~30x in the book's benchmark) than dict-style access (`data['age']`) — weigh convenience vs. performance.
- **Storing related columns as separate parallel arrays (`name`, `age`, `weight` lists)**: loses the structural relationship between them; a structured array (or better, a DataFrame) keeps them unified.

## Code Examples
```python
import numpy as np
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                           'formats': ('U10', 'i4', 'f8')})
data['name'] = name
data['age'] = age
data['weight'] = weight

data['name']               # all names (column)
data[0]                      # first row, as a tuple-like record
data[-1]['name']             # 'Doug'
data[data['age'] < 30]['name']   # Boolean-masked field selection -> ['Alice','Doug']

# alternate dtype specs
np.dtype({'names': ('name', 'age', 'weight'),
          'formats': ((np.str_, 10), int, np.float32)})
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
np.dtype('S10,i4,f8')        # unnamed fields f0, f1, f2

# nested array-valued field
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
X['mat'][0]                  # 3x3 zero matrix

# record arrays
data_rec = data.view(np.recarray)
data_rec.age                  # attribute-style access
%timeit data['age']           # ~121 ns, fastest
%timeit data_rec['age']       # ~2.41 µs
%timeit data_rec.age          # ~3.98 µs, slowest
```
- **What it demonstrates**: constructing/filling compound-dtype arrays, field/row access, Boolean masking combined with field selection, alternate dtype specification syntaxes, nested array fields, and the recarray convenience/performance trade-off.

## Reference Tables
| Type code | Description | Example |
|---|---|---|
| `'b'` | Byte | `np.dtype('b')` |
| `'i'` | Signed integer | `np.dtype('i4') == np.int32` |
| `'u'` | Unsigned integer | `np.dtype('u1') == np.uint8` |
| `'f'` | Floating point | `np.dtype('f8') == np.float64` |
| `'c'` | Complex float | `np.dtype('c16') == np.complex128` |
| `'S'`, `'a'` | String (bytes) | `np.dtype('S5')` |
| `'U'` | Unicode string | `np.dtype('U') == np.str_` |
| `'V'` | Raw/void data | `np.dtype('V') == np.void` |

## Worked Example
Storing `name`/`age`/`weight` for 4 people as one structured array instead of three parallel lists: build `data` with dtype `{'names': ('name','age','weight'), 'formats': ('U10','i4','f8')}`, assign each field, then query `data[data['age'] < 30]['name']` to get `['Alice', 'Doug']` — a single expression combining Boolean masking and field selection, directly foreshadowing Pandas' `df[df['age'] < 30]['name']` syntax.

## Key Takeaways
1. Structured arrays store heterogeneous named fields in one compact, C-struct-compatible array.
2. Dtype can be specified via dict, list-of-tuples, or shorthand comma-separated string; type codes encode endianness, kind, and byte-width.
3. Access fields by name (`data['field']`, a "column") or rows by index (`data[0]`, a "record"); combine with Boolean masks exactly like other NumPy arrays.
4. Fields can themselves hold fixed-shape sub-arrays, useful for direct C/Fortran struct interop.
5. `np.recarray` trades a small performance cost for attribute-style field access.
6. For general tabular "data munging" (joins, groupby, missing data, etc.), prefer Pandas DataFrames over structured arrays — this chapter is explicitly a bridge into Part III.

## Connects To
- **Ch 4**: builds on fixed-dtype array fundamentals.
- **Ch 9**: Boolean masking technique reused directly on structured array fields.
- **Ch 13**: Pandas' `Series`/`DataFrame` are the natural, far more capable successor to structured arrays for heterogeneous tabular data.
