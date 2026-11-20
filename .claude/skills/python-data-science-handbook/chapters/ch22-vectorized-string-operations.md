# Chapter 22: Vectorized String Operations

## Core Idea
Pandas' `.str` accessor brings vectorized, missing-value-aware string operations to Series/Index objects, mirroring Python's built-in string methods plus regex-based operations — essential for cleaning messy real-world text data.

## Frameworks Introduced
- **The `.str` accessor**: `series.str.<method>()` applies a string operation element-wise across a Series, correctly skipping/propagating `None`/`NaN` without manual checks.
  - When to use: any string transformation/extraction/filtering across a column of text, especially with missing values present.
  - How: most Python string methods are mirrored directly (`.str.lower()`, `.str.len()`, `.str.startswith()`, `.str.split()`); regex-based methods (`.str.match`, `.str.extract`, `.str.findall`, `.str.replace`, `.str.contains`, `.str.count`) follow `re` module conventions; vectorized slicing (`series.str[0:3]`, equivalent to `.str.slice(0,3)`) and chained split+index (`series.str.split().str[-1]`) extract substructure.
- **`get_dummies(sep)`**: splits a delimiter-separated "coded indicator" string column into a one-hot-encoded DataFrame of Boolean indicator columns.

## Key Concepts
- **Vectorization applied to strings**: the `.str` namespace is the string-data analog of NumPy's array vectorization (Ch 6) — avoids manual list-comprehension loops and explicit `None`-checks.
- **Return-type diversity**: `.str` methods variously return a Series of strings (`.lower()`), numbers (`.len()`), Booleans (`.startswith()`), or lists (`.split()`) — same `.str` namespace, different output shapes depending on the operation.
- **Regex convention mapping**: `.match`→`re.match` (per-element Boolean... actually returns match), `.extract`→`re.match` returning captured groups, `.findall`→`re.findall`, `.replace`, `.contains`→`re.search` (Boolean), `.count`, `.split`/`.rsplit` (regex-capable).
- **Vectorized indexing/slicing**: `.str.get(i)`/`.str[i]`, `.str.slice(a,b)`/`.str[a:b]` — array-style access into each string.
- **Combining split + index**: `series.str.split().str[-1]` extracts, e.g., the last "word" (last name) from each full-name string.

## Mental Models
- Treat `.str` exactly like vectorized NumPy ufuncs for text: "don't write a loop over strings; write the one `.str.method()` call that expresses the per-element operation, and Pandas handles missing values and combines results for you."
- Regex-based `.str` methods let you treat "pattern matching across an entire column" as a single declarative operation rather than an explicit loop with `re.search`/`re.match` calls.

## Anti-patterns
- **Using a Python list comprehension with manual `None`-checking to apply a string method across a Series**: verbose and error-prone (`[s if s is None else s.capitalize() for s in data]`) — use `series.str.capitalize()` instead, which has missing-value handling built in.
- **Trying to extract full unstructured data (e.g., arbitrary ingredient lists) purely with string methods without accepting some approximation/limits**: real-world messy text often can't be fully normalized this way — the chapter explicitly "cheats" by checking for membership in a known list of ingredients rather than fully parsing free text.

## Code Examples
```python
import pandas as pd
import numpy as np

data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
names = pd.Series(data)
names.str.capitalize()                     # None-safe, vectorized

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                    'Eric Idle', 'Terry Jones', 'Michael Palin'])
monte.str.lower()           # Series of strings
monte.str.len()               # Series of ints
monte.str.startswith('T')      # Series of bools
monte.str.split()               # Series of lists

monte.str.extract('([A-Za-z]+)', expand=False)   # regex group extraction
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')         # regex findall per element

monte.str[0:3]                       # vectorized slice
monte.str.split().str[-1]              # extract last name via split+index

full_monte = pd.DataFrame({'name': monte,
    'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']})
full_monte['info'].str.get_dummies('|')   # one-hot indicator columns

# Recipe database cleaning example
recipes = pd.read_json('data/recipeitems.json', lines=True)
recipes.ingredients.str.len().describe()
recipes.name[np.argmax(recipes.ingredients.str.len())]
recipes.description.str.contains('[Bb]reakfast').sum()
recipes.ingredients.str.contains('[Cc]innamon').sum()
recipes.ingredients.str.contains('[Cc]inamon').sum()   # misspelling check

import re
spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
spice_df = pd.DataFrame({
    spice: recipes.ingredients.str.contains(spice, re.IGNORECASE)
    for spice in spice_list})
selection = spice_df.query('parsley & paprika & tarragon')
recipes.name[selection.index]
```
- **What it demonstrates**: missing-value-safe vectorized string ops, the full breadth of `.str` method categories (plain, regex, indexing/slicing, get_dummies), and an end-to-end real-world data-cleaning/recommendation workflow on a messy 175,000-row recipe dataset.

## Reference Tables
| Category | Example methods |
|---|---|
| Mirrors Python str methods | `lower`, `upper`, `len`, `strip`, `split`, `capitalize`, `startswith`, `endswith`, `isalpha`, etc. |
| Regex-based (`re` module conventions) | `match`, `extract`, `findall`, `replace`, `contains`, `count`, `split`/`rsplit` (regex) |
| Vectorized access | `get`, `slice`, `slice_replace`, `cat`, `repeat`, `normalize`, `pad`, `wrap`, `join`, `get_dummies` |

## Worked Example
Building a simple recipe recommender: construct a Boolean DataFrame `spice_df` where each column is `recipes.ingredients.str.contains(spice, re.IGNORECASE)` for a fixed spice list; then `spice_df.query('parsley & paprika & tarragon')` narrows 173,278 recipes down to just 10 matching all three ingredients, and `recipes.name[selection.index]` recovers their names — demonstrating how vectorized string contains-checks plus boolean querying turn unstructured text into an actionable filter.

## Key Takeaways
1. The `.str` accessor vectorizes string operations across a Series while automatically and safely handling missing values.
2. Most Python string methods are mirrored directly under `.str`; regex-based methods (`match`, `extract`, `findall`, `replace`, `contains`, `count`) follow `re` module semantics.
3. Vectorized indexing/slicing (`.str[i]`, `.str[a:b]`) works like array indexing, and chains naturally after `.str.split()`.
4. `get_dummies(sep)` turns a delimiter-separated coded-indicator column into one-hot Boolean columns.
5. Real-world text cleaning (e.g., a 175K-row messy recipe database) often requires combining `.str.contains` membership checks with `.query()` rather than attempting full structured parsing — pragmatic approximation over perfect extraction.

## Connects To
- **Ch 6**: `.str` vectorization is explicitly framed as the string-data analog of NumPy array vectorization.
- **Ch 16**: missing-value-safe string handling builds on Chapter 16's None/NaN conventions.
- **Ch 24**: `.query()` used here for Boolean filtering previews the dedicated eval/query performance chapter.
