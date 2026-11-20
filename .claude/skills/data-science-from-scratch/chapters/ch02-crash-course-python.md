# Chapter 2: A Crash Course in Python

## Core Idea
A targeted Python primer covering only the language features the book leans on heavily: whitespace blocks, functions, core data structures (list/tuple/dict/defaultdict/Counter/set), control flow, comprehensions, `assert`-based testing, OOP basics, generators, randomness, regex, and type annotations — the last of which the book commits to using throughout.

## Frameworks Introduced
- **Type annotations as documentation**: annotate function signatures (`def dot_product(x: Vector, y: Vector) -> float`) even though Python doesn't enforce them at runtime.
  - When to use: every function from here on in the book.
  - How: use `typing` module generics (`List`, `Dict`, `Tuple`, `Optional`, `Callable`, `Iterable`); check with `mypy` as an external tool, since annotations don't raise errors on their own.
- **assert-based testing**: use `assert` statements as the book's lightweight substitute for a test framework.
  - When to use: validating that functions behave as expected, and documenting function preconditions.
  - How: `assert <condition>, "message"`; the book uses this pervasively instead of unittest/pytest.
- **defaultdict / Counter pattern**: use `collections.defaultdict(int)` or `Counter` to avoid manual "check-then-initialize" key handling.
  - When to use: any "count/collect by key" aggregation (word counts, grouping).

## Key Concepts
- **Pythonic**: idiomatic code following Python's "one obvious way to do it" philosophy (e.g., `"user" in tweet` over `"user" in tweet.keys()`).
- **List comprehension**: `[expr for x in iterable if cond]` — the book's default tool for transforming/filtering lists.
- **Generator**: a lazily-evaluated iterable created via `yield` or a parenthesized comprehension; can only be iterated once.
- **defaultdict**: a dict subclass that auto-initializes missing keys via a zero-argument factory function.
- **Counter**: a defaultdict(int)-like structure for tallying a sequence, with a `.most_common(n)` method.
- **Truthiness**: falsy values in Python are `False, None, [], {}, "", set(), 0, 0.0`; everything else is truthy.
- **args/kwargs**: `*args` captures extra positional args as a tuple, `**kwargs` captures extra keyword args as a dict; used for building argument-agnostic higher-order functions.
- **f-string**: Python 3.6+ string interpolation, `f"{first_name} {last_name}"`, preferred throughout the book over `+` concatenation or `.format()`.

## Mental Models
- **"Forgiveness over permission"**: prefer `try/except KeyError` or `dict.get(key, default)` over checking `if key in dict` first, though `defaultdict` is cleanest of all.
- **Laziness as a resource strategy**: use generators instead of lists when you don't need random access or multiple passes, especially over huge or infinite sequences (`natural_numbers()`).
- **Explicit types force cleaner interfaces**: having to write a type signature (vs. an untyped `Union[str,int,float,bool]` parameter) exposes fragile, overloaded function designs before they ship.

## Anti-patterns
- **Scanning a list for membership repeatedly**: `in` on a `list` is O(n); convert to a `set` first if doing many membership checks.
- **Using mutable defaults or lambdas assigned to variables**: book explicitly prefers `def` over `lambda` assignment (`another_double = lambda x: 2*x  # don't do this`).
- **`from module import *`**: can silently overwrite existing names in your namespace (shown with `re.match`).
- **Iterating a generator twice**: generators are single-use; re-create or materialize to a list if you need multiple passes.
- **args/kwargs overuse**: makes function signatures opaque; the book uses them only when no explicit signature will work (e.g., generic decorators like `doubler_correct`).

## Code Examples
```python
from collections import defaultdict

word_counts = defaultdict(int)  # int() produces 0
for word in document:
    word_counts[word] += 1
```
- **What it demonstrates**: from-scratch word counting without manual key-existence checks.

```python
def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        return 2 * f(*args, **kwargs)
    return g
```
- **What it demonstrates**: `*args, **kwargs` used to build a generic higher-order function wrapper.

```python
def generate_range(n):
    i = 0
    while i < n:
        yield i
        i += 1
```
- **What it demonstrates**: a hand-rolled lazy generator (mirroring what `range` does internally).

```python
from typing import List, Optional

values: List[int] = []
best_so_far: Optional[float] = None
def total(xs: List[float]) -> float:
    return sum(xs)
```
- **What it demonstrates**: type-annotation style used for every function in the rest of the book.

## Reference Tables
| Structure | Mutable | Ordered | Use case |
|---|---|---|---|
| `list` | yes | yes | general sequential collection |
| `tuple` | no | yes | fixed-size records, multi-return values, dict keys |
| `dict` | yes | insertion-order (3.7+) | key -> value lookup |
| `set` | yes | no | fast membership test, distinct elements |
| `defaultdict` | yes | yes | dict with auto-initialized missing keys |
| `Counter` | yes | yes | tallying occurrences, `.most_common()` |

| Falsy values | `False`, `None`, `[]`, `{}`, `""`, `set()`, `0`, `0.0` |

## Worked Example
Building a `word_counts` dict three clumsy ways (manual `if/else`, `try/except KeyError`, `dict.get`) versus the clean `defaultdict(int)` one-liner — all on a hypothetical `document` list of words — motivates every data structure introduced in the chapter. The `CountingClicker` class (constructor, `click`, `read`, `reset`, `__repr__`) and its `NoResetClicker` subclass illustrate OOP, dunder methods, and inheritance with `assert`-based tests.

## Key Takeaways
1. Use `defaultdict`/`Counter` over manual dict-key bookkeeping for counting/grouping tasks.
2. Prefer list comprehensions and generator expressions to explicit loops for transforming/filtering data.
3. `assert` statements are the book's substitute for a formal test suite — use them liberally on both outputs and input preconditions.
4. Type annotations (from here through the rest of the book) serve as documentation and editor/mypy tooling, not runtime enforcement.
5. Use generators for large or infinite sequences you only need to traverse once; use lists when you need repeated/random access.
6. `*args, **kwargs` is a last resort for argument-agnostic functions — prefer explicit signatures.

## Connects To
- **Ch1**: the `friendships`, `Counter`, `defaultdict` idioms shown here are exactly what Ch1's DataSciencester examples already used.
- **Every later chapter**: all subsequent from-scratch code uses these exact idioms (type hints, comprehensions, `assert` tests, `Counter`/`defaultdict`).
- **mypy**: external static type checker the author runs behind the scenes to validate the book's own code, even though the book's narrative never shows it running.
