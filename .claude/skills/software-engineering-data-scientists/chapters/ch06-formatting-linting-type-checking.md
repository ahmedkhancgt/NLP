# Chapter 6: Code Formatting, Linting, and Type Checking

## Core Idea
Formatting, linting, and type checking are tedious-but-important aspects of readable, robust code — and the key message is to automate all of them with tools rather than doing them manually, freeing your attention for the code's actual functionality.

## Frameworks Introduced
- **Automate-don't-manually-police framework**: for any mechanical code-quality concern (spacing, import order, style conformance, type mismatches), use a tool rather than reviewing by eye.
  - When to use: always, for any script-based (non-notebook) Python project.
  - How: formatter (Black) for appearance, linter (Flake8/Pylint/Ruff) for correctness/style errors, import sorter (isort) for import order, type checker (mypy/Pyright/Pyre) for type mismatches — run them standalone, in the IDE, or (per Ch 12) automatically via pre-commit hooks.
- **Separation of concerns: formatting vs. linting vs. type checking**: three distinct categories of tool, each catching a different class of problem.
  - When to use: understand which tool to reach for based on the symptom — ugly/inconsistent appearance (formatter), a runtime error waiting to happen like an undefined name (linter), a type mismatch (type checker).
  - How: run Black/Ruff-format first (fixes style), then a linter (flags remaining logical/style issues it won't auto-fix), then mypy (flags type mismatches) — each pass catches what the previous one doesn't.

## Key Concepts
- **PEP8**: the foundational Python style guide (Van Rossum, Warsaw, Coghlan, 2001); its core principle — "A style guide is about consistency... within a project is more important... within one module or function is the most important" — defines whitespace, blank-line, naming, and comment conventions.
- **Code formatting**: changes only appearance (line breaks, whitespace, blank lines) — never changes behavior; done by Black (opinionated, minimal config, ~90-char lines) or Ruff's format command.
- **isort**: a tool that sorts imports into PEP8's required groups — (1) standard library, (2) third-party, (3) local — alphabetizing within each group.
- **Black**: an opinionated auto-formatter using a PEP8 subset with its own defaults (e.g., double quotes, ~90-char line wraps); refuses to run on code with syntax errors; supports `# fmt: skip` / `# fmt: off` / `# fmt: on` to exempt specific lines/blocks.
- **Linting**: static analysis that flags code likely to error at runtime or violate style, without executing it — named after the lint trap in a clothes dryer (removes "fuzz"); originated for C in 1978.
- **Flake8 / Pylint / Ruff**: common Python linters. Pylint and Flake8 only *flag* issues (don't auto-fix); Ruff both lints and formats (like Black) and is designed for speed. Each uses a slightly different rule set/style guide, so teams should standardize on one.
- **IDE-integrated linting**: IDE extensions (e.g., VS Code's Pylance) can underline errors live while typing, often wrapping Pylint/Flake8/other linters under the hood, configurable to ignore irrelevant warnings.
- **Type**: the category of a Python object (int, str, float, etc.); Python is dynamically typed, meaning a variable's type can change during execution (unlike statically typed languages such as Java).
- **Type annotation / type hint**: optional syntax (`var: type`, `def f(x: type) -> return_type:`) introduced in Python 3.5 (PEP 484) to document expected/returned types; improves readability and documents the function's contract but has zero effect on runtime behavior — an incorrect annotation won't raise an error on its own.
- **Type checker (mypy, Pyright, Pyre)**: static analysis tools that read type annotations and flag mismatches before runtime, serving a similar debugging purpose to writing type-check tests but faster/easier.

## Mental Models
- **"The details of code formatting are boring — don't spend human time there"**: the central theme of the chapter; cognitive and review effort should go to logic/functionality, with aesthetics fully delegated to tooling.
- **Consistency removes a category of team friction**: when formatting is standardized, code reviews can focus entirely on functionality rather than debating whitespace/line-break choices — consistency is itself a collaboration tool, not just an aesthetic one.
- **Tabs-vs-spaces as an allegory**: the long, now largely moot "tabs vs. spaces" debate (resolved by data — Felipe Hoffa's 2016 analysis of 1 billion files found spaces dominant — and by IDEs auto-converting tab keypresses to spaces) illustrates how automation can end a debate that pure preference never would.
- **Type annotations as documentation that can be verified**: an annotation is a readability aid (what Ch 9 would call documentation) that becomes actionable only once paired with a type checker — otherwise it's just a comment that happens to look structured and can silently be wrong.
- **Catch errors at write-time, not run-time**: linting/type checking exist to shift bug discovery earlier — from "crashes after 20 minutes of runtime" to "flagged instantly in the editor."

## Anti-patterns
- **Manually enforcing style in code review**: wastes reviewer attention on things a formatter should have already fixed; use Black/Ruff before requesting review.
- **Writing an incorrect type annotation and assuming it's caught**: annotations alone do nothing at runtime (Example 6-3 shows a `float` annotation on a parameter that's actually a `list`, and the code still runs); you must run a type checker (e.g., mypy) to get value from annotations.
- **Mixing linters/formatters across a team without agreement**: because Flake8, Pylint, and Ruff enforce slightly different rules, inconsistent tool choice across contributors reintroduces the inconsistency formatting was meant to eliminate.
- **Running tools only after code is "done"**: configuring IDE-integrated linting/formatting (live underlines, format-on-save) catches issues as you type rather than in a separate late pass.
- **Treating linter/formatter setup as a one-off manual chore instead of automating it in CI/pre-commit** (foreshadowing Ch 12): setup cost is worth paying once, then automating, rather than repeating by hand every time.

## Code Examples
```python
# PEP8: line break after an if statement
# Correct:
if foo == 'blah':
    do_blah_thing()

# Wrong (runs, but less readable):
if foo == 'blah': do_blah_thing()
```
- **What it demonstrates**: PEP8 enforces readability conventions that don't affect correctness but aid consistency.

```bash
# isort: reorders imports into stdlib / third-party / local groups
$ pip install isort
$ isort my_script.py
```
```python
# After isort:
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
```
- **What it demonstrates**: automatic grouping/alphabetizing of imports per PEP8's three-tier ordering.

```bash
# Black: auto-formats a script; refuses to run with syntax errors
$ black plot_big_o.py
# error: cannot format plot_big_o.py: Cannot parse: 3:25: def plot_big_o(save_path)
# (missing trailing colon must be fixed first)

# Preview changes without applying:
$ black plot_big_o.py --diff
```
- **What it demonstrates**: Black requires syntactically valid code, then normalizes quotes, spacing, and blank lines automatically.

```python
# Opt a line or block out of Black's formatting
x = [1,2,3]  # fmt: skip

# fmt: off
y = {  'a':1,'b':2 }
# fmt: on
```
- **What it demonstrates**: targeted exemptions when auto-formatting would hurt readability for specific hand-tuned code.

```bash
# Linters: Pylint and Flake8 stop at syntax errors, then report style/logic issues
$ pylint plot_big_o.py
# plot_big_o.py:5:8: E0602: Undefined variable 'np' (undefined-variable)
# plot_big_o.py:13:4: C0200: Consider using enumerate instead of iterating with range and len

$ flake8 plot_big_o.py
# plot_big_o.py:5:9: F821 undefined name 'np'
# plot_big_o.py:6:29: E231 missing whitespace after ','

# Ruff: fast linter + separate formatter
$ ruff check plot_big_o.py
$ ruff format plot_big_o.py
```
- **What it demonstrates**: different linters catch overlapping but not identical issues (e.g., missing `import numpy as np` surfaces as `E0602`/`F821` in both tools), and each uses its own rule naming scheme.

```python
# Type annotations: correct vs. incorrect
from collections import Counter
from typing import List

def mode_using_counter(list_of_numbers: List[float]) -> float:
    c = Counter(list_of_numbers)
    return c.most_common(1)[0][0]

# NumPy array annotation
import numpy as np
def array_operation(input_array: np.ndarray) -> np.ndarray:
    ...

# Incorrect annotation that still runs without error:
def mode_using_counter(list_of_numbers: float) -> float:  # should be List[float]
    c = Counter(list_of_numbers)
    return c.most_common(1)[0][0]
```
- **What it demonstrates**: annotations document intended types but do not enforce them at runtime — enforcement requires a type checker.

```bash
# mypy: catches the mismatched annotation
$ mypy mode_using_counter_incorrect.py
# error: No overload variant of "Counter" matches argument type "float" [call-overload]
# Found 1 error in 1 file (checked 1 source file)
```
- **What it demonstrates**: mypy statically verifies that annotated types are used consistently, catching the float/list mismatch before runtime.

## Reference Tables
| Tool category | Tools | Changes code? | What it catches |
|---|---|---|---|
| Formatter | Black, Ruff format | Yes (auto-fixes) | Appearance (whitespace, quotes, line breaks) |
| Import sorter | isort, reorder-python-imports | Yes (auto-fixes) | Import grouping/order |
| Linter | Pylint, Flake8, Ruff check | No (Pylint/Flake8 only flag; Ruff can fix) | Style violations, undefined names, logic smells |
| Type checker | mypy, Pyright, Pyre | No (only flags) | Type mismatches vs. annotations |

| Import group order (PEP8) |
|---|
| 1. Standard library imports |
| 2. Related third-party imports |
| 3. Local application/library imports |

## Worked Example
A script (`plot_big_o.py`, recreating the Big-O plot from Chapter 2) has a syntax error (missing colon on `def plot_big_o(save_path)`), a missing `import numpy as np`, and PEP8 violations (extra spaces, single quotes). Running Black first fails outright ("cannot format... Cannot parse") because of the syntax error — formatters require valid syntax. After adding the colon, Black cleans all whitespace/quote issues but can't detect the missing import (that's a *logic* problem, not a *formatting* one). Pylint and Flake8 both catch the syntax error first (refusing full analysis until fixed), then after the fix, both flag `undefined-variable`/`F821` for `np` along with their own style complaints (`C0200` suggesting `enumerate`; `E231` missing whitespace after commas) — demonstrating that formatting, linting, and type checking are three non-overlapping passes, each needed to fully clean the file.

## Key Takeaways
1. Formatting changes only appearance and never behavior; use PEP8 (or a company style guide) as the standard, applied automatically via Black or Ruff.
2. isort (or Ruff) automatically fixes import order/grouping per PEP8's three-tier rule — don't do this by hand.
3. Linters (Pylint, Flake8, Ruff) catch potential runtime errors (undefined names, misspelled variables) and style deviations before you run the code; pick one tool per team for consistency, since their rule sets differ.
4. Type annotations document expected input/output types and improve readability, but have zero runtime effect by themselves — they're only enforced when paired with a type checker like mypy.
5. mypy (or Pyright/Pyre) statically verifies type-annotation consistency, catching mismatches faster/easier than writing dedicated tests for the same purpose.
6. The overarching lesson: use tools for all of this — formatting, linting, and type checking are not valuable uses of manual review time; invest setup time once and automate (see Ch 12 for pre-commit automation).
7. These tools generally target standalone scripts, not Jupyter notebooks (though Black and some IDE linters have notebook support).

## Connects To
- **Ch 1**: Readability pillar — formatting/linting/naming were introduced there as part of "good code."
- **Ch 7**: Type checking is framed as a faster alternative to writing tests purely to catch type mismatches.
- **Ch 9**: Naming conventions and comments (also governed partly by PEP8) covered in depth.
- **Ch 12**: Automation/deployment — shows how to run these tools automatically via pre-commit hooks, including auto-removing leftover `breakpoint()` calls (from Ch 5).
- **External**: PEP8, PEP 484 (type hints); Google's Python Style Guide; Black, Ruff, Flake8, Pylint, isort, mypy, Pyright, Pyre documentation.
