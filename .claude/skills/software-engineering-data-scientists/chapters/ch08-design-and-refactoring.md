# Chapter 8: Design and Refactoring

## Core Idea
Good design — at both the project-structure level and the individual-function level — reduces cognitive load, enables reuse, and makes code easier to change; refactoring is the normal, test-dependent process of restructuring code without changing its external behavior as requirements evolve.

## Frameworks Introduced
- **Project Design Questions**: a checklist to turn a project plan into a well-structured codebase.
  - When to use: at project start, or once you have enough clarity on direction after initial exploration.
  - How: ask — what's the overall goal? what are the inputs/outputs? who are the stakeholders and how often do you check in? what's the timeline/milestones? how much is exploration vs. solving? how likely is scope change or future reuse? Document the answers; also retrospect on past projects for recurring lessons.
- **Function Design Framework** (adapted from Rachel Tatman): name → inputs → behavior → outputs.
  - When to use: before writing the body of any function, especially when converting notebook code into modular functions.
  - How: choose a descriptive name first (sets intent), decide the function's input arguments (interface) before writing logic, write the single-purpose behavior in the body, decide what's returned/output (also part of the interface) — and keep interface decisions stable afterward.
- **Interfaces as Contracts**: treat a function's accepted inputs and returned outputs as a stable agreement with the rest of the system.
  - When to use: whenever designing or refactoring a function that others — or other parts of the system — depend on.
  - How: keep argument count small (≤3–4; use a config file/object for more), avoid changing input/output shapes once set, enforce with tests + docstrings + type annotations/type checking so violations are caught early.
- **Coupling minimization**: judge code quality by how much changing one piece forces changes elsewhere.
  - When to use: when deciding how to split code into functions/modules, or diagnosing why changes keep cascading.
  - How: avoid hardcoding one function's calls/literals inside another (tight coupling); instead, pass data explicitly between independently callable functions (loose coupling) so each can be modified, tested, and reused on its own.
- **Refactor-Test-Save workflow**: the safe loop for restructuring code.
  - When to use: anytime you want to improve code structure without changing behavior.
  - How: (1) confirm existing tests pass, (2) make one small change (e.g., rename a variable via IDE refactor tool), (3) rerun tests, (4) commit/save; repeat incrementally; larger changes to the function body follow the same loop as long as the interface stays fixed — if the interface/behavior changes, add/update integration tests too.

## Key Concepts
- **Code design ("factoring")**: arranging code into functions, classes, and modules — described by Sandi Metz as "the art of arranging code," with no single correct answer.
- **Modular code**: code broken into small, independent pieces each with one well-defined purpose; the opposite of one giant script/notebook. Benefits: easier reuse, easier to relocate when requirements change, composability (combining pieces into larger systems), easier testing.
- **"And" test for single responsibility**: if describing what a function does requires the word "and" ("cleans the data *and* creates a visualization"), it should likely be split into two functions.
- **Coupling**: the degree of interdependence between functions/modules. Tight coupling = changing one requires extensively changing the other (e.g., a function hardcoding a call to another with fixed arguments); loose coupling = changes to one require little or no change to the other. Lower coupling is generally better.
- **Contract**: the fixed, documented input/output interface of a function, which other code depends on; changing it can break dependents, so contracts should be decided deliberately and kept stable, enforced via tests, docstrings, and type annotations.
- **Refactoring** (Martin Fowler): "the process of changing a software system in a way that does not alter the external behavior of the code, yet improves its internal structure." Done only when there's a specific reason (new functionality needed, readability/maintainability improvement) — not refactoring just for its own sake if code already meets requirements.
- **Notebook-to-script transition**: moving code from Jupyter notebooks (great for exploration/prototyping, inline visualization, instant feedback) to standalone Python scripts (better for repeatable runs, linting/formatting/type-checking, dependency management, debugging, version control, security) once a project matures past prototyping into repeated/production use.
- **Cookiecutter**: a tool to automate generating a standardized project file/folder structure (including packaging/dependency files) from a reusable template.
- **Kedro**: an open-source library for standardizing data science projects as a pipeline of connected "nodes" (each a script/step), runnable end to end.
- **SOLID principles / design patterns**: broader software-engineering design theory referenced as further reading (Real Python article on SOLID; Refactoring Guru's Python design patterns guide), noted as needing adaptation for data science use.

## Mental Models
- **Standardization reduces cognitive load across projects**: if every project in your team follows a similar folder/file layout, switching between projects (or picking up someone else's work) costs much less mental effort.
- **"One file per project step"**: thinking of a data science/ML project as a sequence of discrete steps (load → clean/feature-engineer → train → evaluate), each often ending in a save-to-file, is a natural way to decide how to split scripts.
- **Interfaces as the load-bearing walls of a system**: you can freely renovate the inside of a function (its implementation) as long as you don't move the walls (its interface) — this is exactly what safe refactoring exploits.
- **Tests convert "risky change" into "safe experiment"**: because tests verify behavior is preserved, refactoring becomes a disciplined "change → verify → save" loop instead of a leap of faith; without tests, refactoring is just guessing.
- **Notebooks vs. scripts as a maturity spectrum**: notebooks are for *divergent* exploration (trying many things, non-linear execution order), scripts are for *convergent*, repeatable execution once you know what you're doing — the switch point is when you start running "the same thing" repeatedly.
- **The display-cells-become-assertions trick**: when converting a notebook to a function, the ad-hoc `df.head()`/print cells you used to sanity-check your data are exactly the candidates to formalize into test assertions (e.g., "I always checked the row/column count here" → `assert len(df) == 24`).

## Anti-patterns
- **One giant Jupyter notebook as the entire project**: common in practice, but limits testing, linting, version control, dependency tracking, and reuse as the project scales — acknowledged as something even the author has done.
- **Tight coupling between functions**: e.g., `fit_trendline()` internally hardcoding a call to `process_sdg_data("SG_GEN_PARL.xlsx", [...])` — changing the data-processing function's signature or behavior silently breaks the model-fitting function, and the two can't be tested, reused, or swapped independently.
- **Replicating centrally stored data locally in a project**: risks data-version mismatches and security/privacy exposure; better to include the data-access code (or at least documentation to reproduce) rather than copying data files into the repo.
- **Refactoring without tests**: makes it impossible to know whether a structural change silently broke behavior — tests must exist (or be written) before refactoring.
- **Rewriting everything at once during refactoring**: increases risk of introducing hard-to-isolate bugs; prefer small, incremental, test-verified changes with version-control checkpoints.
- **Changing a function's interface casually during refactoring**: breaks the implicit contract with dependent code; if the interface must change, escalate to integration tests to catch ripple effects.
- **Uploading a notebook containing data to version control**: a specifically called-out security/privacy risk of the notebook workflow.

## Code Examples
```python
# Tightly coupled functions (BAD): fit_trendline hardcodes process_sdg_data's call
def process_sdg_data(input_excel_file, columns_to_drop):
    df = pd.read_excel(input_excel_file)
    df = df.drop(columns_to_drop, axis=1)
    df = df.set_index("GeoAreaName").transpose()
    return df

def fit_trendline(country_name):
    df = process_sdg_data(
        "SG_GEN_PARL.xlsx",
        ["Goal", "Target", "Indicator", "SeriesCode", "SeriesDescription",
         "GeoAreaCode", "Reporting Type", "Sex", "Units"],
    )
    timestamps = [int(i) for i in df.index.tolist()]
    data = df[country_name].tolist()
    result = linregress(timestamps, data)
    slope = round(result.slope, 3)
    r_squared = round(result.rvalue**2, 3)
    return slope, r_squared
```
- **What it demonstrates**: `fit_trendline` can't be used, tested, or changed independently of `process_sdg_data`'s exact filename/columns — a change to one forces a change to the other.

```python
# Loosely coupled functions (GOOD): each takes plain data, caller wires them together
def fit_trendline(year_timestamps, data):
    result = linregress(year_timestamps, data)
    slope = round(result.slope, 3)
    r_squared = round(result.rvalue**2, 3)
    return slope, r_squared

def process_sdg_data(input_excel_file, columns_to_drop):
    df = pd.read_excel(input_excel_file)
    df = df.drop(columns_to_drop, axis=1)
    df = df.set_index("GeoAreaName").transpose()
    return df

df = process_sdg_data("SG_GEN_PARL.xlsx", [...])
timestamps = [int(i) for i in df.index.tolist()]
country_data = df["India"].tolist()
slope, r_squared = fit_trendline(timestamps, country_data)
```
- **What it demonstrates**: both functions are independently reusable/testable; the calling code — not either function — owns the wiring between them.

```python
# Converting notebook cells into a function + test
def process_sdg_data(input_excel_file, columns_to_drop, output_csv_file):
    df = pd.read_excel(input_excel_file)
    df = df.drop(columns_to_drop, axis=1)
    df = df.set_index("GeoAreaName").transpose()
    df.to_csv(output_csv_file)

import os
import pandas as pd

def test_process_sdg_data():
    test_filepath = "test_sgd_data.csv"
    process_sdg_data("SG_GEN_PARL.xlsx",
                     ["Goal", "Target", "Indicator", "SeriesCode",
                      "SeriesDescription", "GeoAreaCode", "Reporting Type",
                      "Sex", "Units"],
                     test_filepath)
    df = pd.read_csv(test_filepath)
    assert len(df) == 24
    assert len(df.columns) == 196
    os.remove(test_filepath)  # cleanup step
```
- **What it demonstrates**: the ad-hoc "display the DataFrame to eyeball it" step from notebook development becomes a formal row/column-count assertion, with an explicit cleanup step removing the test artifact.

```python
# Refactor-test-save loop: step 1, rename for readability (interface-preserving internal change)
def weighted_mean(num_list, weights_list):   # renamed from `weights`
    if not (num_list or weights_list):
        return None
    running_total = 0
    for i in range(len(num_list)):
        running_total += (num_list[i] * weights_list[i])
    return (running_total / sum(weights_list))

# step 2, replace the implementation entirely, same interface/behavior
import numpy as np

def weighted_mean(num_list, weights_list):
    try:
        return np.average(num_list, weights=weights_list)
    except ZeroDivisionError:
        return None
```
- **What it demonstrates**: a two-stage refactor (cosmetic rename, then a complete internal rewrite using `np.average`) where the existing test suite (unchanged) verifies behavior is preserved at each step.

## Reference Tables
| Project artifact | Purpose |
|---|---|
| `README.md`, `requirements.txt` | Documentation + dependency management (Ch 9, Ch 10) |
| `notebooks/` | Exploration and model prototyping |
| `src/` (with `__init__.py`, `utils.py`) | Modular, reusable project scripts (one file per project step) |
| `tests/` | Mirrors `src/` structure, one test file per source file |

| Coupling level | Symptom | Consequence |
|---|---|---|
| Tight | Function A hardcodes a call into Function B with fixed arguments | Change to B forces change to A; can't test/reuse independently |
| Loose | Functions take data as parameters; caller wires them together | Each function independently testable, reusable, swappable |

| Function Design Framework element | Question it answers |
|---|---|
| Name | What is this function's intent? |
| Inputs | What does the interface accept? (keep ≤3–4 args; else use config) |
| Behavior | What single thing does the body do? |
| Outputs | What does the interface return/produce? (also part of the contract) |

## Worked Example
The chapter walks through converting a notebook that loads `SG_GEN_PARL.xlsx`, drops metadata columns, reindexes by country, and writes a CSV — into the reusable `process_sdg_data(input_excel_file, columns_to_drop, output_csv_file)` function, following the Name→Inputs→Behavior→Outputs framework: name chosen to generalize across similar UN SDG Excel files; inputs chosen to be whatever the author expects to vary (filename, dropped columns, output filename); behavior is the single transformation; outputs become the saved CSV. The notebook's data-inspection cells directly suggest the test's assertions (`assert len(df) == 24`, `assert len(df.columns) == 196`), showing a direct, traceable path from "cells I used to eyeball the data" to "formal test assertions."

## Key Takeaways
1. Deliberately design project structure (folders for notebooks, `src`, `tests`, plus README/requirements) once a project's direction is clear enough — standardization reduces cognitive load and eases handoffs.
2. Use the Name → Inputs → Behavior → Outputs framework when designing any function; decide the interface (inputs/outputs) before writing the implementation.
3. Treat a function's inputs/outputs as a contract: keep argument counts small, avoid changing the interface casually, and enforce it with tests, docstrings, and type annotations.
4. Minimize coupling — functions should take data as explicit arguments rather than hardcoding calls to other functions with fixed parameters — so pieces stay independently testable and reusable.
5. Move from notebooks to scripts once you're repeating the same work, need reproducibility, or are approaching production — notebooks are excellent for exploration but weak on linting, dependency tracking, debugging, and version control.
6. Refactoring (restructuring without changing behavior) should only be done for a specific reason, and only safely with a prior test suite: make a small change, rerun tests, save/commit, repeat.
7. Changing a function's internals while preserving its interface requires no changes elsewhere; changing the interface itself requires integration tests to catch ripple effects.
8. Cookiecutter (templated scaffolding) and Kedro (pipeline-based project framework) can automate repeatedly-used project structures.

## Connects To
- **Ch 1**: Modularity pillar — expanded into concrete design guidance here.
- **Ch 4**: OOP/FP as frameworks for deciding how to break code into functions/classes, revisited here.
- **Ch 6**: Type annotations/type checking as one enforcement mechanism for interface contracts.
- **Ch 7**: Tests are the prerequisite for safe refactoring; notebook-derived assertions shown here.
- **Ch 9**: Documentation (README, naming, docstrings) referenced throughout as a companion practice to good design.
- **Ch 10**: Dependency management/packaging files (`requirements.txt`) and version control (committing after each refactor step).
- **External**: Sandi Metz on code design; Martin Fowler's definition of refactoring; Rachel Tatman's "Six steps to more professional data science code" (Kaggle); *Building Machine Learning Pipelines* by Nelson & Hapke; SOLID principles (Real Python); Refactoring Guru's Python design patterns; Cookiecutter, Kedro, nbconvert, Jupytext, Papermill tools.
