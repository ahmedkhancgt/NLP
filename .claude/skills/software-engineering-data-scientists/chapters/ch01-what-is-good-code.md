# Chapter 1: What Is Good Code?

## Core Idea
Good code is code that adapts easily to changing requirements; Nelson frames "good" along five dimensions — simplicity, modularity, readability, performance, and robustness — rather than any single metric like speed or brevity.

## Frameworks Introduced
- **The Five Pillars of Good Code**: simplicity, modularity, readability, performance, robustness.
  - When to use: as a checklist/lens whenever evaluating or writing data science code meant to be reused, maintained, or put into production.
  - How: ask of any piece of code — is it free of repetition and unneeded complexity (simplicity)? is it broken into well-defined functions/components (modularity)? does it follow naming/formatting/documentation conventions (readability)? does it run fast enough and use resources reasonably (performance)? does it fail gracefully and reproducibly (robustness)?
- **DRY (Don't Repeat Yourself)**: all knowledge should have one single representation in code.
  - When to use: whenever you notice near-duplicate code blocks, duplicated logic across projects, or documentation that restates what code already says.
  - How: extract repeated code into a function/loop; watch for cross-project duplication (make processing code flexible/parameterized) and documentation duplication (don't comment what the code already shows — add new knowledge instead).

## Key Concepts
- **Technical debt (tech debt)**: deferred work caused by writing code quickly instead of correctly (missing docs, poor structure, bad names); often a necessary trade-off against deadlines but accrues "interest" as future bug-fixing time.
- **Bit-rot**: the tendency of unused code to require updates later because its dependencies (libraries, OS, language versions) changed; mitigated by well-structured, well-documented code.
- **Complexity (accidental vs. essential)**: essential complexity is inherent to the problem (e.g., many model/feature combinations to try); accidental complexity is complexity introduced by poor structure, e.g., not knowing which function to change — this is what good code reduces. Ousterhout's definition: "Complexity is anything related to the structure of a system that makes it hard to understand and modify a system."
- **Modularity**: breaking a large system into smaller, independent, self-contained components with well-defined inputs/outputs, improving readability, debuggability, reusability, and testability.
- **Broken Window Theory**: visible untidy/low-quality code signals that poor quality is acceptable, lowering the bar for future contributors; cleaning up (removing commented-out code, stray prints) maintains high standards.
- **Refactoring**: changing code's internal structure without changing its external behavior; relies on tests to confirm behavior is preserved.
- **PEP8**: the primary Python style guide (est. 2001) specifying formatting conventions down to spacing details; enforced via linters (Flake8, Pylint) and auto-formatters (Black).
- **Robustness**: code should be reproducible, handle unexpected inputs gracefully (explicit choice between crash / handle / fail silently), and be covered by error handling, logging, and tests.
- **Unit / integration / end-to-end tests**: unit tests check a single function, integration tests check a chunk of interacting code, end-to-end tests check the whole project.

## Mental Models
- **Code as craft**: like a carpenter who takes pride in doors that close smoothly, engineers find intrinsic value in elegant, well-fitted code — not about endless polishing, but about caring at each small decision point.
- **Bridge-building vs. software**: unlike a bridge (design fixed before construction), software requirements are expected to change continuously; good code is built to be adaptable, not just correct-at-a-point-in-time.
- **Growing codebase → growing stakes**: the value of "good code" scales with project size/lifetime/number of contributors; a one-off script needs little investment, but almost everything ends up reused, so revisiting and tidying "one-off" code is usually worth it.

## Anti-patterns
- **Verbose/duplicated code**: unnecessary temporary variables or hand-rolled logic that duplicates built-ins increases bug surface and reading time (but don't over-compress at the cost of readability).
- **Non-descriptive names**: single-letter or generic names (`p`, `x`, `f`) increase cognitive load even though the code runs correctly.
- **Leaving debugging artifacts**: commented-out code and stray `print()` debugging calls left in "finished" code signal low standards and get copied elsewhere (Broken Window Theory).
- **Letting documentation go stale**: outdated documentation is worse than none — it actively misleads and wastes time resolving the confusion.
- **Treating readability as an afterthought**: focusing only on functionality while delaying naming/formatting/docs produces code that is more complex and harder to maintain than if readability were addressed while writing.

## Code Examples
```python
# DRY violation: repeated CSV-processing logic
import pandas as pd

df = pd.read_csv("sdg_literacy_rate.csv")
df = df.drop(["Series Name", "Series Code", "Country Code"], axis=1)
df = df.set_index("Country Name").transpose()

df2 = pd.read_csv("sdg_electricity_data.csv")
df2 = df2.drop(["Series Name", "Series Code", "Country Code"], axis=1)
df2 = df2.set_index("Country Name").transpose()
```
- **What it demonstrates**: the "bad" version — identical logic repeated per file, multiplying maintenance points.

```python
# DRY fix: extract into a function
def process_sdg_data(csv_file, columns_to_drop):
    df = pd.read_csv(csv_file)
    df = df.drop(columns_to_drop, axis=1)
    df = df.set_index("Country Name").transpose()
    return df
```
- **What it demonstrates**: one single representation of the processing logic, callable per file/column-set.

```python
# Verbose vs. concise
i = float(i)
image_vector.append(i / 255.0)
# simplified:
image_vector.append(float(i) / 255)
```
- **What it demonstrates**: removing an unnecessary temporary variable while staying readable.

```python
# Modular skeleton using `pass` as placeholder
def load_data(csv_file):
    pass

def clean_data(input_data, max_length):
    pass

def plot_data(clean_data, x_axis_limit, line_width):
    pass
```
- **What it demonstrates**: sketching a system's component boundaries and I/O contracts before implementation, using `pass` so calls don't error prematurely.

```python
# Naming: poor vs. good
import pandas as p
x = p.read_csv(f, index_col=0)
# vs.
import pandas as pd
df = pd.read_csv(input_file, index_col=0)
```
- **What it demonstrates**: identical behavior, very different cognitive load from naming choices.

## Reference Tables
| Pillar | What it asks | Primary tools/chapters |
|---|---|---|
| Simplicity | No repetition, no unneeded complexity/lines | DRY, concise code |
| Modularity | Broken into well-defined functions/components | Ch 8 (Design & Refactoring) |
| Readability | Standards, names, cleanup, docs | Ch 6 (Formatting/Linting), Ch 9 (Docs) |
| Performance | Runtime & memory efficiency | Ch 2, Ch 3 |
| Robustness | Reproducible, graceful error handling, tested | Ch 5 (Errors/Logging), Ch 7 (Testing) |

## Worked Example
A data scientist builds an NLP pipeline that truncates customer reviews to 512 characters before training a model. The truncation step lives only in the training notebook. When the model is deployed, the preprocessing step is forgotten in the inference code, so production throws errors on inputs longer than 512 characters. This illustrates accidental complexity: the system became hard to reason about not because the ML problem is inherently complex, but because a piece of logic (the truncation step) wasn't modularized into a shared, reusable component used by both training and inference.

## Key Takeaways
1. Good code adapts easily to changing requirements — this matters more as a codebase grows and is passed between people over time.
2. Simplicity (via DRY and avoiding verbosity) is foundational; repetition multiplies maintenance burden and bug risk.
3. Modularity (small, well-defined, independent components) makes code easier to read, debug, reuse, and test.
4. Readability — standards (PEP8), good names, cleanup, and documentation — determines how easily others (including future-you) can work with the code.
5. Performance and robustness (graceful error handling, logging, testing) round out "good code," especially for code that will run repeatedly or at scale.
6. Tech debt and bit-rot are the costs of skipping good-code practices; they're sometimes a reasonable trade-off but accrue real future cost.

## Connects To
- **Ch 2**: Performance pillar — timing, profiling, Big O.
- **Ch 3**: Performance pillar — choosing data structures.
- **Ch 5**: Robustness pillar — errors and logging in depth.
- **Ch 6**: Readability pillar — formatting, linting, type checking (PEP8 tooling).
- **Ch 7**: Robustness pillar — testing strategies in depth.
- **Ch 8**: Modularity pillar — design and refactoring in depth.
- **Ch 9**: Readability pillar — documentation in depth.
- **External**: *Clean Code* (Robert C. Martin), SOLID principles, *The Pragmatic Programmer* (Thomas & Hunt), *A Philosophy of Software Design* (Ousterhout) — all cited as further reading on good-code principles.
