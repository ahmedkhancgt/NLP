# Chapter 9: Documentation

## Core Idea
Documentation — ranging from names to comments, docstrings, READMEs, notebook text, and experiment tracking — is what makes code reproducible and reusable by others (and future-you); it must be audience-appropriate, kept up to date alongside code changes, well structured, and never duplicated across levels.

## Frameworks Introduced
- **Documentation hierarchy (shortest to longest)**: names → comments → docstrings → API documentation → longer documents (READMEs/tutorials).
  - When to use: deciding *where* a given piece of information about your code belongs.
  - How: each piece of information should be documented exactly once, at the appropriate level — function purpose/interface in the docstring (not comments, not README); a single clarifying point in a comment; overall project context in the README; don't duplicate across levels.
- **Data science documentation questions**: beyond general code docs, record the *reasoning* behind data science decisions.
  - When to use: throughout exploratory/experimental work, so decisions are traceable later.
  - How: answer — why this data? what assumptions about the data? why this method over alternatives? when does the method fail? what shortcuts were taken? what are good next steps? what lessons were learned?
- **Naming heuristics**: expressive, appropriately-sized, convention-following names as the first and cheapest form of documentation.
  - When to use: every variable, function, class, file, module, and notebook name.
  - How: favor full words over abbreviations (`image_id` not `im_id`); include units (`distance_km`); use `snake_case` for variables/functions, `CamelCase` for classes, `ALL_CAPS` for constants; use recognized short forms only for well-known conventions (`i`/`j` counters, `df`, `fig`/`ax`); avoid shadowing Python built-ins (`list`, `str`); prefer specific over generic names (`data_preprocessing_utils.py` not `utils.py`); start function names with a verb (`make_`, `load_`, `get_`).

## Key Concepts
- **Audience-first documentation**: before writing, decide who will read it (a future teammate taking over the project, another team reusing a function, or future-you) and calibrate detail/language accordingly.
- **Comment**: a short, single-point annotation (`# ...`) that adds information *not already visible in the code* — a summary, caveat, TODO, external link, or note that something is a temporary fix; should never restate what the code obviously does (violates DRY from Ch 1).
- **Docstring**: a formal, longer block (triple-quoted `"""..."""`) at the start of a function/class/module describing purpose, inputs, and outputs — the Python standard, retrievable via `help()` and used by auto-doc tools (Sphinx, pdocs) and IDEs (`?` in Jupyter). Should state the interface/contract (Ch 8) especially when type annotations aren't used.
- **Docstring templates**: Google style, NumPy style, and reStructuredText style are the three main conventions; pick one and use it consistently across a codebase (many editors can auto-generate a stub in the chosen format).
- **README.md**: the standard entry-point document for a repository/project, written in Markdown, giving an executive summary, getting-started instructions, caveats/limitations, and suggested next steps.
- **Markdown**: a lightweight markup language (`.md` files) using symbols (`#` for headings, etc.) for structure — used in READMEs and Jupyter notebook text cells alike.
- **Notebook documentation**: descriptive notebook filenames (never `untitled1.ipynb`), an introductory description at the top, Markdown text cells (toggle with the `m` hotkey) interspersed with code to summarize/caveat/explain without duplicating what the code shows, and Markdown headings (`#`, `##`, `###`) for navigable structure.
- **Experiment tracking**: structured, automated recording of ML experiment variables — training data, train/val/test split, feature engineering choices, hyperparameters, evaluation metrics — so experiments are reproducible and comparable; manual recording is tedious and error-prone, so dedicated tools are recommended.
- **Experiment tracking tools**: Weights & Biases (integrates with scikit-learn/TensorFlow/PyTorch, logs to a web dashboard), plus Sacred, MLflow, Neptune, and AWS SageMaker Experiments as alternatives.

## Mental Models
- **Documentation as a readability multiplier, not an afterthought**: the chapter opens by naming the common failure mode directly — documentation "commonly left until the end... rushed or omitted" — framing good documentation as integral to, not separate from, writing the code.
- **"Each fact lives in exactly one place"**: the documentation hierarchy is really DRY (Ch 1) applied to knowledge about the code — picking the *right level* for each fact prevents both omission and duplication/staleness.
- **Comments as footnotes, docstrings as abstracts**: a comment makes one small point the way a footnote does; a docstring gives the overview the way an abstract does — different documents for different purposes within the same artifact.
- **The failed-experiment-is-still-valuable-knowledge model**: even a notebook that was a "blind alley" is worth preserving and naming well, because knowing what was tried and failed is itself useful information for future work.
- **Names as a readability investment with a refactor safety net**: because IDEs support global rename, a "good enough for now" name is an acceptable starting point — you can always upgrade it later without manual search-and-replace risk.

## Anti-patterns
- **Comments that restate the code**: e.g., `# Train the classification model` directly above `classifier.fit(X_train, y_train)` — adds no information and must be kept in sync with code changes for no benefit (a DRY violation).
- **Vague or generic names**: `untitled1.ipynb`, `process_data`, `utils.py`, `im_id`, `cl_df` — communicate little and increase the reader's mental translation burden.
- **Overly similar names**: e.g., Python's `strptime` vs. `strftime` — forces readers to hold extra disambiguating knowledge in mind.
- **Shadowing Python built-ins as variable names**: `list = [0, 2, 4]` then calling `list()` later breaks the built-in constructor.
- **Using prefixes instead of subfolders to group related files**: per David Nicholson's recommendation, prefix-based naming (`data_load_x.py`, `data_clean_x.py`) is weaker than organizing related files into subfolders.
- **Letting documentation go stale**: tutorials/examples that no longer run against current code are "more frustrating" than no documentation; prevent this with automated tests for documentation examples and by storing docs alongside code (e.g., in the same repo) rather than in a disconnected external tool.
- **Recording ML experiment parameters manually**: tedious and error-prone — use an automated experiment-tracking tool instead.
- **Storing documentation in proprietary/separate-login tools**: makes it harder to keep updated in sync with code; prefer docs that live with the code (e.g., in the GitHub repo) over tools like Confluence.

## Code Examples
```python
# Bad: unclear names requiring a comment to compensate
# calculate the accuracy of the predictions compared to the test data
a = sum(x == y for x, y in zip(p, t)) / len(p)

# Good: names communicate intent directly
accuracy = sum(x == y for x, y in zip(predictions, test_data)) / len(predictions)
```
- **What it demonstrates**: good names can eliminate the need for a clarifying comment entirely; short names (`x`, `y`) remain fine when scoped tightly inside a comprehension.

```python
# Bad comment (restates the code — DRY violation)
# Train the classification model
classifier.fit(X_train, y_train)

# Good comment (adds info not visible in the code)
def slow_way_to_calculate_mode(list_of_numbers):
    # This is for demo purposes only
    # Use the mode function from the statistics library in real use cases
    ...
```
- **What it demonstrates**: the difference between a comment that duplicates the code and one that adds a caveat the code can't express on its own.

```python
# A high-quality docstring (from the pandas codebase, head() method)
def head(self: NDFrameT, n: int = 5) -> NDFrameT:
    """
    Return the first `n` rows.
    This function returns the first `n` rows for the object based
    on position. It is useful for quickly testing if your object
    has the right type of data in it.
    For negative values of `n`, this function returns all rows except
    the last `|n|` rows, equivalent to ``df[:n]``.
    If n is larger than the number of rows, this function returns all rows.
    Parameters
    ----------
    n : int, default 5
        Number of rows to select.
    Returns
    -------
    same type as caller
        The first `n` rows of the caller object.
    """
```
- **What it demonstrates**: a complete docstring covering overall description, a behavioral caveat, an edge case, parameter type/default, and return type — retrievable via `help(df.head)` or `pd.DataFrame.head?` in Jupyter.

```python
# IDE-autogenerated docstring stub, then completed
def process_sdg_data(input_excel_file, columns_to_drop):
    """_summary_

    Args:
        input_excel_file (_type_): _description_
        columns_to_drop (_type_): _description_

    Returns:
        _type_: _description_
    """
    ...

# Completed version:
def process_sdg_data(input_excel_file, columns_to_drop):
    """Rearrange SDG (Sustainable Development Goals) data to load in API.

    Args:
        input_excel_file (str): Path to the input data file.
        columns_to_drop (list): List of column names to drop.

    Returns:
        pandas DataFrame: DataFrame with columns expected by API code.
    """
    df = pd.read_excel(input_excel_file)
    df = df.drop(columns_to_drop, axis=1)
    df = df.set_index('GeoAreaName').transpose()
    return df
```
- **What it demonstrates**: filling in an IDE-generated docstring template to document a function's purpose, inputs, and outputs — directly serving the "interfaces and contracts" idea from Ch 8.

```markdown
# Jupyter notebook Markdown headings for navigation
# This is a top level heading
## This is a second level heading
### This is a third level heading
```
- **What it demonstrates**: using Markdown heading syntax (toggle a cell to Markdown with the `m` hotkey) to give a notebook navigable structure.

## Reference Tables
| Documentation level | Length | Purpose | Should contain |
|---|---|---|---|
| Names | Shortest | Signal intent at a glance | Expressive, convention-following identifiers |
| Comments | Single point | Footnote-style extra info | Caveats, TODOs, links, summaries — never restated code |
| Docstrings | Function/class/module overview | Describe purpose + interface | Inputs, outputs, edge cases, overall behavior |
| API documentation | Endpoint-level | Describe API contract | Expected input/output per endpoint (Ch 11) |
| READMEs / tutorials | Project-level | Overall context + onboarding | Summary, getting started, caveats, next steps |

| Naming convention | Applies to | Example |
|---|---|---|
| `snake_case` | Variables, functions | `x_train_array` |
| `CamelCase` | Classes | `CountryData` |
| `ALL_CAPS` | Constants/globals | `MAX_RETRIES` |
| Accepted short forms | Scoped/conventional contexts | `i`, `j` (counters); `df`; `fig`, `ax` |

| ML experiment variable to record | Example |
|---|---|
| Training data | Dataset version/source |
| Train/val/test split | Ratios or indices used |
| Feature engineering choices | Which transformations applied |
| Hyperparameters | Regularization strength, learning rate |
| Evaluation metrics | Accuracy, precision, recall |

## Worked Example
The `process_sdg_data` function (seen throughout the book) is used to show the complete documentation lifecycle for a single function: starting from an IDE-generated docstring stub with placeholder text (`_summary_`, `_description_`, `_type_`), the author fills in a one-line purpose statement ("Rearrange SDG... data to load in API"), documents each argument's expected type and meaning (`input_excel_file (str)`, `columns_to_drop (list)`), and documents the return type and meaning (`pandas DataFrame: DataFrame with columns expected by API code`) — directly operationalizing the function's "contract" (from Ch 8) as readable, tool-discoverable text.

## Key Takeaways
1. Documentation is not an afterthought — it's what makes code reproducible, reusable, and maintainable by others and by future-you; write it as you go, not rushed at project's end.
2. Match documentation detail/language to your actual audience (teammate, other team, future self).
3. Each fact belongs at exactly one level of the documentation hierarchy (names < comments < docstrings < API docs < longer documents) — never duplicate the same information across levels.
4. Good names (descriptive, appropriately long, convention-following, unit-bearing) are the cheapest and most pervasive form of documentation; avoid ambiguous abbreviations, overly similar names, and shadowing built-ins.
5. Comments should add information the code doesn't already show (caveats, TODOs, links) — never restate the code.
6. Every function should have a docstring describing its purpose, inputs, and outputs (its interface/contract), ideally in a consistent template (Google/NumPy/reST style).
7. Every repository should have a README.md with an executive summary, getting-started guidance, caveats, and next steps — kept up to date and stored alongside the code.
8. Jupyter notebooks need descriptive filenames, an intro description, and interspersed Markdown text/headings to stay navigable and useful later — even failed experiments are worth documenting.
9. Record ML experiment variables (data, split, features, hyperparameters, metrics) systematically, ideally via an automated tracking tool (Weights & Biases, MLflow, etc.) rather than by hand.

## Connects To
- **Ch 1**: Readability pillar and the DRY principle underpin the "document once, at the right level" rule.
- **Ch 6**: Type annotations as an alternative/complement to documenting input/output types in docstrings.
- **Ch 8**: Docstrings and READMEs operationalize the function "interfaces and contracts" and project-design concepts.
- **Ch 11**: API documentation described further in the context of building APIs.
- **External**: PEP8 and the Google Python Style Guide (naming guidance); David Nicholson's blog post on file organization; Hynek Schlawack's article on test docstrings; the Write the Docs community; Sphinx, pdocs, Read the Docs, GitHub Pages (doc generation/hosting); Weights & Biases, Sacred, MLflow, Neptune, SageMaker Experiments (experiment tracking).
