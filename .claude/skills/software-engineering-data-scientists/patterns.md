# Patterns

## DRY (Don't Repeat Yourself)
**When to use** Any time you notice duplicated logic, near-identical code blocks, or repeated data.
**How** Extract repeated code into a function, loop, or shared module; parameterize instead of copy-pasting; avoid comments that restate the code; store each fact (incl. documentation) in exactly one place.
**Trade-offs** Over-abstracting too early can add unneeded complexity (essential vs. accidental complexity); some duplication is fine if the pieces are conceptually unrelated and likely to diverge.

## Measure-first optimization
**When to use** Whenever tempted to make code "faster."
**How** Confirm correctness first, confirm there's a real performance requirement, then measure (time/profile/memory) before changing anything; fix only the identified bottleneck.
**Trade-offs** Upfront measuring costs a little time; skipping it risks optimizing the wrong thing (premature optimization).

## Choosing the right data structure
**When to use** Any time data needs to be stored/queried, especially before writing a processing loop.
**How** Match the structure to the operation: list for index/order, dict for key lookup, set for membership tests, NumPy array for multidimensional single-type numeric data, pandas DataFrame for tabular/mixed-type data; use Big O awareness to anticipate scaling behavior.
**Trade-offs** Converting between structures (e.g., list→set) has an upfront O(n) cost only worth paying for repeated later operations; NumPy's single-dtype constraint enables speed but disallows mixed types.

## Vectorization
**When to use** Any computation on arrays/DataFrames where a built-in vectorized method exists.
**How** Use NumPy/pandas built-ins (`np.sum`, `.str.lower()`, arithmetic on whole columns) instead of Python loops, `.apply()`, or `.iterrows()`.
**Trade-offs** Vectorized code can be less flexible for arbitrary custom logic; `.apply()` is the fallback when no vectorized method exists, at a performance cost.

## Five Pillars of Good Code
**When to use** As a running checklist when writing or reviewing any code meant to be reused or maintained.
**How** Check Simplicity (no repetition/unneeded complexity), Modularity (well-defined functions/components), Readability (standards, names, docs), Performance/Efficiency (adequate speed/memory), Robustness (error handling, tests).
**Trade-offs** Full rigor on every pillar isn't warranted for throwaway/one-off scripts — be pragmatic about where to invest.

## Function Design Framework (Name → Inputs → Behavior → Outputs)
**When to use** Before writing the body of any new function, especially converting notebook code to modular functions.
**How** Choose a descriptive name first; decide the input arguments (interface) before writing logic; write a single-purpose body; decide outputs (also part of the interface) — keep interface decisions stable afterward.
**Trade-offs** Upfront interface design takes a moment longer than "just start coding," but prevents costly interface churn later.

## Loose coupling
**When to use** Designing how functions/modules call each other.
**How** Pass data explicitly as arguments between functions rather than hardcoding one function's call to another with fixed parameters; let the caller wire pieces together.
**Trade-offs** Slightly more boilerplate at call sites, but each piece becomes independently testable, reusable, and swappable.

## Refactor-Test-Save loop
**When to use** Restructuring code without intending to change its behavior.
**How** Confirm tests pass → make one small change → rerun tests → save/commit → repeat; keep the interface fixed for internal-only changes, add integration tests if the interface must change.
**Trade-offs** Requires an existing test suite to be safe; without tests, "refactoring" is just an unverified gamble.

## Interfaces as Contracts
**When to use** Designing any function/module other code will depend on.
**How** Keep argument count small (≤3–4; else use a config object); decide inputs/outputs deliberately and don't change them casually; enforce with tests, docstrings, and type annotations/checking.
**Trade-offs** Rigid contracts can slow iteration during early exploratory work — fine to defer contract-stability until the design has settled.

## Documentation hierarchy (one fact, one place)
**When to use** Deciding where any piece of information about code belongs.
**How** Names (intent) → comments (single point/caveat) → docstrings (function/class overview + interface) → API docs (endpoint contract) → longer docs (README/tutorials, project overview); never duplicate a fact across levels.
**Trade-offs** Requires discipline to keep each level current as code changes; stale docs are worse than none.

## Arrange / Act / Assert / Cleanup
**When to use** Writing any automated test.
**How** Set up inputs (Arrange), call the function (Act), check the result (Assert), remove side effects (Cleanup).
**Trade-offs** None significant — a lightweight structuring convention; adds negligible overhead.

## Data validation via schema
**When to use** Any pipeline step ingesting external or evolving data.
**How** Define a schema (Pandera for DataFrames, Pydantic for runtime/API input) describing required columns/types/ranges; validate data against it before processing.
**Trade-offs** Schemas need maintenance as data evolves; too-strict schemas can break on legitimate new data variants, too-loose ones miss real problems.

## Automate don't manually police (formatting/linting/type-checking)
**When to use** Always, for script-based Python projects.
**How** Black/Ruff for formatting, isort for imports, Flake8/Pylint/Ruff for linting, mypy/Pyright for type checking — run standalone, in-IDE, or via pre-commit/CI.
**Trade-offs** Initial setup/configuration time; team must agree on one toolset to avoid inconsistent rules across contributors.

## Pre-commit hooks for lightweight per-commit automation
**When to use** Fast, frequent checks you want enforced on every commit (formatting, debug-statement removal, notebook-output stripping).
**How** `.pre-commit-config.yaml` listing hook repos/ids or custom local hooks; `pre-commit install`; runs automatically on `git commit`.
**Trade-offs** Keep hooks fast — not suited to full test suites on large projects (use CI instead); first run is slow (creates its own venv).

## CI/CD pipeline for heavier, remote automation
**When to use** Running full test suites, dependency/security scans, or builds on every push, especially when local pre-commit would be too slow.
**How** GitHub Actions (or Jenkins/CircleCI/Travis CI) workflow triggered on push/PR: checkout → setup environment → install deps → run tests/scans.
**Trade-offs** Runs on remote infra (cost beyond free tier); more setup complexity than pre-commit, often owned by a DevOps function.

## Containerized, reproducible deployment
**When to use** Deploying a service (e.g., an API) so it runs identically regardless of host machine.
**How** Write a Dockerfile (base image → copy requirements → install deps → copy code → CMD), build an image, push to a registry, deploy via a cloud run service.
**Trade-offs** Added complexity vs. running code directly; incurs cloud costs; requires logging setup since you lose direct console access.

## Branch + Pull Request + Code Review
**When to use** Any multi-person change to a shared codebase.
**How** Create a branch, commit changes, push, open a PR describing what/why, get it reviewed, merge into main, delete the branch.
**Trade-offs** Adds review latency; worth it for catching bugs/security issues and sharing knowledge — keep reviews time-bounded to avoid bottlenecks.

## Security: static analysis + dependency scanning
**When to use** Continuously, ideally automated (pre-commit/CI).
**How** Run Bandit against your own code, pip-audit (or Dependabot) against dependencies; triage flagged issues by actual relevance to your deployment context rather than blindly fixing all.
**Trade-offs** Scanners produce false positives/context-irrelevant warnings (e.g., `assert` flagged even in non-optimized test code) requiring human judgment.

## Adapting Agile for data science
**When to use** Structuring exploratory, uncertain-timeline data science work within a team process.
**How** Keep frequent planning meetings, stakeholder/product-owner involvement, demos, and retrospectives; drop the expectation of a deployable increment every sprint.
**Trade-offs** Loses some of Scrum's predictability guarantees; requires explicit buy-in that research timelines are inherently uncertain.
