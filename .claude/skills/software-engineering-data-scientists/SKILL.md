---
name: software-engineering-data-scientists
description: "Knowledge base from \"Software Engineering for Data Scientists\" by Catherine Nelson (O'Reilly). Use when writing production-quality data science code — code performance analysis, effective data structures, OOP/functional programming, error handling/logging/debugging, code formatting/linting/type checking, testing, design and refactoring, documentation, version control/dependencies/packaging, building APIs, automation/deployment, or security for data science code."
---

# Software Engineering for Data Scientists

Knowledge base distilled from Catherine Nelson's *Software Engineering for Data Scientists* (O'Reilly, 2024). The book translates professional software-engineering practice into terms and examples relevant to data scientists, using a running UN Sustainable Development Goals dataset example throughout.

## Core Frameworks

### The Five Pillars of Good Code (Ch 1, revisited Ch 15)
The organizing framework for the entire book. Judge any piece of code — and decide how much engineering rigor it deserves — along five dimensions:
1. **Simplicity** — no repetition (DRY), no unneeded complexity, no unneeded verbosity.
2. **Modularity** — broken into small, independent, well-defined functions/classes with stable interfaces.
3. **Readability** — follows PEP8, has good names, is documented, has no debug/dead-code cruft.
4. **Performance/Efficiency** — doesn't take unnecessarily long or use more resources than needed, but only matters once correctness and a real requirement are established.
5. **Robustness** — reproducible, handles errors predictably, is logged, and is tested.
Be pragmatic: a one-off/deadline hack doesn't need full rigor on all five — but know when that's the trade-off you're making.

### Measure-first performance workflow (Ch 2, Ch 3)
Never optimize before confirming correctness and a real performance requirement. Then: time (`time`/`timeit`) → profile (`cProfile`, `line_profiler`, `Memray`) → fix the identified bottleneck, often by choosing a better **data structure** (list O(n) search vs. dict/set O(1) lookup) or **algorithm** (avoid nested loops / O(n²) when a single pass works), or by using **vectorized** NumPy/pandas operations instead of Python loops. Big O notation (O(1), O(log n), O(n), O(n log n), O(n²), O(2ⁿ)) describes hardware-independent scaling and is the shared vocabulary for this reasoning.

### OOP and FP as decomposition tools (Ch 4)
Two complementary paradigms for breaking code into pieces. **OOP** (class/method/attribute, inheritance, encapsulation, abstraction, polymorphism) fits problems with many instances of similar things needing shared behavior but different data — never create a class for a single instance. **FP** (pure functions, no side effects, `lambda`/`map`/`filter`/comprehensions) fits fixed data you want to transform repeatedly or parallelize. Most real code blends both pragmatically. Vectorized/built-in operations beat `.apply()` with custom functions for simple work.

### Errors, logging, and debugging (Ch 5)
For every failure point, make an explicit choice: let it crash, handle it (`try`/`except`/`else`/`finally` with a same-shaped default return), or `raise` a descriptive (optionally custom) exception. Never use a bare `except: return`. Use the `logging` module (not `print`) for anything beyond a quick script — default level is WARNING, default destination is console, so configure both explicitly; `logging.exception()` captures the full traceback. Debugging escalates: print statements → logging → interactive debugger (IDE breakpoints or `pdb`), alongside non-technical strategies (small experiments, searching the exact error text, rubber-ducking, taking a break).

### Formatting, linting, type checking — automate, don't manually police (Ch 6)
Formatting (Black/Ruff format) changes only appearance; linting (Flake8/Pylint/Ruff check) flags likely runtime errors and style issues; type checking (mypy/Pyright) verifies type-annotation consistency. Type annotations alone have zero runtime effect — they only matter paired with a type checker. Pick one toolset per team; automate via IDE integration or pre-commit (Ch 12). PEP8 is the baseline style guide.

### Testing (Ch 7)
Structure every test as **Arrange → Act → Assert → Cleanup**. Write tests when code will be reused, before deployment, after a bug fix (regression), or before refactoring — not for pure exploration or for testing third-party code. Unit tests (small, fast, deterministic) vs. integration tests (multiple units combined). **Data validation** (Pandera for DataFrames, Pydantic for runtime/API input) tests the *data's* shape/type/range, not code logic — critical since real data drifts over a project's life. **ML testing** checks process properties (weights changed, loss decreased, correct I/O shapes, known examples predict correctly) rather than exact outputs, since training is stochastic.

### Design, modularity, and refactoring (Ch 8)
Design a function via **Name → Inputs → Behavior → Outputs**, treating inputs/outputs as a stable **contract** (interface) other code depends on. Minimize **coupling** — pass data explicitly between functions rather than hardcoding calls with fixed parameters. **Refactoring** (Fowler: changing structure without changing external behavior) is safe only with a pre-existing test suite: make a small change → rerun tests → save/commit → repeat. Move from notebooks to scripts once work becomes repeatable/production-bound.

### Documentation hierarchy — one fact, one place (Ch 9)
Names → comments (single point, never restate code) → docstrings (function/class purpose + interface) → API docs → longer docs (README, tutorials). Never duplicate a fact across levels. Good names are the cheapest, most pervasive documentation (`snake_case` vars/functions, `CamelCase` classes, `ALL_CAPS` constants, no ambiguous abbreviations). Every repo needs a README.md; every function needs a docstring describing inputs/outputs/purpose; ML experiments need structured, ideally automated, tracking (Weights & Biases, MLflow, etc.).

### Sharing code: Git, dependencies, packaging (Ch 10)
Git workflow: `init` → `add` → `commit` (small, focused, tested) → `push`/`pull`; branch → PR → review → merge for collaboration. Never commit data or secrets to version control. Isolate every project in a **virtual environment**; manage dependencies via `requirements.txt` (pip, simple) or `pyproject.toml`+`poetry.lock` (Poetry, reproducible with exact hashes). Package code (pyproject.toml → build tool → PyPI/Twine, or `poetry build`/`publish`) only once it's functional, formatted, tested, and documented.

### Building and consuming APIs (Ch 11)
Client/server, RESTful (GET/POST/PUT/DELETE), JSON responses, status codes (2xx/4xx/5xx). Consume APIs with the `requests` library (`.status_code`, `.headers`, `.json()`). Build APIs with **FastAPI** (served by Uvicorn): keep business logic in its own module, separate from endpoint-wiring code; validate POST input with **Pydantic** models; FastAPI auto-generates Swagger UI docs. Add error handling (Ch 5) before calling an API production-ready.

### Automation and deployment (Ch 12)
**Pre-commit hooks** (local, fast, per-commit: formatting, debug-statement removal, notebook-output stripping) vs. **CI/CD** (GitHub Actions etc., remote, per-push: full test suites, clean-env checks). CI = automated build+test; Continuous Delivery = tested code awaits manual deploy approval; Continuous Deployment = automatic deploy on passing tests. Deploy reproducibly via **Docker** (Dockerfile → image → container) pushed to a cloud registry and run via a container service (Cloud Run / App Service / ECS).

### Security (Ch 13)
Vocabulary: attacker, threat, vulnerability, risk (likelihood × severity), mitigation; C-I-A triad (Confidentiality, Integrity, Availability). Key risks for data scientists: stolen credentials (phishing/social engineering), vulnerable/malicious third-party packages, insecure `pickle` (use JSON/Skops/ONNX instead), secrets/data committed to version control, undefended APIs. Automate scanning (Bandit for code, pip-audit/Dependabot for dependencies). ML-specific attacks: input manipulation, training data extraction, model theft, data poisoning, transfer-learning attacks — mitigated by data validation, adversarial testing, model monitoring.

## Chapter Index

| Ch | Title | Core Topic |
|----|-------|------------|
| 1 | What Is Good Code? | Five pillars: simplicity, modularity, readability, performance, robustness |
| 2 | Analyzing Code Performance | Timing, profiling, Big O / time complexity |
| 3 | Using Data Structures Effectively | Lists/tuples/dicts/sets, NumPy arrays, pandas DataFrames, Dask |
| 4 | OOP and Functional Programming | Classes, inheritance/polymorphism, lambda/map/apply |
| 5 | Errors, Logging, and Debugging | try/except/raise, logging module, pdb, debugging strategies |
| 6 | Code Formatting, Linting, and Type Checking | PEP8, Black, isort, Flake8/Pylint/Ruff, mypy |
| 7 | Testing Your Code | Pytest, unit/integration tests, Pandera/Pydantic, ML testing |
| 8 | Design and Refactoring | Project structure, coupling, interfaces, refactor workflow |
| 9 | Documentation | Names, comments, docstrings, READMEs, notebooks, experiment tracking |
| 10 | Sharing Your Code | Git, virtual environments, pip/Poetry, packaging/PyPI |
| 11 | APIs | REST, requests library, FastAPI, Pydantic validation |
| 12 | Automation and Deployment | Pre-commit hooks, GitHub Actions, Docker, cloud deployment |
| 13 | Security | Threats/vulnerabilities, pickle, Bandit/pip-audit, ML security |
| 14 | Working in Software | SDLC, Agile/Scrum/Kanban, industry roles, community |
| 15 | Next Steps | Recap, AI coding assistants' limitations, pragmatism, ethics |

## Topic Index

- **Abstraction** → Ch 4, Ch 8, Ch 15
- **Agile / Scrum / Kanban** → Ch 14
- **API design/building (FastAPI)** → Ch 11
- **API security** → Ch 13
- **Automation (pre-commit, CI/CD)** → Ch 12
- **Big O notation / time complexity** → Ch 2, Ch 3
- **Branches and pull requests** → Ch 10
- **Coupling** → Ch 8
- **Data structures (lists, dicts, sets, NumPy, pandas, Dask)** → Ch 3
- **Data validation (Pandera, Pydantic)** → Ch 7
- **Debugging (pdb, IDE debuggers)** → Ch 5
- **Dependency management (pip, Poetry, virtual environments)** → Ch 10
- **Deployment (Docker, cloud)** → Ch 12
- **Design patterns / SOLID** → Ch 8
- **Docstrings** → Ch 9
- **DRY principle** → Ch 1
- **Error handling (try/except/raise)** → Ch 5
- **Formatting (Black, PEP8)** → Ch 6
- **Functional programming (lambda, map, apply)** → Ch 4
- **Git / version control** → Ch 10
- **Industry roles (engineer, QA, data engineer/analyst, PM, UX, designer)** → Ch 14
- **Linting (Flake8, Pylint, Ruff)** → Ch 6
- **Logging** → Ch 5
- **Machine learning testing** → Ch 7
- **Machine learning security** → Ch 13
- **Modularity / modular code** → Ch 1, Ch 4, Ch 8
- **Naming conventions** → Ch 9
- **Object-oriented programming (classes, inheritance, polymorphism)** → Ch 4
- **Open source / community** → Ch 14
- **Packaging (pyproject.toml, PyPI)** → Ch 10
- **Performance / profiling** → Ch 2
- **Pickle module security risk** → Ch 13
- **Project design/structure** → Ch 8
- **Readability** → Ch 1, Ch 6, Ch 9
- **Refactoring** → Ch 1, Ch 8
- **REST APIs, HTTP methods/status codes** → Ch 11
- **Security terminology (threat, vulnerability, risk)** → Ch 13
- **SDLC (software development lifecycle)** → Ch 14
- **Testing (Pytest, unit/integration tests)** → Ch 7
- **Type checking / type annotations (mypy)** → Ch 6
- **Vectorization (NumPy/pandas)** → Ch 2, Ch 3
- **Virtual environments** → Ch 10

## Supporting Files

- [glossary.md](glossary.md) — Alphabetical glossary of all significant terms with chapter references
- [patterns.md](patterns.md) — All techniques/patterns with when-to-use, how, and trade-offs
- [cheatsheet.md](cheatsheet.md) — Decision rules, trade-off tables, thresholds, and red-flag checklists
- [chapters/](chapters/) — Full per-chapter notes (ch01–ch15), each with frameworks, key concepts, code examples, worked examples, and takeaways

## Scope & Limits

- This skill covers **Python-specific** software engineering practice aimed at data scientists; it is not a general software engineering or computer science textbook, and deliberately omits data structures more central to general CS (linked lists, heaps, trees, etc.) and most classic OOP/FP theory beyond what's directly useful.
- Deployment/cloud/security chapters give an *introductory overview* only — the book itself repeatedly recommends consulting specialists (security teams, DevOps) for production-grade API security, cloud architecture, and compliance; this skill should not be treated as sufficient for production security decisions.
- ML security and ML testing sections reflect the state of a fast-moving field as of the book's 2024 writing; newer attack types/tools may exist.
- Data privacy (GDPR, CCPA, etc.) is explicitly out of scope of the source book (it points to *Practical Data Privacy* by Katharine Jarmul instead) and so is out of scope here.
- Code examples are drawn from the book's running UN Sustainable Development Goals dataset example and illustrative toy functions (`weighted_mean`, `fit_trendline`) — they are pedagogical, not production-hardened.
- Where the book recommends further reading (e.g., *The Pragmatic Programmer*, *A Philosophy of Software Design*, *High Performance Python*, *Robust Python*), those are noted in chapter "Connects To" sections but not summarized here in depth.
