# Glossary

**Abstraction** — Dealing with a class/system at the appropriate level of detail, hiding internal detail behind an interface (Ch 4, Ch 8)

**Acceptance test** — Checks that a system does what the user expects (Ch 7)

**Agile software development** — Methodology emphasizing frequent updates, flexibility, and rapid feedback over detailed upfront planning (Ch 14)

**API (Application Programming Interface)** — A standardized interface letting two systems communicate/exchange data (Ch 11)

**Attribute** — A data property of a class instance, accessed without parentheses (Ch 4)

**Bandit** — Open-source static-analysis tool scanning Python code for security vulnerabilities (Ch 13)

**Big O notation** — Hardware-independent notation (O(f(n))) describing how runtime scales with input size n (Ch 2)

**Bit-rot** — The need to update code that has been unused for a while, due to changed dependencies (Ch 1)

**Black** — Opinionated Python code auto-formatter (Ch 6)

**Branch (Git)** — An isolated line of development tracked from a point in project history (Ch 10)

**C-I-A Triad** — Confidentiality, Integrity, Availability — a framework for evaluating security (Ch 13)

**Class** — A blueprint for creating objects with shared attributes/methods (Ch 4)

**Coupling** — The degree of interdependence between functions/modules; lower is better (Ch 8)

**cProfile** — Python's built-in profiler, giving function-call-level timing breakdown (Ch 2)

**Dask** — Library for parallel/distributed array and DataFrame operations, including out-of-memory data (Ch 3)

**Data validation** — Confirming that data (not code) matches expected shape/type/range/distribution (Ch 7)

**Dependency** — A third-party library your code requires to function (Ch 10)

**Deployment** — Making a "done" version of code operational for end users (Ch 12)

**Docker** — System for building/managing reproducible containers (Ch 12)

**Docstring** — A formal, longer documentation block at the start of a function/class describing purpose, inputs, outputs (Ch 9)

**DRY (Don't Repeat Yourself)** — Principle that each piece of knowledge should have one single representation in code (Ch 1)

**Dtype** — The single data type stored with/by a NumPy array (Ch 3)

**Edge case** — An extreme/unusual input requiring special handling in tests (Ch 7)

**Encapsulation** — Hiding a class's implementation behind a public interface (Ch 4)

**Exception** — A runtime error from otherwise-valid Python code (e.g., KeyError, ValueError) (Ch 5)

**FastAPI** — Modern Python web framework for building APIs (Ch 11)

**Flake8 / Pylint / Ruff** — Python linters that check code for style/logic issues (Ch 6)

**Functional programming (FP)** — Paradigm built on pure, immutable, side-effect-free functions (Ch 4)

**Git** — The dominant distributed version control system (Ch 10)

**GitHub Actions** — CI/CD platform triggered by GitHub repository events (Ch 12)

**Hash table** — Underlying data structure for Python dicts/sets, enabling O(1) lookup (Ch 3)

**Inheritance** — A child class reusing/extending a parent class's attributes/methods (Ch 4)

**Integration test** — Checks that multiple units of code work correctly together (Ch 7)

**Interface / contract** — A function's stable, agreed input/output shape that other code depends on (Ch 8)

**isort** — Tool that sorts Python imports into PEP8-required groups (Ch 6)

**Kanban** — Visual Agile task-tracking system (planned → in progress → done) (Ch 14)

**Lambda function** — A small, anonymous, single-expression Python function (Ch 4)

**Least privilege** — Security principle: grant only the access essential to a role, no more (Ch 13)

**line_profiler** — Third-party profiler giving per-line timing breakdown within a function (Ch 2)

**Linting** — Static analysis checking code for errors/style issues without running it (Ch 6)

**Logging** — Recording a structured, searchable record of what code did, to a file (Ch 5)

**Memray** — Memory profiling tool for Python (Ch 2)

**Merge conflict** — Occurs when the same line was changed differently on branches being merged (Ch 10)

**Modular code** — Code broken into small, independent pieces, each with one well-defined purpose (Ch 1, Ch 8)

**mypy** — Static type checker for Python, verifying type annotation consistency (Ch 6)

**NumPy array (ndarray)** — N-dimensional, single-typed array enabling vectorized operations (Ch 3)

**OOP (Object-Oriented Programming)** — Paradigm organizing code around objects (data + behavior) (Ch 4)

**OWASP** — Open Worldwide Application Security Project; publishes annual top-ten security risk lists (Ch 13)

**pandas DataFrame / Series** — Tabular (2D) / labeled-array (1D) pandas data structures (Ch 3)

**Pandera** — Lightweight schema-based data validation library for pandas DataFrames (Ch 7)

**pdb** — Python's built-in command-line debugger (Ch 5)

**PEP8** — Python's foundational style guide, defining formatting/naming conventions (Ch 1, Ch 6)

**pickle** — Python's object serialization module; insecure for untrusted data (Ch 13)

**pip-audit** — Tool scanning a project's dependencies for known vulnerabilities (Ch 13)

**Poetry** — Dependency/packaging manager using pyproject.toml and poetry.lock (Ch 10)

**Polymorphism** — Different classes exposing a method of the same name/contract (Ch 4)

**Pre-commit hook** — A Git hook script that runs automatically before each commit (Ch 12)

**Premature optimization** — Optimizing code before establishing correctness and real performance need (Ch 2)

**Profiling** — Measuring which parts of code consume the most time/memory (Ch 2)

**PyPI (Python Package Index)** — The standard public repository for installable Python packages (Ch 10)

**pyproject.toml** — Standardized project metadata/build configuration file (Ch 10)

**Pydantic** — Runtime data validation library based on type annotations (Ch 7, Ch 11)

**Pytest** — The standard Python test framework/runner (Ch 7)

**Refactoring** — Changing code's internal structure without changing its external behavior (Ch 1, Ch 8)

**Repository (repo)** — A Git-tracked directory holding a project's change history (Ch 10)

**REST / RESTful API** — Architectural style for APIs using standard HTTP methods on resources (Ch 11)

**Risk (security)** — Likelihood × severity of a threat exploiting a vulnerability (Ch 13)

**Scrum** — Agile methodology organizing work into fixed-length sprints (Ch 14)

**SDLC (Software Development Lifecycle)** — Plan→design→build→test→deploy→maintain (Ch 14)

**SemVer (Semantic Versioning)** — Major.minor.patch version numbering scheme (Ch 10)

**Set** — Python hash-table-based collection of unique values with O(1) lookup (Ch 3)

**Sparse matrix** — Memory-efficient structure for mostly-zero matrices (Ch 3)

**Static analysis / code scanning** — Checking code against rules without executing it (Ch 13)

**Technical debt (tech debt)** — Deferred work from writing code quickly instead of correctly (Ch 1)

**Test coverage** — Proportion of code lines executed when tests run (not a correctness guarantee) (Ch 7)

**Test-Driven Development (TDD)** — Writing tests before the implementing code (Ch 7)

**Threat / vulnerability (security)** — A potentially harmful event / a weakness that could be exploited (Ch 13)

**Time complexity** — How an algorithm's runtime scales with input size (Ch 2)

**Traceback** — Python's error report showing the call chain leading to a failure (Ch 5)

**Type annotation (type hint)** — Optional syntax documenting expected input/output types (Ch 6)

**Type checking** — Verifying type-annotation consistency statically (e.g., with mypy) (Ch 6)

**Unit test** — Tests a small, isolated component (function/class) (Ch 7)

**Uvicorn** — ASGI web server used to serve a FastAPI application (Ch 11)

**Vectorization** — Applying operations to an entire array at once via compiled code, skipping per-element Python overhead (Ch 2, Ch 3)

**Virtual environment** — An isolated space for a project's installed Python libraries (Ch 10)

**Waterfall software development** — Sequential, non-revisitable SDLC methodology (Ch 14)

**YAML** — Human-readable configuration-file markup language used by pre-commit and GitHub Actions (Ch 12)
