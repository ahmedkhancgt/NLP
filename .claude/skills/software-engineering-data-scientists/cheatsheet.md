# Cheatsheet

## Should I optimize this code?
1. Is it correct? If not, fix that first — never optimize broken code.
2. Is there an actual performance requirement (SLA, user-facing latency)? If no, stop.
3. Have you measured (timed/profiled)? If no, measure before changing anything.
4. Only then: fix the identified bottleneck, and re-measure.

## Data structure decision table
| Need | Use | Why |
|---|---|---|
| Index-based lookup, order matters | List | O(1) index lookup |
| Fixed, small, unchanging collection | Tuple | O(1) lookup, immutable, cached |
| Key→value lookup | Dict | O(1) via hash table |
| Fast membership test / uniqueness | Set | O(1) via hash table |
| Multidimensional, single-type numeric | NumPy array | Vectorization, C speed |
| Tabular, mixed-type, time series | pandas DataFrame | Column types, indexing, time series tools |
| Data > memory, or parallel compute | Dask | Chunked, parallel/out-of-core |

## Big O cheat sheet
| Notation | Name | Rule of thumb |
|---|---|---|
| O(1) | Constant | Index lookup, dict/set ops |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Single loop |
| O(n log n) | — | Efficient sort |
| O(n²) | Quadratic | Nested loop over same data — AVOID if n can grow |
| O(2ⁿ) | Exponential | Naive recursion — AVOID for large n |

## pandas performance ordering (fastest → slowest)
Vectorized operation > `.apply(func)` > `.iterrows()` (avoid)

## When to write a test
- Before deploying code to production or before someone else uses it.
- When code will be reused/modified.
- Right after a bug is fixed (regression test).
- Before refactoring.
- NOT needed: pure one-off exploration; never needed for third-party library code.

## Test type quick picker
| Situation | Test type |
|---|---|
| One function/class in isolation | Unit test |
| Multiple functions chained together | Integration test |
| Data shape/type/range, not code logic | Data validation (Pandera/Pydantic) |
| ML training | Check weights changed, loss decreased, correct I/O shapes |
| ML inference | Known example → expected prediction/range/class |

## Formatting/linting/type-checking tool picker
| Task | Tool |
|---|---|
| Auto-fix appearance | Black, or `ruff format` |
| Sort imports | isort |
| Flag style/logic issues | Flake8, Pylint, or `ruff check` (pick ONE per team) |
| Verify type annotations | mypy (or Pyright/Pyre) |

## Naming conventions
| Element | Convention | Example |
|---|---|---|
| Variable/function | `snake_case` | `x_train_array` |
| Class | `CamelCase` | `CountryData` |
| Constant/global | `ALL_CAPS` | `MAX_RETRIES` |
| Accepted short names | conventional contexts only | `i`, `j`, `df`, `fig`, `ax` |

## Error handling decision
- Expected, recoverable failure → `try`/`except` with a same-shaped default return.
- Caller needs to know something is fundamentally wrong → `raise` a descriptive (optionally custom) exception.
- NEVER: bare `except: return` (silently swallows all errors).

## Logging level picker
| Situation | Level |
|---|---|
| Diagnostic detail, dev-time only | DEBUG |
| Normal progress confirmation | INFO |
| Default threshold — unexpected but non-fatal | WARNING |
| Some function failed | ERROR |
| Program itself may be unable to continue | CRITICAL |
Remember: default logging level is WARNING — INFO/DEBUG calls are silently dropped unless you lower it.

## Git workflow cheat sheet
```
git init → git add <file> → git commit -m "..."
git remote add origin <url> → git push -u origin main
git checkout -b <branch> → ...commit... → git push origin <branch>
git checkout main → git merge <branch>
```
Commit discipline: one logical change per commit; run tests before committing.

## Dependency/versioning tells
- `requirements.txt` (pip freeze) = simple, manual sub-dependency management.
- `pyproject.toml` + `poetry.lock` (Poetry) = exact hashed versions, cross-platform reproducibility.
- SemVer: major.minor.patch — major = breaking, minor = new features, patch = small fixes.

## API smell checklist
- No HTTPS/auth on sensitive data → stop, get expert help.
- Database-backed endpoint → check for SQL injection risk, validate inputs.
- POST endpoint with unvalidated JSON → add a Pydantic model.
- Business logic inline in endpoint function → extract to a separate module.

## Security red flags (fix before shipping)
- Using `pickle` to load data/models from any untrusted source → use JSON/Skops/ONNX instead.
- API keys or data committed to version control → move to env vars / excluded files; rotate the key.
- Outdated third-party dependencies → run pip-audit / enable Dependabot, update regularly.
- Undefended public API serving real/sensitive data → add auth + HTTPS, consult security team.
- ML model trained on sensitive data → validate training data, test adversarial inputs, monitor the deployed model.

## Automation layer picker
| Need | Tool |
|---|---|
| Fast check on every commit (format, lint, strip debug/notebook output) | Pre-commit hook |
| Full test suite / clean-env dependency check on every push | GitHub Actions (CI) |
| Reproducible deployment of a running service | Docker + cloud container-run service |

## "Is this code good enough?" pragmatism check
- Is this a one-off / tight-deadline hack? → "Whatever gets the job done" is fine — but know that's what you're doing.
- Will this be reused, shared, or deployed? → Apply the five pillars (simplicity, modularity, readability, efficiency, robustness) in full.

## AI-generated code checklist
- [ ] Ran the generated code yourself?
- [ ] Independently verified any numeric/expected values (don't trust the tool's own docstring/test values)?
- [ ] Added logging, docs, error handling if not already present (the tool won't add these unprompted)?
- [ ] Checked for simplification opportunities (e.g., unnecessary tolerance/approximation checks)?
