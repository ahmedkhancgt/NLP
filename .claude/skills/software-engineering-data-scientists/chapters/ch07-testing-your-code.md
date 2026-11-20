# Chapter 7: Testing Your Code

## Core Idea
Automated tests give evidence that code works correctly now and continues to work as code, data, environments, and collaborators change; the effort to write them pays off whenever code will be reused, modified, or depended on by others — with special strategies needed for data validation and machine learning, where exact outputs aren't always knowable.

## Frameworks Introduced
- **Arrange / Act / Assert / Cleanup**: the four-stage structure of a test (per Pytest docs).
  - When to use: every test, as a template for organizing the code.
  - How: Arrange = set up inputs/data; Act = call the function under test; Assert = check the result matches expectations; Cleanup = remove any side effects (e.g., close opened files).
- **When-to-test heuristic**: skip tests during pure exploration; add tests once code will be reused/modified, before production deployment or others using it, whenever a bug is fixed (regression test), and when refactoring.
  - When to use: deciding whether a given piece of code warrants a test right now.
  - How: don't test one-off exploratory analysis; test code that's reused, that others depend on, that's about to ship, or that just broke; never bother testing third-party/imported library code — that's not your responsibility.
- **Six types of tests (Viafore, *Robust Python*)**: unit, integration, acceptance, load, security, usability.
  - When to use: to pick the right granularity/kind of test for a given concern; the chapter focuses on unit and integration as most relevant to data science.
  - How: unit tests isolate a function/class with small, fast, deterministic fake data; integration tests chain several units together (e.g., data processing → model fitting) and check the combined output.
- **Data validation steps**: a checklist for verifying that *data* (not code) matches expectations — present columns, correct types, acceptable null proportion, expected statistical properties (mean/std/range), distribution similarity to prior periods, anomaly checks.
  - When to use: as a pipeline step, independent of unit/integration tests of code logic, since data changes over a project's life in ways code doesn't.
  - How: define a schema (e.g., with Pandera or Pydantic) describing required columns/types/ranges and validate incoming data against it, ideally at each pipeline stage.
- **ML-specific testing strategy**: because ML has randomization and unknown target outputs, test *properties of the process* rather than exact output values.
  - When to use: throughout a model's pipeline — feature engineering (unit tests + data validation), training (process checks), inference (unit/integration tests + example-based checks).
  - How: for training, check weights change after a step, loss decreases early on, input/output tensor shapes are correct, and (for small/fast models) that a model artifact is produced at all; for inference, test on fake/edge-case inputs and confirm specific known examples produce expected predictions/ranges/classes before deployment.

## Key Concepts
- **Test**: code that calls a function and checks it does what it's supposed to — automated, as opposed to user/manual testing (A/B testing is explicitly out of scope here).
- **Test-Driven Development (TDD)**: writing tests before the implementation, using tests to specify requirements/behavior first; noted as often a poor fit for data science's exploratory early stages, where the eventual code structure isn't known upfront.
- **Edge case**: an extreme/unusual input requiring special handling (empty lists, wrong types, nulls) — tested in addition to "happy path" inputs.
- **Pytest**: the most popular Python test framework/runner; discovers files matching `test_*.py`/`*_test.py` and functions prefixed `test_`, runs them, and reports pass/fail with detailed assertion diffs. Not in the standard library — `pip install pytest`.
- **unittest**: Python standard-library test framework, more verbose to set up than Pytest but avoids an extra dependency.
- **tox**: a test runner that also manages virtual environments, dependencies, and multiple Python versions — useful for confirming code works across environments.
- **Test coverage**: proportion of code lines executed when all tests run (measured via `pytest-cov`); it shows what lines ran, not whether the logic was actually validated — a necessary-but-insufficient quality signal.
- **Unit test**: tests a small, isolated component (function/class); should be fast, use small/fake data, and be deterministic (no randomness) so the same input always produces the same expected output.
- **Integration test**: tests that multiple units work correctly together (e.g., a data-processing function feeding a trendline-fitting function), closer to how the system is actually used.
- **Pandera**: lightweight schema-based validator for pandas (and Dask) DataFrames — define a `DataFrameSchema` with column names/types/optional range checks, then call `schema(df)` to validate; errors on missing required columns but tolerates extra ones.
- **Pydantic**: a runtime data-validation library built on type annotations; unlike mypy (static analysis), Pydantic validates data *when the code actually runs* — well suited to validating inputs at an API boundary (Ch 11). Define a `BaseModel` subclass with typed fields (required, or `Optional[...]` with defaults); instantiating it with bad data raises a `ValidationError`.

## Mental Models
- **Tests as a safety net, not a chore**: tests let you make changes (bug fixes, refactors, library upgrades) without relying on memory or re-manually-checking everything — confidence that regressions will be caught automatically.
- **Tests as executable documentation**: a passing test with clear "arrange" inputs is simultaneously a concrete example of how the function is meant to be used, aiding readability (connects to Ch 9).
- **Future-proofing against an unstable world**: code can break not because *you* changed anything, but because a library updated, an OS changed, or an upstream dependency's behavior shifted — tests catch these externally-triggered breaks.
- **"You only need to test your own code"**: imported libraries/dependencies are assumed to carry their own test suites; testing them again is wasted effort and not your responsibility.
- **Test data properties, not exact values, for data validation**: because real-world data shifts over a project's lifetime, validation should check *shape/type/range/distribution* rather than pinning to specific values that will inevitably become stale.
- **ML testing shifts from "is the output right?" to "is the process behaving sanely?"**: since training is stochastic and the "correct" model is unknowable in advance, tests check proxies — weights changed, loss decreased, shapes match, known examples predict correctly — rather than exact model parameters.

## Anti-patterns
- **Skipping tests on code that will be reused or depended on**: fine for pure one-off exploration, but risky once code is shared, deployed, or built upon.
- **Running tests manually instead of via a framework**: wastes time and is error-prone at scale — always use Pytest (or equivalent) rather than invoking test functions by hand.
- **Testing exact values for data that will change over time**: pinning assertions to specific real-world data values (rather than properties/ranges) makes data-validation tests brittle as data evolves.
- **Testing third-party/library code**: not your responsibility and a waste of effort — trust the dependency's own test suite.
- **Trying to test exact ML model outputs/weights**: impossible given training randomness — test properties of the training process and specific known-case predictions instead.
- **Treating 100% test coverage as proof of correctness**: coverage only shows which lines executed, not whether the assertions actually validate correct behavior.

## Code Examples
```python
# A basic unit test: Arrange, Act, Assert
from ch07_functions import weighted_mean

def test_weighted_mean():
    list_a = [1, 2, 4]
    list_b = [1, 2, 4]

    result = weighted_mean(list_a, list_b)

    assert result == 3
```
- **What it demonstrates**: the Arrange/Act/Assert pattern applied to a simple function; test file is separate from the implementation and named `test_*.py`.

```python
# Testing edge cases: empty lists and wrong types
def test_weighted_mean():
    result = weighted_mean([1, 2, 4], [1, 2, 4])
    assert result == 3

    empty_list_result = weighted_mean([], [])
    assert not empty_list_result

    wrong_types_result = weighted_mean(['one', 2, 4], [1, 2, 4])
    assert not wrong_types_result

# Updated function to satisfy the new test:
def weighted_mean(num_list, weights):
    if not (num_list or weights):
        return None
    running_total = 0
    for i in range(len(num_list)):
        running_total += (num_list[i] * weights[i])
    return (running_total / sum(weights))
```
- **What it demonstrates**: tests drive implementation changes to handle edge cases gracefully (returning `None` instead of crashing).

```bash
# Running tests with Pytest
$ pip install pytest
$ pytest test_weighted_mean.py
# or, to run all discovered tests in a folder:
$ pytest
```
```
# Failure output example:
> assert result == 3
E assert 2.3333333333333335 == 3
FAILED test_weighted_mean.py::test_weighted_mean
```
- **What it demonstrates**: Pytest auto-discovers and runs tests, printing a clear diff when an assertion fails.

```python
# Integration test combining two functions
from ch07_functions import process_sdg_data, fit_trendline

def test_processing_trendline():
    df = process_sdg_data(
        "../data/SG_GEN_PARL.xlsx",
        ["Goal", "Target", "Indicator", "SeriesCode", "SeriesDescription",
         "GeoAreaCode", "Reporting Type", "Sex", "Units"],
    )
    timestamps = [int(i) for i in df.index.tolist()]
    uk_parl = df["United Kingdom of Great Britain and Northern Ireland"].tolist()

    slope, r_squared = fit_trendline(timestamps, uk_parl)

    assert slope == 0.836
    assert r_squared == 0.868
```
- **What it demonstrates**: testing the combined behavior of a data-processing function and a modeling function chained together, using a real representative dataset.

```python
# Pandera: schema-based DataFrame validation
import pandera as pa

schema = pa.DataFrameSchema({
    "Year": pa.Column(int, checks=pa.Check.in_range(2000, 2023)),
    "India": pa.Column(float, checks=pa.Check.in_range(0, 100)),
    "United States of America": pa.Column(float, checks=pa.Check.in_range(0, 100))
})
schema(df)  # raises if columns/types/ranges don't match
```
- **What it demonstrates**: declarative, reusable validation of column presence, type, and acceptable value ranges.

```python
# Pydantic: runtime data validation via type annotations
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CountryData(BaseModel):
    country_name: str
    population: int
    literacy_rate_2020: Optional[float]
    timestamp: Optional[datetime] = None

sample_data_correct = {
    'country_name': 'India', 'population': 1417242151,
    'literacy_rate_2020': 79.43, 'timestamp': datetime.now()
}
india = CountryData(**sample_data_correct)   # succeeds

sample_data_incorrect = {
    'country_name': 'United States', 'population': None,
    'literacy_rate_2020': None, 'timestamp': None
}
united_states = CountryData(**sample_data_incorrect)
# ValidationError: population — Input should be a valid integer [input_value=None]
```
- **What it demonstrates**: validation happens at runtime (object construction), raising a clear `ValidationError` naming the offending field when required data is missing/wrong-typed.

## Reference Tables
| Test type | What it checks | Data science relevance |
|---|---|---|
| Unit | A single function/class behaves correctly | Primary day-to-day tool |
| Integration | Multiple units work together correctly | Pipelines (process → model) |
| Acceptance | System meets user expectations | Usually a separate QA/product function |
| Load | System works under increased data/users | Production scaling concerns |
| Security | System resists attacks | Ch 13 |
| Usability | System is intuitive | Usually a separate UX function |

| ML pipeline step | Testing approach |
|---|---|
| Data exploration | No tests needed unless code will be reused |
| Feature engineering | Unit tests + data validation |
| Model training | Check weights changed, loss decreased, correct I/O shapes; for small models, confirm a model artifact is produced |
| Model evaluation | A form of test itself; can enforce a minimum accuracy threshold |
| Model inference | Unit/integration tests on fake/edge-case inputs; check specific known examples predict correctly/within range |
| Model monitoring | Often external tooling; add unit tests for custom monitoring code |

## Worked Example
The chapter's running integration-test example ties `process_sdg_data` (reads an Excel file, drops metadata columns, reindexes by country) to `fit_trendline` (fits a regression via `scipy.stats.linregress`). The test loads the real UN SDG Excel file, extracts the UK's "women in parliament" time series and year index, feeds both into `fit_trendline`, and asserts the resulting slope (0.836) and R² (0.868) match known values for that exact dataset — demonstrating that integration tests can use a real, representative dataset (vs. a unit test's small/fake data) to confirm that multiple functions compose correctly end to end.

## Key Takeaways
1. Tests give confidence that code behaves correctly now and continues to as code, data, dependencies, or environments change — critical once code will be reused, modified, or depended upon by others.
2. Structure tests as Arrange / Act / Assert / Cleanup; start with expected "happy path" inputs, then add edge cases (empty inputs, wrong types) as you discover them.
3. Use a test framework (Pytest is standard) rather than running tests manually; files named `test_*.py`, functions named `test_*`.
4. Unit tests should be small, fast, and deterministic; integration tests check that multiple units compose correctly, often with more realistic data.
5. Data validation (Pandera for DataFrame schemas, Pydantic for runtime type-based validation) is a distinct but related discipline: validate the data's shape/type/range/distribution, not code logic, since data changes over a project's lifetime.
6. Only test your own code — trust that imported libraries have their own test coverage.
7. ML code needs property-based testing (weights changed, loss decreased, correct shapes, known examples predict correctly) rather than exact-output testing, because training is stochastic and the "correct" model can't be known in advance.

## Connects To
- **Ch 1**: Robustness pillar — testing is one of its two components (with errors/logging).
- **Ch 2 / Ch 4**: `weighted_mean` and `fit_trendline` functions reused here as running test examples.
- **Ch 5**: Tests as a debugging tool — confirming whether a code change produces the expected result; regression tests written after fixing a bug.
- **Ch 6**: Type checking (mypy) as a faster alternative to writing type-mismatch tests; Pydantic reuses type annotations for runtime validation.
- **Ch 8**: Refactoring relies on tests to confirm behavior is preserved.
- **Ch 10**: Running tests before committing to version control.
- **Ch 11**: Pydantic validation at API boundaries.
- **Ch 12**: Automating test runs (CI).
- **Ch 13**: Security testing.
- **External**: *Robust Python* by Patrick Viafore (six test types); *Expert Python Programming* by Jaworski & Ziadé (TDD); *Building Machine Learning Pipelines* by Nelson & Hapke (data validation in ML pipelines); Faker and Hypothesis libraries for generating fake/property-based test data; Jeremy Jordan's article on ML testing.
