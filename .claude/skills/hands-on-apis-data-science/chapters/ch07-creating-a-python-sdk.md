# Chapter 7: Batteries Included: Creating a Python SDK

## Core Idea
A well-built SDK wraps an API in idiomatic, batteries-included Python code — sane defaults, data validation, retry/backoff, logging, bulk downloads, and documentation — removing friction for data scientists who "just want the data," and is the capstone of the Part I portfolio project.

## Frameworks Introduced
- **Minimum Viable SDK → Feature-Rich SDK progression**: build and verify packaging before adding capability.
  - When to use: starting any new SDK project.
  - How: first ship the smallest possible working package (one class, one method, installable via `pip3 install -e .`) to validate the packaging mechanics (`pyproject.toml`, `src/<pkg>/__init__.py`), then layer in config, validation, retries, logging, and bulk downloads incrementally.
- **Sane Defaults with Override Pattern**: make the 80% case zero-config, the 20% case configurable.
  - When to use: any SDK configuration surface (base URL, retry behavior, file formats).
  - How: a dedicated config class (`SWCConfig`) reads defaults from environment variables (`os.getenv`, `.env` via `python-dotenv`) but accepts constructor parameters that override them; raise an explicit error if a required value (e.g., base URL) is missing from both sources.
- **Exponential Backoff with Jitter**: resilient retry strategy that avoids synchronized retry storms.
  - When to use: any SDK/client making network calls to a service that may transiently fail.
  - How: on `RequestError`/`HTTPStatusError`, retry with wait time doubling each attempt (1s, 2s, 4s, 8s...) up to a max total time, plus a randomized jitter added to each wait so that many clients retrying after a simultaneous outage don't all hit the server at the same instants.
- **Centralized call_api() wrapper**: single chokepoint for HTTP mechanics, error handling, and logging.
  - When to use: any SDK with multiple endpoint-wrapping methods.
  - How: one internal method takes an endpoint + params, strips empty params, uses `httpx.Client` as a context manager, logs at DEBUG on request/response and ERROR on exceptions, and re-raises exceptions so decorators (like backoff) above it can act on them; every public endpoint method calls through this one function.
- **Endpoint Method + Pydantic Validation Pattern**: hide API complexity behind simple typed function calls.
  - When to use: wrapping any REST endpoint in an SDK.
  - How: each public method (`list_leagues`, `get_health_check`, ...) builds a params dict, calls `call_api()`, and converts the raw JSON into validated domain objects via the same Pydantic schemas the API itself was built with, e.g. `[League(**league) for league in response.json()]` (list comprehension + dict-unpacking).

## Key Concepts
- **SDK (Software Development Kit)**: in this book's usage, primarily "a custom software library that acts as a wrapper for your API," letting consumers call it in their native language without handling raw HTTP themselves.
- **pyproject.toml**: the modern Python packaging config file; declares build backend (`setuptools`), project metadata (name, version, description, author, Python version, classifiers), and dependencies — replacing the API project's `requirements.txt` for the SDK.
- **Editable install (`pip3 install -e .`)**: installs the local package in a mode where code edits take effect immediately without reinstalling — used during SDK development.
- **PEP 8 / Pythonic conventions**: the Python community's style standard (4-space indents, `UpperCamelCase` classes, `lowercase_with_underscores` functions); "idiomatic" SDK code follows these plus related PEPs for list comprehensions, context managers, docstrings, build system requirements, and type hints.
- **Context manager (`with ... as`)**: a Python construct (used via `httpx.Client(...)`) that initializes a resource, runs code using it, then guarantees cleanup — central to `call_api()`'s HTTP client handling.
- **Dict/list comprehension**: Pythonic, loop-free syntax for transforming collections — used both to strip `None` params and to build filename dictionaries and Pydantic object lists.
- **Unpacking operator (`**`)**: `League(**league)` expands a dict's key-value pairs into keyword arguments for a constructor, letting Pydantic validate each field as the object is built.
- **Python logging levels**: `DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL`; SDKs should log meaningfully at each tier instead of being an opaque black box.
- **Bulk file formats (CSV vs. Parquet)**: CSV is uncompressed, plain-text, universally readable (pandas, Excel); Parquet is a columnar, compressed binary format optimized for analytical workloads; the SDK supports both via a `bulk_file_format` config option.
- **Backoff decorator (`backoff.on_exception`)**: a Python library decorator that wraps a function with retry logic matching specified exception types, a wait generator (`backoff.expo`), a max total time, and jitter.

## Mental Models
- **"Make the right things easy and the wrong things hard"** (Kathy Sierra epigraph): the SDK's entire purpose is to make correct, resilient, well-typed API usage the path of least resistance for consumers.
- **SDKs as a business lever, not just a convenience**: per Simon Yu (Speakeasy), SDKs reduce "time to hello world," unblock revenue (enterprise buyers often won't pay until they're integrated), and dramatically cut 1:1 integration support costs for API producers.
- **SDK layering mirrors API layering**: the SDK reuses the exact same Pydantic `schemas.py` the API was built with, so validation logic is defined once and shared across producer and consumer code — "deceptively simple," but it's the same pattern (DTOs) carried across the API/SDK boundary.
- **The "DDOS from your own users" trap**: naive retry-on-failure at scale from many clients can itself take down a recovering service; exponential backoff alone still clusters retries — jitter is required to actually desynchronize them.

## Anti-patterns
- **Naive immediate/constant retries on failure**: equivalent to "clicking Refresh continually," which can overwhelm a struggling or recovering API with load from many simultaneous clients.
- **Plain exponential backoff without jitter**: because an outage affects many clients at once, their retry clocks start together and they still cluster at the same retry intervals (1s, 2s, 4s...) even with backoff — visualized in the book's Figure 7-3.
- **Skipping bulk download support**: as the Robin Linacre epigraph puts it, data scientists are frustrated by having to "learn how to query an endpoint, think about data types" just to get a full dataset — an SDK should offer direct bulk file access.
- **No logging / "black box" SDK**: without meaningful log levels, users can't diagnose failures or understand what the SDK is doing under the hood.
- **Incomplete API coverage in the SDK**: if the SDK doesn't support every endpoint/parameter the API offers, users are forced to fall back to raw API calls for some tasks, defeating the SDK's purpose.
- **Supporting only one language**: SDKs should "go where the users are" (Zan Markan) — for data science audiences that means Python (and sometimes R), not necessarily the API's own implementation language.

## Code Examples
```python
# swc_config.py — sane defaults with environment override and validation
class SWCConfig:
    def __init__(self, swc_base_url: str = None, backoff: bool = True,
                 backoff_max_time: int = 30, bulk_file_format: str = "csv"):
        self.swc_base_url = swc_base_url or os.getenv("SWC_API_BASE_URL")
        if not self.swc_base_url:
            raise ValueError("Base URL is required. Set SWC_API_BASE_URL environment variable.")
        self.swc_backoff = backoff
        self.swc_backoff_max_time = backoff_max_time
        self.swc_bulk_file_format = bulk_file_format
```
- **What it demonstrates**: constructor-parameter-or-environment-variable fallback for the base URL, with a hard failure if neither is supplied.

```python
# swc_client.py — conditional backoff decoration and call_api wrapper
if self.backoff:
    self.call_api = backoff.on_exception(
        wait_gen=backoff.expo,
        exception=(httpx.RequestError, httpx.HTTPStatusError),
        max_time=self.backoff_max_time,
        jitter=backoff.random_jitter,
    )(self.call_api)

def call_api(self, api_endpoint: str, api_params: dict = None) -> httpx.Response:
    if api_params:
        api_params = {key: val for key, val in api_params.items() if val is not None}
    try:
        with httpx.Client(base_url=self.swc_base_url) as client:
            response = client.get(api_endpoint, params=api_params)
            return response
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP status error occurred: {e.response.status_code} {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Request error occurred: {str(e)}")
        raise
```
- **What it demonstrates**: conditionally wrapping a method with exponential-backoff-with-jitter at construction time, and a centralized API-call method with param cleanup, a context-managed HTTP client, and exception logging/re-raising.

```python
# swc_client.py — endpoint method with Pydantic validation
def list_leagues(self, skip: int = 0, limit: int = 100,
                  minimum_last_changed_date: str = None,
                  league_name: str = None) -> List[League]:
    params = {"skip": skip, "limit": limit,
              "minimum_last_changed_date": minimum_last_changed_date,
              "league_name": league_name}
    response = self.call_api(self.LIST_LEAGUES_ENDPOINT, params)
    return [League(**league) for league in response.json()]
```
- **What it demonstrates**: a public SDK method hiding endpoint names/params behind a typed function signature and validating/converting raw JSON into Pydantic objects via list comprehension + unpacking.

```python
# swc_client.py — bulk file download (binary, not JSON)
def get_bulk_player_file(self) -> bytes:
    player_file_path = self.BULK_FILE_BASE_URL + self.BULK_FILE_NAMES["players"]
    response = httpx.get(player_file_path, follow_redirects=True)
    if response.status_code == 200:
        return response.content
```
- **What it demonstrates**: a separate, simpler download path (direct `httpx.get`, not through `call_api`) for retrieving a whole CSV/Parquet file from a GitHub-hosted bulk directory.

```python
# tests/test_swcpy.py
def test_bulk_player_file_parquet():
    config = SWCConfig(bulk_file_format="parquet")
    client = SWCClient(config)
    player_file_parquet = client.get_bulk_player_file()
    player_table = pq.read_table(BytesIO(player_file_parquet))
    player_df = player_table.to_pandas()
    assert len(player_df) == 1018
```
- **What it demonstrates**: testing the SDK against a live running API instance (in a second terminal), including reading a returned Parquet byte stream with PyArrow/pandas.

## Reference Tables
**Table 7-1. Python logging levels**

| Level | Meaning |
|---|---|
| `DEBUG` | Detailed info, typically only useful for diagnosing a problem |
| `INFO` | Confirmation things are working as expected |
| `WARNING` | Something unexpected happened, or a near-future problem is possible; software still works |
| `ERROR` | A more serious problem prevented some function from completing |
| `CRITICAL` | A serious error; the program itself may be unable to continue |

**Expert tips implemented in this SDK (checklist)**

| Tip | Implementation |
|---|---|
| Easy to install | `pyproject.toml` + pip install directly from GitHub subdirectory (or PyPI) |
| Consistent & idiomatic | PEP 8 style, `get_`/`list_` naming convention, comprehensions, context managers |
| Sane defaults | `SWCConfig` reads env vars, has defaults, allows overrides |
| Rich functionality | Pydantic validation, exponential backoff+jitter, bulk downloads |
| Logging | Python `logging` module at DEBUG/ERROR levels |
| Hides API complexity | Endpoint methods (`list_leagues`, etc.) instead of raw HTTP calls |
| Bulk downloads | CSV and Parquet files for every table |
| Documentation | Docstrings + README.md with install & usage examples |

## Worked Example
The chapter builds `swcpy`, the SDK for the SWC Fantasy Football API, in stages: (1) a minimum viable package (`pyproject.toml`, `src/swcpy/__init__.py`, a bare `SWCClient` with only `get_health_check`), verified via `pip3 install -e .`; (2) `SWCConfig` added for sane defaults (base URL from env/`.env`, backoff flags, bulk format), loaded via `python-dotenv`; (3) the client rebuilt around a shared `call_api()` method with logging and backoff-with-jitter (using the `backoff` library), plus `list_leagues()` returning validated `League` Pydantic objects reused directly from the API's own `schemas.py`; (4) bulk-file support (`get_bulk_player_file`) pulling CSV/Parquet files hosted in the repo's `bulk/` folder; (5) a `README.md` and docstrings documenting install (`pip install swcpy@git+https://...#subdirectory=sdk`) and usage; (6) `pytest` tests run against a live second-terminal instance of the API, validating health check, league list, and Parquet bulk download (1,018 player records). The chapter explicitly leaves `get_league_by_id`, `get_counts`, `list_teams`, `list_players`, `get_player_by_id`, `list_performances`, and four more bulk-file functions as an exercise, following the same patterns. Finally, the portfolio project is "finished" for Part I by creating a `learning-branch` backup, then flattening the final API (Ch. 6) and SDK (Ch. 7) code into the repo root and deleting the chapter-by-chapter scaffolding.

## Key Takeaways
1. An SDK's core value is removing friction: users shouldn't need to understand HTTP, authentication, versioning, or endpoint names to use your API effectively.
2. Build a minimum viable SDK first to validate packaging (`pyproject.toml`, `src` layout, editable install) before adding features.
3. Reuse your API's own Pydantic schemas in the SDK so both producer and consumer validate data the same way with no duplicated logic.
4. Combine exponential backoff with random jitter for retries — backoff alone still clusters retries across many simultaneously-failing clients.
5. Centralize HTTP mechanics (param cleanup, client lifecycle, exception handling/logging) in one internal method that every public endpoint method calls through.
6. Support bulk downloads (CSV and Parquet) directly from the SDK, since full-dataset access is a core data-scientist need (Chapter 1's "bulk download" design tip, now implemented client-side).
7. Document the SDK with docstrings (for IDE hints and AI coding assistants) and a README with install/usage examples — documentation is as essential for an SDK as for the API itself.
8. Aim for full endpoint parity between the API and the SDK; an incomplete SDK forces users back to raw API calls, undermining its purpose.

## Connects To
- **Ch 1**: Implements "provide an SDK" and "support bulk downloads" design recommendations from the original data-scientist-friendly API checklist.
- **Ch 4**: Reuses the exact `schemas.py` Pydantic classes built for the API as the SDK's own validation layer.
- **Ch 5**: SDKs are framed as a core documentation/DX feature, continuing Chapter 5's developer-experience focus.
- **Ch 6**: The SDK is tested against the deployed/running API from that chapter; bulk files are hosted in the same GitHub repository structure.
- **Ch 9–11 (Part II)**: This SDK (or direct API calls) becomes the consumption layer for analytics, pipelines, and Streamlit apps.
- **External**: PEP 8, PEP 202, PEP 257, PEP 343, PEP 484, PEP 518; AWS Architecture Blog's "Exponential Backoff And Jitter"; SDKs.io (APIMatic); Speakeasy's "How to Build a Best in Class Python SDK"; Apache Parquet project documentation.
