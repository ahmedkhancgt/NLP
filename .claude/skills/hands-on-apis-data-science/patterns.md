# Patterns

## User-Story-Driven API Scoping
**When to use**: Deciding which API products/endpoints to build first out of many possible requests.
**How**: Identify potential users via support tickets, surveys, scraping traffic, competitor analysis. Write user stories ("As a [user], I want to [goal], so that [benefit]"). Filter candidates through three criteria: user desirability, technical feasibility, economic viability.
**Trade-offs**: Takes upfront research time instead of building speculatively, but avoids the "Field of Dreams" trap of building APIs nobody uses.

## Consumer-Centric Architecture Selection
**When to use**: Choosing REST vs. GraphQL vs. gRPC for a new API.
**How**: Default to REST (highest adoption/tooling support). Use GraphQL when consumers need flexible, consumer-defined response shapes and fewer round trips. Use gRPC for high-performance internal/streaming use cases (e.g., LLM-serving infra), not typical external APIs.
**Trade-offs**: REST is simplest and most broadly understood but less flexible; GraphQL reduces over/under-fetching but adds query-language complexity; gRPC is fastest but not browser-callable and uses binary protobufs instead of JSON.

## Read-Replica API Architecture
**When to use**: Building a read-only API on top of an existing production application without degrading its performance.
**How**: Stand up a separate API application reading from a continuously-synced read-only replica database, not the live production database.
**Trade-offs**: Slight data-freshness lag vs. protecting the primary application from API traffic load.

## ORM + CRUD Helper Layer
**When to use**: Any Python app querying a relational database repeatedly and safely.
**How**: Define SQLAlchemy models (`Base` subclasses) mapping tables to Python classes with `relationship()` for foreign keys. Write parameterized CRUD helper functions (one per query need) instead of raw SQL strings.
**Trade-offs**: More boilerplate upfront than raw SQL, but eliminates SQL injection risk, handles type conversion automatically, and is portable across database backends.

## DTO / Schema Separation (Pydantic vs. ORM)
**When to use**: Any API where the wire format should be independent of the database schema.
**How**: Keep SQLAlchemy models (storage) and Pydantic schemas (API contract) as separate classes; use `ConfigDict(from_attributes=True)` to let Pydantic read from ORM objects. Split entities into `XBase` (lightweight/nested use) and `X` (full/primary use) schemas to control payload depth.
**Trade-offs**: Two class hierarchies to maintain instead of one, but decouples internal storage changes from the public API contract and controls over-fetching.

## Delta/Last-Changed-Date Querying
**When to use**: APIs or pipelines needing efficient incremental updates after an initial full load.
**How**: Add a `last_changed_date` column to every table; expose an optional `min_last_changed_date`/`minimum_last_changed_date` filter parameter on list endpoints; pipelines query only records changed since the last run.
**Trade-offs**: Requires disciplined timestamp maintenance on writes, but dramatically reduces pipeline runtime/cost versus full reloads.

## OAS-Driven Documentation Enrichment
**When to use**: Once a baseline FastAPI app exists and default Swagger UI/Redoc output is too generic.
**How**: Add `description`/`title`/`version` to the `FastAPI()` constructor; add `tags`, `summary`, `description`, `response_description`, `operation_id` to each route decorator; wrap query parameters in `Query(default, description=...)`.
**Trade-offs**: More verbose route decorators, but the same work powers human docs, SDK generation, and AI/agent tool-use simultaneously.

## Minimum Viable SDK → Feature-Rich SDK
**When to use**: Starting any new SDK project.
**How**: First ship the smallest installable package (one class, one method) to validate packaging (`pyproject.toml`, `src` layout, editable install). Then layer in a config class (sane defaults + env var overrides), a centralized `call_api()` wrapper (logging, error handling), exponential-backoff-with-jitter, Pydantic-validated endpoint methods, bulk-download methods, and documentation.
**Trade-offs**: More development time than a thin wrapper, but produces an SDK that reduces consumer friction, support burden, and misuse.

## Exponential Backoff with Jitter
**When to use**: Any client/SDK making network calls to a service that may transiently fail.
**How**: On retryable exceptions, retry with wait doubling each attempt up to a max total time, plus randomized jitter so many simultaneously-failing clients don't retry in lockstep.
**Trade-offs**: Slightly higher latency on failure paths and added complexity vs. naive retry, but prevents accidental self-inflicted DDOS on a recovering service.

## Centralized API-Call Wrapper
**When to use**: Any SDK/client with multiple endpoint-wrapping methods.
**How**: One internal function (e.g., `call_api_endpoint()`) handles param cleanup, context-managed HTTP client use, and exception logging/re-raising; every public method calls through it.
**Trade-offs**: An extra layer of indirection, but centralizes retry/logging/error-handling so it doesn't need to be duplicated per endpoint.

## Separation of Concerns for API-Calling Code
**When to use**: Any project (notebook, pipeline, app) that calls an API more than once, or across multiple consuming contexts.
**How**: Use the provider's SDK if available; otherwise write a standalone client module (endpoint constants, logger, backoff-decorated call function) and import it wherever needed, rather than inlining HTTP calls.
**Trade-offs**: Extra file/module to maintain, but enables reuse (demonstrated across notebook → Streamlit app) and consistent error handling.

## Full Load + Delta Update Pipeline
**When to use**: Keeping a local copy of API-sourced data current when the API doesn't support full bulk export, or to minimize ongoing sync cost.
**How**: One-time initial load from a bulk file (or full paginated pull); recurring job queries only changed records (last-changed-date filter) and upserts them into the local store.
**Trade-offs**: Two code paths to maintain (initial vs. incremental) instead of one, but far cheaper/faster for ongoing operation.

## DAG Orchestration (Airflow)
**When to use**: Multi-step pipelines needing scheduling, dependency management, retries, and observability beyond a plain script.
**How**: One Python file per DAG using `@dag`; tasks backed by operators (`HttpOperator`, `PythonOperator`); dependencies declared via `>>`; data passed between tasks via `XCom`; environment config externalized into named Connections.
**Trade-offs**: Operational overhead (running Airflow itself) vs. a simple cron script, but pays off once pipelines are numerous/complex or need robust error handling.

## Upsert for Idempotent Loads
**When to use**: Any recurring load where the same source record may appear across multiple runs.
**How**: `INSERT ... ON CONFLICT(primary_key) DO UPDATE SET ...` with fully parameterized (`?`) values — never string-interpolated.
**Trade-offs**: Slightly more complex SQL than plain insert, but makes repeated pipeline runs safe and injection-resistant.

## Nested-JSON-to-Tidy-DataFrame Flattening
**When to use**: An API returns nested list fields (e.g., `team.weekly_scores`) that need to become individual analyzable rows.
**How**: `pandas.json_normalize(data, record_path, meta)` to explode the nested list while carrying parent identifier columns along.
**Trade-offs**: An extra transform step, but avoids writing manual nested-loop flattening code.

## Multi-Page Streamlit App with Shared Session State
**When to use**: Any Streamlit app with more than one view that shares underlying data.
**How**: One entrypoint file configures `st.set_page_config`, defines `st.Page` objects, builds `st.navigation`; data fetched on one page is cached in `st.session_state` and read (not re-fetched) by subsequent pages.
**Trade-offs**: Slightly more upfront structure than a single-page script, but avoids redundant API calls across pages.

## ML Model Deployment via ONNX + FastAPI
**When to use**: Deploying a trained model (any framework) as a real-time inference REST API.
**How**: Train in native framework (e.g., scikit-learn) → convert to ONNX (`sklearn-onnx`'s `to_onnx()`) → load via `onnxruntime.InferenceSession` in a FastAPI app → define Pydantic input/output schemas → expose a POST `/predict/` endpoint.
**Trade-offs**: An extra conversion step vs. serving the native model directly, but decouples the serving runtime from the training framework and gains portability/acceleration.

## Multi-Quantile Regression for Prediction Ranges
**When to use**: A single point estimate understates decision-relevant uncertainty.
**How**: Train separate models with the same features/algorithm but different quantile targets (e.g., `loss="quantile", alpha=0.1/0.5/0.9`); combine their outputs into one API response.
**Trade-offs**: 3x the training/serving cost of a single model, but communicates a usable range instead of false precision.

## LangGraph Agent with Tool Loop
**When to use**: Building an agent that should reason, decide whether to call tools, and loop until done answering.
**How**: `StateGraph(MessagesState)` with an `agent` node (model call) and a `tools` node (`ToolNode`); a conditional edge (`should_continue`) routes to `tools` if the last message has tool calls, else to `END`; an edge from `tools` back to `agent` closes the loop; bind tools to the model via `model.bind_tools(tools)`.
**Trade-offs**: More moving parts than a single LLM call, but enables genuinely autonomous, multi-step API-calling behavior with developer-controlled routing.

## LangChain Toolkit Wrapping an SDK
**When to use**: Giving an LLM agent a scoped, well-documented set of API capabilities.
**How**: For each capability, define a Pydantic input schema and a `BaseTool` subclass (`name`, `description`, `args_schema`, `_run()` calling the underlying SDK method); bundle into a `BaseToolkit.get_tools()`.
**Trade-offs**: Requires hand-writing tool descriptions carefully (the model relies on them entirely), but gives fine-grained control over what the agent can do and how it decides to do it.

## Custom GPT + Imported OAS Action (Low-Code Agentic API Access)
**When to use**: Giving a ChatGPT-based application API access without writing agent/tool code.
**How**: Add a `servers` entry to the FastAPI OAS pointing at a public base URL; in ChatGPT's GPT Builder, create a custom action and import the OAS file by URL; prune any endpoints that shouldn't be AI-accessible.
**Trade-offs**: Far less code than a LangChain toolkit, but less fine-grained control, and only works because the OAS already has good descriptions — rewards earlier documentation investment (Ch 5/12).
