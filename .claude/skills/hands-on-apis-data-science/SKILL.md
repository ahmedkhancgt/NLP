---
name: hands-on-apis-data-science
description: "Knowledge base from \"Hands-On APIs for AI and Data Science\" by Ryan Day. Use when designing, building, documenting, deploying, or consuming APIs for data science/ML workflows — API architecture choices, FastAPI development, databases, cloud deployment, Python SDKs, API consumption in analytics/pipelines/Streamlit apps, ML model APIs, or LangChain/LLM integration with APIs."
---

# Hands-On APIs for AI and Data Science

Knowledge base distilled from Ryan Day's *Hands-On APIs for AI and Data Science* (O'Reilly, 2025). The book builds one running example — the fictional SportsWorldCentral (SWC) fantasy football platform — across three portfolio projects: **Part I** builds a data API (FastAPI + SQLite/SQLAlchemy + Pydantic + Python SDK); **Part II** consumes that API in analytics/pipeline/app contexts (Jupyter, Apache Airflow, Streamlit); **Part III** connects APIs to AI (deployed ML model, LangChain/LangGraph agents, ChatGPT custom actions).

## Core Frameworks

### 1. Designing APIs Data Scientists Will Love (Ch 1)
Data scientists spend most of their time on data prep (38%), reporting/viz (29%), and model training/deployment (27%). Six concrete API design tips serve this: (1) always return JSON; (2) provide an SDK; (3) include standard external identifiers for joining (e.g., GSIS ID); (4) enforce strict data type conformance (OAS-backed); (5) provide bulk downloads (CSV/Parquet); (6) support query-by-last-changed-date (delta queries). Never build an API speculatively — validate demand via support tickets, scraping traffic, surveys, competitor analysis — then filter candidate features through desirability/feasibility/viability, and capture requirements as user stories ("As a [user], I want to [goal], so that [benefit]").

### 2. API Architecture Selection (Ch 2)
Default to **REST** (client/server, stateless, multiple resource endpoints, ~86% industry adoption) unless a specific need justifies otherwise: **GraphQL** for consumer-defined flexible queries over a single endpoint (POST-only, not browser-GET-able); **gRPC** for high-performance internal/streaming use (protocol buffers over HTTP/2), notably in LLM-serving infrastructure. Structure APIs as `api → version → endpoint`, where an endpoint = HTTP verb + URL path. For a read-only API layered on an existing production app, use a **read-replica architecture**: a separate API app reading from a continuously-synced read-only database copy, protecting the primary app's performance.

### 3. Database + ORM Layer (Ch 3)
Use SQLite for prototyping (file-based, fully SQL-capable). Map tables to Python via **SQLAlchemy** ORM models (`Base` subclasses, `Column()` fields, `relationship()` for FK associations — one-to-many via FK + `back_populates`, many-to-many via an association table with a composite PK). Write parameterized **CRUD helper functions** (never raw/interpolated SQL — SQL injection risk) implementing pagination (`skip`/`limit`) and delta filtering (`min_last_changed_date`) from the start. Include nullable external identifiers (GSIS ID) specifically so consumers can join external data. Unit-test with pytest against known row counts.

### 4. FastAPI + Pydantic (Ch 4)
Three-layer separation: SQLAlchemy models (storage) → **Pydantic schemas** (DTO/wire contract, `schemas.py`, using `ConfigDict(from_attributes=True)`) → FastAPI routes (`main.py`). Split entities into `XBase` (lightweight, nested use) and `X` (full, primary use) schemas to control payload depth. Use `Depends()` for per-request resource lifecycle (DB sessions via generator + try/finally). Path parameters identify a specific resource; query parameters filter/paginate. Return standard HTTP status codes (`HTTPException` for 404s, etc.). Test with `TestClient` before running a live server. Version unstable APIs with a `v0` URL prefix.

### 5. API Documentation (Ch 5)
FastAPI auto-generates the **OpenAPI Specification (OAS)** from your Pydantic schemas and route decorators — the highest-leverage documentation work is annotating code, not hand-writing separate docs. Add `description`/`title`/`version` to `FastAPI()` (OAS `info`); add `tags`, `summary`, `description`, `response_description`, `operation_id` per route; wrap params in `Query(default, description=...)`. **Swagger UI** (`/docs`) is interactive (executes real calls); **Redoc** (`/redoc`) is a non-interactive three-column reference — both generated from `/openapi.json`. Core docs features (getting started, endpoint definitions, OAS, terms of service, feedback channel, SDK) come before extras (sandbox, Postman collection). Different consumers need different trust signals: integration-builders want uptime/versioning assurance; data scientists want data-quality/update-frequency commitments.

### 6. Cloud Deployment & Containerization (Ch 6)
Deploying makes a project "real" — enables sharing, feedback, and downstream AI/tool consumption. Match the platform to your goal: **Render** (source-based PaaS, fast/simple) for shipping quickly; **Docker + AWS Lightsail** for learning containerization. A `Dockerfile` captures the exact runtime (base image → working dir → copy deps manifest → install → copy code/data → `CMD` launching Uvicorn directly, not the `fastapi` CLI wrapper). Cost/credential discipline: budgets+alerts, free tiers, short-lived minimal-scope keys, never commit secrets, delete unused resources.

### 7. Python SDK Development (Ch 7)
An SDK's job is to make correct, resilient, well-typed API usage the path of least resistance. Build **minimum viable SDK** first (packaging via `pyproject.toml` + `src` layout, editable install) then layer in: a config class with **sane defaults** overridable via env vars/constructor; a centralized `call_api()` wrapper (context-managed HTTP client, logging, exception re-raising); **exponential backoff with jitter** (plain backoff still clusters retries across simultaneously-failing clients — jitter is required); endpoint methods returning **Pydantic-validated** objects (reusing the API's own schemas); **bulk file** download methods (CSV/Parquet); comprehensive docstrings + README. Aim for full endpoint parity with the API — incomplete SDKs push users back to raw calls.

### 8. Consuming APIs Responsibly (Ch 8)
As a consumer: respect terms of service/rate limits; retry gently with backoff; store credentials in secrets managers/env vars (never in code/repos); validate inputs and outputs both ways; log systematically. Apply **Separation of Concerns** — use a provided SDK, or build your own standalone client module — rather than inlining API calls through analysis code. Four API testing types (Postman): contract, unit, end-to-end, load.

### 9. API Consumption Patterns (Ch 9–11)
**Analytics (Jupyter + pandas)**: build custom metrics with an explicit Question/Theory/Valid-approach/Data-source/Name design; use `pandas.json_normalize()` to flatten nested API JSON; prefer coefficient of variation over raw stdev when comparing across differently-scaled groups. **Pipelines (Apache Airflow)**: DAGs = acyclic, one-direction graphs of tasks backed by operators (`HttpOperator`, `PythonOperator`); full-load + delta-update split; upsert (`INSERT...ON CONFLICT DO UPDATE`) with parameterized SQL for idempotent loads; externalize config into named Connections. **Apps (Streamlit)**: multi-page apps share data via `st.session_state` to avoid redundant API calls; reuse a standalone API client across projects; enrich API data with external datasets joined on a shared identifier (GSIS ID → `nfl_data_py`).

### 10. AI/API Integration (Ch 12–15)
Four overlap patterns: APIs as training data; trained models served via REST (**inference endpoints**); **RAG** (retrieve via API, feed to LLM context, closing the knowledge gap); **agentic** apps (LLM decides which endpoint to call). AI-readiness checklist for APIs: limit result size (token cost + accuracy), consistent OAS-documented schemas, a scoped SDK, clear `operationId`s, dedicated summary/count endpoints, language-based (not just ID-based) search. **ML deployment**: train (scikit-learn) → convert to **ONNX** (portable, framework-independent) → serve via FastAPI + `onnxruntime.InferenceSession`; use multi-quantile regression for prediction ranges, not false-precision point estimates. **LangGraph agents**: cyclic state graphs (`agent` node ↔ `tools` node via a conditional edge) with tools built as `BaseTool` subclasses wrapping SDK calls — the model only *suggests* tool calls; the agent code executes them. **ChatGPT custom actions**: import the OAS directly (`servers` field required) for a low-code equivalent to a LangChain toolkit — rewards earlier OAS documentation investment. Always apply safeguards (permission restriction, human approval, logging) before letting agents act autonomously, especially on non-read-only endpoints.

## Chapter Index

| # | Title | Key topics |
|---|---|---|
| 1 | Creating APIs That Data Scientists Will Love | User research, design tips, user stories |
| 2 | Selecting Your API Architecture | REST/GraphQL/gRPC, tech architecture, Codespaces |
| 3 | Creating Your Database | SQLite, SQLAlchemy ORM, CRUD, pytest |
| 4 | Developing the FastAPI Code | Pydantic schemas, FastAPI routes, DI |
| 5 | Documenting Your API | OAS, Swagger UI, Redoc, DX |
| 6 | Deploying Your API to the Cloud | Render, Docker, AWS Lightsail |
| 7 | Batteries Included: Creating a Python SDK | pyproject.toml, backoff/jitter, bulk files |
| 8 | What Data Scientists Should Know About APIs | HTTP basics, responsible consumption, SoC |
| 9 | Using APIs for Data Analytics | Jupyter, pandas, custom metrics |
| 10 | Using APIs in Data Pipelines | Apache Airflow, DAGs, upsert |
| 11 | Using APIs in Streamlit Data Apps | Multi-page apps, session state, data enrichment |
| 12 | Using APIs with Artificial Intelligence | RAG, agentic apps, AI-readiness design |
| 13 | Deploying a Machine Learning API | CRISP-DM, scikit-learn, ONNX, FastAPI |
| 14 | Using APIs with LangChain | LangGraph agents, toolkits, Claude |
| 15 | Using ChatGPT to Call Your API | Custom GPT, custom actions, OAS import |

## Topic Index

- **Agents (agentic AI)** → Ch 12, 14, 15
- **Airflow / DAGs / orchestration** → Ch 10
- **AI-readiness API design** → Ch 1, 12
- **Backoff/retry/jitter** → Ch 7, 8, 9, 10
- **Bulk downloads (CSV/Parquet)** → Ch 1, 7
- **ChatGPT / custom GPT / custom actions** → Ch 15
- **Cloud deployment (Render, AWS, Docker)** → Ch 6
- **CRISP-DM** → Ch 13
- **Custom metrics** → Ch 9
- **Database design (SQLite/SQLAlchemy)** → Ch 3
- **Delta queries / last-changed-date** → Ch 1, 3, 10
- **Documentation / OAS / Swagger UI / Redoc** → Ch 5, 12, 15
- **FastAPI** → Ch 4, 5, 13
- **GraphQL** → Ch 2, 8, 12
- **gRPC** → Ch 2, 8
- **HTTP verbs/status codes** → Ch 8
- **LangChain / LangGraph** → Ch 14
- **Machine learning deployment / ONNX** → Ch 13
- **Pandas / data wrangling** → Ch 9, 11
- **Pydantic schemas** → Ch 4, 7, 13
- **RAG (Retrieval Augmented Generation)** → Ch 12, 14
- **REST architecture** → Ch 2, 8
- **SDK development (Python)** → Ch 7, 8, 14
- **Separation of Concerns** → Ch 8, 9, 11
- **SQL injection prevention** → Ch 3, 10
- **Streamlit** → Ch 11
- **Testing (pytest)** → Ch 3, 4, 8
- **User stories / user research** → Ch 1
- **Versioning (API)** → Ch 2, 4, 5

## Supporting Files
- `glossary.md` — alphabetical term definitions with chapter references
- `patterns.md` — reusable techniques with when-to-use/how/trade-offs
- `cheatsheet.md` — decision rules, checklists, smell tests
- `chapters/ch01-*.md` through `chapters/ch15-*.md` — full per-chapter study notes

## Scope & Limits
This skill summarizes a single O'Reilly book (2025) built around one running fictional example (SportsWorldCentral). It is a conceptual/methodological reference, not a substitute for reading current framework documentation (FastAPI, SQLAlchemy, LangChain/LangGraph, Airflow, Streamlit, Anthropic/OpenAI APIs all evolve quickly — version numbers cited here reflect the book's writing time, ~2024–2025). Code snippets are illustrative excerpts reconstructed from the book, not verified runnable code — treat them as patterns to adapt, not copy-paste production code. The book's "AI-readiness" and agentic-API guidance (Ch 12, 14, 15) reflects an emerging, fast-moving field; treat its specific tool/library recommendations as a starting point, not a definitive current best practice.
