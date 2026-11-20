# Cheatsheet

## Architecture style decision
- Need broad consumer compatibility, simple resource CRUD → **REST** (default choice, ~86% adoption).
- Consumers need to pick exactly which fields come back, want fewer round trips → **GraphQL**.
- Internal microservice-to-microservice, need streaming, extreme performance (e.g., LLM serving) → **gRPC**.
- Unsure / no explicit consumer demand for GraphQL yet → stay on REST; add GraphQL only when users ask for it.

## "Is this API data-scientist-friendly?" checklist
- Returns JSON (not XML)? 
- Has a published SDK?
- Includes standard external identifiers (e.g., GSIS ID) for joining?
- Enforces strict data types (OAS-backed)?
- Offers a bulk download (CSV/Parquet)?
- Supports query-by-last-changed-date (delta queries)?
- Missing 2+ of these → data scientists will find the API frustrating to use at scale.

## "Is this API AI/agent-ready?" checklist
- Result size limited (trimmed fields, nested collections excluded, pagination/filters present)?
- Schemas consistent and reused, published in OAS?
- Dedicated summary/count endpoint exists (not relying on the model to infer counts)?
- Search endpoint accepts names/language, not just numeric IDs?
- Every endpoint has a clear `summary`, `description`, unique `operationId`?
- `servers` field set in the OAS (required for external tools like ChatGPT custom actions to call the live API)?
- Missing several → expect erratic/unreliable agent behavior; don't expose to autonomous agents without safeguards.

## HTTP status code triage
| Code range | Meaning | Typical action |
|---|---|---|
| 2XX | Success | Proceed |
| 3XX | Redirect | 301/308: update your stored URL; 302: keep using same URL |
| 4XX | Client error | Don't retry blindly; fix the request (400/401/404) |
| 5XX | Server error | Retry with backoff (503 especially); don't hammer |

## Retry decision rule
- Transient network/server error (`RequestError`, `HTTPStatusError`, 5XX) → retry with **exponential backoff + jitter**, capped at a max total time.
- Client error (4XX) → do not blindly retry; the request itself is wrong.
- Many clients could be retrying simultaneously after an outage → jitter is mandatory, not optional (plain backoff still clusters retries).

## Pipeline source selection
| Need | Source |
|---|---|
| Incremental/delta updates, moderate data volume | REST API |
| Very large one-time/bulk datasets | Bulk file (CSV/Parquet) |
| Near-real-time continuous updates | Streaming (Kafka, Kinesis) |
| Async transactional handoff | Message queue (RabbitMQ, SQS) |
| Internal, same-org data sharing | Direct DB connection |

## Full-load vs. delta-update tell
- First time loading a dataset, or API has no change-tracking → full/bulk load.
- Recurring schedule after initial load, API supports `last_changed_date` filter → delta update only. Reloading everything every run is a smell — check for a delta parameter first.

## SDK completeness smell test
- Does the SDK cover every endpoint the API exposes? If not, consumers will fall back to raw HTTP calls, defeating the SDK's purpose — treat partial coverage as a to-do, not acceptable scope.
- Does every public SDK method have a docstring and a sane default (so `pip install` + zero-config "just works")? If not, DX will suffer.

## Streamlit session-state smell test
- Is the same API endpoint being called on more than one page? → cache the result in `st.session_state` on first fetch instead.

## ML deployment readiness
- Model evaluated for accuracy/fairness? If not — it's a draft, not production-ready; say so explicitly.
- Model converted to a portable serving format (ONNX) rather than coupled to the training framework at serving time?
- Does the use case need a single point estimate or a range (quantile models)? Bidding/decision-under-uncertainty scenarios usually want a range.
- Is there a model card / system card documenting purpose, risks, limitations?

## Agentic AI safety gate
Before letting an LLM call an API/endpoint autonomously, ask:
1. Is it read-only / low-risk (like a public fantasy-sports API). If not — require human approval before execution.
2. Are permissions on the calling system restricted to the minimum needed?
3. Is there logging/review of what the agent actually did?
4. Is the system in a "high-stakes situation" per the model provider's own warnings? If yes, do not let the LLM act alone.

## LangGraph vs. ChatGPT custom action — which to use
| Signal | Choose |
|---|---|
| Need fine-grained control over exactly which calls/params are possible | LangGraph (hand-written toolkit) |
| Want fastest possible setup, OAS is already well-documented | ChatGPT custom action (import OAS) |
| Need the agent embedded in your own app/pipeline | LangGraph |
| Need a conversational assistant end users chat with directly | Custom GPT |

## Credential/cost safety defaults
- New paid API key (Anthropic, AWS, etc.) → set a small spend cap, disable auto-reload, rotate immediately if exposure suspected.
- Never commit credentials to source control; use environment variables / secrets managers / Codespaces secrets.
- Cloud deployment → set budget alerts before provisioning, delete unused resources promptly.

## Red flags in code review (this book's anti-patterns)
- Raw string-interpolated SQL anywhere → SQL injection risk; require parameterized queries.
- API client logic inlined throughout notebook/app code → pull into a separate client/SDK module.
- No `last_changed_date` / delta support on a frequently-polled list endpoint → will force wasteful full reloads.
- Retry loop with no backoff or no jitter → risk of self-inflicted DDOS on the called service.
- LLM-driven agent given write/high-risk endpoints with no human-approval gate → policy violation per most model providers' own guidance.
