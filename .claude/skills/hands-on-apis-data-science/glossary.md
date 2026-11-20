# Glossary

**Agent** — Software that uses an LLM to decide the control flow of an application; the more autonomously the LLM drives behavior, the more "agentic" the system (Ch 12, 14).

**Agentic application** — A generative AI application in which the LLM decides which API endpoints to call, inferred from OAS files, code, or documentation (Ch 12).

**API endpoint (operation)** — A combination of an HTTP verb and a URL path; the basic addressable unit of a REST API (Ch 2).

**API version** — A set of endpoints kept stable over time so consumers can rely on them (Ch 2).

**Arazzo Specification** — An emerging OpenAPI-companion standard defining sequences of API calls and their dependencies, aimed at making multi-step AI/agent workflows more deterministic (Ch 5, 12).

**Association table** — A table with two foreign keys implementing a many-to-many relationship between two parent tables (Ch 3).

**Backoff decorator** — A Python (`backoff` library) decorator adding exponential-backoff-with-jitter retry logic to a function (Ch 7, 9, 10).

**Batch inference** — Processing a whole group of ML predictions at once, typically via a scheduled job (Ch 13).

**Build context (Docker)** — The set of files available to `docker build`, from which `COPY` statements pull files into the image (Ch 6).

**Checkpointer (LangGraph)** — An object (e.g., `MemorySaver`) that persists agent state across `invoke()` calls sharing a `thread_id`, giving conversational memory (Ch 14).

**Classification / Classifier** — A model type predicting which category a value falls into (Ch 13).

**Coefficient of Variation (CV)** — Standard deviation divided by the mean; a dimensionless relative-variability measure usable across different scales (Ch 9).

**Container image / container** — A static built artifact (image) vs. a running instance of it (container), created by a container runtime like Docker (Ch 6).

**Context manager** — A Python `with ... as` construct that initializes a resource, runs code, then guarantees cleanup (Ch 4, 7).

**CRISP-DM** — Cross-Industry Standard Process for Data Mining; six stages (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment) for organizing ML projects (Ch 13).

**CRUD** — Create, Read, Update, Delete; the standard set of data operations (Ch 3 implements only "R").

**DAG (Directed Acyclic Graph)** — A top-level pipeline structure (Apache Airflow) with steps (nodes) flowing one direction via edges, without loops (Ch 10).

**Delta / deltas** — Records that are new or changed since the last pipeline run, enabling incremental updates (Ch 1, 3, 10).

**Dependency injection (FastAPI)** — Using `Depends()` to supply per-request resources (like a DB session) to route functions, with lifecycle managed via generator functions (Ch 4).

**DTO (Data Transfer Object)** — A pattern defining a wire-format class (Pydantic schema) separate from the backend storage model (SQLAlchemy model), decoupling API contract from database structure (Ch 4).

**Edge (graph theory)** — A one-way (or, in LangGraph, cyclic) flow between nodes (Ch 10, 14).

**Exponential backoff with jitter** — A retry strategy where wait time doubles each attempt and includes randomization, avoiding synchronized retry storms across many clients (Ch 7, 9, 10).

**Fitting** — The process of applying a general ML algorithm to specific training data to produce a specialized model (Ch 13).

**Function-calling model** — An LLM that can examine available tools/functions and suggest their use; it never calls tools directly — the surrounding agent code executes the call (Ch 14, 15).

**GSIS ID** — The NFL's Game Statistics and Information System identifier; included as a nullable external identifier in the SWC API specifically so data scientists can join to other NFL data sources (Ch 1, 3, 11).

**Gradient boosting** — An ensemble ML technique combining multiple decision trees into a model more effective than any individual tree (Ch 13).

**gRPC** — A Google-developed RPC framework using protocol buffers over HTTP/2, supporting fast, two-way streaming; common for ML/LLM-serving infrastructure (Ch 2, 8).

**GraphQL** — A query language and runtime where consumers define exactly which fields they want from a single endpoint via HTTP POST (Ch 2, 8).

**HTTPX** — An async-capable Python HTTP client used throughout the book for API calls and testing (Ch 4, 7, 9).

**Knowledge gap** — The limitation that an LLM only knows what was in its training data; addressed via RAG or tool/API access (Ch 12, 14, 15).

**Last-changed-date query** — An API parameter allowing consumers to fetch only records changed since a given date (Ch 1, 3, 10).

**Model card / System card** — Documentation standards for ML transparency; model cards (Google) document a single model's purpose/risks/biases, system cards (Meta) document a broader multi-model AI system (Ch 13, 14, 15).

**ONNX (Open Neural Network Exchange)** — An open standard ML model format enabling models trained in different frameworks to be run via a common runtime (ONNX Runtime), with cross-platform portability and acceleration (Ch 13).

**Operator (Airflow)** — A predefined task template (e.g., `HttpOperator`, `PythonOperator`) used to define a DAG task (Ch 10).

**ORM (Object-Relational Mapping)** — A library (SQLAlchemy) mapping database rows to Python objects, handling type conversion and preventing SQL injection via prepared statements (Ch 3).

**Pagination (skip/limit)** — Returning data in page-sized chunks via offset (`skip`) and page-size (`limit`) parameters (Ch 1, 3).

**Parquet** — An open-source, column-oriented, compressed binary file format popular for bulk analytical data (Ch 1, 7).

**Pydantic** — A Python data validation library (Rust core) used by FastAPI to validate requests/responses and generate the OpenAPI/JSON Schema (Ch 4).

**RAG (Retrieval Augmented Generation)** — A pattern where an application retrieves data via APIs/other sources and feeds it to an LLM alongside the user prompt, to overcome the knowledge gap (Ch 12).

**Read replica** — A read-only copy of a production database, kept in sync, used to serve API traffic without impacting the primary application's performance (Ch 2, 3).

**Regression / Regressor** — A model type predicting a continuous numeric value (Ch 13).

**REST (Representational State Transfer)** — The dominant API architectural style: stateless client/server communication over multiple resource-based endpoints using standard HTTP verbs (Ch 2, 8).

**SDK (Software Development Kit)** — A custom software library wrapping an API so consumers can call it natively in their language without handling raw HTTP (Ch 7).

**Separation of Concerns (SoC)** — Isolating API-calling logic into its own module/SDK/client rather than embedding it throughout application/analysis code (Ch 8, 9, 11).

**SQL injection** — A vulnerability where malicious input is interpreted as SQL code; mitigated by parameterized/prepared statements (Ch 3, 10).

**Swagger UI / Redoc** — Two bundled, OAS-driven documentation UIs in FastAPI; Swagger UI (`/docs`) is interactive (executes test calls), Redoc (`/redoc`) is a non-interactive three-column reference (Ch 5).

**Time to Hello World (TTHW)** — A DX metric for how quickly a new consumer can get a basic working API call; reducing it is a goal of good documentation and SDKs (Ch 5, 7).

**Token (LLM)** — The unit of text LLM providers charge for processing; larger payloads cost more and can reduce calculation accuracy (Ch 12).

**Tool / Toolkit (LangChain)** — A tool is code giving an agent one capability (e.g., wrapping one API call); a toolkit bundles related tools for an agent (Ch 14).

**Upsert** — Insert-or-update logic (`INSERT ... ON CONFLICT DO UPDATE`), making repeated pipeline loads idempotent (Ch 10).

**User story** — An informal requirement statement ("As a [user], I want to [goal], so that [benefit]") used to capture API consumer needs (Ch 1).

**XCom (Airflow)** — Cross-communication mechanism for passing data between DAG tasks (Ch 10).
