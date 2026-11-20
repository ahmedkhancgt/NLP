---
name: fundamentals-data-engineering
description: "Knowledge base from \"Fundamentals of Data Engineering\" by Joe Reis and Matt Housley. Use when designing data engineering systems — the data engineering lifecycle, data architecture, technology selection, source systems, storage, ingestion, queries/modeling/transformation, serving data for analytics/ML/reverse ETL, data security and privacy, or the future of data engineering."
---

# Fundamentals of Data Engineering

Knowledge base distilled from Joe Reis and Matt Housley's "Fundamentals of Data Engineering" (O'Reilly, 2022). Use this skill when designing, evaluating, or discussing data pipelines, data architecture, storage systems, ingestion strategies, data modeling, transformation patterns, serving layers, or data security/privacy.

## Core Frameworks

### The Data Engineering Lifecycle (central organizing model)
Five stages — **Generation → Storage → Ingestion → Transformation → Serving** — wrapped by six cross-cutting **undercurrents**: security, data management, DataOps, data architecture, orchestration, software engineering. Storage is not a single stage but a foundation running underneath all others. Every chapter in this skill maps to this lifecycle; use it as the default mental map for any data engineering question.

### Data Maturity Model (3 stages)
**Starting** (generalist engineer, get buy-in, pick minimal architecture, avoid undifferentiated heavy lifting) → **Scaling** (specializing roles, formal DataOps, scalable architecture) → **Leading** (automation, custom tools only where they create competitive advantage, governance tooling). Calibrate ambition to maturity stage — don't build Stage-3 systems in a Stage-1 org.

### Type A vs. Type B Data Engineer
Type A = **Abstraction** (assembles managed/off-the-shelf tools, avoids reinventing wheels). Type B = **Build** (builds custom systems for genuine competitive advantage). Default to Type A; earn Type B only when justified.

### "Good" Data Architecture — 9 Principles
(1) Choose common components wisely, (2) Plan for failure (availability, reliability, RTO, RPO), (3) Architect for scalability (scale up/down/to-zero), (4) Architecture is leadership (mentor, don't bottleneck), (5) Always be architecting, (6) Build loosely coupled systems, (7) Make reversible decisions ("two-way doors"), (8) Prioritize security (zero-trust, shared responsibility), (9) Embrace FinOps. Architecture = the *what/why/when* (strategic); technology = the *how* (tactical) — always architecture first.

### Technology Selection Framework
Evaluate every technology choice across: team size/capability, speed to market, interoperability, cost (TCO/TOCO/FinOps), time horizon (immutable vs. transitory — apply the Lindy effect), location (cloud/on-prem/hybrid/multicloud), build vs. buy, monolith vs. modular, serverless vs. servers, and benchmark skepticism (vendor benchmarks are frequently rigged — validate against real workloads).

### Source System Evaluation
Before ingesting, understand: schema (fixed vs. schemaless, evolution handling), cadence/volume, consistency guarantees, OLTP vs. OLAP nature, CDC availability (batch timestamp vs. continuous log-based), and whether the source is push, pull, or poll. Formalize the relationship with a **data contract** and **SLA/SLO**.

### Storage Decision Model
Raw ingredients (disk/SSD/RAM/network/serialization/compression/caching) → storage systems (block/file/object/streaming) → abstractions (warehouse/lake/lakehouse/platform). Key axes: hot/warm/cold temperature tiering, schema-on-write vs. schema-on-read, separation vs. colocation of compute and storage, single- vs. multi-tenant.

### Ingestion Design Axes
Bounded vs. unbounded ("all data is unbounded until it's bounded"), frequency (batch/micro-batch/streaming), sync vs. async coupling, throughput/reliability/durability trade-offs, payload characteristics (kind/shape/size/schema), and push/pull/poll pattern. Prefer async, event-level coupling over brittle synchronous batch chains.

### Data Modeling Approaches
**Inmon** (top-down, normalized 3NF enterprise warehouse, single source of truth, department marts downstream) · **Kimball** (bottom-up star schema: fact tables + dimension tables, denormalization-friendly, conformed dimensions) · **Data Vault** (hubs/links/satellites, insert-only, source-fidelity-preserving, agile) · **Wide/denormalized tables** (cheap storage + flexible nested schemas, no join overhead, less rigor). Always model at the **lowest practical grain** — you can aggregate up but never de-aggregate.

### Transformation Patterns
ETL vs. ELT (increasingly a micro-level choice, not an org-wide standard) · broadcast join (cheap, one side fits on a node) vs. shuffle hash join (expensive, full repartition) · update patterns under copy-on-write (truncate-and-reload, insert-only, hard/soft delete, upsert/merge) · materialized views, federated queries, data virtualization · streaming windows (session/fixed/sliding) and watermarks for late-arriving data.

### Serving Principles
**Trust is the root consideration** — the most sophisticated architecture fails if stakeholders don't believe the data. Classify serving needs as business analytics (longer-term, batch-friendly), operational analytics (immediate action, near-real-time), or embedded analytics (customer-facing, strict latency/concurrency). Centralize business logic in a **semantic/metrics layer** rather than duplicating it across ETL scripts. Reverse ETL closes the loop by feeding derived data back into source systems — but needs monitoring to prevent runaway feedback loops.

### Security Priority Order: People → Processes → Technology
People are the weakest link — practice "active"/negative-thinking security (imagine attack scenarios proactively; "data cannot leak if it is never collected"). Processes: least privilege, habitual (not theater) security practices, shared responsibility model. Technology: encryption at rest/in transit, patching, monitoring (access/resources/billing/excess permissions), network whitelisting.

### FinOps & Cost Lenses
**TCO** (direct + indirect costs) · **TOCO** (cost of foreclosed alternatives / exit difficulty — "bear traps") · **FinOps** (continuous, business-value-tied cloud spend monitoring, opex > capex given pace of change). Watch for **data gravity** (egress fees locking you into a cloud).

## Chapter Index

| # | Title | Core Topic |
|---|---|---|
| 1 | Data Engineering Described | Field definition, lifecycle intro, data maturity, Type A/B engineers, stakeholder map |
| 2 | The Data Engineering Lifecycle | Five stages + six undercurrents in full detail |
| 3 | Designing Good Data Architecture | Architecture principles, warehouse/lake/lakehouse, Lambda/Kappa, data mesh |
| 4 | Choosing Technologies Across the Lifecycle | TCO/TOCO/FinOps, build vs. buy, monolith vs. modular, serverless vs. servers |
| 5 | Data Generation in Source Systems | OLTP/OLAP, CDC, logs, messages/streams, APIs, data contracts |
| 6 | Storage | Disk/SSD/RAM physics, block/file/object storage, lakehouse, hot/warm/cold |
| 7 | Ingestion | Batch/streaming, sync/async, CDC mechanics, managed connectors, dead-letter queues |
| 8 | Queries, Modeling, and Transformation | SQL internals, normalization, Inmon/Kimball/Data Vault, joins, update patterns, streaming windows |
| 9 | Serving Data for Analytics, ML, and Reverse ETL | Trust, analytics types, self-service, semantic layers, reverse ETL |
| 10 | Security and Privacy | People/process/technology security, least privilege, encryption, monitoring |
| 11 | The Future of Data Engineering | Cloud-scale data OS, "enterprisey" trends, live data stack speculation |

## Topic Index

- **ACID / BASE / consistency** → Ch 5, 6, 8
- **Architecture principles (9)** → Ch 3
- **Batch vs. streaming** → Ch 2, 7, 8
- **Build vs. buy** → Ch 1, 4
- **CDC (Change Data Capture)** → Ch 2, 5, 7, 8
- **Cloud economics / FinOps** → Ch 3, 4
- **Data contracts / SLA/SLO** → Ch 5, 9
- **Data maturity model** → Ch 1
- **Data mesh** → Ch 3, 9
- **Data modeling (Inmon/Kimball/Data Vault)** → Ch 8
- **Egress fees / data gravity** → Ch 4
- **Lambda / Kappa architecture** → Ch 3, 8
- **Lakehouse** → Ch 3, 6
- **Least privilege** → Ch 2, 3, 10
- **Materialized views / federation / virtualization** → Ch 8, 9
- **Normalization (1NF/2NF/3NF)** → Ch 8
- **Object storage** → Ch 6
- **Orchestration** → Ch 2, 4, 7, 9
- **Reverse ETL** → Ch 2, 5, 9
- **Security (people/process/tech)** → Ch 2, 3, 10
- **Self-service BI/ML** → Ch 9
- **Semantic / metrics layer** → Ch 8, 9
- **Serialization formats** → Ch 6, 7
- **Serving (analytics/ML/reverse ETL)** → Ch 9
- **Slowly Changing Dimensions (SCD)** → Ch 8
- **Source system evaluation** → Ch 5
- **Star schema (facts/dimensions)** → Ch 8
- **Streaming windows/watermarks** → Ch 8
- **TCO / TOCO** → Ch 4
- **Technology evaluation (immutable vs. transitory)** → Ch 4
- **Trust (serving)** → Ch 9
- **Type A/B engineers** → Ch 1, 4
- **Zero-trust security** → Ch 3, 10

## Supporting Files

- [glossary.md](glossary.md) — Alphabetical reference of all significant terms with chapter citations.
- [patterns.md](patterns.md) — Reusable techniques/patterns with when-to-use, how, and trade-offs.
- [cheatsheet.md](cheatsheet.md) — Decision rules, trade-off tables, numeric thresholds, and red-flag "tells & smells."
- [chapters/](chapters/) — Full per-chapter breakdowns (ch01–ch11) with frameworks, key concepts, mental models, anti-patterns, code examples, reference tables, worked examples, and cross-references.

## Scope & Limits

This skill is a study/reference distillation of one book published in 2022. It is strong on: data engineering lifecycle structure, architecture trade-off reasoning, storage/ingestion/modeling/transformation/serving mechanics, and security fundamentals. It is weaker on (and should not be treated as authoritative for): specific current-day product capabilities, pricing, or version details for any named vendor/tool (Snowflake, BigQuery, Airflow, dbt, Kafka, etc.) — the book's technology specifics have inevitably dated since 2022, so verify current tool behavior independently. It is also not a substitute for deep dives into any single modeling methodology (Kimball/Inmon/Data Vault), ML/MLOps, or a specific cloud provider's documentation — treat it as the conceptual scaffolding on which to hang more detailed, current research.
