---
name: data-quality-fundamentals
description: "Knowledge base from \"Data Quality Fundamentals\" by Barr Moses, Lior Gavish, and Molly Vorwerck. Use when building reliable data systems — data quality building blocks, data collection/cleaning/testing, monitoring and anomaly detection, data reliability architecture, fixing data quality issues at scale, data lineage, democratizing data quality, or data observability case studies."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Quality Fundamentals
**Authors**: Barr Moses, Lior Gavish, Molly Vorwerck | **Pages**: ~311 | **Chapters**: 10 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch04`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Data Downtime** — periods when data is missing, inaccurate, or erroneous, manifesting as stale dashboards and bad decisions. The book's central metric, modeled directly on software "uptime/downtime." Formula: `DDT = N × (TTD + TTR)` (incidents × [time to detection + time to resolution]).

**Five Pillars of Data Observability** — the complete measurement framework for data health: **Freshness** (how recent?), **Distribution** (values in expected range/format?), **Volume** (did all data arrive?), **Schema** (what changed, by whom?), **Lineage** (upstream sources / downstream consumers?). Always check all five; no single pillar is sufficient alone.

**Known Unknowns vs. Unknown Unknowns** — known unknowns are predictable issues (nulls, expected freshness, scheduled schema changes) coverable by *testing*; unknown unknowns evade testing and require *monitoring/anomaly detection*. Testing alone covers only ~20% of real-world issues, per the authors' interviews with 100+ data teams.

**SLA / SLO / SLI** — borrowed directly from SRE. SLA = the customer promise + consequence of failure. SLO = the target value for an SLI. SLI = the actual measured number (e.g., hours since last table update). Setting these requires 3 steps: Define (stakeholder interviews + baseline) → Measure (choose SLIs) → Track (set SLO ranges + dashboards).

**Data Reliability Life Cycle** — the data analogue of the DevOps life cycle: **Detect** → **Respond** → **Root Cause Analysis** → **Resolve** → **Prevent**. Treat Detect/Resolve as reactive, Prevent as proactive. Anomaly detection is "a tool, not a silver bullet" — it's only the Detect stage.

**Five-Step Root Cause Analysis** — (1) Look at your lineage (find the most upstream broken node). (2) Look at the code (what logic/query produced it, recent changes?). (3) Look at your data (slice by segment/time/schema). (4) Look at your operational environment (ETL/ELT logs, infra, permissions). (5) Leverage your peers (owners, past incidents).

**Precision / Recall / Fβ** — Precision = TP/(TP+FP) ("of my alerts, how many were real?"); Recall = TP/(TP+FN) ("of real issues, how many did I catch?"). Never optimize for raw "accuracy" on rare-event problems (a detector that always says "normal" can show 99%+ accuracy while being useless). Choose Fβ by cost asymmetry: β>1 favors recall (safety-critical), β<1 favors precision (avoid alert fatigue).

**Data Mesh** (Zhamak Dehghani) — a domain-oriented, self-serve data architecture with federated governance; the data-platform analogue of microservices. Four defining elements: (1) distribute data ownership to domains, (2) give domains long-term accountability + product thinking, (3) empower with self-serve infrastructure, (4) address governance via a federated model. NOT the same as data virtualization, and not buyable as a single product.

**Data Reliability Maturity Curve** — four stages: **Reactive** (firefighting only) → **Proactive** (manual QA + alerts) → **Automated** (scheduled validation + dashboards) → **Scalable** (DevOps-style staging, reusable validation, anomaly detection everywhere).

**Treating Data as a Product** — apply product management discipline (stakeholder alignment, data storytelling, scoping docs/sprints, self-serve tooling, KPI tracking) to data pipelines and platforms, not just to application code. Requires reliability/observability, scalability, extensibility, usability, security/compliance, and release discipline, same as any SaaS product.

**Data Certification** (7 steps) — build observability → assign owners → define "good data" KPIs per pillar → set SLAs/SLOs/SLIs → build communication/incident processes → tag/certify (decentralized) → train the team/consumers. Start with your most critical, highest-traffic assets, not a full retroactive sweep.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Why Data Quality Deserves Attention—Now | Data downtime; DataOps; data mesh/decentralized architecture drivers |
| ch02 | Assembling the Building Blocks of a Reliable Data System | Operational vs. analytical data; throughput/latency trade-off; warehouse vs. lake; data discovery |
| ch03 | Collecting, Cleaning, Transforming, and Testing Data | Entrypoint; circuit breaker; ETL vs. ELT; dbt/Great Expectations/Deequ testing |
| ch04 | Monitoring and Anomaly Detection for Your Data Pipelines | Five pillars; known/unknown unknowns; precision/recall/Fβ |
| ch05 | Architecting for Data Reliability | Six-layer platform; SLA/SLO/SLI; TTD/TTR cost formulas |
| ch06 | Fixing Data Quality Issues at Scale | Data reliability life cycle; 5-step RCA; blameless postmortem; incident commander |
| ch07 | Building End-to-End Lineage | Field-level lineage; SELECT vs. non-SELECT lineage; "controlled freedom" |
| ch08 | Democratizing Data Quality | Data-as-a-product; RACI matrix; data certification; maturity curve |
| ch09 | Data Quality in the Real World: Conversations and Case Studies | Data mesh score; metadata-needs-a-use-case; 7 indicators to invest |
| ch10 | Pioneering the Future of Reliable Data Systems | DDT/labor-cost formulas; lakehouse convergence; emerging roles; automation frontier |

## Topic Index
- **Anomaly detection** → ch04, ch06
- **Blameless postmortem** → ch06
- **Circuit breakers** → ch03, ch06
- **Data catalog / data discovery** → ch02, ch08, ch09
- **Data certification** → ch08
- **Data mesh** → ch01, ch09, ch10
- **Data-as-a-product** → ch08
- **Data warehouse vs. lake vs. lakehouse** → ch01, ch02, ch09, ch10
- **ETL/ELT** → ch02, ch03, ch05
- **Incident management / RCA** → ch06
- **KPIs / cost of downtime (TTD/TTR)** → ch05, ch10
- **Lineage (field-level, end-to-end)** → ch02, ch04, ch06, ch07, ch09
- **RACI matrix / data team structure / ownership** → ch08, ch09
- **SLA/SLO/SLI** → ch05, ch06, ch08
- **Testing (dbt, Great Expectations, Deequ)** → ch03, ch05
- **Case studies**: Blinkist (ch05), PagerDuty (ch06), Fox Networks (ch07), Convoy/Uber (ch08), Toast (ch08), Kolibri Games (ch09), Unity Technologies (ch10)

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — reliability/monitoring techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 10 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
