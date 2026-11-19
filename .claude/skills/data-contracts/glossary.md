# Glossary

**1-10-100 Rule** — Chang & Labovitz's cost-escalation principle: $1 to fix a bug at design time, $10 at QA, $100 post-release; used as shared vocabulary to pitch shift left data to SWEs (Ch 9).

**ACID** — Atomicity, Consistency, Isolation, Durability; transactional database compliance properties. Consistency is an application responsibility, not a database guarantee (Ch 5).

**Alert fatigue** — The result of too many non-actionable monitors, causing all alerts to be ignored by default (Ch 3, Ch 6).

**Change Data Capture (CDC)** — Monitoring CRUD operations on an upstream database and propagating changes downstream in real time (Ch 4, Ch 5, Ch 6).

**Change management** — The discipline of handling change to codebases, business logic, data users, expectations, producers, and consumers; the root problem data contracts solve (Ch 10).

**Clickstream events** — User-interaction records captured by third-party SDKs (Amplitude, Segment) for product analytics (Ch 4).

**Concept drift** — Unforeseeable changes in the underlying data distribution over time; types: sudden, gradual, incremental, recurring (Ch 3).

**Contract coverage detector** — Script that flags data assets present in a contract spec but missing from the live data catalog (Ch 7).

**Contract violation detector** — Script that compares contract spec metadata against catalog metadata at the column level to find mismatches (Ch 7).

**Conway's Law** — "Any organization that designs a system will produce a design whose structure mirrors the organization's communication structure" (Ch 3).

**Data as code** — The view that all data is inseparable from the code that produced/transformed it; understanding data quality requires understanding the generating code (Ch 12).

**Data catalog** — Centralized metadata inventory emphasizing discoverability, glossary, lineage, and governance; batch-oriented (Ch 5).

**Data contract** — An agreement between data producers and consumers, established, updated, and enforced via an API/CI-CD, defining schema and semantic expectations for a data asset (Ch 1, Ch 4).

**Data debt** — The accumulated cost of short-term data infrastructure decisions (skipped modeling, documentation, semantic validity); measured via documentation coverage, dependency counts, backfill frequency, filters per query (Ch 1, Ch 2).

**Data discovery** — The process of determining whether needed data exists, where it lives, who owns it, what it means, how to use it, and whether you have access (Ch 3).

**Data downtime** — Time during which critical business data is inaccessible or untrustworthy; tracked via MTTD/MTTR (Ch 2).

**Data drift** — See concept drift.

**Data lake** — Low-cost centralized repository for structured/unstructured data with minimal inherent governance (Ch 1, Ch 5).

**Data lineage** — A directional graph of how data is sourced, transformed, and moved across tables/databases (Ch 6).

**Data product** — A packaged workflow/process bundling a data asset, a means to operationalize it, and a way to serve it to a consumer; must be reusable, domain-owned, and tied to a verifiable high-value outcome (Ch 11).

**Data quarantining** — Monitoring for data violations and tagging/separating data for caution or manual review ("dead letter queue") (Ch 6).

**Data supply chain** — Mapping of data roles onto physical supply chain roles (producers≈farmers, data engineers≈manufacturers, warehouse≈distributor, BI tools≈retailers) (Ch 10).

**Data warehouse** — Per Bill Inmon, a subject-oriented, integrated, time-variant, nonvolatile collection of data supporting management decision-making (Ch 1).

**Dunbar's Number** — The ~150-person cognitive limit on stable relationships a person can maintain; communication breaks down past this scale (Ch 3).

**Entropic behavior of data** — Data decays at varying rates (like radioactive half-life); decay rate (not value) determines required change management rigor (Ch 3).

**Event sourcing** — Creating an immutable, append-only log of every event via a publish-subscribe pattern (Ch 5).

**Event streaming (windowing)** — Aggregating high-volume data into fixed time intervals rather than logging every individual event (Ch 5).

**Forward-looking / backward-looking problems** — Forward = handling new/changing data; backward = handling legacy data not yet under contract (Ch 10).

**GIGO (Garbage-In, Garbage-Out)** — Data not meeting a stakeholder's expectations enters a pipeline and can only be fixed retrospectively (Ch 1).

**Hard failure / soft failure** — A contract violation either blocks a change outright (hard) or alerts impacted teams while allowing it through (soft); the contract spec should define when each applies (Ch 4).

**Invariant** — A true condition about data that the application (not the database) is responsible for preserving, e.g., "what defines a customer" (Ch 5).

**Issue tree** — McKinsey technique for decomposing a problem into hierarchical subcomponents (Ch 10).

**Kotter's Eight-Step Model** — Organizational change framework: urgency, guiding coalition, vision/strategy, communicate vision, empower action, short-term wins, consolidate gains, anchor in culture (Ch 11).

**MDS (Modern Data Stack)** — Cloud-native data tool suite (Snowflake, dbt, Fivetran, Airflow, etc.) (Ch 1).

**Minimum Viable Metric Tree (MVMT)** — Lightweight hierarchical map from a North Star metric down to the specific node a protected data product affects (Ch 12).

**Monomyth / Hero's Journey** — Joseph Campbell's 3-act story structure (Departure, Transformation, Return), applied as a storytelling framework for persuading stakeholders with metrics (Ch 12).

**OLAP (online analytical processing)** — Database workload optimized for large historical scans and denormalized analysis (Ch 2).

**OLTP (online transaction processing)** — Database workload optimized for fast CRUD transactions, ACID compliance, 3NF modeling (Ch 2).

**POSIWID** — "The purpose of a system is what it does" (Stafford Beer); judge systems by actual behavior, not stated intent (Ch 12).

**Push vs. pull architecture** — Push = producers define/control what's exposed and to whom; pull = consumers accept whatever arrives (Ch 8).

**Pyramid Principle** — McKinsey communication structure: decision → supporting arguments → supporting data (Ch 10).

**SCQA / SCR framework** — Situation, Complication, Question, Answer; a pitch structure for recommendations (Ch 10).

**Semantic events** — Events describing *why* something changed (business meaning) vs. CDC events, which only describe *what* changed (Ch 8).

**Shift left data** — Moving data management responsibility further upstream to where data is generated, modeled on DevSecOps's shift left security movement (Ch 1, Ch 9).

**Source code flow graph** — A directional graph of function/object references within a codebase (Ch 6).

**Steel thread** — A data product whose failure would cause immediate, measurable business disruption; used to select pilot candidates (Ch 11).

**Static Code Analysis (SCA)** — Reasoning about program behavior without executing it; used to detect data-impacting code changes before deployment (Ch 6).

**Tier-one data products** — The subset of downstream data assets important enough to prioritize for contract coverage first (Ch 4, Ch 9).

**Tracer bullets** — Using fast real-world feedback to iteratively adjust aim, rather than assuming first-try accuracy (Ch 11, The Pragmatic Programmer).

**Violated expectations** — Categorized as Schema, Semantics, SLA, or PII; the taxonomy of what a contract should specify (Ch 2).

**Weighted decision matrix** — Scoring candidates (1–5) against weighted criteria (1–10) to select the best pilot steel thread (Ch 11).

**Write-Audit-Publish (WAP)** — Pattern of writing data to staging, auditing it against contract rules, then publishing to production or routing to a dead-letter queue on failure (Ch 6, Ch 8).
