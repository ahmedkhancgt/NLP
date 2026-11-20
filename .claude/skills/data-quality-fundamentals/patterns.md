# Patterns: Data Reliability & Monitoring Techniques

## Circuit Breaker
**When to use**: When bad data must never be allowed to propagate downstream to critical dashboards/models — the highest-stakes incident class.
**How**: Stop pipeline execution ("circuit open") when data fails freshness, volume, distribution, or schema thresholds; requires lineage, data profiling across the pipeline, and automated triggering (per Sandeep Uttamchandi). Implement via Airflow's `catchup=False`, `LatestOnlyOperator`, or custom Python breaks; or SQL check operators that halt on a failed query.
**Trade-offs**: Prevents mixing bad/good data and avoids costly backfills, but overuse blocks unrelated, perfectly fine jobs — reserve for genuinely critical incidents only.

## Freshness Anomaly Detection (Rolling Threshold)
**When to use**: Monitoring whether critical tables/reports are updating on expected cadence.
**How**: Compute days/hours since last update via `LAG()` window functions; set a threshold parameter (e.g., `DAYS_SINCE_LAST_UPDATE > X`) to convert a query into a detector. Tune X to balance recall (catch more, smaller X) vs. precision (catch fewer false alarms, larger X).
**Trade-offs**: Simple and fast to implement, but a single global threshold misses seasonality-driven legitimate gaps (e.g., planned maintenance windows) unless paired with a rolling average.

## Distribution/Null-Rate Monitoring with Rolling Average
**When to use**: Detecting gradual or sudden shifts in null rate, zero rate, or value distribution for a field.
**How**: Compute per-batch null/zero rate; compare against a trailing N-period rolling average rather than a fixed threshold (e.g., `AVG(...) OVER (ORDER BY date ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)`); alert when the deviation (not absolute value) exceeds a threshold.
**Trade-offs**: Reduces false positives from slow drift and seasonality, but requires enough historical data to establish a meaningful baseline and more complex query logic.

## Schema Change Detection via Versioned Metadata Table
**When to use**: Catching structural changes (added/removed/retyped columns) that silently break downstream transformations.
**How**: Maintain a side table recording column lists per date (or use warehouse system tables like `information_schema.columns`); use `LAG()` to compare current vs. previous schema snapshot and alert on diffs.
**Trade-offs**: Requires either custom instrumentation or a mature ingestion pipeline that writes schema snapshots; doesn't capture *why* the schema changed, only *that* it did — pair with lineage/RCA for context.

## Field-Level Lineage Mapping
**When to use**: Root cause analysis and impact analysis when table-level lineage is too coarse to pinpoint the exact broken field.
**How**: Parse SQL (e.g., via ANTLR) to extract SELECT and non-SELECT clause relationships; recursively resolve aliases/subqueries to true source tables/columns; store as a hashed destination-table-ID + query-hash model to support multiple queries updating the same table.
**Trade-offs**: Technically difficult (full SQL clause coverage is hard — initial prototypes commonly cover ~70% of cases); high payoff for RCA/impact-analysis speed once built; requires significant engineering investment or a vendor solution.

## Precision/Recall-Tuned Anomaly Detection
**When to use**: Any statistical/ML-based anomaly detector where raw "accuracy" is misleading (rare-event problems).
**How**: Calculate TP/FP/TN/FN for your detector against a labeled ground truth; optimize an Fβ score chosen by the cost asymmetry of your use case (β>1 prioritizes recall for safety-critical alerts; β<1 prioritizes precision to avoid alert fatigue).
**Trade-offs**: Requires ground-truth labeling effort upfront; the "right" β is subjective and must be revisited as business priorities shift.

## Root Cause Analysis (Five-Step Sequence)
**When to use**: Any unclear data incident where the cause isn't immediately obvious.
**How**: (1) Lineage — find the most upstream broken node. (2) Code — inspect the transformation logic/query that produced the table. (3) Data — slice by segment/time period/schema for anomaly patterns. (4) Operational environment — check ETL/ELT job logs, permissions, infra changes. (5) Peers — consult data owners/users and past incident history.
**Trade-offs**: Systematic and teachable (reduces reliance on one "data pipeline encyclopedia" engineer), but still manual and time-consuming without automation; best paired with automated lineage/observability tooling to speed each step.

## Blameless Postmortem
**When to use**: After every significant data incident, regardless of root cause.
**How**: Document sequence of events, affected parties, technologies, and root cause; frame the system (not individuals) as at fault; update runbooks/monitoring; revisit SLAs/SLIs/SLOs; share broadly across the team.
**Trade-offs**: Takes time away from "moving fast," but prevents tacit-knowledge loss and repeat incidents; skipping it compounds technical and cultural debt.

## Incident Commander Rotation
**When to use**: Scaling data incident management past ad hoc, single-engineer firefighting.
**How**: Assign a rotating role (weekly/daily, or by table/domain ownership) responsible for routing notifications, assessing severity, communicating status, and aligning on SLOs/SLIs to prevent recurrence.
**Trade-offs**: Adds organizational overhead and requires cultural buy-in, but prevents burnout of the few people with tacit institutional knowledge.

## Alert Noise Suppression / Grouping
**When to use**: At enterprise scale, when multiple alerts stem from one underlying incident and overwhelm responders.
**How**: Group/deduplicate related alerts (e.g., via PagerDuty) so one incident generates one notification thread rather than dozens; tune signal-to-noise ratio toward zero.
**Trade-offs**: Requires upfront configuration and tagging discipline; under-tuned suppression risks hiding genuinely distinct issues.

## Data Certification Program
**When to use**: Formalizing which data assets are trustworthy enough for company-wide self-service use.
**How**: Seven steps — build observability, assign owners, define "good data" KPIs per pillar, set SLAs/SLOs/SLIs, build communication/incident processes, tag/certify (decentralized), and train consumers. Start with "go forward" (new/critical tables) rather than retroactive certification of everything; consider bronze/silver/gold tiers.
**Trade-offs**: Significant upfront process investment; payoff is reduced "is this data trustworthy?" friction at scale, but requires sustained cross-team discipline to maintain certification status.

## Data-as-a-Product Treatment
**When to use**: When data serves multiple internal "customers" (analysts, scientists, PMs) rather than a single report/dashboard.
**How**: Apply product management discipline — stakeholder alignment, data storytelling to justify investment, scoping docs/sprints, self-serve tooling, and tracked KPIs (e.g., cost of downtime).
**Trade-offs**: Requires a cultural shift and often a dedicated data product manager role; pays off in increased adoption, trust, and reduced ad hoc firefighting — but is a multi-year, not one-time, investment.

## Hub-and-Spoke Team Structure
**When to use**: Balancing speed (domain expertise) against governance consistency as an organization scales past a single centralized or fully decentralized team.
**How**: Centralized data platform team owns infrastructure/data quality; decentralized embedded analysts/engineers own semantic layers and direct business application.
**Trade-offs**: Works well for fast-growing orgs but can produce duplicated effort between embedded analysts without strong central alignment; expect the right structure to keep changing as headcount scales (per Toast's repeated centralize/decentralize cycle).
