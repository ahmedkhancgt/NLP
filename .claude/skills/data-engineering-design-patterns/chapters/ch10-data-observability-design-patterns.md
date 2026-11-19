# Chapter 10: Data Observability Design Patterns

## Core Idea
Data quality patterns (Ch 9) protect data correctness but can't catch systemic failures like silent flow interruptions or late SLAs; observability closes this gap with two pillars — detection (spotting data/time anomalies: interruptions, skew, lag, SLA misses) and tracking (dataset and column/row-level lineage) — to make the whole pipeline, not just the data, trustworthy.

## Frameworks Introduced
- **Flow Interruption Detector**: alert when expected data stops arriving.
  - When to use: a job can silently stop producing output without throwing an error (e.g., a streaming sync job that runs but writes nothing).
  - How: for continuous delivery, alert on zero records in a fixed unit of time; for irregular delivery, alert only when a no-data window exceeds an accepted threshold (avoids false-positive alarm fatigue); for batch/at-rest data, check metadata layer (last-modified timestamp), data layer (added modification-time column or row-count comparison across periods), or storage layer (last file-write time) — watch for false positives from housekeeping operations like compaction that touch storage without adding new data.
- **Skew Detector**: alert when a dataset's volume deviates abnormally from a comparison baseline.
  - When to use: an upstream producer could deliver a partial/incomplete dataset without failing the job (e.g., half-empty daily extract).
  - How: pick a comparison window (e.g., vs. yesterday) and a tolerance threshold (e.g., ±50%); compute either a window-to-window percentage difference or a standard-deviation ratio (`STDDEV(x)/AVG(x)`) across partitions; best placed as the pre-transform audit step of AWAP (Ch 9); beware the "fatality loop" where comparing against a bad prior day perpetuates false skew alerts — compare against the last known-good run instead.
- **Lag Detector**: measure how far a consumer has fallen behind the most recently available data.
  - When to use: need an early warning for streaming (or any incremental) consumers falling behind their producers.
  - How: define a lag unit matching the store (Kafka offset/append time, Delta Lake commit version, partition timestamp); compute `last_available_unit - last_processed_unit`; for partitioned stores, aggregate via MAX (worst case) and/or a percentile (P90/P95, which is more honest than an average — a single skewed partition can hide behind a deceptively low mean).
- **SLA Misses Detector**: alert when a job's execution time exceeds an agreed maximum.
  - When to use: downstream consumers have a hard deadline, independent of per-record lag.
  - How: batch — compare end time minus start time against the SLA threshold; streaming — apply the same subtraction per microbatch/event-time window, or (if windowing isn't available) measure per-record read-to-write latency and aggregate with MAX/percentile; track both processing-time SLA (purely about your own speed) and event-time SLA (end-to-end, vulnerable to upstream late data) as separate, complementary metrics — they are not interchangeable with Lag Detector (a throughput-limited consumer can meet its SLA while lag silently grows).
- **Dataset Tracker**: build a dependency graph (lineage) among datasets/tables/topics across teams.
  - When to use: need to trace which upstream dataset/team introduced a data quality issue, or to understand cross-team data dependencies in general.
  - How: fully-managed capture (Databricks Unity Catalog, GCP Dataplex) when using supported platform-native services, or self-implemented lineage by extracting input/output dependencies at the orchestration layer (e.g., OpenLineage + Airflow) or by parsing executed queries at the database layer, then visualizing the resulting graph (e.g., via Marquez UI).
- **Fine-Grained Tracker**: trace lineage at the column or row level, not just the dataset level.
  - When to use: need to know exactly which upstream columns compose a given output column (e.g., in a heavily Denormalized table), or which job wrote a specific row.
  - How: column-level — native platform support (Databricks `system.access.column_lineage`, Azure Purview) or execution-plan analysis (works well for native SQL/Spark transforms, but is opaque for custom programmatic mapping functions); row-level — decorate each record with lineage metadata (job name/version, batch number, and a `parent_lineage` field referencing the upstream job's own lineage metadata) via the data decoration patterns (Ch 5), since row-level lineage doesn't integrate with standard dataset/column lineage visualization tools.

## Key Concepts
- **Alarm fatigue**: excessive false-positive alerts (e.g., from a too-strict continuous-delivery threshold) eroding trust in and responsiveness to alerts.
- **Fatality loop**: a window-to-window skew check repeatedly flagging valid data as skewed because the prior comparison baseline was itself invalid.
- **Average trap**: using a mean lag/latency value hides how bad the worst (or even the 90th-percentile) case really is — prefer percentiles for lag/SLA metrics.
- **Processing time vs. event time SLA**: processing-time SLA measures your own job's speed; event-time SLA measures true end-to-end latency and can be broken by upstream late data that isn't your fault.
- **Vendor lock in managed lineage**: fully-managed lineage tools (Unity Catalog, Dataplex) only see dependencies within their own service scope, missing cross-cloud/open-source parts of the pipeline.

## Mental Models
- Think of detection patterns as answering "is something currently wrong?" (data/time anomalies) and tracking patterns as answering "who/what is responsible when something is wrong?" (lineage) — they're complementary, not substitutes.
- Lag Detector and SLA Misses Detector measure different things and can diverge in either direction: a throughput-capped consumer can hit its SLA while lag grows unboundedly, and a batch job with zero lag (daily schedule) can still miss its SLA by running too long.
- Choose MAX vs. percentile aggregation for lag/skew per what question you're asking: "what's the worst case I must plan for?" (MAX) vs. "how is the system doing overall?" (percentile).
- Dataset Tracker gives you the organizational map; Fine-Grained Tracker gives you the street-level detail — reach for the latter only when the former isn't enough to resolve a specific issue.

## Anti-patterns
- **Setting a continuous-delivery flow-interruption threshold too strict (e.g., "at least 1 record/minute, always")**: produces alarm fatigue on days with naturally low but legitimate traffic.
- **Treating storage-layer activity (e.g., compaction writes) as evidence a dataset is still flowing**: compaction rewrites existing data without adding new records — it will mask a real interruption.
- **Comparing skew against the immediately preceding run without checking that run was itself valid**: creates a fatality loop that perpetuates false alerts after one bad day.
- **Using an average to represent lag or latency across partitions**: a severely skewed single partition can hide behind a deceptively low mean — use percentiles (P90/P95) plus MAX for the worst case.
- **Conflating Lag Detector and SLA Misses Detector as interchangeable**: they measure different failure modes and must both be monitored for full coverage.
- **Relying only on a fully-managed lineage tool across a multi-cloud/open-source stack**: produces an incomplete lineage graph outside that platform's scope.
- **Expecting column-level lineage tools to see through custom programmatic transformation functions**: these are opaque to execution-plan-based lineage extraction — document such transforms manually.

## Code Examples
```promql
# Flow Interruption Detector: rate of incoming Kafka messages per minute, alert on sustained zero
sum without(instance)(rate(
kafka_server_brokertopicmetrics_messagesin_total{topic="visits"}[1m]))
```
- **What it demonstrates**: a Prometheus expression feeding a Grafana alert that fires when the last N evaluations are all zero.

```sql
-- Skew Detector: standard-deviation ratio across PostgreSQL partitioned tables
SELECT
 NOW() AS "time", (STDDEV(n_live_tup) / AVG(n_live_tup)) * 100 AS value
 FROM pg_catalog.pg_stat_user_tables
WHERE relname != 'visits_all_range' AND relname LIKE 'visits_all_range_%';
```
- **What it demonstrates**: detecting partition imbalance via coefficient of variation rather than a simple window-to-window comparison.

```python
# Lag Detector: comparing last-written vs. last-read Delta Lake table version
last_version = query.lastProgress["sources"][0]["endOffset"]["reservoirVersion"]
# compared against:
last_written_version = (spark_session.sql('DESCRIBE HISTORY default.visits')
  .selectExpr('MAX(version) AS last_version').collect()[0].last_version)
```
- **What it demonstrates**: computing consumer lag as the difference between the latest committed table version and the latest version the consumer has actually processed.

```python
# Fine-Grained Tracker: row-level lineage propagated through record headers
# visits_reducer_job
(visits_to_save.withColumn('headers', F.array(
 F.struct(F.lit('parent_lineage').alias('key'), F.to_json(F.col('headers'))
  .cast('binary').alias('value'))
)))
```
- **What it demonstrates**: carrying the upstream job's own lineage metadata forward into each downstream record's headers.

## Reference Tables
| Pattern | Measures | Typical trigger |
|---|---|---|
| Flow Interruption Detector | Data arrival | No new data within threshold |
| Skew Detector | Data volume | Volume deviates beyond tolerance vs. baseline |
| Lag Detector | Consumer position vs. producer position | Lag exceeds threshold (MAX or percentile) |
| SLA Misses Detector | Execution duration | Duration exceeds agreed max (processing or event time) |
| Dataset Tracker | Dataset-to-dataset dependency | N/A (exploratory/diagnostic) |
| Fine-Grained Tracker | Column/row-to-source dependency | N/A (exploratory/diagnostic) |

## Worked Example
Blog analytics platform: the streaming sync job feeding batch consumers gets a Flow Interruption Detector via a Prometheus/Grafana rule on `kafka_server_brokertopicmetrics_messagesin_total{topic="visits"}`, alerting if the last five one-minute evaluations are all zero — catching the exact silent-write failure described in the chapter's opening story. The daily visits-loading Airflow pipeline adds a `compare_volumes` task (Skew Detector) comparing the current day's file size against the prior day's within a ±50% tolerance, raising an exception before the transform step runs if the input looks suspiciously small or large — effectively acting as AWAP's pre-transform audit (Ch 9). A Spark Structured Streaming consumer of the same Kafka topic runs a `StreamingQueryListener` that computes per-partition offset lag and pushes it to Prometheus as a gauge (Lag Detector), aggregated with both MAX (worst partition) and P90 (overall health) to avoid the "average trap." The downstream statistics pipeline, scheduled at 6:00 a.m. with a hard 8:00 a.m. consumer deadline, defines an Airflow task-level `sla=timedelta(minutes=40)` (SLA Misses Detector) — tracked separately from the Lag Detector since a throughput-capped job could meet its SLA while quietly falling behind in lag. Finally, when a schema inconsistency surfaces in a far-downstream denormalized table, the team uses OpenLineage + Marquez (Dataset Tracker) to trace the dataset dependency chain back to the originating team, then uses column-level lineage (Fine-Grained Tracker) to pinpoint exactly which upstream column's type change propagated into the broken field, supplementing it with row-level `parent_lineage` headers for jobs whose transformations are too custom for automatic execution-plan analysis.

## Key Takeaways
1. Match the Flow Interruption Detector's strategy (continuous vs. irregular) to the data's real arrival pattern to avoid alarm fatigue.
2. Guard against skew-detection fatality loops by comparing against the last known-good run, not blindly against the immediately preceding one.
3. Always prefer percentiles (plus MAX for worst-case) over averages when aggregating lag or latency across partitions.
4. Monitor Lag Detector and SLA Misses Detector independently — they can diverge in either direction and together give full time-based coverage.
5. Start with Dataset Tracker for cross-team dependency questions; escalate to Fine-Grained Tracker only when column- or row-level detail is actually needed.
6. Expect managed lineage tools to be scoped to their own platform — plan for gaps in multi-cloud/open-source stacks and consider a self-implemented (OpenLineage-style) layer for full coverage.

## Connects To
- **Ch 2 (Data Ingestion)**: Readiness Marker's completeness signaling is the ingestion-side counterpart to Flow Interruption/Skew detection.
- **Ch 5 (Data Value)**: Fine-Grained Tracker's row-level lineage reuses the data decoration/Metadata Decorator patterns to carry lineage metadata.
- **Ch 6 (Data Flow)**: Dataset Tracker formalizes the implicit dependencies that Isolated Sequencer's data-based triggering creates.
- **Ch 9 (Data Quality)**: Skew Detector is explicitly designed to plug into AWAP's pre-transform audit step; both Offline/Online Observer patterns feed metrics that these detectors consume.
- **Apache Kafka, Prometheus, Grafana, PostgreSQL, Apache Spark Structured Streaming, Apache Flink, Delta Lake, Apache Airflow, OpenLineage, Marquez, Databricks Unity Catalog, GCP Dataplex, Azure Purview**: primary technologies implementing these patterns.
