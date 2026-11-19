# Chapter 9: Data Quality Design Patterns

## Core Idea
Trustworthy datasets require three layered strategies: quality enforcement (block or flag bad data before it reaches consumers), schema consistency (control how a dataset's shape can evolve), and quality observation (continuously monitor so today's enforcement rules don't silently go stale).

## Frameworks Introduced
- **Audit-Write-Audit-Publish (AWAP)**: validate both input and output data around the transformation step, with configurable failure handling.
  - When to use: need to catch both bad incoming data and transformation bugs before publishing (e.g., an aggregation bug silently halving a metric).
  - How: first audit validates the input (cheap, fast checks — format, size, schema) before transformation; second audit validates the transformed output (data-level checks — nullability, business rules) before publish; on failure, choose fail / data dispatching (promote valid subset, dead-letter the rest) / nonblocking audit (promote but annotate with a quality flag); for streaming, use window-based (audit after each processing-time window) or staging-based (write to staging, audit, then promote) variants — the pre-transform audit step is typically skipped in streaming.
- **Constraints Enforcer**: delegate validation to the database/format itself via declarative constraints.
  - When to use: want validation without adding custom logic to the pipeline, and the data store supports native constraints.
  - How: apply type constraints (schema-level), nullability constraints (NOT NULL), value constraints (CHECK expressions, e.g., `event_time < NOW()`), and integrity constraints (foreign-key-style references); supported via SQL `CHECK`/`NOT NULL` (Delta Lake `ADD CONSTRAINT`), or serialization-format validators (Protobuf + protovalidate); accept that most are all-or-nothing (transactional) and stop at the first error.
- **Schema Compatibility Enforcer**: prevent data producers from making schema changes that break consumers.
  - When to use: multiple independent producers/consumers share a schema and silent breaking changes have caused outages.
  - How: external service/library validation (Kafka Schema Registry, Avro `SchemaValidator`), implicit enforcement via insert-time constraints (table file formats/relational databases), or DDL event triggers (PostgreSQL/SQL Server rejecting incompatible `ALTER TABLE`); choose a compatibility mode — backward (new schema reads old data), forward (old schema reads new data), full (both), each either transitive (holds across all historical versions) or nontransitive (holds only between consecutive versions).
- **Schema Migrator**: enable genuinely breaking changes (rename, retype, remove) safely over a transition period.
  - When to use: a field must be renamed, retyped, or removed, but existing consumers can't adapt instantly.
  - How: requires nontransitive compatibility; add the new/renamed/retyped field alongside the old one, run both in parallel for an agreed grace period (verify no consumers still depend on the old field via lineage — Fine-Grained Tracker, Ch 10), then remove the deprecated field once consumers have migrated; never execute a rename as an in-place `RENAME COLUMN`, since that breaks consumers immediately.
- **Offline Observer**: run data-quality monitoring as a separate, non-blocking job decoupled from the data pipeline's schedule.
  - When to use: want distribution/null-rate/schema-drift insight without touching production latency or resources, and some delay in detection is acceptable.
  - How: a scheduled observability job tracks a monitoring state (first/last row ID or offset processed) for idempotent, non-overlapping analysis windows, computes statistics (null counts, distributions, data profiles via e.g. ydata-profiling), and writes them to a monitoring store — fully decoupled from the generating pipeline's schedule.
  - **Observability vs. auditing**: observability is non-blocking (detects, doesn't prevent); auditing (AWAP/Constraints Enforcer) is blocking.
- **Online Observer**: run data-quality monitoring as part of the generating pipeline itself for near-real-time insight.
  - When to use: detection latency from Offline Observer is unacceptable (e.g., a week-old data regression surfacing after consumers already noticed).
  - How: integrate the observation step via Parallel Split (observe the just-loaded dataset, risk missing load-time issues) or Local Sequencer (observe the dataset actually exposed to consumers, safer but adds latency) in batch; for streaming, embed observation directly in the processing job (accumulators to avoid double-reading the input) and guard against observer failures crashing the whole job (e.g., Airflow `trigger_rule=all_done` on a final no-op task).

## Key Concepts
- **Audit step**: a validation check against real, production data — distinct from (and complementary to) static local unit tests.
- **Data dispatching**: splitting an audited output into valid (promoted) and invalid (dead-lettered) subsets based on explicit control logic, distinct from the Dead-Letter pattern's runtime-exception trigger.
- **Nonblocking audit**: promoting imperfect data anyway, annotated with a quality flag so consumers can decide whether to trust it.
- **Transitive vs. nontransitive compatibility**: transitive compatibility holds across all historical schema versions; nontransitive holds only between consecutive versions — Schema Migrator requires nontransitive compatibility to allow eventual field removal/rename.
- **Monitoring state table**: tracks the row/offset range already analyzed by an observer job, making repeated observation runs idempotent and non-overlapping.
- **Data profile**: a generated statistical summary (distributions, null rates, cardinality) of a dataset, often rendered as an HTML report.
- **Lag detection**: comparing the most recently committed offset against the most recent available offset to detect processing delay in streaming observers.

## Mental Models
- Treat AWAP's two audits as asking two different questions: "did my provider give me what I expected?" (input audit) and "did my own logic do what I expected?" (output audit) — put a given validation rule at the more exhaustive point to avoid redundant full-dataset passes.
- Constraints Enforcer trades flexibility for simplicity: the database stops you immediately but only reports the first violation, while AWAP can report everything but requires you to write the logic.
- Schema Compatibility Enforcer prevents breaking changes from happening at all; Schema Migrator is how you deliberately make a breaking change anyway, safely, over time.
- Offline vs. Online Observer is a time-accuracy-vs-isolation trade-off: Online catches issues sooner but risks coupling observation failures to the main pipeline; Offline is safe but can detect issues too late.

## Anti-patterns
- **Putting the same validation rule redundantly in both AWAP audit steps**: wastes compute reading/processing the dataset multiple times — put each rule at its most exhaustive point.
- **Treating every audit failure as a critical pipeline-stopping issue**: a legitimate traffic spike (e.g., going viral) can look like a data anomaly; some findings should only trigger investigation, not failure.
- **Renaming or retyping a column with a direct `ALTER TABLE ... RENAME COLUMN`**: immediately breaks any consumer query referencing the old name — always use the Schema Migrator's add-then-deprecate approach.
- **Relying solely on database constraints for consumer-specific expectations**: a field nullable at the database level may still be effectively "required" for particular consumers — they need their own validation on top.
- **Scheduling Offline Observer far less frequently than the data generator "to save resources"**: increases the compute needed per run (processing a full day instead of an hour) and delays issue detection.
- **Running Online Observer via Parallel Split on the loaded (not transformed) dataset**: can miss load-time issues like data type/format problems, since the observer never sees the data the way consumers eventually will — use Local Sequencer and observe the actually-exposed dataset when this risk matters.
- **Letting an Online Observer's own failure crash the whole generating pipeline**: isolate it (e.g., trigger rule `all_done` on a closing task) so observability bugs don't take down data delivery.

## Code Examples
```python
# AWAP: input-dataset audit before transformation
if f_size < min_size:
  validation_errors.append(f'File is to small. Expected at least {min_size} bytes but got {f_size}')
if lines < min_lines:
  validation_errors.append(f'File is too short. Expected at least {min_lines} lines but got {lines}')
if validation_errors:
  raise Exception('Audit failed for the file:\n-'+"\n-".join(validation_errors))
```
- **What it demonstrates**: a lightweight, fast pre-transformation audit catching structural issues before the expensive transform runs.

```sql
-- Constraints Enforcer: Delta Lake type, nullability, and value constraints
CREATE TABLE default.visits (
 visit_id STRING NOT NULL,
 event_time TIMESTAMP NOT NULL
) USING delta;

ALTER TABLE default.visits ADD CONSTRAINT
  event_time_not_in_the_future CHECK (event_time < NOW() + INTERVAL "1 SECOND")
```
- **What it demonstrates**: declarative, database-enforced validation that rejects an entire write transaction if any row violates it.

```sql
-- Schema Migrator: safe rename via add-then-deprecate, never an in-place RENAME
ALTER TABLE dedp.visits ADD COLUMN referral VARCHAR(25) NOT NULL
-- old `from_page` column stays until consumers migrate, then is dropped later
```
- **What it demonstrates**: avoiding the immediate consumer breakage a direct `RENAME COLUMN` would cause.

```sql
-- Offline Observer: idempotent monitoring state tracking a non-overlapping analysis window
INSERT INTO dedp.visits_monitoring_state (execution_time, first_row_id, last_row_id)
  SELECT '{{ execution_date }}' AS execution_time, MIN(id), MAX(id)
  FROM dedp.visits_output
  WHERE id > COALESCE(
    (SELECT last_row_id FROM dedp.visits_monitoring_state WHERE execution_time = '{{ prev_execution_date }}'),
    0)
```
- **What it demonstrates**: recording a row-ID checkpoint so repeated observation runs never re-analyze or skip rows.

## Reference Tables
| Compatibility mode | Who benefits | Allowed change |
|---|---|---|
| Backward | Consumer on new schema reads old data | Add optional/defaulted field |
| Forward | Consumer on old schema reads new data | Remove optional field |
| Full | Both | Only changes safe in both directions |

| Pattern | Blocking? | Runs where |
|---|---|---|
| AWAP | Yes | Inside the pipeline (pre/post transform) |
| Constraints Enforcer | Yes | Database/format write path |
| Schema Compatibility Enforcer | Yes | External registry / implicit insert / DDL trigger |
| Offline Observer | No | Separate, independently scheduled job |
| Online Observer | No (observation) / risk to pipeline if unisolated | Inside or parallel to the generating pipeline |

## Worked Example
Blog analytics platform: the daily visits-statistics ETL job gains an AWAP pipeline — `audit_file_to_load` checks the incoming JSON file's size/line count/JSON validity before `transform_file` runs, and `audit_transformed_file` checks the flattened CSV for unexpected NULLs in required columns before `load_flattened_visits_to_final_table` runs (catching the kind of silent aggregation bug that halved the unique-visitor count in the chapter's opening story). The `visits` Delta Lake table itself enforces `NOT NULL` on `visit_id`/`event_time` and a `CHECK (event_time < NOW() + INTERVAL "1 SECOND")` constraint (Constraints Enforcer) as a second line of defense. Upstream, visit events are registered in Kafka Schema Registry with backward compatibility (Schema Compatibility Enforcer), so a new optional field can be added safely but a required field can never just disappear; when the team later needs to rename `from_page` to `referral` for clarity, they add the new column, run both in parallel for an agreed migration window (checked against consumer lineage), then drop `from_page` only after consumers confirm migration (Schema Migrator) — never a direct `RENAME COLUMN`. Separately, a nightly Offline Observer job tracks a `visits_monitoring_state` table of first/last processed row IDs and writes null-rate and distribution statistics to `visits_monitoring`, decoupled from the hourly generating pipeline. When a data regression in a zip-code field takes a full week to surface through this nightly job, the team adds an Online Observer step via Local Sequencer right after the data-loading task, isolated with an `all_done`-triggered closing task so an observer bug can never take down the main pipeline.

## Key Takeaways
1. Place each validation rule at its most exhaustive point in an AWAP pipeline (input vs. output audit) to avoid redundant full-dataset passes.
2. Prefer native Constraints Enforcer validation when the data store supports it — it's simpler than custom audit logic, but remember it's producer-oriented and may miss consumer-specific expectations.
3. Define an explicit schema compatibility mode (backward/forward/full, transitive/nontransitive) before you need it — it determines what evolutions are even possible later.
4. Use Schema Migrator's add-then-deprecate transition period for any rename/retype/removal; never make a breaking DDL change directly.
5. Choose Offline Observer for safe, decoupled monitoring when some detection delay is tolerable; choose Online Observer when time-to-detection matters more than pipeline isolation — and always isolate the observer so its failures don't take down data delivery.
6. Observability and auditing are different disciplines: auditing blocks bad data from publishing; observability just makes sure you notice when your audit rules themselves need to change.

## Connects To
- **Ch 3 (Error Management)**: AWAP's data-dispatching outcome parallels the Dead-Letter pattern, but is explicitly validation-driven rather than exception-driven.
- **Ch 5 (Data Value)**: schema struct/wrapper design choices interact with Schema Migrator's add-then-deprecate transitions.
- **Ch 6 (Data Flow)**: Online Observer is implemented using Parallel Split or Local Sequencer.
- **Ch 10 (Data Observability)**: Schema Migrator's safe-removal check depends on the Fine-Grained Tracker (lineage) pattern introduced there; this chapter's observers feed directly into Ch 10's broader monitoring/alerting patterns.
- **Apache Airflow, Apache Kafka Schema Registry, Apache Avro, Protobuf/protovalidate, Delta Lake, PostgreSQL, Apache Spark Structured Streaming, ydata-profiling**: primary technologies implementing these patterns.
