# Chapter 3: Error Management Design Patterns

## Core Idea
Errors (bad records, duplicates, late data, buggy filters, hardware failures) are inevitable in data engineering; the goal is not to prevent them but to manage them systematically — isolate unprocessable data, bound duplication, detect/integrate lateness, measure filtering impact, and checkpoint progress — while recognizing that none of this by itself guarantees exactly-once correctness (that needs idempotency, Ch 4).

## Frameworks Introduced
- **Dead-Letter**: redirect unprocessable records to a separate store instead of failing the whole job.
  - When to use: nontransient "poison pill" records in long-running/streaming jobs that must keep running.
  - How: wrap risky transformations in try-catch (or if-else for error-safe functions); write failures to a resilient, easily monitored sink (object store/streaming broker) with error metadata (Metadata Decorator); optionally replay later via a dedicated pipeline.
- **Windowed Deduplicator**: remove duplicate records within a bounded scope.
  - When to use: at-least-once delivery produces duplicates but downstream logic needs each occurrence processed once.
  - How: define dedup key attributes; for batch, dedup over the whole processed dataset (DISTINCT / ROW_NUMBER window); for streaming, dedup within a time-based window backed by a state store (local, local-with-fault-tolerance, or remote).
- **Late Data Detector**: identify records whose event time falls behind the tracked progress (watermark).
  - When to use: need to classify incoming records as on-time vs. late (e.g., for stateful aggregation or partition completion).
  - How: define an event-time attribute; aggregate per-partition with MAX (monotonic); aggregate across partitions with MIN (follow slowest source) or MAX (follow fastest) or a MIN/MAX combination; subtract an "allowed lateness" value to get the watermark = MAX(event_time) − allowed_lateness.
- **Static Late Data Integrator**: reintegrate late data using a fixed lookback window.
  - When to use: late data has a known, fixed maximum delay (e.g., 15 days).
  - How: define a static lookback window; reprocess that fixed set of past partitions every run (sequentially first for stateful pipelines; in parallel/after for stateless); never trigger these backfill tasks as separate pipelines (causes overlapping executions).
- **Dynamic Late Data Integrator**: reintegrate late data using only the partitions that actually changed.
  - When to use: late data can arrive beyond any fixed window, and reprocessing every partition in a static window would waste resources.
  - How: maintain a state table (partition, last_processed_time, last_update_time[, is_processed]); query for partitions where last_update_time > last_processed_time; use native last-modified metadata where available (BigQuery `INFORMATION_SCHEMA.PARTITIONS`, Iceberg's `last_updated_at`) or derive it (e.g., Delta Lake's `DeltaLog.getChanges`); guard concurrency with an `is_processed` flag and `depends_on_past` task sequencing.
- **Filter Interceptor**: instrument filter conditions to measure how many rows each condition removes.
  - When to use: need to diagnose a spike in filtered-out volume (data issue vs. code regression) that query-plan analysis can't isolate because the engine collapsed filters into one step.
  - How: programmatic API — wrap each filter with a counter/accumulator incremented on failure; SQL — expose each filter as a computed status-flag column in a subquery, then GROUP BY it to get per-condition counts.
- **Checkpointer**: persist processing progress (offsets/state) so a restarted job resumes instead of reprocessing from scratch.
  - When to use: any continuously running (especially streaming) job needing fault tolerance.
  - How: framework-managed checkpoints (Spark Structured Streaming, Flink write offsets/state to resilient storage) or store-managed (Kafka `__consumer_offsets`, Kinesis KCL → DynamoDB); configuration-driven (periodic) or code-driven (explicit commit); checkpoint-before-processing = at-most-once, checkpoint-after-processing = at-least-once.

## Key Concepts
- **Transient vs. nontransient errors**: transient errors self-recover (e.g., brief DB outage + retry); nontransient errors ("poison pills") never recover without intervention.
- **Snowball backfilling effect**: replaying dead-lettered or late data forces downstream consumers (and their consumers) to backfill too, cascading compute cost.
- **Watermark**: the minimum event time a job will still accept as "on time" = tracked event time minus allowed lateness; must be monotonically increasing.
- **Event time vs. processing time**: event time is when something happened (can be late); processing time is when the pipeline handled it (never late).
- **State store**: persistent/in-memory store of already-seen keys or accumulated state for stateful streaming operations (dedup windows, sessions).
- **Exactly-once "feeling"**: checkpointing alone gives an impression of exactly-once delivery but doesn't guarantee it — true exactly-once needs idempotency patterns (Ch 4).
- **Delivery modes**: exactly-once (ideal, needs idempotency), at-least-once (checkpoint after write → possible duplicates), at-most-once (checkpoint before write → possible data loss).

## Mental Models
- Use Dead-Letter as a pressure valve: it keeps the pipeline alive but hides failures — always pair it with alerting so a true systemic failure still surfaces.
- Think of the watermark as "how far back am I still willing to accept data" — MAX per partition keeps it moving forward; MIN across multiple sources trades completeness for buffering cost.
- Choose Static vs. Dynamic Late Data Integrator the way you choose any fixed vs. adaptive retry policy: static is simple but wastes resources on empty windows; dynamic is efficient but adds state-tracking complexity.
- Checkpointing frequency is a dial between latency (fewer checkpoints = faster but more reprocessing on failure) and recovery cost (more checkpoints = slower but less to redo).

## Anti-patterns
- **Ignoring dead-lettered records without a replay strategy**: silently loses data permanently.
- **Using MIN to track per-partition event time**: can cause "stuck-in-the-past" (watermark never advances) or "open-close-open" reopening of already-emitted state.
- **Using MAX across highly skewed multi-source streams**: aggressively drops records from slower sources.
- **Backfilling each date independently under a static lookback window**: creates overlapping/duplicated reprocessing (see Table 3-3 in chapter) — only the latest execution needs restarting.
- **Running Dynamic Late Data Integrator concurrently without an is_processed guard**: generates duplicate late-data integration runs for the same partition.
- **Relying on checkpointing alone for exactly-once guarantees**: distributed, asynchronous task failures can still cause reprocessing of already-successful records.

## Code Examples
```python
# Dead-Letter: Spark error-safe transformation split into valid/invalid outputs
devices_to_load_with_validity_flag.persist()
(devices_to_load_with_validity_flag.filter('is_valid IS TRUE')
 .drop('is_valid').write.mode('overwrite')
 .format('delta').save(f'{base_dir}/output/devices-table'))
(devices_to_load_with_validity_flag.filter('is_valid IS FALSE')
 .drop('is_valid').write.mode('overwrite')
 .format('delta').save(f'{base_dir}/output/devices-dead-letter-table'))
```
- **What it demonstrates**: isolating bad records (via an `is_valid` flag derived from an error-safe CONCAT) into a separate Delta table without failing the job.

```python
# Windowed Deduplicator: streaming dedup bounded by a watermark
# ...part 2
.withWatermark("visit_time", "10 minutes")
.dropDuplicates(["visit_id", "visit_time"])
.drop("visit_time", "visit_id"))
```
- **What it demonstrates**: bounding dedup state growth by expiring keys older than the watermark.

```sql
-- Late Data Detector concept extended: Dynamic Late Data Integrator state-table query
SELECT partition FROM state_table WHERE
`Last update time` > `Last processed time` AND
`Partition` < `Processed partition` AND
`Is processed` = false
```
- **What it demonstrates**: selecting only partitions with genuinely new late data, guarded against concurrent duplicate processing.

```python
# Checkpointer: Spark Structured Streaming checkpoint location
write_query = (input_stream_data.writeStream.outputMode('update')
   .option('checkpointLocation', f'{base_dir}/checkpoint')
   .foreachBatch(synchronize_visits_to_files).start())
```
- **What it demonstrates**: framework-managed, configuration-driven checkpointing of stream offsets.

## Reference Tables
| State store type | Speed | Fault tolerance | Notes |
|---|---|---|---|
| Local (in-memory) | Fastest | None | State lost on failure |
| Local + fault-tolerant | Fast | Yes (periodic persist) | Trade-off between consistency and speed depending on persist frequency |
| Remote | Slower | Native | Higher latency/cost |

| Checkpoint timing | Delivery mode | Risk |
|---|---|---|
| Before processing | At-most-once | Data loss on failure |
| After processing | At-least-once | Duplicates on retry |
| + Idempotency pattern (Ch 4) | Exactly-once | — |

## Worked Example
Blog analytics streaming pipeline: a Flink job maps a Kafka `visits` topic to JSON output. Each record passes through `MapJsonToReducedVisit`, where parse failures are caught and routed to a `side output` tagged `invalid_visits`, which is sinked to a Kafka dead-letter topic (Dead-Letter pattern) while valid records go to the main sink. A downstream Spark job later deduplicates arriving visits using `dropDuplicates` with a 10-minute watermark on `visit_time` (Windowed Deduplicator). Separately, the same visit stream is checked for lateness: `WatermarkStrategy.for_bounded_out_of_orderness(5 seconds)` drives a `VisitLateDataProcessor` that compares each record's event time against `ctx.timer_service().current_watermark()`, routing late visits to a side output Kafka topic for reprocessing (Late Data Detector + Static/Dynamic Integrator depending on whether the lateness bound is fixed or derived from a state table). The whole job commits progress every 30 seconds via Flink's checkpointing (`enable_checkpointing(30000, EXACTLY_ONCE)`), ensuring a restart resumes without reprocessing from the start — though true exactly-once output still requires an idempotency pattern from Ch 4.

## Key Takeaways
1. Dead-Letter keeps pipelines alive through bad records, but always add alerting so dead-lettering doesn't mask a real systemic failure.
2. Choose the deduplication scope (whole batch vs. time window) based on whether the job is batch or streaming, and size the window as a space-vs-completeness trade-off.
3. Late-data handling always needs a monotonically increasing watermark (never MIN at the per-partition level) to avoid stuck-in-the-past or reopened-state bugs.
4. Prefer Dynamic over Static Late Data Integrator when lookback duration is unpredictable, but budget for the added state-tracking and concurrency-control complexity.
5. Instrument filters (Filter Interceptor) before assuming a data-volume spike is a data problem rather than a code regression.
6. Checkpointing provides fault tolerance and an "exactly-once feeling," not an exactly-once guarantee — pair it with idempotency (Ch 4) for real correctness.

## Connects To
- **Ch 2 (Data Ingestion)**: External Trigger and Readiness Marker pipelines should be wrapped with Dead-Letter-style error handling.
- **Ch 4 (Idempotency)**: explicitly the next chapter — checkpointing and dedup reduce but don't eliminate the need for idempotent writes; also references the Proxy and Incremental/Stateful Sessionizer patterns.
- **Apache Flink, Apache Spark Structured Streaming, Delta Lake, Apache Kafka, Apache Airflow (Dynamic Task Mapping), BigQuery, Apache Iceberg**: the primary technologies used to implement these patterns.
