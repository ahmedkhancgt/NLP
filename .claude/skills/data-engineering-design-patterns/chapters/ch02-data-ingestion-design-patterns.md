# Chapter 2: Data Ingestion Design Patterns

## Core Idea
Bringing data into a system is never "just moving files" — choosing between full vs. incremental loading, replication vs. transformation, and pull vs. push triggering each carries specific consistency, latency, and cost trade-offs that must be matched to the data's mutability and growth characteristics.

## Frameworks Introduced
- **Full Loader**: ingest the complete dataset every run via extract-and-load (EL) or extract-transform-load (ETL).
  - When to use: slowly evolving, small-ish datasets with no delta/change-tracking column (e.g., a device reference dataset with <1M rows).
  - How: EL for homogeneous stores (native copy commands); ETL with a thin transform layer for heterogeneous stores; protect consumers from partial writes via transactions or a view-swap (single data exposition abstraction).
- **Incremental Loader**: ingest only new/changed data since the last run.
  - When to use: continuously growing, append-mostly datasets.
  - How: (1) delta-column implementation — track last-ingested value (e.g., ingestion_time) and filter `WHERE delta_column BETWEEN last_run AND now`; (2) partition-based implementation — ingest the next time-partition directly, using Readiness Marker to confirm it's complete.
- **Change Data Capture (CDC)**: stream row-level changes directly from a database's commit log.
  - When to use: low-latency requirements (e.g., <30s) or need to capture physical deletes without relying on soft deletes.
  - How: attach a log-based connector (e.g., Debezium + Kafka Connect) to the DB's replication log; treat the output as data-in-motion with CDC metadata (operation type, commit version/timestamp).
- **Passthrough Replicator**: copy data as-is between homogeneous stores, preserving metadata.
  - When to use: need an identical copy across environments (e.g., dev/staging mirroring prod) and the source is non-idempotent.
  - How: simplest possible EL job (raw text API, not structured JSON API) or infrastructure-level replication policy; always push from the owning environment, never pull into it.
- **Transformation Replicator**: replicate with a transformation step (filtering/anonymizing) in between.
  - When to use: replicated data contains PII or sensitive fields that can't cross environments as-is.
  - How: add a mapping function or SQL SELECT between read and write; remove columns (EXCEPT / drop) or alter them (substring, hashing via Anonymizer pattern); ideally driven by a data catalog/data contract tagging sensitive fields.
- **Compactor**: merge many small files into fewer, larger files.
  - When to use: metadata/listing overhead dominates processing time (the "small files problem").
  - How: use native commands — Delta Lake `OPTIMIZE`/`executeCompaction()`, Iceberg rewrite-data-files action, Hudi merge-on-read compaction, Kafka log compaction (keep latest value per key); follow with a VACUUM/cleaning step to reclaim space from superseded files.
- **Readiness Marker**: signal that a dataset is complete and safe to consume.
  - When to use: downstream consumers need to avoid reading partial/incomplete datasets.
  - How: flag file (e.g., Spark's `_SUCCESS`, a custom `COMPLETED` file written last) for file-based datasets; partition-naming convention (wait for partition N+1 to know partition N is closed) for time-partitioned datasets.
- **External Trigger**: event-driven (push-based) ingestion instead of fixed schedules.
  - When to use: data arrival is unpredictable/event-driven (e.g., irregular upstream releases).
  - How: (1) subscribe to a notification channel, (2) a handler filters/reacts to events, (3) the handler triggers an orchestration pipeline or job directly; prefer push (serverless function triggered by event) over pull (long-running polling job).

## Key Concepts
- **EL vs. ETL**: Extract-Load has no transformation (homogeneous stores); Extract-Transform-Load adds a transform layer (heterogeneous stores).
- **Delta column**: a timestamp/version column used to detect new/changed rows since the last run.
- **Soft delete vs. hard delete**: soft delete marks a row removed via UPDATE (detectable incrementally); hard delete physically removes it (invisible to delta-column incremental loads — needs CDC).
- **Insert-only (append-only) table**: accepts only INSERTs, pushing reconstruction of current state onto consumers.
- **Data loading vs. replication**: loading is flexible and can cross heterogeneous stores; replication targets the same storage type and ideally preserves all metadata (keys, offsets).
- **Small files problem**: excess of tiny files causes expensive metadata/listing operations relative to actual data processing time.
- **Pull vs. push semantics**: pull = consumer/trigger repeatedly checks for new data; push = producer actively notifies on new data.

## Mental Models
- Use the Full Loader when there's no reliable "what changed" signal; use the Incremental Loader the moment you have a delta column or time partitions.
- Think of CDC as converting data-at-rest into data-in-motion — joins and "no match" results mean something different (data not yet arrived, not "no match exists").
- Treat compaction as a cost-vs-freshness dial: compact rarely (cheap, batched) vs. compact often/inline (expensive, but consumers always benefit) — no one-size-fits-all answer.
- Readiness Marker is pull-based (consumer polls for a flag); External Trigger is push-based (producer notifies) — pick push when volume/predictability makes polling wasteful.

## Anti-patterns
- **Drop-and-insert full reloads without transactions/view abstraction**: exposes consumers to partial or empty data mid-load.
- **Using event time as the delta column**: silently misses late-arriving data for already-processed timestamps.
- **Ignoring metadata when replicating**: copying only Parquet files from a Delta table, or only key/value from Kafka (omitting headers/order), produces an unusable replica.
- **Compacting without a follow-up cleaning/VACUUM step**: old small files remain and continue to hurt listing performance.
- **Treating Readiness Marker conventions as enforced guarantees**: nothing stops a consumer from reading before the marker exists; must be explicitly communicated.
- **Hardcoding credentials in trigger handlers**: shown deliberately for readability in examples, but called out as bad practice — use a Ch 7 security pattern instead.

## Code Examples
```bash
# Full Loader: simple bucket sync (homogeneous stores)
aws s3 sync s3://input-bucket s3://output-bucket --delete
```
- **What it demonstrates**: simplest EL-style Full Loader implementation using native cloud tooling.

```python
# Incremental Loader: delta-column filter with explicit time boundaries (avoids backfill blow-up)
in_data = (spark_session.read.text(input_path).select('value',
   functions.from_json(functions.col('value'), 'ingestion_time TIMESTAMP')))
input_to_write = in_data.filter(
  f'ingestion_time BETWEEN "{date_from}" AND "{date_to}"'
)
input_to_write.mode('append').select('value').write.text(output_path)
```
- **What it demonstrates**: bounding the ingestion window so reruns/backfills don't balloon in volume.

```json
{
  "name": "visits-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres", "database.port": "5432",
    "database.user": "postgres", "database.password": "postgres",
    "database.dbname": "postgres", "database.server.name": "dbserver1",
    "schema.include.list": "dedp_schema",
    "topic.prefix": "dedp"
  }
}
```
- **What it demonstrates**: CDC via Debezium/Kafka Connect — configuration-driven commit-log streaming into Kafka topics.

```python
# Compactor: Delta Lake OPTIMIZE + VACUUM
devices_table = DeltaTable.forPath(spark_session, table_dir)
devices_table.optimize().executeCompaction()
devices_table.vacuum()
```
- **What it demonstrates**: merging small files into larger ones, then reclaiming space from superseded files.

## Reference Tables
| Pattern | Latency | Setup complexity | Handles hard deletes? |
|---|---|---|---|
| Full Loader | Low frequency, high per-run cost | Low | N/A (full overwrite) |
| Incremental Loader | Medium | Low–Medium | Only via soft deletes |
| CDC | Lowest | High (DB-level access/log setup) | Yes, natively |

| Trigger style | Mechanism | Resource efficiency |
|---|---|---|
| Readiness Marker (pull) | Consumer polls for flag file / next partition | Wastes cycles if polling tight |
| External Trigger (push) | Producer notifies via event/message bus | Efficient, no idle polling |

## Worked Example
Blog analytics case study: legacy visit events land in a transactional database while most visits stream in real time. An Incremental Loader first ingests legacy visits into the Bronze layer using a delta column (ingestion_time), bounded per run to avoid backfill blow-ups. When latency requirements tighten (consumers need changes within 30 seconds) and hard deletes must be captured, the team swaps to CDC via Debezium against PostgreSQL's logical replication stream, publishing changes to a `dedp.dedp_schema.events` Kafka topic. Downstream, an Airflow DAG with a `FileSensor` waits on the Readiness Marker (`_SUCCESS` or a custom `COMPLETED` file) before triggering a `SparkKubernetesOperator` load job, while a separate AWS Lambda function provides an External Trigger whenever a new object lands in a monitored S3 bucket, posting to Airflow's REST API to start the `devices-loader` DAG with full execution context (trigger metadata, event time) for later debugging.

## Key Takeaways
1. Pick Full vs. Incremental vs. CDC based on data mutability, delta-tracking availability, and latency SLA — not habit.
2. Always isolate consumers from partial writes (transactions or view-swap) when overwriting a dataset.
3. Replicate metadata, not just raw bytes — otherwise the replica is functionally incomplete.
4. Compaction needs a paired cleanup (VACUUM) step, and its frequency is a deliberate cost/performance trade-off.
5. Prefer push-based External Triggers over pull-based polling once event volume/unpredictability makes polling wasteful.
6. Readiness Markers are conventions, not enforcement — document them clearly with consumers.

## Connects To
- **Ch 3 (Error Management)**: External Trigger pipelines should be designed for failure using patterns like Dead-Letter.
- **Ch 4 (Idempotency)**: Full Loader's view-swap technique foreshadows the Proxy pattern discussed there.
- **Ch 7 (Data Security)**: Transformation Replicator's field removal connects to the Anonymizer and Fine-Grained Accessor patterns.
- **Apache Spark, Delta Lake, Apache Iceberg, Apache Hudi, Apache Kafka, Debezium, Apache Airflow**: the primary technologies used to implement these patterns.
