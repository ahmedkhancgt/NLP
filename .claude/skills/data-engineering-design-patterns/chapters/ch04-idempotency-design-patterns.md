# Chapter 4: Idempotency Design Patterns

## Core Idea
Error management (Ch 3) alone cannot guarantee correctness because retries and backfills can replay already-successful writes; idempotency patterns ensure that running a pipeline multiple times — whether from an automatic retry or a manual backfill — produces consistent output without unidentifiable duplicates, choosing a strategy based on whether the dataset can be fully overwritten, only partially merged, or must stay immutable.

## Frameworks Introduced
- **Fast Metadata Cleaner**: use metadata-only operations (TRUNCATE/DROP) instead of DELETE to reset a partition/table before rewriting.
  - When to use: large, partitioned/incrementally-loaded datasets where row-level DELETE has become too slow.
  - How: partition the dataset (e.g., weekly tables) at the idempotency granularity; on each run, branch (e.g., Exclusive Choice) on whether to start a new partition (TRUNCATE/DROP + recreate) or continue; expose all partitions through a single view.
- **Data Overwrite**: replace data at the file/row level using a native overwrite command when metadata operations aren't available.
  - When to use: object stores or data stores lacking TRUNCATE/DROP (e.g., raw files), or when the whole dataset is reloaded each run.
  - How: configure the writer's save/write mode to overwrite (Spark `mode('overwrite')`, Flink write mode); use SQL `DELETE + INSERT`, `INSERT OVERWRITE`, or store-native load-with-replace (BigQuery `--replace=true` / `LOAD DATA OVERWRITE`); use `replaceWhere` (Delta Lake) for selective overwrite; follow with VACUUM to reclaim space.
- **Merger** (aka UPSERT): combine incremental new/changed rows into an existing dataset via MERGE.
  - When to use: dataset is incremental (only a subset of rows changes each run) and a full overwrite isn't feasible.
  - How: define immutable uniqueness attributes; use a `MERGE` statement with WHEN MATCHED (update), WHEN NOT MATCHED (insert), and model deletes as soft deletes (an `is_deleted` flag) since MERGE cannot express hard deletes; explicitly guard the INSERT branch against `is_deleted = true` to avoid re-inserting already-removed rows on first run.
- **Stateful Merger**: extend Merger with a state table to restore the dataset to a consistent version before backfilling.
  - When to use: Merger's backfilling inconsistency (replaying an incremental dataset mixes old and new row sets) is unacceptable and the data store supports versioning.
  - How: maintain a state table of {execution_time, table_version}; before merging, compare the version created by the previous execution to the current table version — if different, restore the table to the prior valid version (e.g., Delta Lake `restoreToVersion`) before merging; for non-versioned stores, keep a full raw-history table and rebuild via Windowed Deduplicator; account for no-data operations (compaction) shifting version numbers by always restoring to `current_execution_version - 1`.
- **Keyed Idempotency**: generate deterministic keys so repeated writes overwrite rather than duplicate.
  - When to use: a key-based data store (NoSQL, Kafka) and ability to derive a stable, immutable key per logical entity (e.g., a session).
  - How: build keys from immutable attributes only — never event time (mutable due to late data) — prefer append/ingestion/insertion time (Kafka's append time, Kinesis's approximate arrival timestamp, or a store's insertion time) so the same key is generated across restarts/backfills.
- **Transactional Writer**: use database transactions (begin/commit/rollback) so partial writes are never visible.
  - When to use: node preemption, task failures, or retries could otherwise expose partial/duplicated data to consumers.
  - How: wrap writes in a transaction (explicit `BEGIN`/`COMMIT` or framework-managed, e.g., Flink `DeliveryGuarantee.EXACTLY_ONCE` for Kafka); understand the idempotency scope is limited to the transaction itself (task-local vs. whole-job transactions) — a backfill still creates a brand-new transaction and can duplicate data across runs.
- **Proxy**: insert an indirection layer (view/manifest) over physically immutable, write-once datasets.
  - When to use: data must remain immutable/append-only (e.g., legal/compliance retention) yet consumers need a single, current access point.
  — How: write each run to a new timestamped/versioned table or location with write permissions revoked after creation (WORM — AWS S3 Object Lock, Azure immutability policies, GCP object holds/bucket locks); expose the current version via a passthrough view, alias, or manifest file; for natively versioned stores (Delta Lake, Iceberg, BigQuery), rely on built-in time travel instead of manual table versioning.

## Key Concepts
- **Idempotency**: running a job any number of times yields the same (or clearly identifiable) output, analogous to `absolute(-1) == absolute(absolute(absolute(-1)))`.
- **Idempotency granularity**: the partition/table unit at which metadata-based cleanup operates; also defines the minimum backfill unit (Fast Metadata Cleaner).
- **Soft delete**: representing a deletion as an UPDATE with a flag (e.g., `is_deleted`), required because MERGE cannot express physical deletes for missing rows.
- **Append/ingestion time**: immutable "when this was physically written" timestamp, used for key generation instead of mutable event time.
- **Transaction scope**: commit/rollback boundary determining what "idempotent" actually covers (one task vs. the whole job).
- **Write once read many (WORM)**: storage-enforced immutability (object lock / immutability policy) preventing writes after creation.
- **Dirty read**: a reader under "read uncommitted" isolation seeing uncommitted (possibly-to-be-rolled-back) transactional writes.

## Mental Models
- Use Fast Metadata Cleaner when you can define a clean partition boundary and the backfill granularity cost is acceptable; use Data Overwrite when metadata ops aren't available but you can still afford whole-dataset replacement.
- Think of Merger as "I can't delete-and-replace, so I reconcile row by row" — and remember it inherits MERGE's blind spot on hard deletes.
- Stateful Merger is Merger + "undo button": it adds a state table purely to answer "what version do I roll back to before I replay?"
- Keyed Idempotency works by making the write itself a no-op on replay (same key → same row); Transactional Writer works by making the write atomic (all-or-nothing visibility) — pick the first when you control key generation, the second when you need atomicity across multiple writes.
- Proxy is "the recipe you already used in Ch 2's Full Loader view-swap" generalized to full dataset immutability: never mutate storage, only swap what the access point points to.

## Anti-patterns
- **Using row-level DELETE on huge, continuously growing tables for idempotency**: degrades badly as the table grows — use metadata ops (TRUNCATE/DROP) instead.
- **Omitting the `is_deleted` guard in a Merger INSERT branch**: reinserts already-deleted records on the first run.
- **Backfilling only part of a Fast-Metadata-Cleaner partition's granularity**: produces an inconsistent dataset — must replay from the partition-creation task.
- **Using mutable event time to generate idempotent keys**: late data changes the "first visit time," producing a different session key after a restart.
- **Assuming Kafka's keyed writes deduplicate immediately**: compaction is asynchronous, so consumers can see duplicates for a while.
- **Treating local (task-based) transactions as job-wide idempotency**: a job restart can still rewrite data from already-committed local transactions.
- **Giving broad write/delete permissions to the Proxy pattern's loader**: a user could delete prior immutable tables and break the write-once guarantee.

## Code Examples
```sql
-- Merger: soft-delete-aware MERGE covering insert/update/delete
MERGE INTO dedp.devices_output AS target
USING dedp.devices_input AS input
ON target.type = input.type AND target.version = input.version
WHEN MATCHED AND input.is_deleted = true THEN
DELETE
WHEN MATCHED AND input.is_deleted = false THEN
UPDATE SET full_name = input.full_name
WHEN NOT MATCHED AND input.is_deleted = false THEN
INSERT (full_name, version, type) VALUES (input.full_name, input.version, input.type)
```
- **What it demonstrates**: handling insert/update/soft-delete in one idempotent MERGE, guarding inserts against already-deleted rows.

```python
# Stateful Merger: restore to prior version before replaying a backfill
if not maybe_previous_job_version:
  spark.sql('TRUNCATE TABLE default.devices')
else:
  previous_job_version = maybe_previous_job_version[0].delta_table_version
  if previous_job_version < last_merge_version:
    current_run_version = (...)
    version_to_restore = current_run_version - 1
    (DeltaTable.forName(spark, 'devices').restoreToVersion(version_to_restore))
```
- **What it demonstrates**: detecting a backfill via a state table and rolling the Delta table back to a consistent version before re-merging.

```sql
-- Window operation using append/ingestion time to generate a stable idempotent key
SELECT ... OVER (PARTITION BY user_id ORDER BY ingestion_time ASC, visit_time ASC)
```
- **What it demonstrates**: Keyed Idempotency — sorting by an immutable ingestion time so late data never changes the derived key.

```python
# Proxy: generate a unique, write-once internal table name per run
def get_devices_table_name() -> str:
  context = get_current_context()
  dag_run: DagRun = context['dag_run']
  table_suffix = dag_run.start_date.strftime('%Y%m%d_%H%M%S')
  return f'dedp.devices_internal_{table_suffix}'
```
- **What it demonstrates**: each pipeline run writes to a distinct immutable table, later exposed through a single view.

## Reference Tables
| Pattern | Data availability needed | Mechanism | Backfill-safe? |
|---|---|---|---|
| Fast Metadata Cleaner | Full dataset per run | TRUNCATE/DROP (metadata) | Yes, at partition granularity only |
| Data Overwrite | Full dataset per run | DELETE+INSERT / INSERT OVERWRITE | Yes |
| Merger | Incremental dataset | MERGE/UPSERT | No (mixes old/new rows) — use Stateful Merger |
| Stateful Merger | Incremental + versioned store | MERGE + version-based restore | Yes |
| Keyed Idempotency | Any | Deterministic key reuses same row | Yes, if key built from immutable attrs |
| Transactional Writer | Any | Commit/rollback atomicity | Only within transaction scope |
| Proxy | Immutable requirement | Write-once table/file + indirection view | Yes, by construction |

## Worked Example
Blog analytics case study: a daily batch job merges CDC changes from a Kafka topic into a Delta Lake `devices` table (Merger pattern) using `type`/`version` as the match key and an `is_deleted` soft-delete flag. When the business later demands consistent backfills, the team adds a `versions` state table recording each run's resulting Delta table version; before each MERGE, the pipeline checks whether the previous run's recorded version differs from the table's latest version — if so, it restores the table to `current_run_version - 1` before re-merging (Stateful Merger), correctly accounting for compaction-only commits that don't carry data changes. Separately, a Spark Structured Streaming job builds user sessions from Kafka visit events and writes them to ScyllaDB keyed by `(session_id, user_id)`, where `session_id` is derived from the earliest Kafka *append time* per group (not event time) so that late-arriving records never change the session's idempotent key across job restarts (Keyed Idempotency). Finally, because legal requires every daily device snapshot retained immutably, the team switches to the Proxy pattern: each run writes to a uniquely timestamped internal table with write permissions revoked, and a view `dedp.devices` is repointed to the latest table after each load.

## Key Takeaways
1. Prefer metadata operations (TRUNCATE/DROP) over row-level DELETE for idempotent resets on large partitioned datasets.
2. MERGE-based idempotency (Merger) cannot express hard deletes — model deletions as soft deletes with an explicit flag.
3. Stateful Merger is necessary whenever backfilling an incremental Merger-based pipeline must produce a consistent (not partially-mixed) dataset.
4. Build idempotent keys from immutable timestamps (append/ingestion time), never from mutable event time.
5. Transactions guarantee atomic visibility but only within their own scope — they don't by themselves prevent duplicate data across retried/backfilled runs.
6. When data must be immutable, the Proxy pattern (write-once storage + indirection) is the only safe idempotency strategy — don't try to force overwrite/merge patterns onto immutable requirements.

## Connects To
- **Ch 2 (Data Ingestion)**: Fast Metadata Cleaner and Proxy both build on the Full Loader's view-swap/single-exposition-abstraction idea; Merger builds on CDC and Incremental Loader's soft-delete handling.
- **Ch 3 (Error Management)**: checkpointing gives an "exactly-once feeling" but needs these idempotency patterns for a real guarantee; Windowed Deduplicator is reused inside the non-versioned Stateful Merger implementation.
- **Ch 5 (Data Value)**: idempotency is the technical foundation on top of which business-value-generating transformations (next chapter) can safely run.
- **Delta Lake, Apache Iceberg, Apache Hudi, Apache Kafka, Apache Flink, Apache Spark, ScyllaDB, BigQuery, Amazon Kinesis**: primary technologies implementing these patterns.
