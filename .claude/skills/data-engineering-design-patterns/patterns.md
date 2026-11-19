# All Named Design Patterns

## Full Loader (Ch2)
**When to use** No delta-tracking attribute; small/slow-changing dataset.
**How** EL/ETL job rewrites the full dataset; isolate consumers via transactions or a view-swap.
**Trade-offs** Costly at scale; risk of exposing partial data without consistency protection.

## Incremental Loader (Ch2)
**When to use** Continuously growing dataset with a delta column or time partitions.
**How** Filter by last-ingested value or ingest the next time partition.
**Trade-offs** Hard deletes invisible unless soft-deleted; backfills can balloon to full loads unless window-bounded.

## Change Data Capture (Ch2)
**When to use** Low-latency, delete-aware ingestion from a transactional source.
**How** Stream the DB commit log (e.g., Debezium/Kafka Connect) into a broker/table-feed.
**Trade-offs** Higher setup complexity; adds operation-type metadata; data becomes "in motion" with different join semantics.

## Passthrough Replicator (Ch2)
**When to use** Need an identical, non-idempotent-source copy across environments.
**How** Simplest possible EL job or infra replication policy; always push from the owner.
**Trade-offs** Must preserve metadata exactly; fails for PII (use Transformation Replicator).

## Transformation Replicator (Ch2)
**When to use** Replicated data has PII/sensitive fields needing removal or masking.
**How** Add a filter/mapping transform between read and write.
**Trade-offs** Higher data-corruption risk; needs a governance tool to stay synced with sensitivity tags.

## Compactor (Ch2)
**When to use** Small-files problem is hurting listing/metadata performance.
**How** Native OPTIMIZE/rewrite commands (Delta/Iceberg/Hudi) or Kafka log compaction.
**Trade-offs** Needs a paired VACUUM/cleanup step; frequency is a cost-vs-freshness trade-off.

## Readiness Marker (Ch2)
**When to use** Consumers must avoid reading incomplete datasets.
**How** Flag file (`_SUCCESS`) or partition-naming convention (wait for N+1).
**Trade-offs** Convention, not enforcement; fragile under late data.

## External Trigger (Ch2)
**When to use** Unpredictable/event-driven data arrival.
**How** Subscribe → handler filters/reacts → triggers pipeline/job (prefer push over pull).
**Trade-offs** Needs rich execution context metadata; must design for event-delivery failure.

## Dead-Letter (Ch3)
**When to use** Nontransient bad records must not stop the pipeline.
**How** Try-catch/if-else around risky logic; route failures to a monitored sink; optional replay.
**Trade-offs** Risk of snowball backfilling; can hide real systemic failures without alerting.

## Windowed Deduplicator (Ch3)
**When to use** At-least-once delivery produces duplicates needing single-occurrence processing.
**How** DISTINCT/ROW_NUMBER (batch) or state-store-backed time window (streaming).
**Trade-offs** Space-vs-completeness window sizing; doesn't guarantee exactly-once delivery.

## Late Data Detector (Ch3)
**When to use** Need to classify records as on-time vs. late.
**How** MAX per-partition event time minus allowed lateness = watermark.
**Trade-offs** MIN causes stuck-in-the-past; MAX across sources risks dropping slow-source data.

## Static Late Data Integrator (Ch3)
**When to use** Late data has a known fixed max delay.
**How** Fixed lookback window reprocessed every run.
**Trade-offs** Wastes resources on empty windows; backfills must not overlap.

## Dynamic Late Data Integrator (Ch3)
**When to use** Late data delay is unpredictable/unbounded.
**How** State table tracks per-partition last-update vs. last-processed time.
**Trade-offs** Needs concurrency guard (`is_processed`); heavier state-tracking complexity.

## Filter Interceptor (Ch3)
**When to use** Need to know which filter condition removed how many rows.
**How** Counter/accumulator per condition, or status-flag subquery column in SQL.
**Trade-offs** Harder in declarative SQL and streaming; adds minor runtime overhead.

## Checkpointer (Ch3)
**When to use** Continuous jobs need fault-tolerant restart.
**How** Framework- or store-managed offset/state persistence.
**Trade-offs** Only gives at-least/at-most-once, not exactly-once by itself.

## Fast Metadata Cleaner (Ch4)
**When to use** Large partitioned datasets need fast idempotent resets.
**How** TRUNCATE/DROP instead of DELETE, at partition granularity.
**Trade-offs** Backfill granularity tied to partition size; hits data-store metadata limits.

## Data Overwrite (Ch4)
**When to use** No metadata-cleaner support; full dataset replaced each run.
**How** Native overwrite mode / INSERT OVERWRITE / DELETE+INSERT; follow with VACUUM.
**Trade-offs** Slow on large unpartitioned data.

## Merger (Ch4)
**When to use** Incremental dataset must be reconciled without full overwrite.
**How** MERGE on immutable key; deletes modeled as soft-delete flag.
**Trade-offs** No native hard-delete support; inconsistent mid-backfill results (use Stateful Merger).

## Stateful Merger (Ch4)
**When to use** Merger's backfill inconsistency is unacceptable.
**How** State table of {execution_time, table_version}; restore to prior version before re-merging.
**Trade-offs** Requires a versioned store; compaction-only commits shift version math.

## Keyed Idempotency (Ch4)
**When to use** Key-based store; need writes to naturally dedupe on replay.
**How** Build keys from immutable attributes (append/ingestion time, never event time).
**Trade-offs** Relational DBs need MERGE not INSERT; compaction can desync duplicate visibility.

## Transactional Writer (Ch4)
**When to use** Must prevent partial/duplicated data visibility on failure.
**How** Begin/commit/rollback around writes (native or framework-managed).
**Trade-offs** Idempotency limited to transaction scope; not all distributed frameworks support it (e.g., Kafka+Spark).

## Proxy (Ch4)
**When to use** Dataset must remain write-once/immutable.
**How** Write-once table/location with revoked permissions + indirection view/manifest.
**Trade-offs** Needs WORM/permission enforcement from infra; natively versioned stores skip manual permissions.

## Static Joiner (Ch5)
**When to use** Enrichment dataset is at rest/slowly changing.
**How** JOIN on key; SCD type 2/4 for time-sensitivity; materialize API data for idempotency.
**Trade-offs** Late data breaks consistency; API-backed joins risk non-idempotent results.

## Dynamic Joiner (Ch5)
**When to use** Both join sides are streaming.
**How** Time-bounded buffer + GC watermark on both sides.
**Trade-offs** Space-vs-exactness trade-off; late data still causes missed joins.

## Wrapper (Ch5)
**When to use** Need raw + computed fields both visible to consumers.
**How** Nest computed attrs alongside raw ones (flat/nested/two-table variants).
**Trade-offs** Domain split across two structures; increases payload size.

## Metadata Decorator (Ch5)
**When to use** Technical context must stay hidden from consumers.
**How** Native metadata (Kafka headers, object tags) or hidden column/table + access restriction.
**Trade-offs** Not all stores support native metadata; never put business data here.

## Distributed Aggregator (Ch5)
**When to use** Dataset too large for one machine; needs grouped/reduced stats.
**How** Group-by + reduce with network shuffle; mitigate skew with salting.
**Trade-offs** Shuffle network cost; scaling/reclaiming nodes needs a shuffle service.

## Local Aggregator (Ch5)
**When to use** Data is statically, correctly pre-partitioned by grouping key.
**How** Per-partition operations (bucketed tables, `groupByKey`) avoid shuffle.
**Trade-offs** Breaks silently if partitioning/keys change; rescaling is costly.

## Incremental Sessionizer (Ch5)
**When to use** Batch sessions span multiple time partitions.
**How** Input + completed + pending storage zones; init/accumulate/finalize logic.
**Trade-offs** Forward-dependent backfilling; partial sessions need explicit completeness flags.

## Stateful Sessionizer (Ch5)
**When to use** Near-real-time sessions needed from a streaming layer.
**How** State store + session windows or arbitrary stateful processing, event-time expiration.
**Trade-offs** Only at-least-once state persistence; state rebalancing complicates scaling.

## Bin Pack Orderer (Ch5)
**When to use** Bulk sink has partial-commit semantics and ordering matters.
**How** Sort by key+time; pack one record per key per bin; emit bins sequentially.
**Trade-offs** More complex than plain sort; whole-pipeline retries still break cross-execution order.

## FIFO Orderer (Ch5)
**When to use** Low-volume, strict-order delivery without bulk complexity.
**How** Per-record/small-batch send-then-ack (or idempotent producer, limited in-flight requests).
**Trade-offs** Higher I/O/latency overhead; FIFO ≠ exactly-once.

## Local Sequencer (Ch6)
**When to use** Decompose a monolith job/pipeline for readability/cheap restarts.
**How** Order tasks by data dependency; pick orchestration vs. processing-layer boundary.
**Trade-offs** Wrong boundaries cause scheduling overhead or lost restart isolation.

## Isolated Sequencer (Ch6)
**When to use** Coordinate pipelines owned by different teams.
**How** Data-based (Readiness Marker, loose) or task-based (direct trigger, tight) dependency.
**Trade-offs** Task-based coupling breaks on rename/removal; needs cross-team communication discipline.

## Aligned Fan-In (Ch6)
**When to use** Merge requires all parent branches to succeed.
**How** Parallel branches merge via UNION/JOIN into a common task.
**Trade-offs** Scheduling skew from the slowest branch; infra spikes from parallelism.

## Unaligned Fan-In (Ch6)
**When to use** Partial results acceptable/preferred over blocking.
**How** Relaxed trigger rule (ALL_DONE/Catch) + explicit completeness flag.
**Trade-offs** Reduced readability; must signal partial data clearly to consumers.

## Parallel Split (Ch6)
**When to use** One dataset feeds multiple independent outputs/consumers.
**How** Cache/materialize the shared intermediate once; branch writes from it.
**Trade-offs** Pipeline blocked by slowest branch; duplicate-write risk on retry.

## Exclusive Choice (Ch6)
**When to use** Mutually exclusive logic paths (version cutover, schema branch).
**How** Condition-evaluator task before branching; prefer metadata over data conditions.
**Trade-offs** Too many branches become a "complexity factory."

## Single Runner (Ch6)
**When to use** Pipeline is stateful/incremental; concurrent runs would corrupt results.
**How** Concurrency = 1 (`max_active_runs=1`, `depends_on_past=True`).
**Trade-offs** Backfills become slow and strictly sequential.

## Concurrent Runner (Ch6)
**When to use** Pipeline instances are independent of each other.
**How** Concurrency > 1, bounded by workload management.
**Trade-offs** Resource starvation risk in multi-tenant orchestrators; shared-state hazards if misapplied.

## Vertical Partitioner — Security (Ch7)
**When to use** Cheap personal-data removal across repeated immutable attributes.
**How** Split mutable/PII column groups into separate stores, joined by a key.
**Trade-offs** Query complexity; raw unsplit data needs its own removal strategy.

## In-Place Overwriter (Ch7)
**When to use** Legacy/uncontrolled storage without a split-friendly layout.
**How** Native DELETE+VACUUM, or stage-filter-then-promote for raw formats; Kafka tombstones.
**Trade-offs** High I/O cost; prior version becomes unrecoverable without Proxy/versioning.

## Fine-Grained Accessor for Tables (Ch7)
**When to use** Restrict access below table level (column/row).
**How** GRANT on columns, masking functions, row-security policies or guard views.
**Trade-offs** Limited to session-available attributes; complex nested types need unnesting.

## Fine-Grained Accessor for Resources (Ch7)
**When to use** Least-privilege access on cloud resources (buckets, streams).
**How** Resource-based or identity-based IAM policy, optionally tag-scoped.
**Trade-offs** Many narrow policies hurt maintainability; mixing both models adds complexity.

## Encryptor (Ch7)
**When to use** Data must stay unusable if stolen/intercepted, beneath access control.
**How** Server-/client-side encryption at rest via KMS; TLS minimum version in transit.
**Trade-offs** CPU overhead; key loss risk (mitigate with soft-deleted keys).

## Anonymizer (Ch7)
**When to use** Sensitive data must be irreversibly unidentifiable, even combined with other data.
**How** Remove, perturb, or synthetically replace values.
**Trade-offs** Significant information/utility loss.

## Pseudo-Anonymizer (Ch7)
**When to use** Need more analytical utility than full anonymization allows.
**How** Masking, tokenization, hashing, or encryption.
**Trade-offs** Re-identifiable when combined with other datasets — weaker guarantee, false sense of security.

## Secrets Pointer (Ch7)
**When to use** Avoid embedding credentials in code/config.
**How** Reference-by-name lookup from a secrets manager at runtime.
**Trade-offs** Cache invalidation on rotation; never log the resolved secret.

## Secretless Connector (Ch7)
**When to use** Eliminate static credentials entirely.
**How** IAM/identity-based auth (assume-role/STS) or certificate-based auth.
**Trade-offs** Certificate rotation still needs management overhead.

## Horizontal Partitioner (Ch8)
**When to use** Low-cardinality distribution key, time-bounded/incremental access.
**How** Partition by time/business key at write or table-creation time.
**Trade-offs** High-cardinality keys explode partition count; mutating the key needs costly data movement.

## Vertical Partitioner — Storage (Ch8)
**When to use** Avoid duplicating immutable attributes repeated per row.
**How** Split mutable/immutable column groups into separate locations.
**Trade-offs** Query joins required; producer must multi-write.

## Bucket (Ch8)
**When to use** High-cardinality column heavily used in filters/joins.
**How** Hash(key) % N buckets; enables pruning and shuffle-free joins.
**Trade-offs** Bucket schema is effectively immutable; choosing bucket count is a prediction problem.

## Sorter (Ch8)
**When to use** Queries consistently filter/sort on known columns.
**How** Declare sort/cluster columns, or use curved sort (Z-order) for multi-column access.
**Trade-offs** New writes create unsorted segments; composite-key order must match query predicate order.

## Metadata Enhancer (Ch8)
**When to use** Skip irrelevant files/blocks without reading them.
**How** Columnar footer stats (min/max/null-count) or database table statistics.
**Trade-offs** Stale statistics degrade query plans; slight write-time overhead.

## Dataset Materializer (Ch8)
**When to use** Same expensive query is run repeatedly by many consumers.
**How** Materialized view (auto-refresh) or manually managed table; prefer incremental refresh.
**Trade-offs** Refresh cost; storage duplication; harder consistent access control across source tables.

## Manifest (Ch8)
**When to use** Object-store listing dominates latency/cost.
**How** Native commit-log manifests (Delta/Iceberg/Hudi) or explicit manifest files.
**Trade-offs** Manifest size can grow unbounded with continuous small-file writers.

## Normalizer (Ch8)
**When to use** Consistency and cheap localized updates matter (OLTP).
**How** Normal forms (1NF/2NF/3NF) or snowflake dimensional model.
**Trade-offs** Costly multi-join queries; mitigate via broadcast/colocation joins.

## Denormalizer (Ch8)
**When to use** Read latency/cost from joins outweighs consistency needs (OLAP).
**How** One Big Table (flat/nested) or star schema.
**Trade-offs** Costly, duplicated updates; risk of becoming an unmaintainable "trash bag" table.

## Audit-Write-Audit-Publish (Ch9)
**When to use** Need to catch both bad input and transformation bugs before publishing.
**How** Pre-transform input audit + post-transform output audit; fail/dispatch/nonblocking outcomes.
**Trade-offs** Extra compute cost; rule coverage inevitably lags dataset evolution.

## Constraints Enforcer (Ch9)
**When to use** Validation can be delegated to the database/format natively.
**How** Type, nullability, value, and integrity constraints (e.g., Delta `CHECK`, Protobuf+protovalidate).
**Trade-offs** All-or-nothing, stops at first error; doesn't cover every consumer's specific needs.

## Schema Compatibility Enforcer (Ch9)
**When to use** Prevent producers from making breaking schema changes.
**How** External registry, implicit insert-time constraints, or DDL event triggers; choose backward/forward/full + transitive/nontransitive mode.
**Trade-offs** Interaction overhead; harder schema evolution by design.

## Schema Migrator (Ch9)
**When to use** A genuinely breaking change (rename/retype/remove) is unavoidable.
**How** Add new field alongside old, dual-run during a grace period, then deprecate the old field.
**Trade-offs** Temporary size/storage impact; field removal impossible without a safe migration path if still in use.

## Offline Observer (Ch9)
**When to use** Non-blocking quality monitoring without impacting production latency.
**How** Independently scheduled job with idempotent monitoring-state tracking.
**Trade-offs** Detection can lag; infrequent runs cost more compute per run.

## Online Observer (Ch9)
**When to use** Detection latency from Offline Observer is unacceptable.
**How** Observe within the pipeline (Parallel Split or Local Sequencer); isolate failures from the main flow.
**Trade-offs** Extra pipeline latency; Parallel Split risks missing load-time-only issues.

## Flow Interruption Detector (Ch10)
**When to use** A job could silently stop producing output without erroring.
**How** Continuous (zero-in-window) or irregular (no-data-window-threshold) delivery check; metadata/data/storage layer checks for batch.
**Trade-offs** Threshold tuning risk; compaction/housekeeping can create false positives.

## Skew Detector (Ch10)
**When to use** Upstream could silently deliver a partial/incomplete dataset.
**How** Window-to-window % comparison or standard-deviation ratio across partitions.
**Trade-offs** Seasonality causes false positives; fatality loop if baseline run was itself bad.

## Lag Detector (Ch10)
**When to use** Need early warning for a consumer falling behind a producer.
**How** `last_available_unit - last_processed_unit`, aggregated via MAX and/or percentile.
**Trade-offs** MAX alone conflates consumer slowness with upstream data skew.

## SLA Misses Detector (Ch10)
**When to use** Downstream has a hard deadline independent of per-record lag.
**How** End-start time vs. threshold (batch); per-microbatch or per-record latency (streaming); track processing-time and event-time SLAs separately.
**Trade-offs** Not interchangeable with Lag Detector; event-time SLA vulnerable to upstream late data.

## Dataset Tracker (Ch10)
**When to use** Need cross-team/cross-dataset dependency visibility.
**How** Managed lineage capture (Unity Catalog, Dataplex) or self-implemented (OpenLineage + orchestrator/query parsing).
**Trade-offs** Managed tools are scoped to their own platform; custom task types need manual lineage wiring.

## Fine-Grained Tracker (Ch10)
**When to use** Need column- or row-level data origin detail.
**How** Native column lineage, execution-plan analysis, or row-level lineage metadata in headers.
**Trade-offs** Opaque for custom programmatic transforms; row-level lineage needs a separate query layer.
