# Glossary

**Allowed lateness** — the configured margin subtracted from the tracked event time to form a watermark, defining how late data can be and still be accepted (Ch 3).

**Anonymization** — irreversibly removing/altering sensitive data so re-identification is impossible even combined with other datasets (Ch 7).

**Append time / ingestion time** — the immutable time a record was physically written to a broker/store, used for idempotent key generation instead of mutable event time (Ch 4).

**At-least-once / at-most-once / exactly-once** — delivery modes defined by whether checkpointing happens after (at-least-once, risk of duplicates) or before (at-most-once, risk of data loss) processing; exactly-once needs idempotency on top (Ch 3, Ch 4).

**AWAP (Audit-Write-Audit-Publish)** — validating both input and output data around a transformation step, with configurable fail/dispatch/nonblocking outcomes (Ch 9).

**Backfilling** — reprocessing past data, whether previously processed or not; a key stress-test for idempotency, lateness, and sessionization patterns (Ch 1, Ch 3, Ch 4, Ch 5).

**Backward / forward / full compatibility** — schema evolution modes defining whether a new schema can read old data, an old schema can read new data, or both (Ch 9).

**Bucket pruning** — a query optimization that skips buckets guaranteed not to match a filter predicate (Ch 8).

**Bucketing (clustering)** — colocating high-cardinality key values into a fixed number of hash-based buckets instead of partitioning (Ch 8).

**CDC (Change Data Capture)** — streaming row-level changes directly from a database's commit log for low-latency ingestion (Ch 2).

**Checkpointing** — persisting job progress (offsets/state) so a restart resumes instead of reprocessing from scratch (Ch 3).

**Compaction** — merging small files into larger ones (table formats) or keeping only the latest value per key (Kafka log compaction) (Ch 2, Ch 7).

**Constraints Enforcer** — delegating data validation to database/format-native constraints (type, nullability, value, integrity) (Ch 9).

**Data dispatching** — splitting an audited output into a valid (promoted) and invalid (dead-lettered) subset based on explicit validation logic, not runtime exceptions (Ch 9).

**Data lineage** — tracking dependencies between datasets (Dataset Tracker) or at column/row granularity (Fine-Grained Tracker) (Ch 10).

**Data skew** — an unbalanced dataset where one key/partition/run dominates volume, causing latency or false quality alarms (Ch 5, Ch 8, Ch 10).

**Dead-Letter** — redirecting unprocessable records to a separate store instead of failing the whole job (Ch 3).

**Deletion vector** — a table-file-format technique recording removed rows in a small side file instead of rewriting main data files immediately (Ch 7).

**Denormalization** — flattening joined data into fewer, wider tables (One Big Table, star schema) to eliminate query-time joins (Ch 8).

**Distribution key** — the attribute used to physically divide a dataset for horizontal partitioning (Ch 8).

**Dynamic Task Mapping** — Apache Airflow's feature for generating tasks dynamically at runtime, used to create per-partition backfill tasks (Ch 3).

**Event time vs. processing time** — event time is when something happened (can be late); processing time is when the pipeline handled it (never late) (Ch 3).

**Fast Metadata Cleaner** — using TRUNCATE/DROP (metadata operations) instead of DELETE for fast, idempotent dataset resets (Ch 4).

**Fine-Grained Accessor** — restricting access below table level (columns, rows) or to specific cloud resources (Ch 7).

**Garbage collection (GC) watermark** — the mechanism evicting buffered join/session state too old to ever match or complete (Ch 5).

**Horizontal partitioning** — physically isolating whole rows by a (typically low-cardinality) distribution key (Ch 8).

**Idempotency** — the property that running a job any number of times yields the same (or clearly identifiable) output (Ch 4).

**Lag** — the difference between the most recently available data unit and the most recently processed one, aggregated via MAX (worst case) or percentile (overall health) (Ch 10).

**Late data** — data arriving after its event time's expected processing window has closed, detected via watermarks (Ch 3).

**Manifest** — a recorded list of files belonging to a dataset/version, avoiding repeated expensive storage listing (Ch 8).

**MapReduce / shuffle** — the distributed aggregation model where related records are exchanged across the network to be co-located for reduction (Ch 5).

**Merger (UPSERT)** — combining incremental new/changed rows into an existing dataset via a MERGE statement (Ch 4).

**Metadata Decorator** — attaching technical/operational context (job version, batch number) without exposing it to end users (Ch 5).

**Metadata Enhancer** — persisting per-file/table statistics (min/max, null counts) to skip irrelevant data without reading it (Ch 8).

**Nonblocking audit** — promoting imperfect data anyway, annotated with a quality flag, rather than blocking the pipeline (Ch 9).

**Normal forms (1NF/2NF/3NF)** — progressively stricter rules for eliminating repeating groups, partial key dependency, and transitive dependency in normalized schemas (Ch 8).

**Partial commit** — a bulk write outcome where only some records succeed, risking broken ordering on retry (Kinesis `PutRecords`, DynamoDB `BatchWriteItem`) (Ch 5).

**Passthrough job** — a pipeline moving data without transformation (extract-and-load) (Ch 1, Ch 2).

**Pseudo-anonymization** — replacing sensitive values with a form retaining more utility but theoretically reversible/re-identifiable when combined with other data (masking, tokenization, hashing, encryption) (Ch 7).

**Readiness Marker** — signaling that a dataset is complete and safe to consume, via a flag file or partition-naming convention (Ch 2).

**Sessionization** — building sessions (init → accumulate → finalize) from batch partitions (Incremental Sessionizer) or streaming state (Stateful Sessionizer) (Ch 5).

**Shard** — a physical-machine-level special case of horizontal partitioning that actually moves data across hardware (Ch 8).

**Slowly Changing Dimensions (SCD)** — data modeling strategies (type 2: validity date range; type 4: current + history tables) tracking entity history over time (Ch 5).

**Small files problem** — excess tiny files causing expensive metadata/listing operations relative to actual data processing time (Ch 2).

**Snowball backfilling effect** — replaying dead-lettered or late data forcing downstream consumers (and their consumers) to backfill too, cascading compute cost (Ch 3).

**Soft delete** — representing a deletion as an UPDATE with a flag rather than a physical DELETE (Ch 2, Ch 4).

**Staging/promotion pattern** — writing a transformed dataset to a private staging location first, then atomically promoting it over the public location (Ch 7).

**State store** — a persistent/in-memory store of already-seen keys or accumulated state for stateful streaming operations (Ch 3, Ch 5).

**Tombstone message** — a Kafka record with a real key and null value, used with log compaction to physically remove a key's prior values (Ch 7).

**Transaction scope** — the commit/rollback boundary determining what "idempotent" actually covers (Ch 4).

**Vertical partitioning** — splitting a row's columns across separate storage locations, for PII removal (Ch 7) or deduplication/storage optimization (Ch 8).

**Watermark** — the minimum event time still accepted as "on time," computed as MAX(event time) minus allowed lateness; must be monotonically increasing (Ch 3).

**WORM (Write Once Read Many)** — storage-enforced immutability (object lock / immutability policy) preventing writes after creation (Ch 4).

**Z-order** — a curved (non-lexicographical) sort colocating rows across multiple columns more efficiently than top-to-bottom ordering (Ch 8).
