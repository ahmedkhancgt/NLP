# Cheatsheet: Decision Rules & Reasoning Aids

## Ingestion: which loader?
- No delta attribute, small/slow dataset → **Full Loader**
- Delta column or time partitions available, append-mostly → **Incremental Loader**
- Need <30s latency OR must capture physical deletes → **Change Data Capture**
- Smell: incremental job suddenly doing a full scan on backfill → bound the ingestion window (`delta BETWEEN t AND t+interval`).

## Error handling: which pattern for which failure?
- Nontransient "poison pill" record, job must keep running → **Dead-Letter**
- At-least-once delivery creating duplicates → **Windowed Deduplicator** (batch: DISTINCT/window; streaming: state store + watermark)
- Records arriving after their event-time window closed → **Late Data Detector** + (**Static** if max delay known/fixed, **Dynamic** if unbounded/unpredictable)
- Filtered-row-count spike, unclear if data or code regression → **Filter Interceptor** before blaming either side
- Streaming job needs restart-without-reprocessing → **Checkpointer** (remember: only gives at-least/at-most-once, not exactly-once)

## Idempotency decision tree
1. Is the full dataset available each run?
   - Yes, and store supports metadata ops (TRUNCATE/DROP) → **Fast Metadata Cleaner**
   - Yes, but no metadata-op support → **Data Overwrite**
2. Only an incremental/delta subset available?
   - Backfill consistency not required → **Merger**
   - Backfill must restore to a consistent version → **Stateful Merger** (needs a versioned store)
3. Key-based store, want writes to self-dedupe → **Keyed Idempotency** (build keys from append/ingestion time, never event time)
4. Must prevent partial data visibility on failure → **Transactional Writer** (remember: idempotency scope = transaction only)
5. Dataset must be immutable/write-once (legal/compliance) → **Proxy**

## Joining data: static or dynamic?
- Enrichment side at rest / slowly changing → **Static Joiner** (+ SCD type 2/4 if time-sensitive)
- Both sides streaming → **Dynamic Joiner** (time-bounded buffer + GC watermark)
- Tell: join returns mostly empty results on a stream-stream join → missing time boundary/watermark.

## Aggregation: distributed or local?
- Partitioning/grouping key is NOT guaranteed stable/co-located → **Distributed Aggregator** (shuffle)
- Partitioning is static AND records for a key are already co-located → **Local Aggregator** (no shuffle)
- Smell: "it worked until we rescaled/repartitioned" → you used Local Aggregator somewhere partitioning wasn't actually immutable.

## Sessionization
- Batch, sessions span multiple time partitions → **Incremental Sessionizer** (3 storage zones: input/completed/pending)
- Near-real-time required → **Stateful Sessionizer** (state store, prefer event-time expiration over processing-time)
- Tell: downstream treats a partial session as final → missing `is_completed`/`is_final` flag.

## Ordering under write failures
- Bulk sink with partial-commit semantics (Kinesis, DynamoDB, Elasticsearch bulk) + ordering matters → **Bin Pack Orderer**
- Low volume, strict FIFO, latency-tolerant → **FIFO Orderer**
- Remember: FIFO ≠ exactly-once — pair with an idempotency pattern if duplicates matter.

## Data flow: sequencing, fan-in, fan-out
- Single pipeline, want independent restart boundaries → **Local Sequencer**
- Cross-team pipelines → **Isolated Sequencer**: loose coupling wanted → data-based (Readiness Marker); tight coupling + auto-backfill propagation wanted → task-based (direct trigger)
- Merge requires ALL branches to succeed → **Aligned Fan-In**
- Partial results acceptable (with a completeness flag) → **Unaligned Fan-In**
- One input feeds multiple independent outputs → **Parallel Split** (cache/persist the shared intermediate!)
- Exactly one of several mutually-exclusive paths should run → **Exclusive Choice**
- Pipeline is stateful/incremental → **Single Runner** (concurrency=1); independent instances → **Concurrent Runner**

## Security: removal, access, protection
- New/controllable pipeline, cheap PII removal wanted → **Vertical Partitioner** (security)
- Legacy/uncontrolled storage → **In-Place Overwriter** (always stage-then-promote)
- Need column/row-level restriction on a table → **Fine-Grained Accessor for Tables**
- Need least-privilege on a cloud resource (bucket/stream) → **Fine-Grained Accessor for Resources**
- Must survive access-control failure → add **Encryptor** (defense in depth, not a substitute)
- Sharing externally, re-identification must be impossible → **Anonymizer**
- Need more analytical utility, acceptable re-identification risk when combined → **Pseudo-Anonymizer** (never claim it's as safe as anonymization)
- Credentials: get them out of code → **Secrets Pointer**; eliminate them entirely → **Secretless Connector** (prefer this when platform supports IAM/cert auth)

## Storage: partition, bucket, sort?
- Low-cardinality attribute, range/time queries → **Horizontal Partitioner**
- High-cardinality attribute, heavy filter/join use → **Bucket** (never horizontally partition a high-cardinality key)
- Queries filter/sort on known columns → **Sorter** (plain sort for single leading key; Z-order for multi-column access)
- Repeated expensive query → **Dataset Materializer** (prefer incremental refresh)
- Listing cost dominates → **Manifest**
- Consistency/cheap updates (OLTP) → **Normalizer**; query speed (OLAP) → **Denormalizer** (can do both: normalize then materialize a denormalized view)

## Quality: enforce or observe?
- Want to block bad data before publish → **AWAP** (put each rule at its most exhaustive audit point) or **Constraints Enforcer** (if store supports it natively)
- Want to prevent breaking schema changes → **Schema Compatibility Enforcer** (pick backward/forward/full + transitive/nontransitive)
- Need to actually make a breaking change safely → **Schema Migrator** (add-then-deprecate, never in-place rename)
- Monitoring that shouldn't block/slow the pipeline → **Offline Observer**
- Detection latency unacceptable → **Online Observer** (isolate failures from the main flow!)

## Observability: which detector?
- Job could silently stop producing data → **Flow Interruption Detector**
- Dataset volume could silently drop/spike → **Skew Detector** (compare against last *known-good* run, not just yesterday)
- Consumer could silently fall behind → **Lag Detector** (use percentile + MAX, never a plain average)
- Deadline-driven SLA independent of lag → **SLA Misses Detector** (track processing-time and event-time SLA separately)
- "Who broke this?" → **Dataset Tracker** (dataset-level) then **Fine-Grained Tracker** (column/row-level) if needed

## Trade-off matrix: speed vs. safety
| Need | Faster/cheaper option | Safer/stronger option |
|---|---|---|
| Idempotent reset | Fast Metadata Cleaner | Data Overwrite |
| Joins | Local Aggregator / Denormalizer | Distributed Aggregator / Normalizer |
| Data removal | Vertical Partitioner | In-Place Overwriter |
| PII sharing | Pseudo-Anonymizer | Anonymizer |
| Credentials | Secrets Pointer | Secretless Connector |
| Quality monitoring | Offline Observer | Online Observer |
| Fan-in | Unaligned Fan-In | Aligned Fan-In |

## Thresholds & defaults worth remembering
- Watermark = MAX(event time) − allowed lateness; must be monotonically increasing (never use MIN per-partition).
- Use P90/P95 (not average) for lag/latency — the "average trap" hides skewed partitions.
- Skew tolerance is business-specific; a common starting point is ±50%, but validate against seasonality.
- BigQuery: 4,000 partitions/table; AWS Redshift: 200,000 tables — plan Fast Metadata Cleaner granularity around these.
- Default to backward-compatible schema evolution unless you have a specific reason for forward or full.
