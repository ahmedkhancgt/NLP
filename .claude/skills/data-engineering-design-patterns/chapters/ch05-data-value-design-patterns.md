# Chapter 5: Data Value Design Patterns

## Core Idea
Raw ingested data is not yet a real asset — value is created either by adding information (enrichment, decoration) or by reducing/summarizing information (aggregation, sessionization), and by preserving meaningful order (ordering patterns), each with distinct static-vs-dynamic and batch-vs-streaming implementation trade-offs.

## Frameworks Introduced
- **Static Joiner**: combine a dataset with a slowly changing (at-rest) reference dataset via JOIN.
  - When to use: enrichment dataset is static or slowly changing (e.g., user reference data).
  - How: join on a key attribute; for time-sensitive versions use Slowly Changing Dimensions (SCD) type 2 (validity date range in one table) or type 4 (separate current + history tables); for API-backed enrichment, materialize into a table for idempotency and bulk-fetch to optimize network cost.
- **Dynamic Joiner**: combine two in-motion (streaming) datasets.
  - When to use: both sides of the join are continuously updating streams with different latencies.
  - How: define keys and join method plus explicit time boundaries/buffers so the faster stream waits for the slower one; use a GC watermark to evict buffered records that are too old to ever match (space-vs-exactness trade-off).
- **Wrapper**: envelope a record with raw + computed fields, fully exposed to consumers.
  - When to use: need to preserve original values while adding derived/business attributes, visibly, for debugging or downstream clarity.
  - How: nest computed attributes alongside raw ones (flat-computed/nested-raw, flat-raw/nested-computed, all-flat, or two joined tables); works for semi-structured formats (Avro/Protobuf/JSON) and structured tables alike.
- **Metadata Decorator**: attach technical/operational context without exposing it to end users.
  - When to use: need processing context (job version, batch number) for debugging, but it's irrelevant/confusing to business consumers.
  - How: use native metadata support where available (Kafka headers, object-store tags); otherwise simulate with a hidden column/table and restrict access via views or column-level permissions (Fine-Grained Accessor, Ch 7).
- **Distributed Aggregator**: aggregate across a cluster via shuffle (MapReduce-style).
  - When to use: dataset too large for a single machine; need grouped/reduced statistics (counts, averages) across partitions.
  - How: group-by + reduce; the framework shuffles related records across the network; mitigate data skew with salting (add a random "salt" column, aggregate twice) and use a shuffle service to free compute faster.
- **Local Aggregator**: aggregate without network shuffle by relying on existing, stable partitioning.
  - When to use: data volume is static and already co-located by grouping key (fixed partition count/assignment) or fits on one machine.
  - How: use per-partition operations (Spark `mapPartitions`/bucketed tables with matching bucket count; Kafka Streams `groupByKey`); works only as long as partitioning and grouping-key logic stay immutable — any rescale requires a costly full reorganization.
- **Incremental Sessionizer**: build sessions from batch, partitioned data spanning multiple partitions.
  - When to use: event records for one session can span several time partitions in a batch pipeline.
  - How: maintain three storage spaces — input, completed-sessions (public), pending-sessions (private/internal); each run merges new input with pending sessions and classifies each entity as new/continuing/about-to-expire via initialization/accumulation/finalization logic; flag partial sessions (`is_completed: false`) since backfilling has a forward dependency (regenerating one partition requires regenerating all subsequent ones).
- **Stateful Sessionizer**: build sessions in near real time from a streaming layer with a state store.
  - When to use: data freshness requirements make batch-based (Incremental) sessionization too slow.
  - How: use a state store (session windows with a fixed gap duration, or arbitrary stateful processing with custom expiration logic); prefer event-time/watermark-based expiration over processing-time (processing-time expiration is unreliable under latency spikes); understand checkpointing gives only at-least-once state persistence.
- **Bin Pack Orderer**: guarantee ordered bulk delivery despite partial-commit semantics.
  - When to use: the sink supports bulk writes with partial-commit risk (Kinesis `PutRecords`, DynamoDB `BatchWriteItem`, Elasticsearch bulk) and ordering within an entity matters.
  - How: sort records by grouping key + event time; pack into bins with exactly one record per key per bin; emit bins sequentially so any retry stays local to its own key and never crosses bin boundaries.
- **FIFO Orderer**: guarantee first-in-first-out delivery via per-record or small-batch acknowledged sends.
  - When to use: low-volume/latency-tolerant requirement for strict delivery order without bulk-batching complexity.
  - How: send-and-acknowledge each record before the next (single-record API or bulk API with concurrency=1, or Kafka's idempotent producer allowing up to 5 in-flight requests while preserving order); note FIFO ≠ exactly-once — pair with an idempotency pattern (Ch 4) for that guarantee.

## Key Concepts
- **Slowly Changing Dimensions (SCD) type 2 / type 4**: data modeling strategies tracking entity history — type 2 uses validity date ranges in one table, type 4 splits current vs. historical data into two tables.
- **Shuffle**: the network data-exchange step in distributed aggregation that co-locates records sharing a grouping key.
- **Data skew**: an unbalanced dataset where one key dominates, making its shuffle/processing the bottleneck; mitigated via salting.
- **GC watermark**: the mechanism that evicts buffered join/session state that is too old to ever match or complete.
- **Session window / arbitrary stateful processing**: two state-store-based abstractions for defining when a session starts/ends, the former using a fixed gap duration, the latter fully custom.
- **Partial commit**: a bulk write outcome where only some records succeed, risking broken ordering on retry (seen in Kinesis `PutRecords`, DynamoDB `BatchWriteItem`, Elasticsearch bulk).
- **In-flight requests**: overlapping, unacknowledged bulk requests that improve throughput but can break ordering if an earlier request fails while a later one succeeds.

## Mental Models
- Choose Static vs. Dynamic Joiner by asking "is the enrichment side at rest or in motion?" — at rest → Static Joiner (+ SCD for time-sensitivity); in motion → Dynamic Joiner (+ buffering/GC watermark).
- Use Wrapper when consumers should see the distinction between raw and derived data; use Metadata Decorator when they shouldn't see it at all.
- Think of Distributed vs. Local Aggregator as "pay for shuffle now" vs. "pay for rigidity later" — Local Aggregator trades scalability/flexibility for the one-time cost savings of skipping shuffle.
- Incremental Sessionizer and Stateful Sessionizer are the same conceptual workflow (init → accumulate → finalize, with a pending-sessions store) applied to batch partitions vs. streaming state stores respectively.
- Bin Pack Orderer is "shuffle-then-isolate-by-key" so bulk-write retries can never cross entities; FIFO Orderer is "slow down to one (or a few) in flight" so retries can never overtake each other.

## Anti-patterns
- **Joining streaming datasets without time boundaries/buffers**: produces mostly empty joins because the two streams rarely align perfectly in time.
- **Putting business-sensitive data (e.g., shipment address, invoice amount) into the Metadata Decorator layer**: consumers will never think to query hidden metadata for business data.
- **Ignoring data skew in a Distributed Aggregator**: one oversized key dominates shuffle cost and can stall the whole job — use salting or native skew handling (e.g., Spark Adaptive Query Execution).
- **Using Local Aggregator when partitioning/grouping keys can change**: breaks correctness silently by splitting a key's data across multiple groups after any rescale.
- **Treating a partial/ongoing session as final without an `is_completed` flag**: downstream consumers (e.g., fraud detection) may act on incomplete state.
- **Relying on processing time for session/state expiration**: unexpected latency causes premature expiration — prefer event time/watermark-based expiration.
- **Assuming FIFO delivery implies exactly-once**: an acknowledgment failure after a successful send can cause reprocessing — combine with Ch 4 idempotency patterns.

## Code Examples
```sql
-- Static Joiner: SCD type 2 time-sensitive join using validity dates
SELECT v.visit_id, v.event_time, v.page, u.id, u.login, u.email
FROM dedp.visits v JOIN dedp.users u ON u.id = v.user_id
  AND NOW() BETWEEN start_date AND end_date;
```
- **What it demonstrates**: enriching events with the historically-correct version of a slowly changing reference row.

```python
# Dynamic Joiner: time-bounded stream-to-stream join with watermarks on both sides
visits_from_kafka: DataFrame = (visits_data_stream.withWatermark('event_time', '10 minutes'))
ads_from_kafka: DataFrame = (ads_data_stream.withWatermark('display_time', '10 minutes'))
visits_with_ads = visits_from_kafka.join(ads_from_kafka, F.expr('''
    page = visit_page AND
    display_time BETWEEN event_time AND event_time + INTERVAL 2 minutes
  '''), 'left_outer')
```
- **What it demonstrates**: bounding a stream-stream join with both a business time window and watermark-based lateness tolerance.

```python
# Distributed Aggregator: salting to mitigate data skew
dataset.withColumn('salt', (rand()*3).cast("int"))
 .groupBy('group_key', 'salt').agg(...)
 .groupBy('group_key').agg(...)
```
- **What it demonstrates**: splitting a hot key's aggregation across salted sub-groups, then re-aggregating.

```python
# Bin Pack Orderer: packing sorted rows into per-key bins before bulk delivery
for visit in visits_rows:
    if visit.visit_id != last_visit_id:
      last_visit_id = visit.visit_id
      groups_index = 0
    if len(delivery_groups) <= groups_index:
        delivery_groups.append([])
    delivery_groups[groups_index].append(visit)
    groups_index += 1
```
- **What it demonstrates**: ensuring each delivery bin holds at most one record per key, so retries stay isolated and ordering survives partial commits.

## Reference Tables
| Pattern pair | Data motion | Key risk mitigated |
|---|---|---|
| Static Joiner | At rest ↔ at rest/motion | Idempotency via SCD + materialization |
| Dynamic Joiner | Motion ↔ motion | Empty joins via time-bounded buffer + GC watermark |
| Distributed Aggregator | Any, cluster-scale | Scalability via shuffle |
| Local Aggregator | Any, pre-partitioned | Shuffle cost, at cost of scaling flexibility |
| Incremental Sessionizer | Batch/partitioned | Session continuity via pending-sessions store |
| Stateful Sessionizer | Streaming | Latency via state store + session windows |
| Bin Pack Orderer | Bulk/partial-commit sinks | Ordering under partial commit |
| FIFO Orderer | Low-volume strict order | Ordering via single/limited in-flight requests |

## Worked Example
Blog analytics platform: visit events (streaming) are enriched with a slowly-changing `users` reference table using SCD type 2 validity dates (Static Joiner), and separately joined in near real time with an `ads` stream using a 2-minute business window plus 10-minute watermarks on both sides (Dynamic Joiner). Each visit record is wrapped with a `processing_context` struct containing `job_version`/`batch_version` for debugging (Wrapper) while an unrelated job instead hides the same kind of technical context as Kafka record headers so end users never see it (Metadata Decorator). Downstream, a Spark job builds an OLAP cube by grouping visits across geography and device (Distributed Aggregator with salting for skewed keys), while a separate Kafka Streams job locally aggregates session page-views per partition using `groupByKey` because the input topic is already partitioned by `visit_id` (Local Aggregator). Sessions themselves are built two ways: an hourly batch job using three storage zones — input, completed, pending — for users whose visits span multiple partitions (Incremental Sessionizer), and a low-latency Spark Structured Streaming job using `applyInPandasWithState` with event-time-based expiration (Stateful Sessionizer). Finally, visit events are forwarded to Amazon Kinesis for a partner API: records are sorted by `visit_id`/`event_time` and packed into per-key bins before bulk delivery (Bin Pack Orderer) to survive Kinesis's partial-commit semantics, while a separate low-volume alerting stream uses Kafka's idempotent producer with `max.in.flight.requests.per.connection=5` to guarantee simple FIFO delivery (FIFO Orderer).

## Key Takeaways
1. Match the joiner pattern to data motion: Static Joiner for at-rest enrichment (with SCD for time-sensitivity), Dynamic Joiner for stream-to-stream with explicit buffering and GC watermarks.
2. Use Wrapper when consumers should see raw-vs-computed distinctions; use Metadata Decorator when they should not see the technical context at all.
3. Prefer Local Aggregator only when partitioning and grouping keys are genuinely immutable — otherwise accept Distributed Aggregator's shuffle cost for correctness and flexibility.
4. Sessionization always needs a pending/in-flight state zone (pending-sessions table or state store) and explicit completion flags to avoid misleading partial results.
5. Bulk-write ordering guarantees require pattern-level isolation (Bin Pack Orderer's per-key bins) whenever the sink has partial-commit semantics.
6. FIFO ordering and exactly-once delivery are separate guarantees — combine FIFO Orderer with an idempotency pattern (Ch 4) if duplicates must also be avoided.

## Connects To
- **Ch 2 (Data Ingestion)**: Static Joiner's batch-side readiness mirrors the Readiness Marker pattern; CDC feeds the Dynamic Joiner's streaming enrichment side.
- **Ch 3 (Error Management)**: Dynamic Joiner and Stateful Sessionizer reuse watermark/late-data concepts from the Late Data Detector; state stores reuse the Windowed Deduplicator's state-store mechanics.
- **Ch 4 (Idempotency)**: FIFO Orderer and streaming session generation both require idempotency patterns to achieve true exactly-once outcomes.
- **Ch 7 (Data Security)**: Metadata Decorator's access restriction connects to the Fine-Grained Accessor pattern.
- **Ch 8 (Data Storage)**: Local Aggregator's bucketing connects to the Bucket pattern.
- **Apache Spark (Structured Streaming), Apache Flink, Kafka Streams, Apache Kafka, Amazon Kinesis Data Streams, GCP Pub/Sub, AWS Redshift, BigQuery**: primary technologies implementing these patterns.
