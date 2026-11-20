# Chapter 5: Data Generation in Source Systems

## Core Idea
Before ingesting anything, a data engineer must understand how source systems actually generate data — their operational patterns (OLTP vs. OLAP, CDC, logs, messages/streams, APIs) and their quirks — because source systems and ingestion are typically the biggest bottleneck and risk point of the entire data engineering lifecycle, and they are usually outside the data engineer's direct control.

## Frameworks Introduced
- **OLTP vs. OLAP**: OLTP (application/transactional databases) = low-latency, high-concurrency individual record reads/writes; OLAP = large analytical scans, inefficient at individual lookups.
  - When to use: deciding whether a system is a safe source for direct analytics queries.
  - How: never run heavy analytics directly against a production OLTP system; extract (via CDC/snapshot) into an OLAP-suited store instead.
- **ACID**: Atomicity, Consistency, Isolation, Durability — guarantees for transactional correctness.
  - When to use: evaluating whether a database is appropriate for an application needing strict correctness vs. one that can tolerate eventual consistency for scale.
  - How: check if the source relaxes ACID (e.g., eventual consistency in distributed NoSQL) — this affects what "correct" ingested data even means.
- **CDC (Change Data Capture)**: extracting every insert/update/delete event from a database, typically via database logs (binary write-ahead logs) or triggered messages.
  - When to use: when you need near-real-time replication or a full event history rather than periodic snapshots.
  - How: tail database logs (low source-system impact) or use vendor-provided CDC streams; prefer log-based CDC over timestamp-polling CDC when possible.
- **CRUD vs. Insert-Only patterns**: CRUD updates/deletes rows in place (loses history); insert-only appends new versioned rows with timestamps (preserves history at the cost of table growth and `MAX(timestamp)` lookup overhead).
  - When to use: insert-only when historical state matters (e.g., audit, address history); CRUD when only current state matters and table size must stay bounded.
- **Messages vs. Streams**: a message is a discrete, transient signal delivered once and removed from a queue; a stream is an ordered, append-only, retained log of events that can be replayed/aggregated over time.
  - When to use: messages for routing/triggering actions between decoupled services; streams when you need historical replay, windowed aggregation, or multi-consumer analytics.
- **Four Types of Time**: event time (when it happened), ingestion time (when it entered your systems), process time (when transformed), and processing duration.
  - When to use: any streaming or near-real-time pipeline design.
  - How: log timestamps at every stage so late-arriving or out-of-order data can be reasoned about and reconciled.
- **Data Contract**: written agreement between source system owner and the ingesting team specifying what data, extraction method, cadence, and points of contact.
  - When to use: any production dependency on an upstream team's system.
  - How: store contracts in a discoverable, ideally machine-queryable location (e.g., a repo); pair with an SLA/SLO on uptime and data quality.

## Key Concepts
- **Analog vs. digital data creation**: analog = real-world, often transient (speech, writing); digital = native digital creation or analog-to-digital conversion.
- **Database logs (write-ahead logs)**: binary, database-native files recording every operation before acknowledgment — the backbone of crash recovery and the preferred mechanism for CDC.
- **NoSQL database families**: key-value (fast lookups by key, e.g., caching), document (nested JSON, no joins, flexible/risky schema), wide-column (petabyte-scale, single row-key index, very low latency, e.g., Cassandra/Bigtable), graph (nodes/edges, optimized for connectivity traversal queries, e.g., Neo4j), search (text/log search, e.g., Elasticsearch), time-series (timestamp-ordered measurement/event data, write-heavy, e.g., InfluxDB).
- **REST vs. GraphQL vs. gRPC vs. Webhooks**: REST = stateless HTTP verb-based, dominant but underspecified; GraphQL = flexible multi-model single-request queries (JSON-shaped); gRPC = efficient bidirectional binary RPC over HTTP/2 (Protocol Buffers), common at Google-scale; Webhooks = "reverse APIs" — source system pushes events to a consumer-hosted endpoint on trigger.
- **Message queue properties**: message ordering (often fuzzy/best-effort FIFO, not guaranteed), delivery frequency (exactly-once vs. at-least-once), and the need for idempotent consumers to handle duplicate delivery gracefully.
- **Event-streaming platform mechanics**: topics (named collections of related events, multi-producer/multi-consumer), stream partitions (subdivisions by partition key for parallelism — watch for hotspotting from skewed keys), fault tolerance via distributed replicated storage.
- **Data sharing / data marketplace**: multitenant systems (cloud object storage, cloud data warehouses) with fine-grained permissions enabling row/column/sensitive-data filtering for sharing data across org units or externally — a key enabler of data mesh decentralization.
- **Systems stakeholders vs. data stakeholders**: systems stakeholders build/maintain source systems (software engineers); data stakeholders own/govern access to the data (IT, governance team) — often different people requiring separate relationship management.

## Mental Models
- **Source systems are "someone else's problem" at your peril**: treating upstream systems as out-of-scope causes production failures when schemas change or systems go down without notice — proactive relationship-building with source teams is a direct lever on data quality.
- **Bidirectional feedback loop with source teams**: don't just consume passively — communicate your requirements upstream (schema needs, SLAs) just as you listen for their changes.
- **Streams and messages blur lifecycle stages**: unlike an RDBMS tightly bound to one application, an event-streaming platform can simultaneously be a source system, an ingestion mechanism, and a transformation engine — the same technology serves multiple lifecycle stages at once.
- **Idempotency as a safety net**: design consumers so reprocessing the same message twice produces the same result, hedging against the practical impossibility of guaranteed exactly-once delivery (the Two Generals Problem).

## Anti-patterns
- **Running heavy analytics directly on a production OLTP database**: causes resource contention and can degrade or crash the application backend.
- **Assuming strict message ordering/exactly-once delivery without verifying it**: leads to data corruption or double-processing bugs when the underlying queue only offers best-effort guarantees.
- **Ignoring schema evolution on the source side**: unannounced schema changes break downstream pipelines; mitigate via data contracts and direct communication channels.
- **Choosing a hotspotting partition key** (e.g., partitioning IoT data by US state) — skewed key distribution overwhelms a subset of stream partitions.
- **Poorly managed document-store schema evolution**: unconstrained JSON flexibility becomes "an absolute nightmare to manage and query" without discipline, causing inconsistent, bloated data.
- **No data contract / no SLA with upstream teams**: leaves the data engineer blindsided by outages, schema changes, or quality regressions with no agreed recourse.

## Code Examples
```json
{
  "users":[
     {
     "id":1234,
     "name":{
       "first":"Joe",
       "last":"Reis"
     },
     "favorite_bands":[
       "AC/DC",
       "Slayer",
       "WuTang Clan",
       "Action Bronson"
     ]
     },
     {
     "id":1235,
     "name":{
       "first":"Matt",
       "last":"Housley"
     },
     "favorite_bands":[
       "Dave Matthews Band",
       "Creed",
       "Nickelback"
     ]
     }
  ]
}
```
- **What it demonstrates**: a document-store (e.g., MongoDB) collection structure — nested, schemaless JSON documents retrieved by key, with no joins across collections (illustrates why document stores trade relational rigor for flexible, denormalized storage).

```json
{
  "Key":"Order # 12345",
  "Value":"SKU 123, purchase price of $100",
  "Timestamp":"2023-01-02 06:01:00"
}
```
- **What it demonstrates**: the canonical shape of an event in an event-streaming platform — key, value, and timestamp — the minimal unit that gets published to a topic and partitioned downstream.

## Reference Tables

| Record ID | Value | Timestamp |
|---|---|---|
| 1 | 40 | 2021-09-19T00:10:23+00:00 |
| 1 | 51 | 2021-09-30T00:12:00+00:00 |

*(Insert-only pattern: the same logical record accumulates multiple versioned rows instead of being updated in place.)*

| NoSQL type | Best for | Key limitation |
|---|---|---|
| Key-value | Ultra-fast lookup by key (caching, session state) | No complex queries |
| Document | Flexible, nested, evolving schemas | No joins; eventual consistency common |
| Wide-column | Massive write throughput, sub-10ms latency | Single index (row key); no complex queries |
| Graph | Connectivity/traversal analysis | Needs specialized query languages (Cypher, SPARQL) |
| Search | Text search, log analysis | Narrow use case |
| Time-series | High-velocity timestamped measurement/event data | Poor fit for general BI; joins rare |

| API style | Strength | Weakness |
|---|---|---|
| REST | Ubiquitous, simple HTTP verbs | Underspecified; stateless-only |
| GraphQL | Flexible multi-model single request | Added complexity, less standardized caching |
| gRPC | Efficient bidirectional, strongly typed (Protobuf) | Steeper learning curve, less universal tooling |
| Webhook | Push-based, event-triggered ("reverse API") | Consumer must host and secure an endpoint |

## Worked Example
A data engineering team ingests from (1) a Postgres OLTP order database using log-based CDC to capture every insert/update/delete without impacting production performance, (2) a CRM SaaS platform's REST API via a managed connector (reducing undifferentiated heavy lifting), and (3) an IoT fleet publishing temperature events to a Kafka-like event-streaming platform, partitioned by device ID to avoid hotspotting. The team negotiates a data contract with the CRM vendor (extraction method, cadence, point of contact) and an SLA with the internal order-database team (99% uptime, advance notice of schema changes). Event timestamps (event/ingestion/process time) are logged at every stage so late-arriving IoT readings can be reconciled correctly in downstream aggregation. Consumers of the Kafka topic are built idempotently so duplicate message delivery during a partition rebalance doesn't double-count order events.

## Key Takeaways
1. Understand how a source system is architected and generates data before designing anything downstream from it.
2. Never run heavy analytics directly against production OLTP systems — extract via CDC or snapshot into OLAP-suited storage.
3. CDC (especially log-based) is generally lower-impact and more complete than timestamp-polling extraction.
4. Messages (transient, routed) and streams (durable, replayable, ordered) solve different problems — pick based on whether you need history/replay.
5. Track event, ingestion, and process time explicitly to handle late-arriving and out-of-order data correctly.
6. NoSQL isn't one thing — key-value, document, wide-column, graph, search, and time-series each fit different access patterns; choose deliberately.
7. Formalize relationships with upstream teams via data contracts and SLAs/SLOs rather than relying on informal hope.
8. Design consumers to be idempotent since exactly-once delivery is practically unguaranteeable in distributed systems.

## Connects To
- **Ch 2**: Expands the "Generation: Source Systems" stage introduced there.
- **Ch 3**: Event-driven architecture and data mesh/data sharing concepts reused directly.
- **Ch 6**: Storage systems that receive data extracted from these source systems.
- **Ch 7**: Ingestion mechanics (batch/streaming, push/pull, CDC) applied concretely to these source system types.
- **Ch 8**: Data modeling implications of normalized (RDBMS) vs. denormalized (document/wide-column) source schemas.
- **External concept**: Martin Kleppmann's Designing Data-Intensive Applications, the Two Generals Problem, Protocol Buffers/gRPC, James Denmore's data contract definition.
