# Chapter 7: Ingestion

## Core Idea
Ingestion — moving data from source systems into storage — is where data engineers begin actively designing pipelines, and it is frequently the biggest bottleneck and risk point of the lifecycle; the central design questions span bounded vs. unbounded data, frequency (batch/micro-batch/streaming), sync vs. async coupling, and a wide menu of concrete ingestion mechanisms (direct DB connection, CDC, APIs, message/event streams, files, webhooks, web scraping, physical transfer appliances, data sharing).

## Frameworks Introduced
- **Data pipeline definition**: "the combination of architecture, systems, and processes that move data through the stages of the data engineering lifecycle" — deliberately broad/flexible.
  - When to use: as the umbrella concept encompassing ETL, ELT, reverse ETL, and data sharing.
  - How: don't dogmatically commit to one movement pattern; assemble whichever combination serves the business outcome.
- **"All data is unbounded until it's bounded"**: data exists continuously in reality; batch processing artificially imposes boundaries (time or size cuts) on an inherently unbounded stream.
  - When to use: deciding how to think about any ingestion design.
  - How: treat streaming ingestion as preserving the natural unboundedness of data; treat batch as a deliberate, bounded convenience with known latency trade-offs.
- **Key ingestion design axes**: bounded vs. unbounded, frequency (batch/micro-batch/real-time), synchronous vs. asynchronous, serialization/deserialization compatibility, throughput/scalability, reliability/durability, payload characteristics, push/pull/poll pattern.
  - When to use: as a checklist before building any ingestion pipeline.
  - How: walk each axis explicitly and document the chosen trade-off and its rationale.
- **Snapshot vs. Differential (incremental) extraction**: full snapshot = entire current state each read (simple, more network/storage cost); differential = only changes since last read (efficient, more complex).
  - When to use: full snapshot for simplicity/small data; differential/incremental for large or frequently-changing sources.
- **Batch vs. Continuous CDC**: batch CDC queries an `updated_at` field periodically (misses intermediate changes within a pull window); continuous CDC (usually log-based) treats every write as an event, capturing full history for near-real-time replication or streaming analytics.
  - When to use: continuous/log-based CDC when full change history or near-real-time replication matters; batch CDC when periodic snapshots of "latest state" suffice and simplicity is preferred.
- **Dead-letter queue**: a separate store for events that fail ingestion (bad schema, oversized, expired TTL) so they don't block the main stream and can be diagnosed/reprocessed.
  - When to use: any streaming ingestion pipeline with heterogeneous or evolving event payloads.

## Key Concepts
- **Data ingestion vs. data integration vs. internal transformation**: ingestion = moving data A→B; integration = combining disparate sources into something new (Ch 8); internal copy/cache within one system is part of transformation, not ingestion.
- **Synchronous vs. asynchronous ingestion**: synchronous = tightly coupled sequential stages (a failure anywhere forces a full restart — the chapter's "whack-a-mole" 24-hour-pipeline cautionary tale); asynchronous = event-level independence (e.g., Kinesis buffering between Beam processing stages), allowing partial progress and natural load-smoothing ("shock absorber").
- **Payload characteristics**: kind (type + format, e.g., tabular/CSV vs. image/JPG), shape (dimensions — rows×columns, JSON nesting, pixel dimensions, audio channels/sample rate), size (bytes, possibly chunked for transfer), schema/data types, and metadata — all directly affect downstream handling.
- **Schema registry**: a metadata repository tracking schema versions for streaming messages, enabling consistent serialization/deserialization between evolving producers and consumers.
- **Time to Live (TTL)**: max retention time for an unacknowledged streaming message before it's dropped — balances backpressure (too short) against backlog buildup (too long); platform limits vary (Pub/Sub ≤7 days, Kinesis ≤365 days, Kafka effectively indefinite with tiered object-storage offload).
- **JDBC vs. ODBC**: ODBC = OS/architecture-native binary drivers (vendor must maintain per-platform); JDBC = portable across any JVM language/framework via a single driver — both show their age for ingestion since they transmit data row-wise, forcing re-encoding of native columnar/nested data.
- **Synchronous (native) vs. asynchronous CDC replication**: synchronous replication keeps a read replica fully in sync (good for offloading reads, zero-loss failover) but requires matching database types; asynchronous CDC replication is loosely coupled, can fan out to multiple heterogeneous targets, tolerates slight delay.
- **Managed data connectors**: SaaS/OSS platforms (Fivetran, Airbyte, Matillion) providing turnkey, pre-built connectors — treated as a prime example of "undifferentiated heavy lifting" that should be outsourced rather than custom-built.
- **Transfer appliances**: physical hard-drive shipments (AWS Snowball/Snowmobile) for one-time migrations of massive data (~100TB+), where moving data "by truck" beats network transfer economically.
- **Touchless production**: engineering practice of developing/testing against simulated or cleansed data, with production code deployment automated — minimizing direct human handling of sensitive data; a "broken-glass" process (two-person approval, time-boxed, scoped) covers unavoidable exceptions.

## Mental Models
- **"The bulk of software engineering is just plumbing"** (Karl Hughes, quoted): ingestion work is often unglamorous connective tissue, but it's indispensable — analytics and ML cannot happen without it.
- **Batch frequency as a downstream bottleneck**: once data enters a batch process, that batch's cadence caps the speed of everything downstream of it, regardless of how fast later stages could otherwise run.
- **Streaming and batch coexist, not compete**: even "real-time" pipelines usually feed batch-oriented consumers (e.g., ML model training) downstream — pure end-to-end streaming is rare in practice.
- **Reliability/durability as a cost trade-off, not an absolute**: there is no universally "correct" level of redundancy; evaluate the cost of data loss against the direct and indirect costs of preventing it, and accept that some catastrophic scenarios (internet-wide outages) are simply out of scope.
- **"Data cannot leak if it is never collected"**: the most robust privacy control is not collecting or forwarding sensitive fields you don't actually need, rather than relying solely on encryption/tokenization after the fact.

## Anti-patterns
- **Tightly coupled synchronous batch pipelines**: a single failing step forces a full pipeline restart, as in the chapter's real example of a 24+ hour transformation process that took over a week to diagnose after cascading failures.
- **Many small-batch insert/update operations against columnar/analytical stores**: creates excessive small files and forces costly per-column-file scans on update — know your store's bulk-vs-row-level performance characteristics.
- **Assuming exact or strict message ordering/delivery without verifying platform guarantees**: distributed streaming systems commonly deliver at-least-once and out-of-order.
- **Relying on CSV as a production ingestion format without rigorous metadata**: ambiguous delimiters/escaping, no native schema or nested-structure support, autodetection unsuitable for production — prefer Parquet/Avro/ORC/Arrow/JSON where possible.
- **Building custom API connectors for well-supported sources**: reinventing a wheel that managed connector platforms already provide, wasting engineering time on undifferentiated heavy lifting.
- **Reflexive encryption/tokenization as a "privacy magic bullet"**: applying encryption to a field while leaving access controls loose doesn't solve the underlying access-control problem; naive hashing of identifiers (e.g., unsalted email hashes) can be trivially reversed.
- **Letting schema changes break downstream pipelines silently**: either overly slow command-and-control schema review (6-month lead times observed in practice) or unchecked auto-propagation that silently breaks consumers — both extremes are harmful; aim for communicated, versioned, and ideally branch-like (Git-inspired) schema evolution.
- **Excessive/unthrottled web scraping**: risks DoS-like behavior, IP/account bans, and legal exposure — always check for a legitimate data-sharing alternative first and throttle responsibly.

## Code Examples
None — Chapter 7 is pattern/architecture-focused; no code snippets in source (architectural diagrams only, e.g., AWS webhook pipeline: Lambda → Kinesis → Flink → S3, described in prose).

## Reference Tables

| Ingestion frequency | Typical use case | Trade-off |
|---|---|---|
| Batch (time-interval) | Daily/overnight BI reporting | Simple, but caps downstream latency |
| Batch (size-based) | Streaming data rolled into object storage blocks | Needs a chunking trigger (size/count) |
| Micro-batch | Near-real-time dashboards | Balances latency and overhead |
| Streaming/real-time | IoT, event-driven apps, fraud detection | Higher complexity/cost; rarely "purely" real-time end-to-end |

| CDC approach | Captures full history? | Source impact | Best for |
|---|---|---|---|
| Batch (timestamp-based) | No (only latest state per changed row) | Periodic query load | Simple differential updates, low source load tolerance |
| Continuous, log-based | Yes | Low (tails existing write-ahead log) | Near-real-time replication/analytics |
| Managed CDC (cloud-native triggers) | Yes | Minimal engineering effort | Cloud-hosted databases with native support |

| Ingestion mechanism | Strength | Caveat |
|---|---|---|
| Direct DB connection (JDBC/ODBC) | Universal, well-understood | Row-based; poor fit for nested/columnar data |
| CDC | Low-impact, near-real-time | Resource overhead; test before enabling on prod |
| API (managed connector) | Fast to stand up, maintained by vendor | Custom APIs still require real engineering |
| Message queue / event stream | Real-time, decoupled | Needs schema registry, DLQ, TTL tuning |
| File export (object storage) | Secure, flexible, scalable | Format matters (avoid raw CSV in production) |
| Webhook | Push-based, event-triggered | Consumer must host/secure endpoint; brittle without good tooling |
| Transfer appliance | Only cost-effective option at huge scale (~100TB+) | One-time only, not for ongoing workloads |
| Data sharing | No physical data ownership needed | Access can be revoked by provider |

## Worked Example
A retailer builds a hybrid ingestion architecture. Order transaction data is captured via continuous, log-based CDC from a production PostgreSQL database (chosen over batch CDC because partial-day transaction history matters for fraud analytics), streamed through Kafka with a schema registry to handle firmware/app-driven schema evolution gracefully, and routed through a dead-letter queue for malformed events. A marketing SaaS platform's CRM data is ingested via a managed connector (Fivetran) rather than custom API code, since it's a well-supported, undifferentiated task. Daily aggregated reporting is produced via a time-interval batch ELT job orchestrated by Airflow, reading raw object-storage files exported nightly from the warehouse in Parquet (not CSV) format to preserve schema fidelity. A one-time migration of 150TB of historical archive data from an on-prem data center uses an AWS Snowball transfer appliance rather than network transfer, given the data volume. Throughout, sensitive PII fields are dropped or tokenized at ingestion time (never passed downstream unnecessarily), and all data in transit uses encrypted, VPC-scoped connections.

## Key Takeaways
1. All data is fundamentally unbounded; batch ingestion is a deliberate, convenient boundary-imposition with real latency trade-offs.
2. Prefer asynchronous, event-level pipeline coupling over tightly synchronous batch chains to avoid catastrophic all-or-nothing failures.
3. Log-based continuous CDC captures full change history with low source impact; batch/timestamp-based CDC is simpler but misses intra-window changes.
4. Avoid CSV in production ingestion where possible; prefer self-describing, nested-data-capable formats (Parquet, Avro, Arrow, ORC, JSON).
5. Outsource well-supported API/database connectors to managed platforms; reserve custom connector engineering for genuinely unsupported sources.
6. Build schema registries, dead-letter queues, and TTL policies into any streaming ingestion design from the start.
7. Treat reliability/durability investment as a cost-benefit decision, not an absolute — no architecture survives every conceivable failure scenario.
8. Minimize collection of sensitive data at the source; "data cannot leak if it is never collected."
9. Instrument monitoring (uptime, latency, volume, event/minute rates) into ingestion pipelines from day one — undetected ingestion failures (the chapter cites a six-month-undetected case) are a recurring real-world disaster pattern.

## Connects To
- **Ch 2**: Expands the "Ingestion" lifecycle stage and push/pull/batch/streaming concepts introduced there.
- **Ch 3**: Lambda/Kappa/unified batch-streaming architectures reused directly for ingestion pipeline design.
- **Ch 5**: Source system mechanics (CDC, logs, messages/streams, APIs, webhooks) applied concretely as ingestion mechanisms here.
- **Ch 6**: Object storage as the common landing zone/intermediate stage for most file-based and streaming ingestion.
- **Ch 8**: ETL vs. ELT and data integration/transformation, which this chapter deliberately defers.
- **External concept**: Tyler Akidau et al.'s Dataflow Model paper, Debezium/Kafka CDC tooling, Fivetran/Airbyte/Matillion managed connectors.
