---
name: data-engineering-design-patterns
description: "Knowledge base from \"Data Engineering Design Patterns\" by Bartosz Konieczny. Use when designing data pipelines — data ingestion, error management, idempotency, data value/flow patterns, data security, data storage, data quality, or data observability design patterns."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Engineering Design Patterns
**Author**: Bartosz Konieczny | **Chapters**: 10 | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a pattern or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch05`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

This is fundamentally a design-patterns catalog for data pipelines, organized by the natural flow: ingest → handle errors → guarantee idempotency → create value → connect flows → secure → store → assure quality → observe.

**Ingestion (Ch2)**: Use **Full Loader** when there's no delta-tracking signal; **Incremental Loader** once you have a delta column or time partitions; **Change Data Capture** for low-latency, delete-aware streaming from a DB commit log. **Readiness Marker** signals dataset completeness (pull); **External Trigger** reacts to unpredictable arrivals (push). **Compactor** fixes the small-files problem.

**Error Management (Ch3)**: **Dead-Letter** isolates unprocessable records without stopping the job. **Windowed Deduplicator** bounds duplicate removal to a time/dataset scope. **Late Data Detector** uses a monotonically increasing watermark (MAX(event time) − allowed lateness); **Static/Dynamic Late Data Integrator** replay late data via a fixed or state-tracked lookback window. **Checkpointer** gives fault tolerance, not exactly-once.

**Idempotency (Ch4)**: Full dataset → **Fast Metadata Cleaner** (TRUNCATE/DROP) or **Data Overwrite**. Incremental dataset → **Merger** (MERGE/UPSERT, soft-delete flag); add a state table for backfill-safe restores → **Stateful Merger**. Key-based stores → **Keyed Idempotency** (build keys from append/ingestion time, never event time). Need atomic visibility → **Transactional Writer**. Must stay immutable → **Proxy** (write-once + indirection).

**Data Value (Ch5)**: **Static Joiner** (at-rest enrichment, + SCD for time-sensitivity) vs. **Dynamic Joiner** (stream-to-stream, time-bounded buffer + GC watermark). **Wrapper** exposes raw+computed fields; **Metadata Decorator** hides technical context. **Distributed Aggregator** (shuffle) vs. **Local Aggregator** (no shuffle, requires stable partitioning). **Incremental/Stateful Sessionizer** build sessions from batch/streaming. **Bin Pack Orderer**/**FIFO Orderer** preserve delivery order.

**Data Flow (Ch6)**: **Local Sequencer** (in-pipeline) / **Isolated Sequencer** (cross-team, data-based=loose vs. task-based=tight coupling). **Aligned Fan-In** (all parents succeed) vs. **Unaligned Fan-In** (partial + completeness flag). **Parallel Split** (cache the shared intermediate!) vs. **Exclusive Choice** (one branch). **Single Runner** (stateful pipelines) vs. **Concurrent Runner** (independent instances).

**Data Security (Ch7)**: **Vertical Partitioner**/**In-Place Overwriter** for data removal compliance. **Fine-Grained Accessor** for tables (column/row) and cloud resources. **Encryptor** as defense-in-depth beneath access control. **Anonymizer** (irreversible) vs. **Pseudo-Anonymizer** (reversible, weaker, re-identification risk when combined). **Secrets Pointer** (reference credentials) vs. **Secretless Connector** (eliminate them via IAM/certificates).

**Data Storage (Ch8)**: **Horizontal Partitioner** for low-cardinality keys; **Bucket** for high-cardinality ones. **Sorter**/**Z-order** enable data-block skipping, reinforced by **Metadata Enhancer** statistics. **Dataset Materializer** precomputes expensive queries; **Manifest** avoids repeated listing. **Normalizer** (consistency, OLTP) vs. **Denormalizer** (speed, OLAP) — not mutually exclusive.

**Data Quality (Ch9)**: **AWAP** (Audit-Write-Audit-Publish) validates input and output around a transform. **Constraints Enforcer** delegates validation to the store natively. **Schema Compatibility Enforcer** prevents breaking changes (backward/forward/full, transitive/nontransitive); **Schema Migrator** enables them safely (add-then-deprecate). **Offline Observer** (decoupled) vs. **Online Observer** (real-time, isolate failures).

**Data Observability (Ch10)**: **Flow Interruption Detector** (silent stoppage), **Skew Detector** (volume anomaly), **Lag Detector** (consumer falling behind — use percentiles, not averages), **SLA Misses Detector** (deadline breach — track processing-time and event-time separately). **Dataset Tracker** (lineage graph) and **Fine-Grained Tracker** (column/row-level lineage).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Introducing Data Engineering Design Patterns | Design pattern definition, Medallion architecture, case study |
| ch02 | Data Ingestion Design Patterns | Full Loader, Incremental Loader, CDC, Passthrough/Transformation Replicator, Compactor, Readiness Marker, External Trigger |
| ch03 | Error Management Design Patterns | Dead-Letter, Windowed Deduplicator, Late Data Detector, Static/Dynamic Late Data Integrator, Filter Interceptor, Checkpointer |
| ch04 | Idempotency Design Patterns | Fast Metadata Cleaner, Data Overwrite, Merger, Stateful Merger, Keyed Idempotency, Transactional Writer, Proxy |
| ch05 | Data Value Design Patterns | Static/Dynamic Joiner, Wrapper, Metadata Decorator, Distributed/Local Aggregator, Incremental/Stateful Sessionizer, Bin Pack/FIFO Orderer |
| ch06 | Data Flow Design Patterns | Local/Isolated Sequencer, Aligned/Unaligned Fan-In, Parallel Split, Exclusive Choice, Single/Concurrent Runner |
| ch07 | Data Security Design Patterns | Vertical Partitioner, In-Place Overwriter, Fine-Grained Accessor, Encryptor, Anonymizer/Pseudo-Anonymizer, Secrets Pointer, Secretless Connector |
| ch08 | Data Storage Design Patterns | Horizontal/Vertical Partitioner, Bucket, Sorter, Metadata Enhancer, Dataset Materializer, Manifest, Normalizer, Denormalizer |
| ch09 | Data Quality Design Patterns | AWAP, Constraints Enforcer, Schema Compatibility Enforcer, Schema Migrator, Offline/Online Observer |
| ch10 | Data Observability Design Patterns | Flow Interruption/Skew/Lag/SLA Misses Detector, Dataset Tracker, Fine-Grained Tracker |

## Topic Index

| Topic / Pattern | Chapter(s) |
|---|---|
| Anonymization / PII | ch07 |
| AWAP (audit) | ch09 |
| Backfilling / snowball effect | ch03, ch04, ch05 |
| Bucketing | ch08, ch05 |
| CDC (Change Data Capture) | ch02 |
| Checkpointing | ch03 |
| Credentials / secrets | ch07 |
| Data lineage | ch10, ch07, ch09 |
| Dead-Letter | ch03 |
| Denormalization / Normalization | ch08 |
| Encryption | ch07 |
| Fan-in / Fan-out | ch06 |
| Idempotent keys | ch04 |
| Late data / watermarks | ch03, ch05, ch10 |
| Materialized views | ch08 |
| Merger / MERGE / UPSERT | ch04 |
| Partitioning (horizontal/vertical) | ch08, ch07 |
| Readiness Marker | ch02, ch06 |
| Schema evolution | ch09 |
| Sessionization | ch05 |
| Skew (data/lag) | ch05, ch08, ch10 |
| SLA monitoring | ch10 |
| Sorter / Z-order | ch08 |
| State store | ch03, ch05 |

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — all named design patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 10 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
