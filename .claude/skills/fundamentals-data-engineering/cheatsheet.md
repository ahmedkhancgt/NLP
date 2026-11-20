# Cheatsheet

## Decision Rules

- **Choosing architecture vs. tools**: always architecture first (strategic, the what/why/when), technology second (tactical, the how). Never say "our architecture is tools X/Y/Z."
- **Build vs. buy**: build only if it's a genuine competitive advantage; otherwise use OSS/COSS/managed services (Type A default).
- **Batch vs. streaming**: default to batch/micro-batch unless a concrete, identified business use case justifies the added cost/complexity of true streaming.
- **CDC approach**: use log-based continuous CDC when full change history or near-real-time replication matters; batch/timestamp CDC is fine for simple "latest state" needs.
- **Single-cloud vs. multi/hybrid-cloud**: default single-cloud unless regulatory residency, multi-cloud customer proximity, or a specific best-of-breed service need exists.
- **Monolith vs. microservices for data**: start monolithic if you need speed; be ready to decompose as complexity grows — don't over-engineer microservices prematurely.
- **Serverless vs. servers**: serverless for simple, low-to-moderate-frequency, bursty workloads; servers/containers once scale, customization, or cost crossover favors it (see cost crossover curve).
- **Normalize vs. denormalize**: normalize (3NF, Inmon-style) when integration/consistency is the priority; denormalize (Kimball star schema, wide tables) when query speed/simplicity and iteration speed matter more.
- **Which data model**: Inmon for enterprise-wide single source of truth; Kimball for fast department-level iteration; Data Vault for fast-changing/source-fidelity-preserving ingestion; wide table when you don't care about modeling rigor and storage is cheap.
- **SCD type**: default to Type 2 (new row + effective dates) for full history; Type 1 (overwrite) only if history truly doesn't matter.
- **Self-service BI/ML**: only invest if the audience is genuinely data-literate and wants to build their own (e.g., data-savvy executives); most audiences want predefined dashboards instead.

## Trade-off Tables

| Storage medium | Latency | Relative cost/GB | Best fit |
|---|---|---|---|
| CPU cache | ~1 ns | N/A | Hot-path computation |
| RAM | ~100 ns | ~$10 | Caching, in-memory processing |
| SSD | ~0.1 ms | ~20-30¢ | OLTP, low-latency caching |
| HDD/object storage | ~4 ms | ~3¢ | Bulk analytics, archival-adjacent |

| Join strategy | Cost | When |
|---|---|---|
| Broadcast join | Low | One side fits on a single node |
| Shuffle hash join | High | Neither side fits on one node |

| Update pattern | Cost under COW | Preserves history? |
|---|---|---|
| Truncate and reload | Full rerun cost | No |
| Insert-only | Append cost + expensive "latest" lookup | Yes |
| Upsert/merge | High per-row, efficient in bulk | No (unless combined with insert-only) |

| Consistency model | Trade-off |
|---|---|
| Strong (ACID) | Correctness guaranteed, lower throughput/scale |
| Eventual (BASE) | High scale/throughput, possible stale reads |

## Thresholds & Numbers Worth Remembering

- **Magnetic disk**: ~3¢/GB, ~4ms avg latency, 50-500 IOPS, ~200-300 MB/s transfer.
- **SSD**: ~20-30¢/GB, <0.1ms random access, tens of thousands of IOPS.
- **RAM**: ~$10/GB, ~100ns latency (~1000x faster than SSD).
- **Compression**: a 10:1 ratio can turn 200 MB/s disk scan into effective 2 GB/s; can turn 10 Gbps network into effective 100 Gbps.
- **Transfer appliance threshold**: consider physical data transfer (e.g., AWS Snowball) once data size is around 100 TB+.
- **Cloud scale (repatriation justification)**: exabytes of data or terabits/sec of network traffic — most companies are nowhere near this; don't generalize from Dropbox/Netflix/Cloudflare examples.
- **Two-pizza rule**: no team larger than ~5 people should own a service/domain.
- **Technology reevaluation horizon**: ~2 years for reassessing transitory technology choices.
- **Archival storage example**: S3 Glacier Deep Archive ~$1/TB/month, ~12 hour retrieval, intended for data accessed 1-2x/year over 7-10 years.

## Tells & Smells (Red Flags)

- "Our architecture is Spark + Kafka + Snowflake" → confusing tools with architecture.
- A data team spending months/years deliberating technology choices without shipping → analysis paralysis; "time kills deals."
- Admin/superuser access granted to most users by default → least-privilege violation, security risk.
- Single-row inserts hammering a columnar/OLAP warehouse → anti-pattern; causes file bloat and reclustering overhead.
- Near-real-time CDC merges running directly into a columnar warehouse at production frequency → will bring the warehouse "to its knees"; batch the merges instead.
- `SELECT *` with no predicates in production code against a columnar store → full table scan, wasted cost.
- A report/dashboard built urgently that nobody has opened in weeks → candidate for retirement (reduces security surface too).
- Credentials pasted directly into notebook/script code or committed to version control → urgent security fix needed.
- A single team or architect mandating one proprietary technology org-wide → command-and-control anti-pattern, innovation bottleneck.
- CSV used as a production interchange format with no documented delimiter/encoding/schema → high risk of silent corruption; prefer Parquet/Avro/JSON.
- A vendor benchmark showing a "10x" win with no workload detail → likely cherry-picked (small dataset, asymmetric tuning, non-comparable pricing model).
- A data lake with no catalog, no schema management, "write once read never" files accumulating → data swamp in progress.
- Streaming adopted with no identified business action to trigger → "real-time data without action is an unrelenting distraction."
- Public S3 buckets, 0.0.0.0/0 inbound SSH, or open database ports → common, serious, and real-world-observed security failures.

## Quick Checklists

**Before choosing a source-system ingestion strategy, ask:**
1. What's the use case and destination?
2. Batch or streaming? Push, pull, or poll?
3. What's the payload's kind/shape/size/schema?
4. Is there a data contract/SLA with the source owner?

**Before adopting a new technology, ask:**
1. Have we fixed the architecture first?
2. What's the TCO, TOCO, and FinOps profile?
3. Is this immutable (safe to build on) or transitory (should be swappable)?
4. What's our exit plan if this fails or becomes obsolete?

**Before serving data to a new consumer, ask:**
1. Do they trust this data, and why should they?
2. What SLA/SLO are we committing to?
3. What's the minimum access they need (least privilege)?
4. Is the business logic/definition centralized (semantic layer) or duplicated ad hoc?
