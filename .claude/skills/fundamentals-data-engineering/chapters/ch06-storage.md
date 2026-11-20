# Chapter 6: Storage

## Core Idea
Storage underlies every stage of the data engineering lifecycle ("it's storage all the way down"); choosing the right storage requires understanding raw physical ingredients (disks, memory, networking, serialization, compression, caching), the storage systems built on them (block, file, object, streaming), and the higher-level abstractions (warehouse, lake, lakehouse) best suited to a given access pattern and cost profile.

## Frameworks Introduced
- **Cache hierarchy**: CPU cache → RAM → SSD → magnetic disk/object storage → archival — each tier trades latency/cost for capacity/durability.
  - When to use: designing any system with performance-sensitive and cost-sensitive data simultaneously.
  - How: place frequently accessed ("hot") data in faster/costlier tiers, infrequently accessed ("warm"/"cold") data in slower/cheaper tiers; use lifecycle policies to automate migration.
- **Eventual vs. Strong Consistency (BASE vs. ACID)**: BASE = basically available, soft-state, eventual consistency; the distributed-systems counterpart to ACID.
  - When to use: choosing/configuring a distributed database or object store.
  - How: accept eventual consistency for horizontal scalability and speed; pay for strong consistency (consensus-based writes, slower strongly-consistent reads) only when correctness-on-every-read is required.
- **Storage Lifecycle & Data Retention framework**: evaluate data by value, time, compliance, and cost to decide hot/warm/cold placement and retention duration.
  - When to use: any storage architecture decision involving growing datasets.
  - How: ask "how often is this accessed, how valuable is it, what regulation governs it, and what does keeping/discarding it cost?" then apply automated lifecycle policies (tiering, TTLs, archival/deletion).
- **Compute/Storage Colocation vs. Separation**: colocation (data and compute on the same node) maximizes raw performance; separation (object storage + ephemeral compute) maximizes elasticity, durability, and cost efficiency.
  - When to use: architecting any large-scale batch or analytical system.
  - How: default to separation (cloud-native pattern) for flexibility; use hybrid patterns (multitier caching, hybrid object storage) when raw colocated performance is needed for specific workloads.
- **Schema on Write vs. Schema on Read**: write-time enforcement (data warehouse pattern, requires a metastore) vs. read-time interpretation (data lake pattern, relies on self-describing formats like Parquet/JSON).
  - When to use: deciding how strict to be about structure before storage vs. at query time.
  - How: schema-on-write for consistency/ease-of-future-consumption; schema-on-read for flexibility/speed of ingestion, accepting harder future consumption.
- **Single-Tenant vs. Multitenant Storage**: dedicated storage per tenant vs. shared storage across tenants (isolated via logical controls).
  - When to use: designing storage for SaaS or multi-customer systems.
  - How: single-tenant when strict data isolation is contractually/regulatorily required; multitenant when operational simplicity and resource sharing outweigh isolation needs — enforce isolation via views, row/column-level security.

## Key Concepts
- **HDD vs. SSD**: HDD = cheap per GB (~3¢/GB), slow (seek time, rotational latency, ~4ms avg latency, 50-500 IOPS) but parallelizable at scale (object storage); SSD = fast (sub-0.1ms random access, high IOPS) but ~10x more expensive per GB — SSD dominates OLTP, HDD/object storage dominates bulk analytics.
- **RAM characteristics**: volatile, CPU-attached, ~100ns latency (1000x faster than SSD), ~$10/GB, limited capacity per machine — used for caching/indexes/in-memory processing, never as sole durable storage without backup mechanisms.
- **Serialization**: the process of flattening in-memory data into a standard interchange/storage format (row-based: XML/JSON/CSV/RDBMS rows; columnar: Parquet-style column files) — choice affects compression, query performance, and interoperability.
- **Compression trade-offs**: smaller storage footprint, faster effective disk scan rate, and higher effective network bandwidth — at the cost of CPU time to compress/decompress.
- **Block vs. File vs. Object storage**: block storage = raw addressable disk units (SSD/magnetic), used by transactional DBs and VM boot disks (e.g., EBS, SAN); file storage = hierarchical directory tree with random access/append (local disk, NAS, cloud filesystems like EFS); object storage = flat key-value store of immutable byte streams (no directory tree, no in-place updates — rewrite-whole-object only), e.g., S3/GCS/Azure Blob — the backbone of modern data lakes.
- **Object storage immutability & versioning**: objects can't be updated/appended in place; updates create new object versions; eventual consistency historically applied to S3; versioning adds storage cost but enables rollback and strong per-version referencing.
- **HDFS**: combines compute and storage on the same cluster nodes (NameNode manages metadata, blocks replicated 3x) — distinct from object storage's separated compute/storage model; still common in legacy/large-scale on-prem deployments, less so for new builds.
- **Columnar storage + micro-partitioning/clustering**: columnar formats enable scanning only needed columns and high compression; partitioning (e.g., by date) and clustering (sorting within partitions) further prune scans — Snowflake's micro-partitioning automatically clusters similar rows and prunes via per-partition metadata instead of a traditional single-key partition scheme.
- **Data catalog**: centralized metadata store integrating lineage, discoverability, and a social/wiki layer across operational and analytics systems — essential for data lakehouse table discoverability and cross-org collaboration.
- **Zero-copy cloning**: creating a new logical copy (e.g., a table) via pointer duplication rather than physically copying underlying object storage data — fast and cheap, but deleting underlying raw files can unexpectedly corrupt "cloned" copies unless a deep copy was made.

## Mental Models
- **Storage as "ingredients → systems → abstractions"**: raw physical media (disk, RAM, network, serialization, compression, caching) compose into storage systems (block/file/object/streaming), which compose into higher-level abstractions (warehouse/lake/lakehouse/platform) — understand each layer to make informed trade-offs at the layer above.
- **Hot/warm/cold as a cost-latency dial**: data temperature determines appropriate storage tier; most architectures need automated lifecycle policies to migrate data down the temperature scale as it ages, rather than manual one-off decisions.
- **Archival storage as a "reverse cache"**: optimizes for low cost at the expense of access speed/latency, the mirror image of a cache optimizing for speed at the expense of cost/capacity.
- **Separation of compute and storage is now the cloud-native default**, but real systems hybridize it (multitier caching, hybrid object storage) to recapture colocation-style performance where it matters most.

## Anti-patterns
- **Running random-access, high-frequency update workloads against object storage**: object stores are optimized for large batch reads/writes, not many small transactional updates per second — use block storage or a transactional database instead.
- **Treating RAM as durable storage without backup**: a power/node failure can silently erase unreplicated, unpersisted in-memory data.
- **Ignoring object store eventual consistency**: assuming a freshly-written object will always be immediately visible to all readers can cause subtle correctness bugs; use versioning or a strongly-consistent metadata layer when correctness matters.
- **Hoarding all data in hot storage "just in case"**: drives storage costs needlessly high; apply deliberate lifecycle/retention policies instead.
- **Data hoarding without retention discipline ("we might need this data someday")**: recreates data-swamp dynamics and invites regulatory risk (GDPR/CCPA deletion obligations).
- **Treating cloud object "directories" as real directory trees**: costly prefix-filtering operations (e.g., `ls` on a huge bucket) are often mistaken for fast filesystem traversal.
- **Deleting raw files under a zero-copy "clone" without understanding shallow-copy semantics**: can silently destroy data in what appears to be an independent copy.

## Code Examples
```
S3://oreilly-data-engineering-book/data-example.json
```
- **What it demonstrates**: object storage addressing — `oreilly-data-engineering-book` is the bucket, `data-example.json` is the key; there is no real directory traversal even when paths look hierarchical.

```
WHERE created_date='2022-01-02'
```
- **What it demonstrates**: a predicate that Snowflake's micro-partition metadata uses to prune irrelevant partitions from a scan without a traditional index.

## Reference Tables

| Storage medium | Latency | Relative cost | Best fit |
|---|---|---|---|
| CPU cache | ~1 ns | N/A (on-die) | Hot-path CPU computation |
| RAM | ~100 ns | ~$10/GB | Caching, in-memory processing |
| SSD | ~0.1 ms | ~20-30¢/GB | OLTP, low-latency OLAP caching |
| HDD / object storage | ~4 ms (HDD) | ~3¢/GB | Bulk analytics, data lakes, archival-adjacent |

| Storage abstraction | Core trait | Best for |
|---|---|---|
| Data warehouse | Structured, schema-on-write, OLAP-optimized | BI, reporting, governed structured data |
| Data lake | Raw/unstructured, cheap object storage | Flexible retention of any data type |
| Data lakehouse | Lake storage + warehouse-like ACID/schema management (Delta Lake, Hudi, Iceberg) | Interoperable, open-format analytics with update/delete support |
| Data platform | Vendor-integrated ecosystem on top of object storage | Simplified ops at the cost of lock-in |

| Temperature | Access frequency | Storage cost | Retrieval cost |
|---|---|---|---|
| Hot | Constant/frequent | Highest | Lowest |
| Warm | Periodic (e.g., monthly) | Medium | Medium |
| Cold | Rare/archival | Lowest | Highest |

## Worked Example
A data lakehouse architecture: raw event data lands in cloud object storage (S3) as the durability/ingestion layer (schema-on-read, JSON/Parquet). A lakehouse management layer (e.g., Delta Lake) adds a metadata/transaction log on top, enabling ACID updates/deletes (satisfying GDPR deletion requests) and table versioning (rollback support) while keeping data in open Parquet format for interoperability across query engines. Frequently queried recent partitions (hot data, e.g., last 30 days) are clustered/partitioned by date for scan pruning and may be cached on SSD-backed compute during active ELT jobs (hybrid colocation, like AWS EMR pulling from S3 into local HDFS for intermediate steps). Older partitions (cold data, >1 year) are moved via automated lifecycle policy to an archival storage tier (e.g., S3 Glacier Deep Archive) at ~$1/TB/month, accepting hours-long retrieval latency since they're accessed only for occasional compliance audits. A data catalog scans and indexes all of this, giving analysts a searchable, lineage-aware view across the hot, warm, and cold tiers without needing to know the underlying physical storage details.

## Key Takeaways
1. Understand the raw physical trade-offs (HDD vs. SSD vs. RAM) underlying every storage system you choose.
2. Object storage (key-value, immutable, no directory tree) is the modern backbone of data lakes and increasingly of cloud data warehouses themselves.
3. Eventual vs. strong consistency is a deliberate engineering trade-off, not a bug — know which model your storage system uses and where it matters.
4. Columnar storage plus partitioning/clustering is the standard pattern for scalable analytical scan performance.
5. Data lakehouse architectures (Delta Lake, Hudi, Iceberg) bring warehouse-grade ACID/schema management to lake-style object storage, closing historical data-lake gaps (updates, deletes, versioning).
6. Separate compute from storage by default for elasticity and cost efficiency; hybridize with colocation/caching only where performance demands it.
7. Apply systematic hot/warm/cold tiering and retention policy based on value, time, compliance, and cost — don't hoard data indefinitely in hot storage.
8. Single- vs. multitenant storage design has major implications for isolation, schema uniformity, and cross-tenant querying.

## Connects To
- **Ch 2**: Storage introduced as a foundational stage underlying the whole lifecycle.
- **Ch 3**: Data warehouse, data lake, lakehouse, and data mesh/sharing architectures expanded here at the storage-systems level.
- **Ch 5**: Source-system storage (databases, logs, streams) as the upstream origin feeding these storage systems.
- **Ch 7**: Ingestion mechanics that move data into these storage systems (batch/streaming, replay).
- **Ch 8**: Data modeling and query patterns (normalization, partitioning, indexing) that build on these storage foundations.
- **External concept**: Martin Kleppmann's Designing Data-Intensive Applications, Delta Lake/Apache Hudi/Apache Iceberg, Snowflake's hybrid columnar storage paper.
