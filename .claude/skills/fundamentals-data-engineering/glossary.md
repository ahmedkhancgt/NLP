# Glossary

**ACID** — Atomicity, Consistency, Isolation, Durability; transactional guarantees for database correctness (Ch 5, 8).

**Avro** — Row-oriented binary serialization format with JSON schema metadata, popular in the Hadoop ecosystem (Ch 6).

**BASE** — Basically Available, Soft-state, Eventual consistency; the distributed-systems counterpart to ACID (Ch 6).

**Batch ingestion** — Processing data in discrete chunks based on time interval or accumulated size (Ch 2, 7).

**Big data** — Historical term for the 2000s-2010s complexity regime (3 Vs: velocity, variety, volume) now absorbed into mainstream data engineering (Ch 1).

**Block storage** — Raw, addressable disk storage units underlying SSDs/HDDs and VM boot disks (e.g., EBS, SAN) (Ch 6).

**Broadcast join** — A join where a small table is copied to every node and joined locally against a large distributed table (Ch 8).

**CDC (Change Data Capture)** — Extracting insert/update/delete events from a database, via batch (timestamp polling) or continuous (log-based) methods (Ch 2, 5, 7).

**Columnar storage** — Data organized by column rather than row, enabling selective scans and high compression (Ch 6, 8).

**Conformed dimension** — A dimension table reused across multiple Kimball star schemas (Ch 8).

**COW (Copy-on-Write)** — The mechanism by which object/columnar storage handles updates by rewriting files rather than modifying in place (Ch 8).

**Data architecture** — The design of systems to support an enterprise's evolving data needs via flexible, reversible, trade-off-evaluated decisions (Ch 3).

**Data contract** — A written agreement between a source system owner and an ingesting team specifying data, extraction method, cadence, and contacts (Ch 5, 9).

**Data engineering lifecycle** — Generation → Storage → Ingestion → Transformation → Serving, wrapped by six undercurrents (Ch 1, 2).

**Data gravity** — The tendency for data to become expensive/difficult to move once it lands in a given cloud, due to egress fees and migration friction (Ch 4).

**Data lake** — A storage architecture retaining massive amounts of raw, often unprocessed data, typically on object storage (Ch 3, 6).

**Data lakehouse** — An architecture combining lake-style object storage with warehouse-like schema management, ACID updates/deletes, and table versioning (e.g., Delta Lake, Hudi, Iceberg) (Ch 3, 6).

**Data management** — The set of best practices (governance, modeling, lineage, MDM, ethics/privacy) data engineers use to manage the data lifecycle (Ch 2).

**Data maturity model** — Starting / Scaling / Leading with data; a three-stage model calibrating what a data engineer should prioritize (Ch 1).

**Data mesh** — A decentralized architecture where domain teams own and serve their own data as a product, with federated governance (Ch 3, 9).

**Data Vault** — A modeling methodology using hubs (business keys), links (relationships), and satellites (attributes), insert-only (Ch 8).

**Data warehouse** — A centralized, structured OLAP data architecture for analytics/reporting (Ch 3, 6, 8).

**DataOps** — The application of Agile/DevOps/statistical process control principles to data, built on automation, observability/monitoring, and incident response (Ch 2).

**Dead-letter queue** — A separate store for streaming events that fail ingestion, preventing them from blocking the main pipeline (Ch 7).

**Dimension table** — A wide, short, descriptive table providing context to a fact table in a Kimball star schema (Ch 8).

**Distributed monolith** — A distributed system (e.g., Hadoop cluster, Airflow executors) whose nodes still share a tightly coupled codebase/dependency set (Ch 4).

**ELT (Extract, Load, Transform)** — Loading raw data first, then transforming inside the target system (contrast ETL) (Ch 3, 7, 8).

**ETL (Extract, Transform, Load)** — Transforming data in an external system before loading into the target (Ch 3, 7, 8).

**Eventual consistency** — A distributed-systems model where reads may return stale data temporarily but converge to correctness over time (Ch 6).

**Fact table** — A narrow, long, immutable table of numeric/event measures at the lowest grain, referencing dimensions (Ch 8).

**FinOps** — Cloud financial management discipline treating cloud spend as a continuously monitored, business-value-tied variable (Ch 3, 4).

**Grain** — The resolution at which data is stored/queried, typically at a primary-key level (Ch 8).

**Hadoop (HDFS)** — A distributed file system combining compute and storage on the same cluster nodes, historically central to "big data" (Ch 6).

**Idempotency** — A property where processing the same message/event multiple times produces the same result as processing it once (Ch 5, 7).

**Inmon approach** — Top-down, highly normalized (3NF) enterprise data warehouse modeling, with department marts fed downstream (Ch 8).

**Kappa architecture** — An architecture using a single stream-processing backbone for all data handling (ingestion, storage, serving) (Ch 3, 8).

**Kimball approach** — Bottom-up dimensional modeling using star schemas (facts + dimensions) (Ch 8).

**Lambda architecture** — An architecture with separate batch, streaming, and serving layers reconciled at query time (Ch 3, 8).

**Least privilege (principle of)** — Giving a user or system only the access essential to their function, for only as long as needed (Ch 2, 3, 10).

**Lindy effect** — The heuristic that the longer a technology has existed, the longer it's likely to persist — used to identify "immutable" technologies (Ch 4).

**Live data stack** — The authors' speculative successor to the modern data stack, fusing real-time streaming, analytics, and ML directly into applications (Ch 11).

**Materialized view** — A view whose results are precomputed and stored, acting as a de facto transformation managed by the database (Ch 8).

**MDM (Master Data Management)** — The practice of building consistent "golden record" entity definitions across an organization (Ch 2, 8).

**Metadata** — Data about data; divided into business, technical, operational, and reference categories (Ch 2).

**Modern Data Stack (MDS)** — A trendy, cloud-based, plug-and-play, modular analytics architecture built around ELT and cloud data warehouses (Ch 3, 11).

**Normalization** — A relational modeling discipline (1NF/2NF/3NF) removing data redundancy and ensuring referential integrity (Ch 8).

**Object storage** — A flat, key-value store of immutable byte-stream objects with no directory tree or in-place updates (e.g., S3, GCS) — the backbone of modern data lakes (Ch 6).

**OLAP (Online Analytical Processing)** — Systems optimized for large analytical scans, inefficient at individual record lookups (Ch 5, 8).

**OLTP (Online Transaction Processing)** — Systems supporting low-latency, high-concurrency individual record reads/writes (Ch 5, 8).

**Orchestration** — Dependency-aware coordination of many jobs (DAGs), distinct from simple time-based scheduling (Ch 2, 4, 7).

**Partitioning / Clustering** — Splitting a table by field (partition) and sorting within partitions (clustering) to reduce scan volume (Ch 6, 8).

**Query optimizer** — The database component that reorders/refactors a query's execution plan for lowest cost (Ch 8).

**Reverse ETL** — Feeding processed/served data from an OLAP system back into source systems (e.g., CRM) for action (Ch 2, 5, 9).

**Row explosion** — A join producing an explosive cross-product of rows due to repeated join-key values on both sides (Ch 8).

**Schema on read / Schema on write** — Interpreting structure at query time (flexible, lake-style) vs. enforcing it at write time (strict, warehouse-style) (Ch 6).

**Schema registry** — A metadata repository tracking schema versions for streaming messages (Ch 7, 8).

**SCD (Slowly Changing Dimension)** — Type 1 (overwrite), Type 2 (new row + effective dates), Type 3 (new column) patterns for tracking dimension attribute changes over time (Ch 8).

**Semantic / metrics layer** — A reusable layer encoding business logic/metric definitions once, generating consistent queries wherever referenced (e.g., LookML, dbt) (Ch 8, 9).

**Serialization** — Flattening/packing data into a standard interchange/storage format (row-based or columnar) (Ch 6, 7).

**Shared responsibility model** — Cloud vendor secures "the cloud" (infrastructure); customer secures "in the cloud" (data, access, configuration) (Ch 3, 10).

**Shuffle hash join** — A join where both tables are repartitioned by join key across a cluster — more resource-intensive than a broadcast join (Ch 8).

**SLA / SLO** — Service Level Agreement (the contract stating expectations) and Service Level Objective (the measurable commitment against it) (Ch 5, 9).

**Star schema** — A fact table surrounded by dimension tables in the Kimball modeling approach (Ch 8).

**Streaming windows** — Session (inactivity-gap-based), fixed/tumbling (fixed non-overlapping), and sliding (overlapping fixed-length) mechanisms for bounding streaming computation (Ch 8).

**TOCO (Total Opportunity Cost of Ownership)** — The cost of foreclosed alternatives incurred by committing to a given technology/architecture (Ch 4).

**TTL (Time to Live)** — Maximum retention time for an unacknowledged streaming message before it's dropped (Ch 7).

**Type A / Type B data engineer** — Type A (Abstraction) favors off-the-shelf/managed tools; Type B (Build) builds custom systems for competitive advantage (Ch 1, 4).

**Vacuuming** — Removing dead/obsolete record versions left behind by updates/deletes to reclaim space and keep query plans accurate (Ch 8).

**Watermark** — A threshold in streaming systems determining whether arriving data is within the expected time window or considered late (Ch 8).

**Wide table / denormalized table** — A highly denormalized, often sparse, wide columnar table combining facts and dimensions without join-based modeling rigor (Ch 8).

**Zero-copy cloning** — Creating a new logical table/object copy via pointer duplication rather than physically copying underlying storage (Ch 6).

**Zero-trust security** — A model assuming no implicit trust based on network location; every action requires authentication (Ch 3, 10).
