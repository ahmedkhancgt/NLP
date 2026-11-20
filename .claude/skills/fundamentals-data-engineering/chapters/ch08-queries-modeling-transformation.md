# Chapter 8: Queries, Modeling, and Transformation

## Core Idea
This is where data becomes valuable: queries retrieve/manipulate data, data modeling imposes a coherent structure reflecting business logic, and transformations persist the results of complex multi-step processing so expensive work isn't repeated — together they turn raw ingredients into analytics- and ML-ready data.

## Frameworks Introduced
- **Conceptual → Logical → Physical data model continuum**: conceptual = business entities/rules (ER diagrams); logical = types, primary/foreign keys; physical = actual database/schema/table implementation.
  - When to use: any new data modeling effort.
  - How: start by capturing business entities and relationships with stakeholders (conceptual), then formalize types/keys (logical), then implement in a specific system (physical).
- **Codd's Normal Forms (1NF/2NF/3NF)**: progressively remove redundancy — 1NF (atomic, unique columns + primary key), 2NF (no partial dependencies), 3NF (no transitive dependencies).
  - When to use: relational/OLTP-style modeling where update consistency matters.
  - How: split tables until each non-key column depends only on the whole primary key and nothing but the key.
- **Kimball (star schema: facts + dimensions)**: bottom-up, department-oriented, denormalized-friendly; fact tables (narrow, long, immutable, lowest grain) surrounded by dimension tables (wide, short, descriptive, denormalized, "conformed" when shared across stars).
  - When to use: business/department-facing analytics where query simplicity and speed matter more than strict integration.
  - How: identify grain, build the fact table of numeric/event measures, surround it with dimensions, reuse conformed dimensions across multiple star schemas.
- **Inmon (top-down, normalized data warehouse)**: subject-oriented, integrated, nonvolatile, time-variant; highly normalized (3NF) enterprise warehouse as single source of truth, with department-specific star-schema data marts fed downstream.
  - When to use: enterprise-wide integration is the priority; want one authoritative "corporate image" of all business data.
  - How: ETL from all key source systems into a 3NF warehouse; then ETL again from the warehouse into department-specific data marts.
- **Data Vault (hubs, links, satellites)**: separates structural keys (hubs: unique business keys) from relationships (links: many-to-many connections) from context (satellites: descriptive attributes); insert-only, no notion of "correct" data at load time.
  - When to use: fast-changing, agile, source-fidelity-preserving ingestion layer, especially as a landing zone before further Kimball/Inmon-style modeling.
  - How: load business keys into hubs, relationships into links, attributes into satellites; interpret business logic only at query time.
- **Wide/denormalized tables**: throw (nearly) everything into one wide, sparse, columnar table; join-free queries, cheap in columnar storage (nulls nearly free, selective column reads).
  - When to use: cheap cloud storage + flexible/nested source schemas + speed-of-delivery priority over modeling rigor.
  - How: pick a grain and key(s), load all relevant fields (including nested structures) into a single wide table; accept some loss of encoded business logic.
- **Slowly Changing Dimensions (SCD) Types 1/2/3**: Type 1 = overwrite (no history); Type 2 = insert new row with effective start/end dates (full history); Type 3 = add new columns for old/new values (limited history).
  - When to use: dimension attributes that change over time and where history matters for analysis.
  - How: default to Type 2 in practice for full historical tracking via effective-date ranges.
- **Broadcast join vs. Shuffle hash join**: broadcast = small table copied to every node, joined locally (cheap); shuffle hash = both tables repartitioned by join key across the cluster (expensive).
  - When to use: broadcast when one side of a join is small enough to fit on one node; shuffle hash otherwise.
  - How: prefilter aggressively before joining to maximize the chance of a broadcast join; understand your engine's join-reordering behavior.

## Key Concepts
- **DDL / DML / DCL / TCL**: Data Definition Language (CREATE/DROP — defines objects), Data Manipulation Language (SELECT/INSERT/UPDATE/DELETE/MERGE — manipulates data), Data Control Language (GRANT/REVOKE — access control), Transaction Control Language (COMMIT/ROLLBACK — transaction boundaries).
- **Query optimizer / explain plan**: the optimizer reorders/refactors query steps for lowest cost; the explain plan reveals this chosen execution strategy and resource usage, essential for diagnosing slow queries.
- **Row explosion**: repeated join-key values on both sides of a join cause a cross-product blowup (e.g., 5 × 10 = 50 rows from one value) that can exhaust resources — understand join cardinality before running production joins.
- **Commit models vary by database**: PostgreSQL (ACID, row locking, risk of dirty reads if misconfigured) vs. BigQuery (point-in-time snapshot reads, single-writer-at-a-time queuing) vs. MongoDB (variable/configurable consistency, can silently drop writes under load) — know your database's model before you build on it.
- **Vacuuming**: removing dead/obsolete record versions left behind by updates/deletes to reclaim space, keep query plans accurate, and maintain index performance; handled differently per system (Snowflake time-travel window, BigQuery fixed 7-day history, Databricks manual VACUUM, Postgres/MySQL explicit vacuum).
- **Grain**: the resolution at which data is stored (e.g., order-line-item level vs. daily summary) — always model at the lowest practical grain since you can always aggregate up but can't recover detail once aggregated away.
- **Update patterns**: truncate-and-reload (wipe + full rerun), insert-only (append new versions, query finds latest), delete (hard vs. soft vs. insert-with-deleted-flag), upsert/merge (match-then-update-or-insert) — all complicated in columnar/object-storage systems by copy-on-write (COW), where any change rewrites files/partitions rather than updating in place.
- **Views, materialized views, federated queries, data virtualization**: a view is a saved query (no precomputation); a materialized view precomputes and stores results (a de facto transformation managed by the database); federated queries let an OLAP engine read external sources (object storage, other RDBMSs) directly; data virtualization (Trino/Presto) processes across sources without storing data internally, often using query pushdown to offload work to the source.
- **Streaming windows**: session windows (group by inactivity gaps, per-key), fixed/tumbling windows (fixed non-overlapping time periods), sliding windows (overlapping fixed-length windows); watermarks define the threshold past which arriving data is considered "late."
- **Micro-batch vs. true streaming**: micro-batch applies batch frameworks at high frequency (e.g., Spark Streaming); true streaming (Beam, Flink) processes event-by-event — the right choice depends on acceptable latency, not ideology.

## Mental Models
- **Query vs. transformation**: a query retrieves/computes transiently; a transformation persists results for reuse by other transformations/queries — once compute-intensive logic needs reuse, promote it from query to transformation.
- **Transformations and queries are a continuum, especially in streaming**: the line blurs when streaming computations (windowed aggregation feeding another stream) could be called either.
- **"Batch is a special case of streaming"** (repeated theme): windowed/triggered streaming computation subsumes traditional scheduled batch ETL as a special, less-frequent case.
- **Derived data is unavoidable, not avoidable**: business logic (e.g., multiple profit definitions) must live somewhere — either baked into brittle, duplicated ETL scripts, or centralized in a semantic/metrics layer that generates queries from defined metrics (preferred emerging pattern).
- **Model at the lowest grain, aggregate on demand**: you can always summarize granular data; you can never reconstruct detail from pre-aggregated data.

## Anti-patterns
- **`SELECT *` with no predicates / full table scans**: scans and bills for everything when you need only a subset — always prune columns (columnar) or use indexes (row-oriented).
- **Single-row inserts into columnar/OLAP systems**: creates many small inefficient files requiring later reclustering; load data in micro-batches instead.
- **Near-real-time CDC-driven merges into columnar warehouses at production OLTP frequency**: "will bring most columnar data warehouses to their knees" — batch merges hourly (or use streaming-buffer-aware systems like BigQuery/Druid) instead.
- **Running analytics directly on production OLTP databases**: can crash the production system (the chapter's cited real incident: a new analyst's `SELECT *` brought down a grocery chain's inventory database for three days).
- **Python UDFs in PySpark for performance-critical paths**: forces data out of the JVM into Python, losing Spark Catalyst optimizations — prefer native Spark API or rewrite UDFs in Scala/Java.
- **Blindly trusting GUI/low-code transformation tools**: not understanding the generated SQL/code underneath makes debugging and performance tuning impossible.
- **Ingesting into a data lake "to transform later" with no plan**: Inmon's cited warning — data "put in and transformed later" is often "never touched" six months on; recreates the data-swamp problem.
- **Mismatched architecture for workload**: connecting a real-time pipeline to an RDBMS or Elasticsearch and using it as a data warehouse — these aren't built for high-volume aggregated OLAP and will implode under that load.
- **Cherry-picked vendor streaming benchmarks**: micro-batch vs. true-streaming vendor comparisons are often rigged; validate against your real workload.

## Code Examples
```sql
GRANT SELECT ON data_science_db TO user_name Sarah;
```
```sql
REVOKE SELECT ON data_science_db TO user_name Sarah;
```
- **What it demonstrates**: DCL commands granting/revoking read access — illustrates fine-grained, lifecycle-aware access control (e.g., when an employee leaves).

```sql
SELECT COUNT(*), user_id
FROM user_events
GROUP BY user_id;
```
- **What it demonstrates**: the canonical MapReduce example — each map task counts user_id occurrences in its local data block (embarrassingly parallel), then a shuffle redistributes by key so reduce can sum per-key counts across the cluster.

## Reference Tables

| Modeling approach | Structure | Direction | Best for |
|---|---|---|---|
| Inmon | Highly normalized (3NF) enterprise warehouse + marts | Top-down (enterprise → department) | Single source of truth, strong integration |
| Kimball | Star schema (facts + dimensions) | Bottom-up (department-first) | Fast iteration, query simplicity |
| Data Vault | Hubs/links/satellites, insert-only | Source-structure-preserving | Agility, auditability, fast-changing sources |
| Wide/denormalized | Single wide sparse table | None (no rigor) | Cheap storage, flexible/nested schemas, speed |

| SCD Type | Mechanism | History retained? |
|---|---|---|
| Type 1 | Overwrite in place | No |
| Type 2 | New row + effective start/end dates | Full |
| Type 3 | New column(s) for changed value | Limited (one prior value) |

| Join strategy | Mechanism | Cost |
|---|---|---|
| Broadcast join | Small table copied to all nodes | Low |
| Shuffle hash join | Both tables repartitioned by key | High |

| Update pattern | Mechanism | Trade-off |
|---|---|---|
| Truncate and reload | Wipe + full rerun | Simple, but full reprocessing cost |
| Insert-only | Append new version rows | Preserves history, costly "latest" lookups |
| Hard delete | Permanently remove | Needed for compliance/perf, loses history |
| Soft delete | Flag as deleted | Reversible, needs filter logic |
| Upsert/merge | Match-then-update-or-insert | Powerful but expensive under copy-on-write |

## Worked Example
Normalizing an e-commerce `OrderDetail` table (denormalized, with a nested `OrderItems` JSON array) through Codd's normal forms: 1NF flattens nested items into individual rows (OrderID, Sku, Price, Quantity, ProductName, CustomerID, CustomerName, OrderDate), adding a `LineItemNumber` to form a unique composite key (OrderID, LineItemNumber). 2NF splits off `Orders` (OrderID → CustomerID, CustomerName, OrderDate) from `OrderLineItem` (composite key → Sku, Price, Quantity) since the customer fields depend only on OrderID, a partial key. 3NF further splits `OrderLineItem` from `Skus` (Sku → ProductName) since ProductName is transitively dependent on Sku rather than directly on the composite key. The result: three clean, non-redundant 3NF tables — illustrating exactly how Inmon-style integration would model this data, in contrast to Kimball's approach of keeping an `Orders` fact table referencing `Customer` and `Date` dimensions, or a Data Vault's `HubProduct`/`LinkOrderProduct`/`SatelliteProduct` structure, or simply loading the original nested JSON into one wide denormalized table.

## Key Takeaways
1. Understand DDL/DML/DCL/TCL and how your specific database's query optimizer and commit model behave — these details materially affect performance and correctness.
2. Model data at the lowest practical grain; aggregation is always possible later, de-aggregation is not.
3. Choose a modeling approach (Inmon/Kimball/Data Vault/wide-table/no-model) based on your organization's integration needs, agility needs, and tooling — not dogma; hybrids are common (e.g., Data Vault landing zone feeding a Kimball mart).
4. Transformations differ from queries by persisting reusable, often complex, multi-step results — and rely heavily on orchestration.
5. Copy-on-write makes updates/deletes/merges in columnar and object-storage systems inherently more expensive than inserts; tune partitioning/clustering and merge frequency accordingly.
6. Materialized views, federated queries, and data virtualization blur the line between "a query" and "a stored table," offering flexible trade-offs between freshness, performance, and storage cost.
7. In streaming, windows (session/fixed/sliding) and watermarks are the core mechanisms for bounding otherwise-unbounded computation; micro-batch vs. true-streaming choice should be driven by latency requirements, not fashion.
8. Push recurring, cross-team business logic into a semantic/metrics layer rather than duplicating it across many brittle ETL scripts.

## Connects To
- **Ch 3**: ETL/ELT, Lambda/Kappa architectures, and data warehouse/lake/lakehouse concepts reused directly for transformation patterns.
- **Ch 4**: Benchmark-war skepticism reapplied to streaming framework vendor comparisons.
- **Ch 5**: CRUD, schema, and CDC concepts underpinning update patterns and the fast-follower streaming query pattern.
- **Ch 6**: Columnar storage, partitioning/clustering, and copy-on-write mechanics directly inform update-pattern performance here.
- **Ch 9**: Semantic/metrics layers introduced here are expanded as part of serving data for analytics.
- **External concept**: E. F. Codd's normalization papers, Ralph Kimball's The Data Warehouse Toolkit, Bill Inmon's Building the Data Warehouse, Dan Linstedt's Data Vault 2.0, Google's MapReduce/GFS papers, the Dataflow Model (windowing).
