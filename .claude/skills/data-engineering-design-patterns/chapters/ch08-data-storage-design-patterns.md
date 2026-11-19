# Chapter 8: Data Storage Design Patterns

## Core Idea
Beyond adding compute, query latency and cost can be reduced preemptively through deliberate data organization — partitioning (horizontal/vertical), colocation (bucketing/sorting), read-optimization (metadata, materialization, manifests), and representation choice (normalized vs. denormalized) — each trading some form of mutability, storage, or consistency for speed.

## Frameworks Introduced
- **Horizontal Partitioner**: physically isolate whole rows by a low-cardinality distribution key.
  - When to use: incremental/time-bounded processing (rolling windows) or Fast Metadata Cleaner-style idempotency needs.
  - How: choose a low-cardinality partitioning attribute (time, business key, or nested combinations); declare partitions at table-creation time (BigQuery/Databricks `PARTITIONED BY`) or at write time (Spark `partitionBy`, custom Kafka partitioner); avoid high-cardinality keys (too many small partitions → metadata/small-files overhead) — use Bucket instead.
- **Vertical Partitioner** (storage specialization, distinct from the Ch 7 security variant): split a row's mutable vs. immutable attribute groups into separate storage locations joined by a shared key.
  - When to use: avoiding duplicate storage of immutable attributes repeated across many rows (distinct motivation from Ch 7's PII-removal use case, though mechanically similar).
  - How: classify attributes into groups, pick a merge key, write each group to its own table/location; gains storage savings and group-level retention/access policies at the cost of query-time joins and producer-side multi-write complexity.
- **Bucket**: colocate rows sharing high-cardinality key values into a fixed number of hash-based buckets.
  - When to use: a high-cardinality column (e.g., user ID) is heavily used in filters/joins but is unsuitable for horizontal partitioning.
  - How: choose bucket column(s) and bucket count; the engine computes `hash(key) % buckets`; enables bucket pruning (skip buckets that can't match a filter) and shuffle-free joins when both sides share identical bucketing (Spark `bucketBy`, Athena `CLUSTERED BY ... INTO N BUCKETS`, originally popularized by Hive).
- **Sorter**: physically order stored rows by frequently filtered/sorted columns to enable data-block skipping.
  - When to use: queries consistently filter or sort on known columns, and reorganizing storage order won't conflict with the chosen idempotency/partitioning strategy.
  - How: declare sort/cluster columns at table creation (BigQuery `CLUSTER BY`, Redshift sort keys); for multi-dimensional access patterns use curved sorts like Z-order (Delta Lake/Iceberg `OPTIMIZE ... ZORDER BY`, Redshift interleaved sort keys) instead of lexicographical composite sorting, which only helps when queries reference leading sort columns in order.
- **Metadata Enhancer**: persist per-file/per-table statistics (min/max, null counts, row counts) to skip irrelevant data without reading it.
  - When to use: queries filter on columns where most files/blocks can be excluded by range.
  - How: rely on columnar format footers (Parquet min/max/null-count per column) automatically inherited by Delta Lake/Iceberg/Hudi commit logs, or database-maintained table statistics (refreshed via `ANALYZE TABLE`-style commands); watch for stale statistics on databases with infrequent auto-refresh thresholds.
- **Dataset Materializer**: precompute and persist the result of a costly/repeated query.
  - When to use: the same expensive query (joins, aggregations) is run repeatedly by many consumers.
  - How: materialized view with automatic refresh (Redshift `AUTO REFRESH`, BigQuery, Databricks, Snowflake) or a manually-managed table (more flexible — supports partitioning/bucketing/sorting, but refresh is your responsibility); prefer incremental refresh (MERGE on new rows only) over full recompute when the workload is insert-only.
- **Manifest**: record the exact list of files belonging to a dataset/version to avoid repeated expensive storage listing.
  - When to use: object-store listing operations are the dominant cost/latency source, especially with many small files.
  - How: rely on table file formats' native commit logs as manifests (Delta Lake, Iceberg, Hudi) or generate them explicitly (Delta Lake `generate('symlink_format_manifest')` for BigQuery external tables, Redshift `COPY ... MANIFEST` for idempotent, auditable loads); watch for manifest-size growth with continuous streaming writers.
- **Normalizer**: split data into entity/relationship tables (normal forms, snowflake schema) to eliminate duplication and keep updates consistent.
  - When to use: data consistency and cheap, localized updates matter more than query simplicity (transactional workloads).
  - How: apply first/second/third normal form (atomic non-repeating values + full primary-key dependency + no transitive dependency) for OLTP, or build a snowflake dimensional model (fact table + dimension tables that can reference further sub-dimensions) for OLAP; mitigate join cost via colocation/broadcast joins (Spark `autoBroadcastJoinThreshold`) and handle time-sensitive dimensions with SCD (Ch 5).
- **Denormalizer**: flatten joined data into fewer, wider tables (One Big Table, star schema) to eliminate joins at query time.
  - When to use: read latency/cost from repeated joins outweighs the cost of consistency and storage duplication.
  - How: copy joined columns flat (One Big Table) or nest them as structs; or use a star schema (fact table + single-level, non-nested dimension tables); mitigate update cost by treating denormalized tables as point-in-time snapshots where possible, and storage cost via dictionary encoding; avoid letting One Big Table become a "trash bag" of unrelated attributes — if naming the table requires "and"/"with", it's doing too much.

## Key Concepts
- **Distribution key**: the attribute used to physically divide a dataset (horizontal partitioning).
- **Low vs. high cardinality**: few vs. many distinct values for an attribute — determines Horizontal Partitioner (low) vs. Bucket (high) suitability.
- **Sharding**: a special, physical-machine-level case of horizontal partitioning that actually moves data across hardware, distinct from logical horizontal partitioning.
- **Bucket pruning / shuffle elimination**: query optimizations enabled by matching bucket configuration — skipping irrelevant buckets, or joining co-bucketed tables without network shuffle.
- **Curved sort (Z-order)**: a non-lexicographical, multi-dimensional sort that colocates rows across several columns more efficiently than top-to-bottom ordering.
- **Normal forms (1NF/2NF/3NF)**: progressively stricter rules eliminating repeating groups, partial key dependency, and transitive dependency.
- **Snowflake vs. star schema**: snowflake allows dimensions to reference further sub-dimensions (more normalized); star schema flattens all dimension attributes to one level (fully denormalized dimensions).
- **One Big Table (OBT)**: a fully denormalized table with all related attributes flattened as columns or nested structs.

## Mental Models
- Choose Horizontal Partitioner for low-cardinality, range-queried attributes; switch to Bucket once cardinality makes partition count explode.
- Sorter and Metadata Enhancer work together: sorting makes block-level min/max statistics tight and useful, which is what actually lets the engine skip reading blocks.
- Dataset Materializer and Manifest both answer "can I avoid redoing expensive work on every read?" — materialization avoids recomputation, manifests avoid relisting.
- Normalizer vs. Denormalizer is the classic consistency-vs-speed dial, and they're not mutually exclusive — build a normalized model first (source of truth) and materialize a denormalized version on top of it via a Ch 6 sequence pattern.

## Anti-patterns
- **Partitioning on a high-cardinality column (e.g., per-user ID)**: produces too many tiny partitions, overwhelming metadata and causing the small-files problem — use Bucket instead.
- **Changing a partition key casually**: requires moving/rewriting all existing data (or, for formats like Iceberg, leaves old data under the old scheme, splitting the dataset's physical organization).
- **Querying only on a non-leading composite sort key**: the engine can't skip blocks and must scan nearly everything — order sort keys by actual query predicate usage.
- **Letting materialized view/table statistics or refreshes go stale**: query plans degrade or results go out of date; schedule refreshes deliberately.
- **Relying on repeated object-store listing in hot paths**: unnecessarily slow and costly — use the Manifest pattern (native commit logs or explicit manifest files).
- **Building a fully normalized model for a read-heavy analytical workload with 80% of queries hitting 8 joined tables**: query cost explodes — denormalize (star schema or OBT) for the read path instead.
- **Building a "One Big Table" with unrelated domains crammed together**: becomes an unmaintainable, undocumentable dataset — apply the "can I name this table without 'and'/'with'?" test.

## Code Examples
```python
# Horizontal Partitioner: nested time partitioning with Spark
partitioned_users = (input_users
 .withColumn('year', functions.year('change_date'))
 .withColumn('month', functions.month('change_date'))
 .withColumn('day', functions.day('change_date'))
 .withColumn('hour', functions.hour('change_date')))
(partitioned_users.write.mode('overwrite').format('delta')
 .partitionBy('year', 'month', 'day', 'hour').save(output_dir))
```
- **What it demonstrates**: creating a nested year/month/day/hour partition layout for flexible, low-cardinality time-based access.

```python
# Bucket: bucketing a high-cardinality column to enable pruning and shuffle-free joins
input_dataset.write.bucketBy(50, 'user_id').saveAsTable(table_name)
```
- **What it demonstrates**: colocating rows by hashed `user_id` into 50 buckets instead of creating one partition per user.

```sql
-- Sorter: Z-order curved sort for multi-column access patterns
-- via DeltaTable.forPath(spark, output_dir).optimize().executeZOrderBy(['visit_id', 'page'])
CREATE TABLE `dedp.visits.raw_visits`
PARTITION BY DATE(event_time)
CLUSTER BY visit_id, page
```
- **What it demonstrates**: combining partitioning with clustering/Z-order so queries filtering on either `visit_id` or `page` alone still benefit from data skipping.

```sql
-- Dataset Materializer: incremental refresh combining Incremental Loader + Merger
MERGE INTO dedp.visits_counter AS target
USING (
 SELECT user_id, COUNT(*) AS visits FROM dedp.visits
 WHERE insertion_time > '2024-11-09T03:27:32' GROUP BY user_id
) AS input
ON target.user_id = input.user_id
WHEN MATCHED THEN UPDATE SET count = count + input.visits
WHEN NOT MATCHED THEN INSERT (user_id, count) VALUES (input.user_id, input.visits)
```
- **What it demonstrates**: refreshing a materialized aggregate incrementally instead of recomputing the full dataset each time.

## Reference Tables
| Pattern | Cardinality fit | Mutability cost |
|---|---|---|
| Horizontal Partitioner | Low | High (requires data movement) |
| Vertical Partitioner | N/A (column-based) | Medium (producer multi-write) |
| Bucket | High | High (backfill required to change) |
| Sorter | N/A (order-based) | Medium–high (resort cost) |

| Representation | Consistency | Query cost | Storage cost |
|---|---|---|---|
| Normalizer (NF / snowflake) | High | High (many joins) | Low |
| Denormalizer (OBT / star) | Lower (duplicated updates) | Low (few/no joins) | High |

## Worked Example
Blog analytics platform: the `visits` dataset is horizontally partitioned by `year/month/day/hour` (low-cardinality time key) to support the rolling 4-day aggregate job cheaply, while a separate high-cardinality `user_id` column used in 80% of filters is handled with a 50-bucket `bucketBy('user_id')` table instead of being partitioned (which would create one partition per user). Because most queries filter on `event_time` and sometimes `page` independently, the team applies Delta Lake Z-order (`executeZOrderBy(['visit_id', 'page'])`) rather than plain lexicographical sorting, so filtering on `page` alone still skips most data blocks. Parquet's native per-file min/max statistics (inherited automatically by Delta Lake's commit log) let the query engine skip irrelevant files without any extra configuration (Metadata Enhancer). A frequently-run, expensive "last 3 weeks of visits" query is turned into a materialized view with `MERGE`-based incremental refresh (Dataset Materializer) rather than being recomputed from scratch on every read, and a Delta Lake `generate('symlink_format_manifest')` manifest lets BigQuery read the same dataset as an external table without relisting S3/GCS objects (Manifest). For the data model itself, the team first builds a normalized snowflake schema (separate `dim_page`, `dim_page_category`, `dim_date`, `dim_date_month`, `dim_date_quarter` tables feeding a `fact_visit` table) to keep updates consistent, then builds a denormalized One Big Table on top of it via a Ch 6 sequence pattern, giving analysts both a consistent source of truth and a fast, join-free read path.

## Key Takeaways
1. Match partitioning strategy to cardinality: Horizontal Partitioner for low-cardinality attributes, Bucket for high-cardinality ones used heavily in filters/joins.
2. Sorting (especially curved sorts like Z-order) and metadata statistics work together to enable data-block skipping — order composite sort keys by actual query predicate usage.
3. Materialize expensive, repeated queries, and prefer incremental refresh over full recompute whenever the workload is insert-only.
4. Use manifests (native or explicit) to eliminate repeated, costly object-store listing operations, especially for many-small-files datasets.
5. Normalize for consistency and cheap updates (transactional/OLTP); denormalize for query speed (analytical/OLAP) — and consider building both, keeping them in sync with a Ch 6 sequence pattern.
6. Guard against One Big Table becoming an unmaintainable grab-bag — group only attributes that genuinely belong to the same domain.

## Connects To
- **Ch 4 (Idempotency)**: Horizontal Partitioner underlies the Fast Metadata Cleaner pattern; Manifest-based loading (Redshift COPY) supports idempotent pipelines.
- **Ch 5 (Data Value)**: Local Aggregator's bucketing reuse; Normalizer's time-sensitive dimensions reuse SCD from the Static Joiner pattern.
- **Ch 2 (Data Ingestion)**: the small-files problem and Compactor pattern directly motivate avoiding high-cardinality horizontal partitioning.
- **Ch 6 (Data Flow)**: combining Normalizer and Denormalizer outputs is itself a sequencing/orchestration problem.
- **Ch 7 (Data Security)**: Vertical Partitioner here is the storage-optimization sibling of Ch 7's PII-focused Vertical Partitioner.
- **Apache Spark, Delta Lake, Apache Iceberg, Apache Hudi, Apache Parquet, Apache Kafka, PostgreSQL, Amazon Athena, Amazon Redshift, GCP BigQuery, Databricks, Snowflake**: primary technologies implementing these patterns.
