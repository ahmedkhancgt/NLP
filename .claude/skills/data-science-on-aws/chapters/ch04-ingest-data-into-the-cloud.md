# Chapter 4: Ingest Data into the Cloud

## Core Idea
Amazon S3 is the foundation of a modern "Lake House" architecture that unifies a data lake with a data warehouse (Amazon Redshift) via Redshift Spectrum, letting data scientists query raw S3 data with Athena and business intelligence teams combine warehouse + lake data without physically moving it.

## Frameworks Introduced
- **Amazon S3 + AWS Glue Data Catalog**: object storage + Hive-Metastore-compatible metadata store.
  - When to use: foundation for any data lake; "schema-on-read" avoids predefining rigid schemas.
  - How: land raw TSV/CSV in S3, register table-to-S3 mapping via Glue Data Catalog (shared by Athena, Redshift Spectrum, EMR, Spark).
- **Amazon Athena**: serverless, Presto-based SQL query engine over S3.
  - When to use: ad hoc exploratory analytics and operational debugging on S3 data without infrastructure.
  - How: `CREATE EXTERNAL TABLE` pointing at S3 LOCATION with ROW FORMAT DELIMITED; run SQL directly; use `CREATE TABLE AS` (CTAS) to convert TSV → Parquet.
- **AWS Glue Crawler**: scheduled/triggered service that discovers new S3 data and updates the Glue Data Catalog.
  - When to use: continuously ingesting new data arriving in S3 (e.g., nightly batches).
  - How: `glue.create_crawler()` with a cron Schedule and RecrawlPolicy (`CRAWL_EVERYTHING` or `CRAWL_NEW_FOLDERS_ONLY`).
- **Amazon Redshift + Redshift Spectrum (Lake House Architecture)**: petabyte-scale columnar data warehouse that can directly query S3 via an external schema.
  - When to use: business-critical reports/dashboards needing predefined "schema-on-write," complex joins, or subsecond latency; combine with S3 data via Spectrum when older data stays in the lake.
  - How: `CREATE EXTERNAL SCHEMA ... FROM DATA CATALOG DATABASE '<db>' IAM_ROLE '<arn>'` to query Athena-cataloged S3 tables from Redshift; `COPY`/`INSERT INTO ... SELECT` to load a subset into Redshift proper; `UNLOAD ... TO 's3://...' PARQUET PARTITION BY (...)` to export back to S3.
- **S3 Intelligent-Tiering**: automatic storage-class optimization based on access patterns.
  - When to use: large, aging datasets with unpredictable access patterns, to cut storage cost without performance impact.

## Key Concepts
- **Schema-on-read vs. schema-on-write**: Athena/S3 require no predefined schema until query time; Redshift requires schema definition before loading.
- **DISTKEY**: Redshift distribution style column; rows are physically distributed across cluster nodes by this key's values to minimize data movement during joins.
- **SORTKEY**: column(s) Redshift physically sorts data by on disk; choose based on most frequent filter/range columns (e.g., `product_category`).
- **Partition pushdown**: query engines (Athena, Redshift, Spark) push `WHERE` filters down to the storage layer so only relevant partitions/files are scanned.
- **Bloom filters**: probabilistic structures answering "definitely NO or MAYBE" per cluster instance, letting the engine skip instances that can't match — used automatically by Redshift Spectrum on S3.
- **Materialized views (Redshift Spectrum)**: pretransformed/prejoined views refreshed on a schedule for predictable performance on external S3 data.
- **ETL vs. ELT**: Redshift Spectrum enables ELT — load raw data first, transform/clean after loading, rather than transforming before load.

## Mental Models
- **Athena vs. Redshift decision rule**: Athena for ad hoc, infrastructure-free, exploratory queries on S3; Redshift for performance-critical, relational, frequently-joined, low-latency business reporting. Neither replaces write-heavy OLTP databases (RDS/Aurora).
- **Columnar storage = "mechanical sympathy"**: Parquet's columnar layout lets disk controllers do sequential reads for aggregate queries (AVG/SUM/COUNT), avoiding the random-seek cost of row-oriented formats.
- **Time-series table partitioning strategy**: create one table per retention period (e.g., per month); drop whole tables instead of running DELETE+VACUUM; union tables together with `UNION ALL` views for cross-period queries.
- **"Bring compute to data, not data to compute"**: Redshift Spectrum and Athena federated queries let you query data in place (S3, RDS, DynamoDB) instead of physically relocating it.

## Anti-patterns
- **Using row-based formats (CSV/TSV) for large-scale analytics**: forces full-row scans; convert to Parquet for columnar analytical queries.
- **Compressing the first SORTKEY column**: best practice says the first SORTKEY should remain uncompressed even though dictionary encoding looks attractive — compressing it can hurt query performance.
- **Running large-scale DELETE + VACUUM for time-based data retention**: far slower than dropping whole time-partitioned tables.
- **Blindly trusting `ANALYZE COMPRESSION` suggestions**: suggested encodings are not always appropriate (e.g., the chapter overrides the zstd suggestion for `product_category` because it hurts SORTKEY performance in practice).

## Code Examples
```sql
CREATE EXTERNAL TABLE IF NOT EXISTS dsoaws.amazon_reviews_tsv(
         marketplace string, customer_id string, review_id string,
         product_id string, product_parent string, product_title string,
         product_category string, star_rating int, helpful_votes int,
         total_votes int, vine string, verified_purchase string,
         review_headline string, review_body string, review_date string
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n'
LOCATION 's3://data-science-on-aws/amazon-reviews-pds/tsv'
TBLPROPERTIES ('compressionType'='gzip', 'skip.header.line.count'='1')
```
- **What it demonstrates**: registering raw gzip-compressed TSV data in S3 as a queryable Athena table with zero data movement.

```sql
CREATE TABLE IF NOT EXISTS dsoaws.amazon_reviews_parquet
WITH (format = 'PARQUET',
      external_location = 's3://<BUCKET>/amazon-reviews-pds/parquet',
      partitioned_by = ARRAY['product_category']) AS
SELECT marketplace, customer_id, review_id, product_id, product_parent,
       product_title, star_rating, helpful_votes, total_votes, vine,
       verified_purchase, review_headline, review_body,
       CAST(YEAR(DATE(review_date)) AS INTEGER) AS year,
       DATE(review_date) AS review_date, product_category
FROM dsoaws.amazon_reviews_tsv
```
- **What it demonstrates**: CTAS query converting TSV to partitioned, columnar Parquet directly in Athena — no ETL pipeline needed; followed by `MSCK REPAIR TABLE` to register partitions.

```sql
CREATE EXTERNAL SCHEMA IF NOT EXISTS athena FROM DATA CATALOG
    DATABASE 'dsoaws'
    IAM_ROLE '<IAM-ROLE>'
    CREATE EXTERNAL DATABASE IF NOT EXISTS
```
- **What it demonstrates**: Redshift Spectrum querying Athena-cataloged S3 tables directly from Redshift, unifying lake + warehouse.

```sql
UNLOAD ('SELECT ... FROM redshift.amazon_reviews_tsv_2015')
TO 's3://data-science-on-aws/amazon-reviews-pds/parquet-from-redshift/2015'
IAM_ROLE '<IAM_ROLE>'
PARQUET PARALLEL ON
PARTITION BY (product_category)
```
- **What it demonstrates**: exporting Redshift query results back to S3 in partitioned Parquet format for consumption by Athena/EMR/SageMaker/Spark.

## Reference Tables
| Criterion | Amazon Athena | Amazon Redshift |
|---|---|---|
| Infrastructure | None (serverless) | Cluster (compute+storage) required |
| Schema | Schema-on-read | Schema-on-write (predefined) |
| Best for | Ad hoc analytics, operational debugging | Business-critical reports, complex joins, low-latency |
| Data movement | None (queries S3 directly) | Load into cluster (or Spectrum to query S3 in place) |

| Redshift distribution style | Behavior |
|---|---|
| KEY | Rows distributed by one column's values (good for high-cardinality join keys) |
| ALL | Full table copied to every node (good for small dimension tables) |
| EVEN | Round-robin distribution (default) |

| Redshift column encoding | Use case |
|---|---|
| zstd | Generic, works across most data types/sizes |
| az64 | Default for numeric/date columns |
| bytedict | Low-cardinality string dictionary encoding (e.g., dates with ~7,300 unique values) |
| raw | No compression (used for the uncompressed SORTKEY column) |

## Worked Example
Full ingestion pipeline for the Amazon Customer Reviews Dataset: raw TSV files are copied from the public `s3://amazon-reviews-pds/tsv` bucket into a private bucket via `aws s3 cp --recursive`; Athena registers them as `dsoaws.amazon_reviews_tsv` (schema-on-read); a CTAS query converts the data to a partitioned Parquet table `dsoaws.amazon_reviews_parquet` (partitioned by `product_category`), dramatically speeding up analytical queries; an AWS Glue Crawler is scheduled nightly (`cron(59 23 * * ? *)`) to pick up new data; a Redshift external schema exposes the Athena/Glue-cataloged tables via Redshift Spectrum, letting a `UNION ALL` query combine 2014/2015 data loaded into Redshift proper with 1995–2013 data still sitting in S3/Athena — demonstrating the full Lake House pattern end to end.

## Key Takeaways
1. Use Athena for schema-on-read exploratory queries on S3; use Redshift for schema-on-write, performance-critical business reporting.
2. Convert row-based formats (TSV/CSV) to partitioned, columnar Parquet whenever running large-scale analytical queries.
3. Choose SORTKEY/DISTKEY based on actual query patterns (filter columns → SORTKEY; join/high-cardinality columns → DISTKEY) and validate with `EXPLAIN`.
4. Use Redshift Spectrum/UNLOAD to move data between lake and warehouse without manual ETL pipelines.
5. Automate ongoing ingestion with AWS Glue Crawler schedules rather than manual re-registration.
6. Use S3 Intelligent-Tiering for datasets with unpredictable access patterns to reduce storage cost automatically.

## Connects To
- **Ch 2/3**: Builds on Athena/Redshift ML UDF examples introduced earlier.
- **Ch 5**: Uses this same Amazon Customer Reviews Dataset for deeper data exploration and anomaly detection.
- **Ch 6**: Feature transformation builds on the Parquet-formatted data prepared here.
- **Ch 12**: Securing datasets, encryption at rest/in transit, and IAM roles referenced here are covered in depth.
