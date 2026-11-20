# Chapter 2: Assembling the Building Blocks of a Reliable Data System

## Core Idea
Reliable data quality starts with understanding your core infrastructure — the operational-vs-analytical data divide, warehouses vs. lakes (vs. lakehouses), and data catalogs/discovery — because each building block has distinct failure modes and metadata sources you must learn to query.

## Frameworks Introduced
- **Operational vs. Analytical Data distinction**: operational data runs the business (inventory snapshots, transactions); analytical data manages the business (churn, clickthrough rates). Maps to the OLTP vs. OLAP distinction.
  - When to use: deciding where/how to apply data quality tooling — this book focuses on analytical data.
  - How: trace whether data is used for immediate business operations (low-latency, optimized) or for aggregation/analysis (high-throughput, optimized).
- **Throughput vs. Latency trade-off**: for any system with fixed computational power, you cannot simultaneously maximize throughput (volume processed per unit time) and minimize latency (delay before processing).
  - When to use: explaining why OLTP databases (low latency) and OLAP warehouses (high throughput) are architected differently.
  - How: identify whether your use case needs fast single-record access (optimize latency) or large aggregations (optimize throughput); don't query Snowflake/Redshift from a customer UI or run trillion-row aggregations in Postgres/MySQL.
- **Data Discovery** (data-mesh-inspired, successor to static data catalogs): domain owners are accountable for their data as products; discovery surfaces the *current, real-time* state of data rather than an idealized cataloged state.
  - When to use: when a company has scaled past manual spreadsheet cataloging and has distributed/domain-owned data.
  - How: automate lineage, metadata capture, and ownership assignment rather than relying on manual documentation.

## Key Concepts
- **Schema-on-write**: data warehouses enforce structure at the moment data enters (less flexible, better hygiene).
- **Schema-on-read**: data lakes infer structure only when data is read/used (more flexible, more risk of "blind ETL").
- **Swampification**: the tendency of data lakes to accumulate technical debt and tacit knowledge over time until no one can navigate them ("data swamp").
- **Data lakehouse**: convergence technology (e.g., Delta Lake, Redshift Spectrum) adding warehouse-style schema/SQL/ACID properties to lakes, and lake-style flexibility to warehouses.
- **Data catalog**: an inventory of metadata (location, ownership, lineage) that helps stakeholders find and trust data; legacy catalogs rely on manual spreadsheets.
- **Blind ETL**: transforming lake data while assuming an unenforced schema — dangerous because transformations can silently fail on upstream changes.
- **Query logs**: system tables (Snowflake QUERY_HISTORY, BigQuery AuditLogs, Redshift STL_QUERY) recording who queried what, when, and how — a key metadata source for data quality and lineage.

## Mental Models
- Think of operational data as "running" the business and analytical data as "managing" it — this clarifies which SLAs/latency requirements apply where.
- The cafe analogy for throughput vs. latency: fixed staff (compute) forces a trade-off between fast individual service (latency) and total customers served (throughput).
- Data catalogs answer "where/what/why" questions about data at rest; data discovery answers them dynamically, in real time, as data evolves.

## Anti-patterns
- **Manual, spreadsheet-based data cataloging**: doesn't scale to warehouses with tens of thousands of tables or to unstructured/fluid lake data; becomes stale immediately.
- **Blind ETL on lake data**: transforming data in a lake while assuming a schema that is not actually enforced — transformations fail unpredictably on upstream changes.
- **Letting a data lake run ungoverned**: leads to "swampification," where institutional knowledge about data ownership and meaning is lost.
- **Ignoring query log retention limits**: most warehouses retain only a limited window of query history, so teams must proactively export/store these metrics or lose historical visibility.

## Code Examples
```sql
-- Map warehouse inventory (Snowflake)
SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, TABLE_OWNER, TABLE_TYPE,
       IS_TRANSIENT, RETENTION_TIME, AUTO_CLUSTERING_ON, COMMENT
FROM "ANALYTICS".information_schema.tables
WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
  AND TABLE_TYPE NOT IN ('VIEW', 'EXTERNAL TABLE')
ORDER BY TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME;

-- Health check: completeness, distinctness, format-validity signals on a string field
SELECT
  COUNT(account_id) / CAST(COUNT(*) AS NUMERIC) AS account_id___completeness,
  COUNT(DISTINCT account_id) / CAST(COUNT(*) AS NUMERIC) AS account_id___approx_distinctness,
  AVG(LENGTH(account_id)) AS account_id___mean_length
FROM analytics.prod.client_hub
WHERE DATE_TRUNC('HOUR', measurement_timestamp) >= DATEADD(day, -1, CURRENT_TIMESTAMP());
```
- **What it demonstrates**: how to pull table inventory/schema metadata and compute per-column health metrics (completeness, distinctness, zero-rate, quantiles) directly from warehouse system tables as the foundation for monitoring.

## Reference Tables
| Dimension | Data Warehouse | Data Lake |
|---|---|---|
| Schema | On-write (enforced at ingest) | On-read (inferred at use) |
| Data shape | Structured, tabular | Structured + semi + unstructured |
| Query language | SQL only | SQL + Python/Spark/Hadoop |
| Flexibility | Low (rigid) | High (DIY) |
| Key risk | Limited flexibility, frictional workflows | Integrity risk, swampification, more endpoints = more failure points |
| Examples | Redshift, BigQuery, Snowflake | S3 + Hadoop/Spark, Databricks |

| Basic Data Catalog Fields | Example |
|---|---|
| Table name | LIOR_GOOD_TABLE_3.csv |
| Dashboard/report | Exec Forecasting V3 (Looker) |
| Last updated | March 3, 2022 |
| Owner | Lior Gavish (lior@internet.org) |
| Notes | Used for executive financial forecasts (ARR) |

## Worked Example
A four-step tutorial for pulling data quality metrics from Snowflake: (1) map inventory via `information_schema.tables` and `SHOW VIEWS/EXTERNAL TABLES`; (2) monitor freshness/volume via `ROW_COUNT`, `BYTES`, `LAST_ALTERED` on `information_schema.tables`; (3) build query history from `snowflake.account_usage.query_history` (filtering out system/failed queries) to understand lineage and usage; (4) run a "health check" query computing completeness, distinctness, UUID-format rate, and null-keyword rate for a string field (`account_id`), plus zero-rate, mean, and quantiles for a numeric field (`num_of_users`), bucketed hourly, to surface anomalies like accounts with no IDs or user counts silently zeroing out.

## Key Takeaways
1. Separate your data into operational vs. analytical categories early — their reliability requirements (latency vs. throughput) and tooling differ fundamentally.
2. Choose warehouse vs. lake vs. lakehouse based on how much schema rigidity your team needs, not just cost or trend — each has distinct data quality failure modes.
3. Use query logs (not just table snapshots) to answer "who/when/how often" questions essential for both monitoring and lineage.
4. Manual/spreadsheet data catalogs do not scale — invest in automated metadata extraction (SQL parsers, discovery tools) as data volume grows.
5. A data-quality-first catalog needs self-service discovery, automatic scaling with data growth, and built-in lineage — not just a static inventory.

## Connects To
- **Ch 3**: Picks up where this chapter's warehouse/lake metadata ends, applying it to collection, cleaning, transformation, and testing.
- **Ch 7**: Expands on lineage, introduced here as a feature of data discovery.
- **OLTP/OLAP (Designing Data-Intensive Applications)**: external standard the operational/analytical distinction directly parallels.
