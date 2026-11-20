# Chapter 2: Types of Data Architectures

## Core Idea
Data architecture is not a one-size-fits-all template; this chapter gives a guided tour of six major architecture types — relational data warehouse, data lake, modern data warehouse, data fabric, data lakehouse, and data mesh — tracing their historical evolution and core trade-offs.

## Frameworks Introduced
- **Schema-on-write vs. schema-on-read**: the foundational distinction between relational and lake-style storage.
  - When to use: schema-on-write when consistency/integrity matter upfront (OLTP, RDW); schema-on-read when flexibility for unstructured/semi-structured data is needed (data lake).
  - How: schema-on-write enforces structure at ingestion time; schema-on-read defers structure definition until query time.
- **Relational Data Warehouse (RDW)**: a relational database optimized for analytical queries, decoupled from operational (OLTP) systems.
  - When to use: structured data, need for transaction support, audit trails, schema enforcement, and fast self-service reporting.
  - How: copy data out of OLTP systems into a dedicated compute+storage engine tuned for analytics.
- **Data Lake**: object storage (schema-on-read) with no bundled compute engine.
  - When to use: staging/preparing raw data of any type (structured, semi-structured, unstructured, binary) cheaply before it's modeled.
  - How: land raw files as-is; attach external compute engines (Spark, Hive, etc.) as needed.
- **Modern Data Warehouse (MDW)**: data lake + relational data warehouse side by side.
  - When to use: need both ML/data-science staging (lake) and governed, high-performance BI (warehouse).
  - How: stage/prepare in the lake, then move modeled data into the RDW for serving/security/compliance.
- **Data Fabric**: an evolution of the MDW with more technology for sourcing, securing, ingesting, transforming, querying, and accessing data.
  - When to use: broader integration/governance needs across more sources than MDW handles.
  - How: add layers for metadata-driven automation across ingestion, security, and access.
- **Data Lakehouse**: a single repository (data lake) made transactional via a storage layer (Delta Lake, Apache Iceberg, Apache Hudi).
  - When to use: want to eliminate the separate RDW while keeping warehouse-like guarantees.
  - How: run a transactional layer on top of lake storage so it behaves more like a relational database.
- **Data Mesh**: decentralized architecture organizing data by business domain rather than centralizing it under IT.
  - When to use: large enterprises facing data ownership, quality, and scaling bottlenecks from centralization.
  - How: each domain owns its own data, compute, storage, and builds/exposes "data products"; it's an organizational/cultural concept, not a product you buy.

## Key Concepts
- **Data architecture**: the overall design/organization of data within an information system — storage, processing, access, security/privacy, governance.
- **Relational database**: structured storage with keys defining relationships, queried via SQL (Codd, 1970).
- **OLTP**: online transaction processing systems performing CRUD operations with fast response times for operational apps.
- **CRUD**: create, read, update, delete — foundational data manipulation operations.
- **Object storage**: unstructured storage medium underlying data lakes (vs. relational row/column storage).
- **HDFS**: Apache Hadoop Distributed File System — the original, largely on-prem, data lake storage technology.
- **Centralization vs. decentralization**: MDW/fabric/lakehouse centralize data under IT; data mesh decentralizes by domain.

## Mental Models
- Think of architecture evolution as additive: each new type (MDW, fabric, lakehouse, mesh) builds on or reacts to the limitations of its predecessor rather than replacing it outright.
- "One technology to do everything" is a historical anti-pattern (data lakes marketed as RDW replacements) — later architectures reintroduce complementary components rather than single-tool solutions.

## Anti-patterns
- **Believing data lakes alone replace relational data warehouses**: early Hadoop-era vendors (Cloudera, Hortonworks, MapR) oversold data lakes as doing everything; querying lakes required advanced skills (Jupyter, Hive, Python) that most end users lacked, and lakes lacked transaction support, schema enforcement, and audit trails. Two of three major vendors (Hortonworks, MapR) went out of business as a result.
- **Treating data mesh as a technology you can buy**: there's no "data mesh in a box" — it requires a major organizational/cultural shift and is appropriate only for large enterprises, a point heavily emphasized and expanded in Ch. 13–14.
- **Applying predefined architecture templates**: templates often fail to account for a system's specific requirements/constraints and aren't flexible enough as needs change, causing data quality, performance, and maintenance issues.

## Reference Tables
| Characteristic | RDW | Data Lake | MDW | Data Fabric | Data Lakehouse | Data Mesh |
|---|---|---|---|---|---|---|
| Year introduced | 1984 | 2010 | 2011 | 2016 | 2020 | 2019 |
| Storage type | Relational | Object | Both | Both (extended) | Object + transactional layer | Decentralized (per domain) |
| Schema | On-write | On-read | Mixed | Mixed | On-write (via layer) + on-read | Domain-determined |
| Compute coupling | Tied to storage | Decoupled | Decoupled | Decoupled | Decoupled | Per-domain |

(Note: "Year introduced" row reproduced from the book's Table 2-1; the book's table includes additional comparison rows not captured verbatim here — see original Table 2-1 for full characteristic list.)

## Worked Example
A company with growing nonrelational and streaming IoT data finds its single RDW can't keep up; it adds a data lake beside the RDW for staging/ML (becoming an MDW), later adds governance/ingestion automation across many more sources (data fabric), or — if it wants one unified repository — adopts Delta Lake/Iceberg/Hudi on its lake to get warehouse-like guarantees without a separate RDW (lakehouse).

## Key Takeaways
1. No architecture type is a drop-in template — match to specific business needs, not a predefined list.
2. Schema-on-write (RDW) vs. schema-on-read (lake) is the key technical fork underlying most architecture choices.
3. Data lakes' early promise to fully replace RDWs failed due to skill-gap and missing enterprise features; lakes succeeded instead as staging/prep layers.
4. MDW, data fabric, and lakehouse are progressively more integrated combinations of lake + warehouse concepts.
5. Data mesh is fundamentally organizational (decentralized domain ownership), not a purchasable technology stack.

## Connects To
- **Ch 4, 5**: Deep dives into RDW and data lake respectively.
- **Ch 10, 11, 12**: Deep dives into MDW, data fabric, and lakehouse.
- **Ch 13, 14**: Full treatment of data mesh, its principles, myths, and adoption readiness.
- **External concept**: Gartner Hype Cycle for data management, which tracked data mesh's rise in hype.
