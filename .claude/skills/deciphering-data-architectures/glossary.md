# Glossary

**ACID** — Atomicity, Consistency, Isolation, Durability; transaction reliability properties, supported by Delta Lake only within a single Delta Table (Ch 12).

**Architecture Design Session (ADS)** — a structured discovery/whiteboarding workshop with stakeholders producing an architecture blueprint and action plan (Ch 3).

**Batch processing** — grouping transactions over a period and processing them together at scheduled intervals, usually off-peak (Ch 9).

**Big data** — all data, regardless of size, inside/outside a company, described by the "six Vs": volume, variety, velocity, veracity, variability, value (Ch 1).

**Change Data Capture (CDC)** — a technique that records INSERTs/UPDATEs/DELETEs to enable near-real-time incremental extraction (Ch 4).

**Common Data Model (CDM)** — a standardized, vendor-neutral schema representing data consistently across multiple incompatible source systems (Ch 8).

**Conformed dimension** — a dimension standardized/consistent across multiple fact tables so cross-fact comparisons are valid (Ch 8, 13).

**CRUD** — Create, Read, Update, Delete; the foundational OLTP operations (Ch 2, 7).

**Data contract** — an agreement defining format, structure, schema, and access rules for exchanging a data product between domains (Ch 13).

**Data Fabric** — an MDW evolution adding ≥3 of: data access policies, metadata catalog, MDM, data virtualization, real-time processing, APIs, services, products (Ch 11).

**Data governance Center of Excellence (CoE)** — a dedicated hub defining governance policy, roles, and standards (Ch 9).

**Data lake** — schema-on-read object storage holding raw data of any type without upfront structuring (Ch 5).

**Data lakehouse** — a data lake with a transactional layer (Delta Lake/Iceberg/Hudi) added, replacing the need for a separate RDW (Ch 12).

**Data Mart** — a department-scoped subset of the data warehouse (Ch 6).

**Data Mesh** — a decentralized architecture per Dehghani's four principles: domain ownership, data as a product, self-serve infrastructure platform, federated computational governance (Ch 13).

**Data product** — a self-contained, independently deployable unit of analytical data delivering business value, owned by a domain team (Ch 13).

**Data quantum** — Dehghani's proposed logical unit encapsulating data, metadata, code, policy, and infrastructure for a data product; the author considers it largely unrealized technology (Ch 14).

**Data Vault** — a modeling methodology (Linstedt, 2000) using Hubs (entities), Links (relationships), Satellites (descriptive/historical attributes) (Ch 8).

**Data virtualization / data federation** — a logical integration layer accessing multiple sources without copying/moving data (Ch 6, 11).

**Delta Lake** — Databricks' transactional storage layer on top of a data lake, adding DML, schema enforcement, time travel, ACID (single-table) (Ch 12, 16).

**Dimensional modeling** — organizing data into facts (measures) and dimensions (attributes), using surrogate keys and denormalization; Kimball's methodology (Ch 8).

**Domain-Driven Design (DDD)** — the process of designing domain-oriented data architecture for a data mesh (Ch 13).

**ELT (Extract, Load, Transform)** — raw data loads first, then is transformed in the destination; preferred for data lakes/big data (Ch 9).

**Enterprise Data Warehouse (EDW)** — a data warehouse scoped to serve the whole organization (Ch 4).

**ETL (Extract, Transform, Load)** — data is transformed in-flight before loading into the destination; preferred for smaller, relational-to-relational pipelines (Ch 9).

**Federated computational governance** — Dehghani's 4th data mesh principle: central rule-setting + domain-level implementation/monitoring (Ch 13).

**Kappa architecture** — a real-time-only processing design: single event stream, stateless processing, no batch layer (Ch 7).

**Lambda architecture** — a dual batch+real-time processing design with batch layer, stream/speed layer, and a unified presentation/serving layer (Ch 7).

**Master Data Management (MDM)** — creating a single "golden" master record per entity (customer, product, etc.) across source systems (Ch 6).

**Metadata catalog** — a data fabric component storing structure/relationship/lineage information about data assets (Ch 11).

**Modern Data Warehouse (MDW)** — an RDW + data lake combined, with data duplicated from lake to warehouse; the defining MDW trait (Ch 10).

**MPP (Massively Parallel Processing)** — a "scale-out" database design with multiple servers each with own memory/disk, splitting queries across compute nodes (Ch 7).

**OLAP (Online Analytical Processing)** — analytical, "write-once, read-many" processing optimized for fast multi-dimensional querying, often via cubes (Ch 7).

**OLTP (Online Transaction Processing)** — real-time, CRUD-based transactional processing optimized for concurrency and low latency (Ch 7).

**Operational Data Store (ODS)** — an enterprise-wide, near-real-time operational (not analytical) data integration layer (Ch 6).

**Reverse ETL** — moving analytically-prepared data from the DW/lake back into operational/SaaS systems to drive day-to-day action (Ch 9).

**RDW (Relational Data Warehouse)** — a centrally structured repository of historical, structured data optimized for analytical queries; the original data warehouse architecture (Ch 4).

**Schema-on-read** — schema is applied at query/read time, not ingestion; used by data lakes (Ch 2, 5).

**Schema-on-write** — schema defined and enforced at ingestion; used by relational databases/warehouses (Ch 2).

**Single Version of Truth (SVOT)** — the principle that all data in a central repository represents one consistent, accurate version of information (Ch 4).

**Slowly Changing Dimension (SCD)** — Type 1 (overwrite), Type 2 (versioned history rows), Type 3 (old+new in same row) — techniques for tracking dimension changes over time (Ch 8).

**SMP (Symmetric Multiprocessing)** — a "scale-up" database design where multiple processors share disk/memory in one server (Ch 7).

**Source-aligned / Aggregated / Consumer-aligned domain data** — the three domain-data types in domain-driven design: close-to-source, cross-domain combined, and consumer-fitted respectively (Ch 13).

**Spreadmart** — an informal, decentralized collection of spreadsheets used for data analysis, symptomatic of Stage 1 (Reactive) data maturity (Ch 1).

**Star schema** — a dimensional model with a central fact table surrounded by dimension tables (Ch 8).

**Top-down approach** — designing the full architecture/enterprise vision before building (RDW methodology; also Inmon's DW methodology) (Ch 4, 8).

**Bottom-up approach** — collecting data before generating hypotheses (data lake methodology; also Kimball's DW methodology) (Ch 5, 8).
