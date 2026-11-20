# Chapter 8: Approaches to Data Modeling

## Core Idea
Data modeling is the blueprint-building step between raw data and usable reporting — relational modeling (normalized, 3NF) and dimensional modeling (denormalized, facts/dimensions) serve different needs, and the Kimball vs. Inmon debate is less "either/or" than commonly believed: both are compatible, hybridizable methodologies with more myths than real conflicts.

## Frameworks Introduced
- **Relational Modeling (Codd, 1970)**: tables + primary/foreign/natural keys, normalized via ER diagrams.
  - When to use: OLTP systems, or wherever data integrity and minimal redundancy matter most.
  - How: build an ER diagram, then normalize progressively — 1NF (single-valued attributes, no repeating groups, has a primary key) → 2NF (non-key attributes depend on the whole primary key) → 3NF (non-key attributes depend only on the primary key, not transitively on other non-key attributes). Most OLTP models target 3NF.
- **Dimensional Modeling (Kimball, 1996)**: facts (measures) + dimensions (descriptive attributes/hierarchies), using surrogate keys and denormalization.
  - When to use: when querying/reporting off a normalized relational model becomes too slow or complex.
  - How: identify fact tables (numeric measures, e.g. sales) and dimension tables (attributes, e.g. product/customer/time); use surrogate keys instead of natural keys; denormalize (e.g., repeat CategoryName across product records) to reduce joins; arrange as a star schema (central fact table, surrounding dimension tables) or snowflake schema.
- **Common Data Model (CDM)**: a standardized, vendor-neutral structure for representing data across many source systems.
  - When to use: ingesting from multiple vendor systems (e.g., several CRM products) with incompatible native schemas.
  - How: design (or customize a premade industry CDM — banking, healthcare, retail) a model general enough to support all current and future source formats, rather than picking one vendor's native schema.
- **Data Vault (Linstedt, 2000)**: Hubs (business entities) + Links (relationships) + Satellites (descriptive/historical attributes).
  - When to use: large, complex organizations with high data-relationship complexity needing flexible, auditable historical tracking; sits between 3NF and star schema in rigidity/flexibility.
  - How: model core entities as Hubs with unique IDs, relationships as Link tables of foreign keys, and all descriptive/changing attributes as Satellites linked to Hubs/Links.
- **Inmon's Top-Down Methodology (CIF)**: centralize first in 3NF, then derive dependent data marts.
  - When to use: large/complex organizations valuing governance, data quality, regulatory compliance, enterprise-wide consistency.
  - How: OLTP → staging tables → Corporate Information Factory (CIF, atomic-level 3NF EDW, off-limits to end users) → dependent data marts (dimensional, department-specific) → cubes. "Top-down" refers to starting with enterprise-wide business needs before dimensional execution.
- **Kimball's Bottom-Up ("Agile") Methodology**: build independent, dimensional data marts first, integrate via conformed dimensions.
  - When to use: faster time-to-value per business process, end-user-driven, agile delivery.
  - How: OLTP → staging tables → independent dimensional data marts (per business process) → integrated via a DW bus / conformed dimensions (a "conformed" not physical EDW) → optional cubes. Uses an EDW bus matrix for top-down strategic planning even though execution is bottom-up.
- **Hybrid Model**: combines Inmon's CIF with Kimball's dimensional marts and a mirrored OLTP.
  - When to use: most real-world situations — the book explicitly recommends mixing approaches rather than choosing one exclusively.
  - How: staging → mirrored OLTP (parallel duplicate, freeing the real OLTP quickly) → CIF (3NF) → independent dimensional data marts → cubes (preferred reporting layer for semantic layer, concurrency, aggregation, security, time calculations); use database views throughout to simplify ETL and expose a star schema even over complex underlying structure.

## Key Concepts
- **Primary/foreign/natural/surrogate keys**: primary key uniquely identifies a row; foreign key references another table's primary key; natural key is an existing unique field (e.g., SSN) often used as primary key in relational models; surrogate key is an artificial, usually auto-generated key used in dimensional models when natural keys are unsuitable (longer, sensitive, inconsistent across merged systems).
- **Slowly Changing Dimensions (SCD)**: Type 1 overwrites (no history), Type 2 keeps multiple versioned rows (full history, used for "as of" accuracy), Type 3 keeps old+new values as separate columns within one row.
- **History tables (relational) vs. SCDs (dimensional)**: history tables track changes at the individual-record level for any table; SCDs track changes at the dimension level specifically.
- **Denormalization**: deliberately duplicating data across tables to reduce joins and simplify queries, at the cost of storage and sync complexity — the core mechanic distinguishing dimensional from relational models.
- **Conformed dimensions**: dimensions standardized/consistent across multiple fact tables so cross-fact comparisons are valid (Kimball's key integration mechanism).
- **Dependent vs. independent data marts**: Inmon's marts depend solely on the central CIF; Kimball's marts are independently built per business process and integrated afterward via conformed dimensions.
- **Dimensionalized view**: a SQL view that organizes staging data into a dimensional model, allowing a cube to be built directly from staging without a separate physical data mart.
- **Mirrored OLTP**: a parallel duplicate of the source OLTP system used briefly for mirroring, freeing the real OLTP quickly and allowing custom keys/indexes/subset-column loading for cheaper, faster ETL.

## Mental Models
- "Building a city with a blueprint, iteratively" — both Inmon and Kimball reserve upfront design time, then build/design/build in cycles; the 25%-from-ad-hoc point on a design-upfront sliding scale (ad hoc ↔ waterfall/big-bang) describes both, debunking the myth that Inmon = waterfall.
- Relational model = "how the business works" (operational logic); dimensional model = "how well the business is doing" (performance measurement) — use this framing to decide which to build first.
- Data vault sits on a rigidity spectrum between 3NF (most rigid/normalized) and star schema (most denormalized/query-friendly).

## Anti-patterns
- **Treating Kimball as purely bottom-up/no-enterprise-focus**: Kimball's EDW bus matrix is explicitly top-down strategic planning; only execution is bottom-up/agile.
- **Treating Inmon as requiring a "big bang"/waterfall build**: Inmon explicitly warned against big-bang approaches and favored iterative building; this is one of the most persistent methodology myths.
- **Believing Inmon forbids star schemas**: Inmon endorses star schema data marts for direct end-user access; the two methodologies are not philosophically opposed on this point.
- **Treating Kimball and Inmon as mutually exclusive**: they can be — and often are — hybridized (physical EDW + dimensional marts, or conformed EDW + optional normalized layer).
- **Using natural keys across merged/acquired systems without reconciliation**: duplicate or inconsistently formatted natural keys (e.g., two companies' overlapping customer IDs) create integration conflicts — a reason dimensional modeling prefers surrogate keys.
- **Skipping a mirrored OLTP in heavy ETL environments**: pulling directly and repeatedly from the live OLTP (instead of a mirror) ties up production resources and prevents custom indexing for ETL performance.

## Reference Tables
| Model | Structure | Keys | Optimized For | Weakness |
|---|---|---|---|---|
| Relational (3NF) | Normalized tables | Natural/primary/foreign | Integrity, write consistency | Slower queries (many joins) |
| Dimensional (star/snowflake) | Facts + dimensions, denormalized | Surrogate | Fast analytic queries, ease of use | Redundancy, sync maintenance |
| Common Data Model | Standardized cross-vendor schema | Varies | Interoperability across source systems | Significant upfront modeling effort (mitigated by premade industry CDMs) |
| Data Vault | Hubs + Links + Satellites | Business keys + surrogate | Auditability, lineage, flexibility, large complex orgs | Complexity, duplication, performance, lack of standardization |

| Methodology | Direction | EDW Type | Data Marts | End-User Role |
|---|---|---|---|---|
| Inmon | Top-down | Physical (CIF, 3NF) | Dependent | Passive |
| Kimball | Bottom-up/agile | Conformed (not physical) | Independent | Active |
| Hybrid | Both | Physical CIF + dimensional marts + mirrored OLTP | Independent, mixed atomic/summary | Mixed |

| SCD Type | Behavior | Use Case |
|---|---|---|
| Type 1 | Overwrite, no history | Minor corrections (e.g., phone number fix) |
| Type 2 | New row per change, full history | Accurate "as of" historical reporting (e.g., state changes) |
| Type 3 | Old+new values in same row | Complete before/after snapshot per change |

## Worked Example
A company analyzed sales data six months ago and found New York to be its top-selling state; customers have since moved to New Jersey. Without a Type 2 SCD, rerunning the same historical report today would misattribute those customers' past New York sales to New Jersey, falsely showing declining NY sales and misleading strategic decisions — Type 2 SCD preserves the old state value alongside the new one to keep historical reports accurate.

A second worked example: a company ingesting data from multiple CRM vendors, each with different table/field formats, builds a Common Data Model (rather than adopting any single vendor's schema) so all current and future CRM data — regardless of source — can be represented consistently; many cloud providers offer premade industry CDMs (banking, healthcare, retail) to shortcut this work.

## Key Takeaways
1. Normalize (3NF) for OLTP/integrity; denormalize (star/dimensional) for fast analytical queries — pick based on query pattern, not dogma.
2. Use surrogate keys in dimensional models to avoid natural-key problems (length, sensitivity, cross-system duplication/format conflicts).
3. Choose SCD Type 1/2/3 based on whether you need to overwrite, fully version, or snapshot before/after values — Type 2 is essential for accurate historical "as of" reporting.
4. Use a Common Data Model instead of adopting any single source vendor's schema when integrating multiple similar-but-incompatible systems.
5. Data Vault trades complexity and duplication for auditability/flexibility — best for large, complex organizations, not simple ones.
6. Inmon and Kimball are not opposed: both reserve upfront design time, both can incorporate star schemas and data lakes, and hybrid models (CIF + independent dimensional marts + mirrored OLTP) are common and valid.
7. Debunk the specific methodology myths before letting them drive architecture decisions (Kimball ≠ purely bottom-up; Inmon ≠ waterfall/big-bang; Inmon does allow star schemas; the methodologies are not incompatible).

## Connects To
- **Ch 4**: The top-down approach concept reappears here applied specifically to Inmon's methodology (note: the book explicitly warns these are different "top-down" usages).
- **Ch 5**: Data lakes can replace relational staging tables in both Kimball and Inmon architectures.
- **Ch 6**: MDM-mastered records become dimension tables; data marts (Ch 6) are the output of both methodologies.
- **Ch 7**: OLAP cubes (Ch 7) are the typical consumption layer recommended in the hybrid model.
- **Ch 15**: Team skills (leadership, communication) are noted as prerequisites for any methodology to succeed.
- **External concepts**: Bill Inmon's *Building the Data Warehouse* (1992); Ralph Kimball's *The Data Warehouse Toolkit* (1996); Barry Devlin & Paul Murphy's 1988 IBM Systems Journal architecture paper.
