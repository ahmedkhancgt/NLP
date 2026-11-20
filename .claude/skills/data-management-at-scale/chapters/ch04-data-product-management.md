# Chapter 4: Data Product Management

## Core Idea
A data product should be defined as a technology-agnostic logical entity (not a literal container bundling code+data+metadata+infrastructure as Dehghani originally proposed) so that the same semantic data can have multiple physical representations; building it well requires CQRS-inspired read/write separation, strict design principles, and a well-engineered multi-layer (bronze/silver/gold) data product architecture.

## Frameworks Introduced
- **CQRS (Command Query Responsibility Segregation)**: Separates write-optimized command models from read-optimized query models, kept in sync via events.
  - When to use: As the conceptual template for every data product architecture — treat the data product as the "read store" counterpart of an operational system.
  - How: Keep source/operational writes in their own optimized store; build a separate, denormalized, read-optimized store for consumption; sync via batch/CDC/events.
- **Data Product as Logical Entity** (author's revision of Dehghani's "architectural quantum"): A data product is an autonomous logical entity (name, description, owner, business terms, relationships) that references — but is decoupled from — one or more physical data assets.
  - When to use: Whenever designing your metamodel/catalog for data products, to avoid tightly coupling metadata and physical data.
  - How: Store metadata centrally (catalog); allow many physical representations (e.g., Parquet + Delta) to map to one logical data product.
- **Medallion/Bronze-Silver-Gold layering**: Raw (Bronze) → cleaned/standardized (Silver) → consumption-ready, denormalized (Gold) data layers.
  - When to use: Structuring any lakehouse-based data product pipeline.
  - How: Bronze = 1:1 raw copy (Parquet, partitioned by date); Silver = functional, quality-checked, SCD-processed (Delta); Gold = business-logic-transformed, subject-area-grouped, read-optimized datasets (the actual "data products").

## Key Concepts
- **Data as a product (mindset) vs. data product (architecture)**: The former is the cultural/ownership commitment; the latter is the concrete technical artifact — conflating them causes design confusion.
- **Golden source capture**: Data products must capture data directly from the system of origin, never through intermediate/obfuscating layers.
- **Resource-oriented read-optimized design**: Modeling data products like REST API resources/collections — denormalized, logically grouped by subject area, precomputed joins.
- **Atomicity**: Data attributes must be indivisible, each mapped 1:1 to a business term, so policy rules (e.g., sensitivity labels) can be applied without reshaping physical data.
- **Compatibility/schema drift management**: Data products must detect and version against schema drift so consumers aren't broken by upstream changes.
- **Point-in-time vs. interval (SCD) vs. append-only historization**: Three different strategies for retaining historical data in a data product, each with cost/complexity trade-offs.
- **Output ports**: Standardized, read-only, read-optimized interfaces through which a domain serves data products to consumers (file, relational, columnar, document, graph, event, API).
- **Intermediate domain**: A consuming domain that mediates and abstracts an external/third-party data provider that cannot conform to internal standards.

## Mental Models
- **The flower auction analogy** (Ferd Scheepers, ING): Physical flowers and their metadata (barcodes) are glued together because they're physical objects traveling together; digital data and metadata need not be — and should not be — glued, because metadata (ownership, lineage, classifications) must be managed centrally and consistently across many physical copies.
- **"No raw data" principle**: A data product is definitionally the opposite of raw data — exposing raw data pushes rework onto every consumer and should at most be a flagged, temporary, non-production exception.
- **Don't conform to consumers**: Resist pulling consumer-specific business logic (e.g., accounting's IFRS rules) into a producing domain's data product — it creates reverse dependency and coordination nightmares across potentially hundreds of domains.

## Anti-patterns
- **Gluing code+data+metadata+infrastructure into one container** (literal reading of "architectural quantum"): Makes federated lineage/quality/security queries across all data products a complex federated-query or replication nightmare.
- **Outsourcing business logic to other domains**: E.g., asking every domain to implement IFRS accounting rules internally — assumes false domain knowledge and creates synchronized-change nightmares at scale.
- **One data pipeline per data product (too fine-grained)**: Creates a complex web of repeated processing steps, referential integrity problems between pipelines running at different intervals, and heavy consumer integration burden.
- **Designing data products with data virtualization only**: Fails life-cycle management, increases coupling to source systems, adds network latency/fragility, is limited to RDBMS-style endpoints, and costs more compute for repeated queries than precomputed data.
- **Using Data Vault modeling for data products**: Too complex for "outsiders," slows and raises the cost of the architecture, and requires heavy consumer-side joining/integration.
- **Full domain autonomy on tooling**: 25 teams each choosing their own "modern data stack" independently makes lineage, quality metrics, and interoperability impossible to aggregate — autonomy must start from central standards.

## Code Examples
```text
Data Lake Layering (medallion architecture) for a single domain:
Bronze/  <date-partitioned raw Parquet, copied 1:1 from source>
Silver/  <quality-checked, SCD type-2, Delta format, functional-only columns>
Gold/    <business-logic transformed, denormalized, subject-area folders, Delta>

Pipeline: ADF (orchestration) -> copy activity -> Bronze
          ADF -> Databricks notebook (validation, schema mgmt) -> Bronze
          dbt (SCD2 transform) -> Silver
          dbt templates (joins, business logic) -> Gold
          CI/CD -> publish function -> catalog (Microsoft Purview) -> ACLs for consumers
```
- **What it demonstrates**: A concrete, repeatable Azure-based reference implementation (ADF + ADLS + Databricks + dbt + Purview) for building domain-owned data products end to end, from raw ingestion to governed, discoverable, consumer-ready "Gold" datasets.

## Reference Tables
| Historization Approach | Mechanism | Pros | Cons |
|---|---|---|---|
| Point-in-time (full snapshot) | Insert-only, date-partitioned immutable copies | Simple, supports redelivery/overwrite by partition | Expensive retrospective analysis (must process many files); storage duplication |
| Interval (SCD type 2) | Compare new vs. prior delivery, open/close records with start/end dates | Efficient storage, faster queries, easier GDPR deletion | More ETL/compute to detect+process changes; harder redelivery; different consumer periodicities need extra processing |
| Append-only (+ CDC) | Only new/changed records appended | Works well for transactional/event data | Doesn't by itself give full historical reconstruction without further processing |

| Data Product Design Principle | One-line Rule |
|---|---|
| Resource-oriented design | Group/denormalize by subject area; avoid consumer-side joins |
| Immutability | Data product data is read-only; truth changes only at golden source |
| Capture at source | Never encapsulate another domain's data — preserves clear ownership |
| Clear interoperability standards | Standardize distribution type (batch/API/event), formats, catalog registration |
| No raw data | Always consumption-ready; raw access flagged/temporary only |
| Don't conform to consumers | Serve for maximum reuse, not a single consumer's logic |
| Atomicity | One attribute = one business term, lowest granularity |
| Compatibility | Version and detect schema drift, run parallel pipelines during migration |
| New data = new ownership | Any semantically transformed data creates new ownership |

## Worked Example
Author's real-world Azure design: provisions a data landing zone with ADF, ADLS, and Databricks per domain; ingests via ADF copy activities into Bronze (Parquet, YYYYMMDD partitions); validates/creates schemas via Databricks notebooks; uses dbt to build Silver (Delta, SCD2, functional columns only, one feed per source); uses dbt templates again to build Gold (business-logic joins, denormalized, Delta, one folder per subject area e.g. customer/order/sales data); orchestrates and logs via ADF; publishes data products into a catalog (built on Azure SQL + Azure Functions + Microsoft Purview) via a CI/CD declarative function; grants self-service access via ACLs. Each domain repeats this pattern independently, scaling by adding domains rather than redesigning the pipeline.

## Key Takeaways
1. Separate the logical data product (name, owner, business terms, lineage) from its physical representation(s) — never hard-couple metadata to data containers.
2. Design every data product as a CQRS-style read replica: denormalized, read-optimized, decoupled from the operational write path.
3. Enforce "no raw data," "capture at source," and "don't conform to consumers" as non-negotiable guardrails against scope creep and ownership obfuscation.
4. Choose historization strategy (point-in-time, SCD/interval, append-only) per dataset type (reference, master, transactional) rather than one-size-fits-all.
5. Standardize one pipeline per application/domain state rather than one pipeline per data product to avoid redundant processing and referential-integrity drift.
6. Start small: get one or two domains building data products well before scaling the platform to support every ingestion/consumption variation.
7. Full domain-by-domain tooling autonomy breaks interoperability — standardize identity, governance, and interoperability centrally even in a federated model.

## Connects To
- **Ch 1**: Resolves the "data intensiveness"/read-write ratio problem raised there via CQRS.
- **Ch 2**: Builds directly on domain ownership and golden-source principles.
- **Ch 3**: Data products are hosted inside the data landing zones described there.
- **Ch 5/Ch 6**: APIs and events are the other two distribution patterns (alongside batch data products) covered next.
- **Ch 8/Ch 9**: Data security patterns (masking, classification) and the metamodel for data product metadata are elaborated later.
- **Ch 10/Ch 11**: Historization and enterprise consistency themes return in MDM and value-creation chapters.
- **Data Mesh (Dehghani)**: Directly critiques and revises the "architectural quantum" concept.
