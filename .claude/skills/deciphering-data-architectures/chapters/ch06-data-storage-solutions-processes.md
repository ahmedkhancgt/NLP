# Chapter 6: Data Storage Solutions and Processes

## Core Idea
Beyond the RDW and data lake, organizations rely on a set of complementary data stores (data marts, operational data stores, data hubs) and processes (MDM, data virtualization, data catalogs, data marketplaces) that each solve a distinct problem — department-level access, real-time operational views, cross-system distribution, record consistency, speed-to-market, discoverability, and monetization.

## Frameworks Introduced
- **Data Mart**: a department-scoped subset of the DW.
  - When to use: when a department needs a simpler, faster, self-governed view than the full EDW.
  - How: extract/consolidate relevant DW data into a focused repository with its own access policies.
- **Operational Data Store (ODS)**: an enterprise-wide, near-real-time operational (not analytical) data integration layer.
  - When to use: when source-system reporting is too limited, security access is too restrictive, or minute-level freshness is needed (vs. daily DW refresh).
  - How: integrate data from multiple sources at low granularity, clean/normalize/apply business rules frequently throughout the day; can double as a DW staging area to reduce ETL flows.
- **Data Hub**: a centralized integration/distribution point combining multiple storage types (relational, NoSQL, lake).
  - When to use: need many-to-many data exchange between operational systems without point-to-point interfaces.
  - How: ingest/store copies from varied sources and distribute them; transformation/analysis typically happens outside the hub via external tools.
- **Master Data Management (MDM)**: creating a single "golden" master record per entity (customer, product, supplier).
  - When to use: whenever data from multiple sources could contain duplicate or near-duplicate entities.
  - How: copy entity data into an MDM tool; it cleans/standardizes/deduplicates (some matches need manual review) and builds hierarchies; master records feed back into the lake/DW, typically becoming star-schema dimension tables.
- **Data Virtualization / Data Federation**: a logical, non-replicating integration layer over multiple sources.
  - When to use: as a DW replacement when sources are numerous/volatile and speed-to-market matters more than deep performance tuning; or as an ETL replacement for ad hoc, less-performance-critical queries, regulatory data-residency cases, external-user access control, frequently-changing reference data, cross-store joins, sandbox analytics.
  - How: create a logical view over source systems; query engine pushes down filtering/aggregation to sources where possible and caches results; no data copied/stored centrally.
- **Data Catalog**: centralized metadata repository for discovery/governance (not data itself).
  - When to use: whenever data assets span many stores (DW, lake, ODS, marts) and need a single discovery/lineage/governance reference.
  - How: capture source/location info, schema/column metadata, lineage, governance/compliance details, and expose search/discovery tools.
- **Data Marketplace**: a platform for buying, selling, or internally exchanging datasets.
  - When to use: monetizing data assets externally, or helping internal users find/exchange data.
  - How: combine a data catalog with quality tooling, pricing/licensing, security/compliance controls, and often built-in analytics/visualization.

## Key Concepts
- **Golden source / best version of the truth**: the mastered record produced by MDM.
- **Near-duplicates**: records referring to the same entity but not textually identical (e.g., missing middle initial or suffix) — the specific failure mode simple dedup filters miss, which MDM tools solve.
- **Push-down query**: pushing filter/aggregation operations to the source system itself in a virtualization layer, improving performance but consuming source-side compute.
- **Virtual sandbox**: a secure, isolated environment for self-service analytics via virtualization, without copying data.
- **Data lineage**: the traceable path of data from origin through transformations to its current form — a core data catalog feature.

## Mental Models
- "Microscope vs. wide-angle lens": a data mart gives department-level depth (microscope); an ODS gives an enterprise-wide, real-time breadth (wide-angle lens).
- Data virtualization's responsive-grid mechanics: query results render progressively from cache, falling back to live remote fetch only for uncached data — always keeping the UI responsive.
- Decision test for moving vs. virtualizing data: move data (ETL/DW) when you'll transform once and query many times, or need fast joins across sources; virtualize when queries are ad hoc, sources are volatile, or regulatory/residency constraints forbid copying.

## Anti-patterns
- **Skipping MDM and shipping duplicate customer records**: end users see the same customer multiple times in a report, immediately question its accuracy, and lose trust in the data platform — trust that's hard to win back.
- **Treating data virtualization as a universal DW replacement**: it sacrifices performance, historical "as of" reporting accuracy, and other RDW-specific benefits (Ch 4) — must be evaluated against a specific checklist of questions (performance needs, push-down support, per-server installs, native vs. custom indexing, use of source DB statistics, security, near-duplicate handling, data cleaning location, source-schema fragility, licensing cost/complexity) before adopting.
- **Confusing a data hub/catalog/marketplace's purpose**: a data catalog stores no data (metadata only); a data hub stores copies and focuses on distribution; a marketplace focuses on commercial exchange — conflating these misassigns the wrong tool to the job.

## Reference Tables
| Solution | Primary Focus | Stores Actual Data? |
|---|---|---|
| Data hub | Storage, integration, distribution | Yes (copies) |
| Data lake | Storage, processing, analytics | Yes |
| Relational data warehouse | Storage, processing, analytics | Yes |
| Data catalog | Metadata management, data discovery | No |
| Data marketplace | Buying, selling, sharing | No (typically) |

| Dimension | ODS | Data Warehouse |
|---|---|---|
| Best suited for | Granular, low-level queries on detailed operational data | Complex queries on summary/aggregated analytical data |
| Freshness | Near-real-time (minutes) | Periodic (e.g., daily) |

| Dimension | Data Movement (ETL) | Data Virtualization |
|---|---|---|
| Dev/ops cost | Costly to build/maintain, duplicate storage | Lower; no duplicate storage |
| Time to solution | Slower (build/run jobs) | Faster |
| Security | More copies = more exposure | Centralized access control, no copies |
| Freshness/quality | Can go "stale"; ETL can introduce errors | Real-time, but push-down costs apply |
| Compliance | Moving data across boundaries risks governance issues | Easier to respect data-residency rules |

## Worked Example
ShoesForLess, a retail chain, uses an ODS so a customer service rep can query real-time stock for "men's running shoe, size 10" across all stores (sub-minute freshness), while the marketing manager instead queries the DW for aggregated year-over-year sales trends for that shoe style — illustrating when operational (ODS) vs. analytical (DW) data is the right tool. A second example: ShoesForLess's DW gets duplicate customer records when the same customer buys from multiple stores with slightly different name spellings (near-duplicates); without MDM, the first duplicate-laden report destroys end-user trust in the data.

## Key Takeaways
1. Use a data mart to give a department fast, governed access without navigating the full DW.
2. Use an ODS (not a DW) when the requirement is enterprise-wide, near-real-time operational reporting.
3. Use a data hub when the problem is many-to-many system integration/distribution, not analytics.
4. Always apply MDM when integrating multi-source entity data — near-duplicates are the default failure mode, not the exception.
5. Choose data virtualization over a DW/ETL when speed-to-market, regulatory data-residency, or ad hoc cross-source queries dominate; choose a physical DW when performance, historical "as of" accuracy, or heavy reuse dominate.
6. Use a data catalog for discovery/governance and a data marketplace for buying/selling/sharing — don't conflate either with a data hub, which stores and distributes actual data copies.

## Connects To
- **Ch 4**: Data marts are explicitly a subset of the RDW; MDM feeds dimension tables in a star schema.
- **Ch 5**: Compliance/data-residency drivers for multiple data lakes reappear as a virtualization use case (China data sovereignty example reused).
- **Ch 8**: MDM-mastered records become dimension tables joined to (unmastered) fact tables in dimensional modeling.
- **Ch 9**: ETL/ELT (Ch 9) is the alternative this chapter contrasts against data virtualization.
- **External concepts**: Informatica Enterprise Data Catalog, Collibra Data Catalog, Microsoft Purview (catalogs); Snowflake Marketplace, Datarade, Informatica Cloud Data Marketplace (marketplaces).
