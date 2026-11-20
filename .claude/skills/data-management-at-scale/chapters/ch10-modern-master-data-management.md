# Chapter 10: Modern Master Data Management

## Core Idea
"If it's fast and fluid, break it apart and leave it to domains; if it's stable and truly matters, use MDM" — master data management should be applied narrowly and deliberately (not as enterprise-wide unification) to restore necessary cross-domain consistency without undermining domain autonomy, choosing among four implementation styles and clearly separating MDM from the related but distinct discipline of data curation.

## Frameworks Introduced
- **Gartner's Four MDM Implementation Styles** (Lyn Robison): Consolidation, Registry, Centralized, Coexistence.
  - When to use: Selecting how intrusive/synchronized your MDM solution should be relative to source systems.
  - How: Registry (simplest, no source impact, cross-reference table) → Consolidation (hub for analytics, no feedback to sources) → Centralized (two-way sync, high coupling risk) → Coexistence (most complex, full bidirectional distribution); styles are combinable per dataset and can evolve over time.
- **Domain-Level MDM Classification**: Enterprise data / Domain data / Local data — three tiers of data overlap/importance.
  - When to use: Scoping what should be mastered centrally vs. within a few domains vs. left entirely local.
  - How: Enterprise data = cross-domain/regulatory importance (master centrally); domain data = shared by some but not all domains (master at domain-group level); local data = single-domain only (never put in MDM scope).
- **Data Curation Collaboration Patterns** (DDD-derived, from Eric Evans): Separate Ways, Partnership, Customer/Supplier, Conformist.
  - When to use: Deciding how multiple domains with overlapping data needs should collaborate on shared/curated data.
  - How: Separate Ways (low overlap, high autonomy); Partnership (high overlap, shared commitment); Customer/Supplier (one team owns, serves downstream); Conformist (full conformance, e.g. vendor package constraints).

## Key Concepts
- **Master data vs. reference data**: Master data is core, essential enterprise data (customer, product); reference data classifies/organizes other data (currency codes, country codes) — managed and distributed differently.
- **Master identification number (master identifier)**: A centrally assigned, often surrogate (GUID-based, ideally deterministic) key linking matched records across systems; should not be extrapolated to systems outside MDM scope.
- **MDM state store**: A NoSQL, continuously-updated database (often event-driven) supporting low-latency matching/merging for operational use cases (e.g., order processing needing millisecond customer lookups).
- **MDM domain**: A new domain created by integrating/harmonizing/mastering data for a subject area (customers, products, locations) — its output is distributed as "master data products," distinguished from regular data products.
- **Data curation**: Collecting data from diverse sources and integrating it to become "more valuable than its independent parts" — unlike MDM, curation CAN change semantic context and create new data/ownership.
- **Application key vs. surrogate key**: Business/application identifiers may not be globally unique across systems; MDM-assigned surrogate (ideally deterministic) keys solve this but must be managed carefully to avoid instability on re-ingestion.
- **MDM-as-a-service**: Offering MDM/data-quality capabilities (matching, profiling, standardization) centrally so domains don't each reimplement them.

## Mental Models
- **MDM feeds the lakehouse, never the reverse direct-write**: MDM's golden records (with full lineage/history) should feed into the Silver/Gold "medallion" pipeline as an input source — MDM should never write directly into dimension tables or the Gold layer, which remain the data warehouse/lakehouse team's domain.
- **"Golden records are authoritative because of their lineage"**: What distinguishes an MDM golden record from a mere Gold-layer dimension table is that MDM retains full lineage and history of how the record came to be, not just the final output.
- **MDM vs. DataOps division of labor**: DataOps (data engineers) solves data quality with automated code/pipelines; MDM (data stewards) provides an operational, human-in-the-loop workflow for issues that can't or shouldn't be fully automated.

## Anti-patterns
- **Enterprise data unification via MDM** ("mastering too much data"): Widening MDM scope to everything multiplies integration/governance/coordination effort; only data genuinely shared across domains and truly critical should be mastered.
- **Point-to-point connections between the MDM hub and providing/consuming applications**: Creates data inconsistency risk (consumers get MDM updates before data products reflect them) — always feed MDM from the same data products consumers use.
- **Centralized MDM style as a default choice**: Requires intrusive two-way sync with golden sources, creating tight coupling and difficulty modifying off-the-shelf application designs — "seriously consider any of the other approaches" first.
- **Republishing data through integration hubs without strong standards**: Risks traceability loss, data quality improvements never reaching golden sources, ownership obfuscation, and ultimately a tightly coupled architecture resembling the EDW "big ball of mud."
- **Predefining the complete MDM information model up front**: Master data assets and domain boundaries evolve, so rigid upfront modeling causes rework — take an Agile approach to MDM scope.
- **Extrapolating MDM master identifiers to every source system**: Creates consistency issues between domains that aren't in MDM scope.

## Reference Tables
| MDM Style | Source System Impact | Best For |
|---|---|---|
| Registry | None (no distribution back) | Quick wins, learning data quality/overlap issues |
| Consolidation | None (hub only improves copy, not source) | Analytics/BI/reporting |
| Centralized | High (two-way sync required) | Rare; only when full governance control outweighs coupling risk |
| Coexistence | Full (improvements flow back to golden sources) | Most mature/complex end state |

| Data Scope Tier | Mastering Decision |
|---|---|
| Enterprise data | Master centrally — cross-domain/regulatory significance |
| Domain data | Master within overlapping domain group only |
| Local data | Never put in MDM scope — leave to the single owning domain |

| MDM vs. Data Curation | MDM | Data Curation |
|---|---|---|
| Changes semantic meaning? | No — preserves original facts | Yes — can create new context/data |
| New ownership created? | No — traces back to golden source owner | Yes — new/derived data creates new ownership |
| Governance | Central, strict consistency rules | Domain-aligned, more flexible |

## Worked Example
An order management system needs millisecond-level customer lookups during order processing. The solution: an MDM state store built on a NoSQL database continuously injected with source-system data, optionally enriched with ML for quality, keeping a full transaction log (create/update/delete) for replay. This is kept logically separate from the core MDM solution (to avoid role confusion) while the MDM platform itself still "thinks" in event-streaming terms for consistency — illustrating low-latency MDM distribution patterns as a counterpoint to the standard batch-through-data-products pattern used for analytical/high-latency-tolerant use cases.

## Key Takeaways
1. Default to minimal MDM scope: only data that is both (a) shared across multiple domains and (b) genuinely stable/critical belongs in MDM — tens of attributes, not hundreds.
2. Start with the Registry style for a quick, low-risk MDM foundation; progress toward Coexistence only once governance, metadata, and processes are mature.
3. Never let the MDM hub bypass the data product architecture when distributing to analytical/batch-tolerant consumers — always route through the same data products other consumers use, to avoid inconsistency.
4. Clearly separate MDM (consistency-preserving, no new semantic context) from data curation (can create new data/ownership) for governance purposes — they overlap technically but differ fundamentally in intent.
5. Use master identifiers (ideally deterministic surrogate keys) only within MDM-scoped systems; never extrapolate them enterprise-wide.
6. Offer MDM and data-quality capabilities as a shared service to domains rather than requiring every domain to build its own matching/merging/profiling logic.
7. Feed MDM golden records into the lakehouse/warehouse pipeline (Silver→Gold) as an input, never writing MDM output directly into consumption-layer tables.

## Connects To
- **Ch 1**: Revisits reference/master data management, one of DAMA-DMBOK's 11 functional areas.
- **Ch 2**: Builds on aggregate/constructor domain patterns for distributing mastered data.
- **Ch 4**: Master data products are distributed using the same data product principles, just distinguished by MDM's upstream role.
- **Ch 8**: MDM is explicitly tied to compliant handling of personal data (identity matching, consent) alongside governance and security.
- **Ch 9**: Data curation and knowledge-exchange approaches extend the metadata democratization theme.
- **Ch 11**: Hands off to turning integrated/mastered data into business value next.
