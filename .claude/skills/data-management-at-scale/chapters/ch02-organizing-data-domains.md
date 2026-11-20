# Chapter 2: Organizing Data Using Data Domains

## Core Idea
Enterprise data architecture should be organized around business capabilities (not ad hoc technical groupings or pure DDD), mapping bounded contexts to capability instances to create domains that own their data end-to-end and distribute it only via standardized "common driveway" interfaces.

## Frameworks Introduced
- **Domain-Driven Design (DDD) applied to data** (Eric Evans, adapted by the author): Bounded contexts set logical boundaries around high-cohesion problem areas; domains/subdomains are the problem space, bounded contexts are the solution space.
  - When to use: To decide where to draw ownership and decoupling lines in a large, distributed data landscape.
  - How: Identify domains/subdomains (core, generic, supporting) → map to bounded contexts → assign one team per bounded context → enforce decoupled, standardized interfaces across boundaries.
- **Business Capability Mapping** (Ulrich Homann / BIZBOK Guide): A business capability is "a particular ability a business may possess to achieve a specific purpose," abstracting people, process, technology, and information.
  - When to use: As the primary mechanism for setting domain boundaries at enterprise scale (preferred over raw DDD, which the author finds too abstract/technical for data management).
  - How: Build a hierarchical capability map (L1/L2/L3) that is unique, mutually exclusive, and collectively exhaustive; map value streams → capabilities → applications/processes/data.
- **Common Driveway Pattern**: All cross-domain data distribution must exit through standardized, formalized interfaces (data products/APIs/events), never via direct/point-to-point access to internal data.
  - When to use: Any time a domain's data crosses its bounded-context boundary.
  - How: Domain teams encapsulate physical data models and expose logical, consumable data products via governed interfaces.

## Key Concepts
- **Golden source**: The authoritative application where authentic data for a context is originally created/managed; also called system of record (SOR).
- **Bounded context**: The logical boundary of a domain's solution space (code, DB design, team); polymorphic and context-dependent.
- **Ubiquitous language**: The shared, formalized vocabulary used within one bounded context; changes when the bounded context changes.
- **Source system–aligned domain**: A domain aligned with the golden source where data originates (transactional/operational).
- **Consumer-aligned domain**: A domain that consumes/transforms data from other domains to fit end-user use cases.
- **Aggregate domain / domain aggregate**: A domain that combines data from multiple domains into a single reusable unit for sharing (distinct from data aggregation, which is summarization).
- **Constructor domain**: A domain that creates genuinely new business insight (e.g., predictions) by combining other domains' logic, becoming a new provider.
- **Capability instance (realization)**: A specific implementation of a business capability; one capability can have many realizations (one-to-many), each potentially a separate domain.
- **Shared business capability**: A capability (e.g., customer management) implemented centrally and used by multiple business lines, creating a separation between "application domain" and "data domain" ownership.

## Mental Models
- **The street/house/fence metaphor**: A domain is a street (area of common interest); each house is a subdomain with its own property; the fence is the bounded context; crossing boundaries requires using the shared driveway (standardized interfaces) — captures both domain autonomy and the need for governed interchange.
- **Business capability vs. bounded context**: Capabilities describe the *what* (stable, business-level, abstract); bounded contexts/applications describe the *how* (solution-space, technical, changeable) — align them 1:1 where possible but expect divergence for shared/complex applications.
- **Service mesh → data mesh analogy**: Just as a service mesh decouples microservice-to-microservice calls into a managed layer, a data mesh decouples domain-to-domain data sharing into governed data products ("the smallest architectural quantum" — Dehghani).

## Anti-patterns
- **Point-to-point integration**: Creates exponential complexity (100 apps → ~5,000 connections; 1,000 apps → ~500,000), making dependencies impossible to oversee.
- **Using intermediate/virtualization layers to bypass golden sources**: Obscures lineage, degrades data quality, and adds complexity — domain data should flow directly from golden sources through governed interfaces.
- **Distributing data you don't own**: Any redistribution of another domain's data without its approval breaks ownership/accountability chains.
- **Ignoring shared-capability ownership splits**: Treating a shared application (e.g., a CRM/ERP) as having one single owner, when in reality there may be one application owner but multiple data owners per sub-context.
- **Bottom-up-only domain discovery**: Skipping an enterprise capability map "to save time" risks costly remodeling later when overlapping/conflicting boundaries emerge.

## Reference Tables
| Domain Type | Role | Example (Oceanic Airlines) |
|---|---|---|
| Source system–aligned | Captures data at golden source, distributes via common driveway | Customer management |
| Consumer-aligned | Consumes/transforms data from other domains for own use case | Discount and loyalty management |
| Aggregate domain | Combines multiple domains' data; uses it itself and/or distributes it (customer/supplier pattern) | Emissions and fees |
| Constructor domain | Aggregates + enriches with new business insight before exporting | Income and taxes |

| Capability Pattern | Ownership Shape |
|---|---|
| Generic capability | Single domain, single data/application owner |
| Shared capability | Single application owner, multiple data owners (segregate via multiple data products, or metadata-tagged rows within one product) |

## Worked Example
Oceanic Airlines (fictional airline): the business capability map includes ticket sales, baggage/lost items, aircraft maintenance. "Customer management" is source-system aligned and distributes customer data via a common driveway; "discount and loyalty management" is consumer-aligned, pulling from customer management; "emissions and fees" acts as an aggregate domain combining multiple domains' data (customer/supplier pattern); "income and taxes" behaves as a constructor domain, enriching aggregated data with new predictive insight (e.g., passenger travel habits) before exporting it as a new data product. For the shared "baggage handling and lost items" capability, each region (Asia, Europe) has its own realization/instance, creating separate subdomains.

## Key Takeaways
1. Build your domain architecture around business capabilities, not ad hoc technical or departmental groupings — capabilities are stable; the systems implementing them change.
2. Map bounded contexts 1:1 to capability instances wherever feasible; when an application serves multiple capabilities or is shared across business lines, explicitly separate application ownership from data ownership.
3. Enforce the "common driveway": all cross-domain data flow must go through standardized, governed interfaces — never point-to-point, never through intermediate layers that obscure the golden source.
4. Classify domains by their data-flow role (source-aligned, consumer-aligned, aggregate, constructor) to pick the right integration pattern for each.
5. One bounded context should be owned by one team; a team may own multiple bounded contexts, but boundaries must never overlap.
6. Only capture/modify data at the golden source; syntactic transformations (filter, rename, case conversion) don't create new ownership, but semantic changes do.

## Connects To
- **Ch 1**: Operationalizes the centralization/decentralization trade-off into a concrete domain model.
- **Ch 3**: Maps these logical domains onto a technology architecture (landing zones).
- **Ch 8**: Domain data ownership responsibilities extend into governance and security enforcement.
- **Ch 10**: Revisits aggregate/constructor domain patterns in the context of master data management and reuse.
- **Domain-Driven Design / Data Mesh**: Directly builds on and critiques Eric Evans' DDD and Zhamak Dehghani's data mesh "architectural quantum" concept.
