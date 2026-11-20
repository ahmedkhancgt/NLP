# Chapter 11: Data Fabric

## Core Idea
Data fabric is an evolution of the MDW that adds (per the author's definition) at least three of eight technology features — data access policies, a metadata catalog, MDM, data virtualization, real-time processing, APIs, services, and products — to enhance organization-wide data accessibility, security, discoverability, and availability, with no bright line separating it from an advanced MDW.

## Frameworks Introduced
- **Author's Data Fabric Definition (≥3-of-8 features)**: a practical, non-dogmatic test for when an MDW becomes a data fabric.
  - When to use: deciding whether a given architecture should be called/built as a data fabric vs. staying an MDW.
  - How: count how many of the eight features (below) the architecture implements; ≥3 = data fabric by this book's definition, though many solutions reasonably call themselves a data fabric by philosophy alone, with zero of these technologies.
- **Eight Data Fabric Technology Features**: the toolkit that extends an MDW into a fabric.
  - When to use: as a menu to select from when scaling an MDW for larger, more complex, more regulated data environments (especially >10TB).
  - How:
    1. **Data access policies** — governance rules on who/how/when data can be accessed, enforced via APIs/drivers/virtualization, supporting regulations like GDPR/HIPAA.
    2. **Metadata catalog** — centralized repository of data asset structure/relationships/characteristics plus data lineage, enabling discovery and avoiding duplicate ingestion/reporting work.
    3. **MDM** (from Ch 6) — single authoritative master records for key entities.
    4. **Data virtualization** (from Ch 6) — optional in the author's view (disputed by some, including Gartner, who treat it as defining); provides a logical access layer without copying data.
    5. **Real-time processing** (from Ch 9) — immediate results as data arrives.
    6. **APIs** — standardized, location-independent data access that insulates calling applications from underlying data movement.
    7. **Services** — reusable, generic encapsulated code (e.g., a cleaning/ingestion routine) usable by teams beyond the full fabric's scope.
    8. **Products** — an entire data fabric packaged and sold, especially industry-specific (e.g., healthcare).
  - **Gartner's alternative definition**: an "intelligent data fabric" using knowledge graphs, AI/ML for automated discovery, cataloging, relationship-finding, and orchestration — which the author considers largely aspirational/not-yet-existing technology, recommending focus on data fabric as it exists today instead.

## Key Concepts
- **Data lineage**: within the metadata catalog, the traceable history of a data value's origin and transformations — critical for trusting decisions based on that value and for regulatory compliance.
- **API as an abstraction layer**: decouples applications from underlying data location, so moving data only requires updating the API's internal code, not every consumer.
- **Services as reusable building blocks**: a data fabric built in "blocks" lets teams outside the full fabric reuse specific capabilities (e.g., just the data-cleaning routine) without adopting the whole fabric.

## Mental Models
- The "fabric weaving through a company" metaphor: data fabric is meant to be a pervasive connective layer that reaches all data, inside and outside the organization, regardless of size/speed/type — contrast with the MDW's more contained lake+RDW pairing.
- No bright line: treat the MDW→data-fabric transition as a continuum/upgrade path (adding features incrementally), not a binary architecture swap.

## Anti-patterns
- **Assuming data virtualization is mandatory for a "true" data fabric**: the author explicitly treats it as optional, diverging from Gartner and others who call it defining — don't gate a data fabric initiative on building virtualization if it isn't needed.
- **Adopting Gartner's "intelligent data fabric" vision (AI/ML auto-discovery, knowledge graphs) as achievable today**: the author states much of this technology doesn't yet exist in a way that satisfies the definition — building toward it prematurely risks wasted investment.
- **Transitioning small businesses with simple, limited data sources to a full data fabric**: the added complexity, cost, training, and integration burden isn't justified when an MDW already suffices.
- **Underestimating in-house expertise requirements**: data fabric's complexity makes troubleshooting harder; moving without adequate in-house or partner expertise is a named risk.

## Reference Tables
| Driver to Move MDW → Data Fabric | Why |
|---|---|
| Data type/source variety growing fast | Fabric is designed for flexible, evolving adaptation |
| Need for a unified view across many sources | Fabric weaves sources together seamlessly |
| Need for real-time insight | Fabric supports real-time processing natively |
| Rising data breach / regulatory concern | Advanced access policies strengthen governance |
| Multinational, multi-jurisdiction compliance | Advanced governance capabilities adapt per jurisdiction |
| Volatile, fast-moving markets (trading, ecommerce) | Real-time processing is a necessity, not a luxury |

| Data Fabric Pros | Data Fabric Cons |
|---|---|
| Scalability, future-proofing | Resource-intensive transition (cost, training, integration) |
| Unified view across varied sources | Unneeded for small businesses with simple needs |
| Real-time processing | Increased troubleshooting complexity |
| Stronger governance/security/compliance | Requires significant in-house or partner expertise |

## Worked Example
A healthcare organization enforces a data access policy requiring only authorized medical staff to access patient records: it implements APIs that interface with secure authentication systems, so when a provider requests a patient's record, the API checks the provider's credentials against access policies before granting access — illustrating how data access policies + APIs operationalize compliance (e.g., HIPAA) within a data fabric.

## Key Takeaways
1. Data fabric is best understood as an incremental extension of the MDW via a defined set of 8 features, not a wholly separate architecture — use the ≥3-feature heuristic practically, not dogmatically.
2. Treat data virtualization as optional in a data fabric; don't let disputed industry definitions (e.g., Gartner's) force unnecessary virtualization investment.
3. Be skeptical of "intelligent data fabric" marketing promising full AI/ML-automated discovery/cataloging — much of that technology isn't mature yet.
4. Data fabric is most justified for large (>10TB), fast-evolving, highly regulated, or real-time-sensitive environments — not for small businesses with simple needs.
5. Build data fabric capabilities (especially services) as reusable blocks so teams who don't need the whole fabric can still benefit from individual components.

## Connects To
- **Ch 6**: MDM and data virtualization are directly reused from Ch 6's treatment.
- **Ch 9**: Real-time processing concept is directly reused from Ch 9.
- **Ch 10**: Data fabric follows the same five-stage data journey (ingestion, storage, transformation, modeling, visualization) as the MDW, just with added features layered on top.
- **Ch 13/14**: Data fabric remains a centralized architecture, contrasted explicitly with the decentralized data mesh discussed next.
- **External concept**: Gartner's "intelligent data fabric" definition, explicitly contrasted with the author's own.
