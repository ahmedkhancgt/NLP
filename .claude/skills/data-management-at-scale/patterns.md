# Patterns

## Common Driveway Pattern
**When to use**: Any time data crosses a domain/bounded-context boundary.
**How**: Never distribute data point-to-point or through intermediate/obfuscating layers; always expose via a standardized interface (data product, API, or event) that the domain governs and owns.
**Trade-offs**: Adds an abstraction/governance layer but prevents exponential point-to-point complexity (100 apps → ~5,000 connections).

## CQRS for Data Products
**When to use**: Any time consumers need to read data intensively without impacting operational write performance.
**How**: Separate write-optimized command model from a read-optimized, denormalized query model (the data product); sync via batch/CDC/events; route queries to the read store, commands to the operational system.
**Trade-offs**: Adds sync complexity and latency, but dramatically improves scalability and isolates operational systems from analytical load.

## Medallion Layering (Bronze/Silver/Gold)
**When to use**: Structuring any lakehouse-based data product or DDS pipeline.
**How**: Bronze = raw, 1:1 copy, date-partitioned; Silver = quality-checked, SCD-processed, functional-only; Gold = business-logic-transformed, denormalized, subject-area grouped, consumption-ready.
**Trade-offs**: Clear separation of concerns and reprocessing flexibility, at the cost of extra storage and pipeline stages.

## Governed Domain Topology (Hub-and-Spoke)
**When to use**: Organizations wanting mesh-style domain ownership but needing centralized distribution control (financial institutions, governments).
**How**: Each domain owns its data/applications but must route data products through a centrally managed, shared distribution layer.
**Trade-offs**: Reduces network/connectivity explosion and eases early-stage conformance, but the central layer can become a bottleneck and complicate multicloud deployments.

## Landing Zone Topology
**When to use**: Implementing any domain topology concretely on cloud infrastructure.
**How**: Always provision at least one domain-agnostic data management landing zone (catalog, lineage, MDM, security) plus one or more data landing zones (domain workloads); scale to multiple zones by security, region, pace-of-change, or politics.
**Trade-offs**: More landing zones = more isolation/flexibility but exponentially more configuration/connectivity management overhead.

## API-as-a-Product / Federated API Responsibility
**When to use**: Any cross-domain API exposed beyond internal application boundaries.
**How**: Keep domain logic in the domain; push composite/aggregation logic to the consumer; use lightweight API gateways instead of a central ESB; register all APIs in a service catalog with OpenAPI specs.
**Trade-offs**: Requires domain discipline and central registry investment, but avoids ESB-style monolithic bottlenecks and ownership obfuscation.

## Event Governance Segregation (Internal vs. Domain-to-Domain Streams)
**When to use**: Any shared event broker/streaming platform spanning multiple domains.
**How**: Keep internal-application streams technical and consumable only by the producing domain; make domain-to-domain ("business") streams read-optimized, versioned, cataloged, ubiquitous-language-based.
**Trade-offs**: Adds registry/governance overhead but prevents other domains from consuming unstable internal communication.

## Data Contract Lifecycle
**When to use**: Whenever data crosses a trust boundary (domain-to-domain, consumer-to-provider).
**How**: Start with manual forms → web app/DB → automated schema/quality validation (Great Expectations, dbt test) → workflow-driven owner approval with fine-grained filtering → full self-service with ML-assisted approval.
**Trade-offs**: Slow to build maturity but avoids governance overload early and builds organizational trust incrementally.

## Trust-Boundary Security Model
**When to use**: Designing security controls across a federated architecture.
**How**: Relax controls within a domain (context is known); tighten controls (classification, contracts, lineage-backed monitoring, automated enforcement) specifically at domain/data-product/API/event interfaces.
**Trade-offs**: More nuanced than uniform security, but matches risk to actual trust level instead of over- or under-protecting.

## ABAC with PEP/PDP/PIP/PAP
**When to use**: When RBAC/ACL granularity is insufficient or contextual factors (location, time, device) must influence access.
**How**: Policy Enforcement Point intercepts requests; Policy Decision Point evaluates against policy; Policy Information Point supplies external attributes; Policy Administration Point centrally manages all policies.
**Trade-offs**: More flexible and fine-grained than RBAC, but requires more infrastructure investment and skilled engineering.

## MDM Registry-First Rollout
**When to use**: Starting master data management in any organization, regardless of ultimate target style.
**How**: Begin with the Registry style (cross-reference table, no source system impact) to learn what's duplicated/low-quality; progress to Consolidation, then Coexistence only as governance/metadata matures.
**Trade-offs**: Slower to deliver full consistency, but minimizes risk and avoids premature intrusive integration with golden sources.

## Data Curation Collaboration Patterns (Separate Ways / Partnership / Customer-Supplier / Conformist)
**When to use**: Multiple domains with overlapping data/integration needs deciding how to collaborate on shared/curated data.
**How**: Separate Ways for low overlap/high autonomy; Partnership for high overlap requiring shared commitment; Customer/Supplier when one team owns and serves downstream; Conformist when integration complexity or vendor constraints require full conformance.
**Trade-offs**: Each pattern trades autonomy against consistency/coordination cost — pick per situation, not globally.

## DDS/Data Product Blueprint Separation
**When to use**: Architecting the consuming side of a federated data platform.
**How**: Use two separate blueprints — one for data product creation (providing), one for turning data into value (consuming/DDS) — even when the same domain plays both roles.
**Trade-offs**: Avoids tight coupling between use-case-specific transformation logic and stable data product serving, at the cost of maintaining two blueprint types.

## Phased Implementation Roadmap (Opportunistic → Transformation → Optimization)
**When to use**: Rolling out any large-scale federated data architecture.
**How**: Phase 1 — one use case, single landing zone, central team builds/coaches. Phase 2 — add domains, metadata-driven pipelines, first computational governance. Phase 3 — self-service, MDM, automated governance, data fabric/marketplace.
**Trade-offs**: Slower initial value delivery but builds organizational trust and avoids the "boil the ocean" failure mode of big-bang rollouts.
