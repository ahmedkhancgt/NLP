---
name: data-management-at-scale
description: "Knowledge base from \"Data Management at Scale\" (2nd ed.) by Piethein Strengholt. Use when designing enterprise data architecture at scale — data domains/data mesh, technology architecture mapping, data product management, services/API management, event management, data governance and security, metadata, master data management, or turning data into value."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Management at Scale
**Author**: Piethein Strengholt | **Pages**: ~412 | **Chapters**: 12 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch04`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Decentralization is inevitable, but must be federated on top of central standards.** The book's foundational stance: scalability forces data management to decentralize (domains, data mesh), but federation always starts with centrally defined standards, boundaries, and expertise — pure decentralization without central alignment collapses into incompatible silos.

**Data Mesh vs. Data Fabric** — complementary, not competing. Data mesh emphasizes human/organizational federation (domain ownership, data as a product, self-serve platform, federated computational governance). Data fabric emphasizes the technology/metadata layer (unified metadata, automated integration). This book's architecture is explicitly a **hybrid of both**.

**Business Capability Mapping** (preferred over raw DDD for data management): Build domains from business capabilities (stable, business-level, abstract: *the what*), not from bounded contexts alone (technical, solution-space: *the how*). Align 1:1 where possible; split data ownership from application ownership when a capability is shared across business lines.

**The Common Driveway Pattern**: All cross-domain data distribution must go through standardized, governed interfaces (data products, APIs, events) — never point-to-point, never through obfuscating intermediate layers. The only sanctioned exception is genuine sub-100ms latency-critical use cases.

**Domain Topology Spectrum**: Fully Federated (data mesh) ↔ Governed ↔ Partially Federated ↔ Value Chain–Aligned ↔ Coarse-Grained ↔ Centralized. Choose based on organizational maturity, not ideology — mixing topologies across domains/regions is normal.

**Data Product as Logical Entity** (not Dehghani's literal "architectural quantum" container): Separate the logical data product (name, owner, business terms, relationships) from its physical representation(s), so the same semantic data can have multiple physical forms without duplicating metadata/ownership.

**CQRS as the Universal Read Pattern**: Every data product, read-oriented API, and DDS should be modeled as a CQRS read-replica — denormalized, read-optimized, decoupled from the operational write path, synced via batch/CDC/events.

**Trust Boundaries = Domain Boundaries**: Security should relax within a domain (context known) and tighten specifically at cross-domain interfaces (classification, data contracts, lineage-backed monitoring, automated enforcement).

**MDM Scope Discipline**: "If it's fast and fluid, break it apart and leave it to domains. If it's stable and it truly matters, consider MDM." Never pursue enterprise-wide unification — master only data genuinely shared across domains and truly critical (tens of attributes, not hundreds).

**Metadata is the only central data model you will have**: In a decentralized architecture with no single canonical data model, a shared, ubiquitous metadata layer (metamodel, catalog, knowledge graph) is what provides enterprise-wide coherence.

**Providing vs. Consuming asymmetry**: The providing side (data products) can and should be standardized because it's source-system aligned. The consuming side (DDS, BI, ML) resists standardization because every business use case is unique — use two separate blueprints for the two concerns.

**Phased Rollout**: Opportunistic (one use case, central team builds/coaches) → Transformation (add domains, metadata-driven pipelines, first governance) → Optimization (self-service, MDM, automated governance, data fabric/marketplace). "Control is good; trust is better" describes the maturity arc of the central team's role.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | The Journey to Becoming Data-Driven | Data Mesh vs. Data Fabric, DAMA-DMBOK 11 areas, centralization/decentralization trade-off |
| ch02 | Organizing Data Using Data Domains | DDD + Business Capability Mapping, Common Driveway Pattern, domain classification (source/consumer/aggregate/constructor) |
| ch03 | Mapping Domains to a Technology Architecture | 7 Domain Topologies, Landing Zone Topologies |
| ch04 | Data Product Management | Data Product as Logical Entity, CQRS, Medallion (Bronze/Silver/Gold), historization strategies |
| ch05 | Services and API Management | Federated API Responsibility, Resource-Oriented Architecture, API as a Product |
| ch06 | Event and Notification Management | Event Governance Model, EDA Reference Architecture, Consistency/Delivery Guarantee Framework |
| ch07 | Connecting the Dots | Data Distribution vs. Application Integration decision framework, Three-Model Consistency, Team Topologies |
| ch08 | Data Governance and Data Security | Governance Framework (roles/body/processes), Data Contract Framework, ABAC (PEP/PDP/PIP/PAP), Trust Boundary Model |
| ch09 | Democratizing Data with Metadata | Enterprise Metadata Model, Metalake Architecture (catalog + knowledge graph), semantic web standards |
| ch10 | Modern Master Data Management | Gartner's 4 MDM Styles, Domain-Level MDM Classification, Data Curation Collaboration Patterns |
| ch11 | Turning Data into Value | Domain Data Store (DDS), Managed vs. Self-Service Data, MLOps Reference Process |
| ch12 | Putting Theory into Practice | Three-Phase Implementation Roadmap, Centralized-vs-Decentralized Starting Point, DataOps Dual-Orchestrator Model |

## Topic Index

- **ABAC / RBAC / ACL** → ch08
- **API gateway / ESB** → ch05
- **Bounded context / DDD** → ch02
- **Business capability mapping** → ch02, ch07
- **CQRS** → ch04, ch05, ch06
- **Data catalog** → ch09
- **Data contracts** → ch08
- **Data curation** → ch10
- **Data fabric** → ch01, ch09
- **Data governance framework / roles** → ch08
- **Data lineage** → ch09
- **Data mesh** → ch01, ch03
- **Data products** → ch04
- **Domain Data Store (DDS)** → ch11
- **Domain topologies** → ch03
- **Event-driven architecture / streaming** → ch06
- **Golden source** → ch02
- **Knowledge graph / semantic web** → ch09
- **Landing zones** → ch03
- **Master Data Management (MDM)** → ch10
- **Metadata management / metamodel** → ch09
- **MLOps** → ch11
- **Resource-Oriented Architecture (ROA)** → ch05
- **Team Topologies** → ch07
- **Trust boundaries** → ch08
- **Ubiquitous language** → ch02

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — architecture/governance techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 12 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
