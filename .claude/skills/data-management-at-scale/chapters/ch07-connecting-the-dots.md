# Chapter 7: Connecting the Dots

## Core Idea
A unified "data and integration mesh" approach treats data products, APIs, and events as complementary, combinable patterns originating from the same underlying application architecture and ubiquitous language; successful large-scale rollout requires explicit decision criteria for choosing between patterns and a Team Topologies–based organizational model (domain/stream-aligned, platform, enabling, complicated-subsystem teams).

## Frameworks Introduced
- **Data Distribution vs. Application Integration decision framework**: Separates "making data available" (data products, batch/API/event ingestion for eventually-consistent reads) from "connecting applications in real time" (APIs, events for strongly consistent/low-latency operational use).
  - When to use: Choosing which integration pattern (or combination) fits a specific use case.
  - How: Ask (1) what delay is acceptable between operational change and read reflection, and (2) whether periodic reconciliation/auditing is required (→ batch); if speed matters, choose API (provider/consumer-initiated) or event (reactive/CDC-based) ingestion.
- **Three-Model Consistency Framework** (conceptual/logical/physical, from Ch.1): Ensures semantic consistency and discoverability by linking conceptual (business glossary) → logical (data/interface models) → physical (implemented schemas) models across both provider and consumer sides.
  - When to use: Any time you need enterprise-wide semantic consistency without a monolithic canonical model.
  - How: Centrally catalog conceptual business terms; link logical entities (data products/APIs/events) to them; validate against physical schema metadata.
- **Team Topologies** (Skelton & Pais): Four team types — stream-aligned (domain), platform, enabling, and complicated-subsystem teams — mapped onto data organizations.
  - When to use: Structuring the organization delivering a federated data architecture.
  - How: Domain teams = stream-aligned; central data platform + infrastructure teams = platform team; specialized teams (integration engineering, security/governance, MDM/quality, consuming/BI-ML) = enabling or complicated-subsystem teams depending on complexity.

## Key Concepts
- **Data distribution patterns**: Batch, API-based ingestion, and event-based ingestion — all producing eventually-consistent "data product" copies for intensive reads.
- **Application integration patterns**: Synchronous APIs (strongly consistent reads/commands) and asynchronous event brokers/message queues (resilience, elasticity, decoupling) for operational use cases.
- **Queue-based load leveling**: Combining APIs and events — queuing requests as a buffer in front of an API gateway to prevent overload failures.
- **Circuit breaker**: A resilience pattern that detects repeated failures and stops cascading effects in asynchronous communication.
- **Solution intent**: A Scaled Agile Framework artifact the author extended with a mandatory data/integration section so architecture review boards could evaluate pattern choices.
- **Architecture Review Board (ARB) / architecture design authority**: A governance body of experts who review and approve (or redirect) domain architects' solution proposals.
- **Inner sourcing**: Rotating or lending team members across teams (modeled on open-source contribution) to unblock dependencies without rigid resourcing walls.

## Mental Models
- **"Same underlying architecture" insight**: Data products, APIs, and events all originate from the same application architecture, database, and ubiquitous language — so managing the operational and analytical planes as separate, disconnected disciplines is artificial and counterproductive.
- **Combining patterns as a toolkit, not a choice of one**: E.g., streaming can populate multiple data products simultaneously; CQRS can route query traffic to data products and command traffic to operational APIs — patterns compose rather than compete.
- **Trust-boundary-aware exception handling**: The only legitimate exception to standardized (decoupled) cross-domain integration is latency-critical use cases (e.g., sub-100ms banking payment checks) where point-to-point connections are unavoidable — but all other governance duties (lineage, documentation) still apply even then.

## Anti-patterns
- **Managing operational and analytical architecture as fully separate disciplines**: Ignores that the same teams, context, and language underpin both, causing duplicated standards and governance effort.
- **Skipping domain/stream-aligned boundaries before building platform services**: Without clear business-capability-aligned domains first, platform investment lacks a stable foundation.
- **Overusing point-to-point connections beyond the single latency-critical exception**: Breaks the decoupling discipline established in Ch.2–6 and reintroduces exponential complexity.
- **Platform team attending every governance meeting directly** (no enabling team): Creates a bottleneck; specialized enabling teams should mediate instead.

## Reference Tables
| Need | Recommended Pattern |
|---|---|
| Periodic reconciliation / large historical batches | Batch data distribution |
| Near-real-time, provider/consumer-initiated pull | API-based ingestion |
| Reactive, state-change-triggered (e.g. CDC) | Event-based ingestion |
| Strongly consistent reads / direct commands | Synchronous APIs |
| Resilience, elasticity, no time criticality | Event brokers / message queues |
| Overloaded API gateway | Queue-based load leveling (API + event combo) |

| Team Topologies Role | Data Architecture Mapping |
|---|---|
| Stream-aligned team | Domain team (business owner, app owner, devs, data engineers) |
| Platform team | Central data platform team + central infrastructure team |
| Enabling team | Cross-cutting specialists (governance, security, integration coaching) |
| Complicated subsystem team | Deep-expertise builders (e.g., ML-based onboarding systems, MDM engines) |

## Worked Example
A four-domain solution design: Domains A and B are source-system-aligned and strictly forbid other domains from tapping into their operational application internals directly — they offer both operational reporting (low-latency, direct from the operational system) and analytical reporting (via copied/distributed data products). Domain C is a consumer that needs an extra local transformation step because upstream data products don't exactly fit its needs, so data lands in its own data product architecture before further processing. Domain D is a simpler consumer that virtualizes access to data products without duplicating data. This illustrates how source-aligned and consumer-aligned domains combine standardized landing-zone blueprints with domain-specific flexibility.

## Key Takeaways
1. Treat data products, APIs, and events as a combinable toolkit, not competing architectures — choose based on consistency/latency needs, not ideology.
2. Use the two-question test (acceptable delay? reconciliation needed?) to pick data distribution patterns; use consistency/time-criticality to pick application integration patterns.
3. Build semantic consistency by linking conceptual → logical → physical models centrally, rather than attempting a single canonical enterprise model.
4. The only acceptable exception to decoupled cross-domain integration is genuine latency-critical point-to-point need (e.g., sub-100ms payment checks) — and even then, governance duties remain.
5. Organize delivery using Team Topologies: domain teams as stream-aligned, central platform + infrastructure teams as the platform team, and dedicated enabling/complicated-subsystem teams for governance, security, and deep technical complexity.
6. Use dual planning cadences — sprints (1:1 team interaction) and Program Increment planning (many-to-many, cross-team alignment) — to keep a federated architecture program coordinated.

## Connects To
- **Ch 1**: Resolves the operational/analytical integration debate raised there.
- **Ch 4/5/6**: Synthesizes data products, APIs, and events into one decision framework.
- **Ch 2**: Reaffirms domain boundary and decoupling principles, including the single latency exception.
- **Ch 8**: Hands off into governance and security, the next layer of the architecture.
- **Team Topologies (Skelton & Pais)**: Directly adopted as the organizational design framework.
