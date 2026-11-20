# Chapter 3: Mapping Domains to a Technology Architecture

## Core Idea
There is no single correct domain topology — organizations choose among a spectrum from fully federated (data mesh) to centralized, and implement the chosen topology on cloud infrastructure using "landing zones," balancing autonomy, cost, governance, and maturity.

## Frameworks Introduced
- **Domain Topologies** (author's own framework, 7 variants): Patterns describing how domains are grouped, interrelated, and connect to each other.
  - When to use: To decide the governance/autonomy shape of your enterprise data architecture before choosing technology.
  - How: Pick (or blend) from: Fully Federated (data mesh), Governed, Partially Federated, Value Chain–Aligned, Coarse-Grained, Coarse-Grained and Partially Governed, Centralized.
- **Landing Zone Topology** (Cloud Adoption Framework pattern — AWS/Azure/GCP): Preconfigured, code-provisioned environments (security, networking, storage, compute) for domains.
  - When to use: To implement any domain topology concretely on cloud infrastructure.
  - How: Always provision at minimum one domain-agnostic data management landing zone (catalog, lineage, MDM, security oversight) plus one or more data landing zones (domain workloads); scale to multiple landing zones by security, region, pace of change, politics, etc.

## Key Concepts
- **Fully federated (data mesh) topology**: Peer-to-peer data distribution, no central orchestrator, computational (automated) governance; theoretically ideal but requires high engineering maturity and critical mass.
- **Governed domain topology**: Hub-and-spoke variant of data mesh — domains own data but must route data products through a centrally managed shared distribution layer.
- **Partially federated domain topology**: Centralization on the source/producing side (often via a central engineering team), federation/mesh-style autonomy on the consuming side.
- **Value chain–aligned domain topology**: Larger domains grouped around an end-to-end business value chain (e.g., supply chain); tight internal coupling allowed, standards enforced only at the chain's boundary.
- **Coarse-grained domain topology**: Domains aligned to broad units (business unit/region), common after M&A growth; risks capability duplication and ownership obfuscation.
- **Centralized domain topology**: One shared (often lakehouse) platform instance with domain-aligned storage segregation; appropriate for smaller/less mature organizations.
- **Data management landing zone**: The domain-agnostic foundation zone hosting catalog, lineage, API catalog, MDM, security/monitoring — always singular per "sphere of trust."
- **Data landing zone**: An account/subscription hosting one or more domains' workloads and data-related services (ingestion, storage, compute, reporting).
- **Data lakehouse**: Architecture combining data lake flexibility/cost with data warehouse ACID transaction management, often used as the technology substrate for centralized or single-landing-zone designs.

## Mental Models
- **The "elephant in the room" for data mesh**: Full federation demands (1) organization-wide standards conformance, (2) acceptance of exponential peer-to-peer network/connectivity growth, (3) tolerance for fine-grained performance/reconciliation friction, and (4) a large pool of highly skilled engineers — most organizations lack one or more of these.
- **Universe/galaxy formation analogy**: Early-stage architecture goes through collisions and mergers before settling into harmony — don't expect (or demand) a perfect initial design; "start small and fail fast, fail fast to succeed faster."
- **Topology mixing is normal**: Different domains/regions within the same enterprise can legitimately use different topologies simultaneously (e.g., some fully federated, others hub-and-spoke).

## Anti-patterns
- **Domain-oriented data warehouse as a "supermodel"**: Combining raw retention and harmonization in a single coarse-grained warehouse serving many domains creates contradictory design goals and must be avoided at all costs.
- **Treating the first topology choice as final**: Architectures evolve through phases; locking in a "perfect" initial design sets you up for failure.
- **Letting coarse-grained boundaries obscure ownership**: Distributing through intermediary systems/silos/point-to-point interfaces makes it impossible to trace true data origin.
- **Overprovisioning landing zones without a management layer**: Multiple data landing zones without a unifying data management landing zone lose central auditing/control.

## Reference Tables
| Topology | Centralization Level | Best Fit | Key Risk |
|---|---|---|---|
| Fully federated (data mesh) | Lowest | Cloud-native, microservices-heavy, highly skilled orgs | Standardization & talent requirements; network explosion |
| Governed | Medium-low | Financial institutions/governments wanting scalability + control | Central layer becomes bottleneck |
| Partially federated | Mixed | Orgs with legacy constraints / skill gaps | Inconsistent rules; onboarding overhead |
| Value chain–aligned | Medium (chain-internal low) | Supply chain, product development, transportation | Less data-driven outside chain; unclear boundaries |
| Coarse-grained | Medium-high | Post-M&A, large complex orgs | Capability duplication; political infighting |
| Coarse-grained & partially governed | Medium-high | Large orgs wanting some cross-boundary control | Requires high platform-team maturity |
| Centralized | Highest | Smaller/less mature orgs, lakehouse shops | Loses mesh-style reusability/autonomy benefits |

| Landing Zone Pattern | Use Case |
|---|---|
| Single data landing zone | Simplicity, greenfield, strong standardization desired |
| Source/consumer split landing zones | Complex onboarding separated from consumption services |
| Hub data landing zone | Central control of all data movement (regulated/global orgs) |
| Multiple data landing zones | Separation by security, region, pace of change, politics, SLAs |
| Multiple data management landing zones | Large global enterprises with contrasting governance needs (e.g., per-region MDM) |

## Worked Example
Oceanic Airlines implements three data landing zones (US/"Central," Europe, Asia) for latency/data-residency reasons, each fed by region-specific blueprints (one region even uses a different tool blueprint for trained staff), all connected to one data management landing zone for consistent governance (shared data catalog enforces computational governance automatically). The design is inspired by a real-world Amadeus implementation that added a self-service "infrastructure provisioning plane" letting domains onboard in minutes via a simple form.

## Key Takeaways
1. Match your domain topology to organizational maturity and values: full federation needs massive engineering investment; smaller/control-focused orgs should start centralized and evolve.
2. Don't force a single topology enterprise-wide — mixing topologies per domain/region is normal and often optimal.
3. Always separate a domain-agnostic data management landing zone (governance/catalog/lineage) from domain-specific data landing zones, even in the simplest design.
4. Decide deliberately how data products align with storage accounts/instances (per-domain, shared, or hybrid) — each choice trades autonomy against cost and manageability.
5. Use cloud adoption framework blueprints (AWS/Azure/GCP) as repeatable, code-provisioned templates rather than building landing zones from scratch each time.
6. Get a first landing zone live in 8–16 weeks on a small proof-of-concept rather than trying to design the perfect target state up front.

## Connects To
- **Ch 2**: Directly implements the domain/business-capability model as physical infrastructure.
- **Ch 4**: The data landing zone becomes the host for data product architectures discussed next.
- **Ch 10/Ch 12**: Enterprise consistency and step-by-step transition themes recur when discussing MDM and practical rollout.
- **Data Mesh (Zhamak Dehghani)**: This chapter is a systematic critique and pragmatic adaptation of the pure data mesh topology for enterprise scale.
