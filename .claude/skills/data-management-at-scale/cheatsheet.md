# Cheatsheet

## Choosing a Domain Boundary
1. Start from business capabilities, not technical/departmental groupings (map capability → capability instance → bounded context → domain).
2. One bounded context = one team (a team may own several bounded contexts, but boundaries must never overlap).
3. If an application serves multiple capabilities (shared capability), split data ownership from application ownership explicitly.
4. Smell: if two "domains" use the same application/database and fight over governance rules, you have a shared capability, not two domains.
5. Smell: if integrating two domains' data always requires the same join/lookup, consider a shared kernel or aggregate domain instead of duplicating logic.

## Choosing Sync vs. Async Integration
| Need | Choose |
|---|---|
| Strongly consistent read / direct command | Synchronous API |
| Periodic reconciliation, large batch, cost-sensitive | Batch data distribution |
| Reactive, state-change-triggered, resilient to spikes | Event broker / message queue |
| Sub-100ms latency-critical (e.g., payment fraud check) | Point-to-point (the ONLY sanctioned exception) |
| Overloaded API gateway under bursty load | Queue-based load leveling (API+event combo) |

## Choosing a Domain Topology
- Low engineering maturity / need strong control → **Centralized** or **Governed**.
- High maturity, cloud-native, microservices-heavy → **Fully Federated (data mesh)**.
- Legacy constraints / skills gaps on source side only → **Partially Federated**.
- Supply chain / end-to-end business flow → **Value Chain–Aligned**.
- Post-M&A, many legacy apps → **Coarse-Grained** (plan a transition path out of it).

## Choosing an MDM Style
1. Default to **Registry** first (no source impact, fastest to value).
2. Need analytics-only consistency, no write-back → **Consolidation**.
3. Need full governance with two-way sync AND can tolerate tight coupling → **Centralized** (use cautiously).
4. Need improvements to flow back to golden sources → **Coexistence** (most mature, most complex — end goal, not starting point).
5. Scope test: data used by only ONE domain → never master it. Data used across MANY domains and genuinely critical/stable → master it. Tens of attributes, not hundreds.

## Access Control Model Selection
| Situation | Use |
|---|---|
| Individual object-level permissions | ACL |
| Org-wide, admin-manageable roles | RBAC (watch for "role explosion") |
| Context (location/time/device) must matter, or RBAC too coarse | ABAC (PEP/PDP/PIP/PAP) |

## Data Distribution Pattern Decision Tree
1. Is periodic reconciliation/audit required? → **Batch**.
2. Is near-real-time delay (minutes/seconds) acceptable and provider/consumer-initiated? → **API-based ingestion**.
3. Is near-real-time delay acceptable and reactive/CDC-triggered? → **Event-based ingestion**.
4. Patterns combine: streaming can populate data products; CQRS can route reads to products and commands to APIs.

## Tells and Smells
- **Smell**: a central team is still fixing data quality for other teams' pipelines → governance/ownership not actually federated yet.
- **Smell**: consumers get MDM updates before analytical data products reflect them → MDM is bypassing the data product architecture; always feed MDM from the same products consumers use.
- **Smell**: "data product" used for both provider-side stable data AND consumer-side use-case-specific transformed data → terminology collapse; call the latter DDS/integrated data instead.
- **Smell**: ESB/central integration layer growing "optional attributes for everyone" → recreating the enterprise-canonical-model/EDW failure mode.
- **Smell**: teams building their own matching/merging/data-quality logic independently → should be offered as a shared MDM/data-quality service instead.
- **Tell you're ready to decentralize further**: central team shifts from executing governance to "coaching + automated guardrails" ("control is good, trust is better").
- **Tell you need a DDS instead of direct consumption**: data product platform can't meet performance needs, OR multiple products must be combined/harmonized, OR new business insight must be persisted.

## Thresholds & Defaults
- First landing zone / proof-of-concept: target delivery in **8–16 weeks**.
- MDM scope: **tens of attributes**, not hundreds, per mastered subject.
- Team size for cross-functional domain/DDS teams: **5–10 people** (DevOps-team-sized).
- Point-to-point connection growth: 100 apps → ~5,000 connections; 1,000 apps → ~500,000 — avoid at all costs beyond the single latency exception.
- Big data project failure rate (Gartner/Deloitte/VentureBeat, cited): **>60%** — usually due to raw-data coupling, operational risk, and shared-platform complexity.

## Quick Governance Checklist (before sharing any data)
- [ ] Data owner assigned (not just application owner)
- [ ] Classification + usage classification applied
- [ ] Linked to business glossary term
- [ ] Data contract in place if crossing a trust boundary
- [ ] Lineage captured back to golden source
- [ ] No raw/unvalidated data exposed as production interface
