# Chapter 3: Designing Good Data Architecture

## Core Idea
Good data architecture is the design of systems to support an enterprise's evolving data needs, achieved through flexible and reversible decisions reached by careful evaluation of trade-offs — it is a living process, not a fixed blueprint, and it's measured by how well it balances agility against stability.

## Frameworks Introduced
- **Reis & Housley's definition of data architecture**: "the design of systems to support the evolving data needs of an enterprise, achieved by flexible and reversible decisions reached through a careful evaluation of trade-offs."
  - When to use: as the north star when evaluating any architecture proposal.
  - How: ask whether a decision is reversible (two-way door) and what trade-off it makes.
- **Bezos's One-Way vs. Two-Way Doors**: irreversible (one-way) decisions vs. easily reversible (two-way) decisions.
  - When to use: before committing to any major technology or structural decision.
  - How: default to two-way-door decisions so you can adjust course cheaply as requirements or tools change; reserve one-way doors for rare, carefully-vetted choices.
- **9 Principles of Data Engineering Architecture**: (1) Choose common components wisely, (2) Plan for failure, (3) Architect for scalability, (4) Architecture is leadership, (5) Always be architecting, (6) Build loosely coupled systems, (7) Make reversible decisions, (8) Prioritize security, (9) Embrace FinOps.
  - When to use: as a structured checklist for any architecture review.
  - How: walk each principle against the design — e.g., for (2) define availability/reliability/RTO/RPO targets; for (6) check for tight technical/domain coupling; for (9) model pay-as-you-go cost dynamics up front.
- **Operational vs. Technical Architecture**: operational = what needs to happen (people/process/business requirements, e.g., latency SLAs, data quality ownership); technical = how it happens (ingestion/storage/transform/serving mechanics).
  - When to use: to separate "what the business needs" conversations from "which tool implements it" conversations.
  - How: define operational requirements first, then map technical architecture to satisfy them.
- **Shared Responsibility Security Model**: cloud provider secures "the cloud" (infrastructure); customer secures "in the cloud" (data, access, configuration).
  - When to use: whenever adopting any managed/cloud service.
  - How: explicitly enumerate what the vendor guarantees vs. what your team must configure (IAM, encryption, bucket policies).
- **Data Mesh (Dehghani)**: decentralizes data ownership along domain boundaries instead of a central lake/warehouse team.
  - When to use: when a centralized data platform has become an organizational bottleneck across many domains.
  - How: implement its four components — domain-oriented decentralized ownership, data as a product, self-serve data infrastructure as a platform, federated computational governance.

## Key Concepts
- **Domain vs. service**: a domain is a real-world subject area (e.g., sales); a service is a scoped function within it (e.g., order processing) — domains can share services (e.g., invoicing shared by sales and accounting).
- **Scalability, elasticity, availability, reliability**: scalability = capacity to grow; elasticity = ability to scale dynamically (including to zero); availability = % uptime; reliability = probability of meeting functional standards over an interval.
- **RTO / RPO**: Recovery Time Objective = max acceptable outage duration; Recovery Point Objective = max acceptable data loss, both set by business impact.
- **Tight vs. loose coupling**: tight coupling means components depend heavily on each other's internals (fragile, hard to change independently); loose coupling means components interact only through stable interfaces/APIs (each can evolve independently) — applies to both technical systems and teams.
- **Single-tier vs. multitier (n-tier) architecture**: single-tier colocates data+app on one machine (fine for prototyping, bad for production); multitier (e.g., three-tier: data/logic/presentation) decouples layers for resilience and independent scaling.
- **Monolith vs. microservices**: monolith = single codebase/tight coupling (simple to start, hard to evolve piecemeal); microservices = decentralized, independently deployable services (flexible, but adds operational complexity) — "monoliths aren't necessarily bad... just be prepared to break it into smaller pieces eventually."
- **Event-driven architecture**: production, routing, and consumption of events (state changes) as the core communication mechanism between loosely coupled services.
- **Brownfield vs. greenfield**: brownfield = evolving/refactoring an existing architecture (use the strangler pattern, not a risky big-bang rewrite); greenfield = starting fresh (watch for "shiny object syndrome" and resume-driven development).
- **FinOps**: cloud financial management discipline treating cost as an ongoing, dynamically monitored variable (pay-as-you-go) rather than a fixed capital expenditure.

## Mental Models
- **"Least worst architecture"** (Richards & Ford): there's no single "best" architecture, only trade-offs appropriate to context — stop searching for a perfect answer.
- **Architecture as leadership, not command-and-control**: the architect's highest-value activity is mentoring engineers and disseminating good decisions, not being the sole bottleneck decision-maker (Fowler's "Architectus Oryzus").
- **"Always be architecting"**: architecture is never finished; maintain a moving "target architecture" plus a "sequencing plan" that adapts as business and technology change.
- **Tiers describe "what depends on what," domains describe "who owns what"**: solving technical coupling (tiers) doesn't automatically solve domain/ownership coupling — both axes need separate design attention.
- **Batch as bounded streaming**: the Dataflow Model's insight — unbounded real-time streams and bounded batch datasets can be handled with the same windowed-aggregation code, dissolving the batch/stream dichotomy.

## Anti-patterns
- **Big-bang legacy rewrite**: replacing an entire brownfield architecture at once rather than incrementally (strangler pattern) — "often leads to disaster, with many irreversible and costly decisions."
- **Resume-driven development / shiny object syndrome**: greenfield teams stacking trendy technologies to look impressive rather than serving requirements.
- **Command-and-control architecture**: one architect forcing all teams onto a single proprietary technology — hinders innovation and creates a bottleneck.
- **Data lake 1.0 as a dumping ground**: write-only "data swamp"/"dark data"/WORN patterns with no schema management, cataloging, or deletion capability — technically unmanageable and non-compliant (e.g., with GDPR).
- **Ignoring FinOps until the bill arrives**: treating cloud spend as fixed/ignorable instead of continuously monitored, leading to runaway or "cost attack" scenarios (e.g., public S3 buckets driving egress costs through the roof).
- **Hard-perimeter security thinking in the cloud**: assuming a network boundary alone protects data, when cloud systems are inherently more exposed and insider/phishing threats bypass perimeters entirely.
- **Architecting in a vacuum**: designing domains/services without talking to actual business stakeholders and users.

## Code Examples
None — Chapter 3 is architecture/conceptual; no code snippets in source. (Figures reference workflows like ETL/ELT, Lambda/Kappa diagrams, described in prose only.)

## Reference Tables

| Architecture | Core idea | Known weakness |
|---|---|---|
| Data warehouse (ETL) | Centralize, transform before load, OLAP/OLTP separation | Rigid, expensive historically, slow to adapt |
| Data warehouse (ELT) | Load raw, transform inside the warehouse | Requires powerful compute/storage separation |
| Data lake 1.0 (HDFS/object storage) | Store everything cheaply, process on demand | Became "data swamp" — poor governance, hard deletes, high Hadoop talent cost |
| Data lakehouse | Lake storage + warehouse-like ACID/governance | Convergence still maturing |
| Modern data stack | Plug-and-play, cloud-based, modular off-the-shelf components | Tooling churns quickly |
| Lambda architecture | Separate batch + streaming + serving layers | Dual codebases, hard to reconcile, error-prone |
| Kappa architecture | Single stream-processing backbone for everything | Complex/expensive at scale; limited real adoption |
| Dataflow Model / unified batch-streaming (Beam, Flink) | Batch = bounded stream; same code for both | Still maturing in ecosystem support |
| Data mesh | Domain-owned decentralized "data as a product" | Needs strong federated governance to avoid chaos |

| Principle | One-line test |
|---|---|
| Common components | Is this reusable across teams without forcing a one-size-fits-all mandate? |
| Plan for failure | Have we defined availability, reliability, RTO, RPO targets? |
| Scalability | Can it scale up, down, and (ideally) to zero? |
| Architecture is leadership | Is the architect mentoring, not bottlenecking? |
| Always be architecting | Do we have a living target architecture + sequencing plan? |
| Loose coupling | Can a team ship independently without coordinating with every other team? |
| Reversible decisions | Is this a two-way door? |
| Security | Have we assumed zero-trust and applied least privilege everywhere? |
| FinOps | Is spend monitored continuously, with alerting on spikes? |

## Worked Example
A retailer currently runs a single centralized data warehouse (brownfield, ETL-based, tightly coupled single-tier reporting DB feeding directly off production). As the company scales, the sales and inventory domains each want independent release cycles. The architect applies the strangler pattern rather than a big-bang rewrite: a sales-specific pipeline and warehouse table set is stood up first, decoupled behind an API (per the Bezos API Mandate), while inventory continues on the legacy path. Each new component is a two-way-door decision — built on a cloud data warehouse with separated compute/storage so it can scale elastically for month-end reporting spikes and scale down afterward (principle 3, FinOps). Security follows zero-trust: each service gets least-privilege IAM roles rather than shared superuser credentials. Over 18 months, more domains are peeled off the monolith into this loosely coupled, domain-oriented pattern — effectively evolving the centralized warehouse toward data-mesh-like domain ownership without a disruptive one-time migration.

## Key Takeaways
1. Good architecture favors flexible, reversible ("two-way door") decisions over "best" one-time choices.
2. Data architecture is a living process ("always be architecting"), not a static diagram.
3. Separate operational requirements (what) from technical implementation (how).
4. Tight coupling (technical or organizational) impedes independent team velocity; design for loose coupling deliberately.
5. Brownfield migrations should use incremental patterns (strangler) over risky big-bang rewrites.
6. Security is now every data engineer's job under zero-trust and shared-responsibility cloud models.
7. FinOps must be embraced proactively — cloud's pay-as-you-go model makes cost a continuous, monitorable variable, not a fixed sunk cost.
8. Know the major architecture archetypes (warehouse, lake, lakehouse, modern data stack, Lambda, Kappa, unified Dataflow model, data mesh, IoT) and their specific trade-offs well enough to choose contextually, not dogmatically.

## Connects To
- **Ch 2**: Data architecture is one of the six undercurrents introduced there; this chapter expands it fully.
- **Ch 4**: Choosing specific technologies to implement the architecture principles established here.
- **Ch 6**: Storage systems (warehouse/lake/lakehouse) detailed further.
- **Ch 8**: Data modeling approaches (Kimball, Inmon, Data Vault) referenced for transformation within warehouse/ELT architectures.
- **Ch 10**: Security undercurrent (zero-trust, shared responsibility) expanded.
- **Ch 11**: Future directions, including "live data stack."
- **External concept**: TOGAF, Gartner EA, EABOK, AWS Well-Architected Framework, Google Cloud's Five Principles for Cloud-Native Architecture, Domain-Driven Design (Eric Evans), Zhamak Dehghani's Data Mesh, Martin Kleppmann's Designing Data-Intensive Applications.
