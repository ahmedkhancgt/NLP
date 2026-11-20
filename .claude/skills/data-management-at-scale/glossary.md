# Glossary

**ABAC (Attribute-Based Access Control)** — Access control combining data, user/system, and action attributes via PEP/PDP/PIP/PAP components (Ch 8).

**Aggregate domain** — A domain that combines data from multiple domains, using it for itself and/or sharing it with others (Ch 2).

**API gateway** — Lightweight component providing routing, security, and monitoring for APIs without embedding business logic (Ch 5).

**Application key vs. surrogate key** — Business identifier (may not be globally unique) vs. system-generated unique key (ideally deterministic) used in MDM (Ch 10).

**Bounded context** — The logical (solution-space) boundary of a domain's high-cohesion area, implementing a business capability instance (Ch 2).

**Business capability** — A particular ability a business possesses to achieve an outcome, abstracting people/process/technology/information (Ch 2).

**Centralized domain topology** — Single shared (often lakehouse) platform with domain-aligned storage segregation (Ch 3).

**CEP (Complex Event Processing)** — Pattern detection across multiple event streams/time windows; a subset of ESP (Ch 6).

**Coarse-grained domain topology** — Domains aligned to broad units (business unit/region), common after M&A (Ch 3).

**Coexistence style (MDM)** — Most complex MDM style; improvements flow back to golden source systems (Ch 10).

**Common driveway pattern** — Principle that all cross-domain data distribution must use standardized, governed interfaces (Ch 2).

**Composite service** — An API that executes multiple REST requests in a single call; integration logic pushed to consumers (Ch 5).

**CQRS (Command Query Responsibility Segregation)** — Separating write-optimized command models from read-optimized query models, synced via events (Ch 4).

**Data contract** — Formal agreement between provider and consumer on what data may be consumed, for what purpose, under what SLA (Ch 8).

**Data domain** — The organizational/architectural boundary aligned with a business capability's realization, owning its data end-to-end (Ch 2).

**Data fabric** — Metadata-driven architectural vision for unified, self-service data access via an integrated layer (Ch 1, Ch 9).

**Data lakehouse** — Architecture combining data lake flexibility/cost with data warehouse ACID transaction management (Ch 3).

**Data mesh** — Decentralized data architecture founded on domain ownership, data as a product, self-serve platform, and federated governance (Ch 1, Ch 3).

**Data observability** — An organization's ability to understand freshness, integrity, volume, lineage, and quality trends across its data estate (Ch 8).

**Data product** — An autonomous logical entity describing data meant for consumption, decoupled from its physical representation(s) (Ch 4).

**Data Vault** — A hub/link/satellite data modeling pattern for flexible, maintainable warehousing; complex for non-experts (Ch 4).

**DDS (Domain Data Store)** — Consumption-side building block for storing/managing newly created (copied/integrated) data distinct from data products (Ch 11).

**Dead letter queue** — Holding area for undeliverable/malformed event messages requiring manual investigation (Ch 6).

**DevOps/DataOps** — Software delivery practice (DevOps) and its data-focused counterpart (DataOps) for automated, collaborative data pipeline delivery (Ch 1, Ch 12).

**Domain-Driven Design (DDD)** — Eric Evans' methodology using bounded contexts, ubiquitous language, and domains/subdomains to manage complexity (Ch 2).

**DAMA-DMBOK** — Data Management Body of Knowledge; identifies 11 functional areas of data management with governance at the heart (Ch 1).

**ESB (Enterprise Service Bus)** — Centralized SOA integration platform; prone to becoming a monolithic bottleneck (Ch 5).

**Event broker** — Middleware enabling direct, cross-language event communication and routing (Ch 6).

**Event-carried state transfer** — Event pattern where the event contains full state, avoiding a consumer callback (Ch 6).

**Event store** — Database retaining all events for replay/lookup/enrichment; messages are not deleted after consumption (Ch 6).

**Federated computational governance** — Data mesh principle of automated, platform-enforced compliance rather than manual central control (Ch 3).

**Fully federated domain topology** — Data-mesh-style topology with peer-to-peer distribution and no central orchestrator (Ch 3).

**Golden source** — The authoritative application where authentic data for a context originates/is managed (also "system of record") (Ch 2).

**Governed domain topology** — Hub-and-spoke variant where domains own data but route products through a shared distribution layer (Ch 3).

**Knowledge graph** — Graph-structured database of connected entities/relationships, built on semantic web standards (RDF/OWL/SPARQL) (Ch 9).

**Landing zone** — Preconfigured, code-provisioned cloud environment (data management landing zone = domain-agnostic governance; data landing zone = domain workloads) (Ch 3).

**MDM (Master Data Management)** — Discipline for distributing critical, mastered data to ensure consistency/quality/reliability across domains (Ch 1, Ch 10).

**MDM state store** — NoSQL, continuously-updated store supporting low-latency matching for operational MDM use cases (Ch 10).

**Medallion architecture** — Bronze (raw) → Silver (cleaned/validated) → Gold (business-ready, denormalized) data layering (Ch 4).

**Metalake** — Author's term for a unified repository (catalog + knowledge graph) managing all enterprise metadata (Ch 9).

**Metamodel** — A model that manages a set of models; the blueprint defining how metadata entities relate to each other (Ch 9).

**MLOps** — "DevOps for machine learning"; framework for the ML life cycle (data, training, deployment, monitoring) accounting for continuous retraining (Ch 11).

**Output port** — Standardized, read-only, read-optimized interface through which a domain serves a data product (Ch 4).

**Partially federated domain topology** — Centralization on the source side, federation on the consuming side (Ch 3).

**RBAC (Role-Based Access Control)** — Access restricted via user-role-permission assignment; risks "role explosion" at scale (Ch 8).

**Resource-Oriented Architecture (ROA)** — RESTful guideline modeling data as noun-oriented, URI-identified resource collections (Ch 5).

**Service mesh** — Infrastructure layer managing microservice-to-microservice communication, decoupled from external routing (Ch 2, Ch 5).

**Service orchestration vs. choreography** — Centrally coordinated multi-service composition vs. decentralized, autonomous service interaction (Ch 5).

**SOA (Service-Oriented Architecture)** — Architectural style exposing business functionality via (web) services, decoupling applications (Ch 5).

**Trust boundary** — The domain boundary at which security controls tighten because the receiving party lacks the originating context (Ch 8).

**Ubiquitous language** — A shared, formalized vocabulary used within one bounded context; changes when the bounded context changes (Ch 2).

**Value chain–aligned domain topology** — Tight domain alignment around an end-to-end chain of business activities (Ch 3).
