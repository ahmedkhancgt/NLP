# Chapter 5: Services and API Management

## Core Idea
Most organizations actually need a "data and integration mesh" — APIs (real-time, synchronous, operational) and data products (asynchronous, analytical) should follow the same domain-ownership principles; legacy SOA/ESB centralization recreates the same "big ball of mud" problems as enterprise data warehousing, so API responsibility must be federated back to domains using lightweight API gateways rather than monolithic ESBs.

## Frameworks Introduced
- **Federated Responsibility Model for APIs** (author's extension of data mesh to APIs): Business logic, process execution, and context validation stay in the owning domain; the ESB/central integration layer is used only for legacy wrapping, not as the default path.
  - When to use: Whenever designing cross-domain API communication at enterprise scale.
  - How: Expose via REST resource model; keep domain logic in domain; allow orchestration/aggregation only within domain boundaries; use domain ubiquitous language; register in a central API/service repository.
- **Resource-Oriented Architecture (ROA)**: RESTful guideline modeling data as noun-oriented, URI-identified, homogeneous resource collections (vs. SOA's verb-orientation).
  - When to use: Designing API schemas for domain-to-domain consumption, mirroring data product design principles.
  - How: Group by subject area into resources/collections; make each self-discoverable via URI; apply CRUD-style operations (POST/GET/PUT or PATCH/DELETE).
- **API as a Product**: Treat APIs with the same productized ownership discipline as data products.
  - When to use: Any API exposed across a domain boundary.
  - How: Push integration/transformation logic to consumers for composite calls; own API contracts (OpenAPI spec); register in a service catalog/developer portal; separate internal (technical) vs. domain-to-domain (stable, self-explanatory) API design standards.

## Key Concepts
- **Service-Oriented Architecture (SOA)**: An architectural style exposing business functionality via (web) services, decoupling applications and hiding internal complexity (Open Group definition).
- **Enterprise Service Bus (ESB)**: A centralized integration platform (SOAP/XML era) providing orchestration, rule-based routing, format transformation, and adapters — prone to becoming a monolithic bottleneck and even an unintended application state store.
- **Service orchestration**: Centrally coordinating/combining multiple services into a composite flow (via ESB or BPM); contrasts with **service choreography**, where decision logic is distributed and no central party controls the interaction.
- **Service aggregation / composite services**: Combining multiple services into one exposed service; best practice is to push integration/transformation logic to the consuming domain rather than the central layer.
- **API gateway**: A lightweight component (vs. heavyweight ESB) providing routing, security, monitoring, and developer-portal documentation without embedding business logic.
- **Service mesh**: Microservice-to-microservice communication control plane/data plane, analogous to but distinct from an API gateway (internal vs. externally-routed communication).
- **Public (business) services vs. private (technical) services**: Business services deliver meaningful cross-boundary functionality; private/technical services are internal and shouldn't be published broadly.
- **Canonical data model**: A unified, standardized schema meant to align many service models — prone to the same "monster interface" bloat problem as enterprise data warehouses when applied enterprise-wide.
- **Experience APIs / Backend for Frontend**: APIs optimized for human/channel interaction (web, mobile, chatbot) rather than system-to-system integration, requiring caching, state, and UX-specific components.

## Mental Models
- **ESB ≈ EDW**: Centralized ESBs replicate the same failure modes as centralized enterprise data warehouses — bloated canonical models, layering-on-layering complexity, and (worst of all) accidentally becoming a stateful data store.
- **SOA and data mesh share DNA**: Both advocate loose coupling, domain language, product thinking, and federated autonomy — API management best practices should mirror data product management best practices (Ch. 4) because the same underlying application database feeds both.
- **Internal vs. external communication split**: Because external partners (regulatory formats like PSD2, trust/security needs, commercial throttling) cannot be controlled the way internal domains can, an additional external-facing API layer is usually required, separate from internal domain-to-domain APIs.

## Anti-patterns
- **Enterprise-wide canonical data model for APIs**: Creates a "monster interface model" with countless optional attributes that satisfies everyone partially and nobody fully — mirrors the EDW single-source-of-truth failure.
- **ESB as orchestrator for everything**: Makes the ESB a single point of failure and obscures true service ownership when it aggregates services from multiple owners.
- **ESB used as an application state/data store**: Violates the principle that the ESB should be stateless (aside from caching/session management); risks total process-state loss if the ESB goes down.
- **Building composite services on top of other composite services**: Creates a chaotic, untraceable web of nested API calls.
- **Full domain tooling autonomy without central registries**: Makes it impossible to track which services belong to which domains or trace dependencies across a distributed landscape.
- **Mixing technical orchestration with process/business orchestration** (or storing application data inside BPM process layers): BPM software must manage process state/sequencing only, never customer/product/contract data.

## Reference Tables
| Pattern | Centralization | Risk |
|---|---|---|
| ESB-based orchestration | High | Single point of failure, obscured ownership, state-store misuse |
| Consumer-side orchestration | Low | Each consumer reimplements combination logic |
| Service choreography | None (decentralized) | No party can guarantee overall interaction integrity |
| API gateway (federated) | Low-medium | Requires domain discipline + central registry |

| Integration Component | Role |
|---|---|
| API gateway | Lightweight routing, security, monitoring; no heavy orchestration |
| ESB | Legacy modernization wrapper, heavy transformation/orchestration only when unavoidable |
| Service mesh | Internal microservice-to-microservice communication control |
| GraphQL | Federated schema query layer across multiple backend APIs |
| BPM software | Long-running, stateful business process orchestration (not data storage) |

## Worked Example
A six-domain API architecture (A–F): Domains A/B/C are providers, each exposing APIs via their own governed API gateway. Domain D builds a composite API combining multiple provider calls into one, consumed by its own web app. Domain E uses an external API gateway layer to translate/secure APIs for outside consumers (e.g., PSD2-style regulatory formatting). Domain F is a rich "digital experience" domain using a microservices design with a cache for session state, functions that call other domains' APIs and store enriched results, and a GraphQL service to narrow/aggregate requests across domains — illustrating how composite, external, and experience API patterns coexist under one governance umbrella.

## Key Takeaways
1. Treat APIs and data products as two distribution patterns serving the same underlying domain data — apply the same ownership, ubiquitous-language, and read-optimization principles to both.
2. Federate API responsibility to domains; reserve the ESB strictly for legacy wrapping, not as the default integration layer.
3. Push composite/aggregation logic to the consuming domain wherever possible rather than centralizing it, to preserve clear ownership and avoid single points of failure.
4. Separate internal/technical APIs from domain-to-domain (stable, self-explanatory, read-optimized) APIs with different design and governance standards.
5. Split internal and external (ecosystem/public) API communication into separate layers due to differing trust, regulatory, and commercial requirements.
6. Register every API/service centrally (OpenAPI, service registry) regardless of how many gateways or meshes are deployed, to retain enterprise-wide visibility.
7. APIs can serve as the real-time read path for CQRS-based data products — route eventually-consistent reads to the read store and commands to the command store transparently.

## Connects To
- **Ch 2**: Reapplies domain/bounded-context and ubiquitous-language principles to API design.
- **Ch 4**: Direct parallel to data product design principles (CQRS, read-optimization, no raw data, don't conform to consumers).
- **Ch 6**: Hands off to asynchronous/event-driven communication as the complementary integration style.
- **Ch 1**: Resolves the operational/analytical integration trend via APIs as the real-time bridge.
- **Service-Oriented Architecture / REST (Fielding)**: Core external frameworks this chapter builds its principles on.
