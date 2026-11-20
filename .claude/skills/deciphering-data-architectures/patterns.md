# Architecture Patterns

## Relational Data Warehouse (RDW)
**When to use**: Structured data, need for single version of truth, transaction support, schema enforcement, self-service BI for non-technical users, accurate historical ("as of") reporting.
**How**: Top-down approach — formulate hypotheses → define business requirements → design architecture → build data model → build schema (schema-on-write) → develop ETL → deploy BI tools → test/refine → maintain/expand. Populate via full or incremental extraction (timestamps, CDC, partitioning, triggers, or MERGE).
**Trade-offs**: Strong governance, security, performance for reads, protection against app upgrades — but high cost/complexity, maintenance windows (no 24/7 access), limited flexibility for unstructured/streaming data.

## Data Lake
**When to use**: Raw/unstructured/semi-structured/streaming data, exploratory analysis, ML staging, cheap long-term storage, "just in case" data stockpiling.
**How**: Bottom-up approach — collect first, ask questions later. Layer into raw/bronze → conformed → cleansed/silver → presentation/gold → optional sandbox. Organize folders by time, subject, source, security boundary, retention, etc.
**Trade-offs**: Cheap, flexible, fast ingestion (schema-on-read) — but weaker governance/security by default, risk of becoming a "data swamp" without layering discipline, harder reporting experience for non-technical users.

## Modern Data Warehouse (MDW)
**When to use**: Need both ML/exploratory flexibility (lake) and governed, fast self-service BI (RDW); data volume under ~10TB; organizations comfortable transitioning gently from a pure RDW.
**How**: Five-stage journey — ingestion → storage (lake) → transformation (lake compute engines) → modeling (copy to RDW, usually 3NF or star schema) → visualization (RDW-backed reports/dashboards). Data is deliberately duplicated between lake and RDW — this duplication is the architecture's defining trait.
**Trade-offs**: Best-of-both-worlds flexibility and governance — but added complexity, cost (duplicate storage/compute/pipelines), vendor dependency, skill requirements for both lake and RDW toolchains.

## Data Fabric
**When to use**: Large (>10TB), fast-evolving, multi-source, highly regulated, or real-time-sensitive environments where an MDW's capabilities are insufficient.
**How**: Extend an MDW with ≥3 of 8 features: data access policies, metadata catalog (+lineage), MDM, data virtualization (optional), real-time processing, APIs, reusable services, packaged products.
**Trade-offs**: Scalable, adaptive, strong governance/security — but resource-intensive to build, harder to troubleshoot, overkill for small businesses with simple needs.

## Data Lakehouse
**When to use**: Want a single repository instead of lake+RDW duplication; especially attractive for smaller datasets, serverless/pay-per-query economics, or AI/ML-heavy workloads.
**How**: Add a transactional layer (Delta Lake, Apache Iceberg, or Apache Hudi) on top of data lake storage to get DML support, single-table ACID, time travel, schema enforcement, automatic small-file compaction. Build a relational serving layer (SQL views, Hive tables) on top for end-user usability.
**Trade-offs**: Solves MDW's six pain points (reliability, staleness, limited AI/ML support, TCO, governance risk, complexity) — but sacrifices RDW-level query speed (no native advanced indexing/caching/materialized views), native security features (row/column security, encryption, masking), high concurrency, and tightly-coupled metadata; evaluate trade-offs per dataset, not architecture-wide.

## Data Mesh
**When to use**: Very large, domain-oriented enterprises facing genuine organizational/technical scaling bottlenecks from centralization, with the resources and multi-year time horizon for a major cultural/organizational transformation.
**How**: Apply Dehghani's four principles — domain ownership (decentralize to business domains), data as a product (domain teams build data products exchanged via data contracts), self-serve infrastructure platform (central platform team provides standardized tooling), federated computational governance (central rules, domain-level implementation). Most practical implementations use Mesh Type 1 (shared tech stack, one physical lake logically separated by domain). Adopt gradually via a hub-and-spoke model — extend an existing centralized solution, migrate domains incrementally, never attempt a full one-shot conversion.
**Trade-offs**: Solves ownership/quality/scaling problems of centralization — but is slower and more expensive to build than any other architecture here, creates MORE data lakes/warehouses (not fewer), increases data duplication (conformed dimensions, source copies, third-party data purchases), multiplies security surface area, and requires each domain to staff its own mini-IT team. The author estimates almost no company builds a "pure" data mesh, and only ~1% of organizations meet Dehghani's own readiness criteria.

## EDW Augmentation (MDW Stepping Stone)
**When to use**: An existing on-prem EDW can't handle big data (storage/compute/maintenance-window limits) but you're not ready for a full MDW.
**How**: Add a cloud data lake alongside the unchanged on-prem EDW; new big data goes to the lake, existing EDW data/workflows stay as-is.
**Trade-offs**: Fast to stand up, low disruption — but doesn't offload any EDW workload, requires new compute skills for lake cleaning, existing query tools may not work against the lake.

## Temporary Data Lake + EDW (MDW Stepping Stone)
**When to use**: Need to offload EDW transformation/maintenance-window burden without building a full lake-based architecture.
**How**: Use the lake purely as transient staging/refining; all querying/reporting remains in the EDW.
**Trade-offs**: Reduces EDW maintenance window, cost-effective — but doesn't deliver the lake's other benefits (backup, sandbox, single-source-of-truth); easy evolution path to full MDW.

## All-in-One (MDW Stepping Stone / Lakehouse-adjacent)
**When to use**: Startups, small businesses, or technical-expert-only user bases wanting quick prototyping with minimal architecture overhead.
**How**: Skip the RDW entirely; do all reporting/querying from the data lake.
**Trade-offs**: Fast, simple, agile — but sacrifices performance, security, referential integrity, and user-friendliness for non-technical users; needs an RDW added to become a true MDW stepping stone.

## Polyglot Persistence
**When to use**: An application handles multiple distinct data shapes/use cases (e.g., shopping carts, orders, inventory, social graphs) poorly served by one storage paradigm.
**How**: Use the storage technology best suited per data type within one application — key-value for sessions/carts, document store for orders, RDBMS for structured inventory, graph store for relationship data.
**Trade-offs**: Optimal performance/fit per data type — but added complexity from learning/maintaining multiple storage technologies.

## Lambda Architecture
**When to use**: Need both a trusted historical "single source of truth" (batch) and low-latency "I want it now" views (stream) reconciled in one place — e.g., recommendation engines, fraud detection.
**How**: Dual data model (separate batch + stream/speed layers) feeding a single unified presentation/serving layer that defaults to the trusted batch layer but falls back to the stream layer for up-to-the-second needs.
**Trade-offs**: Handles both batch and real-time well — but adds complexity, has limited support for stateful processing (each event processed independently without customer-journey context).

## Kappa Architecture
**When to use**: Need real-time-only processing at scale without batch complexity — e.g., streaming platforms, financial trading systems.
**How**: Single event stream + stateless processing + real-time-only presentation layer; no batch layer.
**Trade-offs**: Simpler than Lambda for pure real-time needs — but weak support for batch processing of historical data and ad hoc queries over history.
