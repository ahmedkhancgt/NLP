# Chapter 11: Turning Data into Value

## Core Idea
The consuming side of the architecture is fundamentally different from the providing side and must be managed separately: data products exist to serve stable, reusable data, while Domain Data Stores (DDSs) exist to consume, integrate, and transform that data into use-case-specific value via BI and MLOps — and the two concerns (providing vs. consuming) should use different blueprints to avoid tight coupling.

## Frameworks Introduced
- **Domain Data Store (DDS)** (author's new building block): The consumption-side counterpart to a data product, used when direct read consumption of data products isn't sufficient (performance limits, need to combine/harmonize multiple products, need to persist newly created insights).
  - When to use: Whenever consumed data needs structural transformation, combination across domains, or persistence of new business insights.
  - How: Classify managed data as Copied / Integrated / (newly created) Data Product data; DDSs can act as consumer, provider, or both; align DDS boundaries with business capabilities/value streams where possible, but allow larger domains to span multiple DDSs or vice versa.
- **Managed vs. Self-Service Data principle**: Clear separation between stable, governed, production data and temporary, exploratory, ad hoc data.
  - When to use: Structuring any BI/analytics environment.
  - How: Managed data flows one-way into self-service (never the reverse); provision a separate "workspace" layer (beyond Bronze/Silver/Gold) for self-service discovery, marked non-certified.
- **MLOps Reference Process** (author's end-to-end framework): Initiate → Experiment & Track → Data Engineering → Model Operationalization, each with defined deliverables.
  - When to use: Any machine learning project moving from exploration to production.
  - How: Define success criteria and repo templates (Initiate); track parameters/metrics/artifacts with tools like MLFlow and choose integration pattern — API, batch, or stream (Experiment); build automated retraining pipelines and feature stores (Data Engineering); parameterize, test, monitor, and integrate (Operationalization).

## Key Concepts
- **Direct data consumption pattern**: Reading data products directly at query time without copying/persisting — preferred when performance and single-source needs allow it.
- **Copied data**: Data with the same semantics as the source data product, possibly with syntactic (not semantic) changes.
- **Integrated data**: Data consumed, combined, and transformed into a new context — specific to the consuming use case, never shared directly with other domains.
- **(Newly created) data product data**: Integrated data that, after going through DDS transformations, is published onward following the same principles as regular data products (Ch.4).
- **Shared kernel** (DDD term): Part of a domain model deliberately shared between teams/subdomains to reduce duplication — used when decomposing/grouping DDS boundaries.
- **Semantic layer**: A business-friendly abstraction over DDS data (predefined joins, consistent naming, caching) built either as proprietary in-memory caching or as a federated query layer over the underlying store.
- **MLOps**: "DevOps for machine learning" — collaboration framework addressing data collection, data validation, model training, deployment, monitoring, and explainability, accounting for the fact that ML models (unlike static software) require continuous retraining on fresh data.
- **Feature store**: A centralized repository standardizing feature definition/storage/reuse for model training — the author is skeptical of fully centralized feature stores, favoring domain-aligned alternatives to avoid unnecessary complexity.
- **Model deployment patterns**: Model as API (synchronous request/response), Model as batches in/out, Model as stream (reactive, incremental).

## Mental Models
- **"It takes a village" / providing vs. consuming asymmetry**: The providing side can be standardized because it's source-system aligned; the consuming side resists standardization because every business use case has unique requirements, tools, and audiences — this asymmetry should drive architecture decisions (standardize providing, allow diversity consuming).
- **80/20 rule for data science**: 80% of a data scientist's time goes to finding, cleansing, and organizing data; only 20% to actual model development — architecture should optimize for reducing that 80%.
- **Classify data before deciding how to govern it**: The same dataset's governance rules differ completely depending on whether it's classified as copied, integrated, or newly-created data-product data — misclassifying it leads to wrong distribution/ownership decisions.

## Anti-patterns
- **Business users self-provisioning point-to-point tools and pipelines around governance**: Leads to architecture that becomes "complex, expensive, and difficult to manage" — provide controlled, self-service environments instead of letting shadow IT emerge.
- **Calling consumer-side transformed data "data products"**: Misleading — consumer-side integrated data lacks the stability, decoupling, and ubiquitous-language inheritance that define true data products; conflating the terms causes governance confusion.
- **Combining DDS and data product architectures without a metamodel to track coupling**: Makes it hard to distinguish newly created data from copied data, increasing coupling risk.
- **Letting self-service results flow back into managed production environments directly**: Forbidden — managed data must never depend on human intervention; the self-service-to-managed handover requires a formal engineering process.
- **Centralized feature stores applied indiscriminately**: The author flags this as adding "unnecessary complexity" to the architecture — prefer domain-aligned alternatives unless genuinely justified.
- **Treating MLOps like standard DevOps**: Software is static; ML data is constantly refreshed, requiring continuous retraining/recalibration/redeployment pipelines, not simple CI/CD.

## Reference Tables
| DDS Data Classification | Shareable With Other Domains? | Governance Note |
|---|---|---|
| Copied data | Limited (same semantics as source) | Lineage tracking still required |
| Integrated data | No — tightly coupled to use case | Needs owner approval of originating data before any sharing |
| New data product data | Yes, as a new data product | Must follow full Ch.4 data product principles |

| Store Type | Purpose |
|---|---|
| Analytical data stores | ML workloads, flattened/denormalized data |
| Reporting stores | BI/dimensional reporting (star/3NF schema) |
| Aggregate stores | Overlapping domain requirements, shared business logic |
| Operational consumers | OLTP-like, consumes+integrates from other domains |
| Data warehouses/lakes | Large historical, cross-domain combination |

| Model Deployment Pattern | Use Case |
|---|---|
| Model as API | Real-time prediction requests |
| Model as batches in/out | Scheduled bulk scoring |
| Model as stream | Reactive, incremental event-driven scoring |

## Worked Example
A large domain spanning multiple subdomains uses a medallion-architecture DDS where Bronze and Silver zones (cleansing, historization) are shared across all subdomains, but the Gold zone branches into subdomain-specific pipelines and transformations — illustrating how DDS boundaries can be more fine-grained than domain boundaries at the point where business logic genuinely diverges, while still sharing common upstream processing to avoid duplication.

## Key Takeaways
1. Never conflate data products (stable, decoupled, built for reuse) with DDS-integrated data (tightly coupled, use-case-specific, inherently unstable) — classify explicitly as copied/integrated/new-data-product data.
2. Default to direct consumption of data products; only build a DDS when performance limits, multi-domain combination needs, or new-insight persistence genuinely require it.
3. Use two separate blueprints — one for data product creation, one for turning data into value — to prevent tight coupling between providing and consuming concerns.
4. Enforce a strict one-way flow from managed to self-service data, never the reverse; use a dedicated non-certified workspace layer for self-service discovery.
5. Build MLOps around the four-phase reference process (Initiate, Experiment/Track, Data Engineering, Operationalize), tracking every experiment's parameters/metrics for reproducibility and regulatory defensibility.
6. Choose database/data-store technology deliberately per use case (structure, query predictability, consistency needs, volume/velocity) — there is no universal best choice.
7. Treat business requirements, target audience/roles (domain owner, SME, data engineer, data scientist, report builder), and nonfunctional requirements as the three pillars of any consumption-side design.

## Connects To
- **Ch 2**: Builds on consumer-aligned domain concepts.
- **Ch 4**: Directly contrasts DDS/integrated data against the data product principles established there.
- **Ch 6**: Real-time ingestion for DDSs reuses event-driven architecture patterns.
- **Ch 8**: Governance model (roles, RACI) extended to BI/ML personas.
- **Ch 10**: Aggregate stores and overlapping domain data connect to MDM/curation patterns.
- **Ch 12**: Hands off into the practical, phased implementation roadmap.
