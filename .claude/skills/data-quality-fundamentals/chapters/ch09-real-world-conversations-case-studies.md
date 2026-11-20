# Chapter 9: Data Quality in the Real World: Conversations and Case Studies

## Core Idea
Data quality's real-world success depends on five interlocking forces — the data mesh, cloud-stack maturity, metadata/lineage applied to actual use cases, data discovery for distributed architectures, and recognizing the right moment to invest — illustrated concretely through Kolibri Games' five-year data evolution and seven leading indicators for when to prioritize data quality.

## Frameworks Introduced
- **Data Mesh Score**: a scoring heuristic summing quantity of data sources, size of data team, number of data domains, data engineering bottleneck severity (1-10), and data governance priority (1-10).
  - When to use: deciding whether your organization should invest in data mesh architecture.
  - How: score 1-15 = likely don't need a mesh; 15-30 = adopt mesh best practices now to ease future migration; 30+ = you're in the "data mesh sweet spot."
- **Four Defining Elements of a True Data Mesh** (per Zhamak Dehghani): (1) distributing data ownership to domain teams, (2) giving those teams long-term accountability + product thinking, (3) empowering teams with self-serve infrastructure, (4) addressing governance via a new federated model.
  - When to use: validating whether an architecture claiming to be a "data mesh" actually is one (vs. being mere data virtualization).
- **Seven Leading Indicators for Investing in Data Quality**: recent cloud migration; data stack scaling (sources/tables/complexity, rule of thumb >50 tables); growing data team; team spending ≥30% of time firefighting; more data consumers than a year ago; moving to self-service analytics; data is part of the customer value proposition.
  - When to use: deciding "when" (not just "whether") to invest in data quality tooling and process.
- **Metadata-Needs-a-Use-Case principle**: metadata (including lineage) is valueless without application to a specific business problem — "lineage without context is just eye candy."
  - When to use: as a check before investing further in metadata/lineage tooling — ask "what purpose is this metadata serving, and for whom?"

## Key Concepts
- **Data mesh** (recap): domain-oriented, self-serve data architecture with federated governance; composed of data sources, data infrastructure, and domain-owned pipelines, connected by a universal interoperability layer.
- **Data virtualization**: an approach exposing transactional/operational databases "as-is" for analytical use — explicitly NOT the same as a data mesh (exposing an OLTP-optimized store for analytics use cases is "probably a bad idea" per Dehghani).
- **Federated governance**: domains own their pipelines/products but adhere to a shared, universal set of standards (formatting, discoverability, metadata, SLAs) for cross-domain interoperability.
- **Data discovery (federated catalog)**: an automated, real-time alternative to manual data catalogs, suited to distributed/lake/mesh architectures where traditional catalogs "drown."
- **Table sprawl**: uncontrolled proliferation of tables/connectors (e.g., from automated ingestion tools like Fivetran) that outpaces manual tracking — a concrete trigger for investing in data observability.
- **"You're Using That Table?!" problem**: as platforms scale, analysts lose track of which tables are actively maintained vs. obsolete — addressed via data certification and end-to-end lineage.

## Mental Models
- A data mesh is a socio-technical shift, not a product you buy — "you couldn't build a microservice architecture with just a database... you wouldn't build a data mesh with just a data warehouse or BI tool."
- Metadata/lineage is like "an elephant riding a bicycle" without a use case: impressive, but not useful unless tied to a concrete business pain point (impact analysis, RCA, or stakeholder communication).
- Building a data-driven culture is "a marathon, not a sprint" — António Fitas's Kolibri Games journey shows multi-year, iterative tech and team evolution, not a single platform decision.
- The irony of growth: the most data-driven organizations have the most data consumers, meaning both the highest error-detection capacity and the highest risk exposure from bad data simultaneously.

## Anti-patterns
- **Treating data mesh as a single technology purchase**: Dehghani explicitly rejects this; a mesh requires organizational and cultural shifts plus multiple interoperating technologies.
- **Letting each data product team run its own separate, duplicated data store**: massively increases cost and data quality risk; a shared storage layer with independent schemas/access/tenancy is preferred.
- **Collecting metadata/lineage for its own sake**: "great for a demo with executives" but worthless without being tied to impact analysis, RCA, or stakeholder communication.
- **Relying on manual data catalogs in a data lake or mesh**: catalogs can't keep pace with distributed, schema-less, rapidly evolving data — leading to the "data swamp" problem recurring at scale.
- **Underinvesting in data quality during early-stage platform building**: many orgs grow their data platform investment without a corresponding increase in trust/adoption — if the data isn't trusted, the platform investment is wasted.
- **Delaying analyst hires in favor of infrastructure-only growth**: António Fitas explicitly flags "we should have hired analysts earlier" as a retrospective mistake at Kolibri.

## Reference Tables
| Data Mesh Score Range | Recommendation |
|---|---|
| 1-15 | Likely don't need a data mesh given size/unidimensionality |
| 15-30 | Adopt data mesh best practices now; a later migration will be easier |
| 30+ | In the "data mesh sweet spot" — mesh will support ongoing/future initiatives |

| Seven Indicators to Invest in Data Quality | Concrete Trigger |
|---|---|
| Recent cloud migration | Must re-establish trust in new platform while maintaining speed |
| Scaling data stack | Rule of thumb: >50 tables, or high severity even with fewer |
| Growing data team | Risk of "You're Using That Table?!" and tacit-knowledge silos |
| ≥30% time firefighting | Direct signal of reactive-stage data reliability maturity |
| More data consumers than a year ago | Higher exposure to bad-data impact despite more error-spotters |
| Moving to self-service analytics | Stakeholders must trust unmediated data access |
| Data is part of the customer value proposition | Data stack must meet product-grade reliability/uptime standards |

## Worked Example
**Kolibri Games** (Berlin-based mobile game studio, acquired by Ubisoft in 2020; handling 100M events/day across 40 event types): a five-year data evolution under former Head of Data Engineering António Fitas. (1) **2016**: lean startup relying entirely on third-party tools (Facebook Analytics, Firebase, GameAnalytics) — data quality wasn't prioritized, leading to scattered analytics, no KPI transparency, and reporting inconsistencies. (2) **2017**: added AppsFlyer for performance marketing attribution, but lacked version control and had "basically blindfolded" bidding operations. (3) **2018**: centralized onto Azure (Data Factory, Event Hubs, Stream Analytics) and migrated Power BI → Looker, but the SQL database became an unpredictable bottleneck with jobs failing and limited alerting. (4) **2019**: replaced some Azure services with Databricks, then replaced the SQL database with Snowflake after finding Spark/Looker performance lacking; added a head of game data and BI developers — but decisions were still intuition-driven, not data-driven. (5) **2020** (post-Ubisoft acquisition): introduced data-specific SLAs, moved from ETL to ELT directly in Snowflake, added dbt for transformation collaboration and Airflow for orchestration; set concrete KPIs (90% of decisions data-backed, time-to-insight <1 hour for 90% of questions). (6) **2021**: pursued full data mesh architecture — domain cross-functional teams, increased testing, monitoring/alerting, and end-to-end lineage investment to speed troubleshooting. António's five key takeaways: building your own stack pays off; know when to switch technology; data observability is essential to establishing trust; get the basics right before advanced applications (hire analysts earlier); and a data-driven culture matters even more than the tech stack itself.

## Key Takeaways
1. Use the Data Mesh Score heuristic to decide whether (and when) your organization is ready for domain-oriented, federated data architecture — don't adopt it just because it's trending.
2. Remember a data mesh is an organizational/socio-technical paradigm, not a product — expect to combine multiple technologies (open source and SaaS) under a shared governance layer.
3. Never invest in lineage or metadata tooling without a concrete use case (impact analysis, RCA, or stakeholder communication) in mind — metadata alone is "eye candy."
4. For lake/mesh architectures, prefer automated data discovery over manual data catalogs — manual catalogs don't scale to distributed, schema-less environments.
5. Use the seven leading indicators (cloud migration, stack scaling, team growth, firefighting time, consumer growth, self-service shift, customer-facing data) as concrete triggers for prioritizing data quality investment, rather than guessing at timing.
6. Expect a multi-year, iterative platform and culture evolution (per Kolibri Games) — data reliability at scale is rarely achieved via a single tooling decision.

## Connects To
- **Ch 1**: Revisits and operationalizes the data mesh concept first introduced there as an industry trend.
- **Ch 2**: Builds directly on the data catalog vs. data discovery distinction established there, applying it specifically to lake/mesh contexts.
- **Ch 5 / Ch 8**: The "30% of time firefighting" and SLA/SLO frameworks recur as concrete decision triggers here.
- **Blinkist, Fox, Toast** (Ch 5/7/8): referenced again as companies that successfully applied these principles, reinforcing the book's cross-chapter case-study thread.
- **DDD (Domain-Driven Design, Eric Evans) / Microservices**: the explicit architectural analogy underlying the entire data mesh discussion.
