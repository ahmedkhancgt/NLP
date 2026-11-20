---
name: deciphering-data-architectures
description: "Knowledge base from \"Deciphering Data Architectures\" by James Serra. Use when choosing or designing a data architecture — relational data warehouses, data lakes, modern data warehouses, data fabric, data lakehouse, data mesh, architecture design sessions, data modeling approaches, data ingestion approaches, or the people/process/technology sides of data architecture adoption."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Deciphering Data Architectures
**Author**: James Serra | **Pages**: ~278 | **Chapters**: 16 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch13`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Six Vs of Big Data**: Volume, Variety, Velocity, Veracity, Variability, Value (Ch1) — the framework for scoping what "big data" means beyond mere size.

**Enterprise Data Maturity Stages**: Reactive (spreadmarts) → Informative (centralized, batch) → Predictive (cloud, ML, real-time decisions) → Transformative (any data, self-service) (Ch1).

**Schema-on-write vs. schema-on-read**: the foundational fork — relational databases/warehouses enforce structure at ingestion; data lakes defer it to read time (Ch2, Ch5).

**Six Architecture Types** (in order of appearance): Relational Data Warehouse (1984) → Data Lake (2010) → Modern Data Warehouse (2011) → Data Mesh (2019) → Data Fabric (2016) → Data Lakehouse (2020). Each responds to limitations of its predecessor rather than replacing it outright (Ch2).

**Architecture Design Session (ADS)**: the structured discovery/whiteboarding workshop that produces an architecture blueprint + action plan. Discovery (vendor-neutral questions) ALWAYS precedes solution talk — "stay away from the solution" until questions are done (Ch3).

**Top-down vs. bottom-up**: RDW/Inmon = top-down (plan the enterprise vision, then build); Data lake/Kimball = bottom-up (collect first, surface hypotheses). Note: these terms mean different things in Ch4/5 (RDW/lake methodology) vs. Ch8 (Inmon/Kimball methodology) — don't conflate them (Ch4, Ch5, Ch8).

**Modern Data Warehouse (MDW) defining trait**: at least some data is deliberately DUPLICATED from lake to RDW. Remove that duplication and you get a lakehouse instead (Ch10). Five-stage data journey: ingestion → storage → transformation → modeling → visualization (Ch10, reused in Ch11, Ch12).

**Data Fabric**: MDW + ≥3 of 8 features (data access policies, metadata catalog, MDM, data virtualization [optional], real-time processing, APIs, services, products) — no bright line, author's own heuristic (Ch11).

**Data Lakehouse**: single repository via a transactional layer (Delta Lake / Apache Iceberg / Apache Hudi) added to a data lake — solves MDW's reliability/staleness/TCO/governance problems at the cost of RDW-native performance/security/concurrency features. Evaluate the RDW-skip decision per dataset, not architecture-wide (Ch12).

**Data Mesh — Dehghani's Four Principles** (Ch13): (1) Domain Ownership — decentralize to business domains; (2) Data as a Product — domain teams build "data products" exchanged via data contracts; (3) Self-Serve Infrastructure Platform — a central platform team provides shared tooling; (4) Federated Computational Governance — central rules, domain-level implementation. **Critical rule**: domain ownership alone (principle #1) is NOT a data mesh — it's data federation. You need principle #2 to realize the mesh's real benefits.

**Data Mesh — the author's skepticism** (Ch13, Ch14): almost no company builds a "pure" mesh; most DW project failures are people/process issues, not scalability limits; data mesh takes LONGER than any other architecture here, not less. Five myths to reject: silver bullet, replaces lake/warehouse, fixes DW "failures," decentralizes everything, virtualization-alone = mesh. Adopt only via gradual hub-and-spoke extension of an existing centralized solution — never a one-shot conversion.

**Architecture Selection (end-of-book guide)**: MDW = small data, gentle transition. Data fabric = diverse multi-source integration need. Lakehouse = flexible default until you hit its limits. Data mesh = very large domain-oriented orgs with resources/time for a multi-year transformation. None are mutually exclusive — most real solutions blend features.

**Project success depends on people/process more than technology** (Ch15): staff the right roles (data architect, engineer, steward, DBA, analyst, scientist, business analyst, PM, privacy/governance/quality managers — plus domain teams + platform teams for a mesh); avoid the enumerated pitfall catalog (unrealistic timelines, wrong tech, requirement gathering extremes, unvalidated reports, bad consultants, mid-project budget cuts, fixed-deadline-backward planning, source-mirrored modeling, poor IT/business communication).

**Technology choices** (Ch16): OSS vs. CSP (usually blended, not exclusive); on-prem only for specific exceptions (connectivity, latency, contracts, sensitivity); PaaS preferred for data warehouses; multi-cloud justified mainly by data sovereignty, not negotiating leverage or redundancy (the latter two are largely myths).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| 1 | Big Data | Six Vs, Enterprise Data Maturity Stages, Self-Service BI |
| 2 | Types of Data Architectures | Schema-on-write/read, six architecture types overview |
| 3 | The Architecture Design Session | ADS structure, ADS Questionnaire, "thinking big, starting small" |
| 4 | The Relational Data Warehouse | Top-down approach, SVOT, ETL, extraction methods, RDW anti-patterns |
| 5 | Data Lake | Bottom-up approach, data lake layering, folder structure, multiple data lakes decision |
| 6 | Data Storage Solutions and Processes | Data marts, ODS, data hubs, MDM, data virtualization, data catalogs, marketplaces |
| 7 | Approaches to Design | OLTP/OLAP, SMP/MPP, Lambda/Kappa architecture, polyglot persistence |
| 8 | Approaches to Data Modeling | Relational vs. dimensional modeling, CDM, Data Vault, Kimball vs. Inmon |
| 9 | Approaches to Data Ingestion | ETL vs. ELT, reverse ETL, batch vs. real-time, data governance |
| 10 | The Modern Data Warehouse | MDW five-stage journey, stepping-stone architectures (EDW augmentation, temp lake+EDW, all-in-one) |
| 11 | Data Fabric | Eight data fabric features, MDW→fabric transition drivers |
| 12 | Data Lakehouse | Delta Lake features, six problems solved, relational serving layer, RDW-skip decision |
| 13 | Data Mesh Foundation | Dehghani's four principles, domain types, mesh topologies, mesh vs. fabric |
| 14 | Should You Adopt Data Mesh? | Five myths, concerns catalog, readiness assessment, hub-and-spoke adoption |
| 15 | People and Processes | Team role taxonomies (MDW/fabric/lakehouse vs. mesh), project pitfall catalog, success tips |
| 16 | Technologies | OSS vs. CSP, on-prem vs. cloud, IaaS/PaaS/SaaS, multi-cloud, Hadoop/Databricks/Snowflake |

## Topic Index
- **ADS (Architecture Design Session)** → Ch3
- **Batch vs. real-time processing** → Ch9, Ch16
- **Common Data Model (CDM)** → Ch8
- **Data Catalog / Data Marketplace** → Ch6
- **Data Fabric** → Ch2, Ch11
- **Data Lake** → Ch2, Ch5, Ch16 (Hadoop/HDFS)
- **Data Lakehouse** → Ch2, Ch12, Ch16 (Databricks)
- **Data Mesh** → Ch2, Ch13, Ch14
- **Data Mesh myths/concerns** → Ch14
- **Data Modeling (relational/dimensional)** → Ch8
- **Data Vault** → Ch8
- **Data Virtualization / Federation** → Ch6, Ch11, Ch14
- **ETL / ELT / Reverse ETL** → Ch9
- **Hadoop** → Ch16
- **Inmon vs. Kimball** → Ch8, Ch14
- **Kappa / Lambda architecture** → Ch7
- **Master Data Management (MDM)** → Ch6, Ch11
- **Modern Data Warehouse (MDW)** → Ch2, Ch10
- **Multi-cloud** → Ch16
- **OLTP / OLAP** → Ch7
- **Operational Data Store (ODS)** → Ch6
- **People/process/team roles** → Ch15
- **Relational Data Warehouse (RDW)** → Ch2, Ch4
- **SMP / MPP** → Ch7
- **Snowflake / Databricks** → Ch16
- **Star schema / Dimensional modeling** → Ch8

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — architecture patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 16 chapters, synthesized as structure, not verbatim text.
