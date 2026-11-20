# Chapter 1: Making Better Decisions Based on Data

## Core Idea
Data collection and analysis exist to support decisions; data science differs from data analytics in that it builds a repeatable, automatable model/algorithm so the same decision can be made systematically, at scale, as a service — and the cloud is what makes "full stack data scientists" (one person doing data prep, analysis, and automation) practical.

## Frameworks Introduced
- **Agile Architecture for Data/AI on GCP**: prefer no-code > low-code > custom code, and ELT over ETL.
  - When to use: default architecture for any new GCP data/ML project.
  - How: land raw data into BigQuery via pre-built connectors (no-code), transform with SQL views (low-code), use AutoML invoked from BigQuery before custom TensorFlow; drop down to Apache Beam/SQL/TensorFlow on serverless infra (Pub/Sub, Dataflow, BigQuery, Vertex AI) only when flexibility is required. The full architecture is a superset of the simple one, so both coexist without maintaining two stacks.
- **Serverless-first service selection**: choose a managed service over self-managed VMs/clusters whenever one exists.
  - When to use: any data processing, storage, or ML workload.
  - How: pick from BigQuery (analytics), Cloud Spanner (databases), Dataflow (pipelines), Pub/Sub (messaging), Bigtable (high-throughput ingest), Cloud Run/Functions (apps/tasks), Vertex AI (ML) — "serverless" = you supply only code, not machine lifecycle management.
- **Probabilistic Decision Framework**: convert a business decision into a CDF threshold.
  - When to use: any decision under uncertainty where historical outcome data exists.
  - How: define decision criterion in terms of a risk-aversion threshold and a probability cutoff (e.g., "cancel if P(arrival delay ≤ 15 min) < 70%"); compute via PDF → integrate → CDF; optionally generalize to expected-value calculations combining probability and cost/benefit.

## Key Concepts
- **Data analytics vs. data science**: analytics answers a single one-off question manually; data science builds a reusable model/algorithm for repeated decisions.
- **Probability density function (PDF)**: the (possibly >1-valued) distribution curve of a continuous variable; must be integrated to yield a probability.
- **Cumulative distribution function (CDF)**: P(X < x), bounded [0,1]; the area under the PDF up to x — easier to reason about than the PDF directly.
- **Elasticity**: the cloud's ability to seamlessly scale compute resources assigned to a job up or down.
- **Full stack data scientist**: a practitioner who can do data preparation, analysis, and automation/engineering end-to-end, enabled by managed cloud services.
- **Agile architecture**: speed of development + flexibility + low maintenance + autoscaling/resiliency.
- **Semantics/governance of data**: capturing what stored data means so it remains usable for future decisions.

## Mental Models
- **Decisions, not dashboards, are the deliverable**: queries, reports, and graphs are not the end goal — a verifiably accurate, repeatable decision is.
- **Principle of Parsimony (Occam's Razor)**: given similar accuracy, prefer the simpler model — fewer assumptions, fewer failure modes, cheaper to implement; maintain multiple candidate models in parallel as insurance.
- **Simple → complex solution path**: always start with the simplest workable approach; let its drawbacks motivate justified complexity (mirrors the book's own chapter progression).
- **Cost-of-elasticity equivalence**: 10 machines × 10 hours ≈ 100 machines × 1 hour in cost — so always prefer finishing faster at equal cost.

## Anti-patterns
- **Writing custom ETL/custom ML models by default**: skips the no-code/low-code options (pre-built connectors, SQL views, AutoML) that are faster and equally capable for most use cases.
- **Treating "the cloud" as merely lift-and-shift to VMs**: doing data prep on cloud VMs (spinning up/managing machines yourself) forfeits the real benefit — true serverless, autoscaling managed services.
- **Ignoring data governance/privacy up front**: collecting data without planning semantics, access control, and retention leads to compliance and usability failures later (e.g., improperly anonymized NYC taxi dataset that was de-anonymized via correlation attacks).
- **Picking a probability threshold without tying it to real cost/benefit**: an arbitrary cutoff (e.g., "30%") is weaker than an expected-value calculation that incorporates monetary upside/downside.

## Code Examples
```bash
git clone \
  https://github.com/GoogleCloudPlatform/data-science-on-gcp
cd data-science-on-gcp
```
- **What it demonstrates**: how to obtain the book's companion repo and chapter-by-chapter README instructions inside GCP Cloud Shell (a free, ephemeral micro-VM with Python, Git, and the Cloud SDK preinstalled).

## Reference Tables
| Need | No-code/Low-code | Full flexibility |
|---|---|---|
| Data ingestion | Pre-built connectors into BigQuery | Dataflow/Apache Beam ETL |
| Transformation | SQL views in BigQuery | Apache Beam pipelines |
| ML modeling | AutoML (from BigQuery) | Custom TensorFlow / Keras on Vertex AI |

| GCP Serverless Service | Primary Use |
|---|---|
| BigQuery | Data analytics / warehouse |
| Cloud Spanner | Globally consistent relational DB |
| Cloud Dataflow | Batch/streaming pipelines (Apache Beam) |
| Cloud Pub/Sub | Messaging / event distribution |
| Cloud Bigtable | High-throughput NoSQL ingest |
| Cloud Run / Functions | Containerized apps / lightweight tasks |
| Vertex AI | End-to-end ML dev/deploy/automation |

## Worked Example
The book's running case study: a traveler's flight is delayed just before departure; should they text the client to cancel a meeting? Decision criterion derived step by step: (1) model time-from-departure-to-arrival-at-client as a random variable with a PDF; (2) define personal risk-aversion buffer = 15 minutes early arrival needed; (3) define cancellation probability threshold = 30% chance of missing that buffer is too much risk, i.e., cancel if CDF(arrival delay ≤ 15 min) < 70%; (4) data source = US Bureau of Transportation Statistics historical flight data (30.6M+ flights, 2015–2019, excluding 2020 due to COVID); (5) platform decision: too large for a laptop, so compute on GCP rather than on-prem Hadoop, specifically using managed services (BigQuery, Dataproc, Dataflow, Vertex AI) instead of self-managed VMs. This flight-delay / meeting-cancellation decision recurs as the dataset and use case for the rest of the book.

## Key Takeaways
1. Design every project around the decision it enables, not around the report or model artifact.
2. Default to no-code/low-code + ELT on GCP; drop to custom code (Beam/SQL/TensorFlow) only when the simpler path is genuinely insufficient.
3. Prefer serverless managed services (BigQuery, Dataflow, Pub/Sub, Bigtable, Vertex AI, Spanner) over self-managed infrastructure for autoscaling and reduced ops burden.
4. Turn a business decision into a probabilistic/CDF threshold so it can be automated and repeated at scale.
5. Favor the simplest model with comparable accuracy (Occam's Razor); keep multiple models maintained in parallel as a hedge.
6. Plan data governance (semantics, privacy, access control, retention) before collection, not after.

## Connects To
- **Ch 2**: The flight-delay dataset introduced here is ingested into the cloud starting in Chapter 2.
- **Ch 12**: The "full dataset" chapter revisits this same probabilistic decision at full scale across all flights.
- **Agile Architecture**: Recurs throughout the book as the organizing principle for tool choice in every later chapter (BigQuery ML vs. Spark vs. TensorFlow, etc.).
