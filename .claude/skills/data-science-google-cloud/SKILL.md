---
name: data-science-google-cloud
description: "Knowledge base from \"Data Science on the Google Cloud Platform\" (2nd ed.) by Valliappa Lakshmanan. Use when building data science/ML pipelines on GCP — data ingestion, dashboards, streaming with Pub/Sub and Dataflow, interactive exploration with Vertex AI Workbench, Spark on Dataproc, BigQuery ML, TensorFlow on Vertex AI, MLOps, or real-time/time-windowed ML features."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Science on the Google Cloud Platform
**Author**: Valliappa (Lak) Lakshmanan | **Pages**: ~462 | **Chapters**: 12 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch08`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The running case study**: every chapter builds the same decision problem — using US BTS flight data to decide whether a traveler should cancel a client meeting, given a 70%/30% probabilistic threshold on arrival delay (Ch1). This case study is the thread tying ingestion → dashboards → streaming → EDA → modeling → MLOps → real-time serving together.

**Agile architecture for GCP data/AI**: prefer no-code > low-code > custom code, and ELT over ETL. Land raw data into BigQuery via connectors, transform via SQL views, try AutoML before custom TensorFlow, and drop to Apache Beam/SQL/TensorFlow only when needed — all on serverless infra (Pub/Sub, Dataflow, BigQuery, Vertex AI). (Ch1)

**Serverless-first**: choose a managed service over self-managed VMs/clusters whenever one exists — BigQuery, Dataflow, Cloud Run, serverless Spark, Vertex AI. "Serverless" means you supply only code, never manage machine lifecycle. (Ch1, Ch2, Ch6)

**Scaling decision rule**: scale up (bigger machine) for jobs that re-read data repeatedly; scale out with sharded data (Dataproc/HDFS) when no high-bandwidth data center is available; scale out with data-in-place (BigQuery/Dataflow/Vertex AI directly on Cloud Storage) is GCP's default advantage, thanks to petabit-class data-center bisection bandwidth. (Ch2, Ch6)

**Causality constraint**: a model input must be knowable at the moment the prediction is needed. Taxi-in duration, actual flight distance, and future time-windowed averages are classic violations. Recurs at every modeling chapter (Ch2, Ch4, Ch8, Ch11).

**Training-serving skew**: a feature must be computed identically in training and at serving time. Solved via BigQuery ML's `TRANSFORM` clause (Ch8) and shared Apache Beam transform modules used by both batch and streaming pipelines (Ch11).

**Date-hash train/test split**: never randomly split correlated rows (e.g., same-day flights share weather). Hash the grouping key (`FARM_FINGERPRINT`) to deterministically assign TRAIN/VALIDATE/TEST; persist the split as a reusable table/column. (Ch5, Ch8, Ch9, Ch10, Ch12)

**Frequency-based outlier filtering**: on large datasets, filter groups by sample count (`HAVING count > N`), not by raw value range — rare-but-frequent-enough values carry real signal; classic small-data outlier rules (μ±3σ) throw away the long tail that matters at scale. (Ch5)

**Model progression across the book**: simple threshold (Ch3) → empirical percentile model (Ch5) → Bayesian binning (Ch6) → Spark logistic regression (Ch7) → BigQuery ML linear/XGBoost/AutoML (Ch8) → TensorFlow wide-and-deep (Ch9) → MLOps-automated + tuned + explainable (Ch10) → real-time streaming features (Ch11) → full 4-year dataset with continuous training (Ch12). Each step is motivated by a concrete limitation of the previous one — follow this progression when deciding how much modeling complexity a new problem actually needs.

**"More data and better features beat a fancier model"**: repeated explicitly in Ch8–Ch9 — hyperparameter tuning and model-architecture upgrades (DNN vs. linear) often move the needle far less than well-chosen domain features (time-of-day, geographic bucketing/crossing, time-windowed averages).

**MLOps separation of concerns**: convert notebook prototypes to plain `.py` files with a single orchestrating entry point; read Vertex AI's injected `AIP_*` environment variables for data paths; never deploy a notebook to production. (Ch10)

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Making Better Decisions Based on Data | Probabilistic decision framework, CDF/PDF, Agile Architecture, serverless-first |
| ch02 | Ingesting Data into the Cloud | Hub-and-spoke/ELT, scaling up/out/in-place, Cloud Run + Scheduler ingestion, causality |
| ch03 | Creating Compelling Dashboards | Dashboard-first development, contingency table threshold optimization, Data Studio |
| ch04 | Streaming Data: Pub/Sub and Dataflow | GCP streaming reference architecture, Beam unified batch/stream, sliding-window aggregation |
| ch05 | Interactive Data Exploration with Vertex AI Workbench | BigQuery+Jupyter EDA, frequency-based outlier filtering, date-hash split, empirical percentiles |
| ch06 | Bayesian Classifier with Spark on Dataproc | Ephemeral Dataproc clusters, histogram equalization, serverless Spark |
| ch07 | Logistic Regression Using Spark ML | Logit/logistic regression, 3-way train/held-out/test, systematic feature selection, one-hot cost |
| ch08 | Machine Learning with BigQuery ML | `CREATE MODEL` SQL workflow, custom split, model-type swap, TRANSFORM clause, Vizier tuning |
| ch09 | Machine Learning with TensorFlow in Vertex AI | Wide-and-deep model, Keras feature columns, hashed buckets, Vertex AI model/endpoint deploy |
| ch10 | Getting Ready for MLOps with Vertex AI | Notebook-to-pipeline conversion, predefined split, cheap hyperparameter tuning, Shapley explain |
| ch11 | Time-Windowed Features for Real-Time ML | Shared transform modules, sliding-window dedup, batched model invocation, watermarks/triggers |
| ch12 | The Full Dataset | True temporal holdout, continuous training, schema-drift fallback, sensitive-data governance |

## Topic Index
- **AutoML** → Ch8 (via BigQuery ML), Ch9, Ch10, Ch12
- **BigQuery ML** → Ch8
- **Bigtable (streaming sink, schema design)** → Ch11
- **Causality** → Ch2, Ch4, Ch8, Ch11
- **Cloud Dataflow / Apache Beam** → Ch4, Ch11
- **Cloud Dataproc / Spark** → Ch6, Ch7
- **Cloud Run / Cloud Scheduler** → Ch2
- **Confusion matrix / contingency table** → Ch3, Ch5, Ch12
- **Dashboards (Data Studio)** → Ch3, Ch4
- **Embeddings** → Ch7, Ch8, Ch9
- **Explainability (Shapley)** → Ch10
- **Feature cross** → Ch8, Ch9
- **Feature engineering / selection** → Ch7, Ch8, Ch9
- **Hyperparameter tuning (Vizier)** → Ch8, Ch10
- **MLOps** → Ch10, Ch12
- **One-hot encoding (and its scaling limits)** → Ch7, Ch8, Ch9
- **Pub/Sub** → Ch4, Ch11
- **Sensitive data / governance** → Ch12 (appendix)
- **Streaming aggregation / time windows** → Ch4, Ch8, Ch11
- **TensorFlow / Keras / Wide-and-deep** → Ch9, Ch10, Ch11
- **Train/test split methodology** → Ch5, Ch8, Ch9, Ch10, Ch12
- **Vertex AI Workbench (Jupyter)** → Ch5
- **XGBoost / gradient boosted trees** → Ch8, Ch9

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — GCP architecture/ML techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 12 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
