---
name: data-science-on-aws
description: "Knowledge base from \"Data Science on AWS\" by Chris Fregly and Antje Barth. Use when building ML/data science pipelines on AWS — automated ML, data ingestion, dataset exploration/preparation, model training (including at scale), model deployment to production, pipelines and MLOps, streaming analytics/ML, or securing data science workloads on AWS."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Science on AWS
**Authors**: Chris Fregly, Antje Barth | **Pages**: ~524 | **Chapters**: 12 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch09`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The AWS ML stack ladder**: AI Services (1-click, e.g. Comprehend/Rekognition/Personalize/Forecast) → SageMaker Autopilot/AutoML (transparent, auto-generated notebooks) → custom SageMaker models (built-in algorithms → Script Mode → Bring Your Own Container). Default to the top of this ladder; descend only when the business problem demands it.

**MLOps maturity (v1.0 → v2.0 → v3.0)**: manual build/train/deploy → manually orchestrated pipelines → fully automated pipelines triggered deterministically (GitOps commits, new S3 data, cron schedule) or statistically (data/model/bias/explainability drift via SageMaker Model Monitor + Clarify). SageMaker Pipelines + Model Registry + Projects implement the full v3.0 strategy.

**Data ingestion/analytics decision**: Athena (serverless, schema-on-read, ad hoc S3 queries) vs. Redshift (schema-on-write, performance-critical BI) vs. Lake House (Redshift Spectrum unifying both). Use Parquet + partitioning for any large-scale analytical workload.

**Feature engineering → BERT pipeline**: raw text → WordPiece tokenization → DistilBERT embeddings (input_ids/input_mask/segment_ids) → TFRecords → SageMaker Feature Store (offline for training, online for low-latency inference). Freeze BERT's base layers and train only a lightweight classifier head unless accuracy demands full fine-tuning.

**Training at scale**: SageMaker Hyper-Parameter Tuning (Bayesian > random > manual; grid search unsupported) with warm start to iteratively narrow ranges; data parallelism (all-reduce, Horovod-compatible) when the dataset — not the model — is the bottleneck; ShardedByS3Key + Pipe Mode + FSx for Lustre for I/O at scale.

**Deployment & rollout ladder**: canary (risk mitigation, small % traffic) → blue/green (safe full cutover, idle rollback cluster) → A/B test (long-running, statistically rigorous, static) → multi-armed bandit (dynamic, minimizes regret, exploit/explore). Choose real-time Endpoints for latency-sensitive single predictions; Batch Transform for high-throughput offline scoring.

**Continuous drift detection**: SageMaker Model Monitor (data quality via Deequ baselines, model quality via ground-truth comparison) + SageMaker Clarify (bias drift, feature-attribution/SHAP drift) — the automated feedback loop that makes MLOps v3.0 possible, wired into SageMaker Pipelines via EventBridge triggers.

**Security baseline (defense in depth)**: shared responsibility model (AWS secures "of" the cloud, you secure "in" it) → least-privilege IAM (separate role per job function/task) → VPC isolation + VPC Endpoints (no public internet path) → KMS encryption everywhere (at rest and in transit) → AWS Organizations/SCPs for org-wide guardrails → CloudTrail/CloudWatch + tagging for auditability.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Introduction to Data Science on AWS | SageMaker, AI Services stack, AutoML, pipeline orchestration options, AWS infrastructure/instance types |
| ch02 | Data Science Use Cases | Personalize, Forecast, Fraud Detector, Rekognition, Comprehend, Textract, Kendra, in-database ML, DeepRacer/DeepComposer, Amazon Braket (quantum) |
| ch03 | Automated Machine Learning | SageMaker Autopilot (transparent AutoML), Athena/Redshift ML inference, Amazon Comprehend Custom |
| ch04 | Ingest Data into the Cloud | S3 data lake, Athena, AWS Glue Crawler, Redshift + Redshift Spectrum, Lake House Architecture, S3 Intelligent-Tiering |
| ch05 | Explore the Dataset | SageMaker Studio EDA, AWS Deequ/PyDeequ, SageMaker Data Wrangler + Clarify (bias/class imbalance), Glue DataBrew, HyperLogLog |
| ch06 | Prepare the Dataset for Model Training | Feature selection/creation/transformation, BERT/DistilBERT tokenization, TFRecords, SageMaker Processing Jobs at scale, SageMaker Feature Store, Lineage/Experiments |
| ch07 | Train Your First Model | SageMaker infrastructure/containers, BERT architecture & history of NLP, fine-tuning, SageMaker Debugger, SageMaker Clarify (post-training bias, SHAP), PyTorch/MXNet/DJL alternatives |
| ch08 | Train and Optimize Models at Scale | SageMaker Hyper-Parameter Tuning (Bayesian, warm start), distributed training (data/model parallelism, parameter server/all-reduce), FSx for Lustre/EFS, ShardedByS3Key, Pipe Mode |
| ch09 | Deploy Models to Production | SageMaker Endpoints vs. Batch Transform, inference pipelines, auto-scaling, canary/blue-green/A-B/MAB rollout strategies, SageMaker Model Monitor, SageMaker Neo/TFLite, TorchServe/DJL |
| ch10 | Pipelines and MLOps | SageMaker Pipelines, Model Registry, GitOps/S3/schedule/drift triggers, Step Functions/Kubeflow/Airflow/MLflow/TFX alternatives, Amazon A2I, SageMaker Ground Truth active learning |
| ch11 | Streaming Analytics and Machine Learning | Kinesis Data Firehose/Analytics/Streams, Amazon MSK, windowed SQL queries (stagger/tumbling/sliding), Random Cut Forest anomaly detection, Enhanced Fan-Out |
| ch12 | Secure Data Science on AWS | IAM (users/roles/condition keys), VPC/VPC Endpoints, S3 Access Points, KMS encryption, AWS Organizations/SCPs/Control Tower, Lake Formation, Secrets Manager, CloudTrail/auditability |

## Topic Index
- **A/B testing** → ch09
- **Amazon Athena** → ch01, ch02, ch03, ch04, ch05
- **Amazon A2I (human-in-the-loop)** → ch01, ch10
- **Amazon Redshift / Redshift Spectrum / Redshift ML** → ch01, ch02, ch04, ch05
- **AutoML / SageMaker Autopilot** → ch01, ch03, ch08
- **AWS Deequ / data quality** → ch05, ch10
- **AWS Glue / DataBrew / Crawler** → ch01, ch04, ch05, ch06
- **AWS Lambda (serverless inference / streaming)** → ch01, ch09, ch11
- **Bias detection (SageMaker Clarify)** → ch05, ch07, ch09
- **BERT / DistilBERT / Transformers** → ch06, ch07
- **Blue/green deployment** → ch09
- **Canary rollout** → ch09
- **Distributed training** → ch08
- **Drift detection (Model Monitor)** → ch05, ch09, ch10
- **EDA / data exploration** → ch05
- **Encryption (KMS)** → ch12
- **Feature engineering** → ch06
- **Feature Store** → ch06
- **Hyper-parameter tuning** → ch03, ch08
- **IAM / security** → ch12
- **Kinesis / streaming** → ch01, ch11
- **Multi-armed bandit (MAB)** → ch09
- **SageMaker Batch Transform** → ch01, ch09
- **SageMaker Debugger** → ch07
- **SageMaker Endpoints** → ch09
- **SageMaker Pipelines / MLOps** → ch01, ch10
- **SageMaker Studio** → ch05, ch12
- **Security / VPC / governance** → ch12
- **VPC / network isolation** → ch07, ch08, ch12

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — AWS architecture/ML techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 12 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
