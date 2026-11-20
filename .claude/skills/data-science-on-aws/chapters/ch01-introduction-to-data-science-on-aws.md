# Chapter 1: Introduction to Data Science on AWS

## Core Idea
Cloud computing removes the infrastructure burden from data science workflows, letting teams move from prototype to production faster; AWS provides a layered stack (AI Services, SageMaker Autopilot/AutoML, SageMaker custom ML) matched to the full ML lifecycle from ingestion through deployment and streaming.

## Frameworks Introduced
- **Amazon SageMaker**: fully managed ML platform spanning data prep, training, tuning, deployment, and pipelines.
  - When to use: any custom ML model development that needs managed infrastructure instead of self-managed clusters.
  - How: point to data in S3, choose built-in algorithm/script mode/BYOC, specify instance type/count, SageMaker provisions and tears down compute automatically.
- **SageMaker Autopilot (AutoML)**: automatically analyzes tabular data, picks problem type (regression/classification) and algorithm (e.g., XGBoost), generates feature-engineering code, and produces a ranked leaderboard of model candidates.
  - When to use: well-known tabular ML problems, baselining, or empowering non-specialists to build models.
  - How: point Autopilot at an S3 dataset and target column; it generates transparent notebooks documenting the data analysis and candidate pipelines.
- **AWS Glue + AWS Lake Formation**: serverless data catalog/ETL and data-lake creation/governance service.
  - When to use: building a centralized S3-based data lake needing cataloging, cleaning, and access control.
  - How: Glue crawlers detect/catalog new data into the Glue Data Catalog; Lake Formation secures and classifies it.
- **Amazon Athena**: serverless, Presto-based distributed SQL engine for querying data directly in S3.
  - When to use: interactive SQL exploration of raw S3 data without ETL or server management.
  - How: define database/tables pointing at S3 locations (mapping stored in Glue Data Catalog); pay per query.
- **AWS Deequ (on SageMaker Processing)**: open-source library (built on Apache Spark) for unit-testing data quality at scale (billions of rows).
  - When to use: before training, to catch anomalies/errors early.
  - How: run as a "serverless Spark" job via SageMaker Processing Jobs.
- **SageMaker Ground Truth**: managed data-labeling service combining human workforces (Mechanical Turk, private, or vendor) with active learning.
  - When to use: supervised learning tasks needing accurate labels at scale.
  - How: define labeling task/workforce; active-learning model auto-labels subsets as it learns, reducing human labeling cost over time.
- **Amazon Kinesis / Amazon MSK**: real-time streaming ingestion and processing services.
  - When to use: need to analyze data streams before they land in the data lake.
  - How: Kinesis Data Firehose (load), Kinesis Data Analytics (process), Kinesis Data Streams (custom stream ingestion); MSK runs managed Kafka.

## Key Concepts
- **Data lake**: centralized, secure, scalable repository (commonly S3) storing structured, semi-structured, and unstructured data for broad analytics access.
- **Script mode**: bringing your own training script while SageMaker supplies the optimized framework container (TensorFlow, PyTorch, MXNet, XGBoost, scikit-learn).
- **Bring Your Own Container (BYOC)**: supplying a fully custom Docker image to SageMaker when built-ins/script mode are insufficient.
- **Managed Spot Training**: using EC2 Spot Instances inside SageMaker Training for up to 90% training cost reduction.
- **SageMaker Endpoints vs. Batch Transform vs. Lambda**: three deployment modes — real-time REST, high-throughput batch, or serverless, respectively.
- **MLOps maturity (v1.0 → v3.0)**: manual build/train/deploy → manual pipeline orchestration → automated retraining triggered by GitOps or statistical drift/bias/explainability signals.
- **Human-in-the-loop (Amazon A2I)**: managed service adding human review/QA into ML workflows with IAM-based access control.

## Mental Models
- **Shared responsibility model**: AWS secures "of" the cloud; the customer secures "in" the cloud (access, encryption, governance, auditability).
- **Pricing mix framework**: blend Savings Plans (steady-state), On-Demand (new/spiky), and Spot (flexible, fault-tolerant) instances to optimize cost — don't default to one model.
- **Right-sizing training vs. inference**: training may need GPU bursts for hours/days; inference runs continuously at scale — size each phase's hardware independently.
- **AWS Well-Architected lens for ML**: evaluate every project against operational excellence, security, reliability, performance efficiency, and cost optimization.

## Anti-patterns
- **Treating model training and inference hardware the same**: wastes money — inference often doesn't need the GPU acceleration training required.
- **Using BYOC without Docker expertise**: adds unnecessary operational burden; prefer built-in containers or script mode unless custom security/dependency needs truly require it.
- **Skipping data-quality checks ("garbage in")**: poor data quality is cited as a leading cause of failed ML projects.
- **Overprovisioning for peak load**: cloud elasticity makes static overprovisioning unnecessary and wasteful.

## Code Examples
```python
import awswrangler as wr

# Retrieve the data directly from Amazon S3
df = wr.s3.read_parquet("s3://<BUCKET>/<DATASET>/"))
```
- **What it demonstrates**: AWS Data Wrangler reads a Parquet dataset from S3 directly into a pandas DataFrame, with support for chunked reads to manage memory on large datasets.

## Reference Tables
| EC2 Family | Model characteristics |
|---|---|
| Inf1 (AWS Inferentia) | Low latency + high throughput at low cost for inference |
| C5 (compute-optimized CPU) | General compute-bound inference workloads |
| G4 (NVIDIA T4 GPU) | Cost-sensitive small-scale training/inference |

| Instance family | Optimized for |
|---|---|
| T | Burstable general purpose |
| M | Balanced compute/memory/network |
| C | Compute-optimized, CPU-bound |
| R | Memory-optimized (e.g., in-memory Spark) |
| P/G/Inferentia/Trainium | Hardware-accelerated training/inference |

## Worked Example
A typical AWS ML workflow for a tabular/product dataset: raw data lands as CSV/Parquet in an S3 data lake → AWS Glue crawlers catalog it → ad hoc exploration via SageMaker Studio notebooks using pandas/AWS Data Wrangler/Athena → data-quality checks with Deequ via SageMaker Processing → labeling via Ground Truth if supervised → feature transformation with SageMaker Data Wrangler or Glue DataBrew → model training with SageMaker Training (built-in XGBoost, script mode, or BYOC) optionally using Managed Spot Training → hyper-parameter tuning via SageMaker Automatic Model Tuning → deployment to a SageMaker real-time Endpoint, Batch Transform job, or Lambda function → orchestrated end-to-end with SageMaker Pipelines so new S3 data automatically retrains and redeploys the model.

## Key Takeaways
1. Match deployment strategy (Endpoint/Batch Transform/Lambda) to latency and throughput requirements rather than defaulting to real-time endpoints.
2. Start with AI Services or SageMaker Autopilot before building fully custom models — reuse before reinvention.
3. Blend Spot, Reserved/Savings Plans, and On-Demand pricing deliberately based on workload flexibility.
4. Track and version datasets, transformation code, hyper-parameters, and models for reproducibility and rollback.
5. Treat MLOps maturity as a progression toward automated, trigger-based retraining (GitOps + statistical drift detection), not a one-time setup.
6. Security is a shared responsibility — the customer must still secure data access, encryption, and governance in the cloud.

## Connects To
- **Ch 2**: Expands on where these AWS services apply across industry use cases.
- **Ch 3**: Deep dive into SageMaker Autopilot/AutoML introduced here.
- **Ch 4–6**: Detailed ingestion, exploration, and preparation services (S3, Athena, Glue, Data Wrangler) introduced here.
- **Ch 7–8**: Detailed model training/tuning (built-in algorithms, script mode, hyper-parameter tuning).
- **Ch 9–10**: Deployment options and MLOps pipeline orchestration (SageMaker Pipelines, Step Functions, Airflow, Kubeflow, MLflow, TFX).
- **Ch 11**: Streaming analytics/ML with Kinesis and MSK.
- **Ch 12**: Security best practices referenced throughout.
