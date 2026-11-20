# Glossary

**Action probability** — In MAB testing, the likelihood that a variant is the best choice given current reward information (Ch 9).
**Active learning** — Labeling workflow where a model progressively takes over high-confidence labels, freeing humans for hard cases (Ch 10).
**All-reduce** — Ring-like MPI distributed communication strategy (Horovod-compatible) for large training clusters (Ch 8).
**Amazon A2I (Augmented AI)** — Managed human-in-the-loop review service triggered by low model-prediction confidence (Ch 1, 10).
**Amazon Athena** — Serverless, Presto-based SQL query engine over S3 data; schema-on-read (Ch 1, 2, 4, 5).
**Amazon Braket** — Managed quantum computing service with Python SDK (Ch 2).
**Amazon Comprehend** — Managed NLP service for sentiment, entities, key phrases, custom classifiers (Ch 2, 3).
**Amazon EMR** — Managed Hadoop/Spark cluster service (Ch 5).
**Amazon Forecast** — Managed time-series demand forecasting service (DeepAR++, ARIMA, ETS, Prophet) (Ch 2).
**Amazon Fraud Detector** — Managed fraud-detection model builder (Ch 2).
**Amazon Kendra** — Cognitive/NLU-based enterprise search service (Ch 2).
**Amazon Kinesis** — Family of streaming services: Data Firehose (delivery), Data Analytics (SQL/Flink processing), Data Streams (custom ingestion) (Ch 1, 11).
**Amazon MSK** — Managed Streaming for Apache Kafka (Ch 1, 11).
**Amazon Neptune ML** — Graph database with Graph Neural Network-based predictions (Ch 2).
**Amazon Personalize** — Managed recommendation engine with built-in cold-start handling (Ch 2).
**Amazon QuickSight** — Serverless BI/dashboard service with ML Insights (RCF) and QuickSight Q (NLQ) (Ch 1, 2, 5).
**Amazon Redshift** — Fully managed columnar data warehouse; schema-on-write (Ch 1, 2, 4, 5).
**Amazon Redshift ML** — Trains/invokes SageMaker Autopilot models via SQL from Redshift (Ch 2, 3).
**Amazon Redshift Spectrum** — Queries S3 data directly from Redshift without loading it (Ch 1, 4).
**Amazon Rekognition** — Managed computer vision service for image/video analysis and content moderation (Ch 2).
**Amazon S3 Access Points** — Per-team named access paths into a shared S3 bucket with individual policies (Ch 12).
**Amazon S3 Intelligent-Tiering** — Automatic storage-class optimization based on access patterns (Ch 4).
**ARN (Amazon Resource Name)** — Universal AWS resource identifier format: `arn:partition:service:region:account-id:resource-id` (Ch 3).
**Attention mechanism** — Weighted average of input vectors, weights from normalized query·key dot products; core of Transformer/BERT (Ch 7).
**AWS Data Wrangler (open source library)** — Connects pandas DataFrames to S3/Athena/Redshift/Glue (distinct from SageMaker Data Wrangler) (Ch 1, 5).
**AWS Deequ / PyDeequ** — Apache Spark-based data-quality unit-testing library (Ch 1, 5, 10).
**AWS Glue** — Serverless ETL + Data Catalog service; includes Crawlers and DataBrew (Ch 1, 4, 5, 6).
**AWS IAM** — Identity and Access Management; controls authentication and authorization via users, roles, and policies (Ch 12).
**AWS Inferentia / Trainium** — Custom AWS silicon for inference and training acceleration, respectively (Ch 1).
**AWS KMS** — Key Management Service for encryption at rest/in transit via Customer Master Keys (Ch 12).
**AWS Lake Formation** — Data lake creation/governance service with fine-grained row/column access control (Ch 1, 4, 12).
**AWS Step Functions** — Managed workflow orchestration service with a Data Science SDK (Ch 1, 10).
**Baseline (Model Monitor)** — Statistical reference (from training data) against which live model inputs/outputs are compared for drift (Ch 9).
**Bayesian optimization** — HPT strategy using a surrogate model + acquisition function to inform sequential search (Ch 8).
**BERT** — Bidirectional Encoder Representations from Transformers; true bidirectional self-attention NLP model (Ch 6, 7).
**Bias metrics (CI, DPL, JS, KL, KS, LP, TVD, CDD, AD, DRR, RD, TE)** — SageMaker Clarify's pre/post-training bias measurements (Ch 5, 7).
**Blue/green deployment** — Full traffic cutover to a new model cluster with the old cluster kept idle for rollback (Ch 9).
**Canary rollout** — Small percentage of traffic routed to a new model variant for risk mitigation (Ch 9).
**Class imbalance (CI)** — Bias metric in [-1,1] measuring facet representation skew; 0 = balanced (Ch 5).
**CMK (Customer Master Key)** — Primary AWS KMS resource representing an encryption key (Ch 12).
**Cold start problem** — Lack of historical data for new users/items in a recommender system (Ch 2).
**Concept shift** — Drift where a label's meaning changes depending on a feature (e.g., region) (Ch 5, 9).
**Covariate shift** — Drift in the distribution of model inputs (Ch 5, 9).
**Data lake** — Centralized, schema-on-read repository for structured/semi-structured/unstructured data, typically on S3 (Ch 1, 4).
**Data leakage** — Information from validation/test improperly influencing training, inflating apparent accuracy (Ch 6).
**Data parallelism** — Sharding the dataset (not the model) across training instances (Ch 8).
**DistilBERT** — Knowledge-distilled, smaller/faster version of BERT (60% faster, 40% smaller, ~97% accuracy) (Ch 6, 7).
**EFA (Elastic Fabric Adapter)** — OS-bypass network interface for low-latency, high-bandwidth inter-instance communication (Ch 1, 8).
**EIA (Elastic Inference Accelerator)** — Low-cost, dynamically attached GPU add-on for inference on CPU instances (Ch 9).
**ELT (extract-load-transform)** — Load-then-transform pattern enabled by Redshift Spectrum (Ch 4).
**Feature engineering** — Selection, creation, and transformation of raw data into model-ready features (Ch 6).
**Feature store (SageMaker Feature Store)** — Centralized, discoverable, online/offline repository of ML features (Ch 6).
**FSx for Lustre** — High-performance, S3-backed, POSIX-compatible distributed file system for training clusters (Ch 1, 8).
**GAN (Generative Adversarial Network)** — Generator + discriminator architecture for generating new content (Ch 2).
**Grid/random/Bayesian search** — Three HPT strategies; SageMaker HPT supports only random and Bayesian (Ch 8).
**HyperLogLog (HLL)** — Probabilistic cardinality-estimation algorithm for fast approximate COUNT DISTINCT (Ch 5, 11).
**Inference pipeline** — Sequence of containers (request handler → model → response handler) behind one SageMaker endpoint (Ch 9).
**Knowledge distillation** — Training a smaller "student" model to mimic a larger "teacher" model (Ch 7).
**Lake House Architecture** — Unified governance model combining an S3 data lake with a Redshift data warehouse (Ch 1, 4).
**Masked Language Model (Masked LM)** — BERT pre-training task masking 15% of tokens and predicting them (Ch 7).
**MLOps v1.0/v2.0/v3.0** — Maturity model: manual → manually-orchestrated pipelines → fully automated, trigger-based pipelines (Ch 1, 10).
**Model parallelism** — Sharding the model itself across instances when it doesn't fit on one (Ch 8).
**Multi-armed bandit (MAB)** — Reinforcement-learning-based dynamic traffic allocation across model variants, balancing exploit/explore (Ch 9).
**Next Sentence Prediction (NSP)** — BERT pre-training task predicting whether two sentences are a valid consecutive pair (Ch 7).
**Parameter server** — Distributed training strategy storing/sharing learned parameters across instances (Ch 8).
**Pipe Mode** — SageMaker training data-loading mode that streams from S3 on demand instead of copying upfront (Ch 8).
**Production variant** — A SageMaker Endpoint's combination of instance type/count/model, used for traffic-splitting strategies (Ch 9).
**Random Cut Forest (RCF)** — Streaming/batch anomaly-detection algorithm producing anomaly + per-column attribution scores (Ch 2, 11).
**SageMaker Autopilot** — Transparent AutoML service generating reproducible notebooks and ranked model candidates (Ch 1, 3).
**SageMaker Batch Transform** — Offline, high-throughput batch inference on S3 data without a standing endpoint (Ch 1, 9).
**SageMaker Clarify** — Bias detection, SHAP-based feature attribution, and explainability analysis service (Ch 5, 7, 9).
**SageMaker Data Wrangler** — ML-focused data ingestion/analysis/transformation/bias-detection tool with full lineage (distinct from AWS Data Wrangler library) (Ch 5, 6).
**SageMaker Debugger** — Zero-code-change training monitoring/profiling with rule-based alerts and early stopping (Ch 7).
**SageMaker Endpoint** — Real-time, distributed REST API hosting for a model (Ch 1, 9).
**SageMaker Experiments** — Tracking/comparison API for training runs (Experiment → Trial → Trial Component) (Ch 3, 6).
**SageMaker Ground Truth** — Managed data-labeling service combining human workforces with active learning (Ch 1, 10).
**SageMaker JumpStart** — One-click access to pre-built solutions and pre-trained models (Ch 1, 7).
**SageMaker Lineage Tracking** — Automatic artifact/action/association graph across the ML lifecycle (Ch 6).
**SageMaker Model Monitor** — Scheduled comparison of live endpoint data against a baseline to detect drift (Ch 9, 10).
**SageMaker Model Registry** — Central model versioning and staging/production approval system (Ch 10).
**SageMaker Neo** — Hardware-specific model compilation service (quantization, graph pruning, layer fusing) (Ch 9).
**SageMaker Pipelines** — SageMaker-native, full-featured ML pipeline orchestrator (Ch 1, 10).
**SageMaker Processing Jobs** — Managed, ephemeral compute for custom data processing/feature engineering/evaluation scripts (Ch 1, 5, 6).
**Schema-on-read vs. schema-on-write** — Athena/S3 require no predefined schema until query time; Redshift requires upfront schema definition (Ch 4).
**Script Mode** — Bring-your-own training script run inside a SageMaker-managed framework container (Ch 1, 7).
**SHAP** — Game-theory-based feature attribution method used by SageMaker Clarify (Ch 7).
**Sharding (ShardedByS3Key)** — Distributing dataset part-files evenly across cluster instances instead of full-copy to each (Ch 8).
**Stagger / Tumbling / Sliding windows** — Three Kinesis Data Analytics streaming-SQL window types (Ch 11).
**TFRecord** — Binary, protobuf-based TensorFlow-optimized training data file format (Ch 6).
**Training-serving skew** — Mismatch between the data distribution a model was trained on and what it sees in production (Ch 5, 10).
**VPC Endpoint (PrivateLink)** — Private ENI-based connection to AWS services bypassing the public internet (Ch 12).
**Warm start (HPT)** — Reusing a prior tuning job's results as a prior for a new tuning job (Ch 8).
**Word-embedding evolution (Word2Vec → GloVe → FastText → ELMo → GPT-n → BERT)** — Historical progression of NLP representation learning (Ch 7).
