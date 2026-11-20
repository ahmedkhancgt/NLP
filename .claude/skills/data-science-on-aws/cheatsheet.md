# Cheatsheet

## Choosing a data-query service
- Ad hoc, schema-on-read, no infra to manage, data lives in S3 → **Athena**.
- Business-critical reports/dashboards, complex joins, subsecond latency, structured data → **Redshift**.
- Need both (recent data fast, full history still in S3) → **Redshift Spectrum (Lake House)**.
- Never use Athena/Redshift as a write-heavy OLTP replacement → use RDS/Aurora.

## Choosing a SageMaker model-development option
- Standard problem (classification/regression/NLP/vision), no custom code wanted → **Built-in algorithm**.
- Need custom training logic but want a managed framework container → **Script Mode** (default recommendation).
- Need full control, custom security/compliance, or unsupported library → **Bring Your Own Container (BYOC)**.
- Want a fast baseline with zero modeling effort → **SageMaker Autopilot** or **SageMaker JumpStart** (pre-trained models).

## Choosing a deployment mode
- Latency-sensitive, single-request → **SageMaker real-time Endpoint**.
- High-throughput, latency-tolerant, large S3 dataset → **SageMaker Batch Transform**.
- Event-driven, serverless, infrequent → **AWS Lambda + API Gateway**.
- Resource-constrained device, need hardware-specific optimization → **SageMaker Neo + Edge Manager**.

## Choosing a rollout strategy
| Need | Use |
|---|---|
| Low-risk test of a new model on a small % of traffic | Canary rollout |
| Safe full cutover with instant rollback | Blue/green |
| Long-running, statistically rigorous comparison | A/B test |
| Fast-reacting comparison that minimizes regret, can add/remove variants | Multi-armed bandit (MAB) |

## HPT strategy decision rule
- Have no prior knowledge of good ranges → explore **logarithmically** first, then narrow with **linear**.
- Default to **Bayesian** optimization with max_parallel_jobs well below max_jobs (e.g., 10 of 100) so it can learn from completed runs.
- Never use grid search (not supported by SageMaker HPT; too slow anyway).
- Already have a prior tuning job → use **warm start** (`IDENTICAL_DATA_AND_ALGORITHM` for narrowed ranges, `TRANSFER_LEARNING` for new data/algorithm version).

## Distributed training decision rule
- Dataset too big for one instance, model fits → **data parallelism** (all-reduce / Horovod-compatible).
- Model too big for one instance's memory → **model parallelism**.
- Need resilience to Spot interruption → **parameter server** (stateful, runs on every instance) + checkpointing.
- Repeated epochs over the same large dataset → **FSx for Lustre** (S3-backed cache).

## Instance-type tells and smells
- **T** family: burstable general purpose (dev/test, light notebooks).
- **M** family: balanced general purpose.
- **C** family: compute-bound (CPU-heavy inference, Spark).
- **R** family: memory-optimized (in-memory Spark, R5 for Deequ jobs).
- **P/G/Inf/Trainium**: hardware-accelerated training (P/G) or inference (Inf/Trainium).
- Default: start small, ramp up only as measured need demands ("right-sizing").
- GPU for training ≠ GPU for inference — most inference workloads run fine on CPU or with an Elastic Inference Accelerator (EIA) attached to a CPU instance.

## Pricing mix defaults
- Steady-state, predictable workload → **Savings Plans / Reserved Instances**.
- New or short-term spiky workload → **On-Demand**.
- Flexible, fault-tolerant, checkpoint-able (training) → **Spot Instances** (up to 90% off; always pair with checkpointing).

## Auto-scaling defaults
- Prefer **target tracking** (least config) unless you need fine control over scale amounts at multiple thresholds (step scaling).
- Scale **out fast** (short cooldown, e.g., 60s), scale **in slow** (long cooldown, e.g., 300-600s) to absorb traffic dips without flapping.
- Keep instance types homogeneous within one auto-scaled cluster.

## Model Monitor drift thresholds — what to check
| Drift type | Detects | Needs |
|---|---|---|
| Data quality | Covariate shift in inputs | Deequ baseline |
| Model quality | Concept shift / accuracy decay | Ground truth labels |
| Bias drift | Shift in post-training bias metrics | Clarify bias baseline |
| Feature attribution drift | Shift in SHAP importances | Clarify explainability baseline |

## Tells and smells (red flags to catch)
- Model "accuracy" ≈ majority-class frequency → **unbalanced dataset**, not a good model.
- `review_body`-style text feature using plain TF-IDF when you have compute budget → prefer **BERT/DistilBERT** for better context understanding.
- HPT job's `learning_rate` found at the edge of its search range → range was mis-specified; widen and re-run.
- SageMaker Debugger flags `CPUBottleneck` or `LowGPUUtilization` → add data loaders/prefetching or increase batch size (don't just add more GPUs).
- A/B test "ends early" because one variant looks bad → should have used MAB, not A/B, if early reaction was ever a requirement.
- Approximate COUNT DISTINCT (HyperLogLog/RCF) used for financial/compliance reporting → wrong tool; needs exact COUNT.
- Notebook instance using a GPU type → move the GPU workload into a Processing/Training Job instead; keep the notebook small/cheap.
- Direct internet access or root access enabled on a SageMaker notebook/Studio domain → disable both unless explicitly justified.
- No KMS encryption specified on a Training Job's `output_kms_key`/`volume_kms_key` → should fail an org-wide SCP check; add it.

## Compression / encoding defaults (Redshift)
- Numeric/date columns → **az64** (default).
- General strings → **zstd**.
- Low-cardinality strings (dates, categories with few repeats) → **bytedict**.
- The SORTKEY column itself → leave **uncompressed (raw)**, even if a dictionary encoding looks attractive.

## Quick thresholds mentioned in the book
- BERT `max_seq_length`: pick near the 80th percentile of your text's word-count distribution (64 was used for Amazon reviews).
- Model-quality gate in a pipeline: don't register/deploy unless evaluation accuracy ≥ a defined threshold (95% was used as an example).
- A2I human-review confidence threshold: route to humans below ~70% confidence (tune per use case).
- Class imbalance (CI) metric: near 0 = balanced; for N facet categories, a perfectly even split gives CI ≈ 1/N.
