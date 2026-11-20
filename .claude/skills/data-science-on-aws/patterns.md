# Patterns

## Lake House Architecture (S3 + Redshift Spectrum)
**When to use**: Need both ad hoc exploratory analytics on raw S3 data and performant, relational business-intelligence reporting on a subset of recent data.
**How**: Land all raw data in S3; catalog it with Athena/Glue Data Catalog; load only the recent (e.g., 2-year) subset into Redshift proper for performance; create a Redshift external schema (`CREATE EXTERNAL SCHEMA ... FROM DATA CATALOG`) to query older S3 data via Redshift Spectrum; `UNION ALL` across Redshift tables and the external schema for full-history queries.
**Trade-offs**: Avoids costly full-data warehouse loads, but requires maintaining two query paths (native Redshift tables + Spectrum) and keeping the Glue Data Catalog in sync.

## Transparent AutoML Baseline (SageMaker Autopilot)
**When to use**: Starting a new tabular ML problem; need a fast, explainable baseline before investing in custom modeling.
**How**: Point Autopilot at an S3 CSV + target column; let it auto-detect problem type, generate 10 dpp[0-9] candidate pipelines, and rank them by validation metric; review the generated Data Exploration and Candidate Definition notebooks before trusting results.
**Trade-offs**: Transparent but slower/costlier than nontransparent AutoML; results are a baseline, not necessarily production-final.

## In-Database ML Inference (Athena/Redshift/Aurora UDFs)
**When to use**: Need to score data that already lives in a warehouse/lake/relational DB without exporting it to a separate inference service.
**How**: Define a `USING FUNCTION ... SAGEMAKER_INVOKE_ENDPOINT` UDF in Athena, or `CREATE MODEL` in Redshift ML, or built-in `aws_comprehend_*` functions in Aurora; reference the function in a normal `SELECT`.
**Trade-offs**: Minimizes data movement and ETL, but couples query performance to endpoint latency and adds per-row invocation overhead for large scans.

## BERT Fine-Tuning with Frozen Base Layers
**When to use**: Domain-specific text classification task where the target vocabulary overlaps with BERT's pre-training corpus (English Wikipedia/Books).
**How**: Load a pre-trained BERT/DistilBERT model, freeze its layers (`trainable=False`), add a lightweight classifier head (e.g., BiLSTM + Dense), train only the head on labeled examples.
**Trade-offs**: Much faster/cheaper than full fine-tuning or training from scratch; may sacrifice a few points of accuracy versus unfreezing BERT layers (which costs more compute/time).

## Bayesian Hyper-Parameter Tuning with Warm Start
**When to use**: Need to find good hyper-parameters efficiently, then further narrow/refine them without starting from scratch.
**How**: Run an initial `HyperparameterTuner` with `strategy='Bayesian'`, modest parallelism (e.g., 10 of 100 jobs at a time so Bayesian search can learn from completed runs), `early_stopping_type='Auto'`; for a follow-up run, pass `WarmStartConfig(warm_start_type=IDENTICAL_DATA_AND_ALGORITHM, parents={prior_job})` with a narrowed range.
**Trade-offs**: Much more sample-efficient than grid search (unsupported by SageMaker anyway) or pure random search; warm start can still underperform the parent if the narrowed range excludes the true optimum — always compare against the parent's best candidate.

## Data-Parallel Distributed Training (All-Reduce)
**When to use**: Dataset too large for a single instance, but the model itself fits comfortably in one instance's memory.
**How**: Shard training data across instances (`ShardedByS3Key` or native sharding), enable SageMaker's built-in all-reduce (`distribution={'smdistributed':{'dataparallel':{'enabled': True}}}`), optionally back the shared dataset with FSx for Lustre for repeated-epoch caching.
**Trade-offs**: Scales near-linearly for parallelizable algorithms (most neural nets); adding instances to non-parallelizable algorithms adds communication overhead with no speedup.

## Progressive Deployment Ladder (Canary → Blue/Green → A/B → MAB)
**When to use**: Rolling out a new model version safely, matched to the risk/goal of the specific rollout.
**How**: Canary (5% traffic, small instance count) to catch production-only bugs → blue/green (0→100% cutover with idle rollback cluster) for a safe full switch → A/B test (50/50, long-running) for rigorous statistical comparison → MAB (Vowpal Wabbit + exploration policy) when you want to minimize regret and dynamically add/remove variants mid-experiment.
**Trade-offs**: Canary/blue-green are about risk mitigation, not statistical rigor; A/B is statistically rigorous but static and slow to react; MAB reacts fast but is more complex to operate and interpret (action probability vs. sample probability).

## Continuous Drift Detection → Automated Retraining (MLOps v3.0)
**When to use**: Any production model that needs to stay accurate as real-world data distributions shift.
**How**: Enable `DataCaptureConfig` on the endpoint; create a Deequ-based data-quality baseline and a Clarify-based bias/explainability baseline from training data; schedule `ModelMonitor`/`ModelQualityMonitor` jobs (cron, e.g., hourly) to compare live traffic against baselines; wire violations into EventBridge to trigger a SageMaker Pipeline retrain.
**Trade-offs**: Closes the MLOps loop fully automatically, but requires ground-truth labels (often via human-in-the-loop) for model-quality (concept-drift) monitoring, which can lag real-time.

## Human-in-the-Loop Confidence Routing (Amazon A2I)
**When to use**: Predictions below a confidence threshold are too risky to auto-serve, but full manual review of all predictions is too slow/costly.
**How**: Set a `CONFIDENCE_SCORE_THRESHOLD`; route only predictions below it to `a2i.start_human_loop()` with a Worker Task Template + Flow Definition; feed human labels back into a continuous retraining pipeline (active learning via SageMaker Ground Truth).
**Trade-offs**: Balances cost/speed against risk; threshold tuning is itself a trade-off (too low = too many human reviews, too high = risky auto-predictions slip through).

## Streaming Enrichment Pipeline (Kinesis/MSK + Lambda + SageMaker)
**When to use**: Need real-time sentiment/classification enrichment of a continuous stream (e.g., product reviews) plus live descriptive analytics (averages, anomalies, counts).
**How**: Kinesis Data Firehose (or MSK) ingests raw messages; a Lambda transform calls a SageMaker Endpoint per record to enrich with a prediction; enriched records land in S3; Kinesis Data Analytics runs parallel windowed SQL queries (tumbling for averages, RCF for anomalies, `COUNT_DISTINCT_ITEMS_TUMBLING` for approximate counts) over the same stream for dashboards/alerts.
**Trade-offs**: Reuses the same trained batch model for streaming inference (no separate streaming-specific model needed); window-type choice (stagger/tumbling/sliding) must match the data's actual arrival pattern or aggregations will be wrong.

## Defense-in-Depth Security Baseline
**When to use**: Any production (or even research) data science environment handling sensitive data.
**How**: Separate IAM roles per job function and per SageMaker task; require VPC-only access for SageMaker Studio/notebooks (disable direct internet + root access); encrypt all data at rest (KMS on S3/EBS/EFS) and in transit (inter-container encryption for distributed jobs); enforce org-wide guardrails with SCPs; separate data-science/staging/production into distinct AWS accounts; log everything via CloudTrail/CloudWatch and tag resources for auditability.
**Trade-offs**: More setup overhead and occasional friction (e.g., no direct internet access means pre-bundling dependencies), but is the only way to pass compliance audits and avoid catastrophic data leaks.
