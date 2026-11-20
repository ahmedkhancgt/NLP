# Patterns

## Hub-and-Spoke / ELT Ingestion
**When to use**: Any ingestion of external data into GCP for analytics/ML.
**How**: Extract raw data from the source as-is, load it unmodified into Cloud Storage then BigQuery (the hub); perform all transformation downstream via SQL views (the spokes), per use case.
**Trade-offs**: Avoids dependency on source-system transformation reliability and keeps a single source of truth; costs some storage for raw duplicated data and defers schema enforcement.

## Serverless Scheduled Ingestion (Cloud Run + Cloud Scheduler)
**When to use**: Periodic (e.g., monthly) data pulls from an external source.
**How**: Wrap ingestion logic in a Flask app, containerize with Docker, deploy to Cloud Run under a least-privilege service account with `--no-allow-unauthenticated`; trigger via Cloud Scheduler hitting the HTTPS endpoint with configured retry/backoff.
**Trade-offs**: Zero infrastructure management and built-in retries/auditability vs. a traditional cron job on a VM, at the cost of some up-front container/IAM setup.

## Causality-Respecting Multi-Event Publication
**When to use**: Streaming/real-time systems modeling an entity (e.g., a flight) whose full record is only progressively knowable.
**How**: Emit one event per state transition (scheduled, departed, wheelsoff, arrived), stripping fields not yet knowable at that point, instead of one event at completion.
**Trade-offs**: More events to process and more complex downstream schema vs. models that can actually be used for real-time decisions before the entity's outcome is known.

## Date-Hash Train/Test Split
**When to use**: Any dataset where rows are correlated by a grouping key (e.g., same-day flights share weather/traffic) and random row-level splitting would leak information.
**How**: Hash the grouping key (e.g., `FARM_FINGERPRINT(date)`) and assign TRAIN/VALIDATE/TEST by hash-mod-100 buckets; persist the split as a reusable table/column so every model trains/evaluates on identical splits.
**Trade-offs**: Slightly more setup than `RAND() < 0.7`, but reproducible, leak-free, and consistent across frameworks (BigQuery, Spark, Vertex AI) and chapters/models.

## Frequency-Based Outlier Filtering
**When to use**: Large datasets (hundreds of thousands+ rows) where rare values may carry real signal rather than noise.
**How**: Filter groups by `HAVING count > N` (N derived e.g. from the three-sigma rule, ~370) instead of filtering by raw value range.
**Trade-offs**: Retains long-tail values that matter at scale vs. classic small-data outlier removal (μ ± 3σ), which would discard real signal.

## Nonparametric (Empirical) Percentile Thresholding
**When to use**: Deriving a decision threshold from data without assuming a distribution shape (e.g., Gaussian).
**How**: Use `APPROX_QUANTILES(value, 101)[OFFSET(k)]` grouped by the conditioning variable, filtered to groups with enough samples.
**Trade-offs**: More robust than a Z-score-based threshold when the true distribution isn't Gaussian; requires enough samples per group to be reliable.

## Model-Type Swap (BigQuery ML / Vertex AI)
**When to use**: Comparing linear vs. nonlinear vs. AutoML models on the same tabular problem.
**How**: Change only the `model_type` option (`logistic_reg` → `boosted_tree_classifier` → `automl_classifier`) or swap `CustomTrainingJob` for `AutoMLTabularTrainingJob`; keep the dataset, split, and surrounding pipeline code unchanged.
**Trade-offs**: Minimal code churn to compare model families; AutoML costs more compute time/money and may not support custom split methods in every context.

## Vizier-Based Hyperparameter Tuning
**When to use**: Tuning multiple hyperparameters where grid search would be prohibitively expensive.
**How**: Specify ranges (`hparam_range`, `IntegerParameterSpec`, `CategoricalParameterSpec`) and a trial budget; let Bayesian optimization (Vizier) search instead of enumerating the full grid; use short/partial training runs per trial, full training only for the winning configuration.
**Trade-offs**: Finds good hyperparameters in a fraction of the compute of grid search, but doesn't guarantee the global optimum and still requires a correctly-scoped validation (not test) set.

## TRANSFORM-Clause / Shared-Transform-Module Feature Baking
**When to use**: Any derived feature (timestamp extraction, time-window aggregate) that must be computed identically at training and serving time.
**How**: In BigQuery ML, use the `TRANSFORM` clause so `ML.PREDICT` reapplies the same derivation automatically. In Beam/TensorFlow pipelines, factor the transform into a shared Python module called identically by both the batch training pipeline and the streaming/serving pipeline.
**Trade-offs**: Eliminates training-serving skew at the cost of slightly more upfront pipeline structuring.

## Sliding-Window Streaming Aggregation with Dedup
**When to use**: Rolling real-time metrics (e.g., trailing-hour average delay per airport) computed every few minutes.
**How**: Key by entity, apply `SlidingWindows(duration, period)`, `GroupByKey`, compute stats; to avoid re-emitting the same entity once per overlapping pane, check `window.start` and only emit from the earliest pane the entity belongs to.
**Trade-offs**: Gives smooth, frequently-updated aggregates; requires explicit dedup logic or risks N-fold duplicate output (N = duration/period).

## Batched, Reusable Model Invocation in Streaming Pipelines
**When to use**: A Beam/Dataflow pipeline calling an external ML prediction endpoint per record.
**How**: Strip windowing with `GlobalWindows()`, batch with `BatchElements(min_batch_size, max_batch_size)` (let Dataflow auto-tune batch size), and initialize the endpoint stub once per worker via `DoFn.setup()` (not `__init__`, which can be passivated/reactivated elsewhere).
**Trade-offs**: Dramatically reduces network calls/cost/latency vs. one-at-a-time invocation; use the heavier `Shared()` handle pattern instead only for large/expensive/non-thread-safe objects like an in-memory loaded model.

## Watermark/Trigger Tuning for Late Data
**When to use**: Streaming aggregations where out-of-order or delayed records are expected.
**How**: Default triggers fire at the watermark then individually on every late record (correctness-first); override with `AfterWatermark(late=AfterCount(n))` and `allowed_lateness` to batch/limit late-record reprocessing.
**Trade-offs**: Trades strict per-record correctness for lower processing overhead; bounded lateness discards very-late records entirely.

## Ephemeral, Job-Scoped Clusters (Dataproc)
**When to use**: Any Spark/Hadoop/Pig/Hive workload on GCP.
**How**: Keep data on Cloud Storage (never HDFS); create a cluster sized for the job, run it, delete it (or use serverless Spark batches) rather than keeping a cluster always-on.
**Trade-offs**: Saves significant cost vs. an always-on cluster; relies on GCP's high-bisection-bandwidth network making Cloud-Storage-backed compute competitive with HDFS for large sequential reads.

## Sliced/Segmented Model Evaluation
**When to use**: Verifying a trained model's behavior is sensible and fair across categorical subgroups (airline, airport, error type), not just in aggregate.
**How**: Export evaluation predictions + ground truth to a queryable table (e.g., BigQuery via AutoML's evaluation export); slice with SQL `GROUP BY` on categorical dimensions and plot feature-vs-prediction and feature-vs-error relationships.
**Trade-offs**: Surfaces non-obvious learned relationships and potential bias/error patterns that aggregate RMSE/accuracy hides; costs extra pipeline plumbing to export and query evaluation data.

## Sensitive-Data Identify → Protect → Govern
**When to use**: Any ML dataset containing or potentially containing personally identifying or otherwise sensitive information.
**How**: Identify sensitive fields (columns, NLP-detectable patterns, unstructured content via Vision/Speech/NLP/DLP APIs, field combinations, embedded context); protect via removal (views/redaction), masking (unsalted strong hash, tokenization, PCA-based dimensionality reduction), or coarsening (reduced precision: zip3, rounded lat/lon, age bucket, IP octet zeroing); govern via documented policy covering sources, locations, access requests, periodic review, and off-boarding.
**Trade-offs**: Every masking technique remains vulnerable to frequency-analysis attacks, especially in combination — technical measures must always be paired with access governance, never relied on alone; removal/over-masking can also materially degrade model accuracy.
