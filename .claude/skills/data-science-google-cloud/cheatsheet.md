# Cheatsheet

## Which ML tool should I use? (decision tree)
1. **Tabular data, want a first model fast?** → **BigQuery ML** (`CREATE MODEL`). Handles categorical columns transparently, no data export needed, scales effortlessly.
2. **BigQuery ML's linear/XGBoost/AutoML model types aren't expressive enough** (need embeddings for high-cardinality feature crosses, custom architectures, GPU-heavy models)? → **TensorFlow/Keras on Vertex AI** (wide-and-deep, custom layers).
3. **Already have a Hadoop/Spark shop, or need Spark-specific tooling (Pig/Hive/MLlib)?** → **Dataproc + Spark ML**, but expect scaling pain once categorical one-hot encoding grows past a few hundred columns — prefer BigQuery ML or TensorFlow for that case.
4. **Need the model to incorporate real-time computed features (moving averages, time windows)?** → **Apache Beam/Dataflow** shared-transform-module pattern (batch training + streaming serving, same code).
5. **Want to skip model-architecture decisions entirely and let the system search?** → **AutoML Tables** (via BigQuery ML `model_type='automl_classifier'` or Vertex AI `AutoMLTabularTrainingJob`); best on large datasets, costs more training time, easy BigQuery evaluation export.
6. **Need production deployment, monitoring, explainability, hyperparameter tuning, versioning?** → **Vertex AI** (Endpoints, Vizier, Shapley explanations, pipelines) regardless of which model type you picked above.

## Which GCP storage/compute should I use?
| Need | Use |
|---|---|
| SQL analytics on large (TB+) datasets, ad hoc queries, no indexing | **BigQuery** |
| Flat files, non-tabular data (images/video/arbitrary), custom code reading whole files | **Cloud Storage** |
| Small, frequently-updated, transactional, indexable relational data | **Cloud SQL** |
| Millions of ops/sec, ms latency, petabyte scale, no transactions | **Cloud Bigtable** |
| Global strong consistency + SQL + massive scale + 99.999% availability | **Cloud Spanner** |
| Real-time message ingestion (pub/sub boundary) | **Pub/Sub** |
| Batch or streaming data transformation pipeline (same code for both) | **Dataflow (Apache Beam)** |
| Hadoop/Spark/Pig/Hive-specific workloads | **Dataproc** (ephemeral clusters, data on GCS not HDFS) |

## Scaling decision rule
- Job re-reads the same data many times (ML training)? → **Scale up** (bigger VM/GPU instance).
- Splittable batch job, no high-bandwidth data center available? → **Scale out with sharded data** (Dataproc + HDFS / MapReduce).
- On GCP, with high-bisection-bandwidth network? → **Scale out with data-in-place** (BigQuery, Dataflow, Vertex AI directly on Cloud Storage) — the default/preferred choice on GCP.

## Default thresholds and rules of thumb
- **Minimum samples per group before trusting a statistic**: ~370 (derived from the three-sigma rule, 1/(1-0.9973)).
- **Train/validate/test split**: 95/5 within training years when a separate future year is held out as test; otherwise commonly 70/30 or 60/20/20 — always split by a correlated grouping key (date), never by raw row, when rows are correlated.
- **"Enough" improvement to justify a new feature/variable**: no universal number; the book uses 0.5% RMSE improvement as its own bar — weigh against retraining burden, runtime cost, and agility loss from the extra feature/dependency.
- **One-hot encoding ceiling**: fine up to a few dozen categories; beyond ~100–300 unique values, switch to hashed buckets (unknown vocabulary) or embeddings (dimensionality reduction) — one-hot encoding ~300 airports was enough to crash a 50-node Spark cluster.
- **Batch size for streaming ML invocation**: let Dataflow's `BatchElements(min_batch_size, max_batch_size)` auto-tune; don't hardcode — observed settling point in the book was ~30 elements per call with max 64.
- **Hyperparameter tuning**: 5–10 trials with Bayesian search (Vizier) is often enough to find near-optimal settings; grid search over the same space can cost orders of magnitude more compute for comparable gains.

## Tells and smells
- **"We removed the NULL rows"** → stop: find the root cause (e.g., canceled/diverted flights) instead of masking a data-quality symptom.
- **Metric improves when you lower the decision threshold** → you're gaming a contingency-table metric (accuracy/precision/recall); use a full-distribution metric (RMSE, logistic loss, cross-entropy) for feature selection or model comparison instead.
- **A feature's real-world value isn't known until prediction time** → causality violation; either drop the feature or replace it with a real-time-computable approximation (e.g., historical average instead of actual).
- **A time-windowed aggregate feature was computed using the whole dataset including evaluation/future rows** → causality leak; recompute using only data genuinely available at prediction time.
- **Each flight/entity appears duplicated N times in sliding-window output** → classic sliding-window dedup bug; check `window.start`/`window.end` and emit only from the earliest matching pane.
- **A Beam `DoFn.__init__` caches a remote connection/stub** → wrong lifecycle hook; move it to `setup()`, since `__init__`-set instance state can be silently resurrected on a different worker.
- **Query "Execution details" shows most time in the "read" step** → query is I/O-bound; consider a summary/aggregate table or partitioning rather than optimizing compute.
- **Global feature importance says a feature is unimportant, but a specific prediction's Shapley explanation says it dominated** → normal; global and local explanations can disagree — check both before concluding a feature doesn't matter.
- **A model trained on recent years performs noticeably worse on a brand-new independent test year** → expected drift, not necessarily a bug; quantify it, and build continuous retraining/fine-tuning rather than assuming a single training run is forever-valid.
- **Dataset contains PII and someone proposes a single hashing/masking pass as "done"** → insufficient; every masking technique remains vulnerable to frequency-analysis attacks (especially combined fields) — pair with access governance, not a one-time technical fix.

## Trade-off matrices
| Masking technique | Repeatable for ML? | Frequency-analysis resistant? | Accuracy impact |
|---|---|---|---|
| Removal | N/A | N/A (gone) | High if field was predictive |
| Salted hash | No | — | N/A (breaks join/repeatability) |
| Unsalted strong hash (SHA-256) | Yes | No | Low |
| Tokenization | Yes | No (same vulnerability as hash) | Low |
| PCA / dimensionality reduction | Yes (vector) | Harder but not immune | Medium (trades accuracy for security) |
| Coarsening (zip3, rounded lat/lon, age bucket) | Yes | Reduced but present at scale | Medium |

| Continuous training approach | Speed | Benefit |
|---|---|---|
| Fine-tune from checkpoint | Fast | Smaller, incremental |
| Retrain from scratch | Slow | Full benefit of all historical + new data |
| Rule of thumb | Fine-tune daily/weekly; retrain from scratch once new data ≈5% of original volume | — |
