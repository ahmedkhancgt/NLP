# Glossary

**AutoML Tables** — No-code Vertex AI service that automatically searches architectures (neural architecture search, TabNet) to build tabular models from a SQL-defined dataset. (Ch 8, 9, 10, 12)

**Bayesian classification (direct binning)** — Estimating P(outcome | bins) by counting outcomes within quantized bins of input variables. (Ch 6)

**BigQuery ML** — Low-code SQL interface (`CREATE MODEL`) for training, evaluating, and predicting with ML models directly inside BigQuery. (Ch 8)

**BigQuery slots** — On-demand, elastically allocated parallel compute units executing a BigQuery SQL query; not statically pre-provisioned. (Ch 2)

**Bucketizing** — Discretizing a continuous variable (e.g., latitude, age) into categorical bins to reduce overfitting risk or encode domain structure. (Ch 8, 9)

**CAP theorem (Brewer's theorem)** — A distributed system cannot simultaneously guarantee Consistency, Availability, and Partition resilience. (Ch 2)

**Causality constraint** — A model input must be knowable at the time a prediction is needed; fields unknowable at serving time (e.g., actual flight distance at takeoff) cannot be used in training. (Ch 2, 4, 8, 11)

**Cloud Bigtable** — Massively scalable NoSQL key-value store; row-wise, sorted by key, sharded into tablets backed by Cloud Storage. (Ch 11)

**Cloud Dataflow** — Fully managed, serverless, autoscaling execution service for Apache Beam batch and streaming pipelines. (Ch 4, 11)

**Cloud Dataproc** — Managed Hadoop/Spark/Pig/Hive/Presto cluster service on GCP; supports ephemeral, job-scoped clusters and serverless Spark batches. (Ch 6, 7)

**Cloud Run** — Serverless, autoscaling execution environment for containerized applications; used for ingestion and other HTTP-triggered jobs. (Ch 2)

**Cloud Scheduler** — Serverless, cron-like job scheduler that invokes HTTP endpoints, Pub/Sub topics, or other targets on a schedule. (Ch 2)

**Confusion matrix / contingency table** — 2×2 (or larger) table of predicted vs. actual outcomes (true/false positive/negative) used to evaluate a threshold-based decision rule. (Ch 3, 5, 12)

**Continuous training** — Repeatedly retraining (from scratch) or fine-tuning (from a checkpoint) a production ML model as new data arrives, since real-world entities and relationships drift over time. (Ch 12)

**Curse of dimensionality** — Statistical reliability of bin-based estimation breaks down as the number of predictor variables (and hence bins) grows, since each bin's sample size shrinks. (Ch 7)

**Cumulative distribution function (CDF)** — P(X < x), bounded [0,1]; the probability of observing a value less than a threshold; easier to use for decision thresholds than the PDF. (Ch 1)

**Data split column (custom/predefined)** — A column explicitly assigning each row to TRAIN/VALIDATE/TEST (e.g., via `FARM_FINGERPRINT` hashing of a date), used instead of random splitting when rows are correlated. (Ch 5, 8, 9, 10, 11, 12)

**ELT (extract-load-transform)** — Architecture that loads raw data as-is into a data warehouse first, then transforms it via SQL views — the recommended default over ETL. (Ch 1, 2)

**Embedding** — A learned low-dimensional dense vector representation of a high-cardinality categorical feature, capturing similarity between categories (unlike one-hot encoding). (Ch 7, 8, 9)

**Feature cross** — Concatenating two or more (often bucketized/categorical) features into a single combined categorical variable to capture interaction effects. (Ch 8, 9)

**Hub-and-spoke architecture** — Ingestion pattern where raw data lands in a central enterprise data warehouse (the hub) and is transformed per-use-case downstream (the spokes). (Ch 2)

**Hyperparameter tuning (Vizier)** — Bayesian-optimization-based search over a specified range of hyperparameters (e.g., L2 regularization, tree depth, batch size), run via BigQuery ML's `hparam_range`/`num_trials` or Vertex AI's `HyperparameterTuningJob`. (Ch 8, 10)

**Logit / log-odds** — The real-valued linear combination `L = Σw·x + b` fit by logistic regression, related to probability by `P(Y) = 1/(1+e^-L)`. (Ch 7, 9)

**MapReduce** — Two-phase distributed processing model (map emits intermediate key-value pairs; reduce merges values per key), the algorithmic basis of Hadoop. (Ch 2, 6)

**MLOps** — The discipline of making ML model development reproducible, version-controlled, automatable, and monitorable from dataset creation through deployment. (Ch 10)

**One-hot encoding** — Representing a categorical variable with K values as K (or K−1) independent binary features; causes a feature-count explosion for high-cardinality columns. (Ch 7, 8, 9)

**Pub/Sub** — GCP's serverless, at-least-once, globally scalable messaging service, the standard ingestion point for streaming event pipelines. (Ch 4, 11)

**Scaling up / out / in-place** — Three strategies for handling datasets too large for one machine: get a bigger machine (up), shard data across nodes (out), or keep data centralized and ship compute to it over a high-bandwidth network (in-place, GCP's default). (Ch 2, 6)

**Serverless** — A service where the user supplies only code, never managing the lifecycle of the machines that run it (BigQuery, Dataflow, Cloud Run, Vertex AI, serverless Spark). (Ch 1)

**Shapley / sampled-shapley explanation** — A feature-attribution technique apportioning credit for a specific prediction among its input features; added to a Vertex AI model purely via deployment configuration. (Ch 10)

**Sliding window** — A time window (e.g., 60 minutes) that advances by a smaller period (e.g., 5 minutes), producing overlapping panes used for moving-average streaming aggregation. (Ch 4, 11)

**Three-sigma rule** — 99.73% of values in a Gaussian distribution lie within μ ± 3σ; used to derive a minimum-sample-size threshold (~370) for trusting a per-group statistic. (Ch 5)

**Training-serving skew** — A mismatch between how a feature is computed during model training versus during live prediction, causing silent model degradation; mitigated by shared transform code (BigQuery ML TRANSFORM clause, shared Beam modules). (Ch 2, 7, 8, 9, 11)

**TRANSFORM clause (BigQuery ML)** — A clause in `CREATE MODEL` that bakes feature derivation (e.g., extracting hour from a timestamp) into the model itself, so `ML.PREDICT` callers only supply raw columns. (Ch 8)

**Vertex AI Endpoint vs. Model** — A Model is an uploaded, versioned trained-model artifact; an Endpoint is the stable, reusable URL that splits traffic across one or more deployed Models. (Ch 9, 10)

**Vertex AI Workbench** — GCP's fully managed Jupyter notebook service, integrated with BigQuery, Cloud Storage, Dataflow, and Vertex AI training/serving. (Ch 5)

**Watermark** — A streaming pipeline's estimate of the oldest unprocessed record's event time; determines when a time window "closes" and governs late-data handling. (Ch 11)

**Wide-and-deep model** — A hybrid architecture combining a linear model over sparse/categorical features (the "wide" part) with a deep neural network over continuous/embedded features (the "deep" part), concatenated before the output layer. (Ch 9, 10, 11)

**XGBoost / gradient boosted trees** — An ensemble of decision trees where successive trees are trained to focus on previously misclassified examples (boosting); strong default for structured/tabular data. (Ch 8, 9)
