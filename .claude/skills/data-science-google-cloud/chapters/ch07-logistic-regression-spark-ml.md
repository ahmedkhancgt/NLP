# Chapter 7: Logistic Regression Using Spark ML

## Core Idea
Logistic regression fits a smooth, multivariable function (weighted sum of features passed through a logistic/sigmoid transform) to overcome the curse of dimensionality that breaks down the Chapter 6 binning approach, but a rigorous ML workflow requires systematic feature selection/engineering via an experimental framework (train/held-out/test split), careful handling of scale and categorical variables, and awareness that model productionization — not training — is where Spark ML becomes painful.

## Frameworks Introduced
- **Logistic regression as classification**: fit a logit (weighted sum of features) and pass it through the logistic function to get a bounded [0,1] probability.
  - When to use: predicting a probability from several continuous/numeric predictors, overcoming the curse of dimensionality that limits bin-based classifiers.
  - How: `P(Y) = 1/(1+e^-L)` where `L = w0*x0 + w1*x1 + ... + b`; train with `LogisticRegressionWithLBFGS.train(examples, intercept=True)` on `LabeledPoint(label, [features])`; `clearThreshold()` to get raw probabilities instead of 0/1 predictions, `setThreshold(p)` to get a binary decision at a chosen cutoff.
- **Three-way data split (train / held-out / test)**: separate datasets for weight-fitting, feature-selection decisions, and final unbiased evaluation.
  - When to use: any time you need to make modeling decisions (which features to keep, which transform to use) beyond simply fitting weights.
  - How: hold back the test set entirely (as in Ch. 5); further split the training set itself into a training portion and a held-out portion (e.g., via a repeatable random column with a fixed seed) used only to decide whether a feature/transform improves performance; never use the test set for any decision that could leak into model selection.
- **Systematic feature selection**: start with all candidate features, remove one at a time, and keep only those whose removal measurably hurts performance.
  - When to use: deciding which of several candidate inputs actually improve a model.
  - How: train once with all features, then retrain with each feature removed; compare a distribution-aware metric (RMSE between predicted probability and true label, or logistic loss/cross-entropy) on the held-out set; keep the feature only if removing it increases error by more than a chosen threshold (e.g., 0.5%) — preferred over the "add features one at a time" approach because it captures interaction effects.
- **Feature engineering for periodic/categorical raw inputs**: transform raw inputs the model can't use directly into usable features.
  - When to use: timestamps (periodic, e.g., hour of day) or category codes (e.g., airport) as candidate predictors.
  - How: for periodic variables, encode as `sin(theta)`/`cos(theta)` (preserves cyclic distance) or bucketize into human-meaningful categories (e.g., night/morning/mid-day/evening); for categorical variables, one-hot encode via `StringIndexer` + `OneHotEncoder`, dropping one category to avoid linear dependence (N categories → N−1 features).

## Key Concepts
- **Curse of dimensionality**: as more variables are added to a bin/quantization-based approach (Ch. 6), bins become sparser and statistically unreliable — motivates moving to a parametric model like logistic regression.
- **Logit / log-odds**: `L = log(P(Y)/(1-P(Y)))`, the real-valued quantity that logistic regression actually fits via linear combination of inputs; probability near 0 or 1 requires large logit swings to move, while near 0.5 small logit changes move probability a lot.
- **Training vs. prediction compute profile**: training is a distributed, batch, CPU-bound optimization (no GPU needed for small models like 3-weight logistic regression); prediction is a cheap single-machine computation, needing GPUs only for very large models or extreme low-latency/high-QPS serving.
- **Metric gameability**: contingency-table metrics (accuracy, precision, recall at a fixed threshold) can be tuned arbitrarily by shifting the decision threshold — not valid for choosing between two models; use a full-distribution metric (RMSE of probability vs. label, logistic loss, cross-entropy) instead for feature selection/hyperparameter tuning.
- **Feature vs. input**: the *input* is the raw value supplied (e.g., a UTC timestamp); the *feature* is what the model actually trains on, possibly a transform or combination of inputs (e.g., local-time-corrected hour of day, bucketized).
- **One-hot encoding cost explosion**: encoding a categorical variable with K distinct values (e.g., ~300 airports) creates K (or K−1) new sparse binary features — can overwhelm cluster memory even for datasets that otherwise fit comfortably, motivating later embedding-based dimensionality reduction (Ch. 10).
- **Soft quota**: GCP's regional CPU/resource limits are deliberately conservative defaults meant to guard against billing surprises; easily raised via the Quotas page, with (per the chapter's experience) approval often arriving within minutes.

## Mental Models
- **Occam's razor / principle of parsimony for ML features**: every added variable increases overfitting risk, data-dependency fragility (missing/undefined values break the model), and retraining burden — prefer the smallest feature set with comparable performance.
- **"There is no substitute for reading the documentation"**: Spark's logistic-loss formulation requires labels in {0,1} even though its underlying math uses {-1,+1}-style notation — a concrete reminder that different ML frameworks use mathematically-equivalent but operationally-different conventions.
- **Bad values are a symptom, not noise to discard**: missing `TAXI_OUT`/`ARR_DELAY` values traced back to canceled/diverted flights — the chapter explicitly rejects blanket NULL-dropping in favor of filtering by the root-cause flag (`CANCELLED`/`DIVERTED`).
- **"Easy to get started, difficult to productionize" (Spark ML)**: feature transforms (scaling, bucketizing, one-hot encoding) must be replicated identically at prediction time, creating a training-serving-skew risk and a hard dependency on running Spark wherever predictions are served — a key motivation for later chapters' move to BigQuery ML / Vertex AI.

## Anti-patterns
- **Dropping NULL rows to "fix" a data problem**: masks the real cause (canceled/diverted flights) instead of addressing it with the appropriate filter — always investigate before discarding.
- **Judging model quality by number of meetings canceled alone**: this secondary metric can be matched or beaten by a worse-calibrated model (Spark logistic regression had 3% better accuracy-at-the-margin than Ch. 6's Bayes model yet canceled more meetings) — not a reliable standalone comparison criterion.
- **Using the test set to make any model-development decision (feature selection, hyperparameter tuning)**: contaminates the test set's independence; always use a separate held-out split carved from the training data for this purpose.
- **One-hot encoding a high-cardinality categorical variable without considering cost**: ~300 airport categories alone caused out-of-memory errors and disk swapping on a 50-node cluster — real production categorical variables (e.g., ad-click datasets) can have thousands of unique values, requiring embeddings instead.
- **Treating RMSE/metric improvement thresholds as objective facts**: "enough" improvement (the chapter uses 0.5%) is a judgment call balancing accuracy gains against the operational cost (retraining burden, runtime, agility loss) of extra features — not a universal constant.
- **Running experimentation on a cold/scaling-up cluster repeatedly**: wastes 5+ minutes per run on startup/autoscaling; for iterative experimentation, use a pre-sized, already-running cluster (with an autoscaling policy and idle timeout) instead of serverless Spark per run.

## Code Examples
```python
def to_example(raw_data_point):
    return LabeledPoint(
        float(raw_data_point['ARR_DELAY'] < 15),  # on-time?
        [raw_data_point['DEP_DELAY'], raw_data_point['TAXI_OUT'], raw_data_point['DISTANCE']])
examples = traindata.map(to_example)
lrmodel = LogisticRegressionWithLBFGS.train(examples, intercept=True)
```
- **What it demonstrates**: the canonical Spark ML pattern — a `to_example` converter producing `LabeledPoint`s, then training logistic regression via L-BFGS; weights/intercept are directly interpretable (e.g., negative departure-delay weight = higher delay lowers on-time probability).

```python
traindays = traindays.withColumn("holdout", rand(SEED) > 0.8)  # 20% held out
# train on is_train_day == 'True' AND holdout == False
# evaluate feature choices on is_train_day == 'True' AND holdout == True
```
- **What it demonstrates**: carving a repeatable (seeded) held-out split out of the training data itself, distinct from the Chapter 5 test split, used purely for feature-selection decisions.

```python
def to_example(raw_data_point):
    def get_local_hour(timestamp, correction):
        t = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S') + datetime.timedelta(seconds=correction)
        theta = np.radians(360 * t.hour / 24.0)
        return [np.sin(theta), np.cos(theta)]
    features = [raw_data_point['DEP_DELAY'], raw_data_point['TAXI_OUT']]
    features.extend(get_local_hour(raw_data_point['DEP_TIME'], raw_data_point['DEP_AIRPORT_TZOFFSET']))
    return LabeledPoint(float(raw_data_point['ARR_DELAY'] < 15), features)
```
- **What it demonstrates**: encoding a periodic feature (hour of day, corrected to local time) as sin/cos pairs so the model doesn't treat hour 23 and hour 1 as far apart.

```python
def add_categorical(df, train=False):
    from pyspark.ml.feature import OneHotEncoder, StringIndexer
    if train:
        indexer = StringIndexer(inputCol='ORIGIN', outputCol='origin_index')
        index_model = indexer.fit(df)
    indexed = index_model.transform(df)
    encoder = OneHotEncoder(inputCol='origin_index', outputCol='origin_onehot')
    return encoder.transform(indexed)
```
- **What it demonstrates**: one-hot encoding a categorical variable (airport code) with Spark's `StringIndexer`+`OneHotEncoder`, and the critical requirement to reuse the fitted `index_model` (not refit) at evaluation/prediction time to avoid training-serving skew.

## Reference Tables
| Experiment | Features | Outcome |
|---|---|---|
| 1 (baseline) | DEP_DELAY, DISTANCE, TAXI_OUT | RMSE 0.205 |
| Remove DISTANCE | DEP_DELAY, TAXI_OUT | Distance had least impact of the three |
| Scaled values | min/max linear scaling to [-1,1] | No change in RMSE (linear model handles scale well) |
| Clipped values | Clamp outliers to bounds | No improvement (slightly worse) |
| + hour of day (raw/sin-cos/bucketed) | — | No RMSE improvement — airline scheduling already accounts for time of day |
| + one-hot airport (ORIGIN) | ~300 categories → ~300 features | Cluster ran out of memory; abandoned |

| Approach | Decision threshold insight |
|---|---|
| Contingency-table accuracy/precision/recall | Gameable by shifting the probability threshold — not valid for model comparison |
| RMSE / logistic loss / cross-entropy | Uses full probability distribution — valid for feature selection and model comparison |

## Worked Example
Builds a 3-variable logistic regression model for the flight-delay decision: (1) on the Dataproc cluster from Ch. 6, load training data via Spark SQL joined against `trainday`, discover inconsistent row counts across columns, trace to canceled/diverted flights (not NULLs to drop), and filter by `CANCELLED`/`DIVERTED` flags instead; (2) convert rows to `LabeledPoint`s with features `[DEP_DELAY, TAXI_OUT, DISTANCE]` and label = on-time (ARR_DELAY < 15); (3) train with `LogisticRegressionWithLBFGS`, inspect weights (departure delay has strongest, negative effect; distance has a small positive effect), save/load the model via `lrmodel.save()`/`.load()`; (4) evaluate on the Chapter 5 test set, finding overall accuracy figures are inflated by easy non-marginal cases, so re-evaluate specifically near the 0.7 probability threshold, getting ~73% correct-to-proceed (close to the 70% target) but canceling slightly more meetings than Chapter 6's Bayes model despite similar accuracy — illustrating that "fewer meetings canceled" is not a reliable standalone comparison metric; (5) build a proper train/held-out/test experimentation framework with a seeded `holdout` column; (6) create a large (50-node) pre-sized, autoscaling-policy cluster (requesting a CPU quota increase) to run many feature-selection experiments quickly, repartitioning data explicitly to actually use all nodes; (7) systematically remove one feature at a time, discovering DISTANCE matters least; (8) test scaling and clipping transforms on the inputs — neither affects RMSE for this linear model; (9) engineer an hour-of-day feature (raw, sin/cos, and bucketed variants) — none improve RMSE, since airline scheduling already bakes in time-of-day effects; (10) attempt one-hot encoding the ~300-value ORIGIN airport categorical variable, which overwhelms the 50-node cluster's memory and is abandoned in favor of embeddings (deferred to Chapter 10's TensorFlow work).

## Key Takeaways
1. Move from bin-based classifiers to logistic regression (or other parametric models) once the number of predictor variables makes binning statistically unreliable (the curse of dimensionality).
2. Use a three-way split — train / held-out (for feature decisions) / test (for final evaluation only) — and never let test-set performance influence model-selection choices.
3. Choose feature-selection/comparison metrics that use the full probability distribution (RMSE, logistic loss, cross-entropy), not threshold-dependent contingency-table metrics, which are easily gamed.
4. Encode periodic variables (hour, day-of-week) via sin/cos pairs or domain-informed bucketing; encode categorical variables via one-hot encoding, always reusing the fitted index/encoder at prediction time to avoid training-serving skew.
5. Watch for one-hot-encoding cost explosions on high-cardinality categorical variables; consider embeddings for large vocabularies.
6. For iterative ML experimentation, use a pre-sized cluster with an autoscaling policy and idle-timeout rather than paying the serverless-Spark startup/autoscale cost on every run.
7. Recognize that Spark ML's biggest pain point is productionization (reproducing transforms exactly at serving time, hard Spark runtime dependency for predictions) rather than training itself — a key motivator for BigQuery ML (Ch. 8) and Vertex AI (Ch. 9–10).

## Connects To
- **Ch 6**: Directly follows on from and quantitatively compares against the two-variable Bayesian classifier; reveals that distance (central to Ch. 6's binning) was actually the least important variable.
- **Ch 5**: Reuses the `dsongcp.trainday` test split and extends it with a new held-out split for feature-selection experiments.
- **Ch 8 & 9**: BigQuery ML and Vertex AI are introduced specifically to resolve the scalability and productionization pain points (one-hot encoding cost, training-serving-skew risk) surfaced here.
- **Ch 10**: Embeddings (for high-cardinality categorical variables like airport) and distributed TensorFlow are the deferred solutions to problems first hit in this chapter.
- **Ch 11**: Real-time aggregate features (e.g., actual vs. average taxi-out time) suggested here as a future improvement are built using Apache Beam in Chapter 11.
