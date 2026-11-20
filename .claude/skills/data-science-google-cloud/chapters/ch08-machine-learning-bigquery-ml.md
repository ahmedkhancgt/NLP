# Chapter 8: Machine Learning with BigQuery ML

## Core Idea
BigQuery ML lets you train, evaluate, tune, and batch-predict machine learning models entirely in SQL directly on data in the warehouse — handling categorical variables (like airport codes) and nonlinear models (XGBoost, AutoML) effortlessly at a scale that overwhelmed Spark ML in Chapter 7 — and is recommended as the first ML framework to try on tabular data, with data/feature improvements prioritized over hyperparameter tuning or AutoML.

## Frameworks Introduced
- **BigQuery ML CREATE MODEL workflow**: train a model as a SQL statement over a SELECT query.
  - When to use: any tabular ML problem where data already lives in (or can be staged into) BigQuery.
  - How: `CREATE OR REPLACE MODEL dataset.model_name OPTIONS(input_label_cols=[...], model_type='logistic_reg'|'boosted_tree_classifier'|'automl_classifier', data_split_method='custom', data_split_col='is_eval_day') AS SELECT ...`; inspect with `ML.TRAINING_INFO`, `ML.WEIGHTS`, `ML.EVALUATE`, `ML.PREDICT`.
- **Custom data split column**: reuse the Chapter 5 date-based train/test split inside BigQuery ML instead of relying on default random splitting.
  - When to use: whenever rows are correlated (e.g., same-day flights) and random splitting would leak information.
  - How: join against the `trainday` table, derive a boolean/string split column (`is_eval_day`, or three-way `TRAIN`/`EVAL`/`TEST` for hyperparameter tuning), and pass `data_split_method='custom', data_split_col='...'`.
- **Model-type swap for nonlinear learning**: upgrade from linear logistic regression to gradient-boosted trees or AutoML by changing one option.
  - When to use: once simple linear models plateau and the problem may benefit from nonlinear feature interactions.
  - How: change only `model_type` (`'logistic_reg'` → `'boosted_tree_classifier'` → `'automl_classifier'`) in the `CREATE MODEL` statement; no other pipeline code changes.
- **Vizier-based hyperparameter tuning**: let Bayesian optimization search a hyperparameter space instead of grid search.
  - When to use: tuning model hyperparameters (e.g., `l2_reg`, `max_tree_depth`) where exhaustive grid search is too slow.
  - How: specify `num_trials=N` and `hparam_range(min, max)` for each hyperparameter in `CREATE MODEL` options; requires a three-way TRAIN/EVAL/TEST split so the tuning process has an independent EVAL set distinct from the final TEST set; inspect results via `ML.TRIAL_INFO`.
- **TRANSFORM clause for training-serving-skew-safe feature engineering**: bake feature transformations into the model itself.
  - When to use: any derived feature (e.g., extracting hour-of-day from a timestamp) that must be computed identically at training and prediction time.
  - How: add a `TRANSFORM(* EXCEPT(col), EXTRACT(...) AS feature, ...)` clause between `CREATE MODEL` and `OPTIONS`; the transformation is automatically reapplied whenever `ML.PREDICT` is called with raw input columns, eliminating the risk of divergent client-side feature computation.

## Key Concepts
- **XGBoost / gradient boosted trees**: an ensemble of decision trees where each subsequent tree is trained to focus on (via gradient-weighted boosting) the examples the previous trees misclassified; generally strong on structured/tabular data.
- **L2 regularization**: a penalty on large model weights that curbs overfitting to outliers by discouraging the optimizer from "warping" the solution space around them; BigQuery ML's default is 1.0 but is dataset-dependent.
- **Max tree depth**: controls how many if-then splits each tree in a boosted-tree ensemble can have; deeper trees are individually more expressive but risk overfitting; default 6.
- **Feature cross**: concatenating multiple categorical/bucketized features (e.g., hour bucket × weekday/weekend × origin airport) into one combined categorical variable via `ML.FEATURE_CROSS`, to capture interaction effects a linear/additive model can't — but at the cost of combinatorial sparsity.
- **Low-code vs. no-code vs. full-code**: no-code (AutoML, point-and-click) is easiest but least flexible; low-code (BigQuery ML, SQL-driven) covers most common ML use cases while delegating real work to C++/TensorFlow under the hood; full code (TensorFlow/Vertex AI) is needed once you hit a capability the low-code layer doesn't expose (e.g., embeddings for high-cardinality feature crosses).
- **"More data beats a better model"**: the chapter's explicit prioritization heuristic — exhaust feature/data improvements (new signals, time-window features, time-based features) before investing in hyperparameter tuning or AutoML, since tuning gains were comparable to just adding two well-chosen features.

## Mental Models
- **Causality creeps back in with time-windowed features**: computing a rolling average (e.g., average taxi-out time per airport, or trailing-hour average delay) risks leaking future/global information (using the full year's average rather than only data available at prediction time) — the chapter explicitly flags its own mistake here as a teaching example.
- **Time features as a cheaper substitute for time-window features**: extracting hour-of-day and day-of-week as plain input features recovered nearly the same RMSE improvement (0.2043) as the causality-risky rolling-average approach (0.2040), with far less engineering complexity — prefer the simpler option when it's competitive.
- **Diminishing returns signal when to stop feature engineering**: once successive feature/transform experiments (categorical day-of-week, feature cross) stop moving RMSE meaningfully, it's time to stop rather than keep iterating.
- **Every low-code/no-code tool has a breaking point**: BigQuery ML's inability (at time of writing) to create embeddings for sparse high-cardinality feature crosses is the specific wall hit here — the lesson is to verify upfront that you can "drop down to code" when a low-code tool's abstraction runs out.

## Anti-patterns
- **Computing a feature's aggregate (e.g., average taxi-out per airport) over the full dataset including evaluation/future data**: leaks information from flights the model shouldn't yet "know about" at prediction time — a causality violation; the chapter catches and calls out its own mistake here.
- **Extracting derived features (hour, day-of-week) in the training SELECT statement without a TRANSFORM clause**: creates an implicit contract that the prediction client must replicate exactly (same timezone, same encoding) — a direct training-serving-skew risk; use TRANSFORM instead so the model owns the derivation.
- **Jumping to hyperparameter tuning or AutoML before exhausting feature/data improvements**: both are expensive (hours) and the chapter found their gains comparable to or smaller than well-chosen new features — sequence data improvements first.
- **Treating a numeric-looking categorical variable (day-of-week as 1–7) as purely numeric**: implies a false ordering (Saturday > Monday) — explicitly re-encode as a categorical string or binary weekday/weekend flag.
- **Replacing individual time features with only their feature cross**: loses the standalone hour/day signal; consider keeping the cross as an additional feature rather than a full replacement.

## Code Examples
```sql
CREATE OR REPLACE MODEL dsongcp.arr_delay_lm
OPTIONS(input_label_cols=['ontime'], model_type='logistic_reg',
        data_split_method='custom', data_split_col='is_eval_day') AS
SELECT
  IF(arr_delay < 15, 'ontime', 'late') AS ontime,
  dep_delay, taxi_out, distance,
  IF(is_train_day = 'True', False, True) AS is_eval_day
FROM dsongcp.flights_tzcorr f
JOIN dsongcp.trainday t ON f.FL_DATE = t.FL_DATE
WHERE f.CANCELLED = False AND f.DIVERTED = False
```
- **What it demonstrates**: the full BigQuery ML training pattern — label derivation, custom date-based split reuse, and cancel/divert filtering — in a single SQL statement.

```sql
CREATE OR REPLACE MODEL dsongcp.arr_delay_airports_xgh
OPTIONS(input_label_cols=['ontime'], model_type='boosted_tree_classifier',
        num_trials=5, l2_reg=hparam_range(0.5, 3.0), max_tree_depth=hparam_range(2, 10),
        data_split_method='custom', data_split_col='is_eval_day')
```
- **What it demonstrates**: Vizier-driven Bayesian hyperparameter search over L2 regularization and tree depth in 5 trials instead of a 200-combination grid search.

```sql
CREATE OR REPLACE MODEL dsongcp.arr_delay_airports_timefeatures
TRANSFORM(
   * EXCEPT(dep_time),
   EXTRACT(hour FROM dep_time) AS dep_hour,
   IF(EXTRACT(dayofweek FROM dep_time) BETWEEN 2 AND 6, 'weekday', 'weekend') AS dep_day
)
OPTIONS(input_label_cols=['ontime'], model_type='boosted_tree_classifier', l2_reg=2.5, max_tree_depth=10)
AS SELECT ..., dep_time, ...
```
- **What it demonstrates**: baking a timestamp-derived feature transform directly into the model via `TRANSFORM`, so `ML.PREDICT` callers only ever need to supply the raw `dep_time`, eliminating client-side feature-derivation drift.

```sql
SELECT ORIGIN,
  AVG(dep_delay) OVER (PARTITION BY ORIGIN ORDER BY UNIX_SECONDS(dep_time)
                        RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING) AS avg_dep_delay
FROM dsongcp.flights_tzcorr
```
- **What it demonstrates**: computing a trailing 1-hour moving average per origin airport using SQL window functions — powerful, but prone to the causality leak discussed in Key Concepts if not scoped carefully to only-training-available data.

## Reference Tables
| Model / feature set | RMSE | Notes |
|---|---|---|
| Logistic regression (dep_delay, taxi_out, distance) | 0.2131 | Baseline, matches Ch. 7 feature set |
| + origin/dest airport (categorical, no special handling needed) | 0.2098 | BigQuery handles ~300-value categoricals transparently |
| XGBoost (boosted_tree_classifier), default hyperparameters | 0.2072 | Nonlinear model improves on logistic regression |
| XGBoost + hyperparameter tuning (l2_reg=2.5, max_tree_depth=10) | 0.2043 | Biggest single improvement in the chapter |
| AutoML classifier | 0.1998 | Best raw number, but not directly comparable (default random split only) |
| + time-window features (avg taxi-out, trailing 1hr avg delay) | 0.2040 | Causality-risky; matches tuning-level improvement |
| + time features (hour, day-of-week) via TRANSFORM | 0.2043 | Simpler than time windows, nearly identical improvement |
| + day-of-week as categorical (weekday/weekend) | 0.2042 | Slight improvement over numeric encoding |
| + feature cross (hour bucket × day × origin) | 0.2043 | No improvement — sparsity problem, needs embeddings |

| Tooling tier | Flexibility | Example |
|---|---|---|
| No-code | Lowest, point-and-click | AutoML Tables (GUI) |
| Low-code | SQL-driven, common use cases covered | BigQuery ML |
| Full code | Maximum, but highest effort | TensorFlow / Vertex AI custom training |

## Worked Example
Rebuilds and extends the flight-delay model entirely in BigQuery ML: (1) replicate the Chapter 7 3-feature logistic regression model via `CREATE MODEL ... model_type='logistic_reg'`, reusing the `trainday` custom split; (2) interrogate it with `ML.TRAINING_INFO`, `ML.WEIGHTS`, `ML.EVALUATE` (threshold 0.7), and compute RMSE via `ML.PREDICT`; (3) add origin/destination airport as categorical features with zero special handling (unlike Spark's crash in Ch. 7), improving RMSE from 0.2131 to 0.2098; (4) swap to `model_type='boosted_tree_classifier'` (XGBoost) for a further improvement to 0.2072; (5) run Vizier-based hyperparameter tuning (`num_trials=5`, `hparam_range` on `l2_reg` and `max_tree_depth`) with a three-way TRAIN/EVAL/TEST split, landing on l2_reg=2.5, max_tree_depth=10 and RMSE 0.2043; (6) try `model_type='automl_classifier'` for RMSE 0.1998 (not directly comparable due to default random split); (7) pause AutoML/tuning and instead add data: validate via a bar chart that average taxi-out time differs meaningfully between on-time and late flights at different airports, then add `avg_taxi_out` (joined from a per-airport aggregate) and a trailing-1-hour windowed average delay feature, reaching RMSE 0.2040 — but catch a causality bug (the aggregate used evaluation/future data); (8) replace the risky time-window features with simple time features (hour, day-of-week) via a `TRANSFORM` clause, matching the improvement (RMSE 0.2043) with far less complexity and no causality risk; (9) refine day-of-week to a categorical weekday/weekend flag (RMSE 0.2042) and try a feature cross of bucketized hour × day-type × origin airport, which doesn't help due to sparsity (24×2×300 combinations) — identifying embeddings as the needed next tool, deferred to Chapter 9/10.

## Key Takeaways
1. Use BigQuery ML as the first ML framework to try on tabular data — it handles high-cardinality categorical variables and scale effortlessly where Spark ML (Ch. 7) struggled.
2. Reuse your date-based custom train/test split inside BigQuery ML via `data_split_method='custom'` rather than accepting default random splitting on correlated data.
3. Prioritize feature/data improvements over hyperparameter tuning or AutoML — they're often comparably impactful and far cheaper to iterate on.
4. Use the `TRANSFORM` clause to bake derived features into the model itself, eliminating training-serving skew for anything computed from raw input columns.
5. Watch for causality leaks when computing time-windowed aggregates (e.g., per-airport averages) — make sure the aggregate only reflects data available at prediction time.
6. When simple time features (hour, day-of-week) achieve comparable gains to complex time-windowed moving averages, prefer the simpler, causality-safer option.
7. Recognize when a low-code tool (BigQuery ML) hits its abstraction ceiling (e.g., no embeddings for sparse feature crosses) and plan to drop to full code (TensorFlow/Vertex AI) at that point.

## Connects To
- **Ch 5**: Reuses the `dsongcp.trainday` date-based split as BigQuery ML's custom data-split column.
- **Ch 7**: Directly replicates and extends the Spark ML feature set, explicitly succeeding where Spark overwhelmed on categorical airport encoding.
- **Ch 9 & 10**: Embeddings (needed for the feature-cross sparsity problem) and AutoML's custom split support are picked up in subsequent TensorFlow/Vertex AI chapters.
- **Ch 11**: Time-windowed features' causality complexity motivates revisiting them with a proper stream-analytics (Apache Beam) system in Chapter 11.
