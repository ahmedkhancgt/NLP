# Chapter 4: Monitoring and Anomaly Detection for Your Data Pipelines

## Core Idea
Testing alone cannot catch "unknown unknowns"; true data reliability requires monitoring and anomaly detection across the five pillars of data observability (freshness, volume, distribution, schema, lineage), tuned using precision/recall trade-offs rather than naive "accuracy."

## Frameworks Introduced
- **The Five Pillars of Data Observability**: freshness, volume, distribution, schema, lineage — the complete set of dimensions a data quality monitor should track.
  - When to use: as the organizing checklist for building any data quality monitoring system.
  - How: build detectors per pillar (freshness = time since last update; distribution = null/zero rates and value ranges; schema = structural diffs over time; lineage = upstream/downstream dependency mapping).
- **Known Unknowns vs. Unknown Unknowns**: known unknowns are predictable issues (nulls, expected freshness, scheduled schema changes) coverable by tests; unknown unknowns are unpredictable issues that evade even comprehensive testing.
  - When to use: deciding whether to invest in testing (known unknowns) or monitoring/anomaly detection (unknown unknowns).
  - How: catalog predictable risks as tests; rely on statistical/ML-based anomaly detection for everything else.
- **Precision/Recall/F-score framework for anomaly detectors**: Precision = TP/(TP+FP); Recall = TP/(TP+FN); Fβ weights recall as β times as important as precision.
  - When to use: tuning any anomaly detection threshold or model — "accuracy" is misleading for rare-event detection (AIDS-test analogy: 99.6% accuracy by always predicting "No").
  - How: choose β based on the cost asymmetry of false positives vs. false negatives for your use case (e.g., missile alerts need high recall, β>1).

## Key Concepts
- **Anomaly detection**: identifying events/observations that deviate from the norm (e.g., fraud, website crashes).
- **Freshness**: how recently a data asset was updated; measured via "days/hours since last update" metrics.
- **Distribution**: the shape and expected range of values in a field (null rate, zero rate, quantiles, mean/std).
- **Schema change**: any alteration to data's structural blueprint (new/removed columns, type changes) — a major, hard-to-manually-track source of breakage.
- **Lineage**: the dependency graph connecting upstream sources to downstream consumers; the "most holistic" pillar because it supplies root-cause context.
- **Seasonality**: predictable periodic fluctuation in a time series (e.g., weekly, yearly) that naive thresholding must account for to avoid false positives.
- **False positive / False negative**: alerting when nothing is wrong ("crying wolf") vs. missing a real issue ("sleeping guard dog") — the two failure modes of any detector.
- **Rolling average thresholding**: comparing a current metric to its own trailing average (e.g., 2-week rolling avg) rather than a fixed threshold, to reduce noise from gradual drift.

## Mental Models
- The car-maintenance analogy: routine checks (testing) can't catch a random flat tire (unknown unknown) — you need real-time monitoring (dashboard warning lights) for that.
- The "boy who cried wolf" (high false-positive, low false-negative detector) vs. "sleeping guard dog" (low false-positive, high false-negative detector) — the two failure extremes any detector must balance.
- Threshold tuning is a "Goldilocks Zone" problem: too sensitive and most incidents are noise; too conservative and real incidents are missed.
- Lineage-equipped root cause explanations ("eccentricity field added upstream → min/max temp extremes → habitability zero-rate spike") are strictly more actionable than isolated anomaly flags.

## Anti-patterns
- **Optimizing for raw "accuracy" in anomaly detection**: with rare events, a detector that always predicts "normal" can show 99%+ accuracy while being useless (the AIDS-test example: 99.6% accuracy by always saying "No").
- **Using a single fixed global threshold without considering seasonality**: naive z-score/Gaussian approaches assume independent random samples, which business data (correlated, day-of-week-biased) violates.
- **Treating every statistically "anomalous" point as "interesting"**: not all 3-sigma deviations represent genuine data downtime worth alerting on.
- **Alerting without lineage context**: an isolated anomaly notification ("habitability zero-rate jumped") is far less useful than one tied to its root cause (an upstream schema change).
- **One-size-fits-all models across heterogeneous lake entrypoints**: different source formats (Postgres, logs, APIs) need different model architectures/hyperparameters, not a single blanket detector.

## Code Examples
```sql
-- Freshness anomaly detector: flag dates where data is more than 1 day stale
WITH UPDATES AS (
  SELECT DATE_ADDED, COUNT(*) AS ROWS_ADDED
  FROM EXOPLANETS GROUP BY DATE_ADDED
),
NUM_DAYS_UPDATES AS (
  SELECT DATE_ADDED,
    JULIANDAY(DATE_ADDED) - JULIANDAY(LAG(DATE_ADDED) OVER (ORDER BY DATE_ADDED)) AS DAYS_SINCE_LAST_UPDATE
  FROM UPDATES
)
SELECT * FROM NUM_DAYS_UPDATES WHERE DAYS_SINCE_LAST_UPDATE > 1;

-- Distribution anomaly: null-rate spike detection vs. a 2-week rolling average
WITH NULL_RATES AS (
  SELECT DATE_ADDED,
    CAST(SUM(CASE WHEN AVG_TEMP IS NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) AS AVG_TEMP_NULL_RATE
  FROM EXOPLANETS GROUP BY DATE_ADDED
),
NULL_WITH_AVG AS (
  SELECT *, AVG(AVG_TEMP_NULL_RATE) OVER (ORDER BY DATE_ADDED ASC ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS TWO_WEEK_ROLLING_AVG
  FROM NULL_RATES
)
SELECT * FROM NULL_WITH_AVG WHERE AVG_TEMP_NULL_RATE - TWO_WEEK_ROLLING_AVG > 0.3;

-- Schema change detection via a versioned columns table
WITH CHANGES AS (
  SELECT DATE, COLUMNS AS NEW_COLUMNS, LAG(COLUMNS) OVER (ORDER BY DATE) AS PAST_COLUMNS
  FROM EXOPLANETS_COLUMNS
)
SELECT * FROM CHANGES WHERE NEW_COLUMNS != PAST_COLUMNS ORDER BY DATE ASC;
```
- **What it demonstrates**: SQL-native implementations of freshness, distribution, and schema-change detectors — the three most directly operationalizable of the five observability pillars, each turned into a "detector" by adding a threshold parameter.

## Reference Tables
| Outcome | Predicted Anomalous | Predicted Not Anomalous |
|---|---|---|
| Actually Anomalous | True Positive | False Negative |
| Actually Normal | False Positive | True Negative |

| Metric | Formula | Meaning |
|---|---|---|
| Precision | TP / (TP + FP) | Of alerts raised, how many were real |
| Recall | TP / (TP + FN) | Of real issues, how many were caught |
| Fβ | weighted harmonic mean | β>1 favors recall; β<1 favors precision; β=1 (F1) balances both |

| ML/Stats Tooling | Purpose |
|---|---|
| Facebook Prophet | Seasonality-aware forecasting for time series |
| TensorFlow / PyTorch / scikit-learn | General ML libraries (autoencoders, ARIMA, isolation forest, k-NN) |
| MLflow | Experiment tracking and model registry/reproducibility |
| TensorBoard | Visualization of training metrics |

## Worked Example
Using a mock EXOPLANETS/HABITABLES database: a distributional anomaly surfaces when average `habitability` unexpectedly halves (from ~0.5 to ~0.25) starting in August 2020. Checking the null rate shows nothing unusual, but checking the **zero rate** for `habitability` reveals it jumps from 0% to ~37-46% starting 2020-07-19. Cross-referencing lineage shows this is the exact date a schema change added `eccentricity` and `atmosphere` fields to the upstream `EXOPLANETS_EXTENDED` table — the new eccentricity values caused `min_temp`/`max_temp` in the downstream `HABITABLES` table to take extreme values, which zeroed out `habitability`. The lineage-informed explanation ("schema change → extreme temps → zeroed habitability") let an engineer resolve the issue far faster than the isolated anomaly report alone ("zero rate jumped") would have.

## Key Takeaways
1. Build detectors for all five observability pillars (freshness, volume, distribution, schema, lineage) — no single pillar is sufficient alone.
2. Never evaluate anomaly detectors on raw accuracy for rare-event problems; use precision, recall, and a task-appropriate Fβ score instead.
3. Pick β based on real-world cost asymmetry: prioritize recall (β>1) when missed incidents are costly (e.g., safety alerts), prioritize precision (β<1) when false alarms burn trust/time.
4. Use rolling averages and seasonality-aware models rather than fixed global thresholds to reduce false positives from predictable periodic patterns.
5. Pair anomaly alerts with lineage data to supply root-cause context — an alert without lineage is a symptom report, not a diagnosis.
6. Design monitors differently for warehouses (fewer, more uniform entrypoints) vs. lakes (many heterogeneous entrypoints requiring per-source models or ensembles).

## Connects To
- **Ch 3**: Picks up where testing (known unknowns) leaves off, covering unknown unknowns via monitoring.
- **Ch 6**: Builds field-level lineage systems from scratch, extending the lineage pillar introduced here.
- **Ch 7**: Expands end-to-end lineage architecture in depth.
- **Statistics / ML**: central limit theorem, Gaussian/z-score methods, ARIMA, clustering (k-NN, isolation forest) — borrowed directly from classical statistics and ML.
