# Chapter 5: Interactive Data Exploration with Vertex AI Workbench

## Core Idea
Exploratory data analysis (EDA) — using BigQuery for fast interactive querying of raw, unindexed data plus Vertex AI Workbench (managed Jupyter) for statistical visualization — reveals data-quality problems and underlying structure that upgrade a naive threshold model into a conditional, empirically-derived probabilistic model, which must then be validated on a held-out, properly-split test set.

## Frameworks Introduced
- **BigQuery + Vertex AI Workbench EDA workflow**: combine SQL-scale aggregation with notebook-based statistical visualization.
  - When to use: any large-dataset exploration where neither SQL alone (no graphing) nor a laptop notebook alone (can't scale) suffices.
  - How: run aggregation/filtering queries directly in BigQuery (fast, unindexed full-table scans); pull result sets into a Vertex AI Workbench-hosted Jupyter notebook via the `%%bigquery` magic or the `google.cloud.bigquery` client library into a Pandas DataFrame; visualize with seaborn/Matplotlib (violin plots, line plots with error bars).
- **Frequency-based outlier filtering (big-data outlier detection)**: replace value-based outlier thresholds with occurrence-frequency thresholds.
  - When to use: datasets large enough that rare-but-real values carry genuine signal (hundreds of thousands to millions of rows).
  - How: filter with `HAVING numflights > N` (grouped-count threshold, e.g., 370 from the three-sigma rule: 1/(1-0.9973)) instead of `WHERE value_column > threshold`; test sensitivity of the resulting model (e.g., linear-fit slope) across several candidate thresholds to confirm the exact cutoff doesn't matter much, only its order of magnitude.
- **Date-based train/test split via hashing**: split time-series-like data into independent train/test sets without random row shuffling.
  - When to use: datasets with intra-day correlation (e.g., weather/traffic affecting all same-day flights) where naive row-level shuffling would leak information between train and test.
  - How: get distinct dates, hash each with `FARM_FINGERPRINT`, assign to train/test by `MOD(hash, 100) < 70`; persist the split as a `trainday` BigQuery table (and export to CSV) so every downstream chapter/model reuses the identical, repeatable split.
- **Empirical (nonparametric) percentile modeling**: replace a Gaussian-assumption Z-score threshold with BigQuery's `APPROX_QUANTILES`.
  - When to use: whenever the normality assumption is doubtful or avoidable with enough data per group.
  - How: `APPROX_QUANTILES(ARR_DELAY, 101)[OFFSET(70)]` grouped by `DEP_DELAY` (with the same frequency `HAVING` filter) directly yields the 70th-percentile arrival delay for each departure-delay value, without assuming a distribution shape.

## Key Concepts
- **Exploratory Data Analysis (Tukey)**: a philosophy/approach (not a fixed method set) for testing assumptions, identifying important variables, discovering structure, building parsimonious models, and detecting outliers — contrasted with confirmatory/hypothesis-testing statistics.
- **EDA vs. dashboard creation**: EDA's audience is the data team itself (statistically sophisticated), aimed at developing insight before modeling; dashboards (Ch. 3) target end users, aimed at explaining a finished model.
- **Anscombe's Quartet**: four datasets with identical summary statistics (mean, variance, correlation, linear fit) but radically different shapes — the canonical argument that graphs, not just summary statistics, are essential.
- **Query execution stages (BigQuery "Execution details")**: wait / read / compute / write times per stage reveal whether a query is I/O-bound, compute-bound, or suffering worker skew; `HAVING` lets BigQuery's optimizer push filters to the earliest stage automatically.
- **Three-sigma rule**: 99.73% of values in a Gaussian distribution lie within μ ± 3σ; used to derive the 370-flights-per-group threshold (1/(1-0.9973) ≈ 370) for "enough examples to trust this value."
- **Violin plot**: a kernel-density (smoothed-histogram) visualization of a distribution's shape — used here to reveal that delayed flights form a distinct, skewed statistical regime vs. on-time flights.
- **Training set vs. test set independence**: splits must be representative of the whole population yet statistically independent — naive random row shuffling fails when rows are correlated (e.g., same-day flights share weather) and fails reproducibility (BigQuery's `RAND()`/row order aren't stable).

## Mental Models
- **"The term outlier is somewhat of a misnomer in big data"**: with enough data, rare values aren't noise to discard — they're signal to model; the filtering criterion should be "do we have enough examples to trust this value," not "is this value far from the mean."
- **Quality flags are themselves suspect**: dataset-provided quality flags often reflect outdated small-data statistical heuristics; prefer your own frequency-based analysis, but use flags that encode real environmental knowledge (e.g., "canceled," "diverted") as model inputs rather than blanket exclusion criteria.
- **A model's apparent accuracy depends on which population you evaluate against**: 95%/82% correctness over the whole test set looks far better than the 70% target because most departure delays aren't near the decision boundary; evaluating only at the marginal decision boundary (departure delay = 15 vs. 16 min) is the honest test of whether the model hits its designed target.
- **Split methodology must match data structure**: always ask "what real-world correlation exists between rows?" before choosing a train/test split strategy — the answer here (same-day correlation) rules out naive shuffling.

## Anti-patterns
- **Filtering out "outliers" by raw value threshold on large datasets**: throws away genuinely rare-but-real long-tail values that may carry business-critical signal; use frequency-of-occurrence filtering instead.
- **Trusting a dataset's quality flags blindly**: many are set by outdated small-sample statistical rules rather than domain understanding — verify independently when you have the data volume to do so.
- **Random row-level shuffling for train/test split on correlated data**: leaks information across train/test when rows share hidden grouping factors (e.g., day-of-flight weather); also non-reproducible in BigQuery since row order and `RAND()` aren't stable across runs.
- **Evaluating a threshold model only over the full population instead of at the decision margin**: produces misleadingly high accuracy numbers that don't reflect how well the model performs exactly where the decision is contested.
- **Developing custom/one-off plotting workflows across many languages and tools** (the chapter's own pre-notebook war story: C++ → Java → R → LaTeX → PDF → email): replaced by a single reproducible Jupyter notebook backed by BigQuery/Dataflow for the heavy computation.

## Code Examples
```sql
SELECT DEP_DELAY, AVG(ARR_DELAY) AS arrival_delay, COUNT(ARR_DELAY) AS numflights
FROM dsongcp.flights_tzcorr
GROUP BY DEP_DELAY
HAVING numflights > 370
ORDER BY DEP_DELAY
```
- **What it demonstrates**: frequency-based outlier filtering using `HAVING` (not `WHERE` on the value itself), letting BigQuery's optimizer push the filter early and avoiding Gaussian-outlier assumptions.

```python
%%bigquery depdelay
SELECT DEP_DELAY, AVG(ARR_DELAY) AS arrival_delay,
       STDDEV(ARR_DELAY) AS stddev_arrival_delay, COUNT(ARR_DELAY) AS numflights
FROM dsongcp.flights_tzcorr GROUP BY DEP_DELAY HAVING numflights > 370 ORDER BY DEP_DELAY
```
- **What it demonstrates**: the Jupyter `%%bigquery` magic pulling a BigQuery result directly into a named Pandas DataFrame inside Vertex AI Workbench, ready for seaborn/Matplotlib plotting.

```sql
SELECT FL_DATE,
  IF(ABS(MOD(FARM_FINGERPRINT(CAST(FL_DATE AS STRING)), 100)) < 70, 'True', 'False') AS is_train_day
FROM (SELECT DISTINCT(FL_DATE) AS FL_DATE FROM dsongcp.flights_tzcorr)
ORDER BY FL_DATE
```
- **What it demonstrates**: deterministic, repeatable 70/30 train/test split by hashing distinct dates (not rows) — persisted as `dsongcp.trainday` so every later chapter's model uses the identical split.

```sql
SELECT DEP_DELAY, APPROX_QUANTILES(ARR_DELAY, 101)[OFFSET(70)] AS arrival_delay_30th,
       COUNT(ARR_DELAY) AS numflights
FROM dsongcp.flights_tzcorr GROUP BY DEP_DELAY HAVING numflights > 370 ORDER BY DEP_DELAY
```
- **What it demonstrates**: nonparametric 70th-percentile estimation per departure-delay group using `APPROX_QUANTILES`, replacing the Gaussian Z-score approach and refining the decision threshold from 13 to 16 minutes.

## Reference Tables
| Train/test split method | Problem |
|---|---|
| Random row shuffle (`WHERE RAND() < 0.7`) | Non-reproducible row order/RAND(); leaks info when rows are correlated (e.g., same-day flights) |
| Split by calendar period (Jan–Sep vs. Oct–Dec) | Not representative if seasonal effects differ (e.g., summer vs. winter delay recovery) |
| Hash distinct dates, split 70/30 | Reproducible, representative, and respects intra-day correlation |

| Decision model | Threshold found | Basis |
|---|---|---|
| Ch. 3 contingency table | 10 min departure delay | Simple binary threshold over whole dataset |
| Ch. 5 Gaussian Z-score | 13 min | Assumes Normal distribution of arrival delay per departure delay |
| Ch. 5 empirical percentile | 16 min | Nonparametric, `APPROX_QUANTILES`-based — most accurate, cancels fewer meetings |

## Worked Example
Refines the flight-delay/meeting-cancellation decision using EDA: (1) explore average delays per airport in BigQuery, discovering January's worst-delay airports are ski resorts (ASE, JAC) — a plausible structural explanation; (2) check BigQuery's "Execution details" to confirm the query is I/O-bound and optimize with `HAVING` over a `WITH`+outer-`WHERE`; (3) create a Vertex AI Workbench notebook, use `%%bigquery` to pull arrival/departure delay into Pandas, and build violin plots revealing a long skewed tail and a genuine statistical-regime split at the 10-minute departure threshold; (4) discover 1000+ "unique" `DEP_DELAY` values — most representing only a handful of flights — and decide to filter by occurrence frequency (`HAVING numflights > 370`, from the three-sigma rule) rather than by value; (5) fit a per-departure-delay average/std of arrival delay, assume normality, and compute the Z=0.52 threshold for 30% probability — yielding a 13-minute departure-delay cutoff; (6) drop the normality assumption and use `APPROX_QUANTILES` to get the empirical 70th percentile directly — yielding a refined 16-minute cutoff; (7) build a proper train/test split by hashing distinct `FL_DATE` values into a persisted `dsongcp.trainday` table; (8) evaluate the 16-minute threshold on held-out test days, finding 95%/82% overall accuracy but only ~70-72% accuracy right at the decision boundary (DEP_DELAY = 15/16) — confirming the model hits its designed 70% correctness target while canceling fewer meetings than the Chapter 3 model.

## Key Takeaways
1. Use BigQuery for fast interactive SQL exploration of raw data; use Vertex AI Workbench (managed Jupyter) for the statistical visualization BigQuery itself can't do.
2. On large datasets, filter "outliers" by frequency of occurrence (`HAVING count > N`), not by raw value thresholds — rare-but-frequent-enough values carry real signal.
3. Prefer nonparametric/empirical quantile estimation (`APPROX_QUANTILES`) over Gaussian Z-score assumptions when you have enough data per group to support it.
4. Split train/test sets along a grouping dimension that matches real-world correlation structure (here, calendar date via hashing), never by naive random row shuffling on correlated data.
5. Evaluate a threshold-based decision model specifically at its decision boundary, not just over the whole population, to get an honest read on whether it hits its designed accuracy target.
6. Persist train/test split assignments as a reusable BigQuery table (and CSV export) so every subsequent model in the book is comparable on the same splits.
7. Treat built-in data quality flags as one more model input, not an automatic exclusion rule, unless they reflect genuine domain/environmental knowledge.

## Connects To
- **Ch 1 & 3**: Directly refines the flight-delay decision threshold (70%/30%, 15-minute criterion) first established in Ch. 1 and modeled simply in Ch. 3's contingency table.
- **Ch 4**: Operates on the `dsongcp.flights_tzcorr` UTC-corrected dataset produced by the Chapter 4 Dataflow pipeline.
- **Ch 10**: The train/test date-split methodology and the eventual need for a truly independent 2019 test set are revisited for MLOps rigor.
- **Ch 6–9**: The `dsongcp.trainday` table and empirical-threshold insight carry forward into every subsequent modeling chapter (Spark ML, BigQuery ML, TensorFlow/Vertex AI).
