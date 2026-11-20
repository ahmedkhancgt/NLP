# Chapter 3: Creating Compelling Dashboards

## Core Idea
Build a dashboard before (or in parallel with) the machine learning model, not after — dashboards force early exploratory data analysis, surface user insight, and provide the explanatory layer that makes a model's recommendations trustworthy and actionable to end users.

## Frameworks Introduced
- **Dashboard-first development**: build an explanatory dashboard as the first "model" artifact.
  - When to use: at project kickoff, before investing in a sophisticated predictive model.
  - How: build a trivial decision rule (e.g., single-threshold rule), visualize it, and ship it to real users to gather fine-grained, contextual feedback that shapes later feature engineering.
- **Contingency table (confusion matrix) threshold optimization**: tune a simple decision rule against a business error-tolerance target using SQL.
  - When to use: establishing or validating a decision threshold against known cost/tolerance for false positives/negatives.
  - How: use `COUNTIF` in BigQuery combined with `UNNEST([...])` to compute confusion-matrix cells for many candidate thresholds in a single table scan; derive accuracy, false-positive rate (FPR), and false-negative rate (FNR); pick the threshold that satisfies the required FPR/FNR, not necessarily the one maximizing overall accuracy.
- **BigQuery view layering for dashboards**: build cleaned "golden" views, then threshold-specific views, as the data sources for BI tools.
  - When to use: whenever a dashboard tool (Data Studio, Looker) needs stable, reusable, pre-cleaned data sources.
  - How: `CREATE OR REPLACE VIEW` to cast/rename raw columns once (avoiding repeated casts in every query); create one view per decision threshold (e.g., `delayed_10`, `delayed_15`, `delayed_20`) to make the model's cutoffs explicit in the dashboard rather than hidden in filters.

## Key Concepts
- **Dashboard vs. exploratory data analysis (EDA)**: dashboards target decision-makers with a tailored subset of data and interactivity; EDA targets data scientists with aggregates of historical data for model-building (Table 3-1).
- **Explainability**: showing users *why* a model recommends what it does, which both builds trust and surfaces hidden bias (cites the Amazon resume-screening tool that penalized terms common in women's resumes).
- **View vs. table vs. materialized view**: a view re-executes its SQL on every query (always current, free); a table is a static snapshot (fast but stale); a materialized view is BigQuery-maintained, kept up to date automatically, at extra storage/compute cost.
- **Apophenia and motivated reasoning**: cognitive biases where humans see patterns that aren't there or justify conclusions they want — a risk when interpreting/explaining model outputs, including to oneself.
- **False positive rate (FPR) / false negative rate (FNR)**: FPR = wrongly canceling when the flight would have been on time; FNR = wrongly not canceling when the flight is late — each must independently satisfy the business risk-tolerance threshold (not just overall accuracy).
- **Cloud SQL vs. BigQuery trade-off**: relational databases (Cloud SQL/PostgreSQL) suit small, frequently updated, indexable, transactional datasets; analytics warehouses (BigQuery) suit large, mostly read-only datasets requiring full-table scans and ad hoc queries without needing indexes.
- **Looker vs. Data Studio**: Data Studio is free, self-service, good for individual/small-team dashboards; Looker adds a shared semantic layer (LookML), multicloud data access, embedded analytics, and scheduled alerts — suited to enterprise-wide consistency.

## Mental Models
- **Dashboards are a user-facing survey tool for model building**: treat dashboard construction as a way to "rope in" eventual model users early, much like building a form to collect requirements.
- **Four fundamental graphic types**: relational, time series, geographical, narrative — pick the type that matches the data relationship you're depicting, never force-fit a narrative-style graphic onto worker-type data.
- **Chart-junk minimization / data-ink ratio**: maximize the ratio of data to decoration (per Tufte); keep gridlines and labels subtle so the data lines dominate.
- **Match the chart to the model, not just the data**: a scatter plot with a linear trendline implies a linear model — don't show it if your actual model is a simple threshold rule (the chapter explicitly rejects its own scatter-chart draft as misleading).

## Anti-patterns
- **Showing a chart that implies a different model than the one actually used**: e.g., a linear-regression-style scatter trendline when the real decision rule is a simple departure-delay threshold — this misleads users about how decisions are actually made.
- **Choosing the overall-accuracy-maximizing threshold instead of the one meeting the actual risk tolerance**: a 20-minute threshold had higher overall accuracy (93%) than 10 minutes but didn't meet the FNR ≤ 30% requirement that matters for the "would you make the meeting" decision.
- **Over-precise thresholds**: picking a threshold like "10.6 minutes" implies false precision; round to a value matching your data's actual resolution and the model's real certainty, especially when shown to end users.
- **Running one query per threshold candidate**: wastes repeated full-table scans; use `UNNEST` + `GROUP BY` to test many thresholds in a single pass.
- **Using CSV header-skipping assumptions without verifying the load tool supports it**: the Cloud SQL import UI has no "skip first row" option, so headers must be handled (e.g., delete the header row afterward) when all columns are typed as TEXT.

## Code Examples
```sql
CREATE OR REPLACE VIEW dsongcp.flights AS
SELECT
  FlightDate AS FL_DATE,
  Reporting_Airline AS UNIQUE_CARRIER,
  CAST(DepDelay AS FLOAT64) AS DEP_DELAY,
  CAST(ArrDelay AS FLOAT64) AS ARR_DELAY,
  IF(Cancelled = '1.00', True, False) AS CANCELLED,
  DISTANCE
FROM dsongcp.flights_raw;
```
- **What it demonstrates**: building a single "golden" cleaned view (casting types, renaming columns) to avoid repeating transformations in every downstream query.

```sql
DECLARE THRESH INT64;
SELECT
    THRESH,
    COUNTIF(dep_delay < THRESH AND arr_delay < 15) AS true_positives,
    COUNTIF(dep_delay < THRESH AND arr_delay >= 15) AS false_positives,
    COUNTIF(dep_delay >= THRESH AND arr_delay < 15) AS false_negatives,
    COUNTIF(dep_delay >= THRESH AND arr_delay >= 15) AS true_negatives,
    COUNT(*) AS total
FROM dsongcp.flights, UNNEST([5, 10, 11, 12, 13, 15, 20]) AS THRESH
WHERE arr_delay IS NOT NULL AND dep_delay IS NOT NULL
GROUP BY THRESH
```
- **What it demonstrates**: computing a full confusion matrix for seven candidate decision thresholds in a single table scan, via `UNNEST` cross join plus `COUNTIF`.

```sql
WITH contingency_table AS ( ... )
SELECT
   ROUND((true_positives + true_negatives)/total, 2) AS accuracy,
   ROUND(false_positives/(true_positives+false_positives), 2) AS fpr,
   ROUND(false_negatives/(false_negatives+true_negatives), 2) AS fnr,
   *
FROM contingency_table ORDER BY accuracy ASC
```
- **What it demonstrates**: deriving FPR/FNR from the contingency table to directly check against the 30% risk-tolerance decision criterion from Chapter 1.

## Reference Tables
| Threshold (min) | FPR | FNR | Accuracy | Verdict |
|---|---|---|---|---|
| 5 | ~0.04 | 0.44 | 0.85 | FNR too high |
| 10 | — | ~0.30 | ~0.90 | Meets 30% FNR target |
| 11 | — | <0.30 | — | Also acceptable |
| 20 | low | low | 0.93 | Higher accuracy but too conservative (misses target intent) |

| Tool | Best for |
|---|---|
| Cloud SQL (PostgreSQL/MySQL/SQL Server) | Small, frequently-updated, indexable transactional data |
| BigQuery | Large, mostly read-only, full-table-scan analytics |
| Data Studio | Free, self-service dashboards |
| Looker | Enterprise semantic layer, multicloud, embedded analytics, scheduled alerts |

## Worked Example
Builds the first "model" for the flight-delay/meeting-cancellation decision: (1) load flight CSVs into a Cloud SQL PostgreSQL instance to demonstrate why RDBMS doesn't scale to this analytics workload; (2) pivot to BigQuery, exploring `flights_raw`'s schema via Preview and Table Explorer to decide which duplicate columns (Year/Month/Day vs. FlightDate; three airline-code columns) to drop; (3) create the `dsongcp.flights` view with cleaned/cast columns; (4) build a naive rule ("cancel if departure delay ≥ 15 min"), compute its contingency table in BigQuery, then sweep thresholds 5–20 minutes to find that a 10–11 minute departure-delay threshold satisfies the ≤30% false-negative-rate requirement from Chapter 1 (achieving ~90% overall decision accuracy); (5) build a Data Studio dashboard with a date-range control, a donut chart of on-time/late proportions, a bar chart of average departure/arrival delay by carrier, and per-threshold views (`delayed_10`, `delayed_15`, `delayed_20`) to visually explain why 10 minutes is the right cutoff (flights delayed ≥10 min jump to ~50 min average delay — a different statistical regime than the ~5 min average for all flights).

## Key Takeaways
1. Build an explanatory dashboard early — even from a trivial rule — to get real user feedback before investing in sophisticated modeling.
2. Match every chart to the actual model; never show a visualization (e.g., linear trendline) that implies a different model than the one driving recommendations.
3. Validate a decision threshold against the specific error-tolerance metrics that matter (FPR, FNR) — not just maximizing overall accuracy.
4. Use `UNNEST` + `COUNTIF` + `GROUP BY` in BigQuery to test many thresholds in one table scan instead of one query per candidate.
5. Prefer BigQuery views over ad hoc per-query casts to keep analysis DRY and reusable across dashboard tools.
6. Choose Cloud SQL for small, transactional, indexable data; BigQuery for large-scale, read-heavy analytics — don't force one tool to do the other's job.
7. Pick Data Studio for free self-service dashboards; move to Looker when you need a shared semantic layer, multicloud sources, or embedded/scheduled analytics.

## Connects To
- **Ch 2**: Consumes the `dsongcp.flights_raw` BigQuery table ingested there.
- **Ch 1**: Directly operationalizes the 70%/30% probabilistic decision criterion established in the flight-delay case study.
- **Ch 4**: Hands off to building real-time, location-aware dashboards once streaming data (Pub/Sub, Dataflow) is introduced.
- **Ch 5**: Full exploratory data analysis (of which this chapter's "just enough" analysis is a preview) is covered there using Vertex AI Workbench.
