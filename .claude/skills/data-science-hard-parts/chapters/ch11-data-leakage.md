# Chapter 11: Data Leakage

## Core Idea
Data leakage occurs when information used in training won't be available (or means something different) at prediction time; it masquerades as unreasonably good test performance and collapses in production. A disciplined windowing methodology, careful feature/transformation audits, and business-process knowledge are the main defenses.

## Frameworks Introduced
- **Data Leakage Definition (3-part symptom check)**: (1) uses data/metadata unavailable at prediction time, (2) correlated with the outcome, (3) creates unrealistically high test performance.
  - When to use: as a checklist whenever test-sample performance looks "too good."
- **Windowing Methodology**: define, relative to prediction time `t_p`: the **Prediction window** `(t_p, t_p+P]` (horizon over which the outcome event is evaluated) and the **Observation window** `[t_p−O, t_p]` (history available to build features). The prediction window is open on the left by design, cleanly separating observed-past from predicted-future.
  - How: the training stage must mirror the scoring stage exactly — training observation window = `[t_p−P−O, t_p−P]`, training prediction window = `(t_p−P, t_p]`.
  - Ownership: Prediction window length (P) — primarily a business decision (timeliness needs); Observation window length (O) — primarily a data-scientist decision (predictive performance, relative weight of distant past).
  - When to use: as the default scaffolding for any supervised model build to make past-only feature use structurally enforced, not just hoped for.
- **Leakage Detection Heuristics**: suspect leakage when AUC > ~0.8 (classification; a personal heuristic, sensitive to class imbalance) or AUC = 1 (near-certain leakage); in regression, compare production vs. test performance directly since MSE/R² thresholds are scale-dependent and less diagnostic.

## Key Concepts
- **Outcome-as-feature leakage**: trivially including y (or `f(y)`) as a feature — the degenerate benchmark case.
- **Function-of-outcome feature**: e.g., including unit price (= Revenue/Sales) as a feature when predicting revenue — easy to miss when the feature is computed upstream by another team.
- **Bad control**: a control variable that is itself an outcome affected by the features (or correlated with y through an unavailable-at-prediction-time channel); differs from a "good" control that's exogenous.
- **Mislabeled timestamp**: ambiguity between beginning-of-period and end-of-period labeling creates leakage if the analyst assumes the metric was known earlier than it actually was.
- **Sloppy time-aggregation joins**: joining subqueries with different implicit time windows can silently pull future information into a feature (e.g., "number of other products" measured after the churn-prediction point).
- **Metadata leakage**: standardizing/scaling using moments (mean, SD) computed over the full dataset (train+test), leaking future-informed statistics into training — not leakage of the raw data values, but of its distributional metadata.
- **Complete separation**: a linear combination of features perfectly predicts y (classification); the likelihood has no maximum. Common with small/imbalanced data or a thresholded continuous variable reused as a feature (definite leakage in the latter case).
- **Quasi-complete separation**: a linear combination of features perfectly predicts a *subset* of observations (e.g., a business-rule-driven dummy variable); may or may not be leakage — requires judgment call.

## Mental Models
- "In ML, the scoring stage reigns" — training is scaffolding; the scoring/production stage is where value is created and is the ultimate test of model validity.
- Training must be the mirror image of scoring: whatever constraints (windows, data availability) apply at scoring time must be imposed identically during training.
- Metadata leakage vs. feature leakage: leakage isn't only about raw data values — summary statistics (means, SDs) computed over future/test data are an often-overlooked leakage vector.

## Anti-patterns
- **Standardizing features/outcome using full-dataset (train+test) moments**: artificially improves test performance, especially when there's a trend component informing the model "the series increases."
- **Including a bad control correlated with the outcome but unavailable at prediction time**: even a lagged version can leak if there's enough autocorrelation.
- **Trusting AUC near 1.0 without scrutiny**: near-perfect AUC is a red flag, not a win — audit for leakage before celebrating.
- **Assuming quasi-complete separation from a dummy variable is harmless because "it will be available at prediction time"**: the dummy may reflect a time-bound business rule (e.g., a rotating regional campaign) that changes the DGP — effectively metadata leakage even if technically "available."
- **One-off ad hoc feature engineering without a windowing discipline**: without explicit O/P window enforcement, it's easy to accidentally pull future information into training features via careless joins.

## Code Examples
```sql
WITH sales AS (
  -- subquery with info for each customer, sales and delta sales, using time window 1
),
prods AS (
  -- subquery with number of products per customer using time window 2
)
SELECT sales.*, prods.*
FROM sales
LEFT JOIN prods ON sales.customer_id = prods.customer_id
AND sales.month = prods.month
```
- **What it demonstrates**: a classic leakage source — two subqueries with inconsistent time windows joined together, silently injecting future information (e.g., "customer still has product next month" when predicting this month's churn).

```python
def query_data(len_obs: int, len_pre: int):
    """Query data enforcing the chosen observation (O) and prediction (P) windows."""
    today = datetime.datetime.today()
    base_time = today - relativedelta(months=len_pre)   # t_p - P
    init_time = base_time - relativedelta(months=len_obs)
    end_time = base_time + relativedelta(months=len_pre)
    # Observation window: [init_time, base_time)
    # Prediction window:  [base_time, end_time]
    my_query = f"""
      SELECT
          SUM(CASE WHEN date >= '{init_time}' AND date < '{base_time}'
          THEN x_metric ELSE 0 END) AS my_feature,
          SUM(CASE WHEN date >= '{base_time}' AND date <= '{end_time}'
          THEN y_metric ELSE 0 END) AS my_outcome
      FROM my_table
    """
    df = connect_to_database(my_query, conn_parameters)
    return df
```
- **What it demonstrates**: enforcing the windowing methodology directly in the query layer so that feature and outcome windows are structurally non-overlapping and reproducible.

## Reference Tables
*Table 11-1: Considerations when choosing window lengths*

| | Prediction window (P) | Observation window (O) |
|---|---|---|
| Owner | Business | Data scientist |
| Key consideration | Feasibility of short- vs. long-term prediction (timeliness of decision) | Relative weight/predictive value of distant past |
| Data constraint | Historical data at your disposal | Historical data at your disposal |
| Optimized for | Business decision timing | Predictive performance |

## Worked Example
Bad-control simulation: training with a leaked bad control gives MSE roughly a quarter of the no-leakage model's MSE (artificially "too good"); when the bad control is independent of the outcome, no leakage occurs and performance is identical, and even a lagged, autocorrelated bad control still leaks. Quasi-complete-separation real case: a cross-selling classifier's top feature by importance was a state dummy variable with no plausible causal link to the product — investigation revealed the sales team had only run that quarter's campaigns in that state (rotating regions to dodge competitor detection); the author classifies this as metadata leakage (not model drift) since the underlying accept/reject DGP is unchanged — fix: drop the state dummy. Simulated separation test: a latent-variable DGP with a third feature constructed by randomly selecting among y=1 observations (degree of separation controlled by p) shows AUC lift of 10–15% over the no-separation baseline for both logistic regression and gradient boosting — demonstrating separation's AUC-inflating effect independent of algorithm.

## Key Takeaways
1. Unreasonably high test-sample performance (AUC > ~0.8, approaching 1.0) is the primary symptom of data leakage — investigate before celebrating.
2. Leakage can come from the outcome itself, a function of the outcome, bad controls, mislabeled timestamps, sloppy time-window joins, or leaked metadata (scaling moments).
3. A strict windowing methodology (observation window + left-open prediction window, training mirroring scoring) structurally prevents the most common leakage pattern: using the future to predict the past.
4. Complete and quasi-complete separation in classification inflate AUC and require judgment to distinguish genuine leakage from legitimate deterministic business rules.
5. The ultimate test of leakage is production performance vs. test-sample performance — if the gap is large and drift is ruled out, suspect leakage.
6. Strong knowledge of the data's business-process origins is often the only way to catch subtle leakage (e.g., rotating regional campaigns).
7. "Scoring reigns" — design every upstream training decision to faithfully mirror what will actually be available at scoring/production time.

## Connects To
- **Ch 2**: The unit-price (Revenue/Sales) feature-leakage example reuses the P×Q decomposition.
- **Ch 9/Ch 10**: Bad controls and omitted/included variable logic connect directly to the omitted-variable-bias and FWL discussions.
- **Ch 12**: The windowing methodology and "scoring reigns" mindset are the direct setup for Productionizing Models.
