# Chapter 6: What's in a Lift?

## Core Idea
A lift is simply the ratio of an aggregate metric between two groups; it's a lightweight, versatile diagnostic for model performance, self-selection, and survivorship bias.

## Frameworks Introduced
- **Lift**: `Lift(metric, A, B) = Metric_aggregate(A) / Metric_aggregate(B)`.
  - When to use: measuring a selection/classification algorithm's performance vs. population baseline, or diagnosing self-selection/survivorship bias in any grouped comparison.
  - How: pick an aggregation (usually mean), define group A (subset) and group B (baseline/population), compute the ratio. Ratio > 1 = uplift, < 1 = downlift, = 1 = no lift.
- **Lift Table for Selection Bias Diagnosis**: rows = candidate features, columns = group A average, group B average, lift — quickly reveals what characteristics drive self-selection into a group.

## Key Concepts
- **Uplift / downlift**: lift ratio > 1 or < 1 respectively.
- **Self-selection**: individuals choose to enroll in a group based on some intrinsic characteristic (formal: political party; informal: product purchasers).
- **Survivorship bias**: a reverse self-selection where only units with a given characteristic "survive" into the sample (classic WWII fighter-plane armor example, Abraham Wald).
- **True positive rate (TPR) by decile**: sort scored users into deciles, compute the outcome rate per decile — the standard way to evaluate classifier lift.
- **Monotonicity (lift sense)**: higher score deciles should show higher TPR/lift if the score is informative.
- **Top-decile performance**: the lift in the highest-scored decile, often the headline number used to sell a model to stakeholders.

## Mental Models
- Lift as a "quick and dirty" selection-mechanism detector: compute it across several candidate features and read off which ones differ most between groups.
- Lift generalizes beyond ML: usable on any 2×2 quadrant (Ch 4), marketing-campaign self-selected responders vs. non-responders, or survey response-rate representativeness checks.

## Anti-patterns
- **Assuming a classifier/selection score is informative without checking monotonicity**: always verify TPR/lift rises across deciles before trusting the score.
- **Measuring campaign impact without checking self-selection**: if users who opted in to a campaign differ systematically from a true control group, naive before/after comparisons are biased — check lifts first.
- **Ignoring survey response-rate bias**: differential response rates across subgroups can bias survey conclusions; lifts flag this quickly.

## Code Examples
```text
Lift = Metric_aggregate(A) / Metric_aggregate(B)
e.g., Lift(women CEOs) = 0.32 / 0.50 = 0.64   (underrepresentation vs. random-selection baseline)
```
- **What it demonstrates**: lift as underrepresentation/overrepresentation measure relative to an expected baseline frequency.

## Reference Tables
*Table 6-1: Lifts in a churn example (active vs. churned)*

| Feature | Active | Churned | Lift |
|---|---|---|---|
| Monthly spend | 29.9 | 32.7 | 1.1 |
| Customer satisfaction score | 10.00 | 10.08 | 1.01 |
| Tenure (months) | 9.84 | 8.14 | 0.83 |

(Income lift mentioned in text as 1.18, an 18% increase for churners, though not shown in this reduced table excerpt.)

## Worked Example
Churn classifier: sort test-sample users by predicted score into deciles, compute churn rate per decile, divide each by overall sample churn rate to get lift per decile. Simulated example: top decile lift = 2.7× the average churn rate — strong evidence the model is useful for targeting a retention incentive, and a number directly comparable against the lift from the company's current (non-ML) selection mechanism. Selection-bias table: churners show higher income (lift 1.18), spend more (lift 1.1), shorter tenure (lift 0.83), and no meaningful difference in satisfaction (lift 1.01) — suggesting a hypothesis that higher-income customers have higher expectations unmet by a lower-SES-targeted product.

## Key Takeaways
1. A lift is just a ratio of aggregates between two groups — simple to compute, broadly applicable.
2. In ML evaluation, decile-based lift and TPR checks reveal both model informativeness (monotonicity) and expected real-world campaign performance (top-decile lift).
3. Lifts are a fast diagnostic for self-selection and survivorship bias — build a feature-by-feature lift table to spot what drives group membership.
4. Benchmark your model's lift against the lift of the existing (often non-ML) selection mechanism, not against an abstract ideal.
5. Apply lifts anywhere you compare a subgroup to a baseline: marketing self-selection, survey representativeness, 2×2 quadrant analysis.

## Connects To
- **Ch 4**: Lifts can be computed within any 2×2 quadrant to understand what characterizes that group.
- **Ch 9**: Bootstrapped confidence intervals (Ch 9) should accompany decile/TPR estimates to avoid over-interpreting noise at fine granularity.
- **Ch 15**: Self-selection bias is revisited in depth in the incrementality chapter as a core threat to causal estimation.
