# Cheatsheet

## Decision Rule: A/B Test vs. Observational/Incrementality Methods
- **Can you randomize, and is it affordable?** → Yes: run an A/B test (Ch 16). It's the only method that guarantees unconfoundedness by construction.
- **No (too costly, e.g., infrastructure, or unethical/infeasible)** → Use observational causal methods (Ch 15): draw a DAG, check if unconfoundedness is plausible given observed covariates, then use matching/propensity score matching or DML.
- **Two-sided marketplace with supply/demand interference?** → Use block/cluster randomization, not unit-level randomization (SUTVA violation risk).
- **Confounder is unobserved?** → Causal effect is not identifiable from this data; look for a natural experiment, instrument, or proxy (understanding the proxy only partially reduces bias).

## Decision Tree: Choosing a Metric Decomposition
1. Does the metric accumulate over time (a "stock")? → Stock-flow decomposition.
2. Is the metric a product of two factors (price×quantity, ARPU×MAU)? → P×Q / multiplicative decomposition.
3. Is the metric a weighted average across segments? → Mix-rate decomposition.
4. Does the metric result from a sequential process? → Funnel decomposition.
5. Want to explain a *change* in any of the above over time? → Apply the matching growth decomposition (additive for sums, multiplicative for products, mix-rate for weighted averages).

## Decision Tree: Which Interpretability Method?
- Need **local** explanation (why this one prediction)? → ICE plot, or individual coefficient contribution in linear model.
- Need **global** explanation, features **uncorrelated**? → PDP is fine (simple, standard).
- Need **global** explanation, features **correlated**? → Use ALE (PDP will be biased).
- Need feature **ranking**? → Standardized coefficients (linear), impurity-based importance (trees only), or permutation importance (any algorithm).
- Quick directional check only? → Heatmap (bivariate only — can mislead with correlated features).

## Thresholds & Defaults
- A/B test significance level (α): **5%** (sometimes 1% for high-stakes decisions).
- A/B test power (1−β): **~85%** (β=15%) is the conventional default.
- Fraction treated (P): **0.5** minimizes MDE — default to equal-sized groups.
- Leakage suspicion threshold: AUC **> 0.8** warrants scrutiny; AUC **≈ 1.0** is a near-certain red flag.
- 2×2 design randomization: aim for "ex ante equal on average" treatment/control groups.
- Business-case decision: proceed only if **net benefit ≥ 0** (breakeven or better), after stripping non-incremental costs.

## Tells and Smells: Data Leakage
- Test-set AUC unusually high (>0.8, especially near 1.0) with no obvious reason.
- A feature is a function of the outcome (e.g., unit price = Revenue/Sales when predicting Revenue).
- A "bad control" correlated with the outcome but not available at prediction time (even lagged, if autocorrelated).
- Ambiguous timestamp labeling (beginning- vs. end-of-period) with no documented convention.
- Joins across subqueries with mismatched implicit time windows.
- Feature scaling/standardization computed on the full dataset (train+test) rather than train-only.
- A dummy variable (e.g., region) ranks as the top feature with no plausible causal story — check for quasi-complete separation from a hidden business rule.
- Production performance is substantially worse than test-sample performance (after ruling out drift) — the definitive diagnostic.

## Tells and Smells: Lift-Metric / Selection-Bias Traps
- A/B-style group comparison without checking pretreatment variable balance — compute lift or difference-in-means on pretreatment features first.
- Self-selected campaign responders analyzed as if randomly assigned — check for adverse selection (Ch 4) or self-selection (Ch 6, 15).
- Survey or campaign response rates differ systematically across subgroups — check representativeness with lift tables before trusting aggregate results.
- A heatmap shows a feature as non-monotonic when the true relationship (per simulation/domain knowledge) is monotonic — suspect a correlated co-feature dominating the ranking.

## Tells and Smells: Underpowered or Misread A/B Tests
- "No significant effect" with a small sample size — check whether the MDE for your sample size exceeds any plausible true effect; if so, this may be a false negative, not evidence of no effect.
- Statistically significant result with an enormous sample (millions+) and a tiny effect — check business significance separately; don't conflate the two.
- Running a test without first defining the metric and hypothesis (how + why) — a leading cause of inconclusive, wasted experiments.

## Quick Reference: Core Formulas
```
Funnel:          M = (M/s_last) × ... × (s1/E) × E
Stock-flow:      Stock_t = Stock_(t-1) + Inflow - Outflow
P×Q:             Revenue = Price × Quantity
Business case:   Net Benefit = Benefits - Costs ≥ 0 (breakeven at 0)
Lift:            Lift(A,B) = Metric_agg(A) / Metric_agg(B)
FWL:             α_k = Cov(ỹ_k, x̃_k) / Var(x̃_k)   (net effect after partialling out)
Bias:            Bias(θ̂,θ) = E[θ̂ - θ]; omitted-var bias ∝ Cov(x1,x2)
Threshold rule:  Action A if p_i ≥ τ else B
Precision/Recall: TP/(TP+FP)  and  TP/(TP+FN)
Causal decomp:   E[Y|D=1]-E[Y|D=0] = ATT + Selection Bias
MDE:             (t_α + t_(1-β)) × sqrt(Var(Y)/(N·P·(1-P)))
```
