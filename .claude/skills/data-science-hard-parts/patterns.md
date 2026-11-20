# Patterns

## Funnel Decomposition
**When to use:** A metric results from a multi-stage sequential process (sales funnel, growth funnel, marketplace funnel).
**How:** Move right to left through the funnel, multiplying by the current-stage submetric and dividing by the previous stage's, finishing by multiplying by the entry-point metric, so terms cancel back to the original metric. Each resulting fraction is a stage conversion rate.
**Trade-offs:** Reveals levers owned by different teams, but choosing *which* intermediate stages to decompose by requires business knowledge; too many stages can fragment ownership and dilute focus.

## Stock-Flow Decomposition
**When to use:** The metric is an accumulating ("stock") quantity, like MAU or account balances.
**How:** `Stock_t = Stock_(t-1) + Inflow_t − Outflow_t`; split inflow/outflow further (e.g., New vs. Resurrected users) for more levers.
**Trade-offs:** Simple and intuitive, but only applies to genuinely accumulating metrics — misapplying it to flow-only metrics produces meaningless decompositions.

## P×Q / Multiplicative Decomposition
**When to use:** The metric is naturally a product of two factors (Revenue = Price × Quantity; Revenue = ARPU × MAU).
**How:** Multiply and divide by a meaningful intermediate metric; growth decomposes as sum of input growth rates plus a small combined/cross effect.
**Trade-offs:** Price and quantity often move in opposite directions (nonlinear interaction), useful for optimization but requires care interpreting the combined effect; drop it if negligible, but check first.

## Mix-Rate Decomposition
**When to use:** The metric is a weighted average across segments (e.g., ARPU across customer segments weighted by share).
**How:** `Δy = Δy^x (rate effect, weights fixed) + Δy^w (mix effect, inputs fixed) + Δw·Δx (combined)`.
**Trade-offs:** Excellent for storytelling ("what if only mix/only rate had changed") but requires careful bookkeeping of lagged vs. current weights in code.

## 2×2 Design
**When to use:** Simplifying a decision/analysis to exactly two key factors — testing a model+lever jointly, segmenting by two drivers (LTV, product-market fit), handling correlated features (adverse selection), or prioritizing projects (value vs. effort).
**How:** Pick two factors, threshold each into high/low, analyze/compare the four quadrants; randomize in experimental settings for "ex ante equal on average" groups.
**Trade-offs:** Great for communication and focus, but cannot rank items that dominate on different axes (use a monetary-scale business case instead), and non-experimental 2×2s can't claim "everything else held constant."

## Business Case Breakeven Analysis
**When to use:** Evaluating any new campaign, model, feature, or purchase before committing resources.
**How:** Enumerate costs/benefits, restrict to incremental ones only, compute at the unit-economics (average customer) level, solve the breakeven inequality (net benefit ≥ 0), net out false-positive and false-negative costs explicitly.
**Trade-offs:** Fast and broadly applicable, but requires discipline to avoid including non-incremental (sunk) costs; aggregate sign usually survives scaling but should be sanity-checked for highly heterogeneous customer bases.

## Lift Analysis
**When to use:** Measuring a classifier's performance vs. baseline, or diagnosing self-selection/survivorship bias in any grouped comparison.
**How:** Compute `aggregate(group A) / aggregate(group B)`; build a feature-by-feature lift table to spot what drives group differences.
**Trade-offs:** Extremely cheap and fast, but purely descriptive — a lift table reveals correlation patterns, not causal drivers; pair with bootstrapped CIs for fine-grained buckets to avoid chasing noise.

## Narrative Structure (What / So What / Now What)
**When to use:** Structuring any delivery-stage presentation, memo, or TL;DR.
**How:** What = business context/sizing; So What = actionable findings and Aha! moments (prefer expected-direction + surprising magnitude); Now What = concrete next steps and owners.
**Trade-offs:** Less dramatic than a classic narrative arc (setup-struggle-resolution), but more reliably drives action in business settings; requires genuine actionable insight to exist before starting.

## Monte Carlo Simulation
**When to use:** Testing algorithm sensitivity to SNR, correlation structure, outliers, or other DGP properties before trusting it on real data.
**How:** Fix parameters/seed/N, define a metric (e.g., bias), fix M simulations, simulate+estimate+save each run, aggregate mean and CI from percentiles.
**Trade-offs:** Requires you to specify (and get right) the DGP; results only generalize to the degree your simulated DGP resembles reality.

## Bootstrapping
**When to use:** Quantifying estimate precision without distributional assumptions, or when analytic variance is hard/costly to derive.
**How:** Resample N rows with replacement B times, recompute the estimate each time, use the spread across B estimates for variance/CI.
**Trade-offs:** Computationally heavier than parametric formulas but nonparametric and broadly applicable; essential for fine-grained metric breakdowns (e.g., ventile TPRs) to avoid false alarms.

## Windowing Methodology (Leakage Prevention)
**When to use:** Any supervised model build, as default scaffolding.
**How:** Define prediction window `(t_p, t_p+P]` and observation window `[t_p−O, t_p]`; ensure training mirrors scoring exactly (shifted by P).
**Trade-offs:** Structurally prevents the most common leakage pattern but doesn't catch all leakage sources (bad controls, metadata leakage still need separate audits).

## Interpretability Toolkit (Feature Importance, PDP, ALE, Heatmaps)
**When to use:** Ex post storytelling for an ML model that already has adequate performance and no leakage.
**How:** Standardize features before comparing coefficient magnitudes; use impurity-based importance for trees, permutation importance for any algorithm; prefer ALE over PDP when features are correlated; use heatmaps only for a first directional check (bivariate only).
**Trade-offs:** No single method is bias-free under correlated features except ALE (and even ALE has some residual bias); always confirm model quality before investing interpretability effort.

## Threshold Decision Rule + Confusion Matrix Optimization
**When to use:** Converting a classifier's probability score into a business action (approve/review, call/don't call, block/accept).
**How:** `Action A if p≥τ`; optimize τ by minimizing expected cost or maximizing expected profit using confusion-matrix cell probabilities and relative costs/benefits.
**Trade-offs:** Needs only relative (not absolute) costs, but requires genuine business input on cost ratios; a capacity-constrained volume rule is a simpler alternative when cost-benefit data isn't available.

## Causal Identification via DAGs + Unconfoundedness
**When to use:** Any time you want to claim a lever caused an outcome change, not merely correlates with it.
**How:** Draw the DAG, apply the back-door criterion (control for confounders, never colliders), check whether unconfoundedness plausibly holds given observed covariates.
**Trade-offs:** The identifying assumption is fundamentally untestable from data alone — relies entirely on getting the causal model (DAG) right; randomization (A/B testing) is the only way to guarantee it by construction.

## Matching / Propensity Score Matching
**When to use:** Causal estimation with observational data when randomization is infeasible (too costly or impossible) but unconfoundedness plausibly holds.
**How:** For each treated unit, find the m nearest untreated units by feature distance or propensity score; average their outcomes as the counterfactual; average individual effects for the ATT.
**Trade-offs:** Intuitive but computationally expensive (O(n_t × n_c × k)); only as valid as the unconfoundedness assumption and completeness of observed covariates.

## A/B Test Design via MDE
**When to use:** Planning any randomized experiment, before running it.
**How:** Set significance level α and power (1−β), estimate outcome variance (from history, Bernoulli formula, or an A/A test), compute MDE for a given sample size (or invert to get minimum sample size for a target MDE).
**Trade-offs:** Protects against underpowered false negatives, but requires upfront agreement on what MDE is business-meaningful — statistically detectable ≠ commercially significant.
