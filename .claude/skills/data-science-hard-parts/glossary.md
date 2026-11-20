# Glossary

**A/A test** — An experiment where both groups receive the same baseline alternative, used to estimate outcome variance or validate test infrastructure (Ch 16).

**A/B test (RCT)** — A randomized controlled trial comparing baseline (A) and alternative (B) on a chosen metric; randomization guarantees unconfoundedness (Ch 15, 16).

**Accumulated Local Effects (ALE)** — An interpretability method that fixes PDPs' correlated-feature bias by computing local slopes within feature-value neighborhoods and accumulating them (Ch 13).

**Additive decomposition** — Growth decomposition for `y = y1 + y2`; output growth is the weighted average of input growth rates (Ch 3).

**ATT (Average Treatment Effect on the Treated)** — The causal effect of treatment averaged over units that actually received it (Ch 15).

**Backdoor criterion** — The rule for valid causal identification: control for confounders, never for colliders (Ch 15).

**Bad control** — A control variable that is itself an outcome affected by the features/treatment, biasing estimates if included (Ch 9, 11, 15).

**Bootstrapping** — Resampling with replacement from your dataset to estimate an estimate's variance without distributional assumptions (Ch 9).

**Breakeven** — The point of zero net benefit in a business case; the worst-case boundary for a decision (Ch 5).

**Cannibalization** — A new product/feature/location reducing sales of an existing one (Ch 15).

**Collider** — A common effect of treatment and outcome; controlling for it introduces bias (Ch 15).

**Complete / quasi-complete separation** — When a linear combination of features perfectly (or near-perfectly for a subset) predicts the classification outcome, inflating AUC and often signaling leakage (Ch 11).

**Confounder** — A common cause of both treatment/feature and outcome; must be controlled for to avoid spurious correlation (Ch 10, 15).

**Confusion matrix** — Cross-tab of actual vs. predicted class (TN, FP, FN, TP), parameterized by a decision threshold (Ch 14).

**DAG (Directed Acyclic Graph)** — A graph of variables (nodes) and causal relationships (directed edges, no loops), used to formalize causal assumptions (Ch 15).

**Data drift** — The joint distribution of outcome/features changes over time, while the DGP stays the same (Ch 12).

**Data leakage** — Using data/metadata unavailable at prediction time, correlated with the outcome, creating unrealistically high test performance (Ch 11).

**Data-ink ratio** — Tufte's ratio of informative "data-ink" to total ink in a chart; a guideline (not law) for avoiding visual clutter (Ch 8).

**DGP (Data Generating Process)** — The full specification of relationships among inputs, noise, and outputs (Ch 9).

**Double Machine Learning (DML)** — Causal-effect estimation method combining ML flexibility, orthogonalization, and sample splitting (Ch 15).

**Dummy variable trap** — Including an intercept and dummies for all categories in OLS, making the cross-product matrix singular (Ch 10).

**Frisch-Waugh-Lovell (FWL) theorem** — Any OLS coefficient equals the result of partialling out all other regressors first, then regressing residuals — proof that coefficients are "net" effects (Ch 10).

**Funnel decomposition** — Breaking a metric into stage-by-stage conversion rates across a sequential process (Ch 2).

**Heatmap (feature)** — Table of average outcome by feature-value decile, revealing bivariate (only) correlation patterns (Ch 13).

**ICE (Individual Conditional Expectation)** — The per-unit, un-averaged version of a PDP; a local-interpretability tool (Ch 13).

**Incrementality** — Causal inference applied to business decisions: whether a lever truly improved an outcome (Ch 15).

**Lift** — The ratio of an aggregate metric between two groups; >1 = uplift, <1 = downlift (Ch 6).

**Matching / Propensity Score Matching** — Estimating causal effects with observational data by pairing treated units with similar untreated units, directly or via the propensity score (Ch 15).

**MDE (Minimum Detectable Effect)** — The smallest true effect size reliably detectable given significance, power, sample size, and variance (Ch 16).

**Metadata leakage** — Leakage from using summary statistics (e.g., scaling moments) computed over data unavailable at prediction time (Ch 11).

**Mix-rate decomposition** — Growth decomposition for weighted-average metrics, splitting change into mix (weight) and rate (input) effects plus a combined term (Ch 3).

**Model drift** — The underlying data generating process itself changes over time (Ch 12).

**Multiplicative decomposition** — Growth decomposition for `y = y1 × y2`; output growth is the sum of input growth rates plus a combined effect (Ch 3).

**Omitted variable bias** — Bias from excluding a feature correlated with an included feature; `Bias ∝ Cov(x1,x2)` (Ch 9, 10).

**Partial Dependence Plot (PDP)** — Predicted outcome as one feature varies, others fixed (at means or per-unit, then averaged); biased under correlated features (Ch 9, 13).

**Permutation importance** — Feature importance measured by performance drop when a feature's values are shuffled; works for any algorithm (Ch 13).

**Precision** — `TP/(TP+FP)`; reflects false-positive cost sensitivity (Ch 14).

**Production ready** — A model deployed, monitored, and maintained for actual consumption (Ch 12).

**Propensity score** — The probability a unit receives treatment given covariates, `p(X) = Pr(D=1|X)`; sufficient for matching under unconfoundedness (Ch 15).

**Recall** — `TP/(TP+FN)`; reflects false-negative cost sensitivity (Ch 14).

**Selection bias** — Deviation between observed treated-vs-control difference and the true causal effect, from non-random assignment into treatment (Ch 15).

**Self-selection** — Individuals choosing to enter a group based on an intrinsic characteristic (Ch 6, 15).

**Signal-to-noise ratio (SNR)** — Ratio of explained signal to unexplained noise; low SNR can hide true effects behind wide confidence intervals (Ch 9).

**Slopegraph** — A line-based plot showing trend differences across categorical segments (Ch 8).

**Stock-flow decomposition** — `Stock_t = Stock_(t-1) + Inflow − Outflow`, for accumulating metrics like MAU (Ch 2).

**SUTVA (Stable Unit Treatment Value Assumption)** — Requires (i) uniform treatment definition, (ii) no interference between units' potential outcomes; violated in marketplaces with supply/demand externalities (Ch 15).

**Survivorship bias** — A reverse self-selection where only units with a given characteristic "survive" into the sample (Ch 6).

**Threshold decision rule** — Converting a probability score into an action via `Action A if p≥τ else B` (Ch 14).

**Unconfoundedness** — The assumption that potential outcomes are independent of treatment assignment conditional on observed covariates; untestable, foundational to causal inference (Ch 15).

**Uplift modeling** — Using treatment/control data to target the "movable middle" segment for incremental campaign impact (Ch 15).

**Waterfall chart** — A chart decomposing a metric change into segment/category contributions (Ch 3, 8).

**Windowing methodology** — Defining observation and prediction windows relative to prediction time to structurally prevent data leakage (Ch 11).
