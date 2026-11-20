# Chapter 10: Linear Regression: Going Back to Basics

## Core Idea
Linear regression's mathematical transparency (covariance formulas, Frisch-Waugh-Lovell theorem) reveals general truths about what any ML algorithm — linear or nonlinear — can and cannot learn: correlation is not causation, variance is required to learn anything, and omitted/irrelevant variables behave predictably.

## Frameworks Introduced
- **Bivariate Coefficient Formula**: `α1 = Cov(y, x1) / Var(x1)`.
  - When to use: as the benchmark intuition for what *any* regression coefficient represents — a scaled correlation, never automatically causal.
- **Frisch-Waugh-Lovell (FWL) Theorem**: `αk` can be recovered by (1) regressing y on all other features, save residuals ỹ; regressing xk on all other features, save residuals x̃k; (2) regressing ỹ on x̃k — the slope equals αk. Equivalently: `αk = Cov(ỹ_k, x̃_k) / Var(x̃_k)`.
  - When to use: explaining to a stakeholder that a coefficient is a *net* effect, already cleaned of other included covariates' influence (e.g., price effect net of state-level pricing differences); controlling for confounders (including a deterministic time trend to avoid spurious time-series regression); building intuition for why PDPs on correlated features (Ch 9) can't cleanly separate direct vs. indirect effects the way OLS can.
  - Caveat: orthogonalization must be applied to the feature of interest, not just the outcome — `αk ≠ Cov(ỹ, xk)/Var(xk)` in general.
- **Variance-Identification Principle**: you need variation in inputs to explain variation in outputs, for *any* ML algorithm, not just OLS — zero variance in a feature means zero covariance means the parameter can't be learned (or doesn't exist, as in the dummy variable trap).

## Key Concepts
- **Confounder**: a third variable affecting both x and y that is otherwise unrelated to the causal question (Figure 10-1); creates non-causal correlation that FWL-style controlling can remove.
- **Direct causation / reverse causation**: covariance is symmetric, so `Cov(x,y)≠0` is consistent with x→y, y→x, or neither (confounding).
- **Orthogonalization**: OLS residuals are, by construction, uncorrelated with the regressors — the mechanical basis for FWL.
- **Dummy variable trap**: including an intercept plus dummies for *all* categories makes X'X singular (non-invertible) in OLS; always drop one reference category. Ensemble methods don't have this computational restriction but gain nothing from the redundant dummy, and will often rank both as "important" since they carry identical information.
- **Irrelevant variables**: in OLS, adding uninformative controls creates no bias, only extra variance; ensembles are more sensitive if the irrelevant feature is correlated with a real signal feature.
- **Trend-stationary variable**: a time series stationary only after removing a deterministic trend; omitting the trend when regressing two independent trending series produces spurious correlation.
- **Perfect multicollinearity via group aggregates**: including both group dummy variables (e.g., state) and the group-level average of another variable provides *exactly* the same information — redundant for OLS (rank-deficient) and for any ML algorithm (no predictive gain).

## Mental Models
- "Correlation is not causation" is mathematically explicit in OLS (via the covariance formula) and should be assumed to hold just as strongly, if less visibly, in any nonlinear ML algorithm.
- Think of every regression coefficient as a *net* effect — ask "net of what?" before interpreting it.
- If a feature transformation (e.g., normalizing by a 95th-percentile benchmark) is just a monotonic rescaling of existing information, it changes nothing about what the model can learn — be suspicious of "clever" feature engineering that doesn't add new variance/information.

## Anti-patterns
- **Causal interpretation of a regression coefficient without justification**: only valid under very specific conditions (see Ch 15); applies to OLS, random forests, and gradient boosting alike.
- **Interpreting GBR/RFR PDPs as cleanly separating direct and indirect effects of correlated features**: PDPs fix other features at their mean, so correlated "direct + indirect" effects get muddled — OLS's FWL-based net effect is cleaner here.
- **Regressing two independently trending time series without a trend control**: produces spurious, statistically significant correlation purely from the shared time trend confounder.
- **Including both group dummies and group-level feature averages**: wastes effort; the feature matrix is rank-deficient and provides zero additional predictive information for any algorithm.
- **Engineering a "ratio-to-benchmark" feature for anomaly detection and assuming it adds signal**: if it's a monotonic transform of the original feature, it carries identical information and will not improve fraud-detection performance.

## Code Examples
```python
def compute_alpha_1feat(ydf, xdf):
    """Compute alpha using var-cov formula and linear regression for y = a + b x"""
    cat_mat = ydf.copy()
    cat_mat['x'] = xdf['x1']
    vcv = cat_mat.cov()
    cov_xy = vcv['y'].loc['x']
    var_x = vcv['x'].loc['x']
    beta_vcv = cov_xy / var_x
    reg = LinearRegression(fit_intercept=True).fit(xdf, ydf.values.flatten())
    beta_reg = reg.coef_[0]
    return beta_vcv, beta_reg
# Alpha vcv formula = 3.531180168, Alpha OLS = 3.531180168
```
- **What it demonstrates**: the bivariate covariance formula and OLS produce numerically identical coefficients — proof that the coefficient is "just" a scaled correlation.

```python
# FWL two-step vs. direct OLS, for x1, x2, x3
# x1: FW two-steps = 3.66436, OLS = 3.66436
# x2: FW two-steps = -1.8564, OLS = -1.8564
# x3: FW two-steps = 2.95345, OLS = 2.95345
```
- **What it demonstrates**: the Frisch-Waugh-Lovell two-step partialling-out procedure reproduces full multi-feature OLS coefficients exactly.

```python
# Dummy-variable / group-aggregate rank check
print(f'Columns of X = {Xdf.columns.values}')
rank_x = LA.matrix_rank(Xdf)
# X: Rank = 4, total columns = 5   -> rank-deficient: perfect multicollinearity
X_nm = Xdf[[col for col in Xdf.columns if col != 'mean_z']]
# X_[-meanz]: Rank = 4, total columns = 4   -> same rank after dropping redundant column
```
- **What it demonstrates**: including both state dummies and a state-level feature average produces a rank-deficient feature matrix; dropping the redundant average loses no rank/information.

## Reference Tables
| Scenario | OLS behavior | Ensemble (GBR/RFR) behavior |
|---|---|---|
| Omitted correlated variable | Biased coefficient | Biased (same direction) |
| Irrelevant uncorrelated variable | Unbiased, more variance | Usually fine, correctly low importance |
| Irrelevant but correlated variable | Unbiased, more variance | Sensitive — can look falsely predictive |
| Dummy variable trap (all categories + intercept) | Estimator does not exist (singular X'X) | No computational issue, but no added information |
| Low feature variance | High coefficient variance / non-identification | High estimate variance too |

## Worked Example
Simulated DGP with a direct+indirect effect structure (x1 affects y both directly and through x2): OLS, via FWL, correctly recovers the true net effect for both x1 and x2 with proper confidence intervals, while GBR's PDP for x1 is distorted because fixing x2 at its mean leaves x1's PDP blending direct and indirect effects the algorithm can't disentangle. Confounder example: two independent trend-stationary AR(1) processes regressed on each other without a trend control show a statistically significant (spurious) relationship; adding a linear time trend as a control removes the spurious significance — illustrating why a deterministic trend should routinely be included as a safeguard in time-series regressions.

## Key Takeaways
1. Linear regression coefficients formalize "correlation, not causation" — treat this as the baseline assumption for any ML algorithm's output.
2. The Frisch-Waugh-Lovell theorem shows every OLS coefficient is a net effect after partialling out all other included features — useful for defending "net of X" claims to stakeholders.
3. Including a deterministic time trend is a cheap, effective safeguard against spurious time-series correlation from a shared confounding trend.
4. Irrelevant features don't bias OLS, only add variance; correlated-but-irrelevant features can mislead ensemble feature importance.
5. Always drop one reference category when using dummy variables in OLS (dummy variable trap); ensembles don't need this but gain nothing from redundant categories either.
6. You need genuine variance in features to learn anything — group-level dummies and group-level feature averages carry identical information and create (near-)perfect multicollinearity.
7. A monotonic feature transformation (e.g., ratio to a benchmark quantile) adds no new information regardless of how clever it looks.

## Connects To
- **Ch 9**: Builds directly on omitted-variable-bias and PDP material from Simulation and Bootstrapping.
- **Ch 11**: The "variance is required to learn" and multicollinearity/redundant-feature principles here foreshadow leakage-detection heuristics (overly-predictive, information-identical features).
- **Ch 15**: FWL's "net effect after controlling for confounders" is the conceptual bridge to the causal-inference tools used for incrementality.
