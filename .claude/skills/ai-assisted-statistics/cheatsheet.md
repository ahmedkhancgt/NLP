# Cheatsheet: AI-Assisted Statistics for Data Scientists

## Which estimator for location/variability?
- Data roughly symmetric, no outliers → mean, standard deviation.
- Outliers/skew present → median, trimmed mean (p=0.1 common), MAD, IQR.
- Need a weighted average (unequal reliability/representation) → weighted mean/median.
- No formula exists for your statistic (median, correlation, model coef) → bootstrap.

## Which significance/resampling test?
| Situation | Use |
|---|---|
| 2 groups, numeric outcome | Permutation test or t-test |
| 2 groups, binary/count outcome | Permutation test or chi-square/`prop.test` |
| 3+ groups, numeric outcome | ANOVA (F-test) or permutation ANOVA |
| r×c categorical contingency table | Chi-square test; Fisher's exact if any cell count ≤5 |
| Want ongoing optimization, not one-time proof | Multi-arm bandit (epsilon-greedy / UCB / Thompson) |
| Many repeated comparisons | Adjust alpha (Bonferroni) or use holdout/permutation to guard against vast search effect |

## p-value decision rules / defaults
- Common alpha thresholds: 0.05 (standard), 0.01 (stricter).
- p-value ≠ "probability hypothesis is true." It's P(result this extreme | null model).
- Statistical significance ≠ practical significance — always check effect size too.
- Multiple tests inflate false-positive risk: n tests at α=0.05 → P(≥1 false positive) ≈ 1 − 0.95^n.

## Sample size / power rule of thumb
- Need 3 of {sample size, effect size, alpha, power} to solve for the 4th.
- Smaller detectable effect → quadratically more sample needed (√n rule: 4x sample → half the standard error).
- Generic rules of thumb ("wait for 1,000 visitors, run for a week") are not statistically meaningful — compute power/sample size from actual baseline rate and desired effect size.

## Which regression/classification model?
| Need | Model |
|---|---|
| Numeric outcome, linear relationship, interpretability | Linear regression |
| Numeric outcome, nonlinear relationship | Polynomial/spline regression, GAM, or tree ensemble |
| Binary outcome, interpretable odds ratios | Logistic regression |
| Categorical predictors only, want simple/fast | Naive Bayes |
| Many predictors, unknown interactions, want accuracy over interpretability | Random forest or XGBoost |
| Best possible accuracy, willing to tune carefully | XGBoost (watch overfitting: tune `eta`, `subsample`, `max_depth`, `lambda`/`alpha`) |
| Sequential/time data | RNN/LSTM (classic) or transformer (modern, large-scale) |
| Image data | CNN (train from scratch only with lots of data; else transfer learning) |
| Need interpretable per-prediction explanation on a black-box model | SHAP values |

## Tells and smells (common statistical pitfalls)
- **Mean ≫ median**: skewed data or outliers present — check before trusting the mean.
- **Negative coefficient that "shouldn't" be negative** (e.g., bedrooms reducing home value): suspect correlated predictors (commission) or an omitted confounder like location (omission).
- **R² improves every time you add a variable**: expected — never use in-sample R²/RMSE alone to select a model; use cross-validation/holdout.
- **Model "99% accurate" on rare-event classification**: check the base rate — if positives are <1%, accuracy is meaningless; check precision/recall/AUC instead.
- **Huge gap between training and validation/test error**: classic overfitting signature — add regularization (L1/L2, dropout, shallower trees) or get more data.
- **"Torture the data long enough, it will confess"**: many variables/models tried on the same data → use holdout, permutation (target-shuffling), or pre-registered hypotheses.
- **p-value just under 0.05 after trying several analyses**: red flag for p-hacking / vast search effect.
- **Standard deviation on highly skewed/zero-inflated data looks absurd** (implies impossible negative values): use mean absolute deviation or domain-appropriate metric instead.
- **AI-generated statistical interpretation sounds fluent and confident**: verify against actual numbers — AI can misread chart density regions, fabricate unsupported qualifiers, or botch simple algebra.
- **Model trained increasingly on web-scraped/AI-generated content**: watch for model collapse — narrowing diversity, blander outputs over successive generations.

## Key thresholds & defaults
| Quantity | Typical default/threshold |
|---|---|
| Significance level (alpha) | 0.05 (sometimes 0.01) |
| Boxplot outlier whisker | 1.5 × IQR beyond box |
| High-leverage point (regression) | hat-value > 2(P+1)/n |
| High-influence point (regression) | Cook's distance > 4/(n−P−1) |
| KNN's K | 1–20, odd number to avoid ties |
| Random forest min leaf size | ~5 (classification)/10 (regression) in R; 1 default in sklearn |
| XGBoost default learning rate | 0.3 (R) / 0.1 (Python); use smaller (~0.1) for noisy data |
| Trimmed mean trim | 10% per tail (common default) |
| MAD scaling factor (to match SD under normality) | ×1.4826 |
| Degrees of freedom in variance | n − 1 |

## Decision tree: diagnosing a "weird" regression/ML result
1. Coefficient sign is wrong/unintuitive → check correlated predictors and omitted confounders.
2. Model fits training data perfectly but fails on new data → overfitting; add regularization/pruning, use cross-validation.
3. Residuals show a funnel/fan shape → heteroskedasticity; model may be missing a predictor or needs a transform.
4. Partial residual plot shows curvature → relationship is nonlinear; try polynomial/spline/GAM.
5. A few records dominate the fit → check for influential points/leverage; verify they aren't data errors.
6. Classifier always predicts the majority class → rare-class problem; rebalance (undersample/oversample/weight/SMOTE) and use precision/recall/AUC, not accuracy.
7. AI-assisted analysis looks too smooth/certain → ask it adversarial questions ("what would make this wrong?"), get a second AI's critique, verify against raw numbers.
