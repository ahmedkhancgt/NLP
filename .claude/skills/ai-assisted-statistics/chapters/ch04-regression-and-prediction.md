# Chapter 4: Regression and Prediction

## Core Idea
Linear regression quantifies the relationship between predictors and a numeric outcome via least squares, but the coefficients can mislead unless you account for correlated predictors, confounding variables, multicollinearity, nonlinearity, and influential outliers — and in data science, predictive accuracy on holdout data matters more than in-sample explanatory fit.

## Frameworks Introduced
- **Simple/Multiple Linear Regression (OLS)**: Y = b0 + b1X1 + ... + bpXp + e, fit by minimizing the residual sum of squares (RSS).
  - When to use: predicting a numeric outcome from one or more numeric/categorical predictors with an approximately linear relationship.
  - How: `lm(Y ~ X1 + X2, data=df)` in R / `LinearRegression().fit(X, y)` in Python; interpret bj as the change in Y per unit change in Xj holding others constant.
- **Model Selection (stepwise regression / AIC / adjusted R²)**: balancing fit against model complexity (Occam's razor) since adding variables always reduces RMSE/increases R² on training data.
  - When to use: many candidate predictors, need a parsimonious model.
  - How: `stepAIC()` (R, forward/backward/both) or cross-validated `SequentialFeatureSelector` (Python `mlxtend`); minimize AIC or maximize adjusted R².
- **Weighted Regression**: give different records different weights (inverse-variance weighting or to represent duplicated cases).
  - When to use: observations vary in reliability (e.g., older sales are noisier) or represent different numbers of underlying cases.
  - How: `lm(..., weight=Weight)` / `model.fit(X, y, sample_weight=w)`.
- **Factor variable encoding (dummy/reference coding vs. one-hot)**: convert categorical predictors to P−1 binary dummies for regression (one-hot's P columns cause multicollinearity in linear models, though it's fine for trees/kNN).
  - When to use: any categorical predictor in a linear model.
  - How: `model.matrix()` / `pd.get_dummies(..., drop_first=True)`.
- **Regression Diagnostics (standardized residuals, leverage/hat-values, Cook's distance)**: identify outliers and influential observations.
  - When to use: smaller datasets, explanatory modeling, or anomaly detection.
  - How: standardized residual = residual / SE(residual); high leverage if hat-value > 2(P+1)/n; high influence if Cook's distance > 4/(n−P−1).
- **Nonlinear extensions (polynomial, spline, GAM)**: capture curvature in the predictor-outcome relationship.
  - When to use: partial residual plots show systematic curvature that a linear term misses.
  - How: polynomial — `poly(X, 2)`; spline — `bs(X, knots=..., degree=3)`; GAM — `gam(Y ~ s(X) + ...)` auto-selects knots.
- **Prediction interval via bootstrap**: resample rows, refit, add a randomly drawn residual to the prediction, repeat to get an interval reflecting both model uncertainty and individual-point noise.
  - When to use: need an uncertainty range around a single predicted value (not a mean).

## Key Concepts
- **Fitted values / residuals**: Ŷi (predictions) and ei = Yi − Ŷi (prediction errors).
- **RMSE vs. R² vs. Residual Standard Error (RSE)**: RMSE is the main out-of-sample accuracy metric; R² (coefficient of determination) is the in-sample proportion of variance explained (0 to 1), most useful for explanatory modeling; RSE normalizes by degrees of freedom instead of n.
- **Confounding variable**: an important omitted predictor that produces spurious/counterintuitive coefficient signs (e.g., location omitted from a house-price model).
- **Multicollinearity**: near-perfect correlation among predictors causing unstable/non-unique coefficient estimates; must be fixed by removing redundant variables (unlike correlated-but-not-collinear predictors, which just make coefficients hard to interpret).
- **Interaction / main effect**: a main effect assumes a predictor's relationship with Y is independent of other predictors; an interaction term (`X1*X2`) lets that relationship vary with another variable's level (e.g., $/sqft depends on zip code group).
- **Heteroskedasticity**: residual variance that is not constant across the range of predicted values — a sign the model may be missing a predictor, though not a major concern for pure prediction accuracy.
- **Extrapolation**: applying a regression model to predictor values outside the range of the training data produces unreliable, sometimes absurd predictions.
- **Prediction interval vs. confidence interval**: a prediction interval (wider) covers uncertainty in one individual outcome; a confidence interval covers uncertainty in a mean/statistic — data scientists usually need prediction intervals.

## Mental Models
- Think of each regression coefficient as "the effect of this variable holding all others fixed" — which only makes causal sense when predictors aren't badly correlated and no important variable has been omitted (confounding).
- Use partial residual plots as a diagnostic lens: they isolate one predictor's true relationship with the outcome net of the rest of the model, revealing nonlinearity that the raw scatterplot or the fitted line alone would hide.
- Treat outliers/influential points as a double-edged signal: in big data for pure prediction they mostly wash out, but in small data or anomaly-detection contexts they are often the entire point (fraud, data errors, mis-recorded units).
- View stepwise/AIC-based model selection as Occam's razor operationalized: always prefer the simpler model unless the more complex one earns its extra variables through meaningfully better (ideally cross-validated) performance.

## Anti-patterns
- **Extrapolating beyond the training data's range**: e.g., predicting a vacant lot's price with a model fit only on built properties yields a negative, nonsensical value.
- **Interpreting a coefficient's sign without checking for confounding/correlated predictors**: e.g., `Bedrooms` getting a *negative* coefficient in the King County model only because `SqFtTotLiving` is correlated with it and location is omitted.
- **Using one-hot encoding (P dummies) in linear regression**: causes multicollinearity; use P−1 reference-coded dummies instead (one-hot is fine for trees/kNN).
- **Judging model quality by in-sample R²/RMSE alone**: these always improve by adding variables — use cross-validation / holdout for genuine model comparison.
- **Treating formal residual-normality assumptions as critical for data science**: they matter for valid p-values/hypothesis tests but rarely for pure predictive accuracy; heteroskedasticity is worth checking (missing signal) but non-normal residuals usually aren't.
- **Trusting AI-suggested model improvements uncritically**: the chapter shows ChatGPT correctly flags confounding/multicollinearity but such suggestions still need human statistical judgment before acting.

## Code Examples
```r
house_lm <- lm(AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms +
    Bedrooms + BldgGrade, data = house, na.action = na.omit)
summary(house_lm)
# Residual standard error: 261300 on 22681 degrees of freedom
# Multiple R-squared: 0.5406, Adjusted R-squared: 0.5405
```
- **What it demonstrates**: fitting and assessing a multiple linear regression on King County housing data; note `Bedrooms` gets a counterintuitive negative coefficient due to correlation with `SqFtTotLiving`.

```r
zip_groups <- house %>%
  mutate(resid = residuals(house_lm)) %>%
  group_by(ZipCode) %>%
  summarize(med_resid = median(resid), cnt = n()) %>%
  arrange(med_resid) %>%
  mutate(cum_cnt = cumsum(cnt), ZipGroup = ntile(cum_cnt, 5))
```
- **What it demonstrates**: consolidating 80 zip codes into 5 groups using regression residuals, then using `ZipGroup` to control for the confounding effect of location.

```r
lm(formula = AdjSalePrice ~ SqFtTotLiving * ZipGroup + SqFtLot +
    Bathrooms + Bedrooms + BldgGrade + PropertyType, data = house)
```
- **What it demonstrates**: an interaction term letting $/sqft vary by location — the highest ZipGroup shows ~3x the per-sqft price boost of the lowest.

```r
df_influential <- tibble(
  std_resid = rstandard(lm_98105),
  cooks_d = cooks.distance(lm_98105),
  hat_values = hatvalues(lm_98105)
)
```
- **What it demonstrates**: computing regression diagnostics (standardized residuals, Cook's distance, leverage) to flag influential observations (Python: `OLSInfluence` from `statsmodels`).

## Reference Tables
| Model extension | Captures | R function | Python equivalent |
|---|---|---|---|
| Linear (OLS) | Linear relationship | `lm()` | `sklearn.LinearRegression` / `sm.OLS` |
| Weighted regression | Unequal reliability/weight per row | `lm(weight=)` | `fit(..., sample_weight=)` |
| Polynomial | Smooth curvature (global) | `poly(X, 2)` | `I(X**2)` in `smf.ols` formula |
| Spline | Piecewise curvature (local, via knots) | `bs(X, knots=, degree=3)` | `bs()` in `statsmodels` formula |
| GAM | Auto-fit spline knots | `gam(Y ~ s(X))` (`mgcv`) | `pygam.LinearGAM(s(...))` |

| Diagnostic | Threshold / rule of thumb |
|---|---|
| Standardized residual (outlier) | large absolute value (no fixed universal cutoff) |
| Hat-value (leverage) | > 2(P+1)/n |
| Cook's distance (influence) | > 4/(n−P−1) |

## Worked Example
King County, WA housing sales (`house_sales.csv`) predict `AdjSalePrice` from `SqFtTotLiving`, `SqFtLot`, `Bathrooms`, `Bedrooms`, `BldgGrade`: R² ≈ 0.54, RMSE/RSE ≈ $261,300. `Bedrooms` initially gets a *negative* coefficient (counterintuitive) because of correlation with house size; dropping size-related variables flips it positive, confirming the correlation artifact. The model also omits location — a confounding variable — so adding `ZipGroup` (80 zip codes consolidated into 5 price-based groups via residual medians) corrects several coefficient signs (e.g., `Bathrooms` turns positive) and shows the most expensive zip group adds ~$340,000. Adding an `SqFtTotLiving × ZipGroup` interaction shows $/sqft ranges from $115 (cheapest zip group) to $342 (most expensive) — location moderates the size effect. Regression diagnostics on the 98105 zip-code subset find one house that sold for only $119,748 despite being 2,900 sqft — a standardized residual of −4.33 — traced via the deed to a partial-interest sale, not a true market transaction. A partial residual plot reveals `SqFtTotLiving`'s relationship with price is actually nonlinear, motivating polynomial/spline/GAM fits that better match the data's curvature in Figure 4-9–4-13. The chapter closes by having ChatGPT interpret and critique the regression output, correctly flagging confounding, multicollinearity, and heteroskedasticity, and suggesting log-transforming the outcome and adding location controls.

## Key Takeaways
1. Always check whether a counterintuitive coefficient sign is due to correlated predictors (commission) or a confounding/omitted variable (omission) before trusting an interpretation.
2. Prefer P−1 dummy (reference) coding for categorical predictors in linear regression; reserve one-hot (P dummies) for tree/kNN-type models.
3. Use RMSE/cross-validated performance, not in-sample R² alone, to judge or compare models — R² always improves with more variables.
4. Check partial residual plots for nonlinearity before assuming a linear term is adequate; escalate from polynomial → spline → GAM as needed for flexibility.
5. Never extrapolate predictions outside the range of the training data's predictor values.
6. Use prediction intervals (not confidence intervals) when you need uncertainty bounds around one new prediction; the bootstrap is a transparent way to build either.
7. In big datasets, individual outliers/influential points rarely distort predictions much, but they're exactly what matters for anomaly detection and data-quality investigation.

## Connects To
- **Ch 3 (Statistical Experiments and Significance Testing)**: ANOVA's F-statistic and variance decomposition directly generalize into regression's model-level F-test and R².
- **Ch 5 (Classification)**: logistic regression extends this chapter's linear-model machinery to categorical/binary outcomes.
- **Ch 6 (Statistical Machine Learning)**: tree-based models automatically discover interactions/nonlinearities that here require manual polynomial/spline/interaction specification.
