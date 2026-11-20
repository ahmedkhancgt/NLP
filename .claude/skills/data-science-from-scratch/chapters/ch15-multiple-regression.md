# Chapter 15: Multiple Regression

## Core Idea
Multiple regression extends simple linear regression to a vector of predictors (`y = dot(x, beta)`), fit via gradient descent rather than a closed form; the chapter covers its key assumptions (no collinearity, errors uncorrelated with inputs), interpreting coefficients "all else equal," goodness of fit, bootstrap-estimated standard errors/significance, and regularization (ridge/lasso) to combat overfitting with many variables.

## Frameworks Introduced
- **Multiple regression model**: `y_i = beta_0 + beta_1*x_i1 + ... + beta_k*x_ik + epsilon`, represented as `predict(x, beta) = dot(x, beta)` after prepending a constant `1` to each input vector.
  - When to use: when more than one predictor explains the outcome (e.g., friends, work hours, PhD status predicting minutes on site).
- **Dummy variables**: encode non-numeric categorical facts (e.g., "has PhD") as 0/1 numeric columns so they fit the linear model.
  - When to use: any categorical predictor in a linear/regression context.
- **Gradient-descent fitting for multiple regression**: `sqerror_gradient(x, y, beta) = [2*err*x_i for x_i in x]`; `least_squares_fit` runs minibatch gradient descent since no simple closed form exists for many predictors (within this book's from-scratch scope).
- **The bootstrap**: resample the dataset with replacement (`bootstrap_sample`) many times, recompute a statistic (e.g., median, or a regression beta) on each resample, and use the spread of those estimates as a standard-error estimate — a general technique for quantifying uncertainty without analytic formulas.
  - When to use: estimating standard errors/confidence for any statistic when you lack (or want to avoid) an analytic formula.
- **Regularization (ridge and lasso)**: add a penalty term to the squared-error objective that grows with the size of beta, trading goodness-of-fit for smaller/sparser coefficients.
  - When to use: many-variable regressions prone to overfitting or hard-to-interpret coefficient sets; ridge shrinks coefficients smoothly, lasso drives some to exactly zero (sparse models) but isn't solvable by plain gradient descent.

## Key Concepts
- **Collinearity (linear dependence) of predictors**: if one column can be written as a weighted sum of others (e.g., `num_acquaintances == num_friends`), beta becomes unidentifiable — no unique solution exists.
- **Omitted-variable bias**: if a predictor correlated with both y and an included predictor is left out, the included predictor's coefficient becomes systematically biased (illustrated with work-hours omission inflating/deflating the friends coefficient).
- **All-else-being-equal interpretation**: each coefficient `beta_j` is the expected change in y per unit increase in `x_j`, holding every other predictor fixed.
- **Multiple R-squared**: same formula as Ch14 but with vector inputs; mechanically can only increase (or stay the same) as more variables are added — not proof that added variables matter.
- **Standard error of a coefficient**: how much a coefficient estimate varies across resamples/datasets; larger standard error means less confidence in that coefficient's value.
- **t-statistic / p-value for a coefficient**: `t_j = beta_j / sigma_j`; tests the null hypothesis `beta_j = 0`; with large degrees of freedom, approximate via the normal CDF instead of the (unimplemented) Student's t-distribution.
- **Ridge penalty**: `alpha * dot(beta[1:], beta[1:])` (L2, squared-magnitude penalty on non-constant coefficients); shrinks coefficients smoothly toward 0 as `alpha` increases.
- **Lasso penalty**: `alpha * sum(abs(beta_i) for beta_i in beta[1:])` (L1 penalty); tends to force some coefficients to exactly 0, yielding sparse models — but not solvable via plain gradient descent (non-differentiable at 0).

## Mental Models
- **More variables always raises R-squared, but doesn't mean they matter**: simple regression is the special case where extra coefficients are forced to 0, so adding any variable to a multiple regression can only help (or not hurt) the training-set fit — significance testing (standard errors, p-values) is needed to tell real signal from noise.
- **Omitted variables bias remaining coefficients when correlated with both the omission and an included predictor**: this is a concrete, worked version of Ch5's confounding/Simpson's-paradox warning, now inside a regression context.
- **Bootstrap replaces analytic formulas with brute-force resampling**: when you can't (or don't want to) derive a standard-error formula, simulate "redrawing the sample" via resampling-with-replacement and measure how much the estimate wobbles.
- **Rescale before regularizing**: a penalty on raw coefficient magnitude is unit-dependent — switching "years of experience" to "centuries of experience" inflates that coefficient 100x and makes it look (wrongly) more penalizable, so standardize features first.
- **Regularization as an overfitting/interpretability dial**: increasing `alpha` trades training-set fit for smaller (ridge) or sparser (lasso) coefficients — useful both to prevent overfitting and to produce simpler, more explainable models.

## Anti-patterns
- **Including perfectly (or near-perfectly) collinear predictors**: makes beta unidentifiable — infinitely many coefficient combinations give identical predictions.
- **Omitting a predictor correlated with both the outcome and an included variable**: silently biases the remaining coefficients (demonstrated via the friends/work-hours example).
- **Trusting R-squared alone to justify adding variables**: it mechanically increases with more predictors regardless of whether they're meaningful — check standard errors/p-values too.
- **Regularizing without rescaling features first**: unit changes (years -> centuries) distort which coefficients get penalized how much.
- **Treating a high p-value coefficient as reliable**: the "PhD" coefficient (0.923, SE 1.249) has p > 0.4 — not significantly different from 0, and indeed vanishes under ridge regularization as `alpha` grows, consistent with it being noise.

## Code Examples
```python
def predict(x: Vector, beta: Vector) -> float:
    """assumes that the first element of x is 1"""
    return dot(x, beta)

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]
```
- **What it demonstrates**: generalizing simple regression's prediction/gradient to an arbitrary-length parameter vector via `dot`.

```python
def bootstrap_sample(data: List[X]) -> List[X]:
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data, stats_fn, num_samples):
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]
```
- **What it demonstrates**: the general bootstrap resampling pattern, later applied to estimate regression-coefficient standard errors.

```python
def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def sqerror_ridge_gradient(x, y, beta, alpha):
    return add(sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))
```
- **What it demonstrates**: adding an L2 penalty to both the loss and its gradient for ridge regression, leaving the constant term unpenalized.

## Reference Tables
| alpha (ridge) | beta (approx) | sum(beta[1:]^2) | R-squared |
|---|---|---|---|
| 0.0 | [30.51, 0.97, -1.85, 0.91] | 5-6 | 0.67-0.69 |
| 0.1 | [30.8, 0.95, -1.83, 0.54] | 4-5 | 0.67-0.69 |
| 1 | [30.6, 0.90, -1.68, 0.10] | 3-4 | 0.67-0.69 |
| 10 | [28.3, 0.67, -0.90, -0.01] | 1-2 | 0.5-0.6 |

| Penalty | Formula | Effect |
|---|---|---|
| Ridge (L2) | `alpha * sum(beta_i^2)` | shrinks coefficients smoothly |
| Lasso (L1) | `alpha * sum(\|beta_i\|)` | drives some coefficients to exactly 0 (sparse); not gradient-descent-friendly |

## Worked Example
Fitting `minutes = beta_0 + beta_1*friends + beta_2*work_hours + beta_3*phd` via 5,000 epochs of minibatch (batch_size=25) gradient descent yields `beta ≈ [30.58, 0.972, -1.87, 0.923]` — matching the exact linear-algebra solution. R-squared rises to 0.68 (up from 0.329 in Ch14's single-variable model). Bootstrapping 100 resampled datasets and refitting beta each time gives standard errors `[1.272, 0.103, 0.155, 1.249]`; converting to p-values shows friends and work_hours are highly significant (p < 0.001) but PhD is not (p > 0.4) — and indeed, as ridge's `alpha` increases from 0 to 10, the PhD coefficient shrinks toward (and essentially to) 0 first, consistent with it being noise rather than real signal.

## Key Takeaways
1. Multiple regression generalizes simple regression via `dot(x, beta)`; without a from-scratch closed form, fit it with gradient descent.
2. Watch for collinearity (unidentifiable beta) and omitted-variable bias (systematically wrong coefficients) — both assumptions must hold for beta estimates to be trustworthy.
3. R-squared always rises (or stays flat) with more variables — use standard errors and p-values (via bootstrap + normal approximation) to judge which coefficients are actually meaningful.
4. The bootstrap is a general-purpose, from-scratch way to estimate any statistic's standard error by resampling with replacement.
5. Regularization (ridge shrinks, lasso zeroes out) trades fit for smaller/sparser, more interpretable, less overfit coefficient sets — but rescale features first.

## Connects To
- **Ch14**: direct generalization of simple linear regression's model, error, and R-squared machinery to vector inputs.
- **Ch5**: the omitted-variable-bias discussion is a concrete regression instance of Ch5's correlation/Simpson's-paradox confounding warnings; bootstrap reuses `median`/`standard_deviation`.
- **Ch8**: gradient descent (including minibatching) is the fitting engine throughout.
- **Ch11**: regularization directly implements the "remove features to fix high variance" fix for overfitting discussed there.
- **scikit-learn / Statsmodels**: external libraries providing exact closed-form linear regression, ridge, lasso, and proper t-distribution-based significance testing.
