# Chapter 14: Simple Linear Regression

## Core Idea
Simple linear regression models `y = beta*x + alpha + error`, finding alpha/beta via least squares (minimizing sum of squared errors) either through an exact closed-form formula or gradient descent, and evaluates fit quality with R-squared — with maximum likelihood estimation given as the statistical justification for why least squares is the right objective under normally-distributed errors.

## Frameworks Introduced
- **Least squares fit (closed-form)**: `beta = correlation(x,y) * stdev(y)/stdev(x)`, `alpha = mean(y) - beta*mean(x)` directly minimizes `sum_of_sqerrors`.
  - When to use: fitting a linear relationship between one predictor and one outcome when you want the exact, fast, closed-form answer.
  - How: `least_squares_fit(x, y)` returns `(alpha, beta)` using only Ch5's `correlation`, `standard_deviation`, `mean`.
- **Least squares via gradient descent**: treat `theta = [alpha, beta]` as parameters, derive `grad_a = sum(2*error(...))`, `grad_b = sum(2*error(...)*x_i)`, and apply Ch8's `gradient_step` repeatedly.
  - When to use: as a sanity check that gradient descent (the book's general tool) recovers the same answer as the closed form, and as a bridge to more complex models (Ch15+) where no closed form exists.
- **R-squared (coefficient of determination)**: `1 - sum_of_sqerrors/total_sum_of_squares`, the fraction of y's variation explained by the model; ranges [0, 1], where 0 means "no better than predicting the mean."
  - When to use: quantifying model fit quality beyond eyeballing a plot.
- **Maximum likelihood estimation (MLE) as justification**: under the assumption that regression errors are normally distributed with mean 0, maximizing the likelihood of the observed data is mathematically equivalent to minimizing the sum of squared errors — explains *why* least squares is a principled choice, not just a convenient one.

## Key Concepts
- **alpha, beta**: intercept and slope of the linear model `y_i = beta*x_i + alpha + epsilon`.
- **error term (epsilon)**: the "hopefully small" residual capturing factors the simple model doesn't account for.
- **sum_of_sqerrors**: `sum((predict(alpha,beta,x_i) - y_i)^2)`; squaring (rather than summing raw errors) prevents positive/negative errors from canceling out.
- **total_sum_of_squares**: total squared deviation of y from its own mean — the "baseline" variation a model of "always predict mean(y)" would leave entirely unexplained.
- **R-squared bounds**: must lie in [0, 1] because the least-squares fit can never do worse than always-predict-the-mean (whose R-squared is 0 by construction) nor have negative sum of squared errors.
- **Likelihood function**: `L(theta | data)`, the probability of the observed data viewed as a function of the unknown parameter; MLE picks the theta maximizing it.

## Mental Models
- **alpha/beta as "mean in, mean out" plus a scaled deviation**: alpha says "predict the average y at the average x"; beta says "a one-stdev increase in x moves the prediction by `correlation(x,y)` stdevs of y" — correlation directly controls how strongly x-changes propagate to predictions.
- **R-squared as "how much better than guessing the mean"**: the least-squares model is guaranteed to be at least as good as a flat "always predict mean(y)" model, which anchors R-squared's lower bound at 0 and gives an intuitive baseline to beat.
- **Squared error over raw error, always**: raw errors can cancel (one prediction too high, another too low, summing near zero while both are bad); squaring forces all errors to contribute positively to the total.
- **Gradient descent as a general-purpose cross-check**: when a closed-form solution exists, running gradient descent anyway and confirming it converges to the same answer builds confidence in the machinery you'll need for models (Ch15+) that have no closed form.

## Anti-patterns
- **Summing raw (unsquared) errors to judge fit**: positive and negative errors cancel, hiding large individual mistakes — always square (or otherwise make errors non-negative) before summing.
- **Using R-squared alone without checking what it actually means**: an R-squared of 0.329 means the model is "only sort of okay" — don't over-trust a single summary number; also inspect the plot and consider what unaccounted factors might exist.
- **Assuming correlation justifies causation when building the model**: the chapter explicitly notes "assuming you've convinced yourself that having more friends causes people to spend more time" — the regression model itself doesn't establish causality (see Ch5's correlation/causation caveats).

## Code Examples
```python
def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta
```
- **What it demonstrates**: the closed-form least-squares solution built entirely from Ch5 statistics primitives.

```python
def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y))
```
- **What it demonstrates**: measuring what fraction of y's variance the fitted line explains.

```python
grad_a = sum(2 * error(alpha, beta, x_i, y_i) for x_i, y_i in zip(xs, ys))
grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i for x_i, y_i in zip(xs, ys))
guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
```
- **What it demonstrates**: deriving and applying the gradient of squared error with respect to alpha and beta, reusing Ch8's `gradient_step`.

## Reference Tables
| Quantity | Formula | Range |
|---|---|---|
| beta | `correlation(x,y) * stdev(y)/stdev(x)` | any real |
| alpha | `mean(y) - beta*mean(x)` | any real |
| R-squared | `1 - SSE/SST` | [0, 1] |
| R-squared = 0 | model no better than predicting `mean(y)` | baseline |
| R-squared = 1 | perfect fit, zero error | best case |

## Worked Example
Fitting DataSciencester's outlier-removed `num_friends_good` / `daily_minutes_good` data (from Ch5) gives `alpha ≈ 22.95`, `beta ≈ 0.903` via both the closed-form formula and 10,000 epochs of gradient descent (learning rate 0.00001, starting from random guess) — confirming both methods agree. The model predicts a user with 0 friends spends ~23 minutes/day on the site, and ~0.9 more minutes per additional friend. R-squared = 0.329, meaning the model explains about a third of the variation in daily minutes — "only sort of okay," signaling other factors are at play (motivating Ch15's multiple regression). A synthetic sanity-check dataset (`y = 3x - 5`) confirms `least_squares_fit` recovers the exact known relationship.

## Key Takeaways
1. Least squares minimizes sum of squared errors; it has both an exact closed form (for simple regression) and a gradient-descent solution that converges to the same answer.
2. R-squared quantifies fit quality as "fraction of variance explained," bounded in [0,1] with 0 as the always-predict-the-mean baseline.
3. Under a normal-errors assumption, least squares is exactly the maximum likelihood estimate — a principled, not just convenient, choice of objective.
4. A low-to-moderate R-squared (like 0.329 here) signals that important explanatory factors are missing — a motivation for adding more predictors (Ch15).
5. Correlation/causation caveats from Ch5 still apply: fitting a regression line doesn't by itself establish that x causes y.

## Connects To
- **Ch5**: `correlation`, `standard_deviation`, `mean`, `de_mean` are the direct building blocks of `least_squares_fit` and `r_squared`.
- **Ch8**: the gradient-descent solution reuses `gradient_step` and the general optimization framework.
- **Ch15 (Multiple Regression)**: the natural next step when one predictor (R-squared 0.329) isn't enough — adds more explanatory variables.
- **Ch16 (Logistic Regression)**: reuses the same gradient-descent-based fitting approach for a classification (rather than regression) target.
