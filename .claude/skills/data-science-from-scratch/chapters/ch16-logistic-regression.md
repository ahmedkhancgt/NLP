# Chapter 16: Logistic Regression

## Core Idea
Linear regression breaks down for a 0/1 classification target (unbounded predictions, biased coefficients), so logistic regression squashes `dot(x, beta)` through the logistic function into a [0,1] probability and fits beta by maximizing the (log) likelihood via gradient descent — with support vector machines introduced as an alternative classification approach that finds a maximum-margin separating hyperplane instead.

## Frameworks Introduced
- **Logistic regression**: model `P(y=1|x) = f(dot(x, beta))` where `f` is the logistic function; fit by maximizing log-likelihood (equivalently minimizing negative log-likelihood) via gradient descent, since (unlike simple/multiple regression) there's no least-squares equivalence here.
  - When to use: binary classification where you want calibrated probability outputs, not just a 0/1 label.
  - How: derive `negative_log_likelihood` and its gradient (`negative_log_gradient`) analytically, then run standard gradient descent (Ch8 machinery) to find beta.
- **Logistic function**: `f(x) = 1/(1+exp(-x))`, maps any real number to (0,1), with convenient derivative `f'(x) = f(x)*(1-f(x))`.
  - When to use: converting an unbounded linear score into a probability-like, bounded output — the canonical "squashing function" for binary classification.
- **Precision/recall evaluation on held-out test data**: threshold predicted probability at 0.5, compute true/false positives/negatives, then precision and recall (as in Ch11).
- **Decision boundary / hyperplane**: the set of points where `dot(x_i, beta) = 0` separates predicted classes — a direct geometric view of what the fitted logistic model is doing.
- **Support Vector Machines (SVM) and the kernel trick** (introduced conceptually, not implemented): find the hyperplane that maximizes the margin to the nearest point of each class; when no hyperplane separates classes in the original space, map data into higher dimensions (e.g., `x -> (x, x**2)`) where it becomes separable, using a kernel function to avoid the cost of explicit mapping.
  - When to use (conceptually): classification problems needing a maximum-margin boundary rather than a probabilistic model; acknowledged as requiring specialized optimization software beyond from-scratch scope.

## Key Concepts
- **Why linear regression fails for 0/1 targets**: predictions are unbounded (can be far outside [0,1], hard to interpret as probability) and, because actual y is capped at 0/1 while the linear model's output isn't, the error term becomes correlated with x for extreme inputs — biasing beta (the same assumption-violation flagged in Ch15).
- **Log-likelihood**: sum over data points of `log(P(y_i | x_i, beta))`; maximizing it (equivalently minimizing negative log-likelihood) is the fitting objective, since log is monotonic so the same beta maximizes both likelihood and log-likelihood.
- **Rescaling before fitting**: logistic regression here is fit on `rescale(xs)`-ed inputs (Ch10's standardization), then coefficients are transformed back to the original scale for interpretation.
- **All-else-equal coefficient interpretation, but nonlinear impact**: a coefficient's effect on the *logistic input* is constant, but its effect on the output *probability* depends on where you are on the S-curve — the same `beta_1` shift matters much more near `dot(x,beta)=0` than far out in either tail.
- **Separating hyperplane**: a decision boundary (`dot(x_i, beta) = 0` for logistic regression, or SVM's margin-maximizing hyperplane) splitting feature space into predicted classes.

## Mental Models
- **Classification needs a squashing function, not a raw linear score**: logistic regression's entire motivation is "we want large positive linear scores to map near 1, large negative scores near 0" — any function with that shape (logistic, or others) could in principle serve this role.
- **Likelihood maximization replaces squared-error minimization as the fitting criterion** when the target isn't continuous: for linear regression under normal errors, least squares and MLE coincided (Ch14); for logistic regression they don't, so the book switches to direct (negative) log-likelihood minimization via gradient descent.
- **Logistic regression (probabilistic) vs. SVM (margin-maximizing) are two different philosophies for the same binary classification task**: logistic regression finds the boundary as a byproduct of fitting a probability model; SVM directly optimizes the boundary's separation margin, with no inherent probability interpretation.
- **Kernel trick as a shortcut around expensive feature mapping**: rather than explicitly transforming points into a higher-dimensional space to make them separable, compute the dot products that would result from that mapping directly via a kernel function — avoiding the cost of the explicit transform.

## Anti-patterns
- **Using plain linear regression for a 0/1 classification target**: produces unbounded, hard-to-interpret predictions and biased coefficients because the error term becomes correlated with x at the extremes.
- **Interpreting a logistic coefficient the same way as a linear regression coefficient**: its effect on *probability* (not just the linear score) is nonlinear and depends on where the baseline prediction already sits — "all else equal" only cleanly applies to the pre-squashing linear score.
- **Expecting a separating hyperplane to always exist**: many real datasets (like the "who pays?" example) have no line that perfectly separates classes — SVMs handle this via margin maximization plus the kernel trick, not by assuming perfect separability.
- **Attempting to implement SVMs fully from scratch**: the book explicitly stops short, noting it "probably [isn't] a good idea" without specialized optimization software (e.g., LIBSVM).

## Code Examples
```python
def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)

def _negative_log_likelihood(x: Vector, y: float, beta: Vector) -> float:
    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else:
        return -math.log(1 - logistic(dot(x, beta)))
```
- **What it demonstrates**: the logistic function's convenient derivative, and per-point negative log-likelihood used as the loss to minimize.

```python
def _negative_log_partial_j(x, y, beta, j):
    return -(y - logistic(dot(x, beta))) * x[j]

def negative_log_gradient(xs, ys, beta):
    return vector_sum([_negative_log_gradient(x, y, beta) for x, y in zip(xs, ys)])
```
- **What it demonstrates**: the analytically-derived gradient of negative log-likelihood, summed across the dataset, fed into standard `gradient_step`.

## Reference Tables
| Model | Output | Fitting objective | Handles non-separable data |
|---|---|---|---|
| Linear regression | unbounded real | sum of squared errors | n/a (not for classification) |
| Logistic regression | probability [0,1] | (negative) log-likelihood | yes (soft boundary) |
| SVM | class label (+ margin) | maximize margin | only with kernel trick |

## Worked Example
Predicting "paid premium account" (0/1) from `[experience, salary]` for ~200 DataSciencester users. A naive linear regression on rescaled inputs gives beta `[0.26, 0.43, -0.43]` but produces many negative/out-of-range predictions. Switching to logistic regression (gradient descent, 5000 epochs, learning rate 0.01) on a 67/33 train/test split gives rescaled `beta ≈ [-2.0, 4.7, -4.5]`, which transforms back to unscaled coefficients `≈ [8.9, 1.6, -0.000288]` — meaning more experience increases the probability of paying, higher salary decreases it (holding the other fixed), matching the earlier scatterplot intuition. On held-out test data, thresholding at 0.5 gives 75% precision and 80% recall. The decision boundary `dot(beta, x_i)=0` is plotted directly; the chapter notes no straight line perfectly separates the paid/unpaid users, motivating the SVM/kernel-trick discussion (illustrated abstractly with a 1D-to-2D `x -> (x, x^2)` mapping that makes an otherwise inseparable dataset linearly separable).

## Key Takeaways
1. Use logistic regression (not linear regression) for 0/1 classification targets — it bounds predictions to [0,1] and avoids the bias linear regression would introduce.
2. Fit logistic regression by minimizing negative log-likelihood via gradient descent, since no least-squares-equivalent closed form applies.
3. Interpret logistic coefficients carefully: their effect on the underlying linear score is constant, but their effect on the final probability is nonlinear and context-dependent.
4. SVMs offer an alternative, margin-maximizing approach to classification, with the kernel trick enabling separation in cases with no simple separating hyperplane — but require specialized optimization tooling beyond from-scratch implementation.
5. Always evaluate on held-out data with precision/recall (not training accuracy) before trusting a classifier.

## Connects To
- **Ch1**: revisits the exact "predict paid account from years of experience" problem first posed there with an eyeballed heuristic.
- **Ch8**: gradient descent is the fitting engine, applied to a new (log-likelihood) objective.
- **Ch10**: `rescale`/`scale` are used before fitting and to transform coefficients back for interpretation.
- **Ch11**: precision/recall evaluation methodology directly reused.
- **Ch15**: the omitted-variable-bias / assumption-violation reasoning from multiple regression explains exactly why linear regression fails here.
- **scikit-learn / LIBSVM**: external libraries providing production logistic regression and SVM implementations.
