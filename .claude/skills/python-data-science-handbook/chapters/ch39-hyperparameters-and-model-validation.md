# Chapter 39: Hyperparameters and Model Validation

## Core Idea
Evaluating a model on the same data it was trained on gives a meaningless (often perfect) score; proper validation (holdout sets, cross-validation) combined with understanding the bias-variance trade-off (via validation curves and learning curves) is essential to choosing the right model complexity.

## Frameworks Introduced
- **Holdout validation**: `train_test_split(X, y, train_size=...)` — train on one split, evaluate on the other never-seen split.
  - When to use: always, instead of evaluating on training data.
- **Cross-validation**: repeatedly splitting data into train/validation folds so every point serves as validation exactly once, averaging results for a more robust performance estimate without losing training data permanently to one holdout.
  - `cross_val_score(model, X, y, cv=k)`: k-fold cross-validation convenience function.
  - `LeaveOneOut()`: extreme case, folds = number of samples.
- **Bias-variance trade-off**: the central lens for model selection.
  - High bias (underfitting): model too simple/inflexible; training score ≈ validation score (both mediocre).
  - High variance (overfitting): model too flexible; training score high, validation score much lower.
  - Validation curve (score vs. model complexity): has a maximum at the bias-variance sweet spot.
- **Learning curve** (score vs. training-set size): reveals whether more data would help (curves still converging) or not (already converged — need a different/more complex model instead).
- **Grid search**: `GridSearchCV(estimator, param_grid, cv=k)` — automates the "try every hyperparameter combination, pick the best cross-validated score" process; exposes `.best_params_` and `.best_estimator_` after `.fit()`.

## Key Concepts
- **R² score (coefficient of determination)**: 1 = perfect fit, 0 = no better than predicting the mean, negative = worse than the mean.
- **`validation_curve(model, X, y, param_name=, param_range=, cv=)`**: computes train/validation scores across a hyperparameter range in one call.
- **`learning_curve(model, X, y, cv=, train_sizes=)`**: computes train/validation scores across increasing training-set sizes.
- **Four "underperforming model" remedies**: more flexible model, less flexible model, more training samples, more features — the *correct* remedy is often counterintuitive and must be diagnosed (not guessed) via validation/learning curves.
- **Pipelines**: `make_pipeline(PolynomialFeatures(degree), LinearRegression())` — chains preprocessing + estimator into one composable object (previewed here, detailed in Ch 40).

## Mental Models
- "Train and evaluate on the same data" is a methodological trap, not just suboptimal — an instance-based model like 1-NN will score 100% every time under this scheme, telling you literally nothing about generalization.
- A validation curve answers "how complex should my model be?"; a learning curve answers "would more data help?" — these are two orthogonal diagnostic questions, and conflating them leads to wrong remedies (e.g., gathering more data when the real problem is model complexity, or vice versa).
- Training score is *always* ≥ validation score (except by chance) — the gap between them, not either score alone, is the signal that distinguishes high-bias (small gap, both low) from high-variance (large gap, train high/val low) regimes.

## Anti-patterns
- **Evaluating model accuracy on the training set**: produces an artificially inflated (sometimes perfect) score that says nothing about generalization — always use a holdout set or cross-validation.
- **Using a single train/test split when data is limited**: wastes a large fraction of data from training and gives a single noisy estimate — use k-fold cross-validation to reuse all data across folds and average for robustness.
- **Assuming "more complex model" or "more data" is always the fix for underperformance**: both can fail to help, or even hurt, depending on whether the problem is bias or variance — diagnose with validation/learning curves before acting.
- **Reading a learning curve's "already converged" state as a reason to collect more data**: once training and validation scores have converged, additional data won't help — a different (often more complex) model is needed instead.

## Code Examples
```python
from sklearn.datasets import load_iris
iris = load_iris(); X = iris.data; y = iris.target

# WRONG: evaluate on training data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)
y_model = model.predict(X)
accuracy_score(y, y_model)                      # 1.0 -- meaningless!

# RIGHT: holdout set
from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
model.fit(X1, y1)
accuracy_score(y2, model.predict(X2))            # ~0.907, realistic

# two-fold cross-validation by hand
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)   # (0.96, 0.907)

# k-fold cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)

# leave-one-out
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
scores.mean()                                      # 0.96

# polynomial regression pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

# validation curve
from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(
    PolynomialRegression(), X, y,
    param_name='polynomialfeatures__degree', param_range=degree, cv=7)
# plot median(train_score) vs median(val_score) across degree -> find peak

# learning curve
from sklearn.model_selection import learning_curve
N, train_lc, val_lc = learning_curve(
    PolynomialRegression(degree), X, y, cv=7, train_sizes=np.linspace(0.3, 1, 25))

# grid search
from sklearn.model_selection import GridSearchCV
param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False]}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)
grid.best_params_          # {'linearregression__fit_intercept': False, 'polynomialfeatures__degree': 4}
model = grid.best_estimator_
```
- **What it demonstrates**: the naive-vs-correct validation contrast, cross-validation at increasing rigor (2-fold → k-fold → leave-one-out), polynomial-regression validation curves (finding optimal degree), learning curves (diagnosing whether more data would help), and automated hyperparameter grid search.

## Reference Tables
| Model regime | Training score | Validation score | Gap | Remedy |
|---|---|---|---|---|
| High bias (underfit) | Low-moderate | Similar to training | Small | More flexible model, more features |
| High variance (overfit) | High | Much lower | Large | Less flexible model, more data |
| Good fit | Moderate-high | Close to training | Small, both decent | — |

| Validation tool | Question answered |
|---|---|
| Holdout / cross-validation | How well does this specific model generalize? |
| Validation curve | What model complexity is optimal? |
| Learning curve | Would more training data help? |
| Grid search | What's the best hyperparameter combination (multi-dimensional)? |

## Worked Example
Polynomial regression on noisy data: fitting degree 1 (underfit, high bias), 3, and 5 polynomials to 40 points shows visually different fit quality; a `validation_curve` over degrees 0-20 confirms degree 3 maximizes validation score (training score rises monotonically, validation score peaks then falls = overfitting beyond that point). Repeating with 5x more data (200 points) shifts the optimal degree to ~6, and even degree-20 no longer seriously overfits — demonstrating that "optimal complexity" is itself a function of dataset size, which is exactly what learning curves diagnose directly. `GridSearchCV` over `{degree: 0-20, fit_intercept: [True, False]}` automates this search, finding `degree=4, fit_intercept=False` as best for the original 40-point dataset.

## Key Takeaways
1. Never evaluate a model on its own training data — always use a holdout set or cross-validation.
2. Cross-validation (`cross_val_score`, `LeaveOneOut`) reuses all data across multiple train/validation splits for a more robust performance estimate than a single holdout.
3. The bias-variance trade-off governs model selection: underfitting (high bias, small train-val gap, both scores low) vs. overfitting (high variance, large train-val gap, train high/val low).
4. Validation curves (score vs. complexity) find the optimal model complexity; learning curves (score vs. training-set size) reveal whether more data would help or whether a different model is needed.
5. `GridSearchCV` automates hyperparameter search across multiple dimensions, exposing `.best_params_` and `.best_estimator_`.
6. The "correct" response to a poorly performing model (more data? more complexity? less complexity? more features?) must be diagnosed via these tools, not assumed.

## Connects To
- **Ch 38**: builds directly on the five-step Estimator API and `train_test_split` introduced there.
- **Ch 40**: pipelines (`make_pipeline`) are introduced here and covered more fully next.
- **Ch 42**: polynomial/linear regression concepts used throughout this chapter's examples are detailed in the dedicated regression chapter.
- **Ch 49/50**: referenced for further grid-search examples in applied contexts.
