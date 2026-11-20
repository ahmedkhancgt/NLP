# Patterns and Techniques

## Vectorize instead of looping
**When to use**: any Python `for` loop touching NumPy array/Series elements one at a time.
**How**: replace with ufunc expressions (`1.0 / arr`), `.str` methods for strings, or Pandas vectorized ops.
**Trade-offs**: orders-of-magnitude speedup; loses nothing in readability once idiomatic — only con is a learning curve for broadcasting rules.

## Boolean masking for filter+aggregate
**When to use**: selecting/counting elements meeting a condition.
**How**: build a Boolean array via comparison ops, then use it both as `np.sum(mask)` (count) and `arr[mask]` (extract); combine conditions with `&`/`|`/`~`, parenthesized.
**Trade-offs**: very fast and composable; must remember `&`/`|` not `and`/`or` on arrays.

## loc/iloc for unambiguous indexing
**When to use**: any Pandas indexing where the index could be integer-valued or ambiguity is possible.
**How**: `.loc[...]` for label-based (inclusive slices), `.iloc[...]` for position-based (exclusive slices); combine mask + fancy indexing in one call: `df.loc[mask, ['col1','col2']]`.
**Trade-offs**: slightly more verbose than bare brackets; eliminates an entire class of indexing bugs.

## groupby / pivot_table for split-apply-combine
**When to use**: summarizing data conditional on one or more categorical keys.
**How**: `df.groupby(key).agg(...)` for flexible multi-op aggregation; `df.pivot_table(values, index, columns, aggfunc=)` for the common 2D case; bin continuous variables first with `pd.cut`/`pd.qcut` to make them groupable.
**Trade-offs**: pivot_table is more readable for 2D summaries; groupby is more flexible (filter/transform/apply) for custom logic.

## Index alignment for safe combination
**When to use**: combining Series/DataFrames from different sources with potentially mismatched labels.
**How**: rely on Pandas' automatic union-based alignment (fills gaps with NaN); use `.add(other, fill_value=)` when a non-NaN default is wanted.
**Trade-offs**: safer than raw NumPy array combination; must remember to check `.isnull().any()` afterward to catch real mismatches.

## Pipeline chaining for preprocessing + model
**When to use**: any multi-step feature engineering + model fitting workflow.
**How**: `make_pipeline(transformer1, transformer2, ..., estimator)`; the result supports `.fit()`/`.predict()` like a single estimator and plugs into `GridSearchCV`/`cross_val_score` directly.
**Trade-offs**: essential for reproducibility and avoiding data leakage (e.g., fitting a scaler on test data); adds a small indirection layer when debugging.

## Basis function / kernel expansion for nonlinearity
**When to use**: a linear model underfits data with a clearly nonlinear relationship.
**How**: expand features via `PolynomialFeatures`, a custom Gaussian-basis transformer, or switch to kernel SVM (`kernel='rbf'`) — "transform the input, not the model."
**Trade-offs**: dramatically increases flexibility; risks overfitting without regularization (ridge/lasso) or careful hyperparameter (degree, gamma) tuning via cross-validation.

## Regularization to control basis-expansion overfitting
**When to use**: a flexible/high-degree basis expansion fits training data well but overfits.
**How**: `Ridge(alpha=)` for smooth shrinkage, `Lasso(alpha=)` for sparse feature selection; tune `alpha` via cross-validation.
**Trade-offs**: ridge is cheap and stable; lasso gives interpretable sparsity but can be less stable with correlated features.

## Validation/learning curves for model diagnosis
**When to use**: deciding whether to increase/decrease model complexity, or whether more data would help.
**How**: `validation_curve` (score vs. complexity) finds the bias-variance sweet spot; `learning_curve` (score vs. training-set size) reveals convergence (more data won't help) vs. non-convergence (more data would help).
**Trade-offs**: adds computation (many refits); essential to avoid guessing wrong about the right fix for underperformance.

## Baseline-then-upgrade model selection
**When to use**: starting any new classification/regression task.
**How**: fit the fastest, simplest, most interpretable model first (naive Bayes for classification, linear regression for regression) to establish a performance floor; only move to SVM/random forest/deep learning if the baseline is insufficient.
**Trade-offs**: saves time and avoids premature complexity; occasionally the simple model is "good enough" and no further work is needed.

## PCA-first exploratory workflow for high-dimensional data
**When to use**: any new high-dimensional dataset (images, many-feature tables).
**How**: run PCA first to visualize relationships (2-3 components), understand variance structure (cumulative explained variance plot), and get a sense of intrinsic dimensionality; reach for manifold learning (Isomap/LLE/t-SNE) only if PCA's linear assumption visibly fails.
**Trade-offs**: PCA is fast, interpretable, and robust to missing data/noise; manifold learning is slower, less interpretable, but captures nonlinear structure PCA misses.

## Ensemble averaging to fight overfitting
**When to use**: a flexible base model (e.g., decision tree) overfits individually.
**How**: bag/average many randomized instances of the base model (`BaggingClassifier`, or the optimized `RandomForestClassifier`/`Regressor`).
**Trade-offs**: trades interpretability for robustness/accuracy; randomization (data subsampling + feature-selection-per-split) is what makes averaging effective.

## Custom Scikit-Learn estimator
**When to use**: a domain-specific model/transformer doesn't exist in Scikit-Learn but needs to plug into its pipeline/grid-search machinery.
**How**: subclass `BaseEstimator` (+ `ClassifierMixin`/`TransformerMixin` as appropriate); `__init__` only assigns args to `self` (no logic, no `*args`/`**kwargs`); `fit()` returns `self` and stores learned attributes with a trailing underscore; implement `predict`/`transform`/`predict_proba` as needed.
**Trade-offs**: unlocks full Scikit-Learn tooling (GridSearchCV, pipelines, cross_val_score) for custom logic; requires strict adherence to conventions or cloning/cross-validation silently breaks.

## Sliding window for spatial/image detection
**When to use**: detecting an object (face, pattern) whose location in a larger image/signal is unknown.
**How**: step a fixed- (or multi-) scale window across the data, extract features (e.g., HOG) per window, classify each with a trained model; merge overlapping positive detections afterward (clustering or non-maximum suppression).
**Trade-offs**: simple and effective at small scale; naive single-scale/no-merge versions miss size variation and produce redundant detections — production use needs multi-scale search and deduplication.
