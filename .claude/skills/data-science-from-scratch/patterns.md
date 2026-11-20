# Patterns

## Index Before Scan
**When to use**: Any lookup repeated many times over the same collection (friend lookups, interest matching, word counting).
**How**: Build a dict/defaultdict mapping key -> value(s) once (O(n) build), then do O(1) lookups thereafter, instead of re-scanning a raw list on every query.
**Trade-offs**: Pays a one-time build cost for much faster repeated access; wasteful if the lookup only happens once.

## Vector/Matrix as Type Aliases
**When to use**: Representing any fixed-length numeric record or dataset from Ch4 onward.
**How**: `Vector = List[float]`, `Matrix = List[List[float]]`; build `add`/`subtract`/`dot`/`magnitude`/`distance` once, reuse everywhere (regression, gradient descent, k-NN, clustering, neural nets).
**Trade-offs**: Transparent and dependency-free for teaching; far slower than NumPy for real workloads — swap in NumPy for production.

## Gradient Descent as Universal Optimizer
**When to use**: Fitting any differentiable loss function's parameters, whenever no closed-form solution exists (or as a cross-check when one does).
**How**: Derive the loss's gradient analytically, pick a step size, repeatedly `theta = theta - step_size * gradient` until convergence; choose batch/minibatch/stochastic based on dataset size.
**Trade-offs**: General and reusable across the whole book, but requires tuning step size and risks local minima on multimodal objectives.

## Train/Test/Validation Split
**When to use**: Every supervised model, to detect overfitting; add a validation set specifically when comparing multiple candidate models.
**How**: `split_data`/`train_test_split` shuffle and partition (e.g., 70/30 or 70/15/15), keeping paired (x,y) correspondence intact.
**Trade-offs**: Essential for honest evaluation; using the test set for model selection silently converts it into a second training set.

## Fail-Soft Parsing (Optional-Returning Parser)
**When to use**: Ingesting real-world delimited/scraped data with inconsistent quality.
**How**: Write a parser that returns `Optional[Record]` — `None` on failure — so the caller explicitly decides whether to skip, fix, or ignore bad rows, rather than crashing the whole pipeline.
**Trade-offs**: Much more robust than letting one bad row crash ingestion; requires an explicit policy decision about what to do with the `None`s.

## Rescale Before Distance/Regularization
**When to use**: Any distance-based technique (k-NN, clustering, PCA) or any regularized regression, when input dimensions have different/incomparable units.
**How**: Standardize each dimension to mean 0, standard deviation 1 (`rescale`) before computing distances or applying a penalty; transform learned coefficients back to original units afterward if needed for interpretation.
**Trade-offs**: Prevents unit choice from silently distorting results; adds a preprocessing/postprocessing step that must be remembered and applied consistently at train and prediction time.

## Bias-Variance Diagnosis Before Fixing
**When to use**: Whenever a model underperforms and you need to decide what to try next.
**How**: Check train-set performance too — poor even on training data = high bias (add features/complexity); great on training, poor on test = high variance (remove features, regularize, or get more data).
**Trade-offs**: More data only ever helps variance, never bias — applying the wrong fix wastes effort.

## Smoothing (Pseudocount) for Probability Estimation
**When to use**: Estimating any per-category probability from finite observed counts (Naive Bayes word probabilities, any frequency-based estimator).
**How**: `(count + k) / (total + 2k)` instead of a raw fraction, so unseen categories get a nonzero, non-catastrophic probability.
**Trade-offs**: Prevents one never-seen value from zeroing out an entire prediction; the smoothing strength `k` is itself a tunable hyperparameter.

## Bootstrap Resampling for Standard Errors
**When to use**: Estimating any statistic's uncertainty when no analytic formula exists or is convenient.
**How**: Resample the dataset with replacement many times, recompute the statistic on each resample, use the spread of results as the standard error.
**Trade-offs**: General-purpose and assumption-light; computationally expensive relative to a closed-form formula when one exists.

## Regularization for High-Dimensional Overfitting
**When to use**: A many-variable regression or embedding model prone to overfitting or hard to interpret.
**How**: Add an L2 (ridge, shrinks smoothly) or L1 (lasso, zeroes out some coefficients) penalty to the loss, scaled by `alpha`; always rescale features first.
**Trade-offs**: Reduces overfitting and can simplify the model (lasso), but the right `alpha` requires tuning and L1 isn't solvable by plain gradient descent.

## Ensemble via Bagging + Feature Subsampling
**When to use**: A single model (especially decision trees) overfits and you want lower variance without a fundamentally different algorithm.
**How**: Train many models, each on a bootstrap resample of the data and (for trees) a random subset of attributes at each split; combine via majority vote/averaging.
**Trade-offs**: Reduces variance by decorrelating errors across the ensemble; costs more compute and loses the interpretability of a single model.

## Layer/Loss/Optimizer Decomposition (Deep Learning)
**When to use**: Building or extending any neural network beyond a single hand-coded example.
**How**: Define a uniform `Layer` interface (forward/backward/params/grads), compose layers via `Sequential`, decouple the loss (SSE vs. SoftmaxCrossEntropy) and the update rule (GradientDescent vs. Momentum) from the architecture.
**Trade-offs**: Enables fast experimentation (swap any piece independently); more upfront abstraction than a single hard-coded training loop.

## Choosing k via the Elbow Method
**When to use**: k-means clustering when `k` isn't externally constrained by the problem.
**How**: Plot total squared clustering error against increasing `k`; pick the point where the error-reduction rate visibly flattens.
**Trade-offs**: A pragmatic heuristic, not a provably optimal choice; different runs/seeds can still converge to different local optima for a fixed `k`.

## Benchmark Against a Trivial Baseline
**When to use**: Before trusting any "sophisticated" model's reported improvement (regression, recommender systems, classifiers).
**How**: Always compute the dumbest reasonable baseline first (predict the mean, predict the majority class, popularity-based recommendation) and report the real model's improvement relative to it.
**Trade-offs**: Cheap insurance against over-crediting a model; skipping it risks mistaking a trivial result for a real win.

## Filter Before Join (Query Optimization)
**When to use**: Any relational query (SQL or from-scratch) combining a filter and a join.
**How**: Apply the `WHERE`/filter predicate to shrink a table before joining it to another, rather than joining first and filtering the (larger) result.
**Trade-offs**: Reduces the number of row-comparisons the join must perform; real SQL engines often do this automatically via a query planner, but a from-scratch or naive implementation requires doing it manually.
