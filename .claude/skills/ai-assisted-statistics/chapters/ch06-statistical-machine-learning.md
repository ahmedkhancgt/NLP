# Chapter 6: Statistical Machine Learning

## Core Idea
Data-driven, non-parametric learners — KNN, decision trees, and especially tree ensembles (random forests, boosting/XGBoost) — automatically discover complex patterns and interactions without imposing a linear structure, but this power comes with a bias-variance trade-off and real overfitting risk that must be actively managed via hyperparameters and cross-validation.

## Frameworks Introduced
- **K-Nearest Neighbors (KNN)**: classify/predict a new record using the majority class (or average) of its K most similar training records.
  - When to use: simple baseline classification/regression; also valuable as a "feature engine" (stage-1 KNN score fed into a stage-2 model, mimicking a realtor's "comps").
  - How: standardize predictors to z-scores first (critical — unscaled variables with large magnitude dominate distance), pick K via cross-validation (typical range 1–20, odd to avoid ties).
- **Decision Trees (CART / recursive partitioning)**: repeatedly split the data on the predictor/threshold that most increases within-partition class homogeneity (minimizes Gini impurity or entropy).
  - When to use: need an interpretable rule-based model, or as the base learner for ensembles.
  - How: `rpart()` / `DecisionTreeClassifier`; control overfitting via `min_n`/`min_samples_leaf`, `tree_depth`/`max_depth`, or cost-complexity pruning (`cp`/`ccp_alpha`), tuned via cross-validation.
- **Bagging / Random Forest**: bootstrap-aggregate many trees, each also restricted to a random subset of predictors at each split, then average/majority-vote their predictions.
  - When to use: strong off-the-shelf predictive accuracy with minimal tuning; need variable importance rankings.
  - How: `rand_forest()` (ranger engine) / `RandomForestClassifier`; monitor out-of-bag (OOB) error to pick number of trees; tune `min_n`/`min_samples_leaf` and `maxnodes`/`max_leaf_nodes` to control overfitting.
- **Boosting (Adaboost / Gradient Boosting / Stochastic Gradient Boosting / XGBoost)**: fit a sequence of models, each focusing more on records the previous models got wrong (reweighted observations or fit to pseudo-residuals).
  - When to use: maximum predictive accuracy when you're willing to carefully tune hyperparameters and guard against overfitting.
  - How: `xgboost`/`XGBClassifier`; key knobs are `eta`/`learning_rate` (shrinkage), `subsample`/`colsample_bytree` (randomness like random forest), `max_depth` (shallow trees, default 3-6), `lambda`/`alpha` (L2/L1 regularization).
- **Regularization (Ridge/L2, Lasso/L1)**: penalize model complexity (coefficient magnitude or tree complexity) in the cost function to prevent overfitting.
  - When to use: whenever a flexible model (boosting, or regression with many predictors) shows a large gap between training and test error.
- **Hyperparameter tuning via cross-validation (grid search)**: systematically try combinations of hyperparameters (e.g., `tree_depth` × `learn_rate`) across K folds and pick the combination with lowest average out-of-sample error.

## Key Concepts
- **Distance metrics**: Euclidean (straight-line, √Σ(xi−ui)²), Manhattan (grid-like, Σ|xi−ui|), Mahalanobis (accounts for correlation via the covariance matrix) — choice matters for KNN and clustering.
- **Standardization (z-score)**: (x−mean)/SD; essential before KNN/PCA/clustering so large-scale variables don't dominate distance calculations; does NOT make data normally distributed.
- **One-hot encoding**: converting categorical variables into P binary dummy columns; fine for KNN/trees (unlike linear/logistic regression, which needs P−1 to avoid multicollinearity).
- **Bias-variance trade-off**: variance = error from sensitivity to the specific training sample (overfitting if too flexible); bias = error from an inadequate model structure (underfitting if too rigid/smoothed); K in KNN and tree depth are direct knobs on this trade-off.
- **Gini impurity / entropy**: measures of class heterogeneity within a partition used to choose tree splits; lower is purer/better.
- **Out-of-bag (OOB) error**: for bagged/random forest models, the error measured on the ~1/3 of data not included in each tree's bootstrap sample — a built-in, "free" cross-validation estimate.
- **Variable importance**: ranks predictors by either Gini-impurity decrease (fast, training-set-based, less reliable) or permutation/accuracy-decrease (slower, OOB-based, more reliable).
- **Hyperparameters**: settings chosen before training (not learned), e.g., K, tree depth, learning rate, regularization strength — critical to tune for ensemble methods, especially boosting.

## Mental Models
- Think of bagging/random forest as "the wisdom of crowds" (Galton's ox-weight-guessing county fair story): averaging many noisy, independent models cancels out individual errors better than any single model.
- Picture bagging as a reliable Honda Accord (little tuning needed, hard to badly overfit) and boosting as a powerful Porsche (best-in-class performance but needs careful hyperparameter tuning or it will overfit badly).
- Use KNN as a mental anchor for "local" learning: a prediction is only as good as the records nearby it in feature space — which is why scaling (standardization) and the choice of K directly control how "local" vs. "smoothed" the prediction is.
- View random forest's dual randomization (bootstrap rows + random predictor subset per split) as deliberately decorrelating the trees in the ensemble — if every tree used the exact same strong predictor first, averaging wouldn't reduce variance much.

## Anti-patterns
- **Running KNN or distance-based methods on unstandardized data**: a dollar-scale variable (e.g., `revol_bal`) will dominate the distance metric and make other informative ratio variables essentially irrelevant.
- **Growing a decision tree to full depth without pruning**: produces 100% training accuracy but is pure overfitting — the model has memorized noise, not learned signal.
- **Using XGBoost/boosting with default hyperparameters on noisy data**: the chapter's own example shows training error down to 10.9% but test error up to 35.9% — boosting overfits aggressively without regularization (`subsample`, `eta`, `lambda`/`alpha` tuning).
- **Trusting Gini-impurity-based variable importance as the final word**: it's computed on training data and can diverge substantially from the more reliable (but costlier) permutation/accuracy-decrease importance.
- **Treating random forest's apparent "black box" smoothness as noise-free**: predictions can still be noisy/overfit for individual records (e.g., high-creditworthiness borrowers still predicted to default due to unusual records).

## Code Examples
```r
knn_model <- workflow() %>%
  add_model(nearest_neighbor(mode = "classification", neighbors = 20)) %>%
  add_formula(outcome ~ payment_inc_ratio + dti) %>%
  fit(train_data)
```
- **What it demonstrates**: fitting KNN classification via tidymodels; K=20 nearest neighbors vote on loan default.

```r
loan_tree <- decision_tree(mode = "classification", engine = "rpart") %>%
  fit(outcome ~ borrower_score + payment_inc_ratio, data = loan3000)
```
- **What it demonstrates**: fitting a simple interpretable decision tree whose rules can be read directly (e.g., `borrower_score >= 0.575` → paid off).

```r
rf <- rand_forest(mode = "classification") %>%
  set_engine("ranger", seed = 123) %>%
  fit(outcome ~ borrower_score + payment_inc_ratio, data = loan3000)
# OOB prediction error (Brier s.): 0.2300053
```
- **What it demonstrates**: random forest fit with built-in OOB error estimate, no separate validation set needed.

```r
xgb_penalty <- xgboost(x = train_x, y = train_y,
  learning_rate = 0.1, subsample = 0.63, reg_lambda = 1000,
  objective = "binary:logistic", nrounds = 250)
# train_error 0.310, test_error 0.330 (vs. unregularized: train 0.109, test 0.359)
```
- **What it demonstrates**: L2 regularization (`reg_lambda`) closing the train/test error gap, directly fixing the overfitting shown in the unregularized XGBoost run.

## Reference Tables
| Method | Interpretability | Typical accuracy | Tuning burden | Key hyperparameters |
|---|---|---|---|---|
| KNN | Low (no explicit model) | Moderate | Low (just K, distance metric) | K, distance metric, standardization |
| Single decision tree | High (readable rules) | Moderate (prone to overfit) | Low-moderate | `max_depth`, `min_samples_leaf`, `cp`/`ccp_alpha` |
| Random forest | Low ("black box") | High, robust | Low (defaults work reasonably) | `min_n`/`min_samples_leaf`, `maxnodes`, # trees |
| Boosting (XGBoost) | Low ("black box") | Highest (if tuned well) | High (easy to overfit) | `eta`/`learning_rate`, `max_depth`, `subsample`, `lambda`/`alpha` |

| Distance metric | Formula | Best for |
|---|---|---|
| Euclidean | √Σ(xi−ui)² | General numeric data, straight-line similarity |
| Manhattan | Σ\|xi−ui\| | Grid/city-block travel-time-like similarity |
| Mahalanobis | Euclidean on PCA-transformed space | Correlated numeric predictors |

## Worked Example
LendingClub loan data (`loan200.csv`, `loan3000.csv`, `loan_data.csv.gz`): a KNN model with K=20 on `dti` and `payment_inc_ratio` for a new loan (dti=22.5, payment_inc_ratio=9) finds 9 defaults vs. 11 paid-off among its 20 neighbors, predicting "paid off" (but K=3 would flip the prediction to "default" — illustrating K's sensitivity). Standardizing 4 predictors (`payment_inc_ratio`, `dti`, `revol_bal`, `revol_util`) before KNN changes the 5 nearest neighbors entirely, since unstandardized `revol_bal` (in dollars) dominated distance before. A simple decision tree on `borrower_score` and `payment_inc_ratio` yields a readable rule set (root split at `borrower_score >= 0.575`). Random forest on the full 18-variable dataset achieves OOB error ≈ 0.212, with error stabilizing around 500 trees; variable importance via Gini-decrease vs. permutation/accuracy-decrease gives notably different rankings. XGBoost on the same full data overfits dramatically without regularization: 10.9% train error vs. 35.9% test error; adding `reg_lambda=1000` narrows this to 31.0%/33.0%. A 5-fold cross-validated grid search over `tree_depth` ∈ {3,6,12} and `learn_rate` ∈ {0.1,0.5,0.9} finds `tree_depth=3, learn_rate=0.5` gives the lowest CV error (33.34%), confirming shallower trees with moderate learning rate generalize best for this noisy data.

## Key Takeaways
1. Always standardize predictors (z-scores) before KNN, PCA, or clustering — unscaled large-magnitude variables silently dominate distance-based methods.
2. Tune K (KNN) and tree complexity (depth/min leaf size/cost-complexity) via cross-validation rather than defaults — this directly manages the bias-variance trade-off.
3. Random forests are a strong, relatively low-maintenance default for predictive accuracy; boosting (XGBoost) can beat them but requires deliberate overfitting defenses (`eta`, `subsample`, `lambda`/`alpha`, shallow `max_depth`).
4. Monitor training vs. test/validation error directly — a large gap is the clearest overfitting signal, especially for boosting.
5. Use OOB error as a free, built-in cross-validation estimate for bagging/random forest models.
6. Prefer accuracy/permutation-based variable importance over Gini-based importance when reliability matters more than compute cost.
7. KNN's value often lies less in its own predictive power and more as a "feature engine" that injects local structure (a derived "comps"-like feature) into a stronger second-stage model.

## Connects To
- **Ch 2 (Data and Sampling Distributions)**: bagging is literally "bootstrap aggregating," directly reusing the bootstrap resampling procedure introduced there.
- **Ch 5 (Classification)**: trees/forests/boosting are compared head-to-head against LDA and logistic regression's decision boundaries (Figure 5-9), showing trees capture nonlinear/interaction effects automatically.
- **Ch 4 (Regression and Prediction)**: ridge/lasso regularization introduced here parallels penalized regression as an alternative to stepwise selection for controlling model complexity.
