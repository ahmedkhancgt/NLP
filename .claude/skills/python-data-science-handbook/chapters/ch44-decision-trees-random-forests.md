# Chapter 44: In Depth: Decision Trees and Random Forests

## Core Idea
Decision trees recursively split data along axis-aligned feature cutoffs (intuitive but prone to overfitting), while random forests — ensembles of many randomized/bagged decision trees whose predictions are averaged — reduce overfitting and often outperform any single constituent tree, demonstrating that "the sum can be greater than the parts."

## Frameworks Introduced
- **Decision trees** (`DecisionTreeClassifier`/implicit in regression too): recursively split data via axis-aligned cutoffs within one feature at a time, assigning the majority-vote label to each resulting region.
  - When to use: as a base learner for ensembles, or for interpretable rule-based classification — but prone to overfitting on its own.
- **Bagging** (`BaggingClassifier(base_estimator, n_estimators=, max_samples=)`): fits many copies of an overfitting estimator on random data subsets, averaging their predictions to cancel out idiosyncratic overfitting.
- **Random forest** (`RandomForestClassifier`/`RandomForestRegressor`): an optimized, automatically-randomized ensemble of decision trees — randomization happens both via data subsampling and via randomized feature selection at each split, giving a smoother, more generalizable model than any single tree or than manual bagging.
  - Classification: `RandomForestClassifier(n_estimators=)`.
  - Regression: `RandomForestRegressor(n_estimators=)` — same API, continuous target.

## Key Concepts
- **Ensemble method**: aggregating predictions from multiple simpler estimators, often yielding better accuracy than any individual estimator.
- **Axis-aligned splits**: each decision-tree node splits data using a cutoff on exactly one feature — the mechanism behind a tree's "20 questions" structure.
- **Overfitting in decision trees**: deep trees produce "strangely shaped classification regions" reflecting noise/sampling idiosyncrasies rather than true structure; different random subsets of the same data can yield very different deep trees — a clear symptom of high variance.
- **Where trees disagree = where classification is uncertain**: regions where independently-trained trees give different answers are exactly the regions that benefit most from aggregating multiple trees' information.
- **`predict_proba`**: random forests provide probabilistic classification via majority-vote fraction across trees — a benefit single deterministic trees lack directly.
- **Nonparametric flexibility**: random forests make no assumption about the functional form of the relationship (e.g., can fit a multi-period oscillating signal in regression without specifying a multi-period model).

## Mental Models
- A single decision tree is "a very confident but overfit opinion"; a random forest is "an averaged consensus of many different, independently overfit opinions" — each tree's idiosyncratic noise tends to cancel out in the average, leaving the shared true signal.
- Randomization is the key ingredient that makes bagging/forests work: if every tree saw the same data, averaging would do nothing — the randomness (in data subsample and/or feature-selection-per-split) is what makes the trees' errors (mostly) independent and thus cancelable by averaging.
- Random forests trade interpretability for predictive accuracy and flexibility — a single decision tree's splits can be read as explicit if/then rules, but an ensemble of hundreds of randomized trees cannot be "read" the same way.

## Anti-patterns
- **Using a single deep decision tree as a final model without acknowledging overfitting risk**: deep trees reliably overfit noise/sampling idiosyncrasies — either limit depth carefully (with cross-validation) or, more robustly, use a random forest instead.
- **Manually implementing bagging with ad-hoc data subsampling when `RandomForestClassifier`/`Regressor` is available**: the built-in random forest estimators already implement optimized randomization (data + feature-selection) automatically — prefer them over manual `BaggingClassifier` setups for standard use cases.
- **Expecting a random forest to yield an interpretable model of feature-effect relationships**: unlike linear regression coefficients (Ch 42), random forests "may not be the best choice" when the goal is to interpret/explain the underlying relationship rather than just predict well.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)       # single tree, prone to overfitting

# bagging: manual ensemble
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1)
bag.fit(X, y)

# random forest: optimized automatic ensemble
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# random forest regression
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)
def model_fn(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))
    return slow_oscillation + fast_oscillation + noise
y = model_fn(x)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)                        # fits multi-period signal, nonparametrically

# Digit classification with random forest
from sklearn.datasets import load_digits
digits = load_digits()
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

from sklearn import metrics
print(metrics.classification_report(ypred, ytest))    # ~98% accuracy

from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('true label'); plt.ylabel('predicted label')
```
- **What it demonstrates**: a single overfitting decision tree, manual bagging, the automatic random forest classifier, random forest regression fitting a nonparametric multi-period signal, and a full real-world application (digit classification) achieving high accuracy with minimal tuning.

## Reference Tables
| Method | What it combines | Randomization source |
|---|---|---|
| Decision tree | — (single estimator) | None |
| Bagging | Many copies of a base estimator | Random data subsamples |
| Random forest | Many decision trees | Random data subsamples + random feature selection per split |

| Random forest advantage | Random forest disadvantage |
|---|---|
| Fast train/predict, parallelizable | Not easily interpretable |
| Probabilistic output via `predict_proba` | — |
| Nonparametric, flexible, resists underfitting | — |

## Worked Example
Digit classification (0-9, 8x8 pixel images): splitting into train/test, fitting `RandomForestClassifier(n_estimators=1000)` directly on raw pixel data (no PCA/feature engineering needed) achieves ~98% overall accuracy with per-class precision/recall mostly ≥0.94, and a confusion matrix shows very few systematic misclassifications — demonstrating that an untuned random forest can be a remarkably strong baseline even on raw high-dimensional pixel data.

## Key Takeaways
1. Single decision trees are intuitive (axis-aligned "20 questions" splits) but prone to high-variance overfitting, especially as depth increases.
2. Bagging (averaging many overfit estimators trained on random subsets) reduces overfitting by cancelling idiosyncratic errors — random forests are the optimized, automatic version of this applied to decision trees.
3. Random forests randomize both the data subsample and the feature selection at each split, making individual trees' errors more independent and thus more effectively cancelable by averaging.
4. Random forests work for both classification (`RandomForestClassifier`) and regression (`RandomForestRegressor`), with identical API patterns, and provide probabilistic output via `predict_proba`.
5. Key trade-off: random forests are fast, flexible, and parallelizable, but sacrifice the interpretability a single decision tree (or linear model) can offer.
6. A simple, untuned random forest is often a strong baseline even on raw, high-dimensional data (e.g., pixel values) without extensive feature engineering.

## Connects To
- **Ch 41/43**: contrasted as a third major classification paradigm (ensemble/nonparametric) alongside generative (naive Bayes) and discriminative (SVM) methods.
- **Ch 38**: the digits dataset classification task is directly reused from the Scikit-Learn introduction chapter, now solved with a different algorithm for comparison.
- **Ch 39**: overfitting/bias-variance concepts from the validation chapter directly explain why single deep trees overfit and why ensembling helps.
