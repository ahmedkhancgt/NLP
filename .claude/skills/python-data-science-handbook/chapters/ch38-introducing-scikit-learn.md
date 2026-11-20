# Chapter 38: Introducing Scikit-Learn

## Core Idea
Scikit-Learn represents all data as a 2D features matrix `X` [n_samples, n_features] plus an optional 1D target array `y` [n_samples], and every model follows the same five-step Estimator API pattern: choose class → set hyperparameters → arrange data → `fit()` → `predict()`/`transform()`.

## Frameworks Introduced
- **Data representation**: `X` (features matrix, 2D, shape `[n_samples, n_features]`, samples=rows=observed objects, features=columns=quantitative measurements) and `y` (target array, 1D, length `n_samples`, the "dependent variable" to predict — absent for unsupervised learning).
- **The Estimator API five-step recipe**:
  1. Choose a model class (import the estimator, e.g. `from sklearn.linear_model import LinearRegression`).
  2. Choose hyperparameters by instantiating the class (e.g. `LinearRegression(fit_intercept=True)`) — instantiation only stores hyperparameters, no data is touched yet.
  3. Arrange data into `X` (2D) and `y` (1D, supervised only).
  4. Fit: `model.fit(X, y)` (supervised) or `model.fit(X)` (unsupervised) — learned parameters appear as attributes with a trailing underscore (e.g. `model.coef_`, `model.intercept_`).
  5. Apply to new data: `model.predict(X_new)` (supervised prediction, or unsupervised cluster assignment) or `model.transform(X_new)` (unsupervised dimensionality reduction/embedding).
- **Train/test split**: `sklearn.model_selection.train_test_split(X, y)` partitions data to evaluate a model on unseen data, the foundation of honest performance assessment.
- **Evaluation utilities**: `sklearn.metrics.accuracy_score(y_true, y_pred)`, `confusion_matrix(y_true, y_pred)` for classification diagnostics.

## Key Concepts
- **Scikit-Learn API design principles**: consistency (common interface across all models), inspection (parameters as public attributes), limited object hierarchy (only algorithms are classes; data stays in standard formats), composition (complex tasks built from simpler ones), sensible defaults.
- **Trailing-underscore convention**: any attribute ending in `_` was learned during `fit()` (e.g., `coef_`, `intercept_`), distinguishing it from hyperparameters set at instantiation.
- **Model class vs. model instance**: "a class of model is not the same as an instance of a model" — the class defines the algorithm, hyperparameters at instantiation configure it, and only `fit()` applies it to data.
- **Scikit-Learn doesn't explain parameter *uncertainty***: it focuses on prediction quality, not statistical inference about internal parameters — use `statsmodels` for that.
- **GaussianNB as a fast baseline classifier**: no hyperparameters to tune, useful as a first-pass benchmark before trying more sophisticated models.
- **Unsupervised "transform" vs. "predict"**: PCA uses `.transform()` to get the reduced-dimension representation; GaussianMixture uses `.predict()` to get cluster labels — both are unsupervised, but the appropriate output-retrieval method differs by algorithm.

## Mental Models
- Think of every Scikit-Learn workflow as the same five-step ritual regardless of algorithm — once you know this pattern, switching from linear regression to SVM to PCA to Gaussian mixtures is "just" changing which class you import and what hyperparameters you pass; the surrounding fit/predict choreography stays identical.
- A confusion matrix is "where did the model go wrong, and in what specific direction" — a single accuracy number hides the *pattern* of mistakes (e.g., "2s often misclassified as 1s or 8s"), which a confusion matrix reveals at a glance.

## Anti-patterns
- **Forgetting the features matrix must be 2D even for single-feature data**: a 1D array `x` must be reshaped via `x[:, np.newaxis]` before passing to `fit()` — a very common first-time error.
- **Evaluating a model only on its training data**: gives an overly optimistic (and uninformative) accuracy estimate — always evaluate on a held-out test set via `train_test_split`.
- **Relying on a single accuracy score as the complete picture of model quality**: masks which classes/categories are being confused with which others — use a confusion matrix (or similar) to diagnose specific failure patterns.
- **Expecting Scikit-Learn to provide statistical significance/uncertainty on fitted parameters (e.g., `coef_`)**: it is not designed for that — use `statsmodels` if parameter interpretation (not just prediction) is the goal.

## Code Examples
```python
import seaborn as sns
iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)      # features matrix, shape (150, 4)
y_iris = iris['species']                     # target array, shape (150,)

# Supervised: Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)   # 1 & 2: choose class + hyperparameters
X = x[:, np.newaxis]                              # 3: reshape 1D -> 2D features matrix
model.fit(X, y)                                     # 4: fit
model.coef_; model.intercept_                        # learned params (trailing underscore)
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)                            # 5: predict

# Supervised: Iris classification with Gaussian naive Bayes
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)             # ~0.974

# Unsupervised: PCA dimensionality reduction
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:, 0]; iris['PCA2'] = X_2D[:, 1]
sns.lmplot(x="PCA1", y="PCA2", hue='species', data=iris, fit_reg=False)

# Unsupervised: Gaussian mixture clustering
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)
iris['cluster'] = y_gmm

# Application: handwritten digits
from sklearn.datasets import load_digits
digits = load_digits()                      # digits.images.shape == (1797, 8, 8)
X = digits.data; y = digits.target             # X.shape (1797, 64), y.shape (1797,)

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)     # unsupervised dimensionality reduction

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
accuracy_score(ytest, y_model)                    # ~0.833

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='Blues')
plt.xlabel('predicted value'); plt.ylabel('true value')
```
- **What it demonstrates**: the full features-matrix/target-array data layout, the complete five-step recipe applied to both supervised (linear regression, Gaussian naive Bayes) and unsupervised (PCA, Gaussian mixture, Isomap) models, and a full real-world application (handwritten digit classification with accuracy + confusion matrix diagnosis).

## Reference Tables
| Data object | Shape | Convention name |
|---|---|---|
| Features matrix | `[n_samples, n_features]` | `X` |
| Target array | `[n_samples]` (or `[n_samples, n_targets]`) | `y` |

| Step | Action |
|---|---|
| 1 | Choose model class (import) |
| 2 | Choose hyperparameters (instantiate) |
| 3 | Arrange data (X, y) |
| 4 | `model.fit(X, y)` |
| 5 | `model.predict(X_new)` / `model.transform(X_new)` |

## Worked Example
Handwritten digits end-to-end: load `digits.data` (1797×64 features, flattened 8×8 pixel grids) and `digits.target`; reduce to 2D with `Isomap` for visualization (reveals zeros/ones cleanly separated, ones/fours overlapping — matching intuition about how people draw these digits); split train/test, fit `GaussianNB`, achieve ~83% accuracy; diagnose errors via `confusion_matrix` + Seaborn heatmap (reveals 2s often confused with 1s/8s) and by visualizing individual test images colored green (correct) / red (incorrect) — a complete, realistic ML diagnostic workflow in under 30 lines.

## Key Takeaways
1. Scikit-Learn data layout is always `X` (2D features matrix) + optional `y` (1D target array) — reshape 1D feature arrays via `x[:, np.newaxis]` when needed.
2. Every model follows the same five-step recipe: choose class → set hyperparameters at instantiation → arrange data → `fit()` → `predict()`/`transform()`.
3. Learned parameters (post-`fit()`) are exposed as attributes with a trailing underscore (`coef_`, `intercept_`).
4. `train_test_split` + `accuracy_score`/`confusion_matrix` is the standard supervised-learning evaluation pattern; a confusion matrix reveals specific misclassification patterns a single accuracy score hides.
5. Unsupervised methods use `.transform()` (dimensionality reduction, e.g., PCA, Isomap) or `.predict()` (cluster assignment, e.g., Gaussian mixture) to retrieve results — no `y` is ever passed to `.fit()`.
6. Scikit-Learn focuses on prediction, not parameter-uncertainty/statistical inference — use `statsmodels` for the latter.

## Connects To
- **Ch 37**: implements the classification/regression/clustering/dimensionality-reduction taxonomy introduced there with concrete code.
- **Ch 39**: model validation (train/test split here is the simple case; cross-validation generalizes it).
- **Ch 41/45/46/47/48**: Gaussian naive Bayes, PCA, Isomap, and Gaussian mixture models are each covered in full depth in their own chapters.
