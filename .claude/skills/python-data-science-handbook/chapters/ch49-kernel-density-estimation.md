# Chapter 49: In Depth: Kernel Density Estimation

## Core Idea
Kernel density estimation (KDE) places one smooth kernel (e.g., Gaussian) per data point and sums them, producing a nonparametric density estimate that avoids the bin-alignment artifacts of histograms; it can be wrapped into a full "not-so-naive" Bayesian generative classifier by replacing naive Bayes's crude per-class Gaussian with a flexible per-class KDE model.

## Frameworks Introduced
- **Histogram as "stack of blocks"**: a block per point, stacked within bin-aligned columns — informative but highly sensitive to bin edge placement/offset, which can produce qualitatively different (even contradictory) visual impressions from the same data.
- **KDE as "unaligned blocks, summed everywhere"**: instead of forcing blocks into bins, place a kernel (block, or smoother Gaussian) centered exactly at each point and sum contributions continuously across the x-axis — eliminates bin-alignment artifacts, trading them for a bandwidth choice.
  - `sklearn.neighbors.KernelDensity(bandwidth=, kernel=)`: `.fit(X)` + `.score_samples(X_grid)` (returns log-density) — supports 6 kernels and ~dozens of distance metrics, works in multiple dimensions, uses a tree-based algorithm with `atol`/`rtol` speed-accuracy trade-off parameters.
- **Bandwidth selection via cross-validation**: wrap `KernelDensity` in `GridSearchCV` (with e.g. `LeaveOneOut` for small datasets) over a `bandwidth` grid, selecting the value maximizing log-likelihood — the standard machine-learning-style alternative to classical statistical bandwidth-selection rules.
- **"Not-so-naive" Bayes custom estimator**: a complete `KDEClassifier(BaseEstimator, ClassifierMixin)` implementing Bayesian generative classification where each class's generative model is a KDE fit (rather than naive Bayes's simplistic per-class Gaussian) — demonstrates the full custom-Scikit-Learn-estimator pattern (`__init__`, `fit`, `predict_proba`, `predict`).

## Key Concepts
- **Bandwidth as the KDE bias-variance knob**: too narrow → high variance (overfitting, noisy/spiky estimate); too wide → high bias (underfitting, structure washed out) — directly analogous to model-complexity trade-offs in Ch 39.
- **Normalization**: both histograms (`density=True`) and KDE outputs are normalized so the area under the curve equals 1, making them valid probability densities.
- **Custom estimator conventions**: inherit `BaseEstimator` (provides clone/copy logic for cross-validation) + `ClassifierMixin` (default `.score()`); `__init__` must *only* assign passed arguments to `self` (no `*args`/`**kwargs`, no other logic) so cloning works correctly; `.fit()` must return `self` to support method chaining; learned attributes get a trailing underscore (`self.classes_`, `self.models_`, `self.logpriors_`).
- **Generative classification recipe** (general, reused from Ch 41 but now with KDE instead of Gaussian): split training data by label → fit a density model per class (`P(x|y)`) → compute class priors `P(y)` from class frequencies → for new `x`, posterior `P(y|x) ∝ P(x|y)·P(y)`, predict the class maximizing this.
- **KDE classifier interpretability advantage**: unlike SVM/random forest, a generative classifier gives "not only... a probabilistic classification, but a full model of the distribution of points we are comparing it to" — more transparent reasoning about *why* a classification was made.

## Mental Models
- KDE is literally "mixture-of-Gaussians (Ch 48) taken to its logical extreme": instead of a small number of fitted Gaussian components, place one Gaussian (or other kernel) *per data point* — a fully nonparametric density estimate.
- Histogram bin-placement sensitivity is a concrete, visceral illustration of why "aligning representation to arbitrary grid boundaries" is fragile — KDE's "align to the data itself" approach is the direct structural fix.
- Writing a custom Scikit-Learn estimator means thinking of `__init__` purely as "remember my configuration," `fit` as "learn from data and store results with trailing underscores," and `predict`/`predict_proba` as "use what I learned to answer questions about new data" — a disciplined separation that makes the object automatically compatible with grid search, pipelines, and cross-validation.

## Anti-patterns
- **Trusting a single histogram's "shape" (e.g., bimodal vs. unimodal) without checking sensitivity to bin placement**: the same small dataset can look qualitatively different depending on arbitrary bin offset — use KDE (bin-placement-independent) for a more robust visual density estimate, especially with limited data.
- **Choosing KDE bandwidth by eye or via classical statistical rules of thumb without cross-validating on your actual task**: in machine learning contexts, empirical cross-validation (via `GridSearchCV`) is the more robust, task-appropriate approach.
- **Writing a custom Scikit-Learn estimator's `__init__` with extra logic, `*args`/`**kwargs`, or without storing every argument verbatim to `self`**: breaks the cloning machinery `BaseEstimator` relies on for cross-validation/grid search — keep `__init__` to pure, explicit attribute assignment.
- **Assuming naive Bayes's crude Gaussian-per-class assumption is always "good enough"**: on the digits dataset, naive Bayes achieves only ~81% cross-validation accuracy vs. ~96%+ for a KDE-based generative classifier with tuned bandwidth — when naive Bayes underperforms, a less "naive" generative model (KDE, or GMM) is a natural next step.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x
x = make_data(1000)

hist = plt.hist(x, bins=30, density=True)     # normalized histogram, area = 1

# bin-placement sensitivity with small data
x = make_data(20)
bins = np.linspace(-5, 10, 10)
for offset in [0.0, 0.6]:
    plt.hist(x, bins=bins + offset, density=True)   # qualitatively different shapes!

# manual "block" KDE
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)      # top-hat kernel, summed per point

# manual Gaussian KDE
from scipy.stats import norm
density = sum(norm(xi).pdf(x_d) for xi in x)             # smooth Gaussian kernel

# Scikit-Learn KDE
from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])
logprob = kde.score_samples(x_d[:, None])                # log-density
plt.fill_between(x_d, np.exp(logprob), alpha=0.5)

# bandwidth selection via cross-validation
from sklearn.model_selection import GridSearchCV, LeaveOneOut
bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
grid.fit(x[:, None])
grid.best_params_                                          # ~1.12

# Custom KDE-based Bayesian classifier
from sklearn.base import BaseEstimator, ClassifierMixin

class KDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi)
                         for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self

    def predict_proba(self, X):
        logprobs = np.vstack([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

# using the custom estimator with grid search on digits data
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
digits = load_digits()
grid = GridSearchCV(KDEClassifier(), {'bandwidth': np.logspace(0, 2, 100)})
grid.fit(digits.data, digits.target)
grid.best_params_

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(), digits.data, digits.target).mean()    # ~0.807, worse than KDE
```
- **What it demonstrates**: the histogram-to-KDE motivation (bin sensitivity → block-per-point → smooth Gaussian-per-point), Scikit-Learn's `KernelDensity` estimator, cross-validated bandwidth selection, and a complete custom Scikit-Learn estimator (`KDEClassifier`) that outperforms naive Bayes on the digits dataset.

## Reference Tables
| Density estimation step | Technique |
|---|---|
| Discrete bins | Histogram (bin-placement sensitive) |
| Block per point, summed | Manual top-hat KDE |
| Smooth kernel per point, summed | KDE (Gaussian or other kernel) |
| Multivariate, tunable | `sklearn.neighbors.KernelDensity` |

| Custom estimator requirement | Reason |
|---|---|
| `__init__` only assigns args to `self` | Needed for `BaseEstimator`'s cloning logic |
| No `*args`/`**kwargs` in `__init__` | Cross-validation routines can't handle them |
| `fit()` returns `self` | Enables method chaining (`model.fit(X,y).predict(X)`) |
| Learned attrs end in `_` | Scikit-Learn convention distinguishing fit results from hyperparameters |

## Worked Example
"Not-so-naive" Bayes on handwritten digits: a `KDEClassifier` (custom estimator fitting a `KernelDensity` model per digit class) tuned via `GridSearchCV` over bandwidth achieves over 96% cross-validation accuracy, compared to `GaussianNB`'s ~80.7% on the same data — demonstrating that relaxing naive Bayes's crude per-class-Gaussian assumption to a flexible KDE-based generative model substantially improves classification while preserving the interpretability benefit (a full per-class distribution model) that discriminative methods like SVM/random forest lack.

## Key Takeaways
1. KDE places a smooth kernel at each data point and sums contributions, producing a nonparametric density estimate immune to the bin-alignment artifacts that plague histograms.
2. Bandwidth is KDE's central bias-variance trade-off parameter — too narrow overfits (high variance), too wide underfits (high bias); cross-validation (`GridSearchCV` + `LeaveOneOut` for small data) is the standard ML-style selection method.
3. `sklearn.neighbors.KernelDensity` supports multiple kernels/metrics and multidimensional data, with `.score_samples()` returning log-density.
4. KDE can replace naive Bayes's crude per-class Gaussian assumption with a flexible per-class density model, yielding a "not-so-naive" Bayesian generative classifier that can substantially outperform naive Bayes.
5. Writing a custom Scikit-Learn estimator requires strict conventions: `BaseEstimator`/`ClassifierMixin` inheritance, explicit-only `__init__`, `fit()` returning `self`, and trailing-underscore learned attributes — following these makes the estimator fully compatible with grid search, pipelines, and cross-validation.
6. Generative classifiers (naive Bayes, KDE-based) trade some raw accuracy (vs. SVM/random forest in some cases) for interpretability — you get a full distributional model explaining each classification, not just a label.

## Connects To
- **Ch 41**: directly extends the generative-classification recipe, replacing naive Bayes's Gaussian assumption with KDE.
- **Ch 28/36**: KDE visualization was previewed in the density/contour plotting chapter and Seaborn's `kdeplot`.
- **Ch 39**: bandwidth-as-bias-variance-knob and cross-validation-based tuning directly reuse the validation chapter's framework.
- **Ch 48**: explicitly framed as "mixture-of-Gaussians taken to its logical extreme" — one Gaussian per point instead of a small fitted number.
