# Chapter 40: Feature Engineering

## Core Idea
Feature engineering ("vectorization") converts real-world, non-numerical, or incomplete data — categorical labels, text, derived relationships, missing values — into well-behaved numerical feature matrices that Scikit-Learn models can consume, and `Pipeline`/`make_pipeline` chains these transformation steps into one reusable estimator-like object.

## Frameworks Introduced
- **One-hot encoding for categorical data**: `DictVectorizer(sparse=, dtype=)` converts a list of dicts (mixed categorical/numerical fields) into expanded binary indicator columns per category, avoiding false ordinal relationships from naive integer mapping.
- **Text vectorization**: `CountVectorizer` (raw word counts per document) vs. `TfidfVectorizer` (term frequency–inverse document frequency, downweighting very common words) — both via `.fit_transform(text_list)`, inspectable with `.get_feature_names_out()`.
- **Derived/polynomial features**: `PolynomialFeatures(degree=, include_bias=)` expands a feature matrix with powers/products of existing features, enabling linear models to fit nonlinear relationships by transforming the *input*, not the model.
- **Missing-data imputation**: `SimpleImputer(strategy='mean'/'median'/'most_frequent')` replaces NaN entries with a column-wise statistic before fitting, as a baseline (vs. more sophisticated matrix-completion approaches).
- **`make_pipeline(step1, step2, ..., estimator)`**: chains any sequence of transformers + a final estimator into one object supporting `.fit()`/`.predict()` exactly like a single estimator — automates multi-step preprocessing.

## Key Concepts
- **Why not naive integer-encode categories**: Scikit-Learn models assume numerical features reflect algebraic quantities, so an arbitrary integer mapping (`'Queen Anne': 1, 'Fremont': 2, ...`) would wrongly imply ordering/arithmetic relationships between categories.
- **Sparse representation**: one-hot/word-count encodings are mostly zeros; `sparse=True` (DictVectorizer) or the default sparse-matrix output (CountVectorizer/TfidfVectorizer) stores only nonzero entries efficiently — most Scikit-Learn estimators accept sparse input directly.
- **TF-IDF**: weights word counts by how distinctively they appear across documents, reducing the influence of ubiquitous/uninformative words relative to raw counts.
- **"Transform the input, not the model"**: fitting `LinearRegression` on `PolynomialFeatures`-expanded input achieves a nonlinear fit using an inherently linear algorithm — this principle generalizes to basis-function regression (Ch 42) and kernel methods (Ch 43).
- **Imputation precedes modeling**: missing values (`NaN`) must be filled before most estimators can `fit()`; `SimpleImputer`'s mean/median/most-frequent strategies are simple baselines, not universally optimal.
- **Pipeline composability**: `make_pipeline` steps run in order (transform → transform → ... → fit/predict), and the whole chain is itself a valid "estimator" for cross-validation, grid search, etc.

## Mental Models
- Feature engineering is "the translation layer" between messy real-world data and the clean `[n_samples, n_features]` numeric matrix every Scikit-Learn algorithm expects — nearly all real ML work happens here, not in algorithm selection.
- One-hot encoding trades dataset width (more columns) for correctness (no false ordinal assumptions) — an explicit, principled cost/benefit choice, mitigated by sparse storage.
- A `Pipeline` is "a recipe object": it doesn't run anything at construction time, only when `.fit()`/`.predict()` is called — mirroring the model-instantiation-vs-fitting distinction from Ch 38/39.

## Anti-patterns
- **Naively mapping categorical labels to arbitrary integers**: implies false ordinal/arithmetic relationships between unrelated categories — use one-hot encoding (`DictVectorizer`, `OneHotEncoder`) instead.
- **Using raw word counts for text classification without considering TF-IDF**: overweights frequently occurring but often uninformative words — TF-IDF is a cheap, standard improvement.
- **Applying transformation steps by hand, one at a time, especially across multiple models/datasets**: tedious and error-prone — use `make_pipeline` to chain and reuse the full preprocessing+modeling sequence.
- **Evaluating a pipeline's `.predict()` on the same data it was `.fit()` on and treating the result as meaningful**: this chapter's own imputation example does exactly this for demonstration simplicity, but the author explicitly flags it refers back to Ch 39's validation lessons — always validate properly in real use.

## Code Examples
```python
# One-hot encoding of categorical + numerical mixed data
data = [{'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
        {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
        {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
        {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)                 # expands neighborhood into 3 binary columns
vec.get_feature_names_out()

vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)                   # efficient sparse storage

# Text features
sample = ['problem of evil', 'evil queen', 'horizon problem']
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)
import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

# Derived (polynomial) features
import numpy as np
x = np.array([1, 2, 3, 4, 5]); y = np.array([4, 2, 1, 3, 7])
from sklearn.linear_model import LinearRegression
# straight-line fit is poor

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(x[:, np.newaxis])   # columns: x, x^2, x^3
model = LinearRegression().fit(X2, y)        # linear regression on expanded features = good fit

# Missing-data imputation
from numpy import nan
X = np.array([[nan, 0, 3], [3, 7, 9], [3, 5, 2], [4, nan, 6], [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)                     # NaNs replaced by column means
model = LinearRegression().fit(X2, y)
model.predict(X2)

# Full feature pipeline
from sklearn.pipeline import make_pipeline
model = make_pipeline(SimpleImputer(strategy='mean'),
                       PolynomialFeatures(degree=2),
                       LinearRegression())
model.fit(X, y)
model.predict(X)
```
- **What it demonstrates**: one-hot encoding for mixed categorical/numeric data (dense and sparse), word-count and TF-IDF text vectorization, polynomial feature derivation that turns a linear model into a nonlinear fitter, missing-value imputation, and chaining all preprocessing + modeling steps into one `Pipeline`.

## Reference Tables
| Feature type | Tool |
|---|---|
| Categorical | `DictVectorizer`, `OneHotEncoder`, `FeatureHasher` |
| Text (word counts) | `CountVectorizer` |
| Text (weighted) | `TfidfVectorizer` |
| Images | Pixel values (simple), or Scikit-Image feature extraction |
| Derived/nonlinear | `PolynomialFeatures` |
| Missing values | `SimpleImputer` (mean/median/most_frequent) |
| Multi-step chains | `make_pipeline` |

## Worked Example
Turning a poor straight-line fit into a good one: data `(x, y)` where `y` clearly isn't linear in `x` fits `LinearRegression` poorly directly, but `PolynomialFeatures(degree=3).fit_transform(x)` expands each `x` into `[x, x², x³]`, and fitting `LinearRegression` on this expanded matrix produces a much closer fit — the model itself stays linear; only the *input representation* changed, the core principle behind basis-function regression (Ch 42) and kernel methods (Ch 43).

## Key Takeaways
1. Real-world data is rarely a clean numeric `[n_samples, n_features]` matrix — feature engineering ("vectorization") is the translation step, and often the most important practical work in an ML pipeline.
2. Categorical data should be one-hot encoded (`DictVectorizer`/`OneHotEncoder`), never naively integer-mapped, to avoid implying false ordinal relationships.
3. Text can be vectorized via word counts (`CountVectorizer`) or TF-IDF weighting (`TfidfVectorizer`), the latter correcting for over-weighting of common words.
4. Derived (e.g., polynomial) features let a linear model fit nonlinear relationships by transforming the input, not the algorithm — a principle generalized later by basis functions and kernel methods.
5. Missing values must be imputed (`SimpleImputer`, baseline mean/median/most-frequent) before most models can be fit.
6. `make_pipeline` chains any sequence of transformers plus a final estimator into one reusable, fit/predict-compatible object, essential for clean, repeatable multi-step preprocessing.

## Connects To
- **Ch 16**: builds directly on Pandas' missing-data (`NaN`) handling conventions.
- **Ch 39**: `PolynomialFeatures` + `make_pipeline` were previewed there for validation/learning curves; this chapter explains the mechanism in full.
- **Ch 41**: TF-IDF text features are used directly in the naive Bayes text-classification example.
- **Ch 42/43**: derived features and kernel methods generalize the "transform the input, not the model" principle introduced here.
- **Ch 50**: image feature extraction via Scikit-Image is applied in the face-detection pipeline chapter.
