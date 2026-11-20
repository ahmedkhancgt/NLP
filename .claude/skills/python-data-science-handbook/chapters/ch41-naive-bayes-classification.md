# Chapter 41: In Depth: Naive Bayes Classification

## Core Idea
Naive Bayes classifiers apply Bayes's theorem with a simplifying ("naive") generative-model assumption per class (e.g., Gaussian or multinomial), yielding extremely fast, interpretable classifiers that make excellent baselines, especially for high-dimensional, well-separated data.

## Frameworks Introduced
- **Bayesian classification via Bayes's theorem**: `P(L|features) = P(features|L)·P(L) / P(features)`; comparing the posterior ratio `P(L1|features)/P(L2|features)` between labels decides classification — requires a generative model `P(features|L)` for each class, which is the hard part made tractable by "naive" simplifying assumptions.
- **Gaussian naive Bayes** (`sklearn.naive_bayes.GaussianNB`): assumes each class's features are drawn from an axis-aligned (no covariance) Gaussian — fit by computing per-class, per-feature mean and standard deviation; produces (generally) quadratic decision boundaries; supports `predict_proba()` for probabilistic/uncertainty-aware output.
- **Multinomial naive Bayes** (`sklearn.naive_bayes.MultinomialNB`): assumes features are counts/count-rates drawn from a multinomial distribution — the standard choice for text classification on word-count or TF-IDF features.
  - Typical pipeline: `make_pipeline(TfidfVectorizer(), MultinomialNB())`.

## Key Concepts
- **Generative model**: specifies the hypothetical random process that generates the data for each class — training a naive Bayes classifier is essentially fitting one of these per label.
- **"Naive" assumption**: simplifying (often unrealistic) independence/distributional assumptions that make computing the generative model tractable, at the cost of some classification accuracy.
- **Decision boundary shape**: Gaussian naive Bayes produces curved (quadratic) boundaries, not straight lines, because the posterior ratio between two Gaussians is quadratic in the features.
- **`predict_proba()`**: returns posterior probability for each class — naive Bayes's Bayesian formulation naturally supports probabilistic/uncertainty-aware prediction, not just hard labels.
- **Confusion matrix diagnosis**: reveals specific category confusions (e.g., "religion" vs. "Christianity" topics) that a single accuracy score would hide.

## Mental Models
- Naive Bayes is "fast, cheap, interpretable, and surprisingly often good enough" — treat it as the default first baseline for any classification task, not a method to reach for only when nothing else is available.
- High dimensionality paradoxically *helps* simplistic classifiers: in high-dimensional space, points must be close in *every* dimension to be truly close, so well-defined clusters tend to be more separated as dimensionality grows — this is exactly the regime where naive Bayes's crude assumptions matter least.
- Naive Bayes classifiers trade model flexibility (bias) for speed/interpretability/few-or-no hyperparameters — appropriate when the task doesn't need (or can't support, given enough data) a more complex decision boundary.

## Anti-patterns
- **Skipping naive Bayes as "too simple to bother with"**: it is specifically valuable as a fast, interpretable *baseline* before investing in more complex models — if it performs well, you're done cheaply; if not, you have a performance floor to beat.
- **Expecting naive Bayes to perform well when its generative assumptions clearly don't hold and data is low-dimensional/not well-separated**: this is precisely where it tends to underperform more flexible models — recognize the regimes (high-dimensional, well-separated, actually-matching-assumptions) where it shines vs. where it doesn't.
- **Using raw word counts instead of TF-IDF for multinomial naive Bayes text classification without considering the weighting trade-off**: not strictly wrong, but TF-IDF (Ch 40) often improves results by downweighting ubiquitous words.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Gaussian naive Bayes
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)                     # posterior probabilities per class

# Multinomial naive Bayes: text classification
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names

categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names, cmap='Blues')
plt.xlabel('true label'); plt.ylabel('predicted label')

def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predict_category('sending a payload to the ISS')             # 'sci.space'
predict_category('discussing the existence of God')            # 'soc.religion.christian'
predict_category('determining the screen resolution')           # 'comp.graphics'
```
- **What it demonstrates**: Gaussian naive Bayes on synthetic 2D blob data (fit, probabilistic prediction, decision boundary visualization), and multinomial naive Bayes on real-world text (20 Newsgroups) with a TF-IDF pipeline, confusion-matrix diagnosis, and a reusable single-string prediction utility.

## Reference Tables
| Variant | Assumption | Best for |
|---|---|---|
| `GaussianNB` | Axis-aligned Gaussian per class | Continuous, roughly normal features |
| `MultinomialNB` | Multinomial distribution per class | Count/frequency features (e.g., text word counts, TF-IDF) |

| Naive Bayes advantage | Benefit |
|---|---|
| Fast training/prediction | Good for large datasets or rapid prototyping |
| Probabilistic output (`predict_proba`) | Built-in uncertainty estimates |
| Easily interpretable | Simple generative model per class |
| Few/no hyperparameters | Minimal tuning burden |

## Worked Example
20 Newsgroups text classification: a `TfidfVectorizer` + `MultinomialNB` pipeline trained on 4 categories (religion, Christianity, space, graphics) correctly classifies "sending a payload to the ISS" as `sci.space` and "determining the screen resolution" as `comp.graphics`, but the confusion matrix reveals systematic confusion between `talk.religion.misc` and `soc.religion.christian` — demonstrating that even a very simple word-frequency model achieves strong, interpretable results, with its failure modes directly explainable by topic overlap.

## Key Takeaways
1. Naive Bayes applies Bayes's theorem with a simplifying per-class generative-model assumption (Gaussian for continuous features, multinomial for counts) to make classification fast and tractable.
2. `GaussianNB` produces (generally) quadratic decision boundaries and supports `predict_proba()` for uncertainty-aware prediction.
3. `MultinomialNB` is the standard choice for text classification on word-count/TF-IDF features, typically combined via `make_pipeline(TfidfVectorizer(), MultinomialNB())`.
4. Naive Bayes classifiers are fast, interpretable, probabilistic, and nearly hyperparameter-free — ideal as a first baseline before trying more sophisticated models.
5. They perform especially well for high-dimensional and/or well-separated data, because points must be close in every dimension to be "close" overall, making clusters naturally more separated as dimensionality grows.
6. A confusion matrix reveals *which* categories get confused, not just an overall accuracy number — essential for diagnosing where a classifier's assumptions break down.

## Connects To
- **Ch 37/38**: directly implements the classification concept and Estimator API pattern introduced there.
- **Ch 40**: TF-IDF text vectorization is used directly in the multinomial naive Bayes example.
- **Ch 39**: confusion matrix / model evaluation tools reused from the validation chapter.
- **Ch 43/44**: contrasted later as more flexible (but slower/less interpretable) classification alternatives.
