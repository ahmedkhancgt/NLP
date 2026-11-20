# Chapter 43: In Depth: Support Vector Machines

## Core Idea
Support vector machines (SVMs) are discriminative classifiers that find the maximum-margin boundary between classes, determined entirely by a small set of "support vectors" near the margin, and become powerful nonlinear classifiers when combined with kernel transformations.

## Frameworks Introduced
- **Maximum-margin classification**: instead of any line that separates classes (infinitely many exist), choose the one that maximizes the margin (distance to the nearest point) — `SVC(kernel='linear', C=...)`.
  - When to use: classification tasks, especially high-dimensional ones, where a clean/robust discriminative boundary (not a generative model) is wanted.
- **Support vectors**: only the points touching/near the margin determine the fit (`model.support_vectors_`); points further away don't affect the boundary at all — explains SVM's insensitivity to distant training points and compact memory footprint.
- **Kernel SVM**: project data into higher dimensions via a kernel function (e.g., RBF, `kernel='rbf'`) to make nonlinearly separable data linearly separable in the projected space — computed implicitly via the "kernel trick" (never materializing the full high-dimensional representation), avoiding the cost of explicit per-point basis functions.
- **Soft margins**: the `C` hyperparameter controls margin hardness — large `C` = hard margin (no points allowed inside), small `C` = soft margin (some overlap tolerated) — essential for non-perfectly-separable real-world data.

## Key Concepts
- **Discriminative vs. generative classification**: SVM (discriminative) finds a boundary directly, vs. naive Bayes (generative, Ch 41) which models each class's distribution — different philosophies for the same classification goal.
- **`SVC` hyperparameters**: `kernel` (`'linear'`, `'rbf'`, etc.), `C` (margin hardness), `gamma` (RBF kernel width, controls how far each point's influence extends).
- **Kernel trick**: fitting on kernel-transformed data without ever building the full N-dimensional projection — makes kernel SVM computationally feasible even when "one basis function per point" would otherwise be `O(N)`-dimensional and expensive.
- **`model.decision_function(xy)`**: used to visualize/plot the learned decision boundary and margins.
- **Classification report / confusion matrix**: `classification_report` gives precision/recall/f1/support per class; `confusion_matrix` + heatmap visualizes specific misclassification patterns.

## Mental Models
- Think of SVM as "find the widest possible street between the two neighborhoods of points" — the street's centerline is the decision boundary, its edges are the margins, and only the houses right on the edge (support vectors) determine where the street runs.
- Kernel SVM extends Ch 42's "transform the input, not the model" basis-function idea to classification: instead of hand-picking a basis (as in Gaussian basis regression), the kernel trick implicitly considers a basis function at every point and lets the optimization sort out which matter (the support vectors).
- SVM's "only support vectors matter" property is a double-edged sword: it gives compactness/speed/robustness to irrelevant distant points, but also means the model provides no natural probabilistic output without extra (costly) calibration.

## Anti-patterns
- **Assuming any line separating two linearly-separable classes is "the" correct classifier**: multiple separators can perfectly discriminate training data yet generalize very differently — maximizing the margin (not just achieving zero training error) is what makes SVM's choice principled.
- **Using a linear kernel on clearly nonlinearly-separable data (e.g., concentric circles)**: a linear SVM will perform poorly — switch to `kernel='rbf'` (or another nonlinear kernel) to enable implicit high-dimensional separation.
- **Reaching for SVM as a first-choice method for large datasets**: SVM scaling is O(N²) to O(N³) in the number of samples, which can be prohibitive — "I generally only turn to SVMs once other simpler, faster... methods have been shown to be insufficient."
- **Expecting direct probabilistic output from SVM**: unlike naive Bayes, SVM decisions don't have a natural probabilistic interpretation — obtaining one (via the `probability` parameter) requires costly internal cross-validation.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
model = SVC(kernel='linear', C=1E10)       # hard margin
model.fit(X, y)
model.support_vectors_                       # the pivotal points

def plot_svc_decision_function(model, ax=None, plot_support=True):
    ax = ax or plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 30); y = np.linspace(*ylim, 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

# kernel SVM on nonlinear data
from sklearn.datasets import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)         # performs poorly: not linearly separable

r = np.exp(-(X ** 2).sum(1))                   # manual radial basis projection (illustrative)

clf = SVC(kernel='rbf', C=1E6)                # kernel trick does this implicitly
clf.fit(X, y)

# soft margins
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
for C in [10.0, 0.1]:
    model = SVC(kernel='linear', C=C).fit(X, y)   # C=10: harder margin, C=0.1: softer

# Face recognition pipeline
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=150, whiten=True, svd_solver='randomized', random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

from sklearn.model_selection import train_test_split, GridSearchCV
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)
param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, ytrain)
grid.best_params_                                # {'svc__C': 10, 'svc__gamma': 0.001}

model = grid.best_estimator_
yfit = model.predict(Xtest)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(ytest, yfit, target_names=faces.target_names))
mat = confusion_matrix(ytest, yfit)
import seaborn as sns
sns.heatmap(mat.T, square=True, annot=True, fmt='d')
```
- **What it demonstrates**: maximum-margin linear SVM with support vector visualization, demonstration that new distant points don't change the fit, kernel SVM solving a nonlinearly separable problem, soft-margin tuning via C, and a complete real-world pipeline (PCA + kernel SVM + grid search) for face recognition with full evaluation (classification report, confusion matrix).

## Reference Tables
| Hyperparameter | Controls | Large value | Small value |
|---|---|---|---|
| `C` | Margin hardness | Hard margin, no overlap tolerated | Soft margin, overlap tolerated |
| `gamma` (RBF kernel) | Kernel width / reach of each point's influence | Tight/local influence | Broad/smooth influence |

| SVM strength | SVM weakness |
|---|---|
| Compact (few support vectors), fast prediction | O(N²)-O(N³) training cost, poor at scale |
| Works well in high dimensions, even N_features > N_samples | Needs careful C (and gamma) tuning via cross-validation |
| Versatile via kernel methods | No native probabilistic output (costly to add) |

## Worked Example
Face recognition on Labeled Faces in the Wild: PCA reduces 62×47-pixel (~3000-feature) images to 150 components, feeding a `kernel='rbf'` SVC; `GridSearchCV` over `C` and `gamma` finds `C=10, gamma=0.001` as optimal; the tuned model achieves 85% overall accuracy, with `classification_report` showing per-person precision/recall (e.g., Junichiro Koizumi: 0.92/1.00, Hugo Chavez: 0.93/0.70) and a confusion matrix revealing specific person-to-person confusions — illustrating the full real-world "feature reduction + kernel SVM + hyperparameter search + evaluation" pipeline.

## Key Takeaways
1. SVM is a discriminative classifier that maximizes the margin between classes, with the fit determined entirely by a small set of support vectors.
2. Kernel SVM (e.g., `kernel='rbf'`) implicitly projects data into higher dimensions via the kernel trick, enabling nonlinear decision boundaries without the cost of explicit per-point basis functions.
3. `C` controls margin hardness/softness, tuned via cross-validation for non-perfectly-separable real data; `gamma` controls RBF kernel reach.
4. SVM strengths: compact, fast prediction, effective in high dimensions, versatile via kernels. Weaknesses: poor scaling (O(N²)-O(N³)), tuning-sensitive, no native probabilistic output.
5. SVM is best reserved for cases where simpler/faster methods have proven insufficient, given its computational cost at scale.
6. Combining PCA (dimensionality reduction) with kernel SVM in a pipeline is a standard, effective pattern for image-based classification tasks.

## Connects To
- **Ch 41**: explicitly contrasted as discriminative vs. generative classification approaches.
- **Ch 42**: kernel SVM is framed as a direct extension of basis-function regression's "project into higher dimensions" idea.
- **Ch 45**: PCA is used here for feature reduction before SVM classification, previewing the dedicated PCA chapter.
- **Ch 39**: `GridSearchCV` hyperparameter tuning directly reuses the validation chapter's tools.
