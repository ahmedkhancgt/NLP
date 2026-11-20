# Chapter 45: In Depth: Principal Component Analysis

## Core Idea
PCA finds the data's principal axes (directions of maximal variance) and uses them for dimensionality reduction, visualization, noise filtering, and feature selection — representing high-dimensional data via a small number of "optimal" basis functions rather than the raw (e.g., pixel) basis.

## Frameworks Introduced
- **`sklearn.decomposition.PCA(n_components=)`**: fits an unsupervised model finding principal axes (`.components_`) and their variance (`.explained_variance_`); `.transform()`/`.inverse_transform()` project to/from the reduced space.
  - When to use: visualizing high-dimensional data in 2D/3D, reducing dimensionality while preserving variance, filtering noise, or selecting compact features before another model.
- **Dimensionality reduction**: zeroing out low-variance principal components; `n_components=k` keeps only the top-k axes.
- **Noise filtering via PCA**: `PCA(variance_fraction)` (e.g., `PCA(0.50)`) automatically picks enough components to preserve a target variance fraction; reconstructing from only the top components preferentially keeps signal (high variance) and discards noise (assumed lower variance, spread across many components).
- **Choosing the number of components**: plot cumulative `explained_variance_ratio_` vs. component count to see how much "information" (variance) is captured as components are added.

## Key Concepts
- **Principal axes**: directions in feature space capturing the most variance, found via (implicitly) an affine transformation (translation + rotation + uniform scaling) of the data.
- **Principal components**: the projections of each data point onto the principal axes — the new, reduced coordinates.
- **`explained_variance_`/`explained_variance_ratio_`**: how much of the total variance each principal axis accounts for (absolute vs. normalized fraction).
- **PCA as optimal basis selection**: instead of the native "pixel basis" (each feature/pixel is its own basis vector), PCA finds basis vectors ("eigenfaces" for face data) such that a weighted sum of just a few of them reconstructs most of the image/data — dramatically more efficient than truncating the native basis (e.g., keeping only the first 8 raw pixels loses far more information than keeping the first 8 PCA components).
- **Affine transformation**: PCA's data-to-principal-axes mapping is a composition of translation, rotation, and uniform scaling — not an arbitrary nonlinear warp.
- **Eigenfaces**: the PCA components of a face-image dataset, visualizable as images themselves; early components often correspond to broad factors (e.g., lighting angle), later ones to finer features (eyes, nose, lips).

## Mental Models
- PCA answers "what is the smallest number of new, custom-built directions I need to describe most of what varies in this data?" — a fundamentally different question from "which original features matter" (that's feature selection; PCA does feature *construction*).
- Truncating a dataset's native basis (e.g., keeping only some raw pixels) and truncating its PCA basis (keeping only the top principal components) are both "throw away some representation," but PCA's basis is *optimized* to concentrate the most information in the fewest components — explaining why 8 PCA components reconstruct a digit far better than 8 raw pixels.
- Noise filtering via PCA works because noise, by assumption, is spread diffusely across many low-variance directions, while true signal concentrates in a few high-variance directions — reconstructing from only the top components is implicitly a signal/noise separation.

## Anti-patterns
- **Discarding dimensions by simply dropping raw features (e.g., "use only the first 8 pixels")**: loses disproportionate information compared to discarding low-variance PCA components, since raw features aren't ordered by informativeness — use PCA's variance-ranked basis instead.
- **Choosing `n_components` arbitrarily without inspecting the cumulative explained variance curve**: risks either discarding too much signal or keeping unnecessary dimensions — always plot/check `cumsum(explained_variance_ratio_)` first.
- **Applying PCA to data with significant outliers without caution**: PCA is "highly affected by outliers" since variance is sensitive to extreme values — consider robust PCA variants (e.g., `SparsePCA`) when outliers are a concern.
- **Expecting PCA to work well on every high-dimensional dataset**: "PCA is not useful for every high-dimensional dataset" — it's a good first exploratory step, not a universal solution.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
pca.components_                 # principal axes (directions)
pca.explained_variance_          # variance along each axis
pca.mean_                          # center of the data

# dimensionality reduction
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)             # (200, 2) -> (200, 1)
X_new = pca.inverse_transform(X_pca)  # project back for visualization

# PCA for visualization: digits
from sklearn.datasets import load_digits
digits = load_digits()                  # (1797, 64)
pca = PCA(2)
projected = pca.fit_transform(digits.data)   # (1797, 2)

# choosing number of components
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components'); plt.ylabel('cumulative explained variance')

# PCA as noise filtering
pca = PCA(0.50).fit(noisy)       # preserve 50% of variance
pca.n_components_                  # 12 (out of 64)
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)   # denoised reconstruction

# Eigenfaces
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
pca = PCA(150, svd_solver='randomized', random_state=42)
pca.fit(faces.data)
# pca.components_[i].reshape(62, 47) -> visualize as "eigenface" image

plt.plot(np.cumsum(pca.explained_variance_ratio_))   # ~90% variance with 150 components

components = pca.transform(faces.data)
projected = pca.inverse_transform(components)   # 150-dim reconstruction, visually recognizable
```
- **What it demonstrates**: fitting/visualizing principal axes on 2D toy data, dimensionality reduction with inverse-transform visualization, 64-dim digit visualization in 2D, explained-variance-based component selection, noise filtering via partial reconstruction, and the full eigenfaces workflow (component visualization, variance accounting, and reconstruction quality check).

## Reference Tables
| Attribute/method | Meaning |
|---|---|
| `.components_` | Principal axes (directions) |
| `.explained_variance_` | Absolute variance along each axis |
| `.explained_variance_ratio_` | Normalized variance fraction per axis |
| `.transform(X)` | Project data onto principal axes |
| `.inverse_transform(X_proj)` | Reconstruct approximate original data from reduced representation |
| `PCA(k)` | Keep top `k` components |
| `PCA(fraction)` | Keep enough components to preserve `fraction` of variance |

## Worked Example
Eigenfaces for face recognition: `PCA(150, svd_solver='randomized')` on ~3000-pixel LFW face images captures just over 90% of total variance; visualizing the first components as images ("eigenfaces") shows early components correspond to broad factors like lighting angle, later ones to specific features (eyes, nose, lips); reconstructing faces from only these 150 components (vs. ~3000 raw pixels) produces images still visually recognizable by eye — directly explaining why the Ch 43 SVM face-classification pipeline, built on this 150-dim PCA representation, worked so well despite a ~20x dimensionality reduction.

## Key Takeaways
1. PCA finds principal axes (directions of maximal variance) and represents data via its projections (principal components) onto them — an affine (translate+rotate+scale) transformation of the original feature space.
2. Dimensionality reduction keeps only the top-k (or a target variance fraction of) components, discarding low-variance directions.
3. PCA constructs an *optimal* basis (unlike truncating the native/raw feature basis) — a few PCA components typically preserve far more information than the same number of raw features.
4. Plotting cumulative `explained_variance_ratio_` is the standard way to choose how many components to keep.
5. PCA can filter noise by reconstructing from only the top (high-variance, signal-dominated) components, discarding the diffuse low-variance (noise-dominated) ones.
6. PCA is sensitive to outliers; robust variants (e.g., `SparsePCA`) exist for when this matters. It's a strong first exploratory step for high-dimensional data, though not universally optimal.

## Connects To
- **Ch 38**: PCA was previewed there for Iris dimensionality reduction.
- **Ch 42**: the "optimal basis function" framing directly parallels basis-function regression's "transform the input" principle.
- **Ch 43**: the eigenfaces example directly explains the PCA+SVM face-recognition pipeline used there.
- **Ch 44**: the digits dataset is reused from the random forest classification chapter, now visualized via PCA.
- **Ch 46**: manifold learning is introduced next as a nonlinear generalization of PCA's dimensionality-reduction goal.
