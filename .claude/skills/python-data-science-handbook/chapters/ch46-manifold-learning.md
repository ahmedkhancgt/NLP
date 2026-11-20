# Chapter 46: In Depth: Manifold Learning

## Core Idea
Manifold learning algorithms (MDS, LLE, Isomap) find low-dimensional representations of data embedded nonlinearly in high-dimensional space — handling cases where PCA's purely linear transformation fails to "unwrap" curved/twisted structure — but are finicky and best used mainly for qualitative visualization, as a complement to (not replacement for) PCA.

## Frameworks Introduced
- **Manifold**: a lower-dimensional surface (e.g., a 2D sheet of paper) embedded in a higher-dimensional space (3D); "linear embedding" (rotate/translate/scale) preserves flat geometry, "nonlinear embedding" (bend/curl/crumple) distorts it — manifold learning recovers the intrinsic lower-dimensional structure regardless of how it's embedded.
- **Multidimensional scaling (MDS)** (`sklearn.manifold.MDS(n_components=, dissimilarity=)`): recovers a D-dimensional coordinate representation from a pairwise distance matrix, preserving *all* pairwise distances — works for linear embeddings, fails on nonlinear ones (can't "unroll" a curved manifold while preserving every long-range distance).
- **Locally linear embedding (LLE)** (`sklearn.manifold.LocallyLinearEmbedding(n_neighbors=, n_components=, method=)`): preserves only *local* (nearest-neighbor) distances rather than all pairwise distances — can successfully "unroll" nonlinear manifolds that defeat MDS.
- **Isomap** (`sklearn.manifold.Isomap(n_components=, n_neighbors=)`): another nonlinear manifold technique, generally more robust than LLE on real-world (vs. toy) high-dimensional data.

## Key Concepts
- **Distance matrix / `pairwise_distances`**: an N×N matrix of distances between all point pairs — invariant to rotation/translation, the quantity MDS operates on.
- **Why MDS fails on nonlinear embeddings**: it insists on preserving distances between *all* pairs, including far-apart points on a curved manifold that can't be simultaneously "unrolled" without distortion.
- **Why LLE succeeds where MDS fails**: by relaxing to only preserve *local* (near-neighbor) distances, it allows the far-apart global structure the needed flexibility to unroll, since there's no single rigid constraint forcing every long-range distance to stay fixed.
- **`n_neighbors`**: the central tunable parameter for LLE/Isomap, with no solid quantitative method to choose it optimally — a known weakness vs. PCA's principled component-count selection.
- **PCA vs. manifold learning comparison** (several dimensions where PCA has a clear edge): handling missing data (PCA has iterative approaches; manifold learning doesn't), noise robustness (PCA filters noise naturally; manifold learning can be "short-circuited" by noise), output dimensionality choice (PCA uses explained variance; manifold learning has no analog), interpretability (PCA components have clear meaning; manifold dimensions often don't), and computational cost (manifold methods scale O(N²)-O(N³); PCA has fast randomized approximations).
- **Practical algorithm recommendations**: LLE (esp. modified LLE) for toy/synthetic problems; Isomap generally more robust for real-world high-dimensional data; t-SNE for highly clustered data (slower but often effective).

## Mental Models
- "Manifold learning's only clear advantage over PCA is preserving nonlinear relationships" — so the recommended workflow is: explore with PCA first, and only reach for manifold learning if PCA's linear assumption is clearly inadequate (visible as curved/twisted structure).
- Think of MDS vs. LLE as "global rigidity vs. local flexibility" — MDS insists every pairwise distance stay fixed (rigid, fails to unroll curves); LLE only insists neighborhoods stay locally consistent (flexible, can unroll).
- Isomap embeddings of real data often reveal interpretable "global features" (e.g., brightness, orientation) along their axes even without being told what to look for — a useful exploratory signal, even if not directly usable as classifier input.

## Anti-patterns
- **Reaching for manifold learning as a default first tool for dimensionality reduction**: "manifold learning techniques tend to be finicky enough that they are rarely used for anything more than simple qualitative visualization" — start with PCA, use manifold learning only when PCA clearly fails (e.g., high cumulative-variance requirement, visibly curved structure).
- **Applying MDS to data you suspect has a nonlinearly embedded (curved/twisted) manifold structure**: MDS cannot unroll such structure and will discard/distort the true relationships — use LLE or Isomap instead.
- **Treating manifold learning output as noise-robust or missing-data-tolerant like PCA**: manifold methods lack PCA's principled handling of these issues — be cautious applying them to noisy or incomplete real-world data without preprocessing.
- **Choosing `n_neighbors` arbitrarily and not checking sensitivity**: results are "highly dependent on the number of neighbors chosen" with no solid selection method — experiment with multiple values rather than trusting a single default.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt

# "HELLO" toy manifold data (2D)
X = make_hello(1000)             # custom function generating 2D points in "HELLO" shape

def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)
X2 = rotate(X, 20) + 5             # linear embedding (rotation+translation)

from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)           # distance matrix, invariant to rotation/translation
D2 = pairwise_distances(X2)
np.allclose(D, D2)                     # True -- confirms invariance

from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1701)
out = model.fit_transform(D)              # recovers 2D coords from distance matrix alone

# linear embedding into 3D -> MDS recovers original 2D structure
def random_projection(X, dimension=3, rseed=42):
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])
X3 = random_projection(X, 3)
model = MDS(n_components=2, random_state=1701)
out3 = model.fit_transform(X3)              # successfully recovers 2D "HELLO" (up to rotation)

# nonlinear embedding into 3D (S-curve) -> MDS FAILS
def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t); y = X[:, 1]; z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T
XS = make_hello_s_curve(X)
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)               # fails to unroll the S-curve

# LLE succeeds where MDS fails
from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2,
                                 method='modified', eigen_solver='dense')
out = model.fit_transform(XS)                 # recovers approximate 2D "HELLO" from the S-curve

# Isomap on real-world face data
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=30)   # (2370, 2914)

from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)                 # 2D embedding, reveals brightness/orientation axes

# Isomap on MNIST digits (subset, and single-digit subset)
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
data = mnist_data[::30]                                  # subsample for speed
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
```
- **What it demonstrates**: distance-matrix invariance to linear transforms, MDS succeeding on linear embeddings and failing on nonlinear ("S-curve") ones, LLE successfully unrolling the same nonlinear data, and real-world Isomap applications (face images revealing brightness/orientation axes, MNIST digit subsets revealing handwriting-style variation and data-quality outliers).

## Reference Tables
| Method | Preserves | Handles nonlinear embeddings? |
|---|---|---|
| MDS | All pairwise distances | No |
| LLE | Local (nearest-neighbor) distances only | Yes |
| Isomap | Geodesic distances along the manifold | Yes |

| Dimension | PCA | Manifold learning |
|---|---|---|
| Missing data | Iterative approaches exist | No good framework |
| Noise robustness | Filters naturally | Can "short-circuit" embedding |
| Output dimension choice | Explained variance | No clear method |
| Interpretability | Components have clear meaning | Often unclear |
| Compute cost | Fast randomized approximations | O(N²)-O(N³) |

## Worked Example
Isomap on a single MNIST digit class ("1"s only): a 2D Isomap embedding of ~2000 "1" images (with image thumbnails plotted at their projected locations) reveals the images arranged along a broad curve tracing the digit's orientation — sparse regions at the extremes correspond to 1s with "hats" or "bases," and some points reveal data-quality issues (stray pixels from neighboring digits) — demonstrating manifold learning's primary practical value as an exploratory/diagnostic visualization tool, even when not directly useful as classifier input.

## Key Takeaways
1. Manifold learning (MDS, LLE, Isomap) recovers low-dimensional structure from data nonlinearly embedded in high dimensions, where PCA's linear-only transformation fails.
2. MDS preserves all pairwise distances (works for linear embeddings only); LLE/Isomap relax to local-neighborhood preservation, enabling nonlinear "unrolling."
3. Manifold learning methods are considerably more finicky than PCA: no principled handling of missing data or noise, no clear way to choose neighbor count or output dimensionality, less interpretable, and more computationally expensive.
4. Practical guidance: PCA first, manifold learning (especially Isomap for real-world data, LLE for toy problems, t-SNE for highly clustered data) as a follow-up when linear methods clearly fail.
5. Manifold learning's main practical use is qualitative visualization/exploration of high-dimensional data (e.g., revealing interpretable structure like orientation or handwriting-style variation), rather than as a production feature-engineering step.

## Connects To
- **Ch 45**: directly extends/contrasts with PCA, explicitly framed as "PCA's nonlinear complement," reusing the LFW faces and digits datasets.
- **Ch 43/44**: the LFW faces and digits datasets are reused from the SVM and random forest chapters respectively.
- **Ch 47/48**: clustering methods covered next are another major unsupervised-learning category, often used alongside manifold learning for exploratory analysis.
