# Chapter 47: In Depth: k-Means Clustering

## Core Idea
k-means clustering finds a predetermined number of cluster centers (each the mean of its assigned points, with each point closer to its own center than any other) via a fast, iterative expectation-maximization procedure — simple and widely applicable, but limited to linear cluster boundaries, sensitive to initialization, and requiring the cluster count to be chosen in advance.

## Frameworks Introduced
- **`sklearn.cluster.KMeans(n_clusters=)`**: the standard estimator — `.fit()`/`.predict()`/`.fit_predict()`, with `.cluster_centers_` giving the learned centers.
- **Expectation-maximization (E-M) procedure**: (1) guess initial cluster centers; (2) repeat until convergence: **E-step** (assign each point to its nearest current center) → **M-step** (recompute each center as the mean of its assigned points) — each iteration provably improves (never worsens) the clustering, though not guaranteed to reach the *global* optimum.
- **`SpectralClustering`**: kernelized k-means variant using a nearest-neighbor graph to project data into a higher-dimensional space before applying k-means — enables discovery of nonlinear cluster boundaries (analogous to kernel SVM, Ch 43).
- **`MiniBatchKMeans`**: batch-based variant that updates centers using only a data subset per iteration — much faster for large datasets, same interface as `KMeans`.

## Key Concepts
- **Two core k-means assumptions**: (1) a cluster center is the arithmetic mean of its points; (2) each point is closer to its own cluster's center than to any other's — together these imply linear (Voronoi-like) decision boundaries between clusters.
- **`n_init`**: Scikit-Learn's default of re-running k-means from 10 different random initializations and keeping the best result, mitigating (but not eliminating) the local-optimum problem.
- **Known limitations** (four explicit caveats): (1) E-M isn't guaranteed to reach the global optimum — sensitive to initialization; (2) the number of clusters (`n_clusters`) must be specified beforehand, unlike e.g. DBSCAN; (3) k-means produces only linear cluster boundaries, failing on complex geometries (e.g., interleaved crescents) unless kernelized (`SpectralClustering`); (4) scales poorly to very large datasets since every iteration touches every point — mitigated by `MiniBatchKMeans`.
- **Permutation-insensitivity of cluster labels**: k-means assigns arbitrary integer labels to clusters — matching learned clusters to true labels (when available, for evaluation) requires a post-hoc alignment step (e.g., via `scipy.stats.mode` per cluster).
- **Color quantization application**: treating image pixels as points in 3D RGB space and clustering them into k colors is a direct, non-obvious application of k-means for lossy compression.

## Mental Models
- E-M is "guess, then alternate between two cheap local-improvement steps" — far cheaper than exhaustively searching the combinatorially huge space of possible cluster assignments, at the cost of only guaranteeing local (not global) optimality.
- k-means' "closer to own center than any other" rule geometrically partitions space into convex (linear-boundary) regions — any dataset whose true clusters have non-convex or curved boundaries will defeat vanilla k-means, which is exactly the signal to reach for kernelized variants.
- Treating any "cloud of points" abstractly (pixels in color space, digit images in pixel space) as clusterable data is a powerful generalization — k-means doesn't care what the "points" represent.

## Anti-patterns
- **Running k-means once with a single random initialization and trusting the result**: risks landing in a poor local optimum (visibly different results from different seeds) — rely on Scikit-Learn's default `n_init=10` reruns (or increase it) rather than disabling this safety net.
- **Applying vanilla k-means to data with nonlinear/non-convex cluster shapes (e.g., interleaved crescents)**: fundamentally cannot succeed, since k-means boundaries are always linear — use `SpectralClustering` or another kernelized/nonlinear clustering method instead.
- **Choosing `n_clusters` arbitrarily without any validation**: an incorrect cluster count (e.g., 6 clusters for 4 true groups) will "happily" produce a confident-looking but meaningless result — consider silhouette analysis or algorithms that infer cluster count (DBSCAN, mean-shift, affinity propagation, or Gaussian mixture models, Ch 48).
- **Using standard `KMeans` on very large datasets (e.g., millions of pixels) without considering `MiniBatchKMeans`**: full k-means revisits every point every iteration, which can be prohibitively slow at scale.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

# manual E-M implementation (illustrative)
from sklearn.metrics import pairwise_distances_argmin
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers)                 # E-step
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])  # M-step
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

# nonlinear boundaries: kernelized k-means
from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)
labels = KMeans(2, random_state=0).fit_predict(X)     # fails: linear boundary assumption

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(X)                             # succeeds via kernel trick

# Example: digits clustering (unsupervised digit "classification")
from sklearn.datasets import load_digits
digits = load_digits()
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]           # align cluster labels to true labels

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)                       # ~0.79

# improve with t-SNE preprocessing
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='random', learning_rate='auto', random_state=0)
digits_proj = tsne.fit_transform(digits.data)
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)
# re-align and re-score -> ~0.94 accuracy

# Example: color compression
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")                        # (427, 640, 3)
data = (china / 255.0).reshape(-1, 3)                            # (273280, 3) RGB points

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]         # recolor with 16 colors
china_recolored = new_colors.reshape(china.shape)
```
- **What it demonstrates**: basic k-means clustering with visualized centers, a from-scratch E-M implementation, the two canonical failure modes (bad initialization, nonlinear boundaries) with the `SpectralClustering` fix, an unsupervised digit-clustering application with label alignment and t-SNE preprocessing boost, and a color-quantization/compression application using `MiniBatchKMeans`.

## Reference Tables
| Limitation | Mitigation |
|---|---|
| Local optima (bad initialization) | `n_init` multiple random restarts (default 10) |
| Must specify `n_clusters` beforehand | Silhouette analysis, or alternatives (DBSCAN, mean-shift, affinity propagation, GMM) |
| Only linear cluster boundaries | `SpectralClustering` (kernelized k-means) |
| Slow at scale (visits all points each iteration) | `MiniBatchKMeans` |

## Worked Example
Color compression: a 427×640 image with up to 16 million possible RGB colors is reshaped into 273,280 points in 3D color space; `MiniBatchKMeans(16)` clusters these into 16 representative colors, and recoloring each pixel with its nearest cluster center produces a visually recognizable image using only 16 colors instead of millions — achieving roughly a millionfold raw storage compression factor (though far below purpose-built codecs like JPEG), illustrating k-means applied to an unconventional "points" interpretation (pixels as 3D color-space coordinates).

## Key Takeaways
1. k-means alternates an E-step (assign points to nearest center) and M-step (recompute centers as means) until convergence — each step provably improves the fit, but only to a local, not necessarily global, optimum.
2. Scikit-Learn mitigates the local-optimum risk via multiple random restarts (`n_init`, default 10).
3. k-means requires specifying `n_clusters` in advance and can only find linear (convex) cluster boundaries — `SpectralClustering` extends it to nonlinear boundaries via a kernel trick analogous to kernel SVM.
4. `MiniBatchKMeans` trades a small amount of accuracy for much better speed/scalability on large datasets.
5. k-means generalizes to any "points in space" interpretation — demonstrated on digit-pixel vectors (unsupervised digit grouping, boosted by t-SNE preprocessing) and image-pixel RGB values (color compression).
6. Evaluating unsupervised clustering against known ground truth requires a post-hoc label-alignment step, since cluster indices are arbitrary.

## Connects To
- **Ch 43**: `SpectralClustering`'s kernel-trick approach to nonlinear boundaries directly parallels kernel SVM.
- **Ch 44/45**: the digits dataset is reused from the random forest and PCA chapters, now clustered unsupervised.
- **Ch 46**: t-SNE (introduced there) is used here to boost clustering accuracy via preprocessing.
- **Ch 48**: Gaussian mixture models are introduced next as a more flexible alternative addressing some of k-means's limitations (notably fixed cluster-count rigidity and hard boundary assumptions).
