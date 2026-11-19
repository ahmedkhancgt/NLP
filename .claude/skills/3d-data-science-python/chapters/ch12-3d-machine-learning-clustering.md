# Chapter 12: 3D Machine Learning: Clustering

## Core Idea
Unsupervised clustering (k-means for centroid-based, DBSCAN for density-based, often hybridized with RANSAC) lets you extract semantic-aware structure from unlabeled 3D point clouds without costly manual annotation — a prerequisite step before scene understanding, labeling, or supervised learning.

## Frameworks Introduced
- **Three-Stage Clustering Chapter Structure**: (1) clustering principles (distances/similarity, cluster shape, cluster stability), (2) centroid-based clustering (k-means), (3) density-based clustering (DBSCAN), with RANSAC hybridization.
  - When to use: as the map for choosing and validating an unsupervised 3D segmentation approach.
  - How: establish similarity/distance metric → pick algorithm family based on expected cluster geometry → validate with stability/shape metrics before trusting results downstream.
- **Four Clustering Algorithm Families**: centroid-based (k-means — fixed k, spherical/ellipsoidal clusters), density-based (DBSCAN — auto cluster count, arbitrary shapes, noise-robust), connectivity-based (graph/neighborhood traversal — deferred to Ch. 13), distribution-based (Gaussian Mixture Models — probabilistic membership, risk of overfitting — deferred to Ch. 13).
  - When to use: match cluster geometry assumption to data — compact/uniform distributions → centroid-based; overlapping/irregular/noisy → density-based; hierarchical spatial relationships → connectivity-based; probabilistic/overlapping distributions → distribution-based.
  - How: this chapter covers k-means and DBSCAN in depth; Ch. 13 covers the other two plus deep-learning-based approaches.
- **Multi-RANSAC + DBSCAN Refinement Loop**: iteratively detect dominant planes with RANSAC, refine each candidate's inlier set to its single largest spatially-contiguous DBSCAN sub-cluster (discarding/recycling the rest), accumulate "clean" planar segments, then DBSCAN-cluster the final leftover points for complete scene segmentation.
  - When to use: segmenting complex, multi-planar scenes (e.g., a kitchen) into clean, non-overlapping structural + object segments.
  - How: loop up to `max_plane_idx` times: `segment_plane()` on the remaining cloud → DBSCAN on the plane's inliers (eps = 10× RANSAC distance_threshold) → keep only the largest DBSCAN sub-cluster as the "clean" plane, push its other points + the RANSAC outliers back into `rest` for the next iteration → finally DBSCAN the residual `rest`.

## Key Concepts
- **Hard vs. soft clustering**: hard assigns each point to exactly one cluster; soft assigns probabilistic membership across multiple clusters.
- **Centroid, homogeneity, separation**: centroid = barycenter of a cluster's points; homogeneity = mean distance from points to their centroid (lower = tighter cluster); separation = distance between cluster centroids (higher = better distinguished clusters).
- **Davies–Bouldin index**: combines homogeneity (intra-cluster) and separation (inter-cluster) into one score — lower is better.
- **Silhouette coefficient**: per-point score `s(x) = (b(x) - a(x)) / max(a(x), b(x))`, range [-1, 1], where a(x) = mean distance to own cluster, b(x) = mean distance to nearest other cluster — closer to 1 means better-assigned.
- **Rand index / adjusted Rand index**: compare predicted clustering to a ground-truth clustering by pairwise agreement; the adjusted version corrects for chance agreement.
- **Cluster stability (reassignment rate, centroid movement, boundary variance)**: measures of how consistent clustering results are across repeated runs/perturbations — low reassignment rate (<5–10%) or small centroid movement (<1–2% of spatial extent) signal robust clustering.
- **Oversegmentation vs. undersegmentation**: oversegmentation fragments one coherent object into too many pieces (local variation prioritized over global coherence); undersegmentation merges distinct objects into one cluster ("bleeding" over natural frontiers).
- **Elbow method (WCSS)**: plot within-cluster sum of squares (`kmeans.inertia_`) against k; the "elbow" point is the heuristically optimal cluster count.
- **DBSCAN parameters (eps, min_points)**: eps defines neighborhood radius; min_points is the density threshold for a point to be a core point; points failing both become noise (label -1).

## Mental Models
- Treat clustering as a semantic-labeling cost multiplier: labeling 3 clusters instead of 18 individual points (the book's chair example) is a 6x effort reduction, and labeling a cluster propagates to every point within it.
- Use the "LeCun cake analogy" framing: clustering/unsupervised learning is the foundational layer beneath classification pipelines — most of the "cake" (effort/value) is unsupervised structure discovery, with supervised labeling as a thin top layer.
- Picture k-means as "assign then relocate, repeat until stable" — it is not guaranteed to reach a global optimum, so initialization (`k-means++`) and multiple runs matter.
- Picture DBSCAN as "contagious density" — a point joins a region only if its local neighborhood is dense enough, and the region keeps growing through chains of dense neighbors until it runs out of density, naturally isolating noise.
- When facing "lines" artifacts from naive iterative RANSAC plane detection, think "RANSAC has no concept of spatial contiguity" — a plane equation fits any coplanar points regardless of whether they're part of the same physical surface; DBSCAN refinement reintroduces that spatial-contiguity constraint.

## Anti-patterns
- **Using k-means when clusters have irregular shapes, varying density, or an unknown count**: k-means assumes spherical/ellipsoidal, roughly equal-density clusters and needs k specified in advance — DBSCAN handles these cases better.
- **Trusting raw iterative RANSAC plane segmentation without density refinement**: produces spatially-incoherent "lines" artifacts that slice unrelated coplanar regions together — always refine each RANSAC plane candidate with DBSCAN to keep only the spatially contiguous sub-cluster.
- **Not filtering the ground plane before clustering a scene with k-means on spatial features alone**: ground points dominate the geometry and distort the resulting clusters (the book's airport-LiDAR example shows triangular, uninterpretable splits before ground filtering).
- **Judging clustering quality by visual inspection alone**: use quantitative metrics (silhouette score, Davies–Bouldin index, reassignment rate) to validate, especially before using cluster output to seed a labeling pipeline.
- **Picking DBSCAN's eps/min_points arbitrarily without reference to point density**: derive eps from the data's characteristic point spacing (e.g., `d_threshold × 10`) rather than a fixed guess, since density varies across datasets and even within a single scan.

## Code Examples
```python
from sklearn.cluster import KMeans
X = np.column_stack((x[mask], y[mask]))
kmeans = KMeans(n_clusters=2).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)

# Elbow method for choosing k
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42).fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss)   # look for the "elbow"
```
- **What it demonstrates**: ground-masked k-means clustering on spatial features, plus the elbow method for automatically choosing the cluster count k (elbow at k=2 in the book's plane-detection example).

```python
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0   # noise points colored black
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
```
- **What it demonstrates**: DBSCAN segmentation of a point cloud (post RANSAC plane removal) with noise-aware coloring for visual QA.

```python
segment_models, segments = {}, {}
max_plane_idx = 20
rest = pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    segments[i] = rest.select_by_index(inliers)
    labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=10))
    candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
    best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])
    rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(
        list(np.where(labels != best_candidate)[0]))
    segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))
```
- **What it demonstrates**: the full Multi-RANSAC + DBSCAN refinement loop that removes spatially-incoherent "lines" artifacts from naive iterative plane detection by keeping only each plane's largest contiguous sub-cluster.

## Reference Tables
*Table 12-1 (partial, paraphrased): Clustering approach comparison*

| Approach | Best use cases | Key algorithm | Key characteristics |
|---|---|---|---|
| Centroid-based | Uniform/compact geometric distributions, urban infrastructure mapping | k-means | Fixed cluster count, spherical shapes, sensitive to outliers |
| Density-based | Complex, nonlinear, overlapping distributions | DBSCAN | Auto cluster count, arbitrary shapes, handles noise |
| Connectivity-based | Hierarchical spatial relationships | Graph-based (k-NN graphs, ε-neighborhoods) | Covered in Ch. 13 |
| Distribution-based | Overlapping/intricate configs (e.g., medical imaging) | Gaussian Mixture Models | Probabilistic membership; risk of overfitting; covered in Ch. 13 |

*Quality/stability metrics*

| Metric | Measures | Good value |
|---|---|---|
| Davies–Bouldin index | Homogeneity vs. separation combined | Lower is better |
| Silhouette coefficient | Per-point cluster fit | Closer to 1 |
| Rand index / adjusted Rand index | Agreement with ground truth | Higher is better |
| Reassignment rate | Stability across runs | <5–10% |
| Centroid movement | Stability across runs | <1–2% of spatial extent |

## Worked Example
The chapter runs two parallel worked examples. (1) **Airport security k-means**: load `KME_planes.xyz` (LiDAR point cloud with X,Y,Z,illuminance,reflectance,intensity,nb_of_returns), visualize X-Z and Y-Z projections with a mean-Z reference line, mask out ground points (`z > mean(z)`), run `KMeans(n_clusters=2)` on the masked X,Y features to automatically separate two stationary aircraft planes, then use the elbow method (WCSS vs. k, 1 to 20) to confirm k=2 is optimal, and export the labeled result as `.xyz`. (2) **Kitchen robot RANSAC+DBSCAN**: load a terrestrial-laser-scanned kitchen (`TLS_kitchen.ply`), first validate RANSAC plane detection alone (`segment_plane`, coloring inliers red/outliers gray) and DBSCAN alone (`eps=0.05, min_points=10`, revealing cabinets/heating controls/lamp as distinct clusters) as isolated proofs of concept; then combine them into the full Multi-RANSAC refinement loop (20 iterations) to extract 20 clean planar segments (walls, countertops, cabinet faces) without the "line" artifacts of naive RANSAC; finally DBSCAN-cluster the loop's final leftover (`rest`) points to capture remaining floating objects (light panel, window shades), achieving complete scene segmentation ready for a robot's spatial understanding. A supporting comparison example ("DBSCAN vs. k-means," `KME_cars.xyz`) shows k-means correctly separating 3 closely-spaced cars (k=3) while DBSCAN (eps=0.1/0.2/0.5) struggles to separate two of them regardless of eps tuning — illustrating that k-means can outperform DBSCAN when the true object count is known and objects are compact/similarly sized.

## Key Takeaways
1. Clustering (unsupervised) is the practical answer to 3D data's chronic lack of labeled datasets — use it as a cost-reducing seed step before manual labeling or supervised learning.
2. Match the algorithm family to the data's cluster geometry: k-means for compact/known-count clusters, DBSCAN for arbitrary-shape/noisy/unknown-count clusters.
3. Use the elbow method (WCSS) to choose k systematically rather than guessing.
4. Naive iterative RANSAC plane detection creates spatially-incoherent artifacts; refine each detected plane with DBSCAN to keep only its largest contiguous sub-cluster.
5. Validate clustering quality quantitatively (silhouette, Davies–Bouldin, reassignment rate) rather than relying on visual inspection alone, especially before using results to seed downstream labeling.
6. k-means can outperform DBSCAN when the object count is known and objects are compact and similarly sized, even though DBSCAN is generally more flexible — test both.
7. Clustering works on any feature combination (spatial, color, intensity, reflectance, number of returns) — don't limit yourself to X,Y,Z features alone.

## Connects To
- **Ch 6, 7**: feature extraction/dimensionality reduction (PCA) referenced as complementary to richer clustering feature spaces.
- **Ch 8**: deviation/statistical analysis groundwork reused for ground-plane filtering.
- **Ch 9**: RANSAC (introduced there) is the hybridization partner for plane detection throughout this chapter.
- **Ch 13**: extends to connectivity-based and distribution-based clustering (graphs, foundation models, deep learning).
- **Ch 14**: builds on clustering output as the seed for semantic injection/supervised learning.
- **scikit-learn (KMeans, DBSCAN), Open3D (segment_plane, cluster_dbscan), Matplotlib**: the core library stack for this chapter's implementations.
