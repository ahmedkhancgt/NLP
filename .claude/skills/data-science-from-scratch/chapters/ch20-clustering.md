# Chapter 20: Clustering

## Core Idea
Clustering is unsupervised learning (no labels, no "correct" answer) that groups similar points; the chapter implements k-means (iteratively reassign points to the nearest of k means, recompute means, repeat) and bottom-up hierarchical clustering (merge the two closest clusters repeatedly until one remains, then "unmerge" to any desired cluster count), applying both to meetup-location and image-recoloring examples.

## Frameworks Introduced
- **k-means clustering**: fix `k`; iterate (1) assign each point to its nearest of `k` means, (2) recompute means from current assignments, (3) stop when no assignment changes.
  - When to use: partitioning points into a fixed number of roughly-spherical clusters when you already have a sense of (or can determine) the right `k`.
  - How: `KMeans(k).train(inputs)` initializes random assignments, then alternates `cluster_means` and `classify` (nearest-mean lookup via `squared_distance`) until convergence.
- **Choosing k via the elbow method**: plot total squared clustering error (`squared_clustering_errors`) against `k` and look for where the curve's rate of improvement "bends" (diminishing returns past that point).
  - When to use: whenever `k` isn't externally fixed by the problem (e.g., budget constraints) and needs to be chosen from the data itself.
- **Bottom-up hierarchical clustering**: start with every point as its own cluster (`Leaf`); repeatedly find and merge the two closest clusters (tracked via `Merged(children, order)`) until one cluster remains; recover any number of clusters by "unmerging" from the most recent merge backward.
  - When to use: when you want a full hierarchy of nested groupings (a dendrogram) rather than committing to one fixed `k`, or want to explore cluster structure at multiple granularities from one computation.
  - How: `cluster_distance(cluster1, cluster2, distance_agg)` with `distance_agg` as `min` (chain-like, "closest touching" clusters), `max` (tight, "smallest enclosing ball" clusters), or average, aggregating all pairwise point distances between the two clusters.

## Key Concepts
- **Unsupervised learning**: works with unlabeled data (or ignores existing labels) to find inherent structure, contrasted with the supervised algorithms (k-NN, Naive Bayes, regression, decision trees, neural nets) earlier in the book.
- **No "correct" clustering**: different clustering schemes can all be reasonable; each is "more optimal" with respect to its own goodness metric, not objectively "more correct."
- **Clusters don't self-label**: you must inspect what's inside each cluster to understand/name what it represents.
- **Leaf / Merged cluster representation**: `Leaf(value: Vector)` for a single unmerged point; `Merged(children: tuple, order: int)` for a merged pair, with `order` tracking merge sequence (lower = more recent merge, so unmerging proceeds from lowest order upward); leaves get `order = infinity` since they were never merged.
- **cluster_distance aggregation choice**: `min` yields chain-like, loosely connected clusters; `max` yields tight, compact (small-diameter) clusters; average is a common middle ground.

## Mental Models
- **k-means' iterative refinement always converges (but not necessarily to the global optimum)**: alternating "assign to nearest mean" and "recompute means from assignments" monotonically reduces total squared error at each full step, guaranteeing termination, but different random starting assignments can converge to different local optima — rerunning with different seeds is standard practice.
- **Hierarchical clustering trades a fixed `k` for a full dendrogram**: instead of committing upfront to one `k`, build the complete merge history once, then slice it at any granularity by unmerging — useful when you want to explore multiple groupings without re-running the algorithm.
- **The distance-aggregation choice shapes cluster geometry**: `min`-based merging favors clusters that are close at their nearest edges (can produce long chains), while `max`-based merging favors clusters that are compact as a whole (closer to k-means-style spherical clusters) — pick based on whether you want loosely-connected or tightly-bounded groups.
- **Clustering as a practical tool for diverse problems**: the same k-means machinery solves both "where to hold meetups" (2D geographic clustering) and "reduce an image to 5 colors" (3D RGB-space clustering) — the algorithm is domain-agnostic; only the input vectors change.

## Anti-patterns
- **Expecting one "correct" clustering or k**: there is no ground truth to validate against in unsupervised learning; different reasonable people (or algorithms) can produce different, equally defensible groupings.
- **Taking the mean of an empty cluster**: undefined; `cluster_means` handles this by randomly reassigning an input as that cluster's "mean" instead of crashing.
- **Recomputing all pairwise distances at every step of hierarchical clustering**: the book's implementation is explicitly flagged as "shockingly inefficient" for exactly this reason — a production implementation should precompute/cache pairwise distances.
- **Using `min`-based (chain-favoring) linkage when you actually want compact, ball-shaped clusters**: produces very different (often less useful) groupings than `max`-based linkage on the same data — choose the aggregation deliberately.

## Code Examples
```python
class KMeans:
    def classify(self, input: Vector) -> int:
        return min(range(self.k), key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        assignments = [random.randrange(self.k) for _ in inputs]
        while True:
            self.means = cluster_means(self.k, inputs, assignments)
            new_assignments = [self.classify(input) for input in inputs]
            if num_differences(assignments, new_assignments) == 0:
                return
            assignments = new_assignments
            self.means = cluster_means(self.k, inputs, assignments)
```
- **What it demonstrates**: the complete k-means train loop — assign, recompute means, repeat until stable.

```python
def cluster_distance(cluster1, cluster2, distance_agg=min):
    return distance_agg([distance(v1, v2)
                         for v1 in get_values(cluster1)
                         for v2 in get_values(cluster2)])

def bottom_up_cluster(inputs, distance_agg=min):
    clusters = [Leaf(input) for input in inputs]
    while len(clusters) > 1:
        c1, c2 = min(((c1, c2) for i, c1 in enumerate(clusters) for c2 in clusters[:i]),
                      key=lambda pair: cluster_distance(*pair, distance_agg))
        clusters = [c for c in clusters if c != c1 and c != c2]
        clusters.append(Merged((c1, c2), order=len(clusters)))
    return clusters[0]
```
- **What it demonstrates**: greedy nearest-cluster merging that builds a complete hierarchy, one merge at a time.

## Reference Tables
| Method | Fixed cluster count upfront? | Shape bias | Efficiency |
|---|---|---|---|
| k-means | yes (`k` chosen in advance) | roughly spherical | fast, iterative |
| Hierarchical (min linkage) | no (choose after the fact) | chain-like | slow (naive: recomputes all distances) |
| Hierarchical (max linkage) | no | tight/compact | slow (same) |

## Worked Example
DataSciencester hometown user locations (2D) are clustered with k=3 via k-means (seed 12), converging to means near `[-44,5]`, `[-16,-10]`, `[18,20]`; with only budget for 2 meetups (seed 0), k=2 converges to `[-26,-5]` and `[18,20]`. Plotting total squared error vs. k from 1 to n shows an "elbow" at k=3, confirming the earlier visual choice. A separate image-recoloring example loads an RGB image, flattens it to a list of 3D pixel vectors, clusters into k=5, and recolors each pixel to its cluster's mean color — reducing an image to a 5-color palette via the same k-means machinery. Bottom-up clustering on the same 20 meetup locations, using `min` linkage, produces chain-like 3-cluster groupings visually different from k-means' spherical ones; switching to `max` linkage recovers results matching the k-means clustering.

## Key Takeaways
1. Clustering is unsupervised: there's no "correct" answer, only clusterings that are more or less useful/optimal for a given metric — and clusters must be interpreted/labeled manually.
2. k-means requires choosing `k` upfront; use the elbow method (plot squared error vs. k) when `k` isn't externally constrained.
3. Hierarchical clustering avoids committing to a fixed `k` by building a full merge hierarchy once, then slicing it at any desired granularity.
4. The linkage/aggregation function (`min` vs. `max` vs. average) in hierarchical clustering strongly shapes resulting cluster geometry — choose deliberately.
5. The same clustering algorithm applies across very different domains (geographic coordinates, RGB pixel colors) — only the input vector representation changes.

## Connects To
- **Ch4**: `vector_mean`, `distance`, `squared_distance` are the geometric primitives k-means and hierarchical clustering are built on.
- **Ch10 (Working with Data)**: PCA/dimensionality reduction (Ch10) is often a useful preprocessing step before clustering high-dimensional data, since clustering (like k-NN) relies on meaningful distances.
- **Ch12 (k-Nearest Neighbors)**: shares the same curse-of-dimensionality vulnerability, since both rely on distance in feature space.
- **scikit-learn (sklearn.cluster)** / **SciPy (scipy.cluster.vq, scipy.cluster.hierarchy)**: external libraries providing production-grade, efficient k-means and hierarchical clustering implementations (including Ward's method, a different merge criterion).
