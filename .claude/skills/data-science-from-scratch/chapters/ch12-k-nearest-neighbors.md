# Chapter 12: k-Nearest Neighbors

## Core Idea
k-Nearest Neighbors (k-NN) classifies a new point by letting its k closest labeled neighbors vote, requiring only a distance function and the assumption that nearby points are similar — simple and assumption-free, but it explains nothing about *why* and degrades badly in high dimensions (the curse of dimensionality).

## Frameworks Introduced
- **k-NN classification**: pick k; for a new point, find the k nearest labeled points (via `distance` from Ch4) and have them vote via majority rule.
  - When to use: when "similar inputs have similar outputs" is a reasonable assumption and you don't need to understand *why* (k-NN gives prediction, not explanation/causal insight).
  - How: `knn_classify(k, labeled_points, new_point)` sorts all labeled points by distance to `new_point`, takes the k closest, and calls `majority_vote` on their labels.
- **Tie-breaking by recursive shrinking**: `majority_vote(labels)` recursively drops the farthest point (`labels[:-1]`) and retries whenever there's a vote tie, guaranteed to terminate since k=1 always has a unique winner.
  - When to use: as one of three legitimate tie-breaking strategies (the others being random pick or distance-weighted voting); the book implements this one for determinism.
- **Choosing k**: too small (e.g., k=1) lets outliers dominate; too large (e.g., k=n) just predicts the dataset's most common class — tune via a validation set in real applications.

## Key Concepts
- **LabeledPoint**: a `NamedTuple(point: Vector, label: str)` pairing a feature vector with its known class.
- **Curse of dimensionality**: in high-dimensional space, points tend not to be meaningfully close to each other — every extra dimension (even pure noise) is another chance for two points to drift apart, so the ratio of nearest-neighbor distance to average distance approaches 1 as dimensions grow.
- **Sparsity of high-dimensional space**: the same number of random sample points covers a unit interval well, a unit square less well, and a unit cube even less — illustrating that fixed sample sizes leave increasingly large empty regions as dimensionality grows.
- **Confusion matrix (predicted, actual) counting**: track `(predicted, actual)` pairs with a `defaultdict(int)` to see where a classifier's errors concentrate, not just its overall accuracy.

## Mental Models
- **k-NN trades interpretability for simplicity**: "predicting my vote based on my neighbors' votes doesn't tell you much about what causes me to vote the way I do" — contrast with a model built on income/marital status, which would be explanatory as well as predictive.
- **Nearest neighbors only uses local information**: unlike most techniques in the book that fit global patterns from the whole dataset, k-NN's prediction for a point depends only on a handful of nearby points — deliberately throwing away most of the dataset's information at prediction time.
- **"Close in every dimension" gets exponentially harder**: two points are only close overall if they're close along *every* axis; adding dimensions (especially noisy/irrelevant ones) multiplies the ways two otherwise-similar points can end up far apart.
- **Dimensionality reduction as a k-NN prerequisite**: when working in many dimensions, reduce dimensionality (Ch10's PCA) first, unless the data has enough intrinsic lower-dimensional structure to behave well anyway.

## Anti-patterns
- **Using k-NN in high dimensions without dimensionality reduction**: the curse of dimensionality means "nearest" stops being meaningfully different from "average distance," so distance-based voting degrades — reduce dimensions first.
- **Picking k arbitrarily without validation**: k=1 overfits to individual outliers; k=n underfits to the global majority class — use a validation set to tune k properly rather than guessing.
- **Breaking ties naively (e.g., always picking the first alphabetically)**: biases predictions systematically; use random choice, distance-weighting, or k-shrinking instead.
- **Expecting k-NN to explain causal drivers**: it predicts well but offers no insight into *why* — don't reach for k-NN when the goal is understanding the relationship, not just predicting an outcome.

## Code Examples
```python
def majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to farthest."""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([c for c in vote_counts.values() if c == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])  # try again without the farthest
```
- **What it demonstrates**: recursive tie-breaking that always terminates (worst case: one label left).

```python
class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    return majority_vote(k_nearest_labels)
```
- **What it demonstrates**: the complete from-scratch k-NN classifier, reusing Ch4's `distance` and Ch10's `NamedTuple` pattern.

```python
def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim)) for _ in range(num_pairs)]
# as dim grows from 1 to 100: avg distance rises, min/avg ratio -> 1
```
- **What it demonstrates**: empirically illustrating the curse of dimensionality by measuring how nearest-vs-average distance ratios degrade as dimensions increase.

## Reference Tables
| k value | Risk |
|---|---|
| Too small (e.g., 1) | outliers dominate predictions (high variance) |
| Too large (e.g., n) | always predicts the majority class (high bias) |
| Well-tuned (via validation set) | balances the two |

| Dimensions | Random-point coverage of unit hypercube |
|---|---|
| 1 | good coverage with modest sample size |
| 2 | noticeably sparser |
| 3 | sparser still, visible empty regions |
| 100 (typical ML feature count) | nearest-neighbor distance ~= average distance; "nearness" loses meaning |

## Worked Example
The Iris dataset (150 flowers, 3 species, 4 measurements: sepal length/width, petal length/width) is split 70/30 into train/test (`split_data(iris_data, 0.70)`, seed 12). Using k=5, `knn_classify` is run over the 45 test points, predictions compared to actual species and tallied in a `(predicted, actual)` confusion matrix. The model predicts almost perfectly — only one versicolor is misclassified as virginica — because the 4D scatterplot matrix shows the species cluster tightly once all four measurements (not just sepal length/width alone) are considered together.

## Key Takeaways
1. k-NN needs only a distance function and the "nearby points are similar" assumption — no training phase, no parameters to fit beyond k.
2. Tune k via a validation set; too small overfits to noise/outliers, too large underfits to the global majority class.
3. k-NN sacrifices interpretability for simplicity — it predicts well but doesn't explain the underlying drivers.
4. The curse of dimensionality degrades k-NN (and any distance-based method) in high dimensions — reduce dimensionality first (Ch10's PCA) when working with many features.
5. Always inspect a confusion matrix, not just overall accuracy, to see where a classifier's mistakes concentrate.

## Connects To
- **Ch4**: `distance()` is the core primitive k-NN is built on.
- **Ch10 (Working with Data)**: PCA/dimensionality reduction is the recommended fix for k-NN's curse-of-dimensionality problem; NamedTuple pattern (`LabeledPoint`) reused directly.
- **Ch11 (Machine Learning)**: train/test split and confusion-matrix evaluation concepts applied directly here.
- **Ch20 (Clustering)**: another distance-based technique sharing the same curse-of-dimensionality vulnerability.
- **scikit-learn**: external library with production-grade nearest-neighbor implementations.
