# Chapter 9: 3D Shape Recognition

## Core Idea
Manmade environments decompose into known geometric primitives (planes, spheres, cylinders), so robust 3D scene understanding can be built from two complementary shape-recognition algorithms — RANSAC (model-fitting, robust to noise/outliers) and region growing (bottom-up segmentation by local similarity) — and their hybrid combination (`detect_planar_patches`) outperforms either alone.

## Frameworks Introduced
- **RANSAC (Random Sample Consensus), 4-step loop**: (1) pick a geometric model type (plane, sphere, etc.), (2) fit it to a minimal random sample (3 points for a plane, 4 for a sphere), (3) count inliers via a distance threshold, (4) repeat N iterations and keep the model with the most inliers.
  - When to use: detecting dominant, well-defined geometric primitives in noisy real-world point clouds (walls, floors, pipes, balls) — robust precisely because it never requires most of the data to be clean.
  - How: randomly sample minimal point set → derive model params algebraically (cross product for plane normal; linear system for sphere center/radius) → compute per-point distance to model → threshold to get inliers → track best-scoring model across iterations.
- **Region Growing, 6-step process**: (1) select a seed point, (2) create an empty region, (3) find neighbors via a search structure (k-d tree), (4) test each neighbor against similarity criteria (distance + normal angle), (5) add passing neighbors to the region and expand, (6) repeat until no new points qualify; then restart with a new seed for the next region.
  - When to use: segmenting a scene into locally-consistent regions when shapes aren't simple/predefined — naturally handles complex, non-parametric geometry.
  - How: build a k-d tree over all points → pop a seed, push its distance-threshold neighbors onto a stack → pop each stack point, check normal-angle similarity to the seed, if it passes remove from "unsegmented," add to region, push its own neighbors → repeat until the unsegmented set is empty.
- **Hybrid RANSAC + Region Growing (`detect_planar_patches`)**: use RANSAC-style robust fitting to find candidate planar regions while using region-growing-style criteria (normal variance threshold, coplanarity threshold, outlier ratio) to delineate contiguous patches — Open3D's built-in method operationalizes this fusion.
  - When to use: when you need both RANSAC's robustness to noise/outliers AND region growing's ability to respect local feature similarity (e.g., color) within a detected shape.
  - How: estimate normals → call `pcd.detect_planar_patches(normal_variance_threshold_deg, coplanarity_deg, outlier_ratio, min_plane_edge_length, min_num_points, search_param)` → get oriented bounding boxes per patch → convert to meshes and sum volumes for quantitative scene analysis.

## Key Concepts
- **Inlier / outlier set**: RANSAC's core output — points consistent with the fitted model (inliers) vs. the rest (outliers); either set can become the object of interest depending on the task.
- **Point-to-plane distance via dot product**: `distance = (a·x + b·y + c·z + d) / sqrt(a²+b²+c²)`, the signed orthogonal distance of a point to a plane defined by normal (a,b,c) and offset d; take absolute value to remove normal-direction ambiguity.
- **Adaptive threshold**: instead of a fixed distance threshold, compute it from the data itself — e.g., `threshold = min(nearest_dist_mean) + std(nearest_dist_mean)` — so the same algorithm works across varying point cloud densities without manual retuning.
- **Oversegmentation vs. undersegmentation**: too-strict (lenient) thresholds in region growing create too many small fragments (oversegmentation) or merge distinct objects together (undersegmentation) — the central tuning trade-off.
- **4-connectivity vs. 6-connectivity**: neighbor-adjacency definitions for region growing — in-plane neighbors only (4) vs. also above/below (6).
- **Oriented bounding box (OBB) from planar patch**: each detected planar patch is represented as a thin oriented bounding box, whose volume summed across patches estimates total occupied structural volume.
- **Mahalanobis distance**: a covariance-aware distance metric, useful for region growing in correlated/non-uniformly distributed feature spaces (vs. Euclidean or Manhattan).

## Mental Models
- RANSAC is "stretching a piece of paper between three pins" (plane) or "finding a bubble that touches four points" (sphere) — a physical, intuitive way to picture the minimal-sample model fit.
- Region growing is "a flood fill in 3D, gated by both distance and surface-normal similarity" rather than a flood fill gated by color alone (as in 2D image segmentation).
- Treat RANSAC and region growing as complementary, not competing: RANSAC answers "does this noisy data contain a known shape?" (model fitting); region growing answers "which contiguous areas share local properties?" (segmentation). Table 9-2 in the book makes this contrast explicit.
- Use "adaptive thresholding via local nearest-neighbor statistics" as the default pattern whenever you'd otherwise hardcode a magic-number distance threshold — it generalizes the same code across sparse and dense scans.

## Anti-patterns
- **Using a fixed distance threshold across point clouds of very different densities**: a threshold tuned for a dense indoor scan will be useless (too tight) on a sparse outdoor LiDAR scan — use the adaptive, k-d-tree-derived threshold instead.
- **Using `np.delete` to remove indices from a large point array**: it copies the entire array before removing elements — use Boolean masking instead for large point clouds (single O(n) allocation vs. a full copy-then-shrink).
- **Picking a bad region-growing seed point (e.g., inside a small isolated object)**: can produce a segmentation that fails to capture the full intended region — seed selection materially affects output quality.
- **Ignoring RANSAC's documented limitations**: sensitivity to noise, difficulty with complex/irregular shapes, and costly parameter tuning (Table 9-1) — don't expect RANSAC alone to handle highly irregular geometry; combine with region growing or other techniques instead.
- **Forgetting the normal-direction sign ambiguity in point-to-plane distance**: a plane's normal can point either of two ways, so a raw signed distance can be arbitrarily negative — always take the absolute value unless side-of-plane information is explicitly needed.

## Code Examples
```python
def ransac_plane(points, num_iterations=1000, threshold=0.1):
    best_inliers, best_plane = [], None
    for _ in range(num_iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
        normal = np.cross(v1, v2)
        a, b, c = normal / np.linalg.norm(normal)
        d = -np.sum(normal * sample[1])
        distances = (np.dot(points, [a, b, c]) + d) / np.sqrt(a**2 + b**2 + c**2)
        inliers = np.where(np.abs(distances) < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers, best_plane = inliers, (a, b, c, d)
    return best_plane, best_inliers
```
- **What it demonstrates**: a complete from-scratch RANSAC plane-fitting loop — minimal-sample fit, inlier counting, best-model tracking across iterations.

```python
tree = KDTree(points, leaf_size=2)
nearest_dist, nearest_ind = tree.query(points, k=8)
nearest_dist_mean = np.mean(nearest_dist[:, 1:], axis=0)
threshold = np.min(nearest_dist_mean) + np.std(nearest_dist_mean)  # adaptive threshold
```
- **What it demonstrates**: data-driven, density-adaptive threshold computation that removes the need for manual, dataset-specific tuning.

```python
angle_threshold = np.pi / 6      # 30 degrees
distance_threshold = 0.1
kd_tree = KDTree(input_points)
segments, unsegmented = [], set(range(len(input_points)))
while unsegmented:
    seed_index = unsegmented.pop()
    segment = [seed_index]
    stack = list(kd_tree.query_ball_point(input_points[seed_index], distance_threshold))
    while stack:
        point_index = stack.pop()
        if point_index not in unsegmented:
            continue
        normal_angle = np.arccos(np.dot(normals[seed_index], normals[point_index]))
        if normal_angle < angle_threshold:
            unsegmented.remove(point_index)
            segment.append(point_index)
            stack.extend(kd_tree.query_ball_point(input_points[point_index], distance_threshold))
    segments.append(segment)
```
- **What it demonstrates**: the complete region-growing implementation using a k-d tree, a distance-threshold neighbor search, and a normal-angle similarity criterion, processed depth-first via a stack.

```python
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=p_radius, max_nn=30))
oboxes = pcd.detect_planar_patches(
    p_normal_variance_threshold_deg, p_coplanarity_deg, p_outlier_ratio,
    p_min_plane_edge_length, p_min_num_points,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
print("Detected {} patches".format(len(oboxes)))   # 31 patches in the book's example
struct_vol = np.sum([obox.volume() for obox in oboxes])  # 10.76 m3
```
- **What it demonstrates**: Open3D's built-in hybrid RANSAC+region-growing planar patch detector, used to both segment and quantitatively estimate occupied volume.

## Reference Tables
*Table 9-1: Limitations of RANSAC*

| Limitation | Description |
|---|---|
| Sensitivity to noise | High noise/random points can cause failure to detect accurate shapes |
| Difficulty with complex shapes | Works best for simple primitives (plane, sphere, cylinder); struggles with irregular shapes |
| Parameter tuning | Distance threshold and iteration count need careful, dataset-dependent tuning |

*Table 9-2: RANSAC vs. Region Growing*

| Aspect | RANSAC | Region Growing |
|---|---|---|
| Goal | Model fitting | Segmentation |
| Approach | Top-down (fit predefined shape to data) | Bottom-up (grow from seed by local similarity) |
| Robust to | Noise and outliers | — |
| Best for | Dominant, predefined geometric primitives | Locally-consistent regions, arbitrary shape |

*Distance metrics for region growing proximity*

| Metric | Use case |
|---|---|
| Euclidean | Straight-line, isotropic distance |
| Manhattan | Grid/axis-aligned ("city block") distance |
| Mahalanobis | Covariance-aware, correlated/non-uniform features |

## Worked Example
The chapter's most complete worked example is "Application 3: RANSAC for Modeling Tasks," which builds a full scene-modeling pipeline on a synthetic point cloud: (1) load `pcd_synthetic.ply`; (2) compute an adaptive threshold from k-d-tree nearest-neighbor statistics (`min(nearest_dist_mean) + std(nearest_dist_mean)`); (3) define both `ransac_plane()` and `ransac_sphere()` (the sphere variant solves a 3×3 linear system from 4 sampled points for center and radius); (4) run both to detect one plane and one sphere, printing their equations; (5) segment the point cloud into plane-inlier, sphere-inlier, and remaining-outlier subsets (using `reduce(np.union1d, ...)` plus Boolean-masking caution for large clouds); (6) generate actual 3D meshes for the detected shapes via `create_plane_mesh()` (grid of vertices + triangulated faces) and `create_sphere_mesh()` (UV-sphere parametrization); (7) refine the plane mesh with `create_oriented_plane_mesh()`, which projects inlier points onto the plane and uses PCA/SVD to find the two in-plane principal directions, producing a mesh that tightly bounds the actual data extent rather than an arbitrary axis-aligned grid; (8) visualize the full composite scene (point clouds + meshes). This demonstrates the complete arc from raw noisy point cloud to clean, parametrically reconstructed 3D geometric model — combining RANSAC fitting, PCA-based orientation, and procedural mesh generation.

## Key Takeaways
1. RANSAC's 4-step loop (sample → fit → count inliers → repeat/keep best) robustly detects simple geometric primitives even in heavily noisy data.
2. Compute distance/iteration thresholds adaptively from local nearest-neighbor statistics instead of hardcoding them — this makes the same code work across point clouds of wildly different densities.
3. Region growing's quality hinges on seed choice and the distance/angle threshold balance between oversegmentation and undersegmentation.
4. RANSAC answers "what known shape is here?" while region growing answers "what's locally consistent here?" — use RANSAC first for robust primitive detection, then region growing to refine within detected regions (the hybrid pattern).
5. Open3D's `detect_planar_patches` operationalizes the RANSAC+region-growing hybrid directly, returning oriented bounding boxes whose summed volume gives a quantitative occupied-space estimate.
6. Use Boolean masking, not `np.delete`, when removing indices from large point arrays.
7. PCA/SVD on inlier points (not just the raw plane equation) gives a tightly-oriented mesh that better represents the true geometric extent of a detected shape.

## Connects To
- **Ch 4, 6, 8**: reuses k-d trees (Ch. 4), PCA (Ch. 6), and point-to-plane distance / deviation analysis (Ch. 8) as building blocks for both RANSAC and region growing here.
- **Ch 10**: 3D modeling (next chapter) builds directly on the shape-detection-to-mesh-generation pipeline demonstrated in "Application 3."
- **Ch 12**: notes that combining region growing with clustering/graph-based segmentation is the subject of that later chapter.
- **Ch 14**: flags that richer similarity metrics (beyond geometric normals/distance) via machine learning are covered there.
- **Open3D (`detect_planar_patches`, KDTreeFlann), SciPy/scikit-learn (KDTree)**: the external libraries central to both the from-scratch and built-in shape-detection implementations.
