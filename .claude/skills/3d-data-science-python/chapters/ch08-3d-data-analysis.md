# Chapter 8: 3D Data Analysis

## Core Idea
3D data analysis is not a single processing module but a cross-cutting discipline — descriptive, exploratory, predictive, and prescriptive — that combines statistical, geometric, and visual techniques to extract and communicate insight from 3D datasets, grounded in concrete tools: bounding boxes/volumes, PCA-derived orientation/aspect ratio, statistical distributions, and point-to-plane/point-to-mesh deviation analysis.

## Frameworks Introduced
- **The Four Buckets of 3D Data Analysis**: Descriptive (what happened — summarize data), Exploratory (why did it happen — geometric/spatial pattern investigation, deformation/clash detection), Predictive (what will happen — feature-engineering-fed models), Prescriptive (what should be done — decision support/optimization).
  - When to use: to classify any 3D analytics task and pick appropriate tools/depth.
  - How: progress roughly in order — DDA informs cleaning/preprocessing, EDA diagnoses deviations/patterns, predictive builds forecasting models (deferred to Ch. 12–18), prescriptive drives decisions/actions.
- **The Five-Step 3D Descriptive Analysis Workflow**: Environment & Data Preparation → Metadata Analysis/Profiling → Geometry & Shape Analysis → Statistical Analysis → Attribute Analysis.
  - When to use: as the concrete, repeatable recipe for characterizing any new 3D dataset before deeper processing.
  - How: profile with pandas (`.info()`, `.shape`, `.head()`, `.describe()`) → compute bounding boxes/volume/orientation/aspect ratio → compute central tendency/spread/percentiles/skew with histograms/boxplots/KDE/correlation heatmaps → analyze RGB/normals/eigenvalues attributes.
- **3D Deviation Analysis (Diagnostic Pattern)**: compute per-point distance from a dataset to a reference geometry (plane or mesh), color-code the result, and histogram the distances to diagnose fit/leveling/design deviation.
  - When to use: monitoring/clash detection — is the floor level? does a manufactured part match its CAD model? does a built structure match its BIM?
  - How: plane case — per-point dot product of (point − plane vertex) with the plane's normal; mesh case — sample the mesh into a dense point cloud, build a k-d tree, and do nearest-neighbor distance search per point.

## Key Concepts
- **AABB (axis-aligned bounding box) vs. OBB (oriented bounding box)**: AABB is faster to compute/test but less accurate/tight; OBB (via convex hull + per-triangle-frame minimal box) gives a tighter true volume but costs more to compute — trade-off between speed and accuracy.
- **Convex hull**: the minimal enclosing polyhedron of a point set; underlies the oriented-bounding-box computation and gives a volume estimate.
- **Aspect ratio (L/h, l/h, L/l from OBB extent)**: a compact shape signature useful for automated classification (e.g., distinguishing a car's elongated OBB from a plane's).
- **Central tendency & spread statistics**: mean, median, standard deviation, percentiles, skewness, mode — median is more robust to outliers than mean; mean≈median suggests symmetric distribution.
- **Distance metrics for shape comparison**: Euclidean (point sets), Manhattan/L1 (grid/voxel data), Hausdorff (max of min distances, for non-congruent shapes), Fréchet (curves/surfaces, non-rigid comparison).
- **Normalization techniques (recap + expansion)**: min-max scaling (bounded data, no outliers), z-score (unbounded data with outliers), unit sphere normalization (shape-preserving), local normalization (nonuniform distributions), PCA normalization (dimensionality reduction + noise removal).
- **KDE (kernel density estimation) plot**: smooth, continuous alternative to histograms for visualizing feature distributions, with area under the curve summing to 1.
- **Correlation heatmap**: pairwise `df.corr()` visualized via Seaborn to detect redundant/derived features (e.g., Z strongly correlating with verticality, since one derives from the other).

## Mental Models
- Treat 3D data analysis like telling a story with few bullet points — it resists clean decomposition into isolated components because it draws on preprocessing, segmentation, and modeling simultaneously; the four-bucket typology is a map, not a strict pipeline.
- Use the "car aerodynamics" running analogy: capture → preprocess (remove outliers) → segment (keep relevant points) → model (fit refined mesh) → monitor deviation (against as-designed model) → simulate — this is the shape of most real 3D analytical workflows.
- When picking a bounding-volume representation, think speed (AABB) vs. accuracy (OBB via convex hull) as a dial, not an either/or.
- Treat deviation histograms as dual-purpose: a QA/diagnostic signal (how good is the fit?) AND a segmentation/thresholding mechanism (use the distance value to split data into two classes, e.g., ground vs. non-ground).

## Anti-patterns
- **Reporting only an axis-aligned bounding box volume for an arbitrarily-oriented object**: AABB volume overestimates true extent for rotated objects — use the minimal oriented bounding box when accuracy matters (book's example: 605.82 m³ AABB vs. 475.83 m³ OBB for the same room).
- **Skipping normalization before comparing/visualizing a feature with large outlier-driven range**: e.g., mean curvature with max=50.16 vs. typical small values signals a clear need for min-max or z-score scaling before further analysis.
- **Treating derived/correlated features as independent inputs**: a correlation heatmap revealing that eigenvalue-3 and verticality are highly correlated (because one derives from the other) means you should drop redundant derived features before downstream modeling.
- **Hardcoding mesh-sampling density without considering its trade-off**: too few sampled points from a mesh (for distance-to-mesh analysis) causes imprecision; too many causes needless performance cost — tune deliberately, don't default blindly.
- **Relying on a single distance metric regardless of data type**: using Euclidean distance on grid/voxel data or non-congruent shapes, where Manhattan or Hausdorff would be more appropriate, misrepresents the comparison.

## Code Examples
```python
import open3d as o3d
xyz_o3d = o3d.geometry.PointCloud()
xyz_o3d.points = o3d.utility.Vector3dVector(np.array(df_pcd[['X','Y','Z']]))

aabb = xyz_o3d.get_axis_aligned_bounding_box(); aabb.volume()      # 605.82 m3
oobb = xyz_o3d.get_minimal_oriented_bounding_box(); oobb.volume()  # 475.83 m3
L, l, h = oobb.extent
ratio_Lh, ratio_lh, ratio_Ll = [L/h, l/h, L/l]   # aspect-ratio shape signature
```
- **What it demonstrates**: computing both bounding-box types and deriving an aspect-ratio "shape signature" usable for automated classification (e.g., car vs. plane).

```python
def compute_distance_to_plane(point_cloud_file, plane_file):
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    plane_mesh = o3d.io.read_triangle_mesh(plane_file)
    distances = []
    for point in point_cloud.points:
        min_distance = np.inf
        for triangle in plane_mesh.triangles:
            v0, v1, v2 = [plane_mesh.vertices[i] for i in triangle]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)
            distance = np.abs(np.dot(point - v0, normal))
            min_distance = min(min_distance, distance)
        distances.append(min_distance)
    return point_cloud, plane_mesh, np.array(distances)
```
- **What it demonstrates**: point-to-plane deviation analysis via the dot-product-with-normal method — the base technique for leveling/clash-detection diagnostics.

```python
def compute_distance_to_mesh(point_cloud_file, mesh_file):
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    pcd_mesh = mesh.sample_points_uniformly(number_of_points=1000000)
    mesh_tree = o3d.geometry.KDTreeFlann(pcd_mesh)
    distances = []
    for point in point_cloud.points:
        _, inds, dist = mesh_tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))
    return point_cloud, mesh, np.array(distances)
```
- **What it demonstrates**: point-to-mesh deviation analysis — sample the mesh to a dense point cloud, build a k-d tree, and do nearest-neighbor distance lookups (the generalized version of registration-style fit assessment, applied for diagnostics instead of alignment).

```python
nn_distances = xyz_o3d.compute_nearest_neighbor_distance()
mean_dist, median_dist = np.mean(nn_distances), np.median(nn_distances)
std_dist = np.std(nn_distances)
skew_dist = sp.stats.skew(nn_distances)
corr = df_pcd.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 20, as_cmap=True))
```
- **What it demonstrates**: the core statistical-analysis toolkit — nearest-neighbor distance as a density proxy, central tendency/spread/skew, and correlation heatmap for redundant-feature detection.

## Reference Tables
*Distance metrics for shape comparison*

| Metric | Best for | Notes |
|---|---|---|
| Euclidean | Point sets in Euclidean space (point clouds) | Straight-line distance |
| Manhattan (L1) | Grid/voxel-coordinate data | Sum of absolute coordinate differences |
| Hausdorff | Non-congruent shape comparison | Max over min point-to-point distances |
| Fréchet | Non-rigid curves/surfaces | Shortest path within a bound of both curves |

*Normalization techniques (recap)*

| Technique | Best for |
|---|---|
| Min-max scaling | Bounded data, no major outliers |
| Z-score | Unbounded data, has outliers |
| Unit sphere | Shape-preserving scale normalization |
| Local normalization | Nonuniform point distributions |
| PCA normalization | Dimensionality reduction + noise removal |

*Example AABB vs. OBB result (book's room dataset)*

| Box type | Volume |
|---|---|
| AABB | 605.82 m³ |
| OBB (minimal) | 475.83 m³ |

## Worked Example
The chapter's two deviation-analysis case studies form the central worked examples. (1) **Planar case**: load an office point cloud (`office.ply`) and a reference `ground_plane.ply`, compute per-point perpendicular distance to the plane via the normal-dot-product method, color points by distance (cool colormap), export a distance histogram — result: a primary mode at distance≈0, confirming the floor is well leveled, with the histogram itself usable as a ground/non-ground segmentation threshold. (2) **Mesh case**: given a manufactured car point cloud (`pcd_car.ply`) and its CAD reference mesh (`mesh_car.ply`), sample the mesh into 1M points, build a k-d tree, and compute nearest-neighbor distance per point of the real scan to the ideal mesh, visualize with the jet colormap and histogram — result: a visible mean/median/mode shift indicating the manufactured part deviates from its as-designed model, directly surfacing a quality/design issue. Both examples share the same three-stage shape (load data → compute per-point distance via geometry/k-d tree → color + histogram the result), establishing deviation analysis as a reusable diagnostic pattern. A supporting worked example walks the full five-step DDA workflow on a 500,000-point wool-factory dataset (20 extracted features): profile with pandas, compute AABB/OBB/volume/centroid, visualize planarity and (post-min-max-scaled) mean curvature as point colors, extract main orientation via PCA, compute aspect ratios on car vs. plane meshes, then run statistical analysis (nearest-neighbor-distance mean/median/std/percentiles/skew/mode), histogram/boxplot/KDE visualizations, and a correlation heatmap revealing which engineered features are redundant.

## Key Takeaways
1. Classify any 3D analytics task into descriptive/exploratory/predictive/prescriptive to pick the right tools and depth.
2. Use OBB (not AABB) when volume/shape accuracy matters; AABB is faster but less precise.
3. Aspect ratios derived from an OBB's extent are a cheap, effective shape signature for automated classification.
4. Deviation analysis (point-to-plane via normal dot product; point-to-mesh via sampled-mesh k-d tree nearest-neighbor) is the general-purpose tool for leveling checks, clash detection, and manufacturing/BIM QA.
5. Nearest-neighbor distance doubles as both a density proxy and the basis for central-tendency/spread statistics.
6. Correlation heatmaps are essential for spotting and removing redundant/derived features before further modeling.
7. Choose the distance metric (Euclidean/Manhattan/Hausdorff/Fréchet) to match the data type and rigidity assumption of the comparison.

## Connects To
- **Ch 4, 6, 7**: reuses point cloud/mesh representations, k-d trees, and PCA/eigenvalue feature extraction established in those chapters.
- **Ch 6**: deviation-to-mesh analysis is explicitly tied back to the registration techniques covered there.
- **Ch 9**: 3D shape recognition picks up directly where this chapter's diagnostic/geometric-decomposition tools leave off.
- **Ch 12, 13, 14**: 3D predictive analysis (segmentation, clustering, supervised AI) is deferred to these chapters to avoid overlap.
- **Open3D, NumPy/SciPy, Matplotlib, Seaborn**: the core library stack used throughout this chapter's statistical and geometric analysis tooling.
