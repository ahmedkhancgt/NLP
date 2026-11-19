# Cheatsheet: 3D Data Science Decision Aid

## Which data representation should I use?
- Raw sensor capture, fast manipulation, canonical hub for conversion → **point cloud**
- Need surface/visualization fidelity, reduced point count via topology → **mesh** (triangle mesh)
- Volumetric simulation, 3D CNN input, physics/occupancy → **voxel grid**
- CAD/engineering, parametric editability, precise geometry → **parametric model / B-rep**
- Training-data-abundant deep learning, RGB+depth needed → **RGB-D image**
- Applying 2D vision models without 3D training → **project to 2D (ortho/perspective/spherical) and map back**

## Which 3D structuring data structure?
- Flexible, nonuniform data, memory-efficient sparse regions, fast build → **k-d tree**
- Uniform/voxel-aligned data, range-query pruning, LoD → **octree**
- Ray tracing, collision detection, dynamic/deformable scenes → **BVH**
- Rule of thumb: k-d tree builds ~4x faster than octree on the same data (book benchmark: 5s vs 23s for 10M points) — default to k-d tree unless you specifically need octree's uniform-grid properties.

## Which meshing algorithm?
- Clean, well-sampled point data, need efficient explicit surface → **Ball-Pivoting Algorithm (BPA)**
- Noisy/incomplete scan, need watertight result → **Poisson surface reconstruction** (filter low-density vertices; crop to bbox)
- Mixed scene (flat + organic) → **hybrid**: planar segmentation for flat surfaces + Poisson/alpha-shape for the rest
- Massing/footprint-level speed over detail → **alpha shape** (tune α per scale: ~0.5 for 2D footprint, ~20 for 3D volume — recalibrate per use)
- Volumetric/isosurface data → **marching cubes**
- Need maximal-angle, FEA-quality triangles → **Delaunay triangulation**

## Which clustering/segmentation algorithm?
| Situation | Algorithm |
|---|---|
| Known cluster count, compact/spherical clusters | k-means (use elbow method for k) |
| Unknown cluster count, arbitrary shapes, noisy | DBSCAN (derive eps from local point spacing × ~10) |
| "Objects shouldn't touch" / connectivity matters | Graph + connected components |
| Dominant planar structures amid noise | RANSAC (iterate + DBSCAN-refine each plane's inliers) |
| Organic/irregular shapes, no predefined model | Region growing (seed + distance + normal-angle criteria) |
| Zero-shot, no 3D training data available | SAM on 2D projection, map back to 3D |
| Need per-point semantic labels, have labeled data | Random forest (classical) or PointNet/Point Transformer (deep) |

k-means vs DBSCAN tie-break: if you know the true object count and objects are compact/similar-sized (e.g., N known cars), k-means can beat DBSCAN even though DBSCAN is generally more flexible — test both.

## Which distance metric?
- Point sets, isotropic space → **Euclidean**
- Grid/voxel coordinates → **Manhattan (L1)**
- Non-congruent shape comparison → **Hausdorff**
- Non-rigid curves/surfaces → **Fréchet**
- Correlated/non-uniform feature space (region growing) → **Mahalanobis**

## Which bounding volume?
- Need speed, rough extent only → **AABB** (axis-aligned bounding box)
- Need accurate volume/shape for a rotated object → **OBB** (oriented bounding box, via convex hull) — book example: AABB overestimated room volume by ~27% vs OBB (605.82 m³ vs 475.83 m³)

## Which normalization for a feature?
- Bounded data, no major outliers, spatial coordinates → **min-max scaling** ([0,1])
- Unbounded, normally distributed, has outliers → **z-score standardization**
- Shape-preserving uniform scale → **unit sphere normalization**
- Nonuniform point distributions → **local normalization**
- Dimensionality reduction + noise removal → **PCA normalization**
- Skewed sensor feature (e.g., LiDAR intensity) → **robust (IQR/median-based) scaling**, never naive min-max

## 3D deep learning architecture by representation
| Data representation | Architecture |
|---|---|
| Voxel grid | 3D CNN |
| Raw point cloud | PointNet / PointNet++ / Point Transformer |
| Mesh (graph-structured) | GNN |
| Multiview 2D renders | MVCNN (can reuse pretrained 2D backbone, e.g. ResNet18) |

## 3D Machine Learning vs. 3D Deep Learning — which to reach for?
| Use classical ML (Ch 14) when | Use deep learning when |
|---|---|
| Small dataset | Large dataset available |
| Interpretability required | State-of-the-art performance is the goal |
| Fast deployment needed | Complex, nonlinear patterns |
| Simple, well-defined features | Automatic feature learning desired |

## Tells and smells
- **98% test accuracy, same-distribution split** → not evidence of real-world generalization. Validate against a genuinely different dataset (different city/sensor/time) before trusting it.
- **Overall accuracy looks fine but per-class F1 is wildly uneven** → class imbalance masking a weak class; always check per-class precision/recall/F1.
- **Polygon area wildly wrong from manually-picked 3D points** → forgot to sort corners by centroid angle before computing area (self-intersecting polygon).
- **RANSAC-detected planes have spatially-incoherent "line" artifacts slicing through them** → naive iterative RANSAC has no spatial-contiguity concept; refine with DBSCAN on each plane's inliers.
- **Height/elevation measurement is clearly wrong for a segmented object** → the segment itself may lack ground-level points (occlusion); borrow nearby ground-classified points for a reliable reference instead.
- **Voxel mesh exports in the wrong position/scale** → forgot the translate(0.5)→scale(voxel_size)→translate(grid_origin) correction to undo the arbitrary voxel grid coordinate system.
- **A trained classifier confidently labels an obviously out-of-domain object** → models always output *some* class; never assume reliability without verifying domain coverage.
- **k-d tree query returns distance 0 for every point's first neighbor** → you queried the cloud against itself; filter out the first column before computing mean neighbor distance.

## Defaults the author commits to
- Statistical outlier removal: `nb_neighbors=16-20`, `std_ratio` 2-10 (lower = more aggressive)
- RANSAC: `distance_threshold` ≈ 10× the mean point spacing (empirical); 1000 iterations is a common default
- DBSCAN: `eps` ≈ 10× the RANSAC distance threshold when chained after plane detection
- Region growing: angle_threshold ≈ π/6 (30°), distance_threshold tuned to point spacing
- PointNet input tile size: 4,096 points per tile (original paper's recommendation)
- Voxel size: ~0.5% of the point cloud's max bounding-box edge (empirical starting point)
- k-NN feature extraction: k=8–20 neighbors for local geometric features (planarity/verticality/etc.)
- Poisson reconstruction depth: 8 default; 5–7 for noisy data (smoother, less detail); higher depth = more detail but bakes in noise
