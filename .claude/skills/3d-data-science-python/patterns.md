# Patterns

## RANSAC Shape Detection
**When to use**: Detecting a known geometric primitive (plane, sphere, cylinder) in a noisy, outlier-laden point cloud — walls, floors, poles, balls.
**How**: Randomly sample the minimal point set needed to define the shape (3 points for a plane, 4 for a sphere) → algebraically derive model params → compute every point's distance to the model → threshold to count inliers → repeat N iterations, keep the model with the most inliers.
**Trade-offs**: Robust to heavy noise/outliers; fails on complex/irregular shapes; needs careful threshold/iteration tuning (mitigate with data-driven adaptive thresholds from nearest-neighbor statistics).

## Region Growing Segmentation
**When to use**: Segmenting a scene into locally-consistent regions when shapes aren't simple predefined primitives.
**How**: Pick a seed point → build a k-d tree → iteratively add neighbors (within distance threshold) whose normal angle to the seed is within an angle threshold → repeat until the region stops growing → start a new region from a new seed.
**Trade-offs**: Handles arbitrary/organic shapes; highly sensitive to seed choice and threshold balance (too strict = undersegmentation, too loose = oversegmentation); struggles with noise.

## Coarse-to-Fine Point Cloud Registration
**When to use**: Aligning two point clouds of the same scene from different viewpoints/times into one coordinate frame, especially with significant initial misalignment.
**How**: Voxel-downsample + estimate normals on both clouds → compute FPFH descriptors → RANSAC-based global registration for a rough initial transform → refine with point-to-plane ICP seeded by that transform.
**Trade-offs**: Global step is robust to large misalignment but imprecise; ICP alone only finds local minima and needs a decent starting alignment — never skip the coarse step.

## Multi-RANSAC + DBSCAN Refinement
**When to use**: Segmenting a complex multi-planar scene (e.g., a kitchen or room) into clean, spatially-coherent structural segments without "line" artifacts.
**How**: Loop RANSAC plane detection on the remaining point cloud → for each detected plane's inliers, run DBSCAN and keep only the largest spatially-contiguous sub-cluster → push the rest back into the remainder → after N planes, DBSCAN-cluster the final leftover points.
**Trade-offs**: Fixes naive iterative RANSAC's spatially-incoherent "lines" artifact; more compute than single-pass RANSAC; iteration count is a tunable (non-RANSAC) parameter.

## Graph-Based Euclidean Clustering (Connectivity-Based)
**When to use**: Object delineation from unstructured point clouds where connectivity/contiguity (not density or centroid proximity) best defines "object" (e.g., furniture in a scanned room).
**How**: Build a k-d-tree-accelerated radius graph (nodes=points, edges=proximity, optionally k-NN pruned) → compute connected components → each component is a candidate object/cluster.
**Trade-offs**: Captures the intuitive "objects don't touch" principle; radius choice is sensitive (derive adaptively from mean nearest-neighbor distance, e.g. 3×); doesn't assign semantic labels, only groups.

## Point Cloud Voxelization for Volume/Coverage Estimation
**When to use**: Computing occupied volume, built coverage, or converting unstructured points into a regular grid for simulation/3D-CNN input.
**How**: Compute bounding box → derive voxel size (e.g., % of max bbox edge or fixed size) → `VoxelGrid.create_from_point_cloud` → optionally binary-color points by target class before voxelizing to avoid color-averaging ambiguity → reduce to top-voxel-per-(x,y)-column for coverage metrics.
**Trade-offs**: Simplifies volumetric analysis and enables grid-based ML; must explicitly undo the arbitrary voxel grid coordinate system (translate+scale+translate) before exporting to real-world coordinates.

## Mesh Reconstruction: BPA vs. Poisson (Matched to Data Quality)
**When to use**: Converting a point cloud to a surface mesh for visualization, printing, or further modeling.
**How**: Use Ball-Pivoting Algorithm for clean, well-sampled point data (fast, explicit, local); use Poisson surface reconstruction (with density-based low-vertex filtering) for noisy/incomplete scans needing a watertight result; for mixed scenes, combine planar segmentation (flat surfaces) with Poisson/alpha-shape (organic elements).
**Trade-offs**: BPA struggles with uneven sampling/gaps; Poisson can bake noise into spurious vertices at high depth and needs bounding-box cropping to remove extrapolation artifacts.

## Deviation Analysis (Point-to-Plane / Point-to-Mesh)
**When to use**: Diagnosing leveling (is the floor flat?), clash detection, or manufacturing/BIM QA (does the as-built match as-designed?).
**How**: Plane case — per-point absolute dot product of (point − plane vertex) with the plane's unit normal. Mesh case — sample the reference mesh into a dense point cloud, build a k-d tree, and compute nearest-neighbor distance per point of the dataset being checked.
**Trade-offs**: Simple and fast; mesh-sampling density is a trade-off knob (too sparse = imprecise, too dense = slow); results double as a segmentation threshold (e.g., separate ground from non-ground via the distance histogram).

## Supervised 3D Semantic Segmentation with Generalization Checks
**When to use**: Classifying every point in a 3D dataset (ground/vegetation/building, etc.) when labeled training data is available.
**How**: Extract geometry-aware features (not just X,Y,Z,R,G,B — add planarity/verticality/omnivariance/normal change rate/number of returns) → normalize/scale → train (random forest is a strong baseline) → validate against a genuinely different dataset (different city/sensor), not just a random split of the same scene.
**Trade-offs**: Same-distribution test accuracy can be dangerously misleading (98%→54% collapse in the book's example); richer features recover generalization far more effectively than switching models.

## PointNet-Style Direct Point Cloud Deep Learning
**When to use**: Classification or semantic segmentation on raw, unordered point clouds without voxelization/meshing overhead.
**How**: Per-point shared-MLP feature extraction → T-Net-learned alignment matrices (regularized toward orthogonality in the loss) → permutation-invariant max-pooling for a global feature vector → classification head (global feature only) or segmentation head (global + per-point local features concatenated).
**Trade-offs**: Avoids voxelization/meshing information loss and cost; needs a fixed input point count per sample (tiling + resampling for large scenes); limited global context vs. newer attention-based architectures (Point Transformer).

## Monocular Image-to-3D Reconstruction
**When to use**: Generating a 3D point cloud/mesh from a single photo or AI-generated image, without multiview capture or specialized hardware.
**How**: Run a pretrained monocular depth model (GLPN/MiDaS/Depth Anything) on the image → build an RGB-D image → define camera intrinsics (calibrated if available, else an assumed pinhole model) → generate a point cloud via `create_from_rgbd_image` → clean (remove outliers, estimate normals) → mesh via Poisson.
**Trade-offs**: Far more accessible than multiview photogrammetry; absolute scale/geometry reliability depends heavily on camera intrinsics accuracy (worse for uncalibrated/AI-generated images).

## Foundation-Model (SAM) 3D Segmentation via Projection
**When to use**: Leveraging a massively pretrained 2D vision foundation model for 3D segmentation without any 3D training data.
**How**: Project the 3D point cloud to a 2D image (e.g., spherical/equirectangular, retaining a pixel→point mapping) → run SAM's automatic mask generator on the image → transfer per-pixel segmentation results back onto the corresponding 3D points via the mapping.
**Trade-offs**: Zero-shot, no 3D training needed; loses information for occluded/unevenly-sampled points (unlabeled after mapping back); encoder size (ViT-B/L/H) trades speed for quality.

## Ground-Referenced Measurement Correction
**When to use**: Computing a height (or any ground-relative quantity) for a segmented object whose own point cloud may lack reliable ground-level points (occlusion, scan geometry).
**How**: Take the object's footprint center, drop Z to the object's own minimum to form a query point → k-NN search a separate ground-classified point cloud around that query → average the nearby ground points' Z as "ground zero" → height = object max Z − ground zero.
**Trade-offs**: Corrects systematic underestimation from occluded ground; requires a separately available ground-classified point set (e.g., from LAS classification).
