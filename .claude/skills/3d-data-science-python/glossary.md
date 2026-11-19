# Glossary

**Affordance** — the perceived possible actions an object offers (e.g., a chair affords sitting) (Ch 18).

**Alpha shape (α-shape)** — a Delaunay-triangulation-derived outline/surface family controlled by parameter α; smaller α = tighter/detailed, larger α = smoother, converging to the convex hull (Ch 11, 9).

**Axis-aligned bounding box (AABB)** — fastest-to-compute bounding volume aligned to coordinate axes; less tight-fitting than an oriented bounding box (Ch 8).

**Ball-Pivoting Algorithm (BPA)** — meshing method that simulates a virtual sphere rolling over a point cloud, forming triangles where it rests on three points (Ch 10).

**B-rep (boundary representation)** — geometric model built from Vertex→Edge→Wire→Face→Shell→Solid→Compound hierarchy (Ch 4, 10).

**Centroid-based clustering** — clustering by proximity to k representative points (e.g., k-means); requires a predefined cluster count (Ch 12).

**Chamfer distance** — point-cloud-specific loss function measuring similarity between two point sets (Ch 15).

**Classification (ML)** — supervised task predicting a discrete categorical label (Ch 14).

**Cluster stability** — consistency of clustering results across repeated runs/perturbations; measured via reassignment rate or centroid movement (Ch 12).

**Confusion matrix** — table of True Positive/False Negative/True Negative/False Positive counts underlying classification metrics (Ch 14).

**Connected component** — a maximal subgraph where every node pair has a path, isolated from the rest of the graph; used as a cluster proxy (Ch 13).

**Connectivity-based clustering** — grouping points via graph connectivity (nodes=points, edges=proximity) rather than density or centroids (Ch 12, 13).

**Constructive solid geometry (CSG)** — building complex shapes via Boolean operations (union/intersection/difference) on primitives (Ch 4, 10).

**Convex hull** — minimal enclosing polyhedron of a point set; basis for oriented bounding box computation (Ch 8).

**Covariance matrix** — matrix of pairwise feature covariances; its eigendecomposition is the basis of PCA-derived 3D features (Ch 6, 7).

**CRS (Coordinate Reference System)** — the projection/datum defining how spatial coordinates map to the Earth (e.g., EPSG:26910) (Ch 11).

**DBSCAN** — Density-Based Spatial Clustering of Applications with Noise; groups points by local density, auto-detects cluster count, isolates noise (Ch 9, 11, 12).

**Degree (graph)** — number of edges connected to a given vertex (Ch 13).

**Depth map** — single-channel raster encoding per-pixel distance along the camera's line of sight; a 2.5D representation (Ch 4).

**Digital twin** — a dynamic, data-driven 3D replica of a physical object/system continuously synchronized with real-time sensor data (Ch 1).

**Eigenvalue / eigenvector** — outputs of PCA on a local point neighborhood's covariance matrix; used to derive planarity, linearity, omnivariance, verticality, and normals (Ch 6, 7, 8).

**Elbow method** — heuristic for choosing k in k-means by plotting WCSS (within-cluster sum of squares) vs. k and finding the inflection point (Ch 12).

**F1-score** — harmonic mean of precision and recall; preferred metric under class imbalance (Ch 14).

**FPFH (Fast Point Feature Histograms)** — local geometric descriptor used to find candidate correspondences during global point cloud registration (Ch 6, 17).

**Fine-tuning** — unfreezing some/all pretrained model layers to adapt them to a new (often larger/similar) target dataset (Ch 15).

**Gaussian splatting (3D)** — scene representation using explicit, oriented, colored 3D Gaussian "splats" (position, covariance, color, opacity) rendered via analytical projection — faster and more editable than NeRF (Ch 18).

**Geometric fingerprinting** — combining density-based analysis, local feature histograms, and statistical profiling into a rich multidimensional surface characterization (Ch 17).

**GNN (Graph Neural Network)** — deep learning architecture processing graph-structured data (nodes=vertices, message passing, aggregation, update functions) — suited to meshes (Ch 13, 15).

**Homogeneity (cluster)** — mean distance from a cluster's points to its centroid; lower = tighter cluster (Ch 12).

**k-d tree** — binary space-partitioning data structure using axis-aligned hyperplanes; fast nearest-neighbor/range queries, flexible for nonuniform data (Ch 4).

**k-NN (k-nearest neighbors)** — classification/feature-extraction method based on proximity to the k closest points (Ch 7, 14).

**LAS classification code** — integer label per LiDAR point denoting semantic class (e.g., 2=ground, 6=building) (Ch 11).

**Level of Detail (LoD)** — multiple simplified versions of the same 3D model for performance-scaled rendering (Ch 10).

**Max pooling (PointNet)** — symmetric, order-invariant aggregation operation that builds a global feature vector from per-point features (Ch 16).

**Mesh (triangle mesh)** — geometric structure of vertices, edges, and (usually triangular) faces describing a surface (Ch 4).

**Monocular depth estimation** — inferring a per-pixel depth map from a single 2D image using a pretrained model (e.g., GLPN, MiDaS, Depth Anything) (Ch 10, 17, 18).

**NeRF (Neural Radiance Field)** — implicit, continuous volumetric scene representation learned by a neural network, rendered via ray marching; photorealistic but slow and hard to edit (Ch 18).

**Normal vector (surface normal)** — vector perpendicular to a surface at a point; estimated via PCA on local neighborhoods or plane fitting (Ch 6, 8).

**Octree** — hierarchical spatial data structure recursively subdividing space into 8 equal cubic octants; good for uniform data and range-query pruning (Ch 4).

**Omnivariance** — PCA-eigenvalue-derived feature `(λ0·λ1·λ2)^(1/3)` indicating uniform variance in all directions (Ch 7).

**Ontology / RDF** — formal, shared vocabulary (classes, properties, instances) plus inference rules used to encode semantic priors for machines (Ch 18).

**Oriented bounding box (OBB)** — minimal-volume bounding box aligned to an object's principal axes (via convex hull); tighter but costlier than AABB (Ch 8).

**Orthographic projection** — top-down flattening of a point cloud into a 2D raster preserving dimensions, without single-viewpoint perspective distortion (Ch 4).

**Overfitting** — a model performing well on training/same-distribution data but failing to generalize to genuinely new data (Ch 14, 15).

**Permutation invariance** — property of an aggregation function (e.g., max pooling) whose output doesn't depend on input point order — essential for unordered point cloud processing (Ch 16).

**Planarity** — PCA-eigenvalue-derived feature `(λ1-λ2)/λ0` indicating flat/planar local geometry (Ch 7, 8).

**Point cloud** — a set of {X,Y,Z} coordinates (+ attributes) representing a captured environment; the canonical hub representation for 3D data conversion (Ch 1, 4).

**PointNet** — pioneering deep learning architecture processing raw point clouds directly via shared per-point MLPs and permutation-invariant max pooling (Ch 15, 16).

**Point Transformer** — self-attention-based point cloud architecture that outperforms PointNet on semantic segmentation by weighting neighbor contributions via geometric+feature similarity (Ch 18).

**Poisson surface reconstruction** — implicit meshing method building a watertight surface via octree + global optimization fit to points and normals; robust to noise (Ch 10).

**Precision / Recall** — precision = avoiding false positives; recall = finding all true positives (Ch 14).

**RANSAC (Random Sample Consensus)** — iterative model-fitting algorithm robust to outliers: sample minimal points, fit model, count inliers via threshold, repeat, keep best (Ch 9).

**Region growing** — bottom-up segmentation: start from a seed point, iteratively add neighbors meeting distance+similarity criteria until no new points qualify (Ch 9).

**RGB-D image** — an RGB image plus a depth channel; abundant and widely used for training 3D deep learning models (Ch 4).

**Semantic segmentation (3D)** — classifying every data unit (point/voxel/triangle) in a scene — contrast with instance/object detection (Ch 14, 16).

**Shap-E** — OpenAI's text-to-3D generative model using an encoder + latent diffusion producing implicit 3D functions (Ch 18).

**Silhouette coefficient** — per-point cluster-fit score in [-1,1]; closer to 1 = better cluster assignment (Ch 12).

**Spatial AI** — AI combining perception (3D reconstruction/preparation), scene understanding (semantics/topology), and reasoning (knowledge graphs) about 3D space (Ch 1, 18).

**SuperPixel / SuperPoint / SuperVoxel** — Gestalt-inspired grouping of base geometric units sharing a feature space into meaningful sub-object patches, used to accelerate labeling (Ch 17).

**Tiling** — splitting a large point cloud into fixed-size spatial chunks each resampled to a fixed point count, required for architectures like PointNet (Ch 16).

**T-Net (Transformation Network)** — PointNet submodule learning a k×k alignment matrix (3×3 input, 64×64 feature), initialized as identity, to spatially/feature-align point clouds (Ch 16).

**Transfer learning** — freezing a pretrained model's weights and training only new task-specific layers on a small target dataset (Ch 15).

**Transformer (deep learning)** — self-attention-based architecture originally for NLP, extended to 3D via Point Transformer and similar models (Ch 15, 18).

**Verticality** — PCA-derived feature `1 - normal_z` indicating vertical surface orientation; isolates ground points (Ch 7).

**VLR (Variable Length Record)** — flexible metadata container in the LAS point cloud file format, used to store CRS info (Ch 11).

**Voxel** — the 3D analog of a pixel; a fixed-size cube used for volumetric representation, simulation, and 3D CNN input (Ch 4, 10).

**Voxel grid / voxelization** — discretizing continuous/unstructured 3D data into a regular grid of occupied/attributed voxels (Ch 4, 10, 17).

**WCSS (within-cluster sum of squares)** — metric plotted against k in the elbow method to choose an optimal cluster count (Ch 12).

**Zero-shot / few-shot learning** — recognizing never-seen classes (zero-shot) or generalizing from a handful of examples (few-shot); core capability of foundation models like SAM and CLIP-based systems (Ch 13, 18).
