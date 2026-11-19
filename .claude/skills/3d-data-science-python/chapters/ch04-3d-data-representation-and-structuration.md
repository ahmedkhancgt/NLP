# Chapter 4: 3D Data Representation and Structuration

## Core Idea
3D data can be represented in many forms (point clouds, meshes, voxels, depth maps, parametric models, graphs/SDFs), but point clouds act as the **canonical link** between all of them, and dedicated 3D data structures (k-d trees, octrees, BVH) are what make processing any of these representations computationally efficient.

## Frameworks Introduced
- **The 3D Data Representation Spectrum**: raw sensor-near data (point clouds, depth maps) on one end, surface/solid models (meshes, parametric B-reps) on the other, with raster/projection and volumetric (voxel) representations occupying intermediate ground.
  - When to use: to choose the right representation for a task (rendering speed vs. memory vs. topological fidelity vs. CAD-editability).
  - How: pick point clouds for raw capture/fast manipulation, meshes for surface-accurate visualization, voxels for volumetric simulation, depth/RGB-D for deep-learning-scale training data, parametric/B-rep for CAD/engineering editability.
- **Point Cloud as Canonical Data Link**: every other 3D representation (mesh, voxel, raster/depth map) can be converted to and from a point cloud, making point clouds the hub format for a coherent multi-modal workflow.
  - When to use: when you need to integrate datasets originating in different modalities into one processing pipeline.
  - How: mesh→point cloud via surface sampling (`sample_points_uniformly`); voxel→point cloud via grid-index-to-coordinate mapping; raster/depth→point cloud via camera intrinsics/extrinsics and projection inversion.
- **3D Data Structures vs. 3D Data Representations**: representations describe/store geometry; data structures (k-d trees, octrees, BVH) organize that geometry for efficient computation — the two concerns are orthogonal.
  - When to use: whenever nearest-neighbor search, range queries, or hierarchical spatial access is needed on top of any representation.
  - How: choose k-d trees for flexible, memory-efficient, nonuniform-data search; choose octrees for uniform spatial subdivision and voxel-aligned range queries; choose BVH for ray tracing/collision detection on dynamic scenes.

## Key Concepts
- **Point cloud**: {X,Y,Z}(+attributes) set; exact but memory-heavy, fast to render/transform, limited combination operations.
- **Depth map**: single-channel raster where pixel intensity encodes distance along the camera's line of sight (2.5D — cannot represent two depths on one line of sight).
- **RGB-D image**: RGB image + depth channel; abundant in public datasets, the go-to representation for training large deep learning models.
- **Projection (orthographic/perspective/spherical)**: mapping a 3D scene into a 2D grid via a virtual camera's intrinsic (focal length, distortion) and extrinsic (position, orientation) parameters; lossy but useful for 2D-vision-style processing.
- **Multiview**: representing a 3D object via multiple 2D images from different viewpoints; reduces noise/occlusion impact but needs "enough" views to avoid overfitting to incomplete geometry.
- **Voxel**: the 3D analog of a pixel — a fixed-size cube; good for volumetric simulation and 3D CNNs, poor for precise/curved geometry (aliasing) and memory-expensive at high resolution.
- **Mesh / B-rep (boundary representation)**: vertices + edges + faces (mostly triangles) describing an object's surface/boundary rather than its volume; the most common 3D model format in graphics.
- **Parametric model**: geometry defined by adjustable parameters (e.g., wall length/height) rather than fixed vertices; supports whole shape *classes*, not just instances; common in CAD/BIM via STEP/IGES.
- **k-d tree**: binary space-partitioning tree using axis-aligned hyperplanes; fast to build, flexible for nonuniform data, memory-efficient for sparse regions.
- **Octree**: recursive subdivision into 8 equal cubic octants; slower to build than k-d trees but better for uniform spatial subdivision, voxel grids, and range-query pruning.
- **BVH (bounding volume hierarchy)**: hierarchy of bounding boxes; used for ray tracing and collision detection in dynamic/deformable scenes.
- **In-memory vs. out-of-core processing**: load entire dataset into RAM (fast, memory-bound) vs. stream it in chunks (slower, scales beyond RAM).

## Mental Models
- Think of the library/out-of-core metaphor: in-memory = "all books open on a table" (fast, limited by table size); out-of-core = "fetch chapters from the basement as needed" (slower, scales past memory).
- Use "k vs. d vs. tree" to decode k-d trees: tree = hierarchical branching structure; d = dimensionality (works for 2D/3D/k-D); k-d = binary splits via hyperplanes that balance point counts per branch.
- Treat octrees as "Lego for space": always eight equal cubic children per node, vs. k-d trees' flexible axis-aligned splits — pick octrees when your data/task is naturally grid-aligned (voxels), k-d trees when it's irregular.
- When converting between representations, think of point clouds as the "universal intermediate format" (like a hub-and-spoke integration pattern) rather than converting directly between every pair of modalities.

## Anti-patterns
- **Treating depth maps as full 3D data**: they are 2.5D — a single viewpoint cannot encode occluded geometry or multiple depths along one ray.
- **Over-sampling images for multiview reconstruction with too few actual viewpoints**: too few views risks overfitting/incomplete shape capture; too many wastes compute — aim for sufficient, well-distributed coverage.
- **Using voxels for precision-critical CAD work**: voxel modeling lacks the mathematical precision of B-rep/parametric methods and introduces aliasing on curved surfaces.
- **Choosing octrees by default for all spatial indexing**: they are slower to build than k-d trees (e.g., ~23s vs ~5s for 10M points in the book's benchmark) and are only preferable for uniform/voxel-aligned data.
- **Loading an entire massive point cloud into memory without checking size**: use out-of-core/chunked processing (pandas `chunksize`) once the dataset exceeds available RAM.

## Code Examples
```python
import open3d as o3d
pcd_o3d = o3d.io.read_point_cloud("../DATA/Seychelles-beach.ply")
o3d.visualization.draw_geometries([pcd_o3d])

# NumPy <-> Open3D bidirectional bridge
pcd_np_o3d = o3d.geometry.PointCloud()
pcd_np_o3d.points = o3d.utility.Vector3dVector(pcd_np[:, 0:3])
pcd_np_o3d.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:6] / 255)
```
- **What it demonstrates**: loading native Open3D point cloud formats and converting a raw NumPy XYZRGB array into an Open3D PointCloud object for visualization.

```python
mesh = o3d.io.read_triangle_mesh("data/bunny.ply")
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=1000)
o3d.io.write_point_cloud("output/bunny_pcd.ply", pcd)
```
- **What it demonstrates**: the canonical mesh→point cloud conversion via uniform surface sampling — the core "link" technique of the chapter.

```python
def cloud_to_image(pcd_np, resolution):
    minx, maxx = np.min(pcd_np[:, 0]), np.max(pcd_np[:, 0])
    miny, maxy = np.min(pcd_np[:, 1]), np.max(pcd_np[:, 1])
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for point in pcd_np:
        x, y, *_ = point
        r, g, b = point[-3:]
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        image[pixel_y, pixel_x] = [r, g, b]
    return image
```
- **What it demonstrates**: a from-scratch orthographic (top-down) projection function converting an unstructured point cloud into a structured raster image, preserving RGB.

```python
# k-d tree vs octree construction
pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
from scipy.spatial import cKDTree
kdtree = cKDTree(np.array(point_cloud.points))

octree = o3d.geometry.Octree(max_depth=8)
octree.convert_from_point_cloud(point_cloud)
octree.locate_leaf_node(point_cloud.points[0])
```
- **What it demonstrates**: building the two main spatial-indexing structures used throughout the rest of the book for nearest-neighbor search and hierarchical spatial queries.

## Reference Tables
*Table 4-1/4-4/4-5/4-6 (synthesized): representation trade-offs*

| Representation | Main benefit | Main drawback |
|---|---|---|
| Point cloud | Exact, fast render/transform | High memory, limited combination ops |
| Depth map | Low memory, simple transforms | 2.5D only, weak topology |
| Voxel | Great for volumetric sim/3D CNNs | Imprecise, aliasing, memory-heavy at high res |
| Mesh/B-rep | Well-adopted, reduces point count via topology | High memory, expensive combinations, curves approximated |
| Parametric/B-rep (CAD) | Precise, editable by parameter, class-level shapes | Limited native Python support, large file sizes (.step/.iges) |

*Common file formats and Python libraries*

| Format | Modality | Python library |
|---|---|---|
| .ply | Point cloud / mesh | open3d, pyvista |
| .las/.laz | Point cloud (LiDAR) | laspy |
| .xyz | Point cloud (simple text) | numpy, pandas |
| .stl | Mesh (3D printing) | meshio, open3d |
| .off | Mesh (academic) | trimesh, open3d |
| .step/.stp | Parametric/CAD | FreeCAD, OCPNet |

*k-d tree vs. octree*

| Aspect | k-d tree | Octree |
|---|---|---|
| Partition rule | Axis-aligned hyperplane, balances point count | Always 8 equal cubic octants |
| Build speed | Faster (~5s/10M pts in book's benchmark) | Slower (~23s/10M pts) |
| Best for | Nonuniform data, memory-efficient sparse regions | Uniform data, voxel grids, range-query pruning |

## Worked Example
The chapter builds a full raster-to-point-cloud spherical (equirectangular) projection pipeline: (1) load a `.las` point cloud with laspy and convert XYZ + 16-bit RGB to a NumPy array; (2) pick a `center_coordinates` (virtual scan station position) and a target `resolution_y`; (3) translate all points relative to the center, then convert to spherical coordinates (`theta = arctan2(y,x)`, `phi = arccos(z/norm)`); (4) map `(theta, phi)` to pixel coordinates `(x, y)` on a `resolution_y × 2·resolution_y` image grid; (5) resolve multiple points mapping to the same pixel by keeping the nearest one (`mapping` array tracks the closest point's index per pixel via `np.linalg.norm` comparison); (6) assign that point's RGB color to the pixel. The result is an equirectangular ("360° scan station") image exported with Matplotlib — a technique the author flags as feeding directly into Chapter 13's deep-learning-on-projected-images workflow. This demonstrates the general recipe for raster↔point-cloud conversion: define a projection geometry (ortho / perspective / spherical), map 3D→2D with a resolvable collision rule, then invert for reconstruction.

## Key Takeaways
1. Point clouds are the canonical hub representation — convert other modalities (mesh, voxel, raster) to/from point clouds to unify a multi-modal pipeline.
2. Depth maps/RGB-D are 2.5D, not full 3D, but their abundance makes them ideal for deep-learning training data.
3. Meshes reduce point count via topology (faces/edges) but approximate curved geometry; parametric/B-rep models trade Python-native support for CAD-grade precision.
4. Voxels unlock volumetric simulation and 3D CNN processing but cost memory and introduce aliasing.
5. Pick k-d trees for flexible, nonuniform spatial search; pick octrees for uniform/voxel-aligned range queries; pick BVH for ray tracing/collision detection.
6. Use out-of-core (chunked pandas) processing once a dataset exceeds available RAM rather than forcing an in-memory load.
7. Orthographic, perspective, and spherical projections are reversible bridges between unstructured 3D point data and structured 2D raster representations — useful for applying 2D image-processing/deep-learning techniques to 3D data.

## Connects To
- **Ch 3**: extends the basic point-cloud loading/visualization patterns (NumPy, pandas, Open3D) introduced there.
- **Ch 5**: the multimodal 3D viewer built next directly reuses these representations and data structures.
- **Ch 10**: deepens voxels, constructive solid geometry, and advanced parametric modeling.
- **Ch 13**: reuses spherical-projection-derived images for graph/foundation-model-based unsupervised segmentation.
- **Ch 16**: reuses voxel grid structure for 3D CNN-based classification (PointNet chapter).
- **Open3D, laspy, NetworkX, SciPy (cKDTree)**: the external libraries exercised for representation conversion and spatial indexing in this chapter.
