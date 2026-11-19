# Chapter 10: 3D Modeling: Advanced Techniques

## Core Idea
Four complementary advanced 3D modeling approaches — high-fidelity meshing (BPA/Poisson), voxelization, parametric modeling (CadQuery/B-rep/CSG), and monocular image-based 3D reconstruction (AI depth estimation) — each trade off fidelity, flexibility, and computational cost differently, and real pipelines often chain them together (point cloud → mesh → voxel → parametric edit → export).

## Frameworks Introduced
- **Four 3D Modeling Approaches**: high-fidelity meshing, voxelization, parametric modeling, image-based (monocular) 3D modeling.
  - When to use: pick based on the target use case — meshing for visualization/printing, voxelization for volumetric analysis/ML, parametric for CAD/engineering editability, image-based for accessible single-photo reconstruction.
  - How: each has its own Python toolchain (Open3D for meshing/voxels, CadQuery for parametric, Transformers+Open3D for image-based) but all can feed into or out of point clouds (the canonical link from Ch. 4).
- **Representation Comparison (Polygonal / NURBS / SDF / Boolean assemblies)**: four ways to mathematically describe 3D geometry, each suited to different domains.
  - When to use: polygonal for general-purpose visualization/interoperability; NURBS for smooth precision surfaces (auto/aero/shipbuilding); SDF for organic/implicit shapes and simulation; Boolean/CSG for combining primitives into complex industrial/architectural shapes.
  - How: SDFs and Boolean assemblies are often used internally then converted to polygonal meshes for visualization/export.
- **B-rep Construction Hierarchy (CadQuery/CAD)**: Vertex → Edge → Wire → Face → Shell → Solid → Compound, each level built from the previous.
  - When to use: whenever precise, parametrically-editable CAD geometry is needed (vs. an unstructured mesh).
  - How: define points (vertices), connect into curves (edges), group into wires, bound into faces, assemble into shells/solids, and combine solids into compounds via workplanes and Boolean operations.

## Key Concepts
- **Ball-Pivoting Algorithm (BPA)**: simulates rolling a virtual sphere of given radius over a point cloud; where it rests on 3 points it creates a triangle, and it keeps pivoting along edges to grow the mesh — good for clean, well-sampled point data.
- **Poisson surface reconstruction**: builds an octree, then globally optimizes an implicit isosurface fit to the points and their normals, producing a watertight mesh — robust to noisy/incomplete scans; controlled by `depth` (detail/noise sensitivity trade-off), `width`, `scale`, `linear_fit`.
- **Mesh simplification/cleanup**: `simplify_quadric_decimation(n)` reduces triangle count; `remove_degenerate_triangles/duplicated_triangles/duplicated_vertices/non_manifold_edges` clean artifacts before export.
- **Level of Detail (LoD)**: multiple simplified versions of the same mesh (different triangle-count targets) for performance-scaled rendering (closer=more detail, farther=less).
- **Voxelization pipeline**: compute bounding box → derive voxel size (e.g., 0.5% of max bbox edge) → `VoxelGrid.create_from_point_cloud` → extract individual voxels → instantiate a cube mesh per voxel, colored and positioned by `grid_index` → undo the arbitrary grid coordinate system (translate by 0.5, scale by voxel size, translate by grid origin) to recover real-world coordinates before export.
- **Workplane (CadQuery)**: a 2D reference surface in 3D space; most sketch/construction operations happen relative to the current workplane, which CadQuery then maps into world coordinates.
- **CSG (Constructive Solid Geometry) Boolean operations**: union (`|`, combine), intersection (`&`, common region), difference (`-`, subtract) — the core primitives for building complex parametric shapes from simple ones.
- **Monocular depth estimation**: inferring a per-pixel depth map from a single 2D image using a pretrained deep model (e.g., GLPN); used as the "sensor" input for 3D reconstruction without needing multiple viewpoints (contrast with photogrammetry's multiview requirement).
- **Camera intrinsics (pinhole model)**: focal length (fx, fy) and principal point (cx, cy) needed to project 2D depth-map pixels back into 3D coordinates; without ground-truth calibration, a basic assumed pinhole model is used for AI-generated/uncalibrated images.

## Mental Models
- Think of the four modeling approaches on a fidelity/flexibility/cost spectrum: meshing = accurate surface at moderate cost; voxelization = simple volumetric grid, ML/simulation-friendly but memory-heavy; parametric = infinitely editable via few parameters but limited native Python tooling; image-based = cheapest acquisition (one photo) but least geometrically certain (estimated, not measured, depth).
- Treat BPA vs. Poisson as "local, deterministic rolling" vs. "global, optimization-based implicit fitting" — BPA for clean data needing an efficient explicit surface, Poisson for noisy/incomplete data needing a robust watertight result.
- When voxelizing, remember the "undo the arbitrary grid" trilogy: translate by half a voxel (correct origin reference from corner to center) → scale by voxel size (grid units to real units) → translate by grid origin (local to global frame) — skipping any step misplaces the result.
- For monocular reconstruction, treat the pipeline as a chain of representation conversions already covered in Ch. 4: image → depth map (2.5D) → RGB-D → point cloud (via camera intrinsics) → mesh (via Poisson) — the AI model only replaces the depth-acquisition step, not the downstream geometry pipeline.

## Anti-patterns
- **Using a fixed BPA ball radius across all datasets**: density varies locally and globally; prefer adaptive radius selection from local nearest-neighbor statistics (or octree-based per-node averages).
- **Setting Poisson reconstruction depth too high on noisy data**: the algorithm cannot distinguish outliers from real geometry and will bake noise into the mesh as spurious vertices — lower depth (5–7) smooths at the cost of fine detail.
- **Skipping the Poisson mesh crop step**: raw Poisson output often extends beyond the actual point cloud extent with unwanted artifacts — always crop to the point cloud's bounding box.
- **Forgetting to invert the voxel grid's arbitrary coordinate system before export**: without the translate→scale→translate correction, the exported voxel mesh will be in the wrong position/scale relative to the original point cloud.
- **Treating an AI-estimated monocular depth map as metrically accurate without calibration**: without true camera intrinsics (focal length, principal point) from calibration or EXIF, the resulting point cloud/mesh has unreliable absolute scale and may be distorted — use a calibration pattern when precision matters.

## Code Examples
```python
distances = pcd.compute_nearest_neighbor_distance()
radius = 3 * np.mean(distances)
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector([radius, radius * 2]))
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
dec_mesh.remove_degenerate_triangles(); dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices(); dec_mesh.remove_non_manifold_edges()
```
- **What it demonstrates**: BPA meshing with data-derived radius, then the standard mesh-cleanup chain before export.

```python
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)
```
- **What it demonstrates**: Poisson reconstruction and the essential bounding-box crop step to remove extrapolation artifacts.

```python
v_size = round(max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.005, 4)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=v_size)
vox_mesh = o3d.geometry.TriangleMesh()
for v in voxel_grid.get_voxels():
    cube = o3d.geometry.TriangleMesh.create_box(width=v_size, height=v_size, depth=v_size)
    cube.paint_uniform_color(v.color)
    cube.translate(v.grid_index, relative=False)
    vox_mesh += cube
vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
vox_mesh.scale(v_size, [0, 0, 0])
vox_mesh.translate(voxel_grid.origin, relative=True)
vox_mesh.merge_close_vertices(0.0000001)
```
- **What it demonstrates**: the complete point-cloud-to-voxel-mesh pipeline, including the critical coordinate-system correction before export.

```python
import cadquery as cq
Box = cq.Workplane("XY").box(1, 1, 1, centered=(False, False, False))
Sphere = cq.Workplane("XY").sphere(1)
union_result = Box | Sphere          # combine
inter_result = Box & Sphere          # common region
diff_result  = Box - Sphere          # subtract

result = cq.Workplane("XY").box(3, 3, 0.5).edges("|Z").fillet(0.125)  # rounded edges
```
- **What it demonstrates**: CadQuery's parametric modeling via workplanes, Boolean CSG operators, and feature operations (fillet).

```python
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
inputs = feature_extractor(images=image, return_tensors="pt")
with torch.no_grad():
    predicted_depth = model(**inputs).predicted_depth

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)
pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]
```
- **What it demonstrates**: the full monocular image → AI depth estimation → RGB-D → point cloud (via assumed pinhole intrinsics) → Poisson mesh pipeline.

## Reference Tables
*Table 10-1 (paraphrased): 3D model representation comparison*

| Representation | Definition | Best for |
|---|---|---|
| Polygonal (mesh) | Vertices/edges/faces | General visualization, interoperability, printing |
| NURBS (B-rep) | Mathematical curve/surface function | Automotive, shipbuilding, aerospace (smooth precision) |
| SDF (implicit B-rep) | Distance-to-surface function | Organic shapes, fluid sim, level design |
| Boolean assemblies | Combination of primitives | Industrial design, architecture, manufacturing |

*Table 10-2: Meshing strategy comparison*

| Algorithm | Noise handling | Complexity | Typical applications |
|---|---|---|---|
| Ball Pivoting (BPA) | Moderate | O(n log n) | CAD, computer graphics |
| Poisson surface reconstruction | Good | O(n log n) | 3D scanning, medical imaging |
| Marching cubes | Moderate | O(n) | Volumetric visualization, isosurface extraction |
| Delaunay triangulation | Moderate | O(n log n) | Finite element analysis, computational geometry |
| Voronoi diagrams | Moderate | O(n log n) | Spatial partitioning, centroidal tessellation |

*B-rep construction hierarchy*: Vertex → Edge → Wire → Face → Shell → Solid → Compound

*File formats (this chapter)*: DXF (2D CAD exchange; good interoperability, not for 3D), STL (simple triangulated 3D surface; widely supported for 3D printing, no internal/material info), PLY/OBJ (mesh export from Open3D).

## Worked Example
The chapter's most complete end-to-end worked example is monocular image-based 3D reconstruction: (1) install torch/pillow/matplotlib/transformers/open3d; (2) acquire a source image (a phone photo or an AI-generated image, e.g., Stable Diffusion "proud knight" prompt); (3) load the pretrained GLPN depth model (`vinvino02/glpn-nyu`) via `GLPNImageProcessor`/`GLPNForDepthEstimation`; (4) resize the image to dimensions that are multiples of 32 (model requirement) while preserving aspect ratio; (5) run inference under `torch.no_grad()` to get `predicted_depth`; (6) postprocess — squeeze/move to CPU/scale by 1000, crop a 16px padding border to remove edge artifacts, crop the matching region from the original image; (7) build an Open3D RGBD image from the color+depth pair; (8) define an assumed pinhole camera intrinsic (`fx=fy=500`, principal point at image center) since no real calibration exists for an AI-generated image; (9) generate the raw point cloud via `create_from_rgbd_image`; (10) clean it up — remove statistical outliers (`nb_neighbors=20, std_ratio=6.0`), estimate and orient normals; (11) run Poisson reconstruction at depth=10, keep the largest connected mesh component `[0]`, apply a 180° rotation correction, and export as `.ply`. This single pipeline chains nearly every technique in the book: deep-learning depth estimation → RGB-D representation (Ch. 4) → point cloud generation via camera intrinsics (Ch. 4) → outlier removal/normals (Ch. 6) → Poisson meshing (this chapter) — demonstrating how a photo alone, with no specialized scanning hardware, becomes a clean exportable 3D mesh.

## Key Takeaways
1. Match representation to task: polygonal mesh for general use, NURBS/SDF for precision/organic shapes, Boolean/CSG for complex assemblies from primitives.
2. BPA suits clean, well-sampled point data; Poisson suits noisy/incomplete scans and produces watertight meshes — pick based on input data quality, not just preference.
3. Always clean meshes (degenerate triangles, duplicated vertices/triangles, non-manifold edges) and crop Poisson output to the original bounding box before export.
4. Voxelization requires explicitly undoing an arbitrary grid coordinate system (translate-scale-translate) to recover real-world placement before export.
5. CadQuery's workplane + B-rep + CSG model gives fully parametric, re-editable CAD geometry that raw meshes cannot offer.
6. Monocular (single-image) 3D reconstruction via pretrained depth models (GLPN, MiDaS, Depth Anything, MoGe, Depth Pro) is now a viable, accessible alternative to multiview photogrammetry, at the cost of relying on assumed rather than calibrated camera intrinsics.
7. The same Chapter-4 representation-conversion machinery (depth map → RGB-D → point cloud → mesh) underlies both classic 3D reconstruction and AI-based monocular reconstruction — the model just replaces how depth is acquired.

## Connects To
- **Ch 4**: all representation conversions here (depth map→RGB-D→point cloud→mesh, voxel↔point cloud) reuse the canonical-link machinery established there.
- **Ch 6**: outlier removal and normal estimation, reapplied in the monocular reconstruction cleanup step.
- **Ch 9**: RANSAC-detected planar patches feed directly into this chapter's mesh-generation functions (`create_plane_mesh`, `create_oriented_plane_mesh`).
- **Ch 12**: voxel-based topology/relationship analysis is deferred to the clustering chapter.
- **Ch 18**: extends photogrammetry, NeRF, and generative-AI-driven 3D dataset generation, contrasted here with monocular depth-based reconstruction.
- **Open3D (BPA, Poisson, VoxelGrid), CadQuery (Open Cascade kernel), Transformers (GLPN), PyTorch**: the core external libraries this chapter is built on.
