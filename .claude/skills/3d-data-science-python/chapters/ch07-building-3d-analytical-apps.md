# Chapter 7: Building 3D Analytical Apps

## Core Idea
Build a fully interactive 3D feature-extraction-and-thresholding application in Python using PyVista + SciPy KDTree + NumPy — turning static point clouds into real-time, explorable datasets where geometric features (planarity, linearity, omnivariance, verticality, elevation differences) can be computed, visualized, and interactively thresholded to segment/label data.

## Frameworks Introduced
- **The Nine-Step App-Building Workflow**: gather dataset → set up Python environment → 3D data I/O with PyVista → k-d tree data structure creation → PCA/covariance feature computation → derive geometric features (planarity/linearity/omnivariance/verticality) → neighborhood definition/selection → automate/scale across the full point cloud → interactive thresholding → export results.
  - When to use: as the master template for any point-cloud feature-extraction application meant to support interactive labeling/segmentation.
  - How: follow the stages in order — each stage's output (k-d tree, features array) is the next stage's required input.
- **Weinmann-style Geometric Feature Set**: planarity, linearity, omnivariance, verticality (+ normal vector) computed from sorted PCA eigenvalues/eigenvectors of a point's local neighborhood.
  - When to use: when you need interpretable, hand-engineered per-point descriptors for segmentation/labeling without training a model.
  - How: `planarity=(λ1-λ2)/λ0`, `linearity=(λ0-λ1)/λ0`, `omnivariance=(λ0·λ1·λ2)^(1/3)`, `verticality=1-normal_z`, where λ0≥λ1≥λ2 are sorted eigenvalues of the local covariance matrix.
- **Interactive Thresholding Pattern**: embed computed scalar features into the point cloud object, open a `pv.Plotter()` with `add_mesh_threshold()` bound to a chosen scalar, interactively slide the threshold, then persist the picked selection for further filtering or export — chainable across multiple features.
  - When to use: for rapid, GUI-driven dataset labeling/segmentation without writing custom filtering code per iteration.
  - How: assign feature arrays as named scalars on the PyVista object → `add_mesh_threshold(scalars=<feature>)` → visually narrow by sliding the threshold → press P to pick/persist the remaining selection → repeat with a second feature on the picked subset → export.

## Key Concepts
- **PolyData (PyVista)**: PyVista's core 3D data container type, flexible for point clouds/meshes, supporting named scalar fields attached per point.
- **Eye dome lighting (EDL)**: a PyVista rendering enhancement that improves depth perception for point cloud visualization without requiring normals/meshing.
- **k-NN search vs. radius search vs. hybrid 2D search**: k-NN (`tree.query(k=20)`) fixes neighbor count; radius search (`query_ball_point`) fixes a distance threshold and returns variable-count neighbor lists; a "knowledge-driven" 2D (x,y-only) k-d tree radius search is tailored for top-down LiDAR data to efficiently find local elevation maxima/minima.
- **d_high / d_low**: vertical distance from a point to the highest/lowest point in its local (2D radius-based) neighborhood — a simple, effective terrain-relative elevation feature.
- **Structured NumPy array**: a single `np.empty(len, dtype=dt)` array with named fields (`dt = {'names': [...], 'formats': [...]}`) used as a lightweight, dependency-free feature matrix alternative to pandas.
- **Scene/mesh picking**: PyVista's interactive capability (press "P") to capture the subset of points remaining after a threshold operation, enabling chained, pipeline-style interactive filtering.

## Mental Models
- Treat feature engineering as "distilling shape" — planarity/linearity/omnivariance/verticality each answer "is this neighborhood flat, elongated, scattered-but-uniform, or vertical?" and combinations of these answer "ground vs. wall vs. pole vs. tree."
- Use "neighborhood definition is a knob, not a fixed choice" — k-NN=20 might be suboptimal; radius-based or 2D-restricted searches can yield qualitatively better features for the same underlying geometry (the chapter explicitly notes verticality/planarity results hint that 20 neighbors "might not be optimal").
- Think of interactive thresholding as a GUI-driven precursor to a manual-labeling pipeline: explore a feature's histogram visually, isolate a value range, persist the subset, and iterate with another feature on that subset — a chainable segmentation funnel.
- Treat PyVista + SciPy KDTree + NumPy as a "minimal dependency" 3D app stack — deliberately avoiding pandas/Open3D redundancy to keep the toolchain lean and understand each library's distinct contribution.

## Anti-patterns
- **Using PyVista's `find_closest_point()` in a loop over all points**: it only processes one query point at a time, making it impractical at scale (500K+ points) — build a single `KDTree` and batch-query with `tree.query(points, k=20)` instead.
- **Assuming a fixed k (e.g., k=20) is always the right neighborhood size**: the chapter's own planarity/verticality results show visible noise/ambiguity, suggesting the neighbor count or search type (radius vs. k-NN vs. 2D-restricted) needs tuning per dataset.
- **Treating verticality and planarity as redundant**: they overlap in some regions but verticality uniquely isolates ground points — don't drop one feature assuming the other covers it.
- **Skipping timing instrumentation on expensive full-point-cloud loops**: feature computation over 500K points takes ~2 minutes; always time such loops to catch performance regressions or justify offline-batch vs. real-time design choices.

## Code Examples
```python
import pyvista as pv
from scipy.spatial import KDTree
import numpy as np

pcd_pv = pv.read('../DATA/structure_verviers.ply')
pcd_pv.plot(eye_dome_lighting=True, rgb=True)

tree = KDTree(pcd_pv.points)
lists, indices = tree.query(pcd_pv.points, k=20)   # batched k-NN for ALL points
neighbors = pcd_pv.points[indices]
```
- **What it demonstrates**: loading/visualizing a point cloud with PyVista and building a batched k-d tree query (avoiding PyVista's single-point-only `find_closest_point`).

```python
def PCA(cloud):
    mean = np.mean(cloud, axis=0)
    centered_data = cloud - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_index = np.argsort(eigen_values)[::-1]
    return eigen_values[sorted_index], eigen_vectors[:, sorted_index]

def PCA_featuring(val, vec):
    planarity = (val[1] - val[2]) / val[0]
    linearity = (val[0] - val[1]) / val[0]
    omnivariance = (val[0] * val[1] * val[2]) ** (1/3)
    _, _, normal = vec
    verticality = 1 - normal[2]
    return planarity, linearity, omnivariance, verticality, normal[0], normal[1], normal[2]
```
- **What it demonstrates**: deriving the four interpretable geometric descriptors (planarity, linearity, omnivariance, verticality) from a point's local PCA eigenvalues/eigenvectors.

```python
p = pv.Plotter()
p.add_mesh_threshold(pcd_pv, scalars='verticality', title='Verticality Pointer (%)',
                      all_scalars=True, render_points_as_spheres=True, point_size=7)
p.show()

# After interacting and pressing P to persist a selection:
selection = p.picked_mesh
pl = pv.Plotter()
_ = pl.add_mesh(selection)
pl.export_obj('../RESULTS/segmentation_live.obj')
```
- **What it demonstrates**: the full interactive-thresholding-to-export pipeline — slide a threshold on a named scalar field, persist the picked subset, export as `.obj`.

## Reference Tables
*Geometric feature formulas (λ0≥λ1≥λ2 sorted eigenvalues; normal = 3rd eigenvector)*

| Feature | Formula | Interpretation |
|---|---|---|
| Planarity | (λ1−λ2)/λ0 | High → flat/planar surface |
| Linearity | (λ0−λ1)/λ0 | High → linear/elongated structure (e.g., poles) |
| Omnivariance | (λ0·λ1·λ2)^(1/3) | High → uniform variance in all directions |
| Verticality | 1 − normal_z | High → vertical surface orientation |

*Neighborhood search strategy timings (book's ~500K-point dataset)*

| Search type | Approx. time | Notes |
|---|---|---|
| k-NN (k=20), full point cloud via k-d tree | ~16.7s | Batched query, returns fixed-count neighbors |
| Radius search (1m), full point cloud | ~15s | Returns variable-count index lists |
| 2D-restricted radius search (x,y only) | ~2 min | Knowledge-driven; good for top-down LiDAR local max/min |
| Full feature computation loop (9 features, 500K pts) | ~2 min | PCA + d_high/d_low per point |

## Worked Example
The chapter's single continuous worked example processes a point cloud of an old wool factory in Belgium to distinguish ground/walls/machinery: (1) load with PyVista, visualize with eye dome lighting; (2) build one global k-d tree and batch-query 20 nearest neighbors for every point; (3) compute per-point PCA (mean-center → covariance → eigendecomposition → sort by eigenvalue) on each point's neighbor set; (4) derive planarity/linearity/omnivariance/verticality/normal from the sorted eigenvalues/eigenvectors; (5) separately build a 2D (x,y-only) k-d tree with 1m radius search to compute `d_high`/`d_low` (vertical distance to the highest/lowest point in the local 2D neighborhood) — a terrain-relative elevation feature; (6) assemble all 9 features (planarity, linearity, omnivariance, verticality, nx, ny, nz, d_high, d_low) into one structured NumPy array, looping over all ~500K points (~2 min); (7) visualize individual features as point-cloud scalar colorings to qualitatively assess them (planarity reveals a mostly flat terrain with noisy walls); (8) launch an interactive `pv.Plotter()` with `add_mesh_threshold` on `verticality` to isolate ground points live; (9) chain a second threshold pass on `planarity` applied only to the previously picked subset; (10) export the final picked selection as `segmentation_live.obj`. This demonstrates a complete code-light, GUI-assisted segmentation/labeling pipeline built from feature engineering plus interactive filtering.

## Key Takeaways
1. Always batch k-d tree queries (`tree.query(all_points, k=...)`) rather than looping single-point lookups — the difference is the gap between seconds and intractable runtime at scale.
2. Planarity, linearity, omnivariance, and verticality (from sorted PCA eigenvalues/eigenvectors) are interpretable, well-established geometric features for distinguishing ground/wall/pole/vegetation.
3. Neighborhood definition (k-NN vs. radius vs. 2D-restricted) materially changes feature quality — tune it per application rather than defaulting to a fixed k.
4. A 2D-restricted (x,y-only) k-d tree radius search is a practical trick for top-down LiDAR elevation features (local max/min), despite being the slowest of the three search strategies shown.
5. PyVista's `add_mesh_threshold` + mesh picking enables a fast, no-code-iteration interactive segmentation/labeling loop.
6. A structured NumPy array (named-field `dtype`) is a viable lightweight alternative to pandas for a fixed, known feature schema.
7. Chaining threshold operations across multiple features (e.g., verticality then planarity) lets you progressively narrow a selection to the target class.

## Connects To
- **Ch 4, 6**: directly reuses k-d tree concepts (Ch. 4) and the PCA/covariance-eigenvalue feature-extraction machinery (Ch. 6), now applied via PyVista instead of Open3D/SciPy alone.
- **Ch 8, 9**: this chapter's interactively-derived features and thresholding approach set up the 3D data analysis and shape recognition techniques covered next.
- **Weinmann et al.**: external academic reference for the planarity/linearity/omnivariance/verticality feature formulation.
- **PyVista, SciPy (KDTree)**: the two external libraries central to this chapter's minimal-dependency app-building approach.
