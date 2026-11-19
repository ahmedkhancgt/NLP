# Chapter 5: Developing a Multimodal 3D Viewer with Python

## Core Idea
Build a single interactive Python viewer that fuses multiple 3D modalities (point cloud + mesh + 2D vector geometry + voxels) into one coherent scene, then use that fused scene to answer concrete spatial questions (POI proximity, parcel area, high/low points, built coverage) — demonstrated end-to-end through a flood-risk assessment scenario in the Netherlands.

## Frameworks Introduced
- **The Multimodal 3D Viewer Workflow (4 stages)**: (1) 3D Python and Code Setup, (2) 3D Data Curation, (3) 3D Data Preparation, (4) 3D Data Visualization → feeding into a Multimodal 3D Experience of concrete analytical tasks.
  - When to use: as the template for any project that must combine heterogeneous open datasets (LiDAR point cloud + CityGML mesh + vector boundary) into one visual/analytical tool.
  - How: set up environment → curate/download aligned datasets → clean/center/subsample → visualize combined → then layer analytical queries on top.
- **Four-Challenge Multimodal Analysis Pattern**: point-of-interest (POI) query → manual boundary selection/area computation → high/low point extraction → point cloud voxelization for built coverage.
  - When to use: as a repeatable recipe for extracting decision-relevant metrics (distance, area, elevation extremes, coverage ratio) from a fused point cloud + mesh scene.
  - How: each challenge follows input→processing pipeline→output; k-d tree search for POI, interactive vertex selection + Shapely for area, argmin/argmax over Z for extremes, VoxelGrid + binary coloring + top-voxel-per-column for coverage.
- **Binary Voxel Coloring Trick**: initialize all point colors to black [0,0,0], then recolor only the target class (e.g., building=red); after voxelization, Open3D averages colors per voxel, so a non-black voxel unambiguously indicates "contains target class," sidestepping the color-averaging ambiguity.
  - When to use: whenever you need an unambiguous binary voxel classification signal despite Open3D's default color averaging on voxelization.
  - How: zero all colors → set target-class points to a saturated color → voxelize → test `np.all(color==0)` to classify each voxel.

## Key Concepts
- **LoD (Level of Detail)**: CityGML building model granularity (LoD1.2, LoD1.3, LoD2.2) — higher LoD = more geometric detail (roof shapes, etc.).
- **Point of interest (POI) query / radius search**: using a k-d tree's `search_radius_vector_3d()` to select all points within a distance threshold of a center point.
- **Manual boundary selection**: an interactive Open3D workflow (`draw_geometries_with_vertex_selection`, Shift+click) for a human to pick parcel corner points directly from a 3D view.
- **Coordinate sorting by polar angle**: resolving a self-intersecting (invalid) polygon by sorting boundary points by their angle relative to the centroid (`arctan2`) before computing area — a necessary fix because Shapely's `Polygon.area` is only correct for a properly ordered (non-self-intersecting) point sequence.
- **Voxel column**: all voxels sharing the same (x,y) grid index across different z — used to find the single highest occupied voxel per ground cell, which is the basis for built-coverage extraction.
- **Built coverage / built ratio**: fraction of ground area covered by building footprint, computed from voxel column classification counts × voxel footprint area.

## Mental Models
- Treat the viewer as a pipeline, not a single visualization call: input (raw multimodal datasets) → processing (clean/center/subsample/query) → output (metric, filtered geometry, or saved file) — the same shape recurs across all four challenges.
- Use point clouds and meshes as complementary "layers" in one scene (like GIS layers) rather than separate viewers — translate both by the same center offset so they share one local coordinate frame and avoid rendering glitches from large absolute coordinates.
- When computing a polygon area from manually picked 3D points, think "centroid-relative angle sort" as the standard fix for ensuring a non-self-intersecting polygon.
- Think of "top voxel per column" as a 2.5D digital surface model (DSM)-style operation — collapsing a 3D voxel grid into a top-down classification map.

## Anti-patterns
- **Computing polygon area directly from unsorted manually-selected corner points**: produces a self-intersecting polygon and a wildly wrong area (43.6 m² instead of ~3,409 m² in the book's example) — always sort corners by centroid angle first.
- **Not translating datasets to a local origin before visualization**: large absolute real-world coordinates (e.g., UTM) can cause rendering precision glitches in Open3D.
- **Relying on Open3D's default voxel color averaging to classify voxel content**: averaging blends multiple classes' colors ambiguously — use the binary black/non-black trick instead.
- **Treating `get_max_bound()`/`get_min_bound()` as sufficient for "find highest/lowest point"**: these return bounding coordinates, not point identities/indexes — use `argmax`/`argmin` on the Z column to get the actual point index.
- **Ignoring noise when finding high/low points**: unfiltered outliers skew high/low point detection (shown visually in the chapter — filtered vs. unfiltered results differ); proper outlier removal (Ch. 6) is a prerequisite for reliable extremes.

## Code Examples
```python
dist_POI = 50
POI = mesh.get_center()
pcd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
[k, idx, _] = pcd_tree.search_radius_vector_3d(POI, dist_POI)
pcd_selection = pcd_o3d.select_by_index(idx)
o3d.visualization.draw_geometries([pcd_selection, mesh])
```
- **What it demonstrates**: a k-d tree radius query to isolate all point-cloud points within a fixed distance of a point of interest (the POI/building center).

```python
def sort_coordinates(XY):
    cx, cy = XY.mean(0)
    x, y = XY.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(-angles)
    return XY[indices]

np_sorted_2D_corners = sort_coordinates(o3d_parcel_corners)
pgon = Polygon(np_sorted_2D_corners)
print(f"This is the parcel area: {pgon.area} m²")
```
- **What it demonstrates**: fixing a self-intersecting polygon from manually-picked 3D corner points by sorting them by angle around the centroid before computing area with Shapely — area corrected from 43.6 m² to 3,408.9 m².

```python
idx_voxels = [v.grid_index for v in voxel_grid.get_voxels()]
color_voxels = [v.color for v in voxel_grid.get_voxels()]

max_voxel, max_color = {}, {}
for idx, v in enumerate(idx_voxels):
    if (v[0], v[1]) in max_voxel.keys():
        if v[2] > max_voxel[(v[0], v[1])]:
            max_voxel[(v[0], v[1])] = v[2]
            max_color[(v[0], v[1])] = color_voxels[idx]
    else:
        max_voxel[(v[0], v[1])] = v[2]
        max_color[(v[0], v[1])] = color_voxels[idx]

count_building_coverage, count_non_building = 0, 0
for col in list(max_color.values()):
    if np.all(col == 0):
        count_non_building += 1
    else:
        count_building_coverage += 1

print(f"Coverage of Buildings: {count_building_coverage*4} m²")
print(f"Built Ratio: {(count_building_coverage*4)/(count_building_coverage*2+count_non_building*2)} m²")
```
- **What it demonstrates**: the full top-voxel-per-column reduction to compute built coverage from a binary-colored voxel grid (result in the book: 17.9% built ratio over a 150m radius).

## Reference Tables
*The book's multimodal data stack for this chapter*

| Dataset | Format | Source |
|---|---|---|
| Terrain point cloud | .laz | GeoTiles.nl |
| Building mesh (LoD1.2/1.3/2.2) | .obj + .mtl (from CityGML) | 3DBAG (TU Delft) |
| Parcel boundary vector | derived corner points | Manual Open3D vertex selection + Shapely |
| Built coverage | VoxelGrid (voxel_size=2) | Derived from point cloud |

*Four analytical challenges and their I/O*

| Challenge | Input | Output |
|---|---|---|
| POI query | point cloud + mesh | distance-filtered point cloud |
| Boundary selection | manually selected vertices | parcel polygon + area (m²) |
| High/low points | selected point cloud | indices/coordinates of Z-extremes |
| Built coverage | voxelized point cloud | % built ratio, area by class |

## Worked Example
The chapter's single running worked example is the Netherlands flood-response scenario: (1) set up a 5-library environment (NumPy, pandas, Open3D, Matplotlib, Shapely); (2) load a LiDAR point cloud (.laz→.xyz, ~7.1M points, columns X,Y,Z,R,G,B,Classification) and a CityGML-derived building mesh (.obj, 674 vertices/488 triangles), profiling with `pcd_df.describe()`; (3) subsample to <5M points (`iloc[::2,:]`), convert to Open3D objects, translate both to a shared local origin via `get_center()`; (4) color the point cloud by LAS classification codes (ground=1, low veg=2, building=6, water=9, bridge=26) for visual QA; (5) run a 50m-radius POI query centered on the building via k-d tree; (6) interactively select parcel corners in the Open3D GUI, strip Z, sort by centroid angle, and compute the parcel area with Shapely (3,408.9 m² after the angle-sort fix); (7) find highest/lowest points via `argmax`/`argmin` on Z and mark them with colored spheres; (8) re-run the POI query at 150m, voxelize (voxel_size=2) with binary building/non-building coloring, and reduce to top-voxel-per-column to compute built coverage (17.9% built, 12,812 m² built vs. 58,372 m² non-built); (9) export results via `np.savetxt` (.xyz) and `o3d.io.write_point_cloud` (.ply).

## Key Takeaways
1. A multimodal viewer is built by translating all datasets into a shared local coordinate frame and passing a list of Open3D geometries to one `draw_geometries` call.
2. k-d tree radius search (`search_radius_vector_3d`) is the standard tool for POI-based spatial filtering of point clouds.
3. Manually-picked polygon corners must be sorted by centroid angle before area computation, or the polygon self-intersects and the area is wrong.
4. `argmax`/`argmin` on the Z column (not `get_max_bound`/`get_min_bound`) is how you locate the actual highest/lowest points.
5. The binary-color-before-voxelize trick cleanly sidesteps Open3D's color-averaging ambiguity when classifying voxel content.
6. Built coverage/ratio is computed by collapsing a voxel grid to its top occupied voxel per (x,y) column, then counting by class.
7. Save complex multi-attribute outputs with NumPy (`.xyz`); save pure geometry+color+normals with Open3D (`.ply`).

## Connects To
- **Ch 4**: directly reuses the representation conversions (mesh↔point cloud↔voxel) and k-d tree/data-structure concepts from this chapter.
- **Ch 6**: the next chapter formalizes the "quick primer" preprocessing steps used here (outlier removal, subsampling, normals) as full 3D data engineering.
- **GeoTiles.nl, 3DBAG**: external open-data portals supplying the point cloud and mesh datasets for the running scenario.
- **Shapely**: external library used for 2D polygon/area computation on extracted parcel boundaries.
