# Chapter 3: 3D Python and 3D Data Setup

## Core Idea
Set up a minimal, robust Python environment (conda + NumPy/pandas/Matplotlib/Open3D + an IDE) in under ten minutes, then learn the taxonomy of where 3D datasets come from (sensor-based vs. mind-based reconstruction) so you can both build and populate your 3D data science lab.

## Frameworks Introduced
- **The Full 3D Data Science Solution Workflow (4 steps)**: (1) set up the Python lab, (2) write your first 3D Python program, (3) apply 3D reconstruction methods to gather data, (4) curate/export datasets.
  - When to use: as the onboarding checklist for any new 3D Python project.
  - How: follow steps in order — environment first, then code, then data acquisition, then curation.
- **Six Ways to Obtain 3D Datasets**: on-site data acquisition (sensor-based), sensor-dependent web scraping, sensor-related 3D modeling, creative 3D assets, generative 3D, and 3D creative web scraping — split into sensor-based (reality-grounded) and mind-based (imagination-grounded) approaches.
  - When to use: when deciding how to source data for a 3D project — matches intent (replicate reality vs. invent content) to method.
  - How: pick sensor-based methods for digital twins/reality capture; pick mind-based (creative/generative) methods for games, hypothetical scenarios, or augmentation.
- **The Four-Step Multimodal Dataset Sourcing Process**: (1) 3D data sourcing (LiDAR/city models), (2) spatial raster data sources (satellite imagery/DEMs), (3) vector data curation (OSM), (4) other sources (street-level imagery, crowdsourced).
  - When to use: when building a multimodal dataset for a region of interest spanning point clouds, meshes, rasters, vectors, and imagery.
  - How: work through each data family's dedicated open portal in sequence, noting that tile extents may differ across sources and must be reconciled during integration.

## Key Concepts
- **Active vs. passive sensors**: active sensors emit and measure their own signal (LiDAR, radar, InSAR — more expensive, more control); passive sensors measure ambient reflected light (cameras/photogrammetry — cheaper, light-dependent).
- **Platform vs. sensor**: the sensor is the measuring device; the platform (backpack, drone/UAV, satellite, static tripod) is what hosts/carries it — the same sensor can be mounted on different platforms for different missions.
- **Photogrammetry acquisition parameters**: focal length, ISO, diaphragm/aperture, shutter speed, white balance — all must be tuned and held consistent across a capture session for valid triangulation.
- **Multimodal data**: data spanning different types/contexts — in this book: 3D point clouds, 3D meshes, city models (CityGML), voxels, spatial rasters, 360° imagery, tabular data, vector data.
- **Relative path convention**: the book's code/data folder structure (`CODE/` + `DATA/`) means scripts reference data via relative paths (`../DATA/...`) from the working directory.
- **Open3D**: the book's primary 3D-specific Python library (C++ core with Python bindings) for creating, manipulating, and visualizing point clouds/meshes/voxels.

## Mental Models
- Treat NumPy + pandas + Matplotlib as your general-purpose "3D-adjacent" toolkit (arrays, tabular I/O, 2D/3D plotting) and Open3D as your dedicated 3D toolkit — use the general tools for lightweight manipulation and Open3D once you need true 3D structures.
- Use the sensor/platform/method triad to decompose any acquisition scenario: "what sensor, on what platform, using what method" fully specifies a dataset's provenance and its likely accuracy/limitations.
- Think of 3D web scraping as a trade-off: near-instant access to existing data vs. zero control over precision/accuracy/timeliness, compared to on-site acquisition's full control but higher time/cost.
- When an object needs to be captured via photogrammetry, think in terms of coverage rings: a circular path at a fixed distance per height tier (e.g., 3 tiers × 8 positions + 1 top-down ≈ 25 images) rather than ad hoc shooting.

## Anti-patterns
- **Installing Python directly from python.org without environment isolation**: leads to version/package conflicts across projects — use conda/Miniconda environments instead.
- **Skipping the `skiprows`/`max_rows` safeguards when loading raw point files**: large .xyz/.las-derived text files can exhaust memory; always cap rows during initial testing.
- **Jumping into custom Python before validating a hypothesis in existing software**: the author explicitly recommends validating ideas in GUI tools (from Ch. 2) before writing pipeline code, to avoid wasted development effort.
- **Using too many images per photogrammetry subject**: more images beyond sufficient coverage only slows reconstruction without improving quality — aim for a right-sized capture pattern (e.g., ~25 images for a small statue), not exhaustive coverage.
- **Ignoring tile-extent mismatches when fusing multiple open data sources**: different providers (e.g., GeoTiles.nl vs. 3DBAG) use different tiling schemes, which must be reconciled before integration.

## Code Examples
```python
import numpy as np
file_data_path='../DATA/sample.xyz'
point_cloud = np.loadtxt(file_data_path, skiprows=1, max_rows=1000000)
point_cloud[0]              # first point's X,Y,Z
point_cloud[1][3]           # Red attribute of second point
point_cloud[:,1]            # Z column for all points
xyz = point_cloud[:,:3]
rgb = point_cloud[:,3:]
mean_Z = np.mean(point_cloud, axis=0)[2]
spatial_query = point_cloud[abs(point_cloud[:,2] - mean_Z) < 1]  # 1m height buffer
```
- **What it demonstrates**: loading a raw XYZRGB point cloud with NumPy, multidimensional indexing to split geometry (xyz) from color (rgb), and a simple spatial/attribute query (height-buffer filter).

```python
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
xyz = spatial_query[:,:3]
rgb = spatial_query[:,3:]
ax = plt.axes(projection='3d')
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb/255, s=0.01)
plt.show()
```
- **What it demonstrates**: visualizing a filtered, colorized point cloud as a 3D scatter plot — colors normalized to [0,1] by dividing by 255.

```python
import open3d as o3d
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector([[1,9,3], [8,5,9], [8,8,9]])
o3d.visualization.draw_geometries([point_cloud])
```
- **What it demonstrates**: creating and visualizing a point cloud object using Open3D's dedicated 3D geometry API rather than raw arrays.

```bash
conda create --name DATAS3D python=3.9
conda activate myenv
conda install pip
pip install numpy matplotlib pandas open3d jupyterlab spyder
```
- **What it demonstrates**: the full environment bootstrap sequence for the book's "DATAS3D" conda environment.

## Reference Tables
*Camera settings used for the book's photogrammetry example (Glyptotek statue)*

| Parameter | Value | Rationale |
|---|---|---|
| Focal length | 28 mm | Balance of distance-to-subject and workable area |
| ISO | 400 (up to 1600 if dark) | Clean exposure without excess noise |
| Aperture (f-stop) | f/3.5 | Enough light, acceptable shallow depth of field for small objects |
| Shutter speed | 1/8 s (125 ms) with tripod | Tripod stability allows a slower shutter; handheld needs <1/200s |
| White balance | 4000 K | Matches ambient scene lighting |

*Open data sources by modality*

| Modality | Source |
|---|---|
| LiDAR / elevation / raster (Netherlands) | GeoTiles.nl |
| 3D city/building models (Netherlands) | 3DBAG |
| Satellite imagery (global) | USGS EarthExplorer |
| Vector / street GIS data (OSM) | Geofabrik |
| 360° street-level imagery | Mapillary |
| General point clouds | OpenTopography (from Ch. 1/2) |

## Worked Example
The chapter's central walkthrough is building a complete "first 3D Python program" against a sampled statue point cloud (`sample.xyz`, from the Glyptotek Museum, captured via the author's own photogrammetry trip): (1) set up conda env `DATAS3D` with Python 3.9, install numpy/pandas/matplotlib/open3d, and an IDE (Spyder or JupyterLab); (2) load the XYZRGB file with `np.loadtxt`, capping rows; (3) index into the array to pull out geometry (`xyz`) vs. color (`rgb`) sub-arrays; (4) compute the mean Z (height) and perform a 1-meter height-buffer spatial query to isolate a horizontal slice of the object; (5) plot the filtered, color-mapped slice in Matplotlib with `ax.scatter(..., c=rgb/255, s=0.01)`. This single pipeline — load → index/query → visualize — is the template reused and extended throughout the rest of the book. The chapter then reconstructs how that statue dataset was itself captured: a 25-image photogrammetry rig (8 angles × 3 height tiers + 1 top-down shot) processed through RealityCapture/Meshroom, demonstrating the full sensor-to-point-cloud pipeline end to end.

## Key Takeaways
1. Use conda/Miniconda environments (not a bare Python install) to isolate 3D project dependencies.
2. NumPy array indexing (`[:, :3]` for XYZ, `[:, 3:]` for RGB) is the fastest way to split and query raw point cloud attributes without extra libraries.
3. Open3D is the dedicated 3D library of choice in this book — reach for it once beyond basic array math/plotting.
4. Classify any 3D dataset you encounter by its sensor/platform/method triad before using it, to understand its accuracy and limitations.
5. Photogrammetry needs disciplined, consistent camera settings and a structured multi-tier capture pattern — not ad hoc photos — for reliable reconstruction.
6. Validate ideas in existing GUI software (Ch. 2 stack) before investing in custom pipeline code.
7. Reconcile tile/extent mismatches across open data sources early, since each provider tiles data differently.

## Connects To
- **Ch 2**: builds directly on the Starter Pack's software stack (CloudCompare, QGIS, etc.) for data visualization/validation before coding.
- **Ch 4**: the structuration/representation chapter that follows this environment setup.
- **Ch 6**: revisits sensor types (ToF, stereo vision, LiDAR) and registration in depth.
- **Ch 10, 18**: deepen photogrammetry, NeRF, and generative 3D reconstruction methods introduced briefly here.
- **Open3D, NumPy, pandas, Matplotlib**: the four core external libraries this chapter installs and exercises.
- **GeoTiles.nl, 3DBAG, USGS EarthExplorer, Geofabrik, Mapillary**: external open-data portals used for the multimodal dataset-sourcing walkthrough.
