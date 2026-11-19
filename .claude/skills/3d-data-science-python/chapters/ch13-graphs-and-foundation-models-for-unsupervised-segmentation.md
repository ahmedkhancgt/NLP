# Chapter 13: Graphs and Foundation Models for Unsupervised Segmentation

## Core Idea
Two complementary unsupervised segmentation strategies extend Ch. 12's clustering: connectivity-based clustering via graph theory (points as nodes, spatial proximity as edges, connected components as objects) for pure 3D geometry, and foundation-model-based segmentation (Meta's Segment Anything Model, applied to 2D projections of the point cloud) for leveraging massive pretrained vision models without any 3D-specific training data.

## Frameworks Introduced
- **Graph-Based Euclidean Clustering, 6-step workflow**: (1) environment setup, (2) build a radius graph (points as nodes, k-d tree-based proximity as edges), (3) graph analytics (connected component statistics), (4) optional visualization, (5) connected-components-as-clusters on synthetic data, (6) apply to a real 3D point cloud.
  - When to use: unsupervised object delineation from unstructured point clouds when connectivity/contiguity (not just density or centroid proximity) best defines "object."
  - How: `build_radius_graph()` connects points within a radius via a k-d tree, optionally pruned to k-nearest-neighbors per node; `nx.connected_components()` then yields one cluster per spatially contiguous subgraph.
- **SAM-Based 3D Segmentation, 4-step workflow**: (1) 3D project setup (PyTorch + Segment Anything install), (2) load/configure SAM (ViT-B/L/H encoder choice), (3) project the 3D point cloud to a 2D image (spherical/equirectangular projection from Ch. 4), (4) run SAM's automatic mask generator on the projected image and transfer per-pixel predictions back to 3D points via the projection's point-to-pixel mapping.
  - When to use: when you want to leverage a general-purpose vision foundation model's segmentation power on 3D data without training anything, accepting some information loss from projection to 2D.
  - How: `generate_spherical_image()` (Ch. 4) produces both the image and a `mapping` array from pixel→point index; `SamAutomaticMaskGenerator(sam).generate(image_rgb)` segments the image; `color_point_cloud()` transfers per-pixel segmentation color back onto the corresponding 3D points via the mapping.
- **Graph Theory Core Vocabulary**: vertices/nodes, edges, order (vertex count), size (edge count), degree (edges per vertex), connected component (maximal subgraph where every pair of nodes has a path, with no path to anything outside it).
  - When to use: as the conceptual vocabulary for reasoning about any graph-based 3D structuring approach.
  - How: map point cloud → graph (point=node, proximity=edge) → connected components = candidate objects.

## Key Concepts
- **Unsupervised segmentation vs. semantic segmentation**: unsupervised segmentation groups points/surfaces without a predefined label meaning (group 1, group 2, ...); semantic segmentation assigns labels with predefined meaning ("chair," "wheel") and typically needs substantial annotated training data.
- **Radius graph**: a graph where nodes (points) are connected by an edge if their distance is below a threshold radius, found efficiently via k-d tree `query_pairs()`.
- **k-NN pruning**: after radius-based edge creation, keep only each node's k nearest edges (by weight) to prevent overconnectivity/graph complexity explosion in dense regions.
- **Radius and max_neighbors as a fragmentation/merging dial**: smaller radius → more, smaller fragmented clusters (risk of oversegmentation); larger radius → fewer, merged clusters (risk of undersegmentation); max_neighbors regularizes graph complexity similarly.
- **Zero-shot / few-shot learning**: zero-shot = recognizing something never seen during training; few-shot = generalizing from a handful of labeled examples per new class — both are core capabilities SAM exhibits as a foundation model.
- **SAM encoder variants (ViT-B/L/H)**: 91M/308M/636M parameters respectively, trading speed for segmentation quality — ViT-H gives the best results but only marginal gains over ViT-L at much higher compute cost.
- **SAM mask dictionary fields**: `segmentation` (boolean WxH mask), `area`, `bbox` (xywh), `predicted_iou`, `point_coords`, `stability_score`, `crop_box` — the structured output of `SamAutomaticMaskGenerator`.
- **Point-to-pixel mapping**: the array returned alongside a spherical/equirectangular projection image that records, for each pixel, which 3D point (if any) produced it — essential for transferring 2D segmentation results back onto the original 3D point cloud.

## Mental Models
- Think of connectivity-based clustering as encoding the intuitive human notion that "objects don't touch (once you remove the ground they rest on)" — connectivity, not just density or distance-to-centroid, is what separates distinct physical objects for us visually.
- Treat the SAM pipeline as "borrowing 2D vision superpowers for 3D, at the cost of a lossy projection" — you trade some 3D structural fidelity (occlusion, point-to-pixel ambiguity) for access to a massively pretrained, zero-shot-capable segmentation model that needs no 3D training data at all.
- Use the "radius/max_neighbors as regularization knobs" model for any graph-based or density-based clustering: too tight → oversegmentation; too loose → undersegmentation; the right value depends on local point density, so prefer data-driven values (e.g., `3 × mean nearest-neighbor distance`) over fixed constants.
- Picture connected components as a strict, binary form of clustering: either a path of edges connects two points (same object) or it doesn't (different objects/noise) — no notion of degree or similarity beyond pure connectivity.

## Anti-patterns
- **Using a fixed radius/max_neighbors value across point clouds of different densities**: derive the graph radius from the data itself (e.g., 3× mean nearest-neighbor distance) rather than hardcoding, since density varies by sensor and scene.
- **Treating unsupervised (connectivity/SAM-based) segmentation output as semantic labels**: these methods produce group IDs ("cluster 1," "cluster 7"), not meaningful class names ("chair") — a separate classification step (Ch. 14+) is needed to attach semantics.
- **Expecting SAM to label every 3D point after projection**: occluded or unevenly-sampled regions produce "unseen"/unlabeled points after the 2D→3D mapping; smaller SAM models (ViT-B/L) leave even more points unlabeled than ViT-H.
- **Not pruning graph edges in dense regions**: unconstrained radius-graph construction in high-density point clouds can create near-exponential edge counts — always consider k-NN pruning for computational tractability.
- **Forgetting to check CUDA/GPU availability and memory before running SAM**: SAM (especially ViT-H) is memory-hungry; the chapter shows explicit GPU-memory diagnostics and cache-clearing steps as standard practice, and recommends a lighter model or reduced `points_per_batch` if memory errors occur.

## Code Examples
```python
def build_radius_graph(points, radius, max_neighbors):
    points = np.asarray(points)
    kdtree = KDTree(points)
    graph = nx.Graph()
    for i in range(len(points)):
        graph.add_node(i, pos=points[i])
    pairs = kdtree.query_pairs(radius)
    for i, j in pairs:
        dist = np.linalg.norm(points[i] - points[j])
        graph.add_edge(i, j, weight=dist)
    if max_neighbors is not None:
        prune_to_k_neighbors(graph, max_neighbors)
    return graph

def analyze_components(graph):
    components = list(nx.connected_components(graph))
    return {
        'num_components': len(components),
        'component_sizes': [len(c) for c in components],
        'largest_component_size': max(len(c) for c in components),
        'isolated_points': sum(1 for c in components if len(c) == 1),
    }
```
- **What it demonstrates**: constructing a k-d-tree-accelerated radius graph over a point cloud and extracting connected-component statistics — the core of connectivity-based clustering.

```python
xyz = np.asarray(pcd.points)
nn_d = np.mean(pcd.compute_nearest_neighbor_distance()[0])
graph = build_radius_graph(xyz, radius=nn_d * 3, max_neighbors=10)
# Real dataset result: 52 components, largest=16,771 pts (bed), 16 isolated points
```
- **What it demonstrates**: data-driven radius selection (3× mean nearest-neighbor distance) applied to a real indoor scan, successfully isolating furniture pieces as connected components.

```python
sam = sam_model_registry["vit_h"](checkpoint=MODEL)
sam.to(device=USED_D)
mask_generator = SamAutomaticMaskGenerator(sam)
result = mask_generator.generate(image_rgb)  # list of mask dicts

spherical_image, mapping = generate_spherical_image(center_coordinates, point_cloud, colors, resolution)
# ... run SAM on spherical_image, then:
def color_point_cloud(image_path, point_cloud, mapping):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    modified_point_cloud = np.zeros((point_cloud.shape[0], point_cloud.shape[1] + 3), dtype=np.float32)
    modified_point_cloud[:, :3] = point_cloud
    for iy in range(h):
        for ix in range(w):
            point_index = mapping[iy, ix]
            if point_index != -1:
                modified_point_cloud[point_index, 3:] = image[iy, ix]
    return modified_point_cloud
```
- **What it demonstrates**: the full SAM-on-3D pipeline — spherical projection with point-to-pixel mapping, automatic mask generation, and transferring segmentation colors back from 2D pixels to 3D points.

## Reference Tables
*SAM encoder variants*

| Encoder | Parameters | Speed | Quality |
|---|---|---|---|
| ViT-B (basic) | 91M | Fastest | Low |
| ViT-L (large) | 308M | Fast | High |
| ViT-H (huge) | 636M | Slow | Highest (marginal gain over ViT-L) |

*Graph theory vocabulary*

| Term | Meaning |
|---|---|
| Order | Number of vertices (nodes) |
| Size | Number of edges |
| Degree | Number of edges connected to a given vertex |
| Connected component | Maximal subgraph where every node pair has a path, isolated from the rest |

*Radius/max_neighbors effects*

| Parameter | Too small | Too large |
|---|---|---|
| Radius | Oversegmentation (fragments one object into many) | Undersegmentation (merges distinct objects) |
| max_neighbors | Disconnects meaningful spatial relationships | Introduces noise, reduces discriminative power |

## Worked Example
The chapter's two full worked examples are: (1) **Graph-based Euclidean clustering on a furnished room**: load `room_furnitures.ply` from a Naavis VLX2 handheld scanner, translate to local origin, build a radius graph with radius = 3× mean nearest-neighbor distance and max_neighbors=10, analyze connected components (52 components found, largest = 16,771 points identified as the bed, 16 isolated noise points), then visualize with `plot_cc_o3d()` which colors each significant component distinctly and small clusters (≤10 points) black — successfully isolating the bed, chairs, table, lamps, basket, and even small toys, including separating overlapping objects like a chair and table despite top-down overlap. Before this, a synthetic 300-point 2D dataset is used to validate `build_radius_graph`/`analyze_components`/`plot_components` and to empirically study parameter sensitivity by sweeping radius (0.05/0.1/0.5) and max_neighbors (2/5/10). (2) **SAM for building reconnaissance**: set up a PyTorch + Segment Anything environment with CUDA, load the ViT-H checkpoint, first validate on a 2D aerial image (`biscarosse.jpg` from Google Earth, successfully segmenting rooftops and pools in ~6 seconds), then apply the full pipeline to an indoor LiDAR scan (`indoorpcd.las`): project to a 500px-resolution spherical/equirectangular image via `generate_spherical_image()` (Ch. 4), run SAM's automatic mask generator on the projection, and use the projection's point-to-pixel `mapping` to transfer segmentation colors back onto the 3D point cloud, exporting the result as a `.las` file — successfully delineating furniture, doors, and windows in the mission's reconnaissance scenario, with results also shown working on aerial top-down point clouds and additional indoor scenes.

## Key Takeaways
1. Connectivity-based clustering (graph + connected components) captures the intuitive "objects are things that don't touch" principle that centroid/density clustering alone may miss.
2. Derive graph radius from the data itself (e.g., 3× mean nearest-neighbor distance) rather than a fixed constant, since point density varies by sensor and scene.
3. k-NN pruning after radius-graph construction keeps dense regions computationally tractable.
4. SAM lets you apply a massively pretrained, zero-shot vision foundation model to 3D data by projecting to 2D and mapping predictions back — no 3D training data required.
5. ViT-H gives the best SAM segmentation quality but only marginally beats ViT-L at much higher compute/memory cost — pick the encoder size based on your speed/quality trade-off.
6. Both techniques here produce unlabeled group IDs, not semantic class names — a separate classification step is still needed for true "this is a chair" semantics.
7. Projection-based 3D-to-2D-to-3D pipelines inevitably lose some information (unlabeled/occluded points); fusing multiple viewpoints is the natural next refinement.

## Connects To
- **Ch 4**: reuses the spherical/equirectangular projection function (`generate_spherical_image`) and point-to-pixel mapping machinery as the bridge for SAM-based 3D segmentation.
- **Ch 9**: RANSAC is suggested as a complementary first step to remove large structural planes (walls, floors) before connectivity-based clustering of remaining objects.
- **Ch 12**: directly extends that chapter's clustering taxonomy, covering the connectivity-based and (implicitly) foundation-model-based families left for this chapter.
- **Ch 14**: the natural next step — attaching actual semantic labels to the unsupervised clusters/segments produced here via supervised learning.
- **NetworkX, SciPy (KDTree), PyTorch, Segment Anything (Meta AI), OpenCV, laspy**: the core external libraries this chapter's two pipelines are built on.
