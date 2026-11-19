---
name: 3d-data-science-python
description: "Knowledge base from \"3D Data Science with Python\" by Florent Poux. Use when working with 3D point clouds, 3D data representation/structuration, 3D shape recognition, 3D machine/deep learning (clustering, PointNet, PyTorch), LiDAR building reconstruction, or the 3D data science workflow."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# 3D Data Science with Python
**Author**: Florent Poux | **Pages**: ~690 | **Chapters**: 18 | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch07`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The 3D Data Science Modular Workflow** (Ch 1, synthesized in Ch 17): the master pipeline — Data Acquisition → Preprocessing → Registration → 3D Data Classification (semantic injection) → Structuration/Modeling → 3D Data Analysis → 3D Data Visualization → Application Development. Treat each module as a swappable "cog": tune the algorithm inside a phase to the application without restructuring the overall pipeline.

**Geometry–Topology–Semantics (GTS) Triad**: a 3D dataset is only reasoning-ready once geometry (shape/size/position) is paired with topology (connectivity/adjacency) and semantics (labels/meaning). Geometry alone is a dead end for automated decision-making.

**Point Cloud as Canonical Data Link**: every 3D representation (mesh, voxel, depth map, multiview image) can convert to and from a point cloud. Use point clouds as the hub format when integrating multi-modal 3D data — never convert directly between every pair of representations.

**3D Data Structures — pick by query pattern**: k-d tree (flexible, nonuniform data, fast build, memory-efficient) vs. octree (uniform/voxel-aligned data, range-query pruning, LoD) vs. BVH (ray tracing, collision detection, dynamic scenes). k-d trees build ~4x faster than octrees on the same data.

**RANSAC vs. Region Growing**: RANSAC answers "does this noisy data contain a known shape?" (robust model fitting via sample→fit→count inliers→repeat); region growing answers "what's locally consistent here?" (bottom-up growth from a seed via distance+similarity criteria). Hybridize: RANSAC for robust primitive detection, then region growing (or DBSCAN) to refine within each detected region and strip spatially-incoherent artifacts.

**Clustering family selection**: centroid-based (k-means — fixed k, compact/spherical clusters) vs. density-based (DBSCAN — auto cluster count, arbitrary shapes, noise-robust) vs. connectivity-based (graph + connected components — "objects don't touch") vs. distribution-based (GMM — probabilistic, overlapping classes). Use the elbow method (WCSS vs. k) to pick k for k-means; derive DBSCAN's eps from local point-spacing statistics rather than guessing.

**Generalization over training accuracy**: a supervised model scoring 98% on same-distribution test data can collapse to 54% on a genuinely different dataset (different city/sensor). Always validate against a distributionally different set. When generalization fails, improve the feature set (geometry-aware descriptors: planarity, verticality, omnivariance, normal change rate) before switching models — this recovered the book's example from 54% to 97% accuracy.

**3D Deep Learning Architecture ↔ Representation Mapping**: voxel grids → 3D CNNs; raw point clouds → PointNet/PointNet++/Point Transformer; meshes (graph-structured) → GNNs; multiview 2D renders → MVCNNs (can reuse pretrained 2D backbones). Match architecture to the data's native representation rather than forcing a lossy conversion.

**PointNet's Core Trick — Permutation Invariance**: process each point independently via a shared MLP, then aggregate via max pooling (a symmetric function) to build an order-invariant global feature vector. T-Net submodules learn their own spatial/feature alignment (regularized toward orthogonality in the loss) instead of requiring hand-designed canonicalization.

**Mesh Reconstruction — match method to data quality**: Ball-Pivoting Algorithm (BPA) for clean, well-sampled data; Poisson reconstruction (with density-based vertex filtering + bbox cropping) for noisy/incomplete scans; hybrid planar-segmentation + Poisson/alpha-shape for mixed scenes with both flat and organic surfaces.

**Photogrammetry → NeRF → Gaussian Splatting**: each generation traded one limitation for another. Photogrammetry is fast but fails on transparency/reflections/fine detail. NeRF handles those via an implicit neural volume but is slow (ray marching) and hard to edit. 3D Gaussian splatting represents scenes as explicit, analytically-renderable Gaussians — real-time AND editable, inheriting most of NeRF's visual fidelity. Strip a Gaussian splat scene down to just its Euclidean positions and you recover a canonical point cloud.

**Spatial AI Stack**: 3D Perception (reconstruction + preparation) → 3D Scene Understanding (semantics + affordances/functions/physical properties + topology) → 3D Spatial AI Reasoning (knowledge graphs + reasoning engines). VR/AR/XR are interfaces to spatial AI, not spatial AI itself.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Introduction to 3D Data Science | 3D Data Science Starter components, GTS Triad, 3D Data Science Modular Workflow (8 modules) |
| ch02 | Resources and Software Essentials | 3D Data Science Starter Pack, 3D Data Expertise Venn Diagram, AI Specialization Hierarchy |
| ch03 | 3D Python and 3D Data Setup | 4-step solution workflow, Six Ways to Obtain 3D Datasets, 4-step multimodal sourcing |
| ch04 | 3D Data Representation and Structuration | 3D Data Representation Spectrum, Point Cloud as Canonical Link, Data Structures (k-d tree/octree/BVH) |
| ch05 | Developing a Multimodal 3D Viewer with Python | Multimodal Viewer Workflow, Four-Challenge Analysis Pattern, Binary Voxel Coloring Trick |
| ch06 | Point Cloud Data Engineering | 4-step Point Cloud Data Engineering workflow, Global vs. Local Feature Extraction, Coarse-to-Fine Registration |
| ch07 | Building 3D Analytical Apps | 9-step App-Building Workflow, Weinmann Geometric Feature Set, Interactive Thresholding Pattern |
| ch08 | 3D Data Analysis | 4 Buckets of 3D Data Analysis, 5-step Descriptive Analysis Workflow, 3D Deviation Analysis |
| ch09 | 3D Shape Recognition | RANSAC 4-step loop, Region Growing 6-step process, Hybrid RANSAC+Region Growing |
| ch10 | 3D Modeling: Advanced Techniques | 4 Modeling Approaches, Representation Comparison (Polygonal/NURBS/SDF/Boolean), B-rep Hierarchy |
| ch11 | 3D Building Reconstruction from LiDAR Data | 4-Phase Reconstruction Workflow, Ground-Referenced Height Correction, 2D Footprint→3D via Alpha Shape |
| ch12 | 3D Machine Learning: Clustering | 3-Stage Clustering Structure, 4 Clustering Algorithm Families, Multi-RANSAC+DBSCAN Refinement |
| ch13 | Graphs and Foundation Models for Unsupervised Segmentation | Graph-Based Euclidean Clustering (6 steps), SAM-Based 3D Segmentation (4 steps), Graph Theory Vocabulary |
| ch14 | Supervised 3D Machine Learning Fundamentals | Training vs. Inference, Three-Way Data Split, 3D ML Development Cycle Checklist |
| ch15 | 3D Deep Learning with PyTorch | 4 Representations→Architecture Mapping, 3D DL Project Backbone, Transfer Learning vs. Fine-Tuning |
| ch16 | PointNet for 3D Object Classification | PointNet 5-Step Pipeline, T-Net, PointNet Loss (classification+orthogonality), Tiling+Normalization |
| ch17 | The 3D Data Science Workflow | Reference 5-Phase Workflow, Three Segmentation Tracks, Geometric Fingerprinting |
| ch18 | From 3D Generative AI to Spatial AI | Five Advanced Projects, Text/Image-to-3D Pipeline, 3D Perception→Understanding→Reasoning Stack |

## Topic Index

- **Affordances/functions/physical properties** → Ch 18
- **Alpha shapes** → Ch 9, 11
- **Attention/Transformer architectures (3D)** → Ch 15, 18 (Point Transformer)
- **Bounding boxes (AABB/OBB)** → Ch 8
- **CadQuery / parametric modeling** → Ch 10
- **Clustering (k-means, DBSCAN)** → Ch 12
- **Deviation analysis (point-to-plane/mesh)** → Ch 8
- **Feature extraction (planarity, verticality, omnivariance)** → Ch 6, 7
- **Gaussian splatting** → Ch 18
- **Generative AI (text/image-to-3D, Shap-E, TRELLIS)** → Ch 18
- **Graph theory / connectivity-based clustering** → Ch 13
- **k-d trees, octrees, BVH** → Ch 4
- **LiDAR building reconstruction** → Ch 11
- **Mesh reconstruction (BPA, Poisson)** → Ch 10
- **Monocular depth estimation / image-to-3D** → Ch 10, 18
- **Multimodal 3D viewer** → Ch 5
- **NeRF** → Ch 18
- **Normalization/scaling strategies** → Ch 8, 14, 16
- **Open-vocabulary scene understanding (OpenScene, CLIP)** → Ch 18
- **PointNet / PointNet++ / Point Transformer** → Ch 15, 16, 18
- **Point cloud representations (voxel, mesh, depth map)** → Ch 4
- **Projections (ortho/perspective/spherical)** → Ch 4, 13
- **PyTorch (tensors, training loop, DataLoader)** → Ch 15
- **PyVista** → Ch 7
- **RANSAC** → Ch 9, 12
- **Region growing** → Ch 9
- **Registration (ICP, FPFH, RANSAC-based)** → Ch 6
- **Segment Anything Model (SAM)** → Ch 13
- **Semantic segmentation (classical ML)** → Ch 14
- **Spatial AI / knowledge graphs / reasoning** → Ch 18
- **Supervised learning / generalization** → Ch 14
- **Voxelization** → Ch 4, 10, 17

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — techniques and pipelines with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 18 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
