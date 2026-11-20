# Chapter 2: Data Architectures for Multimodal AI Systems

## Core Idea
Multimodal AI architectures rest on two foundational patterns — vector-native storage (fast similarity search over embeddings) and composable data lakehouses (unified system of record for structured/unstructured/vector data) — surrounded by embedding/retrieval design, containerized orchestration, reliability/scaling practices, and fabric/mesh governance patterns, illustrated via the fictional social-commerce platform StreamBuy.

## Frameworks Introduced
- **Vector-Native Storage**: purpose-built storage/retrieval for high-dimensional embeddings using ANN indexes.
  - When to use: any system needing similarity search, nearest-neighbor queries, or cross-modal retrieval.
  - How: encode modalities into embeddings, store with metadata, index with ANN (HNSW/IVF/DiskANN), query by embedding + hybrid metadata filter.
- **Three ANN Indexing Strategies**: HNSW (multilayer graph, fast + balanced recall/latency), IVF (clustering into centroids, searches nearest clusters only, trades recall for efficiency), DiskANN (small graph in RAM, vectors on SSD, affordable at billion-scale).
  - When to use: choose based on bottleneck — latency (HNSW), budget/recall trade (IVF), scale without memory cost (DiskANN).
- **Composable Data Lakehouse Architecture**: four layers — cloud object storage (base, transactional table format e.g. Delta Lake/Iceberg/Hudi) → indexing layer (vector/full-text/metadata) → compute/processing layer (streaming, cache, distributed frameworks) → unified API layer (abstracts complexity for apps).
  - When to use: as the enterprise system of record complementing vector-native stores used for real-time serving.
- **Staged Retrieval Pipeline**: metadata filters → ANN index top-k → reranker (cross-modal scorer or lightweight LLM).
  - When to use: balancing cost, latency, and precision across billions of embeddings.
  - How: cheap filter first to shrink candidate pool, then vector similarity, then expensive precise reranking only on the shortlist.
- **Embedding Compression (MRL + Binary Quantization)**: Matryoshka Representation Learning allows truncating embeddings (e.g., 1024→128 dims) while preserving structure; binary quantization reduces floats to single bits (32x memory reduction).
  - When to use: trending/hot items need high-dim precision; long-tail items can use compressed vectors to control storage cost.
  - How: combine — truncated/quantized vectors for fast in-memory candidate retrieval, full-precision vectors on disk for reranking.
- **Four Scaling Strategies (2x2)**: horizontal scaling (stateless pods in parallel), vertical scaling (bigger single machine for monolithic models), cluster scaling (isolate workloads by modality), autoscaling (reactive HPA + cluster autoscaler).
  - When to use: production systems combine all four simultaneously rather than picking one.
- **Data Fabric vs Data Mesh**: fabric = centralized, automated integration/governance across distributed heterogeneous data; mesh = decentralized domain ownership where teams expose data as products via APIs.
  - When to use: fabric when you need uniform, automated integration and governance across environments; mesh when scaling across many teams/domains needing autonomy. Often complementary in the same org.

## Key Concepts
- **ANN (Approximate Nearest Neighbor)**: indexing method trading perfect recall for speed over billions of vectors.
- **Hybrid Query**: a query joining vector similarity with structured metadata filters (e.g., "blue sneakers under $100").
- **Hybrid Storage Tiering**: hot data (active embeddings) in fast/in-memory stores; cold data (archival media) in cheap object storage.
- **Online/Offline Split**: online path incrementally updates the vector index via streaming; offline path does full re-embedding/index rebuilds during model upgrades.
- **Node Pool Specialization**: Kubernetes clusters divided into CPU, GPU, and memory-optimized pools matched to workload type.
- **Columnar Transport (e.g., Apache Arrow Flight)**: protocol for moving batch vectors without serialization overhead between services.
- **Index Health / Staleness**: a vector index can silently degrade retrieval quality (stale, skewed, imbalanced) without throwing errors — must be actively probed.
- **Federated Governance**: global security/privacy/compliance policy enforced consistently across mesh domains while domains retain operational autonomy.
- **Cross-Modal Feature Extraction**: integrating signals from text, video, and structured records into a shared representation space usable consistently by downstream models.

## Mental Models
- Think of **embedding/retrieval as core infrastructure, not a model output** — dimensionality, compression, staging and online/offline split are architectural decisions that determine whether a system scales or buckles.
- Read a container/orchestration stack **bottom-up for deployment, top-down for data flow** — hardware/runtime/dependencies are built first, but a user request flows down from the application layer through APIs into modality-specific pods.
- **Scaling strategies answer "how much capacity exists"; production patterns (load balancing, geo-distribution, performance optimization, resource utilization) answer "how is that capacity routed, placed, observed, shared."** Skipping either layer fails predictably (e.g., horizontal scaling without load balancing = pods nobody sends traffic to).
- **Fabric centralizes, mesh decentralizes** — use fabric for seamless integration across environments, mesh for domain-product ownership at team scale; they are complementary, not competing choices.
- Observability stack layers each catch a different failure class: hardware/pipeline issues (base telemetry) → data quality drift (index probes) → cross-service bottlenecks (distributed tracing) → business impact (dashboards/alerts). You need all four.

## Anti-patterns
- **One-size-fits-all containers**: bundling all modalities into a single bloated container instead of modality-specific isolated containers — breaks independent scaling/updating.
- **Treating orchestration as an invisible detail**: orchestration (node pools, autoscalers, communication backbones) is the architectural spine, not an afterthought bolted onto training/inference.
- **Ignoring index health**: a stale or skewed vector index doesn't error, it just silently returns worse results — must be actively monitored, not assumed healthy.
- **Comparing embeddings across incompatible spaces**: image (512-dim ResNet18) and text (384-dim MiniLM) embeddings cannot be directly compared without a shared projection layer (e.g., CLIP) — a common mistake when building naive cross-modal retrieval.
- **Picking only one scaling strategy**: relying solely on horizontal OR vertical OR cluster scaling instead of combining all four (horizontal + vertical + cluster isolation + autoscaling) as production systems require.

## Code Examples
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_models.py   # fetches ResNet18 (~45MB) + all-MiniLM-L6-v2 (~80MB) locally, no runtime network calls
```
- **What it demonstrates**: reproducible, offline-capable encoder setup — no silent fallback if weights are missing.

```python
# Conceptual: ImageEncoder.encode() behavior
# resize to 224x224 -> ImageNet normalize -> ResNet18 forward pass (head stripped)
# -> unit-normalize to 512-dim vector (required for correct cosine similarity)
```
- **What it demonstrates**: why unit-normalization is critical for downstream cosine-similarity search.

```python
# Milvus collection setup per modality, IVF_FLAT index
# streambuy_image_features / streambuy_review_features / streambuy_video_features
# index: IVF_FLAT, nlist=1024 centroids; query-time nprobe=10
```
- **What it demonstrates**: the IVF indexing trade-off in practice — nprobe controls recall vs latency.

```bash
docker-compose up -d           # Starts PostgreSQL + Milvus (with etcd, MinIO)
uvicorn src.api.main:app --reload
# Endpoints: POST /products/, GET /jobs/{job_id}, POST /search/{modality}
```
- **What it demonstrates**: full-stack dual-database architecture (PostgreSQL for metadata, Milvus for vectors) exposed via a FastAPI service with async background feature extraction.

## Reference Tables
**Table 2-1 (partial, reconstructed): Production patterns for reliability and scale**
| Category | Approach | Description | StreamBuy use case |
|---|---|---|---|
| Load balancing | Network + application-level routing | Spreads traffic evenly at network level; routes intelligently at app level | Network LB distributes video streams across GPU pods; app routing sends chat sentiment to NLP pods, images to vision encoders |
| Geographic distribution | CDN edge caching / regional replication | Proximity and resilience | Cached product imagery served from CDN edges near each viewer |
| Performance optimization | Bottleneck detection / dashboards | Surfaces latency tails | Embedding pipeline latency tail surfaced before users notice |
| Resource utilization | Hardware efficiency / co-location | Shares hardware across workloads | Same GPU node hosts multiple text encoders alongside heavier vision workload |

**Table 2-2 (partial, reconstructed): Data fabric vs. data mesh**
| Feature | Data Fabric | Data Mesh | StreamBuy example |
|---|---|---|---|
| Integration pattern | Centralized automation pipelines, orchestration layers abstract complexity, uniform ingestion/governance | Decentralized, domain-owned, curated products exposed via APIs | Fabric centralizes ingestion from live streams/transactions; mesh domains (e.g., product catalogs) expose curated embeddings via a product metadata API |

**Three ANN strategies**
| Strategy | Mechanism | Best for |
|---|---|---|
| HNSW | Multilayer graph, logarithmic-time search | Balance of recall and latency |
| IVF | Cluster into centroids, search nearest only | Efficiency over exhaustive recall |
| DiskANN | Small graph in RAM, vectors on SSD | Billion-scale at affordable memory cost |

## Worked Example
**StreamBuy Multimodal Feature Store playbook**: build a self-contained feature store across 3 modalities — product images (20 JPEGs, ResNet18 → 512-dim unit-normalized vectors), customer reviews (18 text files, all-MiniLM-L6-v2 → 384-dim vectors), product videos (31 MP4 clips, 8 sampled frames through the same ResNet18 backbone, averaged into a 512-dim vector). Architecture: FastAPI API layer (`POST /products/`, `GET /jobs/{job_id}`, `POST /search/{modality}`) + async `TaskManager`/`FeatureProcessor` worker + dual-database backend (PostgreSQL for metadata/job status via `asyncpg`, Milvus for vector similarity search with per-modality collections and IVF_FLAT indexes, nlist=1024, nprobe=10). Because image and video share the same 512-dim ResNet18 embedding space, cross-modal retrieval works between them (test observed cosine similarity ≈0.62 for same-product image/video pairs) but NOT between image (512-dim) and text (384-dim) without a shared projection model like CLIP. Validation checks: same-category images should show cosine similarity >0.6, dissimilar products <0.3; average pairwise similarity across a 20-image batch should sit in the 0.3–0.5 range (if >0.9, embeddings aren't discriminative enough).

## Key Takeaways
1. Vector-native storage and composable lakehouses are complementary, not competing: vector stores serve real-time similarity search; lakehouses serve as the system of record for analytics, retraining, and compliance.
2. Embedding dimensionality, compression (MRL, binary quantization), and staged retrieval (filter → ANN → rerank) are system-level architecture decisions, not model-training details.
3. Container and orchestration design should isolate by modality — each pipeline should scale, update, and fail independently.
4. Production systems combine all four scaling strategies (horizontal, vertical, cluster, autoscaling) together with production patterns (load balancing, geo-distribution, performance optimization, resource utilization) rather than relying on just one lever.
5. Observability requires four distinct layers — pipeline telemetry, index health probes, distributed tracing, and dashboards/alerts — because each catches a different class of failure, and stale indexes fail silently.
4. Data fabric (centralized integration/governance) and data mesh (decentralized domain ownership) are complementary strategies for scaling multimodal data platforms across environments and teams.
6. Cross-modal similarity is only valid between encoders sharing an embedding space; mixing incompatible embedding spaces (e.g., 512-dim image vs 384-dim text) without a shared projection model (CLIP) produces meaningless comparisons.

## Connects To
- **Ch 1**: Builds directly on the semantic layer and data contract concepts, applying them to concrete storage/retrieval architecture decisions.
- **Ch 3**: Moves from architecture to implementation platforms and patterns for realizing these designs in production.
- **Ch 6**: Expands on vector-native storage and retrieval strategies for multimodal data in much greater depth.
- **Kubernetes / Milvus / Kafka**: external tooling referenced throughout as concrete instances of the architectural categories (orchestration, vector DB, event streaming).
