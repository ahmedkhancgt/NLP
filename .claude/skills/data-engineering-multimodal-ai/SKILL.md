---
name: data-engineering-multimodal-ai
description: "Knowledge base from \"Data Engineering for Multimodal AI\" by Vasundra Srinivasan. Use when designing data pipelines for multimodal AI systems — data architectures, ingestion/transformation, feature engineering, storage/retrieval, streaming pipelines, MLOps, cloud-native scaling, model deployment/serving, or governance/security for multimodal AI."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Engineering for Multimodal AI
**Author**: Vasundra Srinivasan | **Pages**: ~549 | **Chapters**: 12 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch07`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The Semantic Layer (expanded)** — beyond analytics-style metric consistency, add ontological reasoning, entity resolution, and cross-modal integration to transform disconnected modality signals into coherent, queryable events. "Multimodal systems fail not because models are weak but because representation is incoherent."

**Data Contracts** — the non-negotiable gate at ingestion: required fields, validation rules, violation handling (reject/log/quarantine). PII masking happens at ingestion regardless of any other principle. Without contracts, entity resolution and cross-modal joins produce garbage.

**ELT over ETL (default for multimodal)** — land raw signals first, transform on demand, because schemas and embedding models change faster than you can commit to early transformation. ETL remains right for stable-schema, governance-heavy domains (payments, strict compliance).

**Kappa over Lambda Architecture** — a single append-only, replayable log beats dual batch/speed layers once you're joining streams sampled at wildly different rates (30Hz video vs 1Hz sensor). Event-time alignment — not arrival/processing order — is the ground truth for every multimodal join.

**Five Fusion Architectures** (supersede early/late fusion): encoder-decoder (shared latent, preserve resolution), attention-based (dynamic modality weighting by context), graph-based (relational reasoning over entities), generative (impute missing modalities), constraint-based (contrastive cross-modal alignment, e.g. CLIP-style).

**Staged Retrieval** — metadata filter → ANN top-k → reranker. "The bottleneck is almost never retrieval; it is almost always representation" — invest in parsing/extraction quality before optimizing vector infrastructure.

**Feature Store Maturity Path** — dual-layer (offline/online, same transform code) → vector-augmented hybrid → metadata-governed feature mesh → graph-integrated. Progress as modality diversity and team scale grow; a feature store is "the only component that survives a modality change, a model upgrade, and a shift in time horizon."

**Layered Drift Monitoring** — input drift, cross-modal drift (correlation decay — invisible if modalities monitored independently), latency-induced drift (`align_gap` tail), semantic drift (operator override rate), contextual drift (silent upstream API changes). Multimodal models are more fragile than unimodal ones because they depend on cross-modal alignment.

**Safe Retraining (Shadow/Canary Gate)** — candidate model runs on real traffic without affecting users; promoted only if it beats production across ALL dimensions (latency, accuracy, acceptance) over a sustained window; otherwise logged and the production model stays untouched.

**Modality-Specific Autoscaling + Cross-Modal Dependency Orchestration** — no uniform CPU-threshold policy works; each modality scales on its own signal (GPU batch size, watermark lag, memory pressure, queue depth), and scaling decisions must also be gated on cross-modal readiness (fused-readiness markers) to avoid downstream starvation or premature eviction. "Autoscaling is a graph problem, not a pod problem."

**The Inference Triangle** — data preparation, latency budget, and compute architecture pull against each other in every real-time multimodal serving decision; there is no free lever.

**Composite Manifest Versioning** — version every heterogeneous deployed component (encoders, fusion model, preprocessing, serving config) together as one immutable, auditable bundle, not just model weights.

**Governance as a Data Property, not a Checkpoint** — lineage, policy tags, and compliance travel WITH the data at every stage (ingestion → storage → inference → audit). "The cheapest data to govern is data you never ingested." Detect-tag-redact PII at the earliest point; right-to-erasure requires cascading deletion across every derived artifact.

**Isolation + Contracts + Tiered Inference + Three-Layer Observability** — the production-grade pattern for always-on multimodal systems: independent per-modality lanes/queues/autoscaling, versioned schema contracts at every fusion boundary, inference split by latency tier (edge/batch/analytics), and trace-back observability (lane → fusion boundary → inference output). "Build for uptime and build for change."

**Shadow Mode → Canary → Production (for evolution)** — the safe pattern for adding new modalities, upgrading models (dual-support deprecation), bridging incompatible embeddings (async upgrade bridge), and surviving partial outages (graceful degradation with explicit null-reason propagation) without downtime or full rewrites.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | The Multimodal AI Data Engineering Landscape | Semantic layer (expanded), data contracts, multimodal data lifecycle (9-stage loop), cloud-native tenets |
| ch02 | Data Architectures for Multimodal AI Systems | Vector-native storage, ANN indexing (HNSW/IVF/DiskANN), composable lakehouse, staged retrieval, data fabric vs. mesh |
| ch03 | Platforms and Patterns for Multimodal AI Data Engineering | Three platform archetypes (managed/composable/streaming), representation, temporal state assembly, selective context retrieval, decision bundle |
| ch04 | Ingestion and Transformation for Multimodal AI Systems | Traditional/hybrid/agentic ingestion maturity, ELT vs ETL, schema evolution, modality-aware DLQ |
| ch05 | Feature Engineering and Management for Multimodal Data | Tensor hierarchy, preprocessing-as-feature-engineering, five fusion architectures, four feature store patterns |
| ch06 | Data Storage and Retrieval for Multimodal AI | Vector DB retrieval loop, graph DB multi-hop reasoning, hybrid retrieval, three-tier caching (edge/hot/persistent) |
| ch07 | Streaming and Adaptive Pipelines for Multimodal AI | Event-time alignment, kappa architecture, windowed fusion, four drift types, safe retraining pipeline, federated learning |
| ch08 | Applying MLOps for Multimodal AI Applications | End-to-end MLOps lifecycle, reproducibility triplet, six-layer validation, multimodal observability stack |
| ch09 | Scaling Cloud-Native Multimodal AI Data Pipelines | Three execution frameworks (Beam/Spark/Ray), modality-specific scaling profiles, cross-modal dependency orchestration, five cost levers |
| ch10 | Model Deployment and Serving for Multimodal AI | Inference triangle, multistage inference pipelines, composite manifest, canary deployment, session-oriented serving (LBMs) |
| ch11 | Governance, Security, and Responsible Practices for Multimodal AI | Four governance foundations, detect-tag-redact PII, three security layers, multimodal explainability stack |
| ch12 | Putting It All Together: Building End-to-End Multimodal AI Pipelines | Five integration patterns, three orchestration responsibilities, three-layer observability, evolution patterns (shadow/dual-support/async bridge) |

## Topic Index

- **Agentic ingestion pipelines** → ch04
- **ANN indexing (HNSW/IVF/DiskANN/PQ)** → ch02, ch06
- **Autoscaling (modality-specific)** → ch09
- **Back-pressure control** → ch07, ch12
- **Bias detection** → ch11
- **Caching strategies** → ch06
- **Canary deployment** → ch08, ch10, ch12
- **CI/CD for multimodal AI** → ch08, ch10
- **Cost optimization** → ch09
- **Cross-modal drift** → ch07
- **Data contracts** → ch01, ch04, ch12
- **Data fabric vs. data mesh** → ch02
- **Data lineage / provenance** → ch11
- **Data sovereignty** → ch11
- **Dead letter queues (modality-aware)** → ch04
- **Decision bundle (selective retrieval)** → ch03
- **Drift detection** → ch07, ch08, ch10
- **Edge-cloud partitioning** → ch07, ch09, ch10
- **ELT vs. ETL** → ch04
- **Embedding compression (MRL, quantization)** → ch02
- **Entity resolution** → ch01
- **Explainability (multimodal)** → ch11
- **Federated learning** → ch07, ch11
- **Feature stores** → ch05, ch08
- **Feature engineering (tensor hierarchy)** → ch05
- **Fusion architectures** → ch05
- **Governance architecture** → ch11
- **Graph databases / knowledge graphs** → ch06
- **Hybrid retrieval** → ch06
- **Inference routing** → ch07, ch09, ch10
- **Kappa vs. lambda architecture** → ch07
- **Large behavior models (LBMs)** → ch08, ch10
- **Lineage and provenance** → ch08, ch11
- **Model optimization (quantization, distillation, pruning)** → ch10
- **Multistage inference pipelines** → ch10
- **Observability (three-layer)** → ch12
- **Orchestration (DAG, workflow)** → ch08, ch12
- **PII / privacy-preserving ML** → ch11
- **Platform archetypes** → ch03
- **Predictive/anticipatory autoscaling** → ch09
- **Retraining pipelines (automated)** → ch07, ch08
- **Scaling strategies (cloud-native)** → ch09
- **Schema contracts / versioning** → ch04, ch08, ch12
- **Security (pipeline)** → ch11
- **Semantic layer** → ch01
- **Session-oriented serving (LBMs)** → ch10
- **Shadow mode (new modality rollout)** → ch12
- **Streaming ingestion** → ch02, ch04, ch07
- **Vector databases** → ch02, ch06
- **Vector-native storage** → ch02

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — architecture/pipeline patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 12 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
