# Glossary

**Align Gap** — the delta between event timestamps of joined records (e.g., frame_ts − soil_ts); the primary health signal for temporal joins (Ch 7, Ch 9).

**Alignment DAG** — a declarative, versioned graph of interpolators/time-window joins/sequence synchronizers encoding how modalities are temporally fused (Ch 8).

**ANN (Approximate Nearest Neighbor)** — indexing method (HNSW, IVF, DiskANN, PQ) trading small recall loss for dramatically better performance/scalability over billions of embeddings (Ch 2, Ch 6).

**Behavioral Ontology Trace** — manifest metadata for large behavior models recording session window, schema version, modality weights, and detected behaviors — the audit answer to "why did the model think X" (Ch 10).

**Causal Attribution Map** — gradient-based explainability (integrated gradients, layer-wise relevance propagation) extended across modalities, annotating each inference with per-input-stream contribution (Ch 10).

**Composite Manifest** — a versioned, immutable bundle describing every heterogeneous component of a deployed multimodal system (encoders, fusion model, preprocessing, serving config) (Ch 10).

**Context Budgeting** — explicitly constraining an assembled decision bundle to fit model token limits and latency targets (Ch 3).

**Context Intelligence (Context Engineering)** — the data architecture transforming unstructured/structured data into synchronized, retrievable, auditable session state that agents/models reason over (Ch 3).

**Copy-on-Write (COW) Data Lake** — storage pattern where modifications create new versions while preserving originals, enabling atomic snapshots and rollback (Ch 8).

**Cross-Modal Drift** — decay in the statistical relationship between modalities (e.g., correlation between NDVI and soil moisture) even when each modality looks individually healthy (Ch 7).

**Data Contract** — a formal guarantee about shape, content, and quality of every record entering a system: required fields, validation rules, violation handling (Ch 1).

**Data Fabric** — unifying architectural layer providing centralized, automated integration and governance across distributed heterogeneous data (Ch 2).

**Data Mesh** — decentralized data architecture where domain teams own and expose data as products via APIs, with federated governance (Ch 2).

**Decision Bundle** — a bounded, reproducible, policy-safe slice of session state (authoritative state + evidence + audit trace + constraints) handed to a reasoning model (Ch 3).

**Detect-Tag-Redact Pattern** — the universal PII handling pattern: automated detection, policy tagging, then redaction/pseudonymization before data enters the feature layer (Ch 11).

**Disaggregated Prefill-Decode Serving** — splitting LLM inference into a compute-bound prefill phase and a memory-bound decode phase, each with its own scaling pool (Ch 9).

**Edge-Cloud Partition Framework** — architectural split where everything above a message bus must survive intermittent connectivity (edge: preprocessing, lightweight inference) and everything below assumes full context (cloud: fusion, retraining) (Ch 7).

**Entity Resolution** — linking disparate identifiers for the same real-world entity across systems, using deterministic rules, ML-based matching, and human adjudication (Ch 1).

**Federated Fine-Tuning** — downloading a pretrained base model and fine-tuning locally with parameter-efficient methods (LoRA/adapters), sharing only lightweight adapter weights (Ch 11).

**Feature Store** — a centralized repository ensuring batch training and real-time inference use the same feature definitions, preventing training-serving skew (Ch 5).

**Feature Store Patterns (4)** — dual-layer, vector-augmented hybrid, metadata-governed feature mesh, graph-integrated — progressive architectures for multimodal feature serving (Ch 5).

**Fused-Readiness Coordination** — gating a batch framework's ingestion on an end-of-window marker written by the stream layer, so autoscaling aligns with multimodal data completeness, not just metrics (Ch 9).

**Fusion Architectures (5)** — encoder-decoder, attention-based, graph-based, generative, constraint-based; the modern taxonomy replacing early/late fusion (Ch 5).

**Governance Event ID** — a single identifier assigned at ingestion anchoring all downstream metadata (frames, OCR, telemetry, model manifest, feature snapshot) for one multimodal event (Ch 11).

**Inference Triangle** — the three-way tension between data preparation, latency budget, and compute architecture governing every multimodal serving decision (Ch 10).

**Kappa Architecture** — single append-only log processed by stream workers in real time, replayable for reprocessing; the event-driven alternative to lambda architecture for multimodal fusion (Ch 7).

**Lambda Architecture** — dual-layer batch + speed processing pattern; works for homogeneous data but creates latency gaps unsuitable for multimodal fusion (Ch 7).

**Late-Arriving Data Problem** — notes/labs/signals appearing after the event they describe, endemic to healthcare-style domains (Ch 1).

**Lineage Chain (Append-Only)** — cryptographically-linked lineage events (hash chain) where tampering breaks the chain; corrections are new revision events, not overwrites (Ch 11).

**Memory-Tiered Sharding** — classifying GPU workload shards by modality/size/decode cost into heavy/medium/light tiers BEFORE dispatch to prevent fragmentation (Ch 9).

**Model Card / Datasheet** — governance artifacts documenting a trained model's intended use/limitations (card) and a dataset's provenance/composition (datasheet) (Ch 11).

**Model Parking** — intentional dormancy: idle models scaled to zero replicas, restorable via checkpoint streaming in under 30 seconds (Ch 9).

**Modal Contract** — latency tolerances, input shape, and quality requirements codified per-modality as part of pipeline design, not bolted on afterward (Ch 10).

**Modal Tag / Header Contract** — the universal metadata (event_ts, field_id/session_id, modal_tag) every streaming message carries so downstream operators can key, watermark, and deduplicate (Ch 7, Ch 12).

**Modality-Specific Scaling Profile** — the formal relationship among a modality's input characteristics, compute demands, and required autoscaling behavior (Ch 9).

**Multimodal Explainability Stack** — three dimensions (per-modality attribution, cross-modal interaction, temporal explainability) needed for a complete multimodal decision explanation (Ch 11).

**Multistage Inference Pipeline** — architecture where one model's output becomes structured input to the next (entity detection → sequencing → threshold analysis → fusion/decision) (Ch 10).

**On-the-Wire Contract** — a versioned, immutable schema published by a modality before data crosses the fusion boundary, enabling safe evolution and rollback (Ch 12).

**Point-in-Time Correctness** — a model at inference should see only features that would have been available at that exact point in time (Ch 5).

**Predictive (Anticipatory) Autoscaling** — prewarming GPU containers based on upstream signal prediction (rules + LLM cross-signal reasoning) rather than reacting to incoming load (Ch 9).

**Reproducibility Triplet** — data hash + code hash + config hash; the minimal unit of experiment identity in multimodal MLOps (Ch 8).

**Right to Erasure (Multimodal)** — deletion requests that must cascade across every modality and derived artifact (fused frames, embeddings, model memorization), tractable only with lineage tracking (Ch 11).

**Selective Context Retrieval** — constructing a bounded decision bundle from structured queries + semantic search + metadata filters rather than passing full session state to a model (Ch 3).

**Semantic Drift** — drift in the meaning of data itself, surfaced through human feedback signals like operator override rates rather than statistical distribution shifts (Ch 7).

**Semantic Layer (Expanded)** — architecture adding ontological reasoning, entity resolution, and cross-modal integration atop analytics-style metric consistency (Ch 1).

**Session-Oriented Serving Loop** — stateful LBM serving where a context cache carries state across timesteps, turning a frame classifier into a behavior-aware system (Ch 10).

**Shadow Mode** — running a new modality/model in parallel with production without influencing live decisions, promoted only after meeting explicit sustained performance criteria (Ch 12).

**Shard-Aware Scheduling** — prewarming GPU memory blocks based on a request's memory signature hash before dispatch, avoiding cache swaps/fragmentation (Ch 9).

**Staged Retrieval Pipeline** — metadata filter → ANN top-k → reranker; the standard multi-pass pattern balancing cost, latency, precision (Ch 2).

**Temporal State Assembly** — incrementally updating session state as asynchronous events arrive, never recomputing from scratch (Ch 3).

**Tensor Hierarchy** — scalar → vector (rank-1) → matrix (rank-2) → rank-3/4/5; the unifying representation making all modalities interoperable (Ch 5).

**Tiering by Frequency** — separating components by update cadence (continuously/episodically/frozen) to limit blast radius and enable independent scaling — recurs across model updates, request routing, and governance (Ch 10).

**Tolerance-Based Inner Join** — a composite training example is only produced when every required modality provides a valid payload within its time window; missing payload means no example at all (Ch 8).

**Watermark** — a streaming-system marker advancing conservatively on event time, used to seal windows and divert late arrivals to a side output rather than forcing recomputation (Ch 7).
