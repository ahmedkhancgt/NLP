# Cheatsheet: Data Engineering for Multimodal AI

## Decision: ELT vs. ETL
- Structured, stable schema, governance-heavy domain (healthcare billing, finance, payments)? → **ETL**
- Unstable schemas / embedding models change frequently / need to replay & re-embed history? → **ELT** (default for multimodal)
- Exception that always applies regardless: **mask/tokenize PII at ingestion**, never deferred.

## Decision: ANN Index Choice
| Bottleneck | Choose |
|---|---|
| Latency + balanced recall | HNSW |
| Budget / willing to trade recall for efficiency | IVF |
| Billion-scale, limited RAM | DiskANN |
| Edge/low-memory | Product Quantization (PQ) / binary quantization |

## Decision: Platform Archetype (managed / composable / streaming)
- Need speed, low team overhead, OK with vendor lock-in? → **Managed** (Bedrock/Vertex/Azure AI Foundry)
- Need full control over parsing/embedding/retrieval/governance? → **Composable** (Databricks Lakehouse AI)
- Need in-session, real-time decision latency regardless of the other choice? → **Streaming infra is non-negotiable** (Kafka/Flink)
- Volume rule of thumb: ~10K events/month → managed pay-per-call is simplest; ~500K+/month → composable's cost levers become essential.

## Decision: Fusion Architecture
| Situation | Use |
|---|---|
| Modalities complementary, need resolution preserved | Encoder-decoder |
| Modality reliability varies by context (lighting, noise) | Attention-based |
| Entities/relationships are the point | Graph-based |
| Modality intermittently missing | Generative (impute) or placeholder/null-token or bypass |
| Need cross-modal retrieval/alignment | Constraint-based (contrastive/CLIP-style) |

## Decision: Missing Modality Handling
1. Essential modality + semantic weight → **impute via generative model**
2. Optional modality, low semantic weight → **fixed placeholder (zero vector / null token)**
3. Modality effectively broken/unavailable → **bypass fusion, run reduced pipeline**

## Decision: Feature Store Pattern
- Just need train/serve consistency → **Dual-layer**
- Need similarity search over embeddings too → **+ Vector-augmented hybrid**
- Multiple teams/domains need discoverable, governed features → **+ Metadata-governed mesh**
- Need semantic reasoning over feature relationships → **+ Graph-integrated**

## Decision: Caching Tier
| Data type | Tier | TTL |
|---|---|---|
| Frequently reused, rarely changing reference (drone images, fault clusters) | Static (edge+cloud) | Days–weeks |
| Short-lived inference/session output (LLM summaries, RCA results) | Dynamic (agent memory) | Seconds–minutes |
Always try: edge → hot cache → persistent, in that order.

## Decision: Drift Type → Detection Technique
| Drift type | Detect with |
|---|---|
| Input (single modality) | Statistical monitor vs. training baseline, schema version checks |
| Cross-modal | MMD, HSIC, rolling Pearson/cosine on fused pairs |
| Latency-induced | `align_gap` histogram tail percentage |
| Semantic | Operator override rate over rolling window |
| Contextual (upstream API/partner) | Contract tests, heartbeat records |

## Decision: Modality-Specific Drift Metric
| Modality | Metric |
|---|---|
| Video / continuous embeddings | Fréchet distance |
| Audio spectrograms | KL divergence |
| Text / token sequences | Entropy / perplexity shift |
| Sensor vectors | Statistical shift (PSI/MMD) |

## Decision: Model Optimization for Deployment
| Need | Technique | Trade-off |
|---|---|---|
| Lowest-risk ~50% memory cut | FP16/BF16 | Negligible accuracy loss |
| Edge deployment, aggressive compression | INT8/INT4 quantization | Small accuracy drop |
| 5–20x size reduction | Knowledge distillation | Task-dependent accuracy impact |
| Remove whole neurons/heads | Structured pruning | Needs fine-tune after |
| Speed up without shrinking | Speculative decoding | None (different mechanism) |

## Decision: Inference Engine
- Fixed architecture, static shape (CNN classification) → **Compiled runtime (TensorRT)** — fast but fragile/inflexible
- Dynamic, variable-length, multi-user generative workload → **vLLM/SGLang/TensorRT-LLM** (paged attention, continuous batching)

## Decision: Scaling Lever by Modality
| Modality class | Scales on |
|---|---|
| Batch, GPU-heavy, predictable (imaging) | GPU memory, batch size |
| Continuous, CPU-light, latency-sensitive (telemetry) | Event rate, watermark lag |
| Sporadic, memory-bound (text/notes) | Concurrency, memory pressure |
| Bursty, latency-dominated (generative inference) | Queue depth, tail latency |

## Decision: Edge vs. Cloud Placement
- Preprocessing, lightweight/binary classification, must survive connectivity loss → **Edge**
- Cross-stream fusion, ensemble inference, retraining, long-window analytics → **Cloud**
- Rule of thumb: if network RTT eats a meaningful fraction of your latency SLO, push more compute to the edge.

## Decision: Cost Lever
| Workload | Lever |
|---|---|
| Training, tolerant of interruption | Preemptible/spot GPU |
| Non-urgent inference | Batch/off-peak + modal batching by encoder type |
| Rarely-used model, must stay available | Model parking (checkpoint streaming, <30s restore) |
| Raw media/logs aging out | Modality-aware TTL tiering (hot→nearline→coldline) |
| High-volume generative inference | Token budgeting by criticality class |

## Decision: Retraining Trigger
- Sustained (not spike) drift over a defined window → **Yes, trigger candidate retrain**
- Statistically different but operationally insignificant → **No** — encode operational-significance thresholds, not just statistical difference
- Always: fork from **last stable commit**, never the live buffer; promote only if candidate beats production baseline on ALL dimensions (shadow/canary gate).

## Tells and Smells (symptom → likely cause)
- "Agent guidance quality declining but no errors anywhere" → check upstream modality silently dropping records (e.g., codec mismatch) — instrument each lane, not just fusion output.
- "Fusion latency crept up after a model deploy" → check cross-modal correlation timeline; a slightly slower upstream stage can push a downstream stage past the fusion cutoff window.
- "Index returns technically valid but worse results over time" → stale/skewed vector index — doesn't throw errors, just quietly degrades; needs active index-health probing.
- "Model accurate in eval, wrong in production" → check representation/preprocessing layer first — "the bottleneck is almost never retrieval; it is almost always representation."
- "Two embeddings don't compare meaningfully" → check they're in the same embedding space/dimensionality (e.g., 512-dim image vs 384-dim text need a shared projection model like CLIP).
- "GPU sits idle waiting for input" → tune the data loader (modal-aware interleaving, prefetch), not the model — "if a GPU is waiting, the system is already losing ground."
- "Retrieval results too vague OR missing obviously relevant items phrased differently" → need hybrid retrieval (semantic + structured filter), not vector-only or keyword-only.
- "A regression appears only hours after a deploy" → check for a breaking upstream schema change (e.g., CRM API field) that doesn't throw an error but corrupts fusion context.

## Thresholds/Defaults Mentioned in the Book
- Fusion cutoff / late-arrival window: commonly 500ms–5s depending on SLA (FieldSense: 500ms; StreamKit screen+voice: 5s)
- Model entropy escalation threshold: ~0.45–0.55 (route to heavier/cloud model above this)
- Shadow model promotion: must beat baseline across ALL dimensions (latency, accuracy, acceptance) for a sustained window (StreamKit: 7+ consecutive days, >88% agreement)
- Canary traffic split: commonly starts at 10%, ramps to 50/50 before full promotion
- KB/content freshness bands: fresh <1hr, aging 1–24hr, stale >24hr
- Vector pairwise similarity sanity check: same-category items should show >0.6 similarity; dissimilar items <0.3; if average pairwise similarity across a batch is >0.9, embeddings aren't discriminative enough
