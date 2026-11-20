# Patterns

## Semantic Layer
**When to use**: Any system fusing 2+ modalities that must support cross-modal reasoning, not just metric lookup.
**How**: Layer ontological reasoning (entity/relationship types), entity resolution (deterministic rules → ML matching → human adjudication), and cross-modal integration (fusing text/image/audio/sensor into coherent events) on top of storage/processing layers.
**Trade-offs**: Adds engineering overhead vs. a simple analytics semantic layer, but without it, cross-modal joins produce garbage and agents hallucinate tool calls.

## Data Contract
**When to use**: At ingestion, for every modality, before data reaches the feature layer.
**How**: Define required fields (entity_id, modality, timestamp, provenance, quality_score), validation rules, and violation handling (reject/log/quarantine — never silent failure).
**Trade-offs**: Rigid contracts slow onboarding of new data sources but prevent silent corruption downstream; PII masking must happen at ingestion regardless of the ELT-first principle.

## Vector-Native Storage + ANN Indexing
**When to use**: Any system needing similarity search, nearest-neighbor queries, or cross-modal retrieval.
**How**: Encode modalities to embeddings, index with HNSW (latency/recall balance), IVF (efficiency over recall), or DiskANN (billion-scale affordability); combine with metadata filters for hybrid queries.
**Trade-offs**: Each index trades memory cost, search speed, and recall differently — choose by bottleneck (latency vs. scale vs. budget).

## Composable Data Lakehouse
**When to use**: As the enterprise system of record complementing real-time vector stores.
**How**: Cloud object storage (transactional table format) → indexing layer (vector/full-text/metadata) → compute layer (streaming/cache/distributed) → unified API layer.
**Trade-offs**: More operational surface than a single managed service, but gives full control over parsing, embedding, chunking.

## Staged Retrieval (Filter → ANN → Rerank)
**When to use**: Balancing cost, latency, and precision across billions of embeddings.
**How**: Cheap metadata filter shrinks candidate pool → ANN index surfaces top-k by similarity → expensive reranker (cross-modal scorer or LLM) finalizes order.
**Trade-offs**: Added latency per stage vs. dramatically better relevance and lower compute than scoring everything with the expensive reranker.

## Three Platform Archetypes (Managed / Composable / Streaming)
**When to use**: Choosing where multimodal context-layer complexity should live.
**How**: Managed (Bedrock/Vertex/Azure AI Foundry) for speed, pay-per-call pricing; Composable (Databricks Lakehouse AI) for full control, explicit cost levers; Streaming (Kafka/Flink) for continuously updated, replayable session state, always-on cost.
**Trade-offs**: Managed trades control for speed; composable trades engineering time for flexibility; streaming is non-negotiable when in-session real-time decisions are a regulatory/business requirement.

## Representation → Temporal Assembly → Selective Retrieval
**When to use**: Universal checklist for any production multimodal context layer.
**How**: Transform raw artifacts into typed, schema-conformant objects with provenance (representation); incrementally update session state as events arrive, never recompute (temporal assembly); construct a bounded, ranked, redacted decision bundle rather than passing full history (selective retrieval).
**Trade-offs**: Skipping any one degrades to either incoherent joins, race conditions/drift, or context-overwhelmed/noisy model inputs.

## ELT over ETL for Multimodal
**When to use**: Default baseline for multimodal pipelines; ETL remains appropriate for governance-heavy, stable-schema domains (payments, strict healthcare validation).
**How**: Land raw multimodal signals first (preserving fidelity/replayability); apply AI-driven transformations on demand/late-bound; except PII, which must be masked at ingestion regardless.
**Trade-offs**: ELT preserves optionality as models/schemas evolve but requires a versioned schema contract layer to govern transformation-boundary changes.

## Modality-Aware Dead Letter Queue (DLQ)
**When to use**: Whenever validation can fail silently in ways that corrupt downstream embeddings/models.
**How**: Partition DLQs by modality; each failed record carries a reason code; build a reprocessing pipeline that can retry intelligently after model upgrades/threshold adjustments/manual review.
**Trade-offs**: More infrastructure than a single error table, but prevents silent corruption of training data and enables targeted remediation.

## Five Fusion Architectures
**When to use**: Choosing a fusion strategy by workload shape — encoder-decoder (complementary modalities needing resolution preservation), attention-based (modality reliability varies by context), graph-based (entities/relationships are primary), generative (handling intermittent missing modalities), constraint-based (cross-modal retrieval/alignment).
**How**: Match the fusion category to the reasoning task rather than defaulting to concatenation; design the data-engineering responsibility (time-aligned embedding delivery, attention diagnostics, graph schema versioning, conditioning-input supply, contrastive-pair tracking) per architecture.
**Trade-offs**: More expressive fusion (graph, generative) costs more engineering/compute than simple concatenation but is necessary when missing modalities or relational reasoning are core requirements.

## Feature Store Maturity Progression
**When to use**: Progress through patterns as modality diversity and team scale grow — dual-layer (most production needs) → vector-augmented hybrid (semantic recall needed) → metadata-governed feature mesh (cross-team discoverability) → graph-integrated (semantic reasoning over feature relationships).
**How**: Start with offline/online stores sharing transformation code; add a vector store as a peer when embeddings dominate; federate with a central registry as teams multiply; link embeddings via a knowledge graph when reasoning over feature relationships is required.
**Trade-offs**: Each step adds governance/complexity overhead; jumping straight to the most advanced pattern without the organizational need wastes engineering effort.

## Vector + Graph + Hybrid Retrieval
**When to use**: Vector alone for "find similar"; graph alone for "how are these connected"; hybrid (both + metadata filters + reranking) for real enterprise operational queries.
**How**: Run dual-phase retrieval — semantic recall via ANN, then metadata-aware filtering/reranking (LLM-based or rule-based) — rather than relying on cosine similarity against a flat index alone.
**Trade-offs**: Hybrid retrieval is more engineering work but is "what separates production systems from demos."

## Three-Tier Caching (Edge / Hot / Persistent)
**When to use**: Any agentic system needing fast, context-rich, cost-controlled recall.
**How**: Try edge cache first (submillisecond, colocated); escalate to cloud hot cache (in-memory, cross-zone recent results); fall back to persistent cache (lifecycle-managed object storage, feeds retraining) only when necessary.
**Trade-offs**: More cache tiers to manage and invalidate, but avoids both idle-GPU waste and unacceptable latency on every request.

## Event-Time Alignment + Kappa Architecture
**When to use**: Always, for multimodal stream fusion at different sampling rates.
**How**: Bind joins to the authoritative event_ts carried in every message (not arrival order); use a single append-only replayable log (kappa) rather than dual batch/speed layers (lambda).
**Trade-offs**: Requires disciplined timestamp propagation and watermark management, but is the only approach that scales when joining streams at wildly different rates (e.g. 30Hz video vs 1Hz sensor).

## Windowed Fusion (Tumbling vs. Sliding) with Fallback
**When to use**: Any temporal join of asynchronous multimodal streams.
**How**: Use sliding windows with a hop when SLA is tight (fresher pairings); tumbling windows when simpler/looser latency tolerance suffices; always route unmatched records to an explicit fallback path (e.g., single-modality inference) rather than dropping them.
**Trade-offs**: Tighter windows improve freshness but increase the fallback rate; missing-modality handling must be a first-class design decision, not an afterthought.

## Layered Drift Monitoring
**When to use**: Every production multimodal pipeline, continuously.
**How**: Instrument four boundaries independently — input drift (per-modality distribution vs. baseline), cross-modal drift (correlation decay via MMD/HSIC/rolling Pearson), latency-induced drift (align_gap histogram tail), semantic drift (operator override rate) — rather than one aggregate health score.
**Trade-offs**: More instrumentation surface, but necessary because each drift type is invisible to the others and requires a different detection technique.

## Safe Retraining via Shadow/Canary Gate
**When to use**: Every model update in a production multimodal system.
**How**: Candidate model runs on real traffic without affecting users (shadow) → compared head-to-head on latency/accuracy/acceptance → promoted only if it beats the baseline across all dimensions, else logged and the production model stays.
**Trade-offs**: Slower rollout than direct deployment, but eliminates the risk of silently degrading production with an undertested model.

## Edge-Cloud Partitioning
**When to use**: Deciding where computation should happen based on latency budget, bandwidth, and resilience to connectivity loss.
**How**: Preprocessing + lightweight inference run at the edge (must survive intermittent connectivity); fusion, ensemble inference, retraining, and analytics run in the cloud (assume full context available); a message bus is the synchronization boundary.
**Trade-offs**: Edge compute is constrained and must be explicitly profiled for worst-case load; cloud-only designs fail when connectivity degrades at exactly the moment a decision is needed.

## Federated Learning Lifecycle
**When to use**: When raw data cannot be centralized (privacy, bandwidth, partner constraints) but local adaptation is still needed.
**How**: Local training on-device → transmit only signed model deltas (never raw data) → aggregate with weighting by freshness/reliability/coverage (FedAvg/FedProx) → shadow-evaluate the merged model → promote via OTA only if it beats baseline, else roll back.
**Trade-offs**: Shifts the data-engineering challenge from data movement to decentralized orchestration (modality-coverage weighting, cross-device compatibility, update integrity).

## Modality-Specific Autoscaling + Cross-Modal Dependency Orchestration
**When to use**: Any multimodal pipeline where modalities have different cadences/compute profiles AND a single inference depends on multiple of them arriving together.
**How**: Scale each modality independently on its own signal (GPU batch size for imaging, watermark lag for telemetry, memory pressure for text, queue depth for inference); additionally gate scaling decisions on cross-modal readiness (fused-readiness markers, pub/sub checkpoint handshakes, modal contract validators) rather than system metrics alone.
**Trade-offs**: More complex autoscaling logic than a uniform CPU-threshold policy, but prevents both downstream starvation (idle GPUs waiting on lagging modalities) and premature eviction (partial data propagating).

## Cost Optimization Levers (5)
**When to use**: Design-time cost discipline embedded at every stage, not a post-hoc billing audit.
**How**: Preemptible training (reserved for critical phases, spot for opportunistic); inference tiering by urgency/criticality; model parking (scale idle models to zero, restore via checkpoint streaming); storage-aware TTL tiering per modality; token-efficient inference (criticality-based budgets).
**Trade-offs**: Requires upfront classification of workloads by true latency/criticality requirement — misclassifying a workload as real-time when it can tolerate batch wastes an order of magnitude of spend.

## Memory-Tiered Sharding + Shard-Aware Scheduling
**When to use**: Distributed multimodal training/serving clusters with heterogeneous memory footprints per modality.
**How**: Classify shards (heavy/medium/light) by modality/tensor size/decode cost BEFORE dispatch; prewarm GPU memory blocks based on a request's memory signature hash so allocation is planned before the batch reaches the GPU.
**Trade-offs**: Added classification overhead vs. eliminated fragmentation, out-of-memory crashes, and reduced throughput variance.

## Predictive (Anticipatory) Autoscaling
**When to use**: On top of reactive serverless elasticity, when upstream signals (schedules, census, dispatch patterns) predictably precede demand surges.
**How**: Combine rule-based deterministic triggers with LLM-based cross-signal correlation in a conservative decision loop (confidence threshold + time window + resource-cost gating) to prewarm GPU containers before demand materializes.
**Trade-offs**: Risk of wasted prewarming if predictions are wrong — mitigated by conservative confidence/time-window gating; closes the cold-start latency gap that pure reactive elasticity cannot.

## Composite Manifest Versioning
**When to use**: Any deployed multimodal system with multiple heterogeneous components (encoders, fusion model, preprocessing, serving config).
**How**: Version every component together as one immutable bundle with an integrity hash; store in a model registry; trace every telemetry record/rollback trigger back to a specific manifest configuration.
**Trade-offs**: More upfront schema design than versioning model weights alone, but is what makes multimodal canary/rollback/audit tractable.

## Multimodal Canary with Single-Component Isolation
**When to use**: Updating one component (e.g., a video encoder) of a composite multimodal model.
**How**: Split traffic (e.g., 90/10) between stable and candidate manifests, isolating the change to ONE modality's component to detect cross-channel fusion artifacts, not just aggregate accuracy.
**Trade-offs**: More granular (and slower) than conventional A/B testing, but catches fusion-specific degradation that whole-model A/B testing misses.

## Graceful Degradation Rollback
**When to use**: Any detected drift or failure in a live multimodal serving system.
**How**: If drift is isolated to one modality, drop only that modality and serve a reduced-modality fallback; only execute full rollback to last-known-good manifest when drift is multi-modal/simultaneous.
**Trade-offs**: Requires designing an explicit intermediate degraded state for every modality combination, but avoids unnecessary full outages for partial failures.

## Governance-as-Data-Property
**When to use**: Any pipeline handling regulated, sensitive, or auditable multimodal data.
**How**: Assign a governance event ID at ingestion that anchors all downstream metadata; enforce policy tags (compliance, region) that travel with the data through every stage; query the catalog before loading any asset into training/inference; reconstruct full asset journeys on demand for audit.
**Trade-offs**: More upfront metadata engineering, but is what turns a forensic, weeks-long investigation into a query-driven review resolved in hours.

## Detect-Tag-Redact for PII
**When to use**: At ingestion, for every modality, before data enters the feature layer.
**How**: Automated detection scans incoming assets → policy tags mark PII fields → redaction/pseudonymization runs before data crosses into training/serving.
**Trade-offs**: Adds ingestion-time latency/compute, but "the cheapest data to govern is data you never ingested" — far cheaper than post-hoc remediation or breach exposure.

## Multimodal Explainability Stack
**When to use**: Any high-stakes multimodal decision requiring regulator, clinician, or auditor-facing explanation.
**How**: Cover all three dimensions — per-modality attribution (what did each input contribute), cross-modal interaction (how did modalities influence each other), temporal explainability (which features mattered, and when) — rather than relying on a single technique like SHAP or Grad-CAM alone.
**Trade-offs**: More tooling/compute to generate three explanation types instead of one, but a single-dimension explanation misses cross-modal and temporal failure modes entirely.

## Isolation + Contracts for Production Multimodal Pipelines
**When to use**: Any always-on, mission-critical multimodal system (contact centers, robotics, healthcare operations).
**How**: Give each modality its own ingestion lane, queue, and autoscaling policy; enforce versioned, immutable schema contracts at every fusion boundary; split inference into tiers by latency sensitivity (edge/batch/analytics).
**Trade-offs**: More operational surface (more queues, more contracts to version) than a monolithic pipeline, but is the only design that survives real production load without cascading failures.

## Three-Layer Observability with Trace-Back
**When to use**: Any multimodal production system where failures can originate in multiple independent lanes and cascade through fusion.
**How**: Instrument each modality lane independently (Layer 1), the fusion boundary (Layer 2), and inference outputs (Layer 3); when a symptom appears at Layer 3, trace back through Layer 2 to find the root cause at Layer 1.
**Trade-offs**: Requires per-stage instrumentation discipline, but is the only way to convert a generic "something is slow" alert into an actionable, enriched diagnosis.

## Shadow Mode → Canary → Production for New Modalities
**When to use**: Introducing any new, unproven modality or model into a live system.
**How**: Run the new component in parallel, consuming the same inputs but not influencing live decisions; evaluate against explicit sustained criteria (agreement rate, latency, degradation tolerance) over a defined window; promote to a small canary traffic percentage before full production.
**Trade-offs**: Slower time-to-production than direct deployment, but is the only safe path to validate an unproven component's accuracy and failure modes against real traffic.

## Dual-Support Deprecation + Async Upgrade Bridge
**When to use**: Upgrading a model whose output format or embedding dimensionality changes in ways that break downstream dependents.
**How**: Run old and new models side by side with an adapter layer translating between formats; migrate traffic gradually via A/B flag with metrics-based stop conditions (dual-support deprecation); for embedding-dimension changes, use a learned projection matrix so fusion can accept either format while preferring the newer (async upgrade bridge).
**Trade-offs**: Slower than a direct cutover, but avoids rewriting and revalidating a large body of dependent downstream logic all at once.
