# Chapter 10: Model Deployment and Serving for Multimodal AI

## Core Idea
Real-time multimodal inference shifts data engineering into "inference engineering" — serving the model is only ~20% of the problem; the other 80% is engineering modality coherence, temporal alignment, and observability under strict latency budgets — illustrated through Eighteen Labs, a fictional company deploying large behavior models (LBMs) in enterprise robotics.

## Frameworks Introduced
- **The Inference Triangle**: three corners in constant tension — data preparation (how much work to get raw inputs model-ready), latency budget (the countdown clock), compute architecture (available hardware). Every choice shifts the balance; there is no free lever.
  - When to use: as the mental model for every multimodal serving design decision — heavier preprocessing costs latency, lower latency constrains model complexity, cheaper hardware degrades fusion quality.
- **Five Canonical Inference Subsystems**: Ingestion and alignment (capture + common time axis) → Preprocessing (resize/denoise/normalize to model-ready form) → Encoding (modality-specific encoders producing dense embeddings) → Fusion (temporal fusion graph — nodes are timestamped multimodal events, edges are cross-modal attention weights, bounded by a sliding session window) → Retrieval and output (ranked, timestamped, actionable, traceable results).
  - When to use: as the reference decomposition for building any real-time multimodal inference pipeline, regardless of domain.
- **Multistage Inference Pipeline Pattern**: output of one model becomes structured input/prompt to the next (e.g., Stage 1 entity detection → Stage 2 temporal sequencing → Stage 3 threshold analysis → Stage 4 fusion/decision, with all earlier stages also feeding Stage 4 directly).
  - When to use: for layered reasoning tasks (narrative interpretation, hierarchical labeling) — requires strict schema contracts between stages, interstage latency management, and explicit failure-propagation policies (retry? fall back to fewer modalities? skip?).
- **Monitoring Architecture (3 parallel logging streams)**: input logger (full-fidelity raw inputs + metadata) + prediction logger (output distributions, confidence, per-modality latency) + feedback collector (post-prediction outcomes, overrides) → converge into observability stack (real-time alerting + retrospective analytical store).
  - When to use: always, for deployed multimodal models — "a prediction log alone just tells you what went wrong. The input log tells you why, and the feedback log tells you how often it matters."
- **Drift Detection via Embedding Space Monitoring**: frozen encoders compute embeddings over time per modality; divergence measured via PSI or MMD against the training distribution; crossing a threshold fires an alert and flags the batch for review.
  - When to use: continuously post-deployment — each modality channel can drift independently and silently degrade overall fused performance.
- **Shadow Evaluation and Promotion Workflow**: production traffic forks to current model + candidate model (shadow mode, doesn't affect users) → KPI comparison (latency, accuracy, operator acceptance) → promote only if candidate wins across all dimensions, else retain current and flag for review.
  - When to use: every model update — never risk degrading production quality by requiring the candidate to "earn its place" on real traffic before promotion.
- **Three-Tier Model Update Cadence**: Continuously updated (streaming feedback, e.g. speech-to-text) / Episodically updated (scheduled, e.g. monthly) / Frozen (stable, manual-review-only, e.g. low-level telemetry encoders). This is a recurring "tiering by frequency" design principle that reappears in request-routing tiers and governance tiers.
- **Canary Deployment for Composite Models**: traffic split (e.g., 90/10) between stable and candidate manifests; multimodal canarying often isolates a SINGLE modality's component (e.g., swap only the video encoder) to detect whether an improvement in one channel degrades overall fusion quality — granularity that distinguishes it from conventional A/B testing.
- **Rollback Decision Graph (graceful degradation)**: performance monitor → if within threshold, continue; if not, drift detection asks "which modality drifted?" → isolated drift: drop that modality, serve reduced-modality fallback; multi-modality drift: full rollback to last known-good manifest. Never goes straight from fully operational to fully down.
- **Session-Oriented Serving Loop (for LBMs)**: at each timestep, new multimodal input + context cache (serialized state from T-1) converge at the LBM inference node, producing a prediction AND an updated context written back for T+1 — turns a frame-level classifier into a behavior-aware system that detects patterns unfolding over time.
- **Dual-Path Architecture (shallow + deep inference)**: shallow inference head gives rapid, lower-confidence predictions on short-context input (real-time alerts); deep inference head asynchronously processes longer sequences for higher-confidence refined predictions (feeds analytics/retraining). Resolves the latency-vs-behavioral-depth trade-off by serving both simultaneously rather than choosing one.

## Key Concepts
- **Modal Contract**: latency tolerances, input shape, and quality requirements codified per-modality as part of the pipeline design, not bolted on as runtime configuration afterward.
- **Composite Manifest**: a versioned, immutable YAML bundle describing every component of a deployed system (encoders per modality with quantization/target hardware, fusion model, preprocessing pipeline version, ontology, serving framework config) — the unit of reproducibility and rollback.
- **Causal Attribution Map**: gradient-based explainability (integrated gradients, layer-wise relevance propagation) extended across modalities, annotating each inference with per-input-stream contribution to the final prediction — used both for debugging and as an implicit adversarial defense (cross-modal consistency check: if OCR contradicts vision, flag as low-confidence and route to human review).
- **Action-Validation Gate**: a runtime guardrail sitting between an LBM's output and a robot's motor controller, checking predicted commands against safe operational envelopes/proximity zones/confidence thresholds before execution — downgrades to safe default and flags for human review on failure (relevant to EU AI Act high-risk classification for ML-based safety functions).
- **Length-Bucketed Batching**: grouping multimodal inference requests by similarity in sequence length/modality presence to avoid massive zero-padding overhead when naively batching variable-length sequences (e.g., 48 video frames vs. 17 vibration tokens).
- **Tiered Serving Hardware Architecture**: preprocessor pods (CPU-optimized) → encoder pods (mid-tier GPUs, T4/L4) → fusion inference pods (high-memory A100/H100) — cost efficiency from matching hardware tier to compute profile at each stage.
- **Tiered Request Routing**: Tier-1 (high-accuracy, high-cost, full-fidelity) / Tier-2 (approximate, downsampled) / Tier-3 (deferred, batch/archival) — context-aware load balancing assigns requests dynamically based on system load, operator priority, latency SLA breaches.
- **Compute Receipt**: metadata logged per inference (inference time, accelerator class, energy estimate, model version) enabling downstream teams to query the true cost of any prediction.
- **Behavioral Ontology Trace**: extended manifest metadata for LBMs recording session window, schema version, modality weights, and inferred behavior structures — the audit answer to "why did the robot think X was happening?"

## Mental Models
- **"Serving the model is only 20% of the problem. The other 80% is engineering modality coherence, temporal alignment, and system observability."**
- The inference pipeline has a **funnel shape**: wide at the top (heterogeneous raw streams), narrow in the middle (fusion/synchronization), narrow out to a single semantic response — what looks like ETL is fundamentally different because of the demand for temporal coherence and subsecond round-trip latency.
- **Multistage pipelines mirror human cognition**: recognize entities → infer interactions → contextualize in time → explain/respond — but each stage must produce schema-compliant output or drift becomes "no longer isolated," silently degrading every downstream stage.
- **The deployment progression is "serve a model" → "orchestrate a reasoning graph"** — this is the core shift the chapter asks the reader to internalize; everything in monitoring, scaling, and lifecycle management builds on this reframing.
- **Closed loop as operating system**: observe → detect → attribute → retrain → promote. Remove any link and the chain breaks — no telemetry means no drift detection, no drift detection means no retraining trigger, no shadow evaluation means no safe promotion, no causal attribution means no explainability when it matters most.
- **Dual-path serving is "not choosing between speed and accuracy; you're serving both simultaneously"** — the shallow path buys time, the deep path delivers the verdict.
- **Graceful degradation, never binary failure**: the rollback decision graph always has an intermediate fallback state (drop one modality) before resorting to full rollback — "you never go from 'fully operational' to 'completely down.'"

## Anti-patterns
- **Treating multimodal serving as a simple model-as-a-service problem**: misses that 80% of the engineering effort is in modality coherence and alignment, not model invocation.
- **Letting a mismatch in structured output (inconsistent types, missing keys) between pipeline stages pass silently**: can silently break the downstream model in a multistage pipeline.
- **Naive batching across modalities with wildly different sequence lengths**: causes massive zero-padding memory overhead; use length-bucketed batching with dynamic masking instead.
- **Promoting a candidate model without shadow evaluation against real production traffic**: risks degrading production quality with only benchmark-dataset validation.
- **Canarying a composite multimodal model the same way you'd A/B test a single model**: misses fusion artifacts — isolate changes to a single modality's component to detect cross-channel degradation.
- **Full rollback when drift is isolated to one modality**: unnecessarily sacrifices availability — drop only the drifting modality and serve a reduced-modality fallback first.
- **Using a one-size-fits-all accelerator tier for every pipeline stage**: wastes budget running video decoding on expensive A100s or running cross-modal attention on CPU nodes — match hardware tier to the compute profile of each stage.

## Code Examples
```yaml
# Composite manifest (Eighteen Labs) - the unit of versioning and rollback
manifest:
  version: "2.4"
  status: PRODUCTION
  integrity_hash: "sha256:a3f8b2c91d..."
encoders:
  video: { model: resnet50_v3, quantization: FP16, target: edge-tpu }
  audio: { model: whisper_v2.1, quantization: INT8, target: mid-tier-gpu }
  text: { model: bert_ocr_v1.2, quantization: FP16, target: mid-tier-gpu }
  telemetry: { model: ffn_v1.0, quantization: FP32, target: cpu }
fusion:
  model: xattn_v2.4
  target: high-memory-gpu
serving:
  framework: triton
  max_batch_size: 8
  timeout_ms: 2000
  fallback_policy: drop_modality
```
- **What it demonstrates**: every heterogeneous component (encoders, fusion, serving config) versioned together as one reproducible, auditable bundle — not just model weights.

```yaml
# Behavioral ontology trace attached to each high-value LBM inference
behavioral_trace:
  session_window: 300s
  schema_version: behavior_v3.1
  modality_weights: { video: 0.40, audio: 0.25, telemetry: 0.20, text: 0.15 }
  behaviors_detected: [fan_degradation_sequence, operator_intervention_pattern, thermal_anomaly_progression]
  confidence: 0.87
  attribution_hash: "sha256:7b2f..."
```
- **What it demonstrates**: the audit-ready answer to "why did the model think this was happening" — session window, per-modality influence, detected behaviors, and a hash linking back to the exact explainability computation.

```python
# model.py - MultimodalInferenceModel (RLBench playbook)
# ResNet18 vision encoder (frozen, classification head removed) -> 512-dim features
# + 7 telemetry (joint position) features -> fusion: 519 -> 256 -> 256 -> 7 (ReLU)
# Output: 7 joint commands (one per robot arm DOF)
```
- **What it demonstrates**: a minimal but faithful vision+telemetry fusion architecture mirroring the production pattern (modality-specific encoder + concatenation fusion + task head).

```bash
python simulate_canary.py --max-samples 5 --canary-traffic 20.0
# Baseline v1.0.0 vs Candidate v1.1.0, 80/20 traffic split
# Avg action divergence: 0.023 -> Decision: PROMOTE (all thresholds passed)
```
- **What it demonstrates**: canary promotion gated on comparative latency, success rate, AND action divergence (how much the candidate's predictions differ from baseline on the same inputs) — not accuracy alone.

## Reference Tables
**Table 10-1 (partial, reconstructed): Model optimization techniques**
| Technique | Size reduction | Accuracy impact | Deployment target |
|---|---|---|---|
| FP16/BF16 mixed precision | ~50% memory | Negligible | Any GPU |
| INT8/INT4 quantization | Progressively larger | Small drop | Edge deployment |
| Knowledge distillation | 5x–20x | Task-dependent | Small/student models |
| Structured pruning | Variable | Needs fine-tune after | GPU/edge |
| Speculative decoding | None (speeds inference, doesn't shrink model) | None | Latency-sensitive serving |

**Tiered Serving Hardware**
| Tier | Hardware | Role |
|---|---|---|
| Preprocessor pods | CPU-optimized nodes | Video decode, audio conversion, tokenization |
| Encoder pods | T4/L4 GPUs | Convert raw frames/audio into embeddings |
| Fusion inference pods | A100/H100 (high-memory) | Cross-modal attention over fused embedding space |

**Tiered Request Routing**
| Tier | Fidelity | Use case |
|---|---|---|
| Tier-1 | High-accuracy, high-cost, full-resolution | Live operator display, critical detection |
| Tier-2 | Approximate (downsampled/truncated) | Real-time feedback, A/B testing of lighter models |
| Tier-3 | Deferred | Batch diagnostics, archival analysis |

## Worked Example
**Eighteen Labs Multimodal Inference Service Playbook**: Built on RLBench robot-manipulation demonstrations (front/wrist/overhead RGB images + proprioceptive state in `low_dim_obs.pkl` with joint positions, gripper state). Restricted to front camera + joint positions to keep the exercise lightweight while still exercising visual+proprioceptive fusion. Steps: (1) validate a YAML configuration manifest (version, vision=resnet18, telemetry=joints_v1, fusion=concat, model_settings with vision_embedding_dim=512/telemetry_dim=7/hidden_dim=256/output_dim=7) for required fields; (2) implement `MultimodalInferenceModel` — frozen ResNet18 vision encoder (ImageNet pretrained, classification head removed) + telemetry fusion network (519→256→256→7 with ReLU) producing 7 joint commands per inference; (3) expose via FastAPI (`GET /health`, `POST /infer`, `GET /metrics` for Prometheus-compatible monitoring); (4) simulate canary deployment (`simulate_canary.py`) comparing baseline v1.0.0 against candidate v1.1.0 on an 80/20 traffic split, evaluating latency, success rate, and action divergence (0.023 in the observed run) before promoting; (5) containerize with Docker, mounting real RLBench episode data, and test `/infer` with actual `front_rgb` frames and `low_dim_obs.pkl` paths. The exercise explicitly mirrors — at small scale — the exact scaffolding (manifest-driven config, staged rollout, telemetry, rollback logic) that governs how Eighteen Labs would operate production-grade LBMs in real enterprise robots.

## Key Takeaways
1. Real-time multimodal serving requires engineering modality coherence and temporal alignment as the primary challenge — the model invocation itself is a minority of the work.
2. Multistage inference pipelines turn deployment into orchestrating a dependency graph of reasoning tasks, each with its own latency budget, schema contract, and failure-propagation policy — not a single model behind an API.
3. Monitoring deployed multimodal models requires three parallel logging streams (input, prediction, feedback) because any one alone can't diagnose a failure — you need all three to know what happened, why, and how often it matters.
4. Drift must be tracked per-modality via embedding-space divergence (PSI/MMD) because each input channel can drift independently and silently degrade fused reasoning quality.
5. Shadow evaluation and canary promotion (with multimodal-specific single-component isolation) are the only safe way to update composite models without risking production quality.
6. Rollback should follow graceful degradation — drop only the drifting modality when possible, reserving full rollback for simultaneous multi-modality drift.
7. Large behavior models require session-oriented, stateful serving (context cache across timesteps) and a dual-path (shallow/deep) architecture to resolve the latency-vs-behavioral-depth trade-off without sacrificing either.

## Connects To
- **Ch 7**: Extends drift detection, shadow/canary promotion, and edge-cloud topology concepts from FieldSense into enterprise robotics serving infrastructure.
- **Ch 8**: Builds directly on CI/CD, versioning, and retraining pipeline patterns, extending them to composite multimodal manifests and sequence-aware CI/CD for behavior models.
- **Ch 9**: Continues the tiered hardware scheduling and disaggregated serving concepts (prefill/decode) into full production inference engineering.
- **Ch 11**: Sets up governance, safety, and auditability requirements (EU AI Act, behavioral ontology traces) that the next chapter formalizes.
- **Triton / vLLM / TensorRT-LLM / SGLang**: model-serving frameworks referenced as concrete implementations of the tiered serving and inference-engine selection patterns.
