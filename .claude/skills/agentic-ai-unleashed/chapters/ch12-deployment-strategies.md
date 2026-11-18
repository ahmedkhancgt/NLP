# Chapter 12: Deployment Strategies

## Core Idea
Agentic systems break the stateless-service assumptions underlying most deployment patterns — an agent mid-negotiation or mid-research-task cannot simply restart — so production deployment requires explicit strategies across cloud/edge/hybrid placement, multidimensional versioning, and graceful degradation, all guided by one principle: **plan for every component's failure rather than assuming reliability**, because "perfect reliability is impossible."

## Frameworks Introduced

### Cloud Infrastructure Trade-offs
- **Container strategy**: embedding models in images (15–30GB, slow rebuild-to-update) vs. external loading at startup (2–5GB, independent model updates but added startup latency) vs. hybrid/on-demand streaming — pick based on update frequency vs. startup-latency tolerance.
- **Kubernetes for stateful agents**: use **StatefulSets** (not Deployments) for long-running agents needing stable network identity + persistent storage across pod rescheduling. Size CPU *requests* for steady-state, not inference bursts, leaving a request-limit gap for spikes — but an overly generous *limit* risks the OOMKiller terminating neighboring pods. Observe actual usage over **weeks, not days**, since burst patterns hide in short windows.
- **Serverless mismatch**: serverless assumes ephemeral/stateless execution; agents maintain state. Viable only for simple tool execution and lightweight routing; stateful conversation and large-model inference need external state stores or dedicated containers instead — cold start (10–20s for transformer init) and the typical 15-minute timeout make serverless actively wrong for long-running reasoning (use Step Functions orchestration instead).
- **Multi-region**: driven by latency, availability, *and* data-residency compliance (GDPR, China/Russia localization) — the last turns multi-region from an optimization into a mandatory architecture constraint enforced at the infrastructure level.
- **Service mesh (Istio/Linkerd/Consul)**: justified once a system has enough services that observability/traffic-control value exceeds sidecar overhead (~50–100MB/pod) — roughly 16+ services; below 5 services, skip it.

### Edge Deployment Constraints and Compression
Edge is necessary (not optional) when cloud round-trip latency is physically incompatible with the task (collision avoidance), when data cannot leave the device (patient data), or when connectivity can't be assumed (vehicles). Four compression techniques, combinable: **quantization** (INT8: 4x smaller, 95-98% accuracy, no retraining; INT4: 8x smaller, 85-92% accuracy, needs quantization-aware training), **knowledge distillation** (5-10x smaller, 80-90% accuracy, needs extensive training), **structured pruning** (2-4x smaller, 90-95% accuracy, needs fine-tuning), **combined** (15-30x smaller, 75-85% accuracy — the ceiling case).
- When to use which: start with post-training INT8 quantization (free accuracy-for-size trade, no retraining); escalate to distillation/pruning/combination only when size/speed requirements still aren't met.

### Edge-Specific Architectural Patterns (exploit constraints, don't fight them)
- **Mixture-of-Experts**: a lightweight router loads only the relevant expert sub-network per inference (e.g., navigation vs. conversation), dropping peak memory from 4GB (all experts loaded) to 1GB (one expert + router).
- **Speculative execution**: run multiple candidate inference paths concurrently, keep the first to finish, cancel the rest — masks latency by spending otherwise-idle edge compute, rather than reducing actual compute.
- **Progressive enhancement**: a minimal always-working core, with richer capability layers (better model, more knowledge, cloud augmentation) activating only when resources/connectivity permit — the agent degrades in capability, never to zero.

### Distributed State Synchronization
- **Event sourcing**: sync the sequence of state-producing events, not current state snapshots — upload events since last sync, download missed events, replay to reconcile.
- **CRDTs (Conflict-free Replicated Data Types)**: mathematically guarantee all replicas converge to the same state regardless of message order or partitions — Last-Write-Wins registers, grow-only sets, observed-remove sets handle common cases without bespoke conflict-resolution logic.
- **Differential sync**: transmit only deltas, not full state (a 100MB knowledge base might sync in 500KB) — essential under edge bandwidth constraints.
- When to use which: event sourcing for append-only logs (no conflicts by construction); LWW-CRDT for simple preference-style data; full CRDT/OT machinery only for genuinely collaborative, concurrently-edited data; delta sync + checksums for large reference knowledge bases.

### Hybrid Tiered Processing (the realistic middle ground)
Mirrors human cognition: **reflexive** (edge, milliseconds), **considered** (gateway, seconds), **deep** (cloud, minutes). Each tier handles what it can locally, escalating only when necessary — providing natural failure isolation (losing the cloud tier degrades capability, doesn't eliminate it).
- **Cognitive function split**: perception preprocessing at the edge (bandwidth: a 25Mbps 4K stream compresses to 1-2Mbps once reduced to motion-detected regions before transmission) + perception deep analysis in the cloud (GPU-dependent); planning cached at edge/gateway (60%+ cache hit rate typical) + novel planning in the cloud; execution always local (cloud latency is incompatible with real-time control) with optional cloud safety-validation (50-100ms budget, "edge proposes, cloud validates, edge executes"); immediate reflection local, deep reflection (accumulated-experience learning) in the cloud.
- **Latency budget decomposition**: `L_total = L_network + L_queue + L_processing + L_serialization`, evaluated at **P95, not average** — a 50ms-average component can be 200ms at P95 from GC pauses or network hiccups, and P95 is what actually determines whether a latency-sensitive deployment is viable.
- **Cost-driven placement**: edge amortizes quickly (6-12 months) for high-volume steady workloads; cloud elasticity wins for bursty/unpredictable load. Data egress charges ($0.08-0.12/GB) can by themselves justify edge hardware cost at scale. Hybrid (steady baseline on edge, burst overflow to cloud) typically achieves 40-60% cost reduction vs. pure cloud.

### Multidimensional Versioning for Stateful Agents
Traditional `MAJOR.MINOR.PATCH` is insufficient — agentic version compatibility spans model behavior and state schema, not just API contracts. Use `RUNTIME.MODEL.SCHEMA.CAPABILITIES`, letting each dimension evolve independently (a model upgrade needn't touch schema; a capability addition needn't touch runtime).
- **State schema migration spectrum**: lazy (migrate on first access post-upgrade — distributes load but risks failures on long-dormant agents), eager (migrate everything at deploy time — guarantees consistency, costs deployment-time delay), dual-schema (read both, write new, background-migrate — safest rollback path, temporarily doubles complexity).
- **Seamless update via blue-green + session handling**: connection draining (let old sessions finish naturally — works for short sessions) vs. session migration (serialize+transfer active state to the new environment — needed for long-running agents, costs a sub-second pause but zero conversation loss).

### Canary Deployment with Agent-Specific Gates
Standard staged rollout (0.1% → 1% → 10% → 50% → 100%) but agentic canary success criteria must extend beyond error-rate/latency to: **reasoning coherence**, **tool hallucination rate** (inventing non-existent tools), **context retention**, **safety-violation rate** — automated metrics miss subtle quality/personality degradation that human review catches.

### Graceful Degradation (SLO-tiered, not binary)
Define **target SLO → degraded mode → minimum acceptable** for each capability (e.g., response generation: 95%<2s target → shorter responses degraded → 90%<5s minimum). **Circuit breakers** (closed→open→half-open state machine) stop calling a failing dependency once a failure threshold trips, returning cached/simplified results instead of wasting resources on doomed calls. **Tiered capability reduction** escalates through 5 load levels (Normal → Elevated → High → Critical → Emergency), each disabling progressively more expensive features while preserving some response capability at every level.

## Key Concepts
- **OOMKiller**: Linux kernel mechanism (not Kubernetes itself) that terminates processes under memory pressure — explains why generous resource *limits* can collaterally kill neighboring pods.
- **Multi-armed bandit allocation**: dynamically shifts traffic toward better-performing variants while still exploring alternatives — contrasted with fixed-split A/B tests.
- **Orthogonal parallel experimentation**: multiple A/B experiments can share the same traffic pool via independent randomization, as long as they modify non-interacting aspects of the system — but each additional concurrent experiment adds variance to the shared user pool, so total "experimentation capacity" at fixed traffic is finite and must be prioritized.
- **Feature flag lifecycle**: development (always off) → internal testing (on for internal) → canary (percentage-based) → full rollout (always on, convert to permanent code) → deprecated (remove flag and code) — flags are explicit technical debt requiring disciplined cleanup, not permanent infrastructure.

## Mental Models
- Before choosing cloud, edge, or hybrid for any component, ask which cognitive function it performs (perception/planning/execution/reflection) and match it to that function's natural latency/compute affinity, rather than defaulting to "cloud for everything" or "edge for everything."
- Treat P95/P99 latency, not average, as the number that determines architectural viability — averages hide exactly the tail behavior that breaks user experience.
- Assume every component will eventually fail, and design the SLO-tiered degraded mode *before* the failure happens, not as an incident-response improvisation.

## Anti-patterns
- **Forcing stateful agents into serverless** "for the scaling story": the chapter is explicit that this requires "architectural contortions that often negate the platform's purported benefits" — state size, cold starts, and timeout limits make pure serverless wrong for anything beyond simple tool execution.
- **Sizing Kubernetes resource limits without weeks of observation**: short observation windows miss real burst patterns and lead to either OOMKiller collateral damage or wasted capacity.
- **Treating rollback as "just redeploy the old code"**: state schema changes make rollback a distinct, harder problem than code rollback — plan bidirectional migrations or accept explicit data-loss trade-offs in advance.
- **Running unlimited concurrent A/B experiments on the same traffic pool**: each additional experiment reduces every other experiment's statistical power — prioritize by business impact rather than running everything simultaneously.
- **Treating feature flags as permanent infrastructure**: accumulating un-cleaned flags is explicit technical debt; every flag needs a planned end-state (convert to permanent code or remove).

## Reference Tables

**Compression technique comparison**

| Technique | Size reduction | Accuracy retention | Training required |
|---|---|---|---|
| INT8 quantization | 4x | 95–98% | No (post-training) |
| INT4 quantization | 8x | 85–92% | Yes (QAT recommended) |
| Knowledge distillation | 5–10x | 80–90% | Yes (extensive) |
| Structured pruning | 2–4x | 90–95% | Yes (fine-tuning) |
| Combined approach | 15–30x | 75–85% | Yes (multi-stage) |

**Measured latency: edge beats cloud for vision, cloud beats edge for generation**

| Component | Location | P50 | P95 | Bottleneck |
|---|---|---|---|---|
| Object detection | Cloud GPU | 45ms | 120ms | Network variability |
| Object detection | Edge TPU | 15ms | 22ms | Thermal throttling |
| Text generation (50 tok) | Cloud | 180ms | 450ms | Queuing + cold start |
| Text generation (50 tok) | Edge | 3500ms | 4100ms | Limited compute |

Lesson: edge wins on vision tasks purely by eliminating network latency despite weaker hardware; cloud wins on generation because its compute advantage outweighs network cost — there's no universal "edge is faster" or "cloud is faster" rule, only per-workload analysis.

## Worked Example

**Canary rollout gate progression (the chapter's concrete thresholds):**
Internal (0.1% traffic, 2-4h, rollback on any error spike) → Early canary (1%, 12-24h, rollback if error rate >1%) → Medium canary (10%, 24-48h, rollback if task success <90% of baseline) → Large canary (50%, 48-72h, rollback on sustained degradation) → Full rollout (100%, ongoing monitoring). At each stage, automated statistical process control can trigger rollback *before* human on-call response time — but the chapter recommends a hybrid: automatic rollback only for severe triggers (error rate >10%, complete failure), human approval for marginal cases, to avoid "rollback thrashing" when both versions have issues.

## Key Takeaways
1. Match each cognitive function (perception/planning/execution/reflection) to its natural tier (edge/gateway/cloud) rather than deploying uniformly.
2. Evaluate latency viability at P95/P99, never average — tail latency is what breaks real deployments.
3. Compress models with the lightest technique that meets your size/accuracy target (INT8 first) before escalating to distillation or combined approaches.
4. Version agentic systems across runtime/model/schema/capabilities independently — this is what makes safe partial updates possible.
5. Plan state-schema rollback strategy (bidirectional migration, forking, or accepted data loss) before you need it, not during an incident.
6. Extend canary success criteria beyond error rate/latency to reasoning coherence, tool hallucination, and context retention for agentic systems specifically.
7. Design SLO-tiered graceful degradation (target → degraded → minimum) for every capability — binary up/down thinking guarantees catastrophic rather than graceful failure.
8. Treat feature flags as temporary, lifecycle-managed infrastructure, not a permanent pattern.

## Connects To
- **Ch 9 (Performance Optimization)**: model compression techniques here directly extend that chapter's quantization/distillation content into the edge-deployment context.
- **Ch 10 (Testing and QA)**: canary deployment's statistical-significance and agent-specific evaluation criteria build on that chapter's AB-testing framework.
- **Ch 11 (Monitoring and Observability)**: canary health evaluation and automated rollback triggers depend directly on that chapter's monitoring infrastructure.
- **Ch 13 (Security and Safety)**: the next chapter's attack-surface analysis (prompt injection, unauthorized tool access) builds on this chapter's deployment architecture.
