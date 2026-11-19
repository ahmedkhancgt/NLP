# Patterns

## Model Selection Funnel Pattern
**When to use**: Comparing multiple candidate LLMs/SLMs for a specific enterprise application.
**How**: Run every candidate through the ~21-question checklist (regulatory compliance, CapEx/OpEx, latency/throughput, cost per token, governance, scalability, data proximity, security, managed/unmanaged, automation, availability, resiliency, hosting, virtualization, sustainability, hardware, software licensing, collaboration ease, inference-task fit, accuracy, upgrade path, technical debt, skills, open/closed-source, decommissioning, indemnity, context limits, LLM vs. SLM). Eliminate on hard constraints (regulatory, security) first; rank survivors on soft criteria.
**Trade-offs**: More rigorous but slower than picking a "best known" model; may surface more than one viable model, requiring a portfolio rather than single-winner mindset.

## Deployment Pattern Selection
**When to use**: After model selection, deciding where/how to host the model.
**How**: Choose among cloud service, open-on-premises, own-on-premises, open-cloud-IaaS, own-cloud-IaaS based on regulatory intensity (push toward on-premises) vs. agility/cost needs (push toward cloud); hybrid splits sensitive data on-premises from less-critical cloud workloads.
**Trade-offs**: Cloud = speed/elasticity but less control and potential data-sovereignty/latency issues; on-premises = full control/compliance but high CapEx and ongoing maintenance burden; hybrid = balanced but operationally complex to manage consistently.

## Blue/Green, Canary, A/B Release Strategy
**When to use**: Releasing new versions or features of an LLM-powered application (not the underlying model).
**How**: Use Canary for cheap, fast early-feedback on a user subset; use Blue/Green for safe, fully reversible cutover between two parallel environments; use A/B Testing (combinable with either) to measure business-metric impact of variations.
**Trade-offs**: Blue/Green costs double infrastructure during transition; Canary risks exposing issues to real (if limited) users; A/B requires sufficient traffic/statistical power to draw conclusions.

## Cloud/On-Premises Demarcation and Co-location
**When to use**: Deciding where enterprise data, models, and applications physically/logically reside for a given workload.
**How**: Commit fully to one side of the cloud/on-premises demarcation line per workload; use Kubernetes to co-locate the GenAI platform, data, and applications to minimize latency and maximize resource utilization.
**Trade-offs**: Staying entirely above or below demarcation preserves latency/service quality; mixing across it degrades both. Co-location reduces latency but increases architectural coupling.

## LoRA / PEFT Fine-Tuning Pattern
**When to use**: Specializing a pre-trained LLM for enterprise-specific jargon/tasks when the dataset is narrow and full retraining would be costly/overfitting-prone.
**How**: Apply low-rank trainable matrix updates per layer instead of retraining all weights; use weight decay to further control overfitting; auto-scale GPU compute (0 → N nodes) only during active fine-tuning jobs.
**Trade-offs**: Much cheaper and faster than full fine-tuning, with lower overfitting risk, but may not capture as deep a specialization as full retraining for very divergent tasks.

## Production-Scale Prompt Engineering Pipeline
**When to use**: Moving a validated single-input prompt flow to enterprise production volume (hundreds to thousands of inputs).
**How**: Build the flow as chained steps (fetch → process via LLM with system prompt → classify/transform → structure output); wrap in an automated Batch Run; evaluate on accuracy, duration, and token count together; iterate by refining data, swapping models, or deploying as an endpoint.
**Trade-offs**: Batch automation adds engineering overhead upfront but is the only way to scale beyond manual trial-and-error; token-count optimization can trade off against accuracy/fluency.

## RAG + Hybrid Retrieval + Semantic Ranking
**When to use**: Enterprise Q&A/assistant applications needing grounded, current, enterprise-specific answers at scale.
**How**: Combine vector search (semantic/embedding matching) with keyword search (BM25 exact-term matching) for L1 retrieval; apply L2 semantic re-ranking (cross-attention LLM scoring, 0-4 scale) on top ~50 L1 results; chunk long documents using a high-token-limit embedding model (8K+) so relevant passages rank first; pass top 3-5 ranked chunks as LLM grounding context.
**Trade-offs**: Higher infrastructure/engineering complexity than naive single-method retrieval, but substantially better relevance and reduced hallucination; chunking strategy must be tuned per document type.

## RAG + Prompt Tuning + Fine-Tuning Hallucination-Reduction Triad
**When to use**: High-stakes or regulated domains where hallucination risk must be minimized.
**How**: Layer all three together — RAG supplies factual grounding via retrieved documents, prompt tuning aligns generative focus to query context, fine-tuning adapts the model's comprehension of the retrieved domain/format.
**Trade-offs**: Requires investment in all three techniques simultaneously rather than relying on one; higher engineering and evaluation overhead, but substantially more reliable output.

## Model-as-a-Service Consumption Decision (PAYG vs. Real-Time Endpoint)
**When to use**: Choosing how to consume an LLM via a cloud MaaS platform, per environment (dev/test vs. production).
**How**: Use PAYG for variable/low-volume dev/test workloads; switch to a dedicated real-time endpoint when production token volume is extremely high and demand is forecasted/consistent, justifying continuous billing for guaranteed capacity/latency.
**Trade-offs**: PAYG avoids idle-capacity cost but lacks guaranteed capacity; real-time endpoints guarantee performance but cost more and require accurate demand forecasting to be worthwhile.

## Multi-MaaS / Hybrid MaaS Resilience Pattern
**When to use**: Enterprise-class production deployments needing resilience against single-provider outages, regional disruption, or wanting best-of-breed capabilities across providers.
**How**: Distribute AI workloads across multiple cloud platforms (or MaaS + on-premises) based on each provider's strengths (tooling, cost, regional compliance); invest in a cross-platform orchestration/management layer; design applications with containerization and provider-agnostic interfaces to reduce lock-in.
**Trade-offs**: Significantly higher architectural and operational complexity; benefits (resilience, optimization, lock-in avoidance) only materialize with sufficient orchestration investment.

## Confidential AI Deployment Pattern (VM TEE + GPU TEE)
**When to use**: AI workloads processing regulated or highly sensitive data (healthcare, finance, defense) on shared or cloud infrastructure.
**How**: Deploy VM TEE to isolate the broader processing environment (memory encryption, secure boot, hardware root of trust) and GPU TEE to isolate GPU-intensive computation (encrypted GPU memory, secured CPU-GPU offload); pair with procedural safeguards covering OS, hypervisor, sysadmin, and physical-access risk layers.
**Trade-offs**: Added hardware/software cost and operational complexity, offset by reduced breach/fine exposure, improved compliance posture, and enhanced customer/partner trust.

## Latency Diagnosis and Mitigation Pattern
**When to use**: Any production GenAI application where response time is degraded or must meet real-time expectations.
**How**: Profile all five latency factors independently (input network, hardware, model inference, integration point, output network) to locate the true bottleneck; apply targeted mitigations — edge computing for network latency, pruning/quantization/distillation for inference latency, co-location for integration-point latency, optimized middleware/APIs for integration, efficient post-processing for output latency.
**Trade-offs**: Thorough profiling takes more upfront diagnostic effort than guessing, but avoids wasted optimization effort on the wrong bottleneck; model-size-reduction techniques trade some accuracy for speed.

## Multi-Agentic Assistant Composition Pattern
**When to use**: Enterprise problems spanning multiple specialized domains/tasks (e.g., customer service + technical support + internal knowledge) where a single monolithic model is insufficient.
**How**: Define each agent with four components (toolset, data, function, instruction); back each agent with an LLM chosen for its specific inference-type/modality needs; connect agents via a shared API layer enabling interoperability and model-switching per task.
**Trade-offs**: Higher architectural complexity and coordination overhead than a single model, but better specialization, scalability, and resilience; requires careful human-oversight design to keep agents augmenting rather than replacing human judgment.
