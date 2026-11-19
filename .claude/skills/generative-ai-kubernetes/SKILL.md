---
name: generative-ai-kubernetes
description: "Knowledge base from \"Generative AI with Kubernetes\" by Jonathan Baier (BPB Publications, 2025). Use when deploying generative AI/foundational models on Kubernetes — infrastructure architecture, data pipelines, model refinement, observability/monitoring, or securing ML/GenAI pipelines on K8s."
---

# Generative AI with Kubernetes
**Author**: Jonathan Baier | **Chapters**: 10 | **Generated**: 2026-11-18

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch07`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The book's architecture, in one line**: foundations (Ch1-2: what GenAI is + how K8s supports it) → models on K8s (Ch3-4: deploying/evaluating foundational models) → pipelines (Ch5-7: MLOps/LLMOps, model serving, data management) → refinement/operations (Ch8-10: RAG/fine-tuning, observability, security). Every hands-on example builds on minikube + kubectl conventions established in Ch2.

**K8s-for-AI architecture model (Ch2-4)**
- Control plane (API server, scheduler, controllers, etcd) / data plane (worker nodes, kubelet) separation underlies all resilience, scaling, and security patterns in the book.
- GPUs are exposed to Kubernetes as a schedulable "extended resource" (`nvidia.com/gpu`) via the Device Plugin Framework — request in `resources.limits`, add a toleration for GPU-node taints.
- Three autoscaling layers: HPA (pod count by metric), VPA (pod resource limits), Cluster Autoscaler (node count). GPU-based autoscaling needs custom metrics (no native GPU utilization signal in HPA).
- Namespaces + ResourceQuotas give organizational grouping and resource caps (including GPU caps) but are NOT a security isolation boundary by themselves — pair with RBAC + NetworkPolicies for real isolation.
- Quantization (GGUF for CPU, GPTQ/AWQ for GPU) is the first lever for fitting large models into constrained infrastructure; GPU inference is roughly ~10x faster than CPU for diffusion-style workloads in the book's own benchmarks.

**Pipeline design model (Ch5-7)**
- Three interlocking pipelines — code, data, model — each need their own tooling (data/model registries ≠ Git for source code).
- MLOps stages: Tracking (version control/lineage) → Build & Test (data curation + model tuning) → Deploy → Monitor & Feedback. LLMOps differs from MLOps mainly in non-determinism (same input ≠ same output) and transfer learning (broad generalization vs. narrow task-specific models).
- MLflow (tracking + registry + deployment server) is the book's chosen consolidation tool, deployed via Helm with MinIO as its S3-compatible artifact backend. Kubernetes `Jobs` (not Deployments) are the right primitive for batch comparison/evaluation runs, with `ConfigMaps` injecting server connection details.
- Model serving decouples from the training/tracking pipeline via Seldon Core (or KServe) using the Open Inference Protocol V2 — this lets the serving layer scale (via native HPA integration) independently of MLflow/storage.
- Data governance (MinIO tagging, versioning, lifecycle/tiering, object locking) prevents "data lake → data swamp" decay. Embeddings + vector databases (ChromaDB) enable semantic similarity search, the foundation for RAG.

**Refinement trade-off spectrum (Ch4, Ch8)**
- Always work cheapest-to-costliest: Prompt engineering (zero-shot → few-shot → chain-of-thought → least-to-most/plan-and-solve/self-consistency/role prompting) → RAG (vector retrieval + augmented prompt) → PEFT (LoRA/IA3, freeze base model, train small adapter) → full fine-tuning → training from scratch (almost never justified; LLaMA-3 scale used 24,000+ GPUs).
- RAG's core value: context-window efficiency — retrieve only relevant chunks instead of stuffing an entire knowledge base into the prompt. Pure vector similarity can misrank on negation/intent; the LLM generation step + evaluators (ContextRelevancyEvaluator, FaithfulnessEvaluator) are necessary safety nets, not optional extras.
- Sampling parameters (temperature, top-k, top-p, context length, max tokens) are tuning knobs, not universal quality dials — low temperature = deterministic/repeatable, high = creative/less coherent.

**Observability + security operational model (Ch9-10)**
- Two monitoring planes, both required: infrastructure (Prometheus/Grafana/kube-state-metrics for cluster health) and model behavior (LangKit-style tools for toxicity, sentiment, injection/jailbreak similarity, PII leakage — monitor BOTH prompts and responses).
- Alignment (RLHF: supervised fine-tune → train reward model from ranked outputs → RL fine-tune, e.g. PPO; or the lighter DPO) reduces but never eliminates harmful/hallucinated output — because LLMs are non-deterministic, continuous monitoring is mandatory, not a one-time gate.
- Threat model: NIST's CIA-triad-plus-Abuse taxonomy (Availability/Integrity/Confidentiality/Abuse) and the OWASP Top 10 for LLMs (prompt injection, insecure output handling, training data poisoning, model DoS, supply-chain vulnerabilities, sensitive info disclosure, insecure plugin design, excessive agency, overreliance, model theft) are the two standard frameworks for classifying GenAI-specific risk.
- Direct injection (malicious user prompt) vs. indirect injection (poisoned RAG/external data source) require different mitigations — filtering only user input misses indirect attacks entirely.
- Kubernetes security primitives: RBAC (namespace-scoped Roles + RoleBindings, least-privilege verbs), Secrets + tiered EncryptionConfiguration (KMS → local KMS → plaintext fallback, applied selectively by resource sensitivity), NetworkPolicies, Pod Security Standards (Baseline/Privileged/Restricted), auditing. Safetensors (not pickle) for model weight storage — pickle allows arbitrary code execution on deserialization.
- A GenAI program office operationalizes governance: AI-use policy → usage inventory (surface shadow AI) → data classification/lineage → continuous monitoring (NIST TEVV) → cost management (model/prompt optimization, LLM cascading) → third-party risk management.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-introduction-to-generative-artificial-intelligence.md) | Introduction to Generative Artificial Intelligence | AI/ML taxonomy, transformer attention, InstructGPT alignment, GANs, diffusion |
| [ch02](chapters/ch02-kubernetes-for-generative-ai.md) | Kubernetes for Generative AI | Control/data plane, HPA/VPA/Cluster Autoscaler, Device Plugin Framework, Namespaces+Quotas |
| [ch03](chapters/ch03-introduction-to-foundational-models-on-kubernetes.md) | Introduction to Foundational Models on Kubernetes | Model card evaluation, Gradio Interface/Blocks, container-then-deploy pattern, diffusion scheduler/VAE |
| [ch04](chapters/ch04-working-with-foundational-models.md) | Working with Foundational Models | Model evaluation questions, LLM benchmarks (ARC/MMLU/etc.), sampling controls, quantization/pruning/PEFT, GPU sharing |
| [ch05](chapters/ch05-process-and-pipelines.md) | Process and Pipelines | MLOps pipeline stages, MLOps vs LLMOps, MLflow components, data/model version control |
| [ch06](chapters/ch06-process-and-pipelines-on-kubernetes.md) | Process and Pipelines on Kubernetes | MLflow metrics/custom metrics, Seldon Core + Open Inference Protocol V2, namespace strategy |
| [ch07](chapters/ch07-managing-data-for-generative-ai.md) | Managing Data for Generative AI | Data lake/lakehouse, MinIO governance (tagging/lifecycle/locking), PV/PVC, embeddings + vector DB |
| [ch08](chapters/ch08-refining-and-improving-results.md) | Refining and Improving Results | Refinement trade-off spectrum, RAG architecture, advanced prompting techniques, PEFT adapters |
| [ch09](chapters/ch09-observability-and-monitoring.md) | Observability and Monitoring | Prometheus+Grafana stack, RLHF/DPO alignment, LangKit metrics (toxicity/injection/PII) |
| [ch10](chapters/ch10-securing-ml-genai-pipelines-on-k8s.md) | Securing ML/GenAI Pipelines on K8s | NIST taxonomy, OWASP Top 10 for LLMs, K8s RBAC/encryption, GenAI program office governance |

## Topic Index
- **Alignment (RLHF/DPO)** → ch01, ch09
- **Autoscaling (HPA/VPA/Cluster Autoscaler)** → ch02, ch06
- **Benchmarks (ARC, MMLU, HellaSwag, TruthfulQA, GSM8K)** → ch04
- **ChromaDB / vector databases** → ch07, ch08
- **Data governance (tagging, lifecycle, object locking)** → ch07
- **Data lake / lakehouse / data swamp** → ch07
- **Embeddings** → ch01, ch07, ch08
- **Fine-tuning / PEFT (LoRA, IA3)** → ch04, ch08
- **GPU scheduling & sharing (MIG, MPS, time-slicing)** → ch02, ch04
- **MLflow (tracking/registry)** → ch05, ch06
- **Monitoring (Prometheus/Grafana)** → ch09
- **Monitoring (model behavior, LangKit)** → ch09
- **Namespaces & RBAC** → ch02, ch06, ch10
- **OWASP Top 10 for LLMs / NIST adversarial ML taxonomy** → ch10
- **Prompt engineering (zero/few-shot, CoT, least-to-most, self-consistency)** → ch04, ch08
- **Prompt injection (direct/indirect) & jailbreaking** → ch09, ch10
- **Quantization (GGUF, GGML, GPTQ, AWQ)** → ch01, ch04
- **RAG (Retrieval Augmented Generation)** → ch07, ch08
- **Safetensors vs. pickle** → ch03, ch10
- **Secrets & encryption-at-rest** → ch02, ch10
- **Seldon Core / KServe / Open Inference Protocol V2** → ch06
- **Transformers & latent space** → ch01

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — architecture/deployment patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 10 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive, as are recurring publisher promotional asides ("Join our book's Discord space"). For topics beyond this book's scope, check other skills or ask directly.
