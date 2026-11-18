# Patterns

## Containerized Model Demo Deployment
**When to use**: standing up any model-serving demo or early-stage inference service (Gradio/Streamlit + Hugging Face pipeline).
**How**: write a Dockerfile (`FROM python:3.x-slim` → install requirements → `COPY . .` → `EXPOSE <port>` → `CMD [...]`) → `minikube image build . -t <tag>` → apply a Deployment + Service YAML pair → `kubectl port-forward` to access.
**Trade-offs**: fast to stand up and iterate; not production-grade (no autoscaling/standard inference API) until layered with HPA (see below) or migrated to a dedicated server (Seldon/KServe).

## GPU Device Plugin + Quota Isolation
**When to use**: any multi-tenant cluster running GPU-backed training/inference where GPU scarcity must be governed.
**How**: install vendor GPU drivers + container toolkit on nodes → deploy the Device Plugin DaemonSet (e.g. NVIDIA) → request `nvidia.com/gpu` in pod `resources.limits` with a matching toleration → enforce per-namespace `ResourceQuota` on `requests.nvidia.com/gpu` (set to 0 to block GPU use in a namespace, or N to cap it).
**Trade-offs**: simple and effective for coarse allocation; does not provide memory isolation between pods sharing a GPU — for finer sharing, see GPU-sharing trade-offs in cheatsheet.md.

## Horizontal Pod Autoscaler for Inference Workloads
**When to use**: stateless inference services where request volume fluctuates and replica count should track load.
**How**: `kubectl autoscale deployment <name> --cpu-percent=<N> --min=<N> --max=<N>` or apply an HPA manifest; ensure `metrics-server` addon is enabled; set `resources.requests`/`limits` on the Deployment so HPA has a baseline to compute utilization against.
**Trade-offs**: effective for CPU-bound scaling signals; GPU-based autoscaling requires custom metrics (e.g. NVIDIA GPU monitoring + Prometheus adapter) since HPA's native metrics don't cover GPU utilization out of the box.

## MLflow Tracking + Registry on Kubernetes
**When to use**: any team needing reproducible experiment tracking, parameter/metric logging, and model versioning without standing up multiple point tools.
**How**: deploy via the Bitnami MLflow Helm chart (also provisions MinIO as artifact store) → inject server connection details into workloads via a Kubernetes `ConfigMap` → run comparison/evaluation code as Kubernetes `Jobs` (not long-running Deployments) → promote a chosen model version to a named registry stage (e.g. "Staging").
**Trade-offs**: easy to adopt, Apache 2.0 licensed, good starting point; lacks the broader end-to-end orchestration of Kubeflow and the dedicated model-serving sophistication of Seldon Core/KServe — pair with those for full lifecycle coverage.

## Seldon Core Model Serving with Native HPA
**When to use**: production model serving needing a standard inference API (Open Inference Protocol V2), framework-agnostic prepackaged servers, and scaling decoupled from the training/tracking pipeline.
**How**: install a service mesh (Istio) → install the Seldon Core operator via Helm → define a `SeldonDeployment` CRD pointing `modelUri` at an object-storage path (e.g. `s3://mlflow/...` via MinIO) with the appropriate `implementation` (e.g. `HUGGINGFACE_SERVER`) → optionally add `componentSpecs.hpaSpec` for CPU-based replica autoscaling.
**Trade-offs**: standardizes serving across frameworks and integrates cleanly with K8s primitives (HPA, Istio ingress); adds operational complexity (service mesh dependency) and requires checking current licensing terms (Seldon shifted parts of its offering to Business Source License in Jan 2024).

## Object Storage as Decoupled Pipeline Backbone (MinIO/S3-compatible)
**When to use**: any multi-stage pipeline (tracking → registry → serving, or data lake → RAG) needing a durable, S3-API-compatible intermediate storage layer.
**How**: deploy MinIO (often alongside MLflow via Helm) backed by a Kubernetes `PersistentVolumeClaim` → apply bucket tagging (cost center, classification, owner), versioning, lifecycle/tiering rules (e.g. transition to Azure Blob after N days), and object-lock retention (Governance/Compliance mode) as governance controls.
**Trade-offs**: S3-API compatibility means skills transfer directly to AWS/Azure/GCP object storage; self-hosting MinIO adds operational burden (PVC sizing, backup, patching) that a managed cloud object store avoids.

## Vector Database + RAG Augmentation
**When to use**: grounding LLM responses in domain-specific/proprietary data without the cost of fine-tuning, especially when context-window budget is limited.
**How**: embed source documents into a vector database (e.g. ChromaDB via Helm chart) → at query time, run a similarity search to retrieve top-k relevant documents → construct an augmented prompt combining retrieved context + user query + explicit grounding instructions ("only use descriptions from...") → send to the LLM → optionally score the response with an evaluator (e.g. Llama-Index's `ContextRelevancyEvaluator`, `FaithfulnessEvaluator`).
**Trade-offs**: cheaper and faster to iterate than fine-tuning; adds retrieval-latency and infrastructure complexity; retrieval quality depends heavily on data cleanliness and chunking strategy, and pure similarity ranking can misrank on negation/intent (requires LLM generation + evaluation as a safety net).

## Fine-Grained RBAC for Application Administrators
**When to use**: granting a human or team scoped (not cluster-admin) control over their application's Kubernetes resources.
**How**: issue a certificate via a `CertificateSigningRequest` (CSR), approve it, configure kubectl credentials/context for the new user → define a namespace-scoped `Role` granting only needed verbs/resources (e.g. full control over Pods/Deployments/Services/ConfigMaps, but only `list` on Secrets) → bind with a `RoleBinding`.
**Trade-offs**: minimizes blast radius of a compromised credential; requires more upfront design work (per-team roles, namespace strategy) than a single shared admin credential — pays off at multi-team scale.

## Tiered Encryption-at-Rest for Cluster Secrets
**When to use**: any cluster storing sensitive configuration (API keys, credentials, Secrets) in etcd.
**How**: define an `EncryptionConfiguration` with ordered `providers` per resource type — strongest (cloud KMS, falling back to local KMS, falling back to plaintext) for `secrets`; explicit plaintext for low-sensitivity `configmaps`/`events`; a wildcard default for everything else (K8s 1.27+) — and pass it to the API server at startup.
**Trade-offs**: layered fallback balances security against availability (a KMS outage doesn't necessarily block the cluster, depending on fallback choice); requires careful ordering since provider priority determines which encryption is actually applied.

## GenAI Program Office Governance Loop
**When to use**: organizations scaling GenAI usage beyond isolated pilots, needing to manage security, compliance, and cost risk centrally.
**How**: publish an AI-use policy → inventory existing AI/ML usage (including shadow AI, tracked via AIBOM-style bills of material) → classify and establish lineage for data feeding AI systems → implement continuous monitoring (NIST TEVV framework) → manage cost via model optimization (smaller/specialized/quantized models, LLM cascading) and prompt optimization (caching, conciseness instructions) → extend third-party risk management to cover AI-specific attack vectors and data-retention practices.
**Trade-offs**: an ongoing organizational process, not a one-time project; underinvesting risks compliance/security exposure, but overinvesting in process before any real GenAI usage exists can slow legitimate experimentation — scale governance rigor to actual usage maturity.
