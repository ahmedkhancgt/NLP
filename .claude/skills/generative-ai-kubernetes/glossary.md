# Glossary

**Admission Controller** — Kubernetes component validating/mutating objects at creation (e.g. CertificateApproval, ImagePolicyWebhook, Pod Security Admission). (Ch10)

**AIBOM** — AI-specific extension of Software Bill of Materials, tracking models, dependencies, training data, metadata. (Ch10)

**ARI (Automated Readability Index)** — readability metric based on characters/words and words/sentences ratios. (Ch6, Ch9)

**AWQ (Activation-aware Weight Quantization)** — GPU quantization method prioritizing precision for weights with high activation importance. (Ch4)

**CLIP (Contrastive Language-Image Pre-training)** — maps image content into language's latent space, enabling text-to-image generation (used in DALL-E). (Ch1)

**ClusterRole / Role** — Kubernetes RBAC objects; ClusterRole is cluster-wide scope, Role is namespace-scoped. (Ch10)

**ConfigMap** — Kubernetes object storing environment-specific key-value configuration independent of application code. (Ch5, Ch6)

**Context window** — the token limit (input+output) an LLM can process/recall in a single interaction. (Ch4, Ch8)

**Controller / control loop** — Kubernetes mechanism continuously reconciling actual state toward desired state. (Ch2, Ch6)

**CRD (Custom Resource Definition)** — Kubernetes mechanism for defining custom resource types (e.g. SeldonDeployment) with accompanying controllers. (Ch2, Ch6)

**Data lakehouse** — matured data lake pattern combining lake flexibility with governance/metadata maturity. (Ch7)

**Data swamp** — a data lake that lost governance/traceability, becoming hard to extract value from despite scale. (Ch7)

**Device Plugin Framework** — Kubernetes extensibility mechanism letting vendors expose specialized hardware (e.g. GPUs) as schedulable resources. (Ch2)

**Diffusion model** — generative model that learns to reverse a noise-adding process to produce new images from noise, guided by latent-space prompts. (Ch1, Ch3)

**DPO (Direct Preference Optimization)** — RLHF alternative that skips training a separate reward model. (Ch9)

**Embeddings** — numeric vector representations of text/images capturing semantic relationships; the basis of similarity search. (Ch1, Ch7, Ch8)

**EncryptionConfiguration** — Kubernetes API server resource defining tiered encryption-at-rest providers per resource type. (Ch10)

**Fine-tuning** — further training a pretrained model on task/domain-specific data; costlier than prompting/RAG but more targeted than training from scratch. (Ch4, Ch8)

**Flesch-Kincaid grade level** — readability metric based on syllables-per-word ratio. (Ch6, Ch9)

**Foundational Model** — a model built via ML at massive scale (data+compute), general-purpose, not limited to language. (Ch1)

**GAN (Generative Adversarial Network)** — two competing models (generator vs discriminator) trained jointly to produce realistic synthetic data. (Ch1)

**GGUF / GGML** — quantized model file formats for CPU inference (GGUF is GGML's successor). (Ch4)

**GPTQ** — GPU quantization method: 4-bit stored weights, dynamically dequantized to 16-bit during inference. (Ch4)

**Hallucination** — plausible-sounding but factually incorrect model output. (Ch1, Ch8, Ch9)

**HPA (Horizontal Pod Autoscaler)** — scales Pod replica count based on a metric (e.g. CPU%). (Ch2, Ch6)

**IA3** — PEFT technique injecting learned vectors to amplify/suppress inter-layer activations. (Ch4, Ch8)

**Indirect Prompt Injection** — malicious content injected via an external data source (e.g. a poisoned RAG document) rather than directly by the user. (Ch10)

**Inference** — using a trained model to generate output from input; far less compute-intensive than training. (Ch1, Ch3)

**Instruct-tuned model** — a model with added supervised fine-tuning for instruction-following (naming convention: "-Instruct" suffix). (Ch4)

**kube-state-metrics** — Kubernetes add-on exposing object state metrics via HTTP for Prometheus scraping. (Ch9)

**KServe** — independent (formerly Kubeflow KFServing) OSS model-serving platform supporting Open Inference Protocol V2. (Ch6)

**Knowledge distillation** — training a smaller "student" model on a larger "teacher" model's outputs. (Ch4, Ch6)

**LangKit** — WhyLabs open-source library for monitoring LLM prompts/responses (text quality, toxicity, injection, PII). (Ch9)

**Latent space** — the numeric/vector encoding space where transformers compress meaning and context. (Ch1)

**Legal hold** — indefinite object lock, settable/removable only by privileged users. (Ch7)

**Lifecycle/Tiering rule** — automated storage policy to transition/expire objects to another tier after a time period. (Ch7)

**LLM Cascading** — cost-optimization technique: query the cheapest model first, escalate to pricier models only if needed. (Ch10)

**LoRA (Low-Rank Adaptation)** — popular PEFT technique adding small low-rank adapter weights on top of a frozen base model. (Ch4, Ch8)

**MIG (Multi-Instance GPU)** — GPU-sharing approach providing hardware-level isolation; requires MIG-capable GPUs and higher maintenance. (Ch4)

**MLflow** — open-source model tracking, registry, and deployment server; central tool in Ch5-6's pipeline examples. (Ch5, Ch6)

**MLOps / LLMOps** — DevOps-derived practices for ML (MLOps) and the LLM-specific variant (LLMOps) accounting for non-determinism and alignment needs. (Ch5, Ch8)

**Model registry** — tracks model versions, lineage, and metadata/metrics (MLflow's core registry function). (Ch5, Ch6)

**Model theft** — extraction of proprietary model weights/parameters via prompt-based extraction or infrastructure breaches. (Ch10)

**MPS (Multi-Process Service)** — GPU-sharing approach; not production-ready/not officially supported in NVIDIA's device plugin as of writing. (Ch4)

**Namespace** — Kubernetes grouping mechanism for resources; provides organization and quota enforcement but not full isolation by itself. (Ch2, Ch6)

**Object locking (Governance/Compliance mode)** — S3-standard retention enforcement; Governance allows privileged override, Compliance enforces for all users. (Ch7)

**Open Inference Protocol V2** — standard inference API adopted by Seldon Core, KServe, NVIDIA Triton for model-server portability. (Ch6)

**Operator pattern** — community-defined CRD plus controller managing custom resource lifecycle beyond core Kubernetes primitives. (Ch2, Ch6)

**OWASP Top 10 for LLM Applications** — community-maintained list of LLM-specific application security risks with examples/mitigations. (Ch10)

**PEFT (Parameter-Efficient Fine-Tuning)** — fine-tuning only a small added parameter subset while freezing the base model (e.g. LoRA, IA3). (Ch4, Ch8)

**PersistentVolume / PersistentVolumeClaim (PVC)** — Kubernetes constructs decoupling storage from Pod lifecycle. (Ch7)

**Pod Security Standards** — Baseline/Privileged/Restricted levels controlling container access to host processes, namespaces, volumes, ports. (Ch10)

**Presidio** — Microsoft's PII-detection framework, integrated into LangKit. (Ch9)

**Probes (startup/liveness/readiness)** — Kubernetes health checks gating restarts and traffic routing. (Ch2)

**Prompt engineering** — structuring prompts (zero-shot, few-shot, chain-of-thought, etc.) to improve model output without changing weights. (Ch4, Ch8)

**Prompt injection** — using crafted language/encoding to compromise a model's safeguards or intended behavior (direct or indirect). (Ch10)

**Pruning** — removing/zeroing redundant or low-impact model weights/layers. (Ch4)

**Quantization** — reducing numeric precision of model weights to cut storage/compute (e.g. GGUF for CPU, GPTQ/AWQ for GPU). (Ch1, Ch4)

**RAG (Retrieval Augmented Generation)** — combining vector-based retrieval of relevant documents with LLM generation to ground responses in domain data. (Ch7, Ch8)

**RBAC (Role Based Access Control)** — Kubernetes authorization model using Roles/ClusterRoles and RoleBindings. (Ch2, Ch10)

**ResourceQuota** — Kubernetes object enforcing hard resource caps (CPU/memory/GPU) per namespace. (Ch2)

**RLHF (Reinforcement Learning with Human Feedback)** — alignment technique: supervised fine-tuning + reward model + RL (e.g. PPO) fine-tuning. (Ch1, Ch9)

**ROUGE** — summarization-quality metric comparing generated summaries to human references. (Ch6)

**Safetensors** — secure model weight storage format, alternative to pickle (which risks arbitrary code execution). (Ch3, Ch10)

**Self-consistency prompting** — sampling the same prompt multiple times and selecting the most common/consistent answer. (Ch8)

**Seldon Core** — open-source Kubernetes-native model-serving platform using the Open Inference Protocol V2 and SeldonDeployment CRD. (Ch6)

**Service Account** — namespace-scoped, non-human Kubernetes identity for application access to Secrets/ConfigMaps. (Ch10)

**StorageClass** — Kubernetes construct defining how PersistentVolumes are provisioned (local, NFS, cloud, etc.). (Ch7)

**TEVV (Testing, Evaluation, Verification, Validation)** — NIST-recommended continuous evaluation practice across the AI model lifecycle. (Ch10)

**Token** — sub-word/word/character unit models operate on, converted to vectors. (Ch1, Ch4)

**Toxicity metric** — pretrained classifier scoring presence of hate/abusive speech (0-1 scale). (Ch6, Ch9)

**Transformer** — neural architecture using attention to model context/relationships across a sequence, enabling parallelization and long-range context. (Ch1)

**VADER** — NLTK's rules-based, social-media-tuned sentiment analyzer (-1 to 1). (Ch9)

**VAE (Variational Autoencoder)** — encodes/decodes between pixel space and latent space in diffusion models. (Ch3)

**Vector database** — database storing embedding vectors and supporting similarity (nearest-neighbor) search, e.g. ChromaDB. (Ch7, Ch8)

**VPA (Vertical Pod Autoscaler)** — raises CPU/memory limits of individual Pods based on resource pressure. (Ch2)
