# Chapter 9: Deployment of Applications

## Core Idea
Deploying LLMs at scale requires solving three interlocking challenges — infrastructure (GPU/CPU/RAM), storage (model parameters + training data, often optimized via quantization), and the deployment process itself (serialization → containerization → API → scaling) — which major cloud providers (AWS, Azure, GCP) offer managed services to address.

## Frameworks Introduced
- **LLM deployment challenge framework**: Infrastructure (hardware: GPU/TPU/CPU/RAM) + Storage (training data + model parameters) + Deployment process (serialization → API → containerization → scaling).
  - When to use: as a checklist before deploying any LLM-based application to production.
- **Deployment process pipeline**: (1) model serialization (save in a deployment format, e.g., TensorFlow SavedModel, PyTorch TorchScript), (2) API development (typically RESTful), (3) containerization (Docker — package code + dependencies into a portable, consistent unit), (4) deployment (transfer + configure in target environment), (5) scaling (horizontal: add instances; vertical: increase resources per instance), (6) security + monitoring/alerting.
- **Cloud deployment comparison**: AWS (SageMaker for managed hosting/scaling + S3/Lambda integration; ECS for granular container orchestration), Azure (Azure Machine Learning for model management/real-time + batch inference; AKS for managed Kubernetes), GCP (Vertex AI for unified ML platform with model registry/endpoints; AI Platform Pipelines for full ML lifecycle management).
  - When to use: core deployment steps are the same across providers — pick based on existing cloud investment, need for granular container control (ECS/AKS) vs. managed simplicity (SageMaker/Vertex AI/AML).
- **Hardware-challenge mitigation approaches**: custom AI hardware (TPUs), model sparsity (fewer parameters, similar performance), software optimizations (better framework/algorithm efficiency).

## Key Concepts
- **Horizontal scaling**: adding more LLM instances across machines to distribute load.
- **Vertical scaling**: increasing CPU/GPU/memory resources allocated to an existing instance.
- **Containerization (Docker)**: packaging code + dependencies into a self-contained, portable unit that runs consistently across environments.
- **Model serialization**: saving a trained model's architecture and weights in a deployment-ready format.
- **Quantization (storage context)**: converting 32-bit float parameters to lower-precision (e.g., 8-bit) representations to reduce storage footprint (4 bytes/parameter at FP32 vs. less at lower precision).
- **TPU (Tensor Processing Unit)**: specialized AI hardware chip optimized for ML workloads, an alternative/complement to GPUs.

## Mental Models
- Think of LLM deployment as three stacked layers: hardware (can it run?) → storage (can it fit?) → process (can it serve requests reliably and scale?) — each layer has distinct bottlenecks and mitigations.
- Use the horizontal-vs-vertical scaling choice as a first lever: horizontal for distributing growing request volume across machines; vertical for giving a single bottlenecked instance more resources.
- Map cloud provider services to a common deployment skeleton (serialize → containerize → expose via API → deploy → scale) rather than treating AWS/Azure/GCP tooling as fundamentally different processes — only the managed-service names differ.

## Anti-patterns
- **Ignoring storage-precision trade-offs**: defaulting to 32-bit float storage for all parameters when quantization could substantially cut storage/compute needs with an acceptable precision trade-off.
- **Skipping containerization for "simple" deployments**: inconsistent environments across dev/staging/production cause hard-to-debug failures that Docker's self-contained packaging prevents.
- **Treating vertical scaling as a universal fix**: there are practical ceilings to how much you can scale a single instance — horizontal scaling is often necessary as load grows, and the two approaches solve different bottlenecks.
- **Deploying without monitoring/alerting in place**: performance degradation, resource exhaustion, or errors in a live LLM service go undetected without active monitoring.

## Code Examples
No code examples in this chapter (infrastructure/process-focused chapter with reference links to Kubernetes/Argo CD deployment of LLMs).

## Reference Tables
None in this chapter (Figure 9.1 illustrates containerization conceptually, no data table).

## Worked Example
General cloud deployment walkthrough (not tied to one specific model): serialize the trained LLM (TensorFlow SavedModel / PyTorch TorchScript) → select cloud infrastructure (SageMaker / Azure ML / Vertex AI) → optionally containerize with Docker → build a RESTful API to expose the model → upload model + container + API code to the chosen service → configure horizontal or vertical scaling based on expected load → apply security (auth/authorization) and monitoring/alerting. The chapter points to a reference implementation for automated Kubernetes + Argo CD autoscaling deployment of LLMs (github.com/halilagin/LLMs-on-kubernetes).

## Key Takeaways
1. LLM deployment at scale requires addressing infrastructure, storage, and process challenges together, not in isolation.
2. Quantization (lower-precision parameter storage) directly reduces both storage footprint and compute requirements — the same lever introduced for fine-tuning in Ch6 applies at the deployment/storage layer too.
3. Docker-based containerization is the standard way to ensure consistent LLM behavior across local, staging, and cloud environments.
4. Horizontal and vertical scaling solve different bottlenecks — plan for both as load grows.
5. AWS, Azure, and GCP offer functionally similar deployment pipelines (serialize → infra select → containerize → API → deploy → scale) with different managed-service names (SageMaker/ECS, Azure ML/AKS, Vertex AI/AI Platform Pipelines).
6. On-premise deployment offers greater transparency/control; cloud deployment offers superior scalability — the choice depends on organizational priorities.

## Connects To
- **Ch6**: Extends quantization (LoRA/QLoRA/PEFT) from a training-efficiency technique into a deployment/storage-efficiency technique.
- **Ch7**: Builds directly on the Hugging Face → AWS SageMaker TGI deployment pattern shown in Chapter 7 (image URI → model config → deploy → predict).
- **Ch10**: Deployment-stage security and monitoring considerations here connect to the broader responsible-AI and data-privacy discussion in the closing chapter.
