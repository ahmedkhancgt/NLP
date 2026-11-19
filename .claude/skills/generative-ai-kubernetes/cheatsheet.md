# Cheatsheet

## Decision: Which Refinement Technique First?
| Situation | Try First | Escalate To |
|---|---|---|
| Wrong format/tone/length | Prompt engineering (instructions, few-shot) | — |
| Multi-step reasoning fails | Chain-of-thought / least-to-most / plan-and-solve | Self-consistency (sample N, take mode) |
| Needs domain/proprietary knowledge | RAG (vector search + augmented prompt) | PEFT if RAG retrieval insufficient |
| Needs deep behavioral/style change | PEFT (LoRA/IA3) | Full fine-tuning |
| Needs net-new capability no base model has | Full fine-tuning | Training from scratch (rarely justified) |

Rule: always work right-to-left on cost — never reach for fine-tuning before confirming prompting/RAG can't do the job.

## Decision: Job vs. Deployment vs. SeldonDeployment
| Workload shape | Use |
|---|---|
| Runs once, produces a result, exits (batch eval, data processing) | Kubernetes `Job` |
| Long-running, stateless, needs continuous availability | `Deployment` + `Service` |
| Model inference needing standard API, framework portability, native autoscaling | `SeldonDeployment` (or KServe) |

## GPU Sharing Decision Table
| Need | Choose | Caveat |
|---|---|---|
| Simplicity, no isolation needed | Time-slicing | No memory isolation/limits |
| Hard isolation between tenants | MIG | Requires MIG-capable GPU; high maintenance (ConfigMap-managed) |
| Memory limits without full isolation | MPS | Not production-ready / unofficial in Device Plugin as of writing |

## Resource-Sizing Defaults (from book's own hands-on examples)
| Task | Minimum observed | Notes |
|---|---|---|
| Minikube + MLflow stack | 16GB RAM, 128GB disk, 4 CPUs | `minikube start --cpus 4 --memory no-limit` |
| CPU inference on 7B GGUF model (Q4/Q5) | ~16GB RAM | Slow (minutes); GPU gives ~10x speedup |
| Phi-2 GGUF (Q4_K_M) | <2GB storage, ~4GB+ RAM | Smallest model used for RAG example |
| LLaMA-2 70B (full) | ~35GB GPU VRAM minimum to run | Training needs far more (24,000+ GPUs for LLaMA-3 scale) |

## Quantization Format Quick Reference
| Format | Target | Notes |
|---|---|---|
| GGUF (successor to GGML) | CPU inference | Favorite for CPU-only; variable compression levels (e.g. Q4_K_M, Q5_K_M) |
| GPTQ | GPU | 4-bit storage, dequantized to 16-bit at inference |
| AWQ | GPU | Activation-aware; preserves precision for most-activated weights |
| Safetensors | Secure storage (any) | NOT a quantization format — a safe alternative to pickle for weight storage |

## Security Checklist (Ch10)
- [ ] RBAC: grant least-privilege verbs per resource (e.g. `list` only on Secrets, not describe/update/delete) — never default to cluster-admin for app teams.
- [ ] Secrets encrypted at rest via tiered `EncryptionConfiguration` (KMS → local KMS → plaintext fallback), applied selectively (strong for Secrets, optional for ConfigMaps/Events).
- [ ] Model weights loaded as Safetensors, not pickle (pickle risks arbitrary code execution on deserialization).
- [ ] Prompt injection defenses layered: input formatting/sanitization + alignment/fine-tuning + continuous output monitoring — no single layer is sufficient.
- [ ] Indirect injection considered: filters must cover RAG/external data sources, not just direct user input.
- [ ] Model/data supply chain scanned: verify provenance of any downloaded model (public hubs have had 100+ compromised models found in studies).
- [ ] NetworkPolicies and Pod Security Standards (Baseline/Restricted) applied, not left at cluster defaults.
- [ ] Audit policy enabled and scoped to capture who/what/when/where of cluster events.
- [ ] Control plane and data plane separated onto independent hosts in production (not co-located as in minikube demos).
- [ ] Cloud NSGs/security groups restricted to known IPs — never leave SSH/demo ports open to "Any Source."

## Observability Signal Thresholds / Tells
| Signal | Tell / Smell |
|---|---|
| Jailbreak similarity score | Weak separation observed between benign (>0.30) and malicious prompts — don't treat as a hard gate; tune empirically per use case |
| Toxicity outlier in parameter sweep | A config scoring ~4x average toxicity (e.g. specific temp/max_length combo) signals that configuration, not just the model, drives safety risk |
| Refusal-similarity score | High score ≠ definite abuse, but is a signal worth correlating with other metrics for UX/abuse patterns |
| Readability scores (ARI/Flesch-Kincaid) drifting over time | Possible model/prompt drift — track continuously, not just at launch |
| ContextRelevancyEvaluator fails on RAG response | Retrieval dataset likely lacks coverage for that query — do not treat the generated answer as reliable |

## Pipeline Misconfiguration Smells
- Hardcoded server names/ports in app code instead of `ConfigMap` injection → breaks across dev/test/prod.
- No registered baseline before fine-tuning/prompt-tuning experiments → comparisons become meaningless.
- No namespace strategy decided before multi-team scale → resource-naming collisions and governance debt accumulate.
- Data lake with no tagging/lifecycle/versioning → "data swamp" symptoms (unclear ownership, uncontrolled cost, poor traceability).
- RAG system with no evaluator wired in → hallucinated/off-topic answers go undetected in production.
- GPU quotas absent at the namespace level → one team/project can monopolize all cluster GPUs.

## Cost-Optimization Levers (in rough order of ease)
1. Prompt conciseness instructions (e.g. "Please be concise") — cuts output tokens, meaningfully reduces API cost.
2. Quantized models (GGUF/GPTQ/AWQ) over full-precision — large resource savings, often small accuracy loss.
3. Prompt caching for repeated/FAQ-style queries (LangChain/LlamaIndex/LlamaCPP support this) — trade-off: cache-staleness and reduced response variety.
4. Specialized/smaller models for narrow domains instead of general large foundational models.
5. LLM Cascading — try the cheapest model first, escalate only if response quality is insufficient.
