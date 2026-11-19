# Chapter 8: Large Language Model Operations

## Core Idea
LLMOps splits into Phase 1 (building foundation models: data preparation, pre-training, alignment/fine-tuning, experimentation) and Phase 2 (productionizing pre-built models: serving, security, monitoring) — most enterprises operate only in Phase 2, but understanding the full lifecycle (quantization, PEFT, drift detection, guardrails) is essential to deploy LLMs reliably and efficiently at scale.

## Frameworks Introduced
- **LLMOps two-phase lifecycle**: Phase 1 (data prep → pre-training → AI alignment → experimentation) for building foundation models; Phase 2 (model serving → security → monitoring) for deploying pre-built models.
  - When to use: most enterprises only need Phase 2; Phase 1 is reserved for organizations with unique data, compliance needs, or model-as-a-service ambitions.
- **Data quality pipeline for pre-training**: language filtering → quality filtering (classifier/heuristic-based) → deduplication (exact MD5, fuzzy via locality-sensitive hashing/MinHash) → metric-based filtering (perplexity, coherence) → text-repetition removal (suffix arrays) → sensitive-data removal → decontamination (test-set overlap removal) → machine-generated-text detection.
  - When to use: any foundation-model pre-training effort, where data quality directly determines model quality.
- **AI alignment technique ladder**: continued pre-training → full fine-tuning → PEFT (LoRA, prefix tuning, P-tuning) → RLHF/RLAIF → preference optimization (DPO, ORPO).
  - When to use: escalate based on how much the task diverges from general capability, available compute, and whether human-preference alignment (not just task accuracy) is needed.
  - How: continued pre-training updates all weights on new domain data (self-supervised); full fine-tuning updates all weights on labeled data; PEFT updates only a small parameter subset (LoRA's low-rank decomposition, prefix/P-tuning's learned soft prompts); RLHF trains a reward model from human preference ratings and optimizes the policy via PPO; DPO/ORPO achieve RLHF-like alignment more cheaply via direct preference-pair losses without a separate reward model.
- **Model merging techniques**: linear merge, SLERP (pairwise spherical interpolation), task arithmetic (vector addition/subtraction of fine-tuned deltas), TIES (trim + elect sign + disjoint merge), DARE (random pruning + rescaling), Frankenmerging (layer stitching across architectures).
  - When to use: combining multiple task-specialized models into one multi-task model without additional training.
- **Parallel computation strategies for pre-training**: data parallelism, tensor parallelism, pipeline parallelism, combined as 3D parallelism.
  - When to use: scaling training across many GPUs for models too large for a single device.
- **Model serving deployment strategies**: shadow, A/B testing, multi-armed bandit (MAB), blue-green, canary, recreate.
  - When to use: match to risk tolerance — shadow/canary for low-risk gradual rollout, blue-green for instant cutover with rollback, MAB for adaptive optimization vs. static A/B.
- **Inference optimization stack**: architecture-level (KV cache, Multi-Query/Grouped-Query Attention, PagedAttention, FlashAttention) + model-level (quantization, pruning, knowledge distillation) + serving-level (continuous/in-flight batching, speculative decoding).
  - When to use: reducing latency/cost at whichever layer is the current bottleneck.
- **LLM security/guardrails categorization**: data protection, model protection, application security, infrastructure security, regulatory compliance — covering training data poisoning, prompt injection, excessive agency, data leakage, API security.
  - When to use: any production LLM application exposed to untrusted input or acting with real-world permissions (agents).
- **Drift detection framework**: concept drift, data drift, prediction drift, prompt drift, upstream drift, embedding drift — monitored via KS statistic, Wasserstein distance, KL/JS divergence, PSI, or PCA+clustering (embedding drift).

## Key Concepts
- **PEFT (Parameter-Efficient Fine-Tuning)**: updates a small subset of parameters instead of the full model; includes LoRA (low-rank weight decomposition), prefix tuning (learned prefix vectors per transformer block), P-tuning (learned soft prompts via a small trainable encoder).
- **LoRA**: decomposes weight updates into low-rank matrices, drastically reducing trainable parameters while allowing multiple task-specific adapters on one base model.
- **RLHF vs. DPO/ORPO**: RLHF trains a separate reward model and optimizes via PPO (expensive, human-feedback-dependent); DPO/ORPO achieve similar preference alignment directly from (chosen, rejected) response pairs without a separate reward model (cheaper, simpler).
- **KV cache**: caches previously computed key/value attention vectors during autoregressive generation to avoid redundant recomputation; size = `2·p·h·L·s·b` (precision × head dim × layers × sequence length × batch).
- **PagedAttention**: manages KV cache in non-contiguous memory blocks (like OS paging) to eliminate memory waste from fixed-size pre-allocation.
- **Quantization (PTQ)**: post-training compression of weights to lower precision (GPTQ, AWQ, GGUF, EXL2) trading a small accuracy loss for large memory/speed gains.
- **Knowledge distillation**: training a smaller "student" model to mimic a larger "teacher" model's behavior (response-based, feature-based, or relationship-based loss; offline, online, or self-distillation).
- **Continuous/in-flight batching**: executes multiple inference requests concurrently without waiting for the whole batch to finish, improving GPU utilization over static/dynamic batching.
- **Model drift types**: concept drift (input-output relationship changes), data drift (input distribution changes), prediction drift (output distribution changes), prompt drift (same prompt yields different responses over time), upstream drift (data pipeline changes).
- **Model card**: a standardized transparency document (model description, dataset details, evaluation results, limitations) serving developers, business, governance, and ethics teams differently.

## Mental Models
- Think of **Phase 1 vs. Phase 2 as "build vs. buy" for LLMs**: Phase 1 (pre-training foundation models) is almost never worth it for most enterprises given its cost; Phase 2 (serving, securing, monitoring pre-built models) is where the overwhelming majority of real LLMOps work happens.
- Use the **alignment-cost ladder**: full fine-tuning (expensive, all params) → PEFT/LoRA (cheap, few params) → RLHF (expensive, needs reward model + human labels) → DPO/ORPO (cheaper, direct preference pairs) — pick the cheapest technique that meets your accuracy/alignment bar.
- Think of **inference optimization as attacking three independent bottlenecks**: memory (quantization, KV cache management), compute (attention variants, batching), and I/O (request scheduling) — a real production system typically needs techniques from all three, not just one.
- Use the **guardrails-as-layered-defense model**: no single control stops prompt injection, data leakage, or excessive agency — defense requires input/output filtering, scoped permissions, human-in-the-loop for agents, and continuous monitoring together.
- Think of **drift as a compounding operational cost**: a model that was accurate at launch degrades silently as the real world diverges from training-time assumptions — monitoring (not just initial evaluation) is what catches this.

## Anti-patterns
- **Running full fine-tuning when PEFT (LoRA) would suffice**: wastes compute/storage and increases catastrophic-forgetting risk — reserve full fine-tuning for cases with abundant supervised data and a genuine need for complete parameter updates.
- **Naive static batching for LLM inference**: batch completion time is bottlenecked by the longest sequence, wasting GPU time on already-finished sequences — use continuous/in-flight batching instead.
- **Pre-allocating KV cache memory for maximum possible sequence length on every request**: wastes GPU memory on unused reservation — use PagedAttention's non-contiguous block allocation instead.
- **Relying solely on built-in model guardrails for agent systems**: excessive agency (unscoped permissions/autonomy) is a distinct, serious risk requiring explicit permission scoping, human-in-the-loop approval, and bounded task definitions — not just prompt-level filtering.
- **Treating benchmark scores at face value without checking for benchmark leakage**: a model trained on data overlapping its benchmark test set will score artificially high; decontamination (n-gram overlap removal) is required for valid evaluation.
- **Skipping drift monitoring after deployment**: a model evaluated well at launch can silently degrade as real-world data distributions shift — concept/data/prediction drift must be continuously monitored, not just checked once pre-deployment.

## Code Examples
```python
# Serving an open-source LLM with vLLM (PagedAttention-based high-throughput inference)
from vllm import LLM, SamplingParams
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")

user_query = ["what is the capital of France?", "What is the smallest prime number that divides the number 111"]
params = SamplingParams(temperature=0.9, top_p=0.8)
llm_responses = llm.generate(user_query, params)
for response in llm_responses:
    print(f"Prompt: {response.prompt!r}, Generated text: {response.outputs[0].text!r}")
```
```python
# Hugging Face transformer pipeline for summarization (abstraction over tokenize -> infer -> postprocess)
from transformers import pipeline
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
summarizer(article, min_length=5, max_length=30)
```
```python
# Loading a dataset from Hugging Face Hub for fine-tuning experimentation
from datasets import load_dataset_builder, load_dataset
ds_builder = load_dataset_builder("Open-Orca/SlimOrca")
print(ds_builder.info.description, ds_builder.info.features)
dataset = load_dataset("Open-Orca/SlimOrca", split="train")
```
- **What it demonstrates**: (1) production-grade LLM serving via vLLM with sampling parameters; (2) the Hugging Face `pipeline` abstraction that wraps tokenization, inference, and postprocessing into one call; (3) programmatic dataset discovery and loading for experimentation/fine-tuning workflows.

## Reference Tables
**AI alignment technique comparison**

| Technique | Params updated | Needs reward model? | Cost |
|---|---|---|---|
| Continued pre-training | All | No | High |
| Full fine-tuning | All | No | High |
| LoRA / PEFT | Small subset | No | Low |
| RLHF | All (policy) + reward model | Yes | Very high |
| DPO / ORPO | All or subset | No (direct preference pairs) | Moderate |

**Inference optimization techniques by bottleneck**

| Bottleneck | Techniques |
|---|---|
| Memory | Quantization (GPTQ/AWQ/GGUF), KV cache, PagedAttention |
| Compute | FlashAttention, MQA/GQA, pruning, knowledge distillation |
| Scheduling/throughput | Continuous (in-flight) batching, speculative decoding |

**Deployment strategy comparison**

| Strategy | Mechanism | Risk profile |
|---|---|---|
| Shadow | New model runs parallel, no live traffic exposure | Very low risk |
| Canary | Small traffic subset to new model, gradually increased | Low risk, gradual |
| Blue-green | Instant cutover between two full environments | Fast rollback, resource-costly |
| A/B testing | Fixed split for hypothesis testing | Exposes some users to suboptimal model |
| Multi-armed bandit | Adaptive, dynamically favors best performer | More complex, better outcomes |
| Recreate | Full scale-down then scale-up | Downtime, simplest |

**LLM observability layers and example metrics**

| Layer | Example metrics |
|---|---|
| Training/fine-tuning | Loss, accuracy, learning rate, gradient norm |
| Inference | Latency (TTFT, TPOT), throughput, MFU, MBU, tail latency |
| Safety | Refusal ratio, jailbreak attempt detection |
| Drift | KS statistic, Wasserstein distance, KL/JS divergence, PSI |

**Model drift types**

| Drift type | Cause |
|---|---|
| Concept drift | Input-output relationship (ground truth) changes |
| Data drift | Input data distribution changes |
| Prediction drift | Output distribution changes (signals underlying data drift) |
| Prompt drift | Same prompt yields different responses over time (model change/migration) |
| Upstream drift | Data pipeline changes |
| Embedding drift | Similarity shift between reference and incoming request/response embeddings |

## Worked Example
**RAG experimentation with MLflow + Giskard**: Using Elastic Cloud (vector store with ELSER model), LangChain, and Hugging Face models, build a RAG pipeline and systematically test different LLM model responses for a fixed set of user queries. Track each experiment run (model, prompt, retrieved chunks, response) with MLflow's LLM tracking tool, and evaluate output quality/vulnerabilities (hallucination, bias, toxicity, security issues) automatically using the Giskard scanner, which combines heuristic-based detectors with LLM-assisted detectors (using GPT-4 to probe the system). The resulting dashboard highlights specific vulnerabilities per configuration, enabling data-driven selection of the best-performing RAG + model combination rather than relying on ad hoc manual comparison.

## Key Takeaways
1. Most enterprises should operate in LLMOps Phase 2 (serving/securing/monitoring pre-built models) rather than Phase 1 (building foundation models from scratch) given the cost and expertise required for the latter.
2. PEFT techniques (especially LoRA) are the practical default for task-specific adaptation — reserve full fine-tuning and RLHF for cases where their added cost is clearly justified.
3. Inference optimization requires attacking memory (quantization, KV cache/PagedAttention), compute (attention variants, pruning/distillation), and scheduling (continuous batching) simultaneously — no single technique solves latency/cost alone.
4. LLM security is categorically different from traditional ML security: prompt injection, jailbreaking, excessive agency, and vector-database-specific attacks (embedding inversion) require LLM-specific guardrail tools (NeMo-Guardrails, Guardrails AI, Vigil, Rebuff, Garak).
5. Drift monitoring is not optional post-deployment — concept, data, prediction, prompt, and embedding drift can silently degrade a model that performed well at launch.
6. Model merging (LoRA composition, TIES, DARE, task arithmetic) offers a low-compute alternative to fine-tuning for combining task-specific capabilities into one model.
7. Observability must span training/fine-tuning, inference, safety, and drift layers — a model evaluated only once pre-deployment has no visibility into real-world performance degradation.

## Connects To
- **Ch 1**: data parallelism, distributed training strategies, and mixture-of-experts concepts introduced there are directly operationalized here into 3D parallelism and MoE-based LLM scaling.
- **Ch 5-6**: fine-tuning concepts (introduced lightly in text generation and ChatGPT chapters) are expanded here into the full PEFT/RLHF/DPO alignment technique landscape.
- **Ch 7**: the LangChain/LlamaIndex application components built there are the systems LLMOps serves, secures, and monitors in production.
- **Ch 9**: the security/guardrails and risk frameworks here feed directly into the enterprise AI risk and responsible-AI discussion that follows.
