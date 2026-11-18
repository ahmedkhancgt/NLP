# Cheatsheet

## Is this task GenAI at all?
| Output shape | Verdict |
|---|---|
| Number / class / score | Discriminative task — not GenAI |
| Natural language, image, audio, code | Candidate GenAI task |

## Learning paradigm selection
| Data situation | Approach |
|---|---|
| Fully labeled | Supervised learning |
| No labels, want structure | Unsupervised learning (clustering/dim. reduction) |
| Mostly unlabeled + small labeled set | Semi-supervised learning (pseudo-labeling/consistency/hybrid) |
| Sequential decisions + reward signal | Reinforcement learning |

## Image generative model family selection
| Need | Family | Why |
|---|---|---|
| Sharp, realistic samples, training stability manageable | GAN | Implicit density, adversarial training |
| State-of-the-art fidelity, conditional generation (text-to-image) | Diffusion | Stable MSE-on-noise training objective |
| Exact likelihood / lossless reconstruction required | Flow model (Real NVP/GLOW) | Tractable, invertible transforms |
| Fast training/sampling at scale | Latent diffusion | Diffuses in compressed latent space |
| Unpaired domain translation | CycleGAN | Cyclic-consistency loss, no paired data needed |
| Paired domain translation | cGAN (pix2pix) | Faster/better when paired data exists |

## Transformer architecture selection
| Need | Architecture | Examples |
|---|---|---|
| Fixed representation (classification, NER, sentiment) | Encoder-only | BERT, RoBERTa |
| Open-ended generation, completion, QA | Decoder-only | GPT, LLaMA, PaLM |
| Sequence-to-sequence transformation (translation, summarization) | Encoder-decoder | T5, BART |

## Decoding strategy selection
| Need | Strategy |
|---|---|
| Precision/reproducibility (translation) | Greedy / beam search |
| Creativity/diversity (open writing) | Top-k / top-p (nucleus) / temperature sampling |
| k=1 top-k | Equivalent to greedy search |

## Alignment/fine-tuning technique selection (cheapest-first)
1. **Prompt engineering** (zero/few-shot, CoT) — no training needed
2. **RAG** — ground with retrieval, no training needed
3. **PEFT/LoRA** — small param subset, cheap, fast
4. **Full fine-tuning** — all params, needs abundant labeled data
5. **DPO/ORPO** — preference alignment, no reward model needed
6. **RLHF/RLAIF** — most expensive, needs reward model + PPO
7. **Continued pre-training** — domain adaptation at pre-training scale (rare for most enterprises)

Default rule: never reach for a more expensive technique than the cheapest one that meets your accuracy/alignment bar.

## RAG chunking/chain strategy by data shape
| Data shape | Strategy |
|---|---|
| Fits in 1-2 chunks | Stuffing |
| Independent, parallelizable documents | Map-reduce |
| Need single best-confidence source | Map-rank |
| Need incrementally-built accuracy | Refine |
| Answer spans multiple sections | Small-to-big (auto-merging) retrieval |

## Inference optimization: which lever for which bottleneck
| Bottleneck symptom | Fix |
|---|---|
| GPU out of memory | Quantization (GPTQ/AWQ/GGUF), gradient/activation checkpointing |
| High latency per token | FlashAttention, MQA/GQA, KV cache |
| Low GPU utilization across requests | Continuous/in-flight batching |
| Memory wasted on padding reservations | PagedAttention |
| Model too large to deploy | Pruning, knowledge distillation, quantization |

## LLMOps deployment strategy by risk tolerance
| Risk tolerance | Strategy |
|---|---|
| Very low (no live exposure) | Shadow |
| Low, gradual | Canary |
| Need fast rollback | Blue-green |
| Hypothesis testing with users | A/B testing |
| Adaptive optimization | Multi-armed bandit |
| Simplicity over uptime | Recreate |

## Drift monitoring tells (what to watch, when)
| Symptom | Likely drift type | Metric |
|---|---|---|
| Same prompt → different answers over time | Prompt drift | Manual/version tracking |
| Model accuracy degrading on live data | Concept or data drift | KS statistic, PSI, KL/JS divergence |
| Output distribution shifting | Prediction drift | Compare output distributions |
| Embedding similarity to reference dropping | Embedding drift | PCA + clustering (inertia, silhouette) |
| Upstream pipeline changed silently | Upstream drift | Pipeline/schema monitoring |

## LLM security red flags (jailbreak/injection smells)
- Prompt asks the model to "pretend," role-play, or adopt a persona with no restrictions (DAN-style) — likely direct injection.
- Prompt asks the model to reveal its own system instructions — prompt leaking.
- Unusually long prompt, high toxicity score, or encoded/obfuscated text (base64, synonym substitution) — jailbreak attempt.
- External content (webpage, document, plugin output) contains instructions addressed to "the AI" — indirect injection.
- An agent is asked to perform an action beyond its stated scope — excessive agency risk.
- Mitigation defaults: input/output filtering + delimiters separating instructions from data + scoped agent permissions + human-in-the-loop for consequential actions + red-teaming before deployment.

## Hallucination mitigation defaults
1. Ground with RAG (retrieval context) before trusting direct generation for factual queries.
2. Tune "creativity" (temperature/top-p) down for factual tasks.
3. Add moderation/validation pass on outputs for compliance-sensitive use cases.
4. Add explicit disclaimers where appropriate.
5. Start enterprise GenAI rollout with low-consequence-of-error use cases.

## Carbon/sustainability defaults
- Prefer the smallest/task-specific model that meets requirements over a general-purpose foundation model.
- Measure both training AND inferencing emissions (inferencing often dominates over model lifetime).
- Image generation > text generation in carbon intensity — account for this when choosing modality.
- Prefer cloud regions with cleaner energy grids and higher PUE (Power Usage Effectiveness).

## Quick numeric anchors (approximate, from the book)
- GPT-3: 175B params; GPT-4: ~1T params (reported).
- Context windows: BERT 512 tokens, BART/PEGASUS 1024, LongT5 4096 (up to 16k w/ sparse attention), GPT-4o 128k.
- Memory requirement grows sharply (~exponentially) past ~4096 tokens of context.
- 1 token ≈ 4 English characters (including spaces).
- GPU memory for full-precision (FP32) training ≈ 4 bytes/parameter (plus optimizer/gradient/activation overhead).
