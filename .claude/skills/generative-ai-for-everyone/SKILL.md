---
name: generative-ai-for-everyone
description: "Knowledge base from \"Generative AI for Everyone\" by Karthikeyan Sabesan, Sivagamisundari, and Nilip Dutta (BPB Publications, 2025). Use when exploring generative AI broadly — AI/GenAI fundamentals, image generation and transformation, text generation and ChatGPT, LLM frameworks and operations (LLMOps), enterprise GenAI applications, or sustainability/advances in the field."
---

# Generative AI for Everyone
**Authors**: Karthikeyan Sabesan, Sivagamisundari, Nilip Dutta | **Chapters**: 10 | **Generated**: 2026-11-18

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch06`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Foundations (Ch1-2)**: AI is scoped by capability (ANI/AGI/ASI) and by learning paradigm (supervised/unsupervised/semi-supervised/RL). Deep learning's full training stack (activation functions, cost functions, gradient descent, backpropagation, regularization) underlies every GenAI architecture. GenAI is distinguished from ordinary ML by output shape: natural language/image/audio generated from a learned distribution, not a score or class. Foundation models (GPT-1 through GPT-4) are pre-trained once at massive scale and adapted via prompting or fine-tuning rather than retrained per use case. Transfer learning — reusing pre-trained weights — is the default, not the exception, given the infeasibility of training GenAI-scale models from scratch per use case.

**Images (Ch3-4)**: CNNs (AlexNet→VGG→ResNet→MobileNet→DenseNet→EfficientNet) extract hierarchical visual features via convolution/pooling; skip connections (ResNet, U-Net) are the general fix for vanishing gradients in deep networks. Representation learning (autoencoder→VAE→VQ-VAE) maps images to a navigable latent space; VAEs generalize autoencoders by learning a *distribution* over latent codes (via the reparameterization trick), making them generative rather than merely compressive. Three generative-model families compete for image synthesis: GANs (implicit density, adversarial min-max training — generator vs. discriminator), diffusion models (explicit/approximate density, learn to reverse a noise-adding Markov process — current SOTA: DALL-E 2, Stable Diffusion, Midjourney), and flow models (explicit/tractable density, invertible transformations with exact likelihood). Latent diffusion (diffusing in a VAE's compressed latent space) is the key efficiency unlock behind Stable Diffusion. Evaluate generative image models with FID/IS/LPIPS/SSIM, never with classification metrics.

**Text / ChatGPT (Ch5-6)**: The transformer (encoder-only=BERT for classification; decoder-only=GPT for generation; encoder-decoder=T5 for translation/summarization) replaced RNN/LSTM via parallelizable self-attention. Tokenization (sub-word: BPE/WordPiece/SentencePiece) → embeddings → positional encoding → multi-head self-/cross-attention → language-modeling head is the universal pipeline. Virtually any NLP task reduces to next-token prediction, which is why decoder-only LLMs generalize so broadly. Decoding strategy (greedy/beam=deterministic; top-k/top-p/temperature=stochastic) is a separate lever from the trained model itself. ChatGPT's enterprise value extends beyond chat to fine-tuning, plugins, and Enterprise security features; prompt engineering (zero-shot→few-shot→CoT→ToT, escalating only as task reasoning complexity demands) is a disciplined, evaluable process, not guesswork. Jailbreaking (direct role-play/DAN prompts, indirect poisoned-data injection, prompt leaking) is a concrete, documented risk requiring layered defenses, not reliance on default guardrails.

**LLM Frameworks / Operations (Ch7-8)**: Production LLM apps need an orchestration framework (LangChain's chains/memory/document loaders/vector stores/agents, or LlamaIndex for retrieval-specialized apps) — raw API calls alone don't scale to real applications. RAG (retrieve relevant chunks → augment query → generate) is the dominant enterprise pattern for grounding LLMs in current/proprietary knowledge without the cost/staleness of fine-tuning; small-to-big (auto-merging) retrieval resolves the precision-vs-context trade-off of naive fixed-chunk RAG. Agents (ReAct pattern) extend LLMs from text generators into action-takers via tool use. LLMOps splits into Phase 1 (building foundation models: data prep→pre-training→alignment→experimentation, rarely worth it for most enterprises) and Phase 2 (serving/securing/monitoring pre-built models, where most real work happens). Alignment technique cost ladder: prompt engineering/RAG (free) < PEFT/LoRA (cheap) < full fine-tuning < DPO/ORPO < RLHF/RLAIF (most expensive) — never reach for a costlier technique than the cheapest one meeting your bar. Inference optimization attacks memory (quantization, KV cache/PagedAttention), compute (FlashAttention, MQA/GQA, pruning/distillation), and scheduling (continuous batching) simultaneously. LLM security requires LLM-specific guardrails (prompt injection, jailbreaking, excessive agency, vector-database attacks) beyond traditional ML security. Drift (concept/data/prediction/prompt/embedding) must be continuously monitored post-deployment, not just checked once pre-launch.

**Enterprise / Sustainability (Ch9-10)**: GenAI's enterprise value spans nearly every function (coding, marketing, customer service, drug discovery, fraud detection), but adoption pace should scale inversely with regulatory stakes. GenAI introduces risk categories beyond general AI risk (hallucination, anthropomorphization, untraceability, unreproducibility) requiring a dedicated Responsible AI strategy (accountability, transparency, fairness, privacy/security, governance) as a prerequisite for sustainable deployment, not an optional add-on. Validate GenAI hype across three independent dimensions — business (economic/job impact), technology (architecture/optimization advances like DoReMi, Sophia, attention-free architectures), and regulation (EU AI Act's risk tiers and evolving global frameworks) — strength in one dimension doesn't guarantee durable impact if the others lag. Sustainability accounting must cover the full lifecycle (training + inferencing + hardware embodied cost) since inferencing emissions often dominate over a model's operational life; prefer the smallest task-fit model and carbon-aware infrastructure choices over defaulting to the largest general-purpose model.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-ai-fundamentals.md) | AI Fundamentals | ANI/AGI/ASI, supervised/unsupervised/SSL/RL, design patterns (encoding, embedding, transfer learning, ensembling, MoE) |
| [ch02](chapters/ch02-genai-foundation.md) | GenAI Foundation | Deep learning math (activations, cost functions, gradient descent, backprop, regularization), discriminative vs. generative, foundation models, GenAI use-case selection |
| [ch03](chapters/ch03-genai-for-images.md) | GenAI for Images | CNN architectures, U-Net segmentation, autoencoders/VAE/VQ-VAE representation learning |
| [ch04](chapters/ch04-transforming-images-with-genai.md) | Transforming Images with GenAI | GANs (cGAN/CycleGAN/SRGAN/VQ-GAN), diffusion models (DDPM/latent diffusion), flow models, image-gen evaluation metrics |
| [ch05](chapters/ch05-genai-for-text.md) | GenAI for Text | Sequence models (RNN/LSTM), transformer architecture (encoder/decoder/encoder-decoder), tokenization, decoding strategies, NLP task patterns (QA, summarization, translation) |
| [ch06](chapters/ch06-chatgpt.md) | ChatGPT | OpenAI product ecosystem, prompt engineering process and techniques (zero/few-shot, CoT, ToT), prompt playground, jailbreaking/prompt injection |
| [ch07](chapters/ch07-large-language-model-frameworks.md) | Large Language Model Frameworks | LangChain (chains, memory, document loaders, vector stores, agents), RAG sequence, small-to-big retrieval, LlamaIndex, property-graph extraction |
| [ch08](chapters/ch08-large-language-model-operations.md) | Large Language Model Operations | LLMOps Phase 1/2, data prep pipeline, AI alignment (PEFT/LoRA/RLHF/DPO/ORPO), model merging, inference optimization, LLM security, drift detection |
| [ch09](chapters/ch09-generative-ai-for-the-enterprise.md) | Generative AI for the Enterprise | Enterprise use-case catalog, GenAI vendor landscape, AI risk/vulnerability taxonomy, Responsible AI principles |
| [ch10](chapters/ch10-advances-and-sustainability-in-genai.md) | Advances and Sustainability in GenAI | Three-dimension hype validation (business/tech/regulation), post-transformer architectures, federated/split learning, GenAI business strategy, carbon/sustainability measurement |

## Topic Index

- **Activation functions** → ch02
- **Agents (LLM)** → ch07, ch08
- **AI alignment (PEFT, LoRA, RLHF, DPO, ORPO)** → ch08
- **AI regulation (EU AI Act, etc.)** → ch10
- **Attention mechanism (self/cross/multi-head)** → ch05
- **Autoencoders / VAE / VQ-VAE** → ch03
- **Benchmarking (LLM evaluation datasets, metrics)** → ch08
- **Carbon footprint / sustainability measurement** → ch10
- **ChatGPT / OpenAI products** → ch06
- **Chunking / document splitters** → ch07
- **CNN architectures (ResNet, EfficientNet, etc.)** → ch03
- **Confidential AI / federated / split learning** → ch10
- **Decoding strategies (greedy, beam, top-k/p, temperature)** → ch05
- **Deep learning math (gradients, backprop, loss functions)** → ch02
- **Diffusion models (DDPM, latent diffusion)** → ch04
- **Drift detection (concept/data/prediction/prompt)** → ch08
- **Enterprise GenAI use cases / vendors** → ch09
- **Ensemble learning / Mixture of Experts** → ch01
- **Fine-tuning (full, PEFT, RLHF, DPO)** → ch08
- **Flow models (Real NVP, GLOW)** → ch04
- **GANs (variants: cGAN, CycleGAN, SRGAN, VQ-GAN)** → ch04
- **Guardrails / LLM security** → ch06, ch08
- **Hallucination** → ch02, ch09
- **Hyperparameters (temperature, top-p, learning rate)** → ch02, ch05, ch06
- **Inference optimization (quantization, KV cache, batching)** → ch08
- **Jailbreaking / prompt injection** → ch06
- **Knowledge distillation / pruning** → ch08
- **LangChain / LlamaIndex** → ch07
- **Learning paradigms (supervised/unsupervised/SSL/RL)** → ch01
- **LLMOps lifecycle** → ch08
- **Memory (conversation)** → ch07
- **Model merging (LoRA composition, TIES, DARE)** → ch08
- **Model serving (vLLM, deployment strategies)** → ch08
- **Monitoring / observability metrics** → ch08
- **Prompt engineering (process, techniques)** → ch06
- **RAG (retrieval-augmented generation)** → ch05, ch07
- **Reinforcement learning (Q-learning, policy gradient)** → ch01
- **Responsible AI** → ch09
- **Sequence models (RNN, LSTM)** → ch05
- **Tokenization (BPE, WordPiece, SentencePiece)** → ch05
- **Transformer architecture (encoder/decoder types)** → ch05
- **Transfer learning** → ch01
- **U-Net** → ch03, ch04
- **Vector stores / embeddings** → ch01, ch07

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — techniques and patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 10 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive, as are any recurring publisher promotional asides. For topics beyond this book's scope, check other skills or ask directly.
