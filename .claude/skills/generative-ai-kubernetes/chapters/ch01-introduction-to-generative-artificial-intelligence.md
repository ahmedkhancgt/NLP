# Chapter 1: Introduction to Generative Artificial Intelligence

## Core Idea
GenAI is the result of combining transformer architecture (context/attention) with massive scale (data + GPUs); understanding this lineage — AI → ML approaches → neural networks/deep learning → transformers → foundational models — is the prerequisite for reasoning about the infrastructure GenAI workloads demand.

## Frameworks Introduced
- **AI/ML taxonomy**: AI is the broad field; ML (supervised, unsupervised, reinforcement learning) is the subfield driving recent progress; deep learning (multi-layer ANNs) is the subfield of ML behind GenAI.
  - When to use: to scope a project — classification/regression (supervised), anomaly/clustering (unsupervised), or simulated reward-driven tasks (reinforcement learning).
  - How: identify whether labeled outcomes exist (supervised), only pattern grouping is needed (unsupervised), or an agent acts in an environment for reward (RL).
- **Transformer attention mechanism**: from "Attention Is All You Need" (Google, 2017) — tokens are weighted by contextual relevance ("attention") rather than processed strictly sequentially, enabling parallelization and long-range context (latent space encoding).
  - When to use: any task needing contextual understanding across a sequence (language, and by extension image/audio generation via latent spaces).
  - How: tokenize input, compute attention-weighted relationships between tokens, encode into latent space vectors.
- **InstructGPT 3-step alignment**: (1) supervised fine-tuning on human-written example responses, (2) humans rank multiple model outputs to train a reward model, (3) reinforcement learning uses the reward model to fine-tune the base model.
  - When to use: converting a raw/base foundational model into one that follows instructions and avoids offensive output.
  - How: collect human examples → train reward model from ranked outputs → RL fine-tune against reward model.

## Key Concepts
- **Foundational Model**: a model built via ML at massive scale (data + GPUs), general-purpose and not limited to language (Stanford-coined term).
- **LLM**: a foundational model specialized for language (GPT, PaLM, LLaMA, Claude, etc.).
- **Tokens**: sub-word/word/character units models operate on, converted to vectors; not the same as "words."
- **Latent space**: the numeric/vector encoding space where transformers compress meaning and context.
- **Hallucination**: a plausible-sounding but factually incorrect model output, since models predict likely next tokens, not verified truth.
- **GAN (Generative Adversarial Network)**: two competing models — generator (creates fakes) vs. discriminator (detects fakes) — trained jointly (Ian Goodfellow, 2014).
- **Quantization**: reducing numeric precision of model weights (e.g., 32-bit float → 8-bit int) to cut storage/compute at some accuracy cost.
- **CLIP (Contrastive Language-Image Pre-training)**: maps image content into the same latent space as language, enabling text-to-image generation (used in DALL-E).
- **Diffusion model**: generates images by learning to reverse a noise-adding process, guided by latent-space prompts.

## Mental Models
- Think of **algorithm vs. model** as process vs. artifact: the algorithm trains; the model is the trained result used for new predictions.
- Use **open-source vs. closed-source** as a lens on model weights (not training code) — open weights let you inspect/run locally; closed weights may protect competitive advantage but aren't always ahead in benchmarks.
- Think of **inference vs. training** as the key infrastructure-cost split: training needs massive, sustained parallel compute (GPUs, months); inference is comparatively lightweight per-call but still GPU-preferred for throughput.
- Use **quantization as a dial**: lower precision = less storage/compute but lower fidelity; tune per use case rather than assuming full precision is required.

## Anti-patterns
- **Assuming you need to train a foundational model from scratch**: training costs scale into the millions of dollars and thousands of GPUs (e.g., LLaMA-3 used 24,000+ GPUs); almost no team should attempt this — fine-tune or augment existing models instead.
- **Ignoring model licensing**: closed vs. open weight license terms affect whether a model can be used commercially; always check the model card/license tag before adoption.
- **Deploying base (non-instruction-tuned) models for chat/instruction use cases without fine-tuning or RLHF**: base models complete text patterns but do not reliably follow instructions or avoid offensive output.

## Code Examples
No code examples in this conceptual/background chapter (first hands-on infrastructure code starts in Chapter 2).

## Reference Tables
| Model | Training Parameters | Year Released |
|---|---|---|
| GPT-1 | 117 Million | 2018 |
| GPT-2 | 1.5 Billion | 2019 |
| GPT-3 | 175 Billion* | 2020 |
| GPT-3.5 | 1.3, 6.7, 175 Billion* | 2020 |
| GPT-4 | Undisclosed (rumored ~1.76 Trillion) | 2023 |
| GPT-4o | Undisclosed | 2024 |

*GPT-3 has variants from 125M to 175B parameters. (Table 1.1: Evolution of GPT models)

## Worked Example
**InstructGPT alignment walkthrough (as described for GPT 3.5/4):**
1. Start with a raw pretrained GPT model that completes text but ignores instructions (e.g., asked to "explain to a 6-year-old," it just restates the question).
2. Collect a small set of human-written ideal responses to representative prompts; supervised-fine-tune the base model on these.
3. Generate multiple candidate responses per prompt from the fine-tuned model; have humans rank them best-to-worst; use these rankings to train a separate reward model that learns to score response quality.
4. Generate new responses from the fine-tuned model and use the reward model's scores to further fine-tune (RL) toward higher-scored behavior.
Result: offensive/incorrect responses are greatly reduced (not eliminated) — hallucination still occurs because the model has no built-in fact-checking mechanism, only improved pattern-matching toward "good" responses.

## Key Takeaways
1. Deep learning's leap to GenAI came from combining transformer attention/context with massive scale (data + GPU hardware) — both pieces are necessary infrastructure considerations.
2. Inference is dramatically cheaper than training — plan infrastructure tiers differently for each (this motivates Ch2's Kubernetes scaling and Ch3-4's inference-focused hands-on work).
3. Quantization (precision reduction, e.g., 32-bit → 8-bit) is a key lever for running large models on constrained hardware — revisited heavily in Ch4.
4. Alignment (RLHF/InstructGPT-style fine-tuning) reduces but does not eliminate hallucination and offensive output — monitoring remains necessary in production (Ch9).
5. Software-defined storage, GPU/CPU/RAM planning, and high-throughput networking (e.g., InfiniBand) are foundational infrastructure needs for GenAI, driving the rest of the book's Kubernetes focus.

## Connects To
- **Ch2**: Kubernetes concepts (scaling, GPU scheduling, resiliency) directly address the infrastructure needs for AI/ML (GPUs, storage, network) introduced here.
- **Ch3-4**: Hands-on work with foundational models (GPT-2, diffusion models) builds directly on the model taxonomy and techniques (CLIP, diffusion, quantization) introduced in this chapter.
- **Ch9**: Alignment and hallucination concepts resurface in the context of continuous monitoring and safety evaluation.
