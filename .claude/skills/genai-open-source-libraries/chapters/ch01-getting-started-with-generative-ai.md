# Chapter 1: Getting Started with Generative AI

## Core Idea
Generative AI learns patterns in existing data (via architectures like GANs, VAEs, and especially Transformers with self-attention) to create new, original content, and the transformer's attention mechanism is the breakthrough that made today's large language models (LLMs) possible.

## Frameworks Introduced
- **Three-layer gen AI architecture**: data processing layer → generative model layer → feedback loop.
  - When to use: framing any generative system at a high level.
  - How: (1) collect/clean/prepare data, (2) generate new content via a model, (3) feed results back to improve performance.
- **Transformer architecture (Vaswani et al., "Attention Is All You Need," 2017)**: encoder-decoder structure using stacked self-attention and pointwise fully-connected layers; replaces sequential RNN processing.
  - When to use: any NLP task needing long-range dependency capture (translation, text generation).
  - How: encoder maps input symbols to continuous representations z; decoder auto-regressively generates output symbols one at a time, each step conditioned on previously generated symbols.
- **Self-attention / multi-head attention**: maps a query and key-value pairs to a weighted-sum output; multi-head splits input into parallel attention computations, combined into one output.
  - When to use: need the model to dynamically focus on relevant tokens regardless of position.
- **Positional encoding**: injects token-order information into embeddings since self-attention itself has no inherent sense of sequence order.

## Key Concepts
- **Generative AI (gen AI)**: AI subset that creates new content (text, images, music, code) rather than only analyzing existing data.
- **Foundational model (FM) precursor concept**: large pre-trained models like BERT and GPT that underlie downstream NLP tasks.
- **Attention mechanism**: lets a model dynamically weight relevant parts of input, capturing long-range dependencies that RNNs/LSTMs struggled with.
- **GAN (generative adversarial network)**: generator network vs. discriminator network trained adversarially.
- **VAE (variational autoencoder)**: learns a latent representation of data; new content sampled from latent space.
- **Hallucination**: LLM generates factually incorrect or misleading text due to training-data noise.
- **Encoder-only / decoder-only / encoder-decoder**: three architectural families (e.g., BERT is encoder-only, GPT is decoder-only, T5 is encoder-decoder).

## Mental Models
- Think of self-attention as a lookup: each token asks "which other tokens matter to me right now?" and weights them by compatibility (query·key), not by fixed position.
- Use the three-layer architecture (data → generate → feedback) as a quick checklist when scoping any new generative-AI project.
- Treat limitations (hallucination, bias, lack of transparency, cost) as a standing checklist to revisit before deploying any gen AI system, not a one-time disclaimer.

## Anti-patterns
- **Trusting LLM output as factually authoritative**: models can produce "plausible sounding but false information" (hallucination) — always verify in informational/educational contexts.
- **Ignoring training-data bias**: LLMs reflect biases present in their training corpora and can perpetuate harmful stereotypes if unchecked.
- **Assuming common-sense reasoning**: models can answer plausibly without true contextual understanding — don't rely on them for tasks needing grounded reasoning without verification.

## Code Examples
No code examples in this chapter (conceptual/foundational chapter — implementation begins in Ch02).

## Reference Tables
None in this chapter.

## Worked Example
The chapter walks conceptually (no code) through how GPT-3's family of models differs by parameter count, and contrasts GPT-4 (multimodal: text + image in/out) against earlier text-only LLMs, situating Claude, LaMDA, Jurassic-2, ERNIE 3.0, Llama, PaLM, BLOOM, and Bard as comparable LLMs with distinct strengths (e.g., LaMDA for creative formats, Jurassic-2 for factual text, Claude for reasoning on open-ended questions).

## Key Takeaways
1. Transformer self-attention solved RNN/LSTM's long-range-dependency and sequential-processing-speed problems.
2. Positional encoding and multi-head attention are the two mechanisms that let transformers model word order and multiple relationship types simultaneously.
3. Gen AI already has concrete applications across retail, customer service, manufacturing, healthcare, and finance — each domain pairs specific LLM capabilities (personalization, prediction, automation) with specific business problems.
4. Gen AI's core limitations — hallucination, bias, poor transparency (SHAP infeasible at LLM scale), compute cost, legal/ethical risk — must be weighed before any production use.

## Connects To
- **Ch2**: Foundational models (BERT, GPT, etc.) introduced here conceptually are formally defined, implemented, and customized in Chapter 2.
- **Ch3-4**: The NLP/transformer foundations here underpin the text-processing, embeddings, and vector-database techniques used to operationalize LLMs later.
- **Ch10**: The limitations (bias, hallucination, ethics) flagged here are revisited as responsible-AI concerns in the closing chapter.
