# Chapter 2: Types of Foundation Models

## Core Idea
Foundation models (FMs) are built on a small set of common architectures (transformer, mixture of experts, GAN) and go through a defined pre-training lifecycle (data collection → curation → tokenization → training → evaluation); selecting the right FM for a financial-services use-case means filtering by attributes — modality, context window, licensing, parameter count, and language support.

## Frameworks Introduced
- **Transformer architecture (encoder-decoder + attention)**: six-layer encoder (multi-head attention + feed-forward sub-layers) paired with a six-layer decoder (adds masked multi-head attention over encoder output).
  - When to use: the default architecture behind nearly all modern text FMs.
- **Mixture of Experts (MoE)**: replaces each feed-forward network (FFN) layer in a transformer with an MoE layer (a router + sparse group of FFNs); router sends each token to one or more specialized experts.
  - When to use: need faster inference/efficiency — only specialized experts activate per token.
- **Generative Adversarial Network (GAN)**: generator vs. discriminator network compete; generator tries to fool the discriminator, discriminator tries to catch fakes; both improve over iterations until discriminator can no longer tell real from fake.
  - When to use: image/audio/video synthetic data generation (less central to this book's text-heavy financial use-cases).
- **Foundation Model Training Lifecycle**: Data collection → Data curation → Tokenization → Training → Evaluation (iterative; failing evaluation sends you back to an earlier step).
  - How: financial organizations must additionally run legal/compliance approval on public datasets and apply strict bias/toxicity/skew quality gates during curation and evaluation to satisfy auditors/model-risk/compliance.

## Key Concepts
- **Token**: integer representation of input data; roughly 0.75 of a word on average.
- **Self-supervised learning**: FM training approach where the model generates its own labels from input data (distinct from supervised/unsupervised predictive ML).
- **Modality**: input/output data type a model handles — text, code, image, audio, video, embedding, or multimodal.
- **Context window**: total input + output tokens a model can process per request (e.g., GPT-4o 128K, Gemini-1.5-flash 1M, Claude 3.5 200K).
- **Licensing categories**: Permissive (e.g., Apache 2.0, Falcon-180b, Mistral) / Restricted commercial use (e.g., Llama 3) / Proprietary (e.g., GPT-4, Gemini, Titan, Claude, Jurassic, Cohere — closed weights, EULA + payment required).
- **Embedding**: numerical (vector) representation of tokens used for semantic search/retrieval.
- **Number of parameters**: not a reliable sole proxy for quality — smaller models (Mistral 7B, Falcon-180B) can match or beat larger ones on benchmarks.

## Mental Models
- Use the licensing lens first for financial services: "Can legal/compliance approve this model's weights, training data provenance, and EULA?" before any technical evaluation.
- Treat parameter count as one signal among several, not the deciding one — benchmark against actual task performance, not model size.
- Think of context window as a hard capacity constraint: if your document (e.g., a financial report) doesn't fit, the model is disqualified regardless of other attributes.

## Anti-patterns
- None explicitly named here (Ch6 covers anti-patterns); implicit caution: assuming "bigger model = better" is called out as a myth.

## Code Examples
No executable code in this chapter — it is conceptual/attribute-based model selection guidance. (Environment setup and code begin in Ch3.)

## Reference Tables
**Table 2.1: Foundation Models' Context Window**

| Provider | Model | Modality | Context window |
|---|---|---|---|
| OpenAI | GPT-4o | Multimodal | 128K |
| Google | Gemini-1.5-flash | Multimodal | 1 million |
| Anthropic | Claude 3.5 | Multimodal | 200K |
| Meta | Llama 3.2 | Text | Up to 128K |
| Mistral | Mistral large | Text | 128K |
| AI21Labs | Jurassic-2 ultra | Text | 8191 |
| Cohere | Command R | Text | 128K |

**Table 2.2: Foundation Model Language Support** — GPT-4, Gemini, Claude 2/3/3.5, Titan, Command/Embed, Llama 3.2, Mistral-large, Jurassic-2 ultra are all multilingual; Falcon-180b has "limited proficiency." Note: Common Crawl is ~60% English, so non-English representation remains a gap relevant to globally operating financial institutions.

**Popular datasets used for FM pre-training**: Common Crawl (petabytes, web since 2008), C4 (~750GB cleaned Common Crawl), GitHub (>1TB code), Wikipedia (~21GB English), Books, Stack Exchange (via EleutherAI).

**Evaluation dataset categories**: Commonsense reasoning (PIQA, SIQA, HellaSwag), World knowledge (Natural Questions, TriviaQA), Code (HumanEval, MBPP), Reading comprehension (SQuAD, QuAC, BoolQ), Math (GSM8K, MATH).

## Worked Example
Choosing between two candidate FMs for a financial report summarization use-case: Model A has more parameters but an 8K context window (Jurassic-2 ultra-like); Model B has fewer parameters but a 128K+ context window. If the business reports routinely exceed 8K tokens, Model A is disqualified outright on the context-window attribute alone — illustrating why attribute filtering (not raw parameter count) should drive the shortlist before any benchmarking.

## Key Takeaways
1. Filter FM candidates by modality, context window, licensing, and language support before deep technical evaluation — licensing is the single most important gate for regulated financial organizations.
2. Open-source/permissive-license models with available weights are needed for in-house fine-tuning; proprietary closed-weight models restrict you to provider-hosted fine-tuning offerings.
3. Model size is not a reliable proxy for quality — benchmark on your actual task.
4. Code models require extra scrutiny of EULA terms for copyright/licensed-code contamination risk, a financial-services-specific compliance concern.
5. Data curation and evaluation stages must include explicit bias/toxicity/skew quality gates to satisfy financial auditors and model-risk management.

## Connects To
- **Ch1**: Elaborates the "choose and evaluate FM" step of the generative AI lifecycle introduced there.
- **Ch3**: Uses the attributes here (especially licensing/weight access) to determine which customization approaches (fine-tuning vs. prompt engineering/RAG) are even feasible for a given model.
- **Ch12**: Revisits model/approach selection trade-offs in a decision-framework form.
