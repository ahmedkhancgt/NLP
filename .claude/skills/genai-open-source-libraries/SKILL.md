---
name: genai-open-source-libraries
description: "Knowledge base from \"Building Generative AI Applications with Open-source Libraries\" by Srikannan Balakrishnan (BPB Publications, 2025). Use when implementing LLM applications with open-source tools — foundational models, text processing and embeddings, vector databases, LangChain, Hugging Face, or deploying generative AI applications responsibly."
---

# Building Generative AI Applications with Open-source Libraries
**Author**: Srikannan Balakrishnan | **Chapters**: 10 | **Generated**: 2026-11-18

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or library; I find and read the relevant chapter
- **With a chapter** — ask for `ch05`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The implementation stack (Ch1-10 in order)**: foundations (transformers/attention, foundational models) → text/embeddings/vector-DB fundamentals → LangChain orchestration → LLM implementation (training, PEFT, prompting) + Hugging Face → developments/deployment/responsible use. Every later chapter's code builds on the same recurring pattern: **chunk → embed → index in a vector store → retrieve → generate (RAG)**.

**RAG is the backbone pattern**: (1) create embeddings from a document corpus, (2) store as indexes in a vector database (Faiss/Chroma/Pinecone/etc.), (3) on query, retrieve top-k by similarity and inject as LLM context. This single pattern reappears in Ch2 (Falcon+PDF), Ch4 (vector DB theory), Ch5 (LangChain `RetrievalQA`), Ch6 (Llama 2 & Falcon with config-driven pipelines), and Ch7 (Hugging Face Hub framing).

**Three ways to adapt an LLM** (Ch2, Ch6): fine-tuning (parameter updates, needs labeled data + compute), prompt engineering (no retraining, fastest/cheapest), RAG (grounds in external knowledge). Pick by what you have: data+compute → fine-tune (prefer PEFT: LoRA/QLoRA over full fine-tuning); need current/external facts → RAG; neither → prompt engineer.

**Chunking → Embedding → Vector Store pipeline** (Ch3, Ch4): text must be chunked (fixed-size / sentence-splitting / recursive / hybrid) before embedding, because LLM embeddings lose quality and efficiency on unchunked long text. Embeddings (BERT, Instructor, sentence-transformers, or paid APIs) map text to dense vectors; vector databases (Faiss for lightweight/in-process, Chroma for metadata+persistence, Pinecone/Weaviate/Milvus for managed scale) index them for similarity search (cosine similarity is the NLP default metric).

**LangChain as the orchestration layer** (Ch5): composes prompt + model + output parser declaratively via LCEL; `RetrievalQA` chains wrap a vector store retriever + LLM + prompt template for RAG; `MapReduceDocumentsChain` handles documents exceeding context windows by mapping a summarization step per-chunk then reducing results. LangChain is model-agnostic — the same chain code works across OpenAI, Hugging Face, Anthropic, Cohere, Bedrock.

**PEFT over full fine-tuning** (Ch6): LoRA freezes the base model and trains small low-rank adapter matrices (up to 10,000x fewer trainable parameters, 3x less GPU memory than full fine-tuning); QLoRA adds 4-bit quantization on top, enabling 65B-parameter fine-tuning on a single 48GB GPU. DPO is a simpler, more stable alternative to RLHF for preference alignment (no separate reward model/RL loop needed).

**Prompt engineering technique selection** (Ch6): match technique to task shape — explicit/few-shot for format control, chain-of-thought (CoT) for multi-step reasoning, self-consistency sampling for higher-stakes reasoning (sample multiple CoT paths, pick most frequent answer), tree-of-thought (ToT) for exploration/backtracking tasks.

**Hugging Face as the open-source hub** (Ch7): `transformers` library + Hub (200,000+ models) + TGI (Text Generation Inference) for serving + managed cloud deployment (AWS SageMaker/Azure ML/Vertex AI). Tensor parallelism (`SM_NUM_GPUS`) splits large models across GPUs; `HF_MODEL_QUANTIZE` serves a quantized version for lower memory cost.

**Deployment reality** (Ch9): LLM deployment needs infrastructure (GPU/TPU/CPU/RAM sizing), storage (model parameters + training data, quantization reduces footprint), and a process (serialize → containerize with Docker → expose via REST API → deploy → scale horizontally/vertically). AWS/Azure/GCP offer functionally similar pipelines under different service names (SageMaker/AML/Vertex AI).

**Responsible AI is not optional** (Ch1, Ch8, Ch10): bias enters through training data, model architecture, and optimization process — real studies (Midjourney, Stable Diffusion, DALL-E 2) show consistent gender-representation bias across commercial and open-source models alike. Mitigation requires data-centric + model-centric + human-oversight techniques combined, plus a responsible-AI framework spanning transparency, accountability, legal compliance, and values alignment. Data privacy requires anonymization, differential privacy, access control, and regulatory compliance (GDPR/HIPAA) at every stage from preprocessing to deployment.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-getting-started-with-generative-ai.md) | Getting Started with Generative AI | Transformer architecture, self-attention, GANs/VAEs, gen AI limitations |
| [ch02](chapters/ch02-overview-of-foundational-models.md) | Overview of Foundational Models | FM customization (RAG vs. prompt engineering vs. fine-tuning), first chat-with-PDF app (Falcon-7B) |
| [ch03](chapters/ch03-text-processing-and-embeddings-fundamentals.md) | Text Processing and Embeddings Fundamentals | Chunking strategies, embedding methods (TF-IDF, word2vec, BERT, USE), secure embedding |
| [ch04](chapters/ch04-understanding-vector-databases.md) | Understanding Vector Databases | Vector DB taxonomy, RAG architecture, Faiss & Chroma implementation, similarity metrics |
| [ch05](chapters/ch05-exploring-langchain-for-generative-ai.md) | Exploring LangChain for Generative AI | LCEL, RetrievalQA chains, Map-Reduce summarization, Streamlit integration |
| [ch06](chapters/ch06-implementation-of-llms.md) | Implementation of LLMs | Training/RLHF/DPO, prompt engineering catalog, PEFT (LoRA/QLoRA), LLM evaluation |
| [ch07](chapters/ch07-implementation-using-hugging-face.md) | Implementation Using Hugging Face | Hugging Face Hub/transformers/TGI, SageMaker deployment, Falcon-7B RAG app |
| [ch08](chapters/ch08-developments-in-generative-ai.md) | Developments in Generative AI | Agentic AI architecture, Agentic RAG, Devin, data privacy |
| [ch09](chapters/ch09-deployment-of-applications.md) | Deployment of Applications | Infrastructure/storage/deployment challenges, cloud provider comparison, scaling |
| [ch10](chapters/ch10-generative-ai-for-good.md) | Generative AI for Good | Bias sources & mitigation, ethical use principles, positive-impact use cases |

## Topic Index
- **Agentic AI / Agentic RAG** → Ch8
- **Attention mechanism / Transformers** → Ch1
- **Bias & fairness mitigation** → Ch10 (also Ch1, Ch8)
- **Chunking strategies** → Ch3
- **Cloud deployment (AWS/Azure/GCP)** → Ch9 (also Ch7)
- **Devin / autonomous coding agents** → Ch8
- **Embeddings (methods & selection)** → Ch3
- **Falcon / Llama implementation** → Ch2, Ch6, Ch7
- **Fine-tuning** → Ch2, Ch6
- **Foundational models** → Ch2
- **Hugging Face Hub / transformers / TGI** → Ch7
- **LangChain (LCEL, chains, retrievers)** → Ch5
- **LoRA / QLoRA / PEFT / quantization** → Ch6 (also Ch7, Ch9)
- **Map-Reduce summarization** → Ch5
- **Mistral 7B** → Ch5, Ch6
- **Prompt engineering (CoT, ToT, few-shot, etc.)** → Ch6
- **RAG (Retrieval-Augmented Generation)** → Ch2, Ch4, Ch5, Ch6, Ch7
- **RLHF / DPO** → Ch6
- **Vector databases (Faiss, Chroma, Pinecone, etc.)** → Ch4
- **Data privacy / GDPR / anonymization** → Ch3, Ch8, Ch10
- **LLM evaluation (BLEU, ROUGE, perplexity, benchmarks)** → Ch6

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — implementation patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 10 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive, as are any recurring publisher promotional asides. For topics beyond this book's scope, check other skills or ask directly.
