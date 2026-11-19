# Glossary

**Agentic AI** — AI systems designed to mimic autonomous-agent cognition: perceive their environment, process information, make decisions, execute actions, and learn/adapt. (Ch8)

**Agentic RAG** — Combines autonomous AI agents with RAG so agents can plan/execute retrieval strategies, use external tools, and reason over results adaptively rather than performing one static retrieval pass. (Ch8)

**ANN (Approximate Nearest Neighbor)** — Indexing technique trading exactness for speed when finding the most similar vectors at scale in a vector database. (Ch4)

**Attention mechanism** — Lets a model dynamically focus on the most relevant parts of input data, capturing long-range dependencies; the core innovation behind transformers. (Ch1, Ch3)

**BERT (Bidirectional Encoder Representations from Transformers)** — Encoder-only transformer model pre-trained on massive text corpora, used for embeddings and NLU tasks. (Ch1, Ch3)

**BLEU (Bilingual Evaluation Understudy)** — 0-1 metric evaluating machine-translation quality against reference translations (1 = perfect overlap). (Ch6)

**Chain (LangChain)** — Sequencing of LLM tasks/components (LLM chain, router chain, sequential chain, transformation chain) into a pipeline. (Ch5)

**Chain-of-Thought (CoT) prompting** — Prompting technique that guides an LLM to articulate step-by-step reasoning, improving multi-step task performance. (Ch6)

**Chroma DB** — Open-source vector store with collections (table-like groupings), built-in embedding (default `all-MiniLM-L6-v2`), and metadata filtering. (Ch4)

**Chunking (text chunking)** — Dividing text into syntactically/semantically meaningful segments before embedding; strategies include fixed-size, sentence-splitting, recursive, and hybrid. (Ch3)

**Cosine similarity** — Similarity metric (max score 1) measuring the angle between two vectors, independent of magnitude; dominant metric for NLP embedding comparison. (Ch3, Ch4)

**CUDA (Compute Unified Device Architecture)** — NVIDIA's parallel-computing architecture used to accelerate model inference/embedding on GPU. (Ch2)

**Devin** — Cognition Labs' autonomous AI software engineer (2024) that plans, codes, debugs, and deploys applications end-to-end in a sandboxed dev environment. (Ch8)

**Differential privacy** — Adding statistical noise to data/outputs to prevent inferring individual data points while preserving aggregate statistical properties. (Ch3, Ch8)

**DPO (Direct Preference Optimization)** — Directly optimizes a model's policy to satisfy human preferences via a binary cross-entropy objective, without a separate reward model or RL loop; simpler/more stable than RLHF. (Ch6)

**Embedding (text embedding)** — Mapping text to dense numeric vectors capturing semantic meaning, such that similar meanings produce nearby vectors. (Ch3)

**Euclidean distance (L2)** — Straight-line distance metric between two points in vector space; used by Faiss's `IndexFlatL2` for exact similarity search. (Ch3, Ch4)

**Faiss** — Facebook's open-source library for efficient similarity search/clustering of dense vectors, with in-RAM index structures (e.g., `IndexFlatL2`). (Ch2, Ch4)

**Few-shot prompting** — Providing n examples of desired input/output in the prompt to set tone/format expectations for the LLM. (Ch6)

**Fine-tuning** — Adjusting a pre-trained model's parameters using additional task/domain-specific training data. (Ch2, Ch6)

**Foundational model (FM)** — Large model pre-trained via self-supervised learning on massive unlabeled data, adaptable via fine-tuning, RAG, or prompting. (Ch2)

**GAN (Generative Adversarial Network)** — Generative architecture with a generator network and a discriminator network trained adversarially; introduced by Ian Goodfellow in 2014. (Ch1, Ch8)

**Hallucination** — An LLM generating factually incorrect or misleading text, including "legal hallucinations" with legal consequences. (Ch1, Ch10)

**Hugging Face Hub** — Open-source platform/community hosting pre-trained models, datasets, and demos; home of the `transformers` library. (Ch7)

**Instructor embeddings** — Instruction-tuned sentence embedding model family (e.g., `hkunlp/instructor-base`) accessed via `HuggingFaceInstructEmbeddings`. (Ch2, Ch7)

**KNN (K-Nearest Neighbors)** — Finds the k most similar data points to a query using a distance metric. (Ch4)

**LangChain** — Open-source, model-agnostic framework for composing LLMs, prompts, retrievers, and memory into chains for building gen AI applications. (Ch5)

**LCEL (LangChain Expression Language)** — Declarative way to compose LangChain chains, supporting streaming, parallelism, and logging; scales from prototype to production without code changes. (Ch5)

**LoRA (Low-Rank Adaptation)** — PEFT technique that freezes pretrained weights and injects trainable low-rank decomposition matrices into transformer layers, reducing trainable parameters up to 10,000x vs. full fine-tuning. (Ch6)

**MapReduceDocumentsChain** — LangChain chain that summarizes documents too large for one context window by mapping a summarization step over chunks, then reducing (combining) the results. (Ch5)

**MTEB (Massive Text Embedding Benchmark)** — Standard benchmark for comparing embedding models' performance across NLP tasks. (Ch3)

**PEFT (Parameter-Efficient Fine-Tuning)** — Family of fine-tuning approaches (LoRA, QLoRA, quantization) that update only a small subset of parameters instead of the full model. (Ch6)

**Perplexity** — Metric measuring how well a language model predicts the next word; lower is better. (Ch6)

**Prompt engineering** — Crafting input prompts (zero-shot, few-shot, CoT, ToT, explicit, positive/negative) to steer LLM output without modifying model parameters. (Ch2, Ch6)

**Quantization** — Converting model weights/activations from high precision (e.g., 32-bit float) to lower precision (e.g., 8-bit/4-bit integers) to reduce memory/compute footprint. (Ch6, Ch7, Ch9)

**QLoRA** — LoRA combined with 4-bit quantization of the base model, enabling fine-tuning of very large models (e.g., 65B parameters) on a single GPU. (Ch6)

**RAG (Retrieval-Augmented Generation)** — Pattern where a retrieval model fetches relevant external content and a generation model uses it as context to produce grounded, informative responses. (Ch2, Ch4)

**Retriever (LangChain)** — Component that fetches relevant documents from an external source given a query string, returning a list of `Document` objects. (Ch5)

**RLHF (Reinforcement Learning from Human Feedback)** — Trains a reward model from human-ranked LLM outputs, then uses RL to align the LLM toward helpful, less harmful responses. (Ch6)

**ROUGE** — Metric for evaluating the quality of machine-generated summaries. (Ch6)

**Self-attention** — Mechanism mapping a query and key-value pairs to a weighted-sum output, letting each token dynamically weight relevant other tokens regardless of position. (Ch1)

**Self-supervised learning** — Training technique that auto-generates labels from data itself, converting an unsupervised problem into a supervised one; used to pretrain foundational models. (Ch2)

**Text Generation Inference (TGI)** — Hugging Face's toolkit for high-performance serving of open-source LLMs (Llama, Falcon, StarCoder, BLOOM, GPT-NeoX). (Ch7)

**Tensor parallelism** — Splitting a model's tensors across multiple GPUs when the model is too large to fit on one GPU; configured via `SM_NUM_GPUS` in SageMaker deployments. (Ch7)

**Tree-of-Thought (ToT) prompting** — Prompting framework maintaining a tree of intermediate reasoning "thoughts," searched via BFS/DFS with lookahead/backtracking for complex problem-solving. (Ch6)

**Transformer architecture** — Encoder-decoder (or encoder-only/decoder-only) neural network using stacked self-attention and feed-forward layers, introduced in "Attention Is All You Need" (Vaswani et al., 2017). (Ch1)

**Vector database** — Database specialized for storing/retrieving/searching high-dimensional vector embeddings via similarity rather than exact match; categories include indexed, graph, spatial, document, and full-text search DBs. (Ch4)
