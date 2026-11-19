# Chapter 1: Introducing New Age Generative AI

## Core Idea
Modern GenAI systems are not just generators — they are stacks combining retrieval (vector databases, dense/sparse/hybrid search), reranking, guardrails, and agentic reasoning (tools, planning, action, memory), all glued together by orchestration frameworks and standards like MCP.

## Frameworks Introduced
- **Retrieval vs. Generation systems**: Retrieval finds existing relevant info (grounding); generation creates new content (text/image/audio). Combined, they form RAG.
  - When to use retrieval: factual grounding, up-to-date knowledge, reducing hallucination.
  - When to use generation: open-ended creativity, reasoning beyond existing data, personalization.
- **RAG (Retrieval-Augmented Generation)**: Two-step pipeline — (1) retrieve top-k relevant chunks from a vector DB, (2) generate an answer conditioned on retrieved context.
  - How: query understanding → retrieval (dense/hybrid) → context preparation (chunk/truncate/format) → generation → output delivery.
- **Single-stage vs. Two-stage RAG**: Single-stage = retrieve→generate (low latency, needs high-quality retrieval). Two-stage = retrieve→rerank→generate (reduces hallucination by filtering before generation).
  - When to use two-stage: when raw retrieval is noisy and generation quality is sensitive to irrelevant context.
- **Bi-encoders vs. Cross-encoders**: Bi-encoders embed query and document independently (fast, scalable, used for first-pass retrieval via cosine similarity/dot product). Cross-encoders jointly encode query+document with cross-attention (slow, but far more accurate) — used for reranking.
  - How: retrieve top-k (e.g., 100) with bi-encoder → rerank with cross-encoder → keep top-n (e.g., 5-10) for generation.
- **Agentic orchestration**: Agent = reasons, chooses actions (tools/APIs/retrieval), executes, reflects, adjusts. Orchestration adds tool selection, multi-step planning, memory management, error handling, goal checking.
- **Model Context Protocol (MCP)**: Anthropic's open, language-agnostic client-server standard ("USB-C for AI") exposing three primitives — tools (actions), resources (data), prompts (behavior guidance) — over JSON-RPC (stdio/HTTP/SSE transports).
  - When to use: standardizing how agents discover/call external tools and data without custom per-backend integration.

## Key Concepts
- **Tokenization**: Breaking text into discrete units (word/subword-BPE/character/byte-level) mapped to token IDs, then embedded into high-dimensional vectors; determines cost, context window, and truncation behavior.
- **Vector database**: Stores dense embeddings and performs similarity search (cosine/Euclidean) rather than exact match; examples — Faiss, Milvus, Qdrant, Pinecone, Weaviate, Chroma, Azure AI Search.
- **ANN (Approximate Nearest Neighbor) search**: Trades exactness for speed at scale; techniques — HNSW (graph-based), IVF (clustering), PQ (compression), IVF+PQ, ScaNN.
- **Embedding models**: Text (OpenAI ada-002, SBERT, MiniLM), Image (CLIP, DINOv2), Multimodal (CLIP, Flamingo).
- **Reranking**: Second-pass reordering of retrieved candidates using deeper relevance scoring (typically cross-encoders) to maximize top-k precision.
- **Guardrails**: Structured controls (input + output) that constrain model behavior for safety/alignment — e.g., NVIDIA NeMo Guardrails, Azure AI Prompt Shields, OpenAI Moderation API.
- **Agentic RAG vs. non-agentic RAG**: Non-agentic = fixed linear pipeline (embed→retrieve→generate). Agentic = dynamic — agent assesses sufficiency, reformulates queries, multi-hop retrieves, reflects, and adapts.
- **Orchestration frameworks**: LangChain, LlamaIndex, Haystack — automate query understanding, retrieval, context prep, prompt construction, generation, post-processing.

## Mental Models
- Think of retrieval as the "grounding engine" and generation as the "creativity engine" — production systems need both, not one or the other.
- Use the bi-encoder→cross-encoder pipeline as the universal retrieval-quality lever: fast-and-broad first, slow-and-precise second.
- Treat guardrails as a two-gate system (input gate before the model, output gate after) — reranking alone never guarantees safety or alignment.
- MCP is the "USB-C" mental model: one standardized plug (tools/resources/prompts) instead of N custom integrations per tool/data source.

## Anti-patterns
- **Relying on pure vector similarity as final relevance**: top results by cosine similarity can be only loosely relevant to true user intent — always consider reranking before generation.
- **Treating guardrails as optional or an afterthought**: non-deterministic LLMs can produce unsafe/biased output even with clean, well-retrieved input; input-only or output-only filtering is insufficient — need both.
- **Using exhaustive exact search at scale**: brute-force flat search is accurate but computationally infeasible beyond small datasets; use ANN (HNSW/IVF/PQ) for production scale.
- **Using cross-encoders for first-pass retrieval over large corpora**: joint encoding of every query-document pair is too slow; reserve cross-encoders for reranking a small candidate set.

## Code Examples
No executable code in this chapter — it is conceptual/architectural. The tokenization example given is illustrative:
```text
Input: "Today is a beautiful day outside."
Tokens (subword): To | day | is | a | be | aut | iful | day | out | side
Token IDs: To->98, day->1452, beautiful->(2932,1709) ...
Token IDs -> Embedding layer -> high-dimensional vectors (e.g., 768-d) -> attention/transformer layers
```
- **What it demonstrates**: The text-to-token-ID-to-embedding pipeline that underlies all downstream retrieval and generation.

## Reference Tables
| Year | Milestone | Description |
|---|---|---|
| 1970s-2000s | TF-IDF, BM25 | Early keyword-based (sparse) retrieval |
| 2020 | Dense Passage Retrieval (DPR) | Dense embeddings for semantic query-doc matching |
| 2021 | Hybrid retrieval | Combines sparse (BM25) + dense (DPR) |
| 2020-2022 | RAG | Tight retrieval-generation integration |
| 2023+ | In-context/memory-augmented retrieval | Dynamic, reasoning-driven retrieval inside LLM workflows |

| Indexing method | Accuracy | Speed | Notes |
|---|---|---|---|
| Flat (brute-force) | Highest | Slowest | Not scalable |
| IVF | Good | Fast | Clusters, searches relevant clusters only |
| HNSW | High recall | Low latency | Graph-based, real-time friendly |
| PQ | Reasonable | Very fast | Compressed vectors, very large datasets |

| Architecture | Speed | Accuracy | Use |
|---|---|---|---|
| Bi-encoder | Fast | Lower | First-pass retrieval at scale |
| Cross-encoder | Slow | Higher | Reranking small candidate sets |

## Worked Example
A two-stage RAG + guardrail pipeline: user query → input guardrail check → bi-encoder retrieves top-100 candidates from a vector DB (e.g., Faiss) → cross-encoder reranks to top-5/10 → LLM generates answer conditioned on reranked context → output guardrail (moderation API) validates/filters the response before returning to the user. This composite architecture (Figure 1.11 in the book) is the reference blueprint reused and extended throughout later chapters (reranking in Ch9, guardrail-adjacent HITL in Ch5, multimodal RAG in Ch8).

## Key Takeaways
1. Retrieval and generation are complementary, not competing — most production-grade systems are RAG systems at their core.
2. Two-stage retrieval (bi-encoder retrieve + cross-encoder rerank) is the standard pattern for balancing speed and precision.
3. Vector databases and ANN indexing (HNSW/IVF/PQ) are the infrastructure backbone enabling retrieval to scale to billions of embeddings.
4. Guardrails must operate on both input and output; reranking and retrieval quality do not substitute for safety controls.
5. Agentic RAG adds dynamic reformulation, multi-hop retrieval, and reflection on top of static non-agentic RAG, needed for ambiguous/complex queries.
6. MCP standardizes tool/resource/prompt access for agents, replacing ad hoc per-tool integration code.

## Connects To
- **Ch 2**: This chapter's token/embedding and retrieval foundation extends directly into multimodal embeddings (text+image+audio in a shared space).
- **Ch 5**: The agent/tools/reasoning/planning/action primitives introduced here are implemented in full as HITL agentic systems.
- **Ch 8-10**: The RAG architecture, reranking, and bi-/cross-encoder framing here are the direct precursor to multimodal RAG, dedicated reranking systems, and retrieval optimization.
- **Ch 14-15**: MCP and agentic orchestration ideas reappear in agentic text-to-SQL architecture decision-making.
