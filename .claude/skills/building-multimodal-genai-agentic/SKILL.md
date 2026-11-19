---
name: building-multimodal-genai-agentic
description: "Knowledge base from \"Building Multimodal Generative AI and Agentic Applications\" by Indrajit Kar (BPB Publications, 2026). Use when building multimodal generative AI systems — local/API-based unimodal systems, agentic GenAI with human-in-the-loop, multi-stage pipelines, multimodal retrieval/RAG/reranking, voice input, advanced multimodal systems, text-to-SQL (including agentic), OCR/text-extraction from images, integrating traditional AI/ML, or LLMOps and GenAI evaluation."
---

# Building Multimodal Generative AI and Agentic Applications
**Author**: Indrajit Kar | **Chapters**: 18 | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a technique or architecture; I find and read the relevant chapter
- **With a chapter** — ask for `ch08`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Foundations (Ch1-2)**: Modern GenAI = Retrieval (vector DBs, bi-/cross-encoders, ANN indexing) + Generation + Guardrails + Agentic reasoning (tools, planning, memory), orchestrated by frameworks like LangChain/LangGraph and standardized via MCP. VLMs (unified vision-text models) differ from Multimodal GenAI systems (modular, retrieval-augmented pipelines spanning any modality) — pick VLM for tightly-coupled vision-text tasks needing low latency; pick multimodal GenAI systems for tool use, retrieval augmentation, and multi-modality beyond vision+text.

**Local vs. API-based unimodal systems (Ch3-4)**: Local (Ollama + Chroma/Faiss + LangChain) gives privacy/cost-control at the expense of model power; API-based (OpenAI) unlocks stronger models and an agentic ecosystem (Responses API, Agents SDK, Operator, Codex) at the expense of data locality. The Seven RAG Failure Points (Missing content, Missed top-ranked documents, Not in context, Not extracted, Wrong format, Incorrect specificity, Incomplete answers) is the standard diagnostic checklist for any underperforming RAG pipeline.

**Agentic/HITL design (Ch5)**: 19 multi-agent design patterns (Parallel, Sequential, Loop, Router, Aggregator, Network, Hierarchical, HITL, Shared tools, Database-with-tools, Memory-transformation, Planner-executor, Critic/validator, Negotiator, Multimodal-agent, Voting/consensus, Supervisor-subordinate, Watchdog/recovery, Temporal-planner) are composable building blocks — real systems combine several. HITL = halt after generation → human approves/rejects → bounded retry (e.g., max 3) → graceful fallback. Agentic AI (multi-agent, planning, persistent memory, adaptive learning) ≠ AI agents (narrow, reactive, single-step).

**Retrieval/RAG/reranking/optimization stack (Ch6-10)**: Interaction depth is the central lever — No interaction (bi-encoder, fast/coarse, first-stage retrieval) → Late interaction (ColBERT/MaxSim, scalable token-level precision) → Full interaction (cross-encoder, most accurate, reserved for reranking ≤100 candidates). Multi-stage RAG variants (Simple, +memory, Branched, HyDE, Adaptive, Corrective/CRAG, Self-RAG, Agentic) each patch one weakness of baseline two-stage RAG. Grading mechanisms (retrieval relevance, hallucination, answer-quality graders) must use a *different* LLM than the generator to avoid self-evaluation bias. Six named retrieval drawbacks (recall/precision trade-off, limited semantic understanding, modality mismatch, index staleness, ranking inefficiency, lack of contextual awareness) each map to a specific mitigation (query expansion, modality routing, embedding normalization, hybrid retrieval + score fusion, cross-encoder rerank + prefiltering, adaptive index refresh, contextual filtering).

**Multimodal retrieval & generation (Ch7-9, 12-13)**: Bidirectional multimodal retrieval uses a shared bi-encoder (CLIP) + dual-index vector stores (per modality) + L2-normalized vectors. Multimodal RAG adds a generator module (separate LLM) consuming retrieved multimodal context. Eleven reasoning types (deductive, inductive, abductive, analogical, commonsense, causal, spatial, temporal, mathematical, tool-based/ReAct, multimodal-fusion) are a toolbox — match reasoning type to problem shape. Advanced prompting (ToT, GoT, self-consistency, CoS, scratchpad, typed CoT, generate-then-select, multi-agent debate, Auto-CoT) all specialize CoT with more structure/branching/redundancy. CoT reasoning can be injected directly into reranking as a fused scoring signal alongside vector similarity.

**Voice & advanced multimodal (Ch11)**: Voice is a thin STT/TTS adapter bracketing an unchanged text-based RAG core — don't build parallel voice-specific pipelines. LLM-based query routing (local vectorstore vs. web search fallback) needs fault-tolerant JSON parsing to be production-safe.

**Text-to-SQL and agentic SQL (Ch14-15)**: Hardest problems are schema linking and domain generalization, not SQL syntax. Production systems need the full pipeline: prompting → schema integration → validation (static/schema-aware/logical) → fallback/retry cascade → governance → observability. Agentic text-to-SQL uses ONE ReAct agent for flexible schema interpretation + plain deterministic functions for SQL generation/grading (don't over-agentify every stage). Execution accuracy is the right production metric; exact-match is for benchmarks only.

**OCR, traditional-ML integration, LLMOps (Ch16-18)**: Three OCR approaches trade modularity against integration (wrapped OCR+LLM > OCR foundation model > fine-tuned multimodal LLM, in modularity order). Wrap existing validated ML models (classifiers, CNNs, forecasters) as LangChain tools rather than retraining LLMs to replicate them — LLM does reasoning/orchestration/explanation, legacy model does prediction. RAGOps = identification (per-stage failure points) + benchmarking (dev-time, version-tied) + continuous monitoring (post-deployment) + dual observability (span-based tracing for process, metric-based LLM-as-judge for outcome).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-introducing-new-age-generative-ai.md) | Introducing New Age Generative AI | Retrieval vs. generation, RAG, vector DBs, reranking, bi-/cross-encoders, guardrails, agents, MCP |
| [ch02](chapters/ch02-deep-dive-into-multimodal-systems.md) | Deep Dive into Multimodal Systems | VLM categories, cross/bi-encoder VLMs, multimodal RAG, Qdrant concepts, output-type classification |
| [ch03](chapters/ch03-implementing-unimodal-local-genai-system.md) | Implementing Unimodal Local GenAI System | GPU decision framework, Ollama, seven RAG failure points, hybrid retrieval |
| [ch04](chapters/ch04-implementing-unimodal-api-based-genai-systems.md) | Implementing Unimodal API-based GenAI Systems | OpenAI API/models, agentic ecosystem (Responses API, Agents SDK), multi-document RAG |
| [ch05](chapters/ch05-implementing-agentic-genai-systems-with-hitl.md) | Implementing Agentic GenAI Systems with HITL | 19 multi-agent design patterns, HITL, LangGraph StateGraph, agentic AI vs. AI agents |
| [ch06](chapters/ch06-two-and-multi-stage-genai-systems.md) | Two and Multi-stage GenAI Systems | Interaction taxonomy, multi-vector representations, multi-stage RAG variants, grading mechanisms |
| [ch07](chapters/ch07-building-a-bidirectional-multimodal-retrieval-system.md) | Building a Bidirectional Multimodal Retrieval System | CLIP bi-encoder, dual-index Qdrant, L2 normalization |
| [ch08](chapters/ch08-building-a-multimodal-rag-system.md) | Building a Multimodal RAG System | Generator module, dual-LLM grading, LLM-as-a-judge, multimodal recommenders |
| [ch09](chapters/ch09-building-genai-systems-with-reranking.md) | Building GenAI Systems with Reranking | 5 reranker categories, multimodal cross-encoder reranking, multi-index embedding |
| [ch10](chapters/ch10-retrieval-optimization-for-multimodal-genai.md) | Retrieval Optimization for Multimodal GenAI | 6 drawbacks→mitigations, prefiltering, score normalization, GA optimization, adaptive refresh |
| [ch11](chapters/ch11-building-multimodal-genai-systems-with-voice-as-input.md) | Building Multimodal GenAI Systems with Voice as Input | STT/TTS, LangGraph hybrid routing, grounded-generation prompting |
| [ch12](chapters/ch12-advanced-multimodal-genai-systems.md) | Advanced Multimodal GenAI Systems | 11 reasoning types, ReAct, reasoning benchmarks |
| [ch13](chapters/ch13-advanced-multimodal-genai-systems-implementation.md) | Advanced Multimodal GenAI Systems Implementation | ToT/GoT/self-consistency/CoS, CoT-based reranking, recommendation-stage reasoning |
| [ch14](chapters/ch14-building-text-to-sql-systems.md) | Building Text-to-SQL Systems | Schema linking, 13-stage pipeline, fallback/retry, text-to-SQL metrics |
| [ch15](chapters/ch15-agentic-text-to-sql-systems-and-architecture-decision-making.md) | Agentic Text-to-SQL Systems and Architecture Decision-Making | 14-step agentic pipeline, single-agent/multi-tool design, federated query execution |
| [ch16](chapters/ch16-genai-for-extracting-text-from-images.md) | GenAI for Extracting Text from Images | 3 OCR approaches, Mistral OCR stack, multimodal receipt extraction |
| [ch17](chapters/ch17-integrating-traditional-ai-ml-into-genai-workflow.md) | Integrating Traditional AI/ML into GenAI Workflow | ML-model-as-tool pattern, LLM's 4 roles, imbalanced-classification evaluation |
| [ch18](chapters/ch18-llm-operations-and-genai-evaluation-techniques.md) | LLM Operations and GenAI Evaluation Techniques | LLM vs. RAG evaluation, RAGOps lifecycle, continuous monitoring, MLflow instrumentation |

## Topic Index
- **Agentic design patterns** → Ch5, Ch15
- **Bi-encoder / cross-encoder** → Ch1, Ch2, Ch6, Ch9
- **CoT / advanced prompting** → Ch1, Ch12, Ch13
- **Embeddings & vector databases** → Ch1, Ch2, Ch7, Ch10
- **Fallback/retry mechanisms** → Ch3, Ch11, Ch14
- **Grading / LLM-as-judge** → Ch6, Ch8, Ch15, Ch18
- **Guardrails** → Ch1
- **HITL** → Ch5
- **Hybrid retrieval (BM25+vector)** → Ch3, Ch9, Ch10
- **LangGraph / StateGraph** → Ch5, Ch11, Ch13, Ch15
- **LLMOps / RAGOps / observability** → Ch18
- **Local vs. API-based LLMs** → Ch3, Ch4
- **MCP (Model Context Protocol)** → Ch1
- **Multimodal RAG** → Ch2, Ch7, Ch8, Ch10
- **OCR** → Ch16, Ch17
- **Reasoning types** → Ch12, Ch13
- **Reranking** → Ch1, Ch6, Ch9, Ch10, Ch13
- **Retrieval optimization** → Ch10
- **Text-to-SQL** → Ch2, Ch14, Ch15
- **Traditional ML + GenAI integration** → Ch17
- **Voice / STT / TTS** → Ch11

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — architecture/implementation patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 18 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive, as are any recurring publisher promotional asides. For topics beyond this book's scope, check other skills or ask directly.
