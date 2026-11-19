# Cheatsheet: AI-Ready Data Blueprints

## Decision: Which architectural pattern to build first?
- Document/knowledge retrieval need? → Semantic search + lakehouse (3–4 months to production)
- Cross-domain insights need? → Knowledge graph (4–6 months)
- Real-time decision support need? → Event-driven architecture (3–5 months)
- Unified analytics + AI need? → Lakehouse (4–6 months)
- External agent access need? → Agent-ready properties (6–9 months)
- Rule of thumb: start with ONE pattern matching your highest-value use case, prove value in 3–6 months, THEN expand. Never try to implement all five patterns simultaneously.

## Decision: Vector database selection (match ALL FOUR dimensions, not just one)
| If you have... | Choose |
|---|---|
| <10M vectors | Chroma, pgvector |
| 10M–1B vectors | Qdrant, Weaviate |
| 1B+ vectors | Milvus, Pinecone |
| High sensitivity (finance/health/gov) | Self-hosted on-prem: Milvus, Weaviate |
| Existing PostgreSQL investment | pgvector (extend, don't replace) |
| Existing MongoDB investment | Atlas Vector Search |
| Limited budget, engineering capacity available | Open source (Milvus/Qdrant/Weaviate) + build governance yourself |
| Minimal ops overhead priority | Managed service (Pinecone, OpenSearch Service, Zilliz Cloud) |
- Common two-tier pattern: Chroma/pgvector for dev+prototyping → Milvus/Pinecone for production.
- Red flag: choosing on raw benchmark speed alone — always check compliance certification (SOC 2, GDPR-ready isolation) before committing.

## Decision: Which index algorithm?
| Scale | Algorithm | Recall |
|---|---|---|
| <500M vectors | HNSW | 95–99% (default choice for most RAG) |
| 100M–1B vectors | IVF-Flat | 90–95% |
| 1B+ vectors | IVF-PQ | 85–90% (memory-constrained) |
| 10B+ vectors | DiskANN | 90–95% (disk-resident, minimal RAM) |
| GPU/TPU available | ScaNN | fast at any scale |

## Decision: Which chunking strategy?
| Content type | Strategy | Chunk size | Overlap |
|---|---|---|---|
| Legal contracts | Hierarchical | 512–1024 tok | 40–50% |
| Financial reports | Hierarchical | 512–1024 tok | 30–40% |
| Technical docs | Recursive | 256–512 tok | 20–30% |
| Medical/clinical | Semantic | 256–512 tok | 40–50% |
| Conversational logs | Sentence/turn-pair | 128–256 tok | 20–30% |
| FAQs | QA-pair (atomic) | 128–256 tok | n/a |
- Default heuristic: smaller chunks (64–128 tok) for fact lookups; larger (512–1024 tok) for broad-context tasks.
- Never default to fixed-size chunking for legal/medical/financial content — it fragments meaning.

## Decision: When to add a knowledge graph vs. stay with pure vector search?
- Stay with vector search if: question is "find similar things" (semantic similarity suffices).
- Add a knowledge graph (GraphRAG) if: question requires "why are these connected" or multihop traversal (e.g., regulation → product → business unit → audit finding) — vector embeddings cannot represent explicit relationship *type*, only proximity.

## Decision: Which RAG architecture tier?
Simple RAG (basic Q&A) → + memory (conversational) → Branched (route by source type) → Adaptive (adjust by query complexity) → Self-RAG (autonomous query generation) → Agentic RAG (full planning/reflection/tool-use).
- Rule: start at the simplest tier that solves the business problem. Even "basic" RAG delivered a 90% productivity increase in JPMorgan's KYC case — sophistication is not automatically better.

## Decision: Reasoning technique by task shape
- Simple task, no explanation needed → NO reasoning tokens / no CoT (pure overhead).
- Multistep/math/logical task → Chain-of-thought (CoT).
- Multiple competing approaches to evaluate → Tree-of-thought (ToT).
- Concepts interrelate non-linearly (e.g., medical diagnosis) → Graph-of-thought (GoT).
- Iterative drafting/refinement → Interleaved thinking.
- Self-similar hierarchical problem (org → team → individual) → Recursive reasoning.
- Clearly phased problem (scope → analyze → propose → plan) → Multistage reasoning.
- ALWAYS empirically test reasoning-token allocation on representative prompts: the book's benchmark showed 1,024 reasoning tokens roughly doubled both output tokens AND latency for an identical correct answer vs. zero tokens.

## Thresholds and defaults
- Context window math: ~200k tokens ≈ 500 pages of text.
- Hybrid search gain: 35%+ on broad QA benchmarks vs. single-method retrieval.
- Cross-encoder reranking gain: up to +28% NDCG@10 over baseline (at added latency cost).
- Chunking strategy impact: up to 35% retrieval accuracy swing (vs. only 6% from LLM swap, 27% from embedding model choice) — chunking is 5–6x more important than model selection.
- Metadata enrichment gain: up to 21% retrieval accuracy; precision 73%→83%.
- Semantic caching gain: 40–50% latency reduction, ~70% fewer redundant LLM calls.
- Vector DB production health targets: retrieval precision >80% (investigate <70%), recall >90% (review indexes <80%), freshness <1hr (refresh >24hr), index quality recall@10 >95% (rebuild <90%).
- Embedding dimensionality cost: doubling dimensions ≈ doubling storage (100M vectors: 768-dim ≈300GB vs. 3072-dim ≈1.2TB).
- Compression: scalar quantization (float32→int8) = 75% size reduction, minimal quality loss; binary quantization = 97% reduction, moderate loss; product quantization = 10–30x reduction.

## Tells and smells
- "We rolled back our GenAI deployment because of model/prompt issues" → usually a data-quality problem masquerading as a model problem (explicitly flagged pattern in Ch 1).
- Business glossary written as short human-friendly one-liners → will fail for AI agents; glossaries must be verbose and non-overlapping.
- Vector store performance degrading "silently" while queries still return results → embedding/index drift; unlike traditional DBs, vector DB quality decay is gradual and invisible without monitoring.
- Flattening structured records (CRM/ERP) into independent embedding chunks → destroys relational context; use a knowledge graph layer instead.
- Reaching for a bigger/better LLM when RAG underperforms → check chunking and data prep FIRST; the leverage ratio strongly favors data prep over model upgrades.
- A PII-redaction pipeline that relies only on prompt-based LLM detection → will miss SSNs, account numbers, and other structured sensitive formats; pair with dedicated DLP tooling (Macie, Comprehend, Bedrock Guardrails, Azure PII Detection, GCP DLP).
- Router/supervisor agent complexity growing unchecked as task variety increases → single point of failure risk; monitor and refactor before it degrades response quality.
- Switching embedding models without budgeting for full reindex → embeddings from different models are NOT compatible in the same index; this requires full corpus re-embedding (QDC or dual-search can soften the transition).

## HITL (Human-in-the-loop) — non-negotiable stages
- ALWAYS required: data labeling/annotation; evaluation and validation (factuality, safety, bias, ethics).
- Conditionally required: data collection/prep, model training/fine-tuning, deployment/integration (especially high-risk), monitoring/feedback review.
- Critical/high-impact inference → real-time human intervention capability required (EU AI Act and similar regulations mandate this).
