# Chapter 5: Knowledge Bases and Vector Databases

## Core Idea
Knowledge bases, vector databases, and RAG are one tightly integrated retrieval architecture rather than three separate technologies, and data preparation quality (especially chunking strategy) has far more leverage on RAG performance than model selection — the chapter quantifies this as roughly 5–6x more impactful.

## Frameworks Introduced
- **Knowledge Base / Vector DB / RAG as One Architecture**: the knowledge base coordinates sync of enterprise sources into vector DBs; RAG is the consumption pattern that routes queries through the knowledge base to ground LLM responses.
  - When to use: as the master mental model before designing any retrieval system — don't treat these as independently selectable components.
- **Five-Layer Knowledge Base Architecture** (extends Ch. 3's semantic layer): Data Foundation → Metadata/Ontology Management (now explicitly including graph databases/GraphRAG) → Transformation/Enrichment → Vectorization/Indexing → APIs/Reasoning, with governance embedded throughout rather than as a separate layer.
  - When to use: as the concrete blueprint for building a production knowledge base; graph databases (Neptune, Neo4j) become essential once multihop, relationship-rich reasoning is required that pure vector similarity cannot represent.
- **GraphRAG**: combines vector indexes (semantic similarity) with graph traversal (explicit relationships) in one retrieval system.
  - When to use: when questions require connecting multiple entities across a relationship chain (e.g., "What compliance requirements affect our derivatives trading desk?") that pure embedding search would miss.
  - How: vector search surfaces semantically relevant documents while graph traversal surfaces related regulations/products/org units that a flat embedding search can't connect; ontology should be mutually exclusive and collectively exhaustive (MECE) for high-confidence retrieval.
- **Seven-Stage Data Transformation Pipeline**: Data selection → cleansing → enhancement → filtering → chunking → embedding → automation (continuous refresh).
  - When to use: as the end-to-end checklist for turning raw enterprise data into retrievable knowledge; chunking and cleansing each carry up to 35% retrieval-accuracy impact — the two highest-leverage stages.
- **RAG Architecture Evolution (6 tiers)**: Simple RAG → Simple RAG with memory → Branched RAG → Adaptive RAG → Self-RAG → Agentic RAG.
  - When to use: match architecture sophistication to task complexity — start simple ("start the learning cycle," don't "boil the ocean"), escalate only as business needs demand it.
  - How: Simple RAG for basic Q&A; add memory for conversational continuity; Branched RAG routes to specific sources by input type; Adaptive RAG adjusts strategy by query complexity; Self-RAG autonomously generates its own retrieval queries; Agentic RAG adds full planning/reflection/tool-use.
- **Vector Database Selection Framework (4 dimensions)**: data scale, data sensitivity, existing infrastructure, budget — evaluated jointly, not as a single ranking.
  - When to use: whenever choosing/justifying a vector database; avoids the common mistake of picking "fastest" or "most feature-rich" without checking compliance fit or infrastructure alignment.
  - How: <10M vectors → Chroma/pgvector; 10M–1B → Qdrant/Weaviate; >1B → Milvus/Pinecone; high sensitivity → self-hosted on-prem (Milvus/Weaviate); existing PostgreSQL/MongoDB → extend with pgvector/Atlas Vector Search; limited budget → open source + engineering time; low ops overhead priority → managed services.
- **Chunking Strategy Selection**: ten named strategies (fixed-size, sentence-based, recursive, semantic, sliding window, hierarchical, topic-based, AI-driven dynamic, plus domain-specific variants) mapped to content types in Table 5-10.
  - When to use: this is the single highest-leverage RAG optimization lever (35% performance swing) — select deliberately per content type rather than defaulting to fixed-size chunking.
- **Drift Mitigation Decision Path**: Query Drift Compensation (QDC, cheap/approximate) → Dual-search strategy (accurate/2x cost) → Full reindexing with shadow index + blue/green cutover (expensive/slow but exact).
  - When to use: whenever an embedding model is upgraded or drift metrics exceed thresholds; choose based on acceptable accuracy loss vs. cost/time budget.

## Key Concepts
- **Vector embedding dimensionality trade-off**: higher dimensions improve recall/nuance but cost scale linearly in storage/memory/latency (e.g., 768 vs. 3,072 dims = ~300GB vs. 1.2TB for 100M vectors); Matryoshka Representation Learning (MRL) lets one embedding serve multiple truncated dimensionalities without recomputation.
- **ANN (Approximate Nearest Neighbor) search**: trades small recall loss (90–99%) for 10–100x faster queries than brute-force exact search — the foundational trick behind all production vector indexes.
- **Index types**: HNSW (graph-based, very high recall, high memory, <500M vectors), IVF-Flat (clustered partitions, balanced, 100M–1B), IVF-PQ (clustered+compressed, low memory, 1B+, 85–90% recall), DiskANN (disk-resident graph, 10B+, minimal RAM), ScaNN (Google, hardware-accelerated GPU/TPU).
- **Quantization/compression**: scalar (float32→int8, 75% reduction, minimal loss), binary (float32→1-bit, 97% reduction, moderate loss), product quantization (10–30x reduction); Milvus RaBitQ achieves 32x compression at 95% recall.
- **Hybrid search**: combining dense vector similarity with sparse keyword matching (BM25) — necessary because pure semantic search misses precise terminology (e.g., "Basel III" as an exact regulatory term).
- **Reranking**: a second-pass precision step (cross-encoder, multistage, contextual, LLM-based) applied after initial fast retrieval; cross-encoders show up to +28% NDCG@10 over baseline at the cost of added latency.
- **Query transformation**: query expansion (broaden with synonyms), query rewriting (clarify phrasing), query decomposition (split multipart questions), and HyDE (Hypothetical Document Embeddings — generate a hypothetical answer first, then search for documents similar to it; demonstrated 38% relative NDCG@10 gain in the original study).
- **Semantic caching**: uses locality-sensitive hashing (LSH) or embedding comparison to recognize differently-worded queries seeking the same answer, returning cached results without a new retrieval/generation cycle — reported to cut latency 40–50% and redundant LLM calls by ~70%.
- **Embedding incompatibility across models**: embeddings from different models exist in incompatible vector spaces — there is no partial migration path; switching embedding models requires full re-embedding/reindexing of the entire corpus (a "high-stakes decision with long-term cost implications").

## Mental Models
- Treat knowledge base + vector database + RAG as one system to design holistically, not as three products to shop for independently.
- Use the "5-6x more important than model choice" finding as a forcing heuristic: whenever RAG underperforms, debug data preparation (chunking, cleansing, metadata enrichment) before reaching for a bigger/better LLM.
- Think of vector embeddings as powerful for *implicit* similarity but blind to *explicit* relationship type — when a question needs to know *why* two things are related or how to traverse a multihop chain, reach for a knowledge graph, not a bigger embedding model.
- View chunking strategy selection as a direct function of document structure: legal/financial documents need hierarchical chunking (preserve clause/section structure); medical texts need semantic chunking (preserve diagnosis/treatment relationships); FAQs need atomic question/answer-pair chunking.
- Treat vector database quality degradation as *silent* and *gradual*, unlike traditional databases' binary correctness — this is why continuous drift monitoring (centroid shifts, cosine similarity variance) is mandatory, not optional.
- When evaluating a vector database, weigh governance/compliance capability (encryption, RBAC, SOC 2/GDPR certification) as seriously as query latency — a fast database that fails compliance review is disqualified regardless of benchmarks.

## Anti-patterns
- **Flattening structured records into independent chunks for embedding**: destroys relational context (a customer record severed from its transactions, a product spec separated from its regulatory approvals) — preserve relationships via knowledge graphs instead.
- **Chasing a "better" LLM to fix poor RAG performance**: empirically, chunking strategy improvements (35%) and embedding improvements (27%) dwarf LLM-swap improvements (6%) — model upgrades are the lowest-leverage fix.
- **Selecting a vector database on a single dimension (speed or feature richness alone)**: leads to discovering months later that compliance, infrastructure integration, or actual-scale costs were misjudged; always evaluate scale + sensitivity + existing infrastructure + budget jointly.
- **Assuming embedding models can be swapped incrementally**: old and new embeddings cannot coexist in one index (incompatible vector spaces) — plan for full reindexing cost and downtime risk, or use QDC/dual-search mitigation.
- **Using fixed-size chunking indiscriminately across all content types**: appropriate only for simple/homogeneous documents; applying it to legal contracts or medical texts fragments meaning and degrades retrieval.
- **Relying on built-in exact query-caching alone for high query diversity**: misses the majority of real traffic where differently worded questions seek the same answer — semantic caching captures this long tail that exact-match caching cannot.
- **Treating open source vector databases as free**: they shift cost from licensing to engineering time (manual encryption, RBAC, backups, monitoring, compliance self-certification) — a "hidden cost" the chapter explicitly flags as often underestimated.

## Code Examples
No executable code is presented in this chapter (figures, comparison tables, and architectural descriptions carry the technical content); the chapter's practical artifacts are its extensive comparison tables (chunking strategies, indexing algorithms, vector DB options) rather than source listings.

## Reference Tables
*Vector Database Indexing Algorithm Comparison (condensed)*

| Algorithm | Recall | Best Scale | Memory |
|---|---|---|---|
| HNSW | 95–99% (very high) | <500M vectors | High |
| IVF-Flat | 90–95% (high) | 100M–1B vectors | Medium |
| IVF-PQ | 85–90% (medium) | 1B+ vectors | Low |
| DiskANN | 90–95% (high) | 10B+ vectors | Very low (disk-resident) |
| ScaNN | Fast at any scale with GPU/TPU | Any (with specialized hardware) | — |

*Open Source Vector Database Comparison (condensed)*

| Database | Scale | Differentiator | License |
|---|---|---|---|
| Milvus | 10B+ vectors | 14 index types, RaBitQ 72% memory reduction | Apache 2.0 |
| Qdrant | Billions | GPU acceleration, advanced filtering | Apache 2.0 |
| Weaviate | Billions | Hybrid search, MUVERA encoding | BSD 3-Clause |
| Chroma | <1M vectors | Simple API, auto-embedding (prototyping) | Apache 2.0 |
| pgvector | 50M+ | PostgreSQL extension, SQL integration | PostgreSQL |

*Vector DB Selection by Scale*

| Vector Count | Recommended |
|---|---|
| <10M | Chroma, pgvector |
| 10M–1B | Qdrant, Weaviate |
| 1B+ | Milvus, Pinecone |

*Chunking Recommendations by Content Type (sample)*

| Content Type | Strategy | Chunk Size | Overlap |
|---|---|---|---|
| Legal contracts | Hierarchical | 512–1,024 tokens | 40–50% |
| Financial reports | Hierarchical | 512–1,024 tokens | 30–40% |
| Technical documentation | Recursive | 256–512 tokens | 20–30% |
| Medical/clinical texts | Semantic | 256–512 tokens | 40–50% |
| Conversational content | Sentence/turn-pair | 128–256 tokens | 20–30% |
| FAQs/knowledge articles | QA-pair chunking | 128–256 tokens | n/a (atomic) |

*Key Vector DB Production Metrics*

| Metric | Target | Action Trigger |
|---|---|---|
| Retrieval precision | >80% | <70% investigate |
| Retrieval recall | >90% | <80% review indexes |
| Data freshness | <1 hour | >24 hours refresh |
| Index quality (recall@10) | >95% | <90% rebuild index |

## Worked Example
Goldman Sachs' GS AI Platform demonstrates the chapter's hybrid-data and governance principles together: it combines structured transaction data and regulatory rules with unstructured content via RAG, routes all queries through an internal compliance gateway (adding proprietary context, applying prompt filtering, data anonymization, and policy checks before reaching any of its integrated models — GPT-4, Gemini, Llama, Claude), and maintains complete audit trails of every AI interaction. This architecture achieved over 50% adoption among 46,000+ employees, a 20% developer productivity increase, and a 35% reduction in false positive rates in AML/transaction monitoring. A second example — JPMorgan's basic RAG implementation for KYC processing — shows that even Simple RAG (the least sophisticated tier) delivers a 90% productivity increase (20% fewer employees processing 48% more files) when applied to a well-defined document workflow, reinforcing the chapter's point that architecture sophistication should match actual task complexity, not maximize it by default.

## Key Takeaways
1. Design knowledge base, vector database, and RAG as one integrated retrieval system from the start — they are not independently swappable components.
2. Prioritize data preparation (chunking strategy especially) over model upgrades: chunking can swing performance by 35%, versus only 6% from changing the LLM.
3. Add a knowledge graph layer (GraphRAG) whenever retrieval needs to answer "why" or "how are these connected" questions that pure vector similarity cannot represent.
4. Select chunking strategy per content type using the content-type-to-strategy mapping (hierarchical for legal/financial, semantic for medical, QA-pair for FAQs) rather than one default strategy for everything.
5. Choose vector database infrastructure against four joint dimensions — data scale, sensitivity, existing infrastructure, budget — never on benchmark speed alone; many teams run a two-tier strategy (Chroma/pgvector for dev, Milvus/Pinecone for production).
6. Treat embedding model selection as a high-stakes, hard-to-reverse decision because of full-reindex costs on any future switch; budget for periodic reindexing and consider QDC or dual-search strategies to soften transitions.
7. Implement hybrid search (dense + sparse/BM25) by default for enterprise RAG — pure semantic search alone underperforms on exact-terminology queries.
8. Monitor vector database health continuously (precision, recall, freshness, index quality, embedding drift) since quality degrades silently and gradually, unlike traditional databases' binary correctness.

## Connects To
- **Ch 3**: Directly builds on the five-layer semantic architecture, extending the metadata/ontology layer with graph databases and GraphRAG, and the vectorization layer with concrete indexing/compression techniques.
- **Ch 4**: Vector store access control (row-level-security analog) and the Data Privacy/Security capability matrix connect directly to this chapter's governance feature comparison (open source vs. commercial) and end-to-end encryption needs.
- **Ch 6**: The chapter's cost optimization (caching, tiered models, batching), reranking, and query transformation techniques are precursors to Chapter 6's full production-optimization focus.
- **RAG / Knowledge graphs / Embeddings**: this chapter is the book's most detailed technical treatment of these three concepts, positioning GraphRAG as the synthesis of vector search and graph traversal.
