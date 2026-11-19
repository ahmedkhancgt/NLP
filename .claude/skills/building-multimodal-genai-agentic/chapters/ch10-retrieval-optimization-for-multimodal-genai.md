# Chapter 10: Retrieval Optimization for Multimodal GenAI

## Core Idea
Retrieval systems suffer six named structural drawbacks (recall/precision trade-off, limited semantic understanding, modality mismatch, index staleness, ranking inefficiency, lack of contextual awareness), each addressed by a specific, composable optimization technique — multi-index embedding, modality routing, query expansion, embedding normalization, hybrid retrieval with score fusion, cross-encoder reranking with prefiltering, and adaptive index refresh.

## Frameworks Introduced
- **Six retrieval drawbacks → six mitigations** (the chapter's central framework — see Reference Tables): each drawback maps 1:1 to an optimization technique.
- **Prefiltering thresholds**: Apply a score-cutoff or top-N cutoff to the initial candidate pool *before* expensive cross-encoder reranking, to control reranking latency without losing most truly relevant results.
  - How: first-stage retriever returns e.g. 1000 candidates → prefilter to top-64 by score → cross-encoder reranks only those 64 → return top-k (5-10) to user/LLM.
- **Score normalization for hybrid fusion**: Min-max normalization, z-score normalization, rank-based fusion (e.g., Reciprocal Rank Fusion/RRF), or learned/weighted fusion — needed because BM25 scores and embedding similarity scores live on incommensurable scales.
- **Weighted embedding fusion**: `fused = text_weight * text_emb + (1 - text_weight) * image_emb` (then re-normalized) — lets a domain bias toward the more reliable modality (e.g., text-heavy weighting for legal, image-heavy for e-commerce) instead of naive 50/50 averaging.
- **Adaptive index refresh**: Incremental/scheduled/event-driven reindexing to combat staleness and embedding drift — triggers include new/changed documents, embedding-model upgrades, or detected recall@k degradation.
- **Genetic algorithm (GA) retrieval-parameter optimization**: Population-based evolutionary search (selection, crossover, mutation) over retrieval configuration vectors (e.g., `[text_weight, query_expansion_flag]`), guided by a fitness function (NDCG@k, MRR, precision@k) — handles non-convex, multi-objective, mixed discrete/continuous parameter tuning without requiring differentiability.
- **Dense + multi-vector (ColBERT) + image unified vector store**: Store three embedding types per document (dense_text 384-d via BAAI/bge-small-en, colbert_text 128-d token-level via colbertv2.0, image 512-d via CLIP) in one Qdrant collection with per-field indexing config — dense_text and image get HNSW indexing (fast ANN), colbert_text has HNSW disabled (reserved for MaxSim-based reranking).

## Key Concepts
- **Modality-based routing**: Direct a query to the appropriate modality-specific index (`routing_table = {"text": "text_index", "image": "image_index", "multimodal": "hybrid_index"}`) instead of forcing all modalities into one homogeneous representation.
- **Query expansion**: Add synonyms/related terms to a query (`"climate" → ["environment", "weather"]`) to bridge lexical-semantic gaps and boost recall, at some cost to precision.
- **Embedding normalization**: L2-normalize all vectors so similarity depends on angle (cosine) not magnitude — prevents longer-norm vectors from artificially scoring higher.
- **Contextual filtering**: Metadata-based query-time filters (source, year, region) via Qdrant's `Filter`/`FieldCondition`/`MatchValue` to exclude technically-relevant-but-contextually-inappropriate results.
- **Index staleness / embedding drift**: A static index omits new content and may become incompatible if the embedding model itself changes — requires re-embedding/reindexing to stay aligned.
- **Two-stage retrieval with multi-vector reranking**: Stage 1 = fast dense retrieval (HNSW on `dense_text`); Stage 2 = ColBERT-style token-level reranking via `MAX_SIM` comparator on `colbert_text` multi-vectors — combining speed and token-level precision.

## Mental Models
- Treat each retrieval drawback as a named disease with a named cure — don't reach for a generic "better embedding model" fix when the actual problem is staleness (needs refresh), modality mismatch (needs routing), or ranking inefficiency (needs reranking + prefiltering).
- Think of prefiltering thresholds as a "funnel": wide fast first-stage retrieval → narrowing score/rank cutoff → expensive precise reranking only on the narrowed set — never run the expensive stage on the wide end of the funnel.
- GA-based parameter tuning is a practical substitute for manual grid search when the retrieval configuration space is large, non-convex, and mixes continuous (weights) and discrete (flags) parameters.
- Adaptive index refresh turns a "frozen snapshot" retrieval system into a "living" one — the index must track both data changes and embedding-model changes, not just data changes.

## Anti-patterns
- **Averaging embeddings from different modalities without considering reliability**: naive 50/50 fusion can dilute the dominant, more trustworthy modality's signal — use weighted fusion with domain-appropriate weights instead.
- **Merging BM25 and vector-similarity scores without normalization**: the two scoring schemes are on different numeric scales; merging raw scores lets one method dominate arbitrarily — always normalize (min-max, z-score, or rank-based) before fusion.
- **Running cross-encoder reranking on the entire candidate pool**: computationally prohibitive at scale — always prefilter to a manageable top-N/score-threshold subset first.
- **Treating the vector index as static/one-time-built**: without adaptive refresh, new content is invisible to retrieval and embedding-model upgrades leave the index silently incompatible/degraded.
- **Expanding queries indiscriminately**: uncontrolled query expansion increases recall but can hurt precision if synonym sets are too broad or ungrounded — tune expansion strategy (e.g., feedback-driven) to balance both.

## Code Examples
```python
def route_query(query: str, modality: str = "text") -> str:
    routing_table = {"text": "text_index", "image": "image_index", "multimodal": "hybrid_index"}
    return routing_table.get(modality, "text_index")

def weighted_embedding_fusion(text_emb, image_emb, text_weight: float = 0.6):
    fused = text_weight * text_emb + (1 - text_weight) * image_emb
    return normalize_embedding(fused)

def score_fusion(results_a: list, results_b: list, method: str = "reciprocal_rank") -> list:
    def reciprocal_rank(score, rank):
        return 1 / (rank + 1)
    fused_scores = {}
    for rank, item in enumerate(results_a):
        fused_scores[item.id] = fused_scores.get(item.id, 0) + reciprocal_rank(item.score, rank)
    for rank, item in enumerate(results_b):
        fused_scores[item.id] = fused_scores.get(item.id, 0) + reciprocal_rank(item.score, rank)
    merged = [{"id": k, "fused_score": v} for k, v in fused_scores.items()]
    return sorted(merged, key=lambda x: x["fused_score"], reverse=True)
```
- **What it demonstrates**: Modality routing, weighted multimodal fusion, and Reciprocal Rank Fusion score aggregation as composable optimization primitives.

```python
def genetic_optimization(generations=20, pop_size=10):
    population = initialize_population(pop_size)
    best_config, best_score = None, -np.inf
    for gen in range(generations):
        scores = [evaluate_config(*ind) for ind in population]
        top_individuals = select(population, scores)
        new_population = top_individuals[:]
        while len(new_population) < pop_size:
            p1, p2 = random.sample(top_individuals, 2)
            new_population.append(mutate(crossover(p1, p2)))
        population = new_population
        if max(scores) > best_score:
            best_score = max(scores)
            best_config = population[np.argmax(scores)]
    return best_config
```
- **What it demonstrates**: A genetic algorithm evolving retrieval configuration (text_weight, query_expansion_flag) toward higher fitness (retrieval quality) over generations.

```python
# Two-stage retrieval: dense first-pass, then ColBERT multi-vector rerank
prefetch = models.Prefetch(query=dense_query, using="dense_text")
# ... then rerank using colbert_text multi-vectors with MAX_SIM comparator
query=colbert_query, using="colbert_text"
```
- **What it demonstrates**: Qdrant-based two-stage retrieval combining fast dense HNSW search with precise ColBERT late-interaction reranking.

## Reference Tables
| Drawback | Impact | Mitigation technique |
|---|---|---|
| Poor recall vs. precision trade-off | Must compromise completeness vs. accuracy | Multi-index embedding, query expansion (tuned) |
| Limited semantic understanding | Misses paraphrases/synonyms, poor multimodal semantics | Query expansion, dense embeddings |
| Modality mismatch | Wrong-modality or missed cross-modal results | Modality-based routing, joint embedding spaces (CLIP) |
| Index staleness | Misses new info, serves outdated content, embedding drift | Adaptive index refresh (incremental/scheduled) |
| Ranking inefficiencies | Fast ranking buries relevant results | Cross-encoder reranking + prefiltering thresholds |
| Lack of contextual awareness | Technically relevant but practically unhelpful/misleading results | Contextual filtering (metadata), preserving chunk context |

| Vector type | Model | Dim | Purpose | Indexing |
|---|---|---|---|---|
| dense_text | BAAI/bge-small-en | 384 | Fast first-stage retrieval | HNSW enabled |
| colbert_text | colbert-ir/colbertv2.0 | 128 (per token) | Token-level MaxSim reranking | HNSW disabled |
| image | openai/clip-vit-base-patch32 | 512 | Visual similarity search | HNSW enabled |

## Worked Example
A multimodal RAG laptop assistant extended with adaptive index refresh: `rag/refresh.py` defines `refresh_text_index()` and `refresh_image_index()`, each deleting and rebuilding its ChromaDB collection from current folder contents (`refresh_all_indexes()` runs both); `run_refresh.py` triggers this on demand, and the Streamlit `app.py` exposes a "Refresh Indexes" button. A more advanced variant stores three embedding types (dense text, ColBERT multi-vector text, CLIP image) in one unified Qdrant collection, using per-vector-field indexing configuration (HNSW for dense_text/image, no HNSW for colbert_text) to support a two-stage pipeline: fast dense retrieval followed by MaxSim-based ColBERT reranking, before context assembly and generation via a local Mistral LLM with a ReAct-style prompt.

## Key Takeaways
1. Every retrieval drawback has a specific, named mitigation — diagnose which drawback is occurring before reaching for a generic fix.
2. Score normalization (min-max, z-score, or rank-based/RRF) is mandatory before fusing lexical and semantic retrieval scores.
3. Prefiltering thresholds make cross-encoder reranking computationally tractable by shrinking the candidate pool before the expensive stage.
4. Adaptive index refresh must account for both new/changed data *and* embedding-model upgrades (drift), not just data changes alone.
5. GA-based hyperparameter search is a viable alternative to manual tuning for multi-objective, non-differentiable retrieval configuration spaces.
6. Combining dense (fast, coarse) + multi-vector/ColBERT (precise, token-level) + image embeddings in one unified, per-field-indexed vector store is the state-of-the-art pattern for production multimodal RAG.

## Connects To
- **Ch 6**: Directly extends the multi-vector representation and late interaction concepts from the two/multi-stage RAG chapter into concrete Qdrant implementation.
- **Ch 9**: Builds on the cross-encoder reranking pipeline, adding prefiltering thresholds as the missing latency-control mechanism.
- **Ch 7-9**: Reuses and extends the same CLIP/ChromaDB/Streamlit laptop-assistant codebase across Chapters 7-10, incrementally adding retrieval, generation, reranking, and now adaptive refresh.
- **Ch 11**: Explicitly previewed as the next chapter — multimodal GenAI systems with voice as input, which will need its own modality-routing and embedding considerations.
