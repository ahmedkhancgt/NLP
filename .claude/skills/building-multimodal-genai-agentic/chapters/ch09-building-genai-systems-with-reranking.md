# Chapter 9: Building GenAI Systems with Reranking

## Core Idea
Rerankers (cross-encoder, late interaction, hybrid, learning-to-rank, LLM-based) act as a precision filter between fast bi-encoder retrieval and generation, and in multimodal RAG they must jointly reason over text+image inputs; a working system combines multi-index embeddings, vector fusion, and cross-encoder reranking before passing results to an LLM.

## Frameworks Introduced
- **Five reranker categories**: Cross-encoder (highest accuracy, joint encoding, expensive — e.g., Cohere Rerank API), Late interaction (ColBERT-style, token-level MaxSim, moderate scalability), Hybrid (score fusion of sparse BM25 + dense signals, e.g., Reciprocal Rank Fusion), Learning-to-rank (classical ML — LambdaMART, RankSVM — combining multiple features), LLM-based (fine-tuned or zero-shot prompted LLM ranks passages).
  - When to use: cross-encoder for small candidate pools needing max precision (legal/academic/QA); late interaction for large collections needing scalable token-level matching; hybrid for enterprise search combining lexical+semantic; LLM-based for flexible/dynamic criteria on small/high-value candidate sets.
- **Multimodal cross-encoder reranking pipeline**: Stage 1 bi-encoder retrieval (top-k via ANN) → Stage 2 cross-encoder reranking (joint query+candidate encoding across text/image modalities via VLMs like CLIP/BLIP/Flamingo/GIT/OFA/Qwen-VL) → Stage 3 top-N selection for generation.
- **Multi-index embedding**: Maintain separate, modality-specialized vector indexes (text index via text-embedding-3-large, image index via CLIP, code index via CodeBERT) instead of one undifferentiated index; route/aggregate/rerank across indexes at query time.
  - How: construct per-modality indexes → route query to relevant index(es) → aggregate retrieved results → rerank with cross-encoder → generate from top-ranked results.
- **Vector fusion for hybrid (image+text) queries**: Average fusion (`joint_vec = [(i+j)/2 for i,j in zip(image_vec, text_vec)]`) or concatenation (`joint_vec = image_vec + text_vec`) to combine same-length embeddings into one query vector for joint retrieval.

## Key Concepts
- **Cross-encoder mathematical formulation**: Input `[CLS] Q [SEP] D [SEP]` processed jointly through transformer layers (`H^l = TransformerLayer(H^{l-1})`), final `[CLS]` token representation `z` scored via `s = sigmoid(w^T z + b)`; trained with binary cross-entropy (pointwise) or hinge/pairwise ranking loss.
- **Indexable vs. non-indexable embeddings**: Bi-encoders produce reusable, storable vectors (suitable for vector DB indexing); cross-encoders produce only a single relevance score per query-document pair — not reusable, not indexable, used exclusively for reranking small candidate sets.
- **Model distillation (for cross-encoder cost mitigation)**: Train a lightweight model to approximate cross-encoder scores, avoiding full joint-encoding cost at scale.
- **Query-aware pruning**: Only pass documents with overlapping metadata to the (expensive) cross-encoder, reducing the candidate set before reranking.
- **Commercial rerankers**: Cohere Rerank API (transformer cross-encoder, multilingual, code/semi-structured support), Azure Cognitive Search semantic reranking (Turing-series models, highlights/explanations), Amazon Kendra/Bedrock (hosted Cohere reranker access), OpenSearch/Elasticsearch external-API reranking, Answer.AI reranker library.

## Mental Models
- Think of the retrieval→rerank→generate pipeline as "wide net, then careful sort" — bi-encoders cast a wide net (recall), cross-encoders sort precisely but only within that net (precision), generation only sees the sorted top.
- Multi-index embedding is the retrieval-layer analogue of dual-index design from Chapter 7 — generalized from 2 modalities (text/image) to N modalities (text/image/code/tables), each with its own best-fit embedding model.
- Cross-encoders "do not create multi-index embeddings" — remember the clean division of labor: multi-index embeddings provide breadth/modality coverage at retrieval time; cross-encoders provide precision at reranking time; neither substitutes for the other.
- Vector fusion (average vs. concatenation) is a simple but consequential design choice — averaging assumes commensurable scales/dimensions; concatenation preserves distinct information but requires consistent strategy at both indexing and query time.

## Anti-patterns
- **Using a cross-encoder for first-stage retrieval over a large corpus**: computationally infeasible — cross-encoders must score each query-document pair individually, making them unsuitable beyond small candidate sets.
- **Expecting cross-encoders to produce storable/indexable embeddings**: they output only a scalar relevance score per pair, not a reusable vector — never try to cache/index cross-encoder "embeddings" for retrieval.
- **Inconsistent vector fusion strategy between indexing and query time**: switching between averaging and concatenation (or using different normalization) breaks retrieval correctness — fusion strategy must be identical at index-build and query time.
- **Leaving reranking out of a multimodal RAG pipeline**: coarse bi-encoder similarity scores can surface loosely relevant candidates; skipping the reranking stage risks hallucination-inducing noisy context reaching the LLM.
- **Ignoring commercial rerankers when building from scratch is unnecessary**: Cohere Rerank, Azure semantic reranking, and similar hosted options can replace significant in-house model development/maintenance effort.

## Code Examples
```python
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_text_ollama(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    outputs = clip_model.get_text_features(**inputs)
    return outputs[0].tolist()

def embed_image_ollama(image_path):
    image = Image.open(image_path).convert("RGB")
    outputs = clip_model.get_image_features(**clip_processor(images=image, return_tensors="pt"))
    return outputs[0].tolist()
```
- **What it demonstrates**: Shared CLIP model producing both text and image embeddings for a multi-index ChromaDB setup.

```python
joint_vec = [(i + j) / 2 for i, j in zip(image_vec, text_vec)]  # average fusion
# or
joint_vec = image_vec + text_vec  # concatenation fusion (longer vector)
```
- **What it demonstrates**: Two vector-fusion strategies for combining image and text query embeddings into one joint query vector.

```python
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, metadatas):
    pairs = [(query, doc.get("file", "")) for doc in metadatas]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(metadatas, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]
```
- **What it demonstrates**: Cross-encoder reranking applied to candidate metadata retrieved from ChromaDB, using a pretrained MS MARCO MiniLM cross-encoder.

## Reference Tables
| Feature | Bi-encoder | Late interaction | Cross-encoder |
|---|---|---|---|
| Encoding | Independent | Independent | Joint |
| Interaction | None | Token-level (post-encode) | Full (within encoder) |
| Scalability | High | Moderate | Low |
| Accuracy | Moderate | High | Highest |
| Use case in RAG | First-stage retriever | Lightweight reranker | Precision reranker (small N) |

| Encoder type | Indexable embeddings | Primary use |
|---|---|---|
| Bi-encoder | Yes | Vector search and retrieval |
| Cross-encoder | No | Semantic reranking |

## Worked Example
A modular multimodal laptop-spec assistant: `config.py` centralizes ChromaDB paths/model names; `embedding_utils.py` uses a shared CLIP model for both `embed_text_ollama()` and `embed_image_ollama()`; `index_builder.py` creates two ChromaDB collections (`laptop_images`, `laptop_texts`) via `run_once.py`. The Streamlit `app.py` offers four modes: Image→specs (embed image, retrieve similar images+specs), Image+text→specs (average-fuse vectors, retrieve+rerank), Text→image+specs (pure text query matched to spec doc + image), Text→generated answer (direct LLM call via Ollama). A `reranker.py` module applies a `cross-encoder/ms-marco-MiniLM-L-6-v2` CrossEncoder to rerank ChromaDB results before display. The chapter leaves `generation.py` and `run_once.py` reorganization as explicit reader exercises to complete the production-ready pipeline.

## Key Takeaways
1. Rerankers come in five distinct families (cross-encoder, late interaction, hybrid, learning-to-rank, LLM-based) — pick based on accuracy/scalability/cost trade-offs for your candidate-set size.
2. In multimodal RAG, cross-encoder reranking requires VLMs (CLIP, BLIP, Flamingo, GIT, OFA, Qwen-VL) capable of jointly fusing visual and textual inputs.
3. Cross-encoders never produce indexable embeddings — they are scoring-only and belong exclusively in the reranking stage, never in retrieval/indexing.
4. Multi-index embedding (separate vector stores per modality) plus consistent vector fusion (averaging or concatenation) enables flexible hybrid/multimodal querying.
5. Commercial reranking APIs (Cohere, Azure semantic reranking, Amazon Bedrock/Kendra) offer production-grade reranking without building custom cross-encoders in-house.
6. Mitigation strategies (top-k pre-filtering, distillation, caching, query-aware pruning) make cross-encoder reranking computationally feasible despite its cost.

## Connects To
- **Ch 1 & 6**: Directly extends the bi-encoder/cross-encoder and interaction-taxonomy frameworks (no/late/full interaction) into a concrete, implemented reranking system.
- **Ch 7-8**: Builds on the bidirectional retrieval system and multimodal RAG generator, inserting the missing reranking stage between retrieval and generation.
- **Ch 10**: Explicitly previewed as the next chapter — retrieval optimization techniques that complement reranking.
- **Ch 15**: Query-aware pruning and multi-index routing here recur in agentic text-to-SQL architecture decision-making.
