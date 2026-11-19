# Chapter 7: Building a Bidirectional Multimodal Retrieval System

## Core Idea
A production multimodal retrieval system uses a shared bi-encoder (e.g., CLIP) to independently embed text and images into a common latent space, stores each modality in its own normalized vector collection, and supports bidirectional queries (text→image, image→text) via vector similarity search alone — with no generative component yet.

## Frameworks Introduced
- **Bidirectional multimodal retrieval architecture**: Dual-index vector store (separate TEXT_COLLECTION and IMAGE_COLLECTION) populated by a shared CLIP-based embedder; queries in either modality are embedded and matched against the *other* modality's collection for cross-modal retrieval.
  - How: load/chunk documents + load images → embed both via shared CLIP bi-encoder → L2-normalize vectors (for cosine similarity) → upload to Qdrant collections with metadata payload (source text or image filename) → query by embedding the input and searching the target collection.
- **Four output-based multimodal classifications recap**: Text-to-image (translation), Image-to-text (translation), Text+image-to-image (fusion), Text-to-specs+image (translation + structure generation + fusion) — framed as translation/alignment/fusion paradigms that inform how retrieval indexing/querying must be designed.
- **Technical architecture of a multimodal retrieval system**: query intake (text/image/hybrid) → modality-specific embedding → chunking (text) / direct embedding (image) → multimodal vector store (unified indexing, metadata filtering, configurable similarity metric) → query encoding + similarity search (optionally late fusion, score normalization, cross-modal reranking) → result mapping back to original content → optional LLM-based summarization/generation layer.
- **Modality routing + embedding caching + retrieval augmentation**: Optimization layer — detect dominant query modality and route accordingly; cache embeddings of frequent queries to cut latency; apply query expansion/pseudo-relevance feedback to improve recall.

## Key Concepts
- **CLIP as bi-encoder**: `openai/clip-vit-base-patch32` independently encodes text (via transformer) and images (via ViT) into a shared latent space with no cross-attention — fast, precomputable, ideal for retrieval-scale matching.
- **L2 normalization for cosine similarity**: `v / norm(v)` ensures unit-length embeddings so cosine similarity depends only on direction, giving consistent distance calculations across the vector space.
- **Dual-index (separate collection) design**: Text and image embeddings stored in distinct Qdrant collections (`vdr_text`, `vdr_images`) rather than one mixed collection — enables modality-optimized retrieval and optional late-fusion logic.
- **Document chunking for retrieval granularity**: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` — larger chunk size than Ch3/4 examples, chosen to align with CLIP's token limits while preserving context continuity.
- **Payload-based result mapping**: Vector search returns IDs; metadata payload (`{"source": chunk_text}` or `{"image": filename}`) maps results back to human-readable content for display.
- **Retrieval-only vs. RAG**: This chapter's system performs semantic retrieval only (no natural language generation) — explicitly scoped as a precursor to full multimodal RAG (addressed in Ch8).

## Mental Models
- Think of CLIP's shared latent space as the "universal translator" that lets a text query and an image document be compared directly, even though they were encoded by entirely separate network branches.
- Treat dual-index design as the multimodal analogue of the single-collection-vs-multiple-collections trade-off from Chapter 2 — here, splitting by modality (not by date/language) is the natural partition because similarity search is always cross-modal, never within-collection for the same query.
- Retrieval-only vs. retrieval-augmented-generation is a staged-capability model: get cross-modal matching correct and fast first (this chapter), then layer an LLM on top for synthesis (next chapter) — don't conflate the two engineering problems.

## Anti-patterns
- **Skipping vector normalization before cosine similarity search**: un-normalized embeddings make similarity scores magnitude-dependent, corrupting ranking — always L2-normalize when using cosine distance.
- **Mixing text and image embeddings in one undifferentiated collection without a clear retrieval policy**: makes it harder to apply modality-specific optimizations (e.g., different HNSW tuning) and to reason about which collection a query should target.
- **Assuming retrieval alone equals RAG**: a working cross-modal retriever without a generation layer only returns "most similar item," not a synthesized answer — readers must explicitly add an LLM wrapper (RetrievalQA / RAG chain) to close the loop.
- **Feeding retrieved images directly into a text-only LLM**: most LLMs cannot consume raw image embeddings/files — either caption images first (image-to-text) or use a native multimodal LLM (GPT-4V, LLaVA, Kosmos-2).

## Code Examples
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
_MODEL_ID = "openai/clip-vit-base-patch32"
def get_mm_embedder(device: str = "cpu"):
    return HuggingFaceEmbedding(model_name=_MODEL_ID, device=device, trust_remote_code=True)
```
- **What it demonstrates**: Initializing a shared CLIP bi-encoder for both text and image embedding via llama_index's HuggingFace wrapper.

```python
def normalize(vecs):
    return [v / norm(v) for v in vecs]

text_vecs = normalize(embedder.get_text_embedding_batch([...]))
image_vecs = normalize(embedder.get_image_embedding_batch([...]))

client = QdrantClient(path=DB_PATH)
if not client.collection_exists(...):
    client.create_collection(...)  # cosine distance, dim inferred from first vector

client.upload_points(
    TEXT_COLLECTION,
    [models.PointStruct(id=i, vector=text_vecs[i], payload={"source": d.page_content}) for i, d in enumerate(text_docs)],
)
client.upload_points(
    IMAGE_COLLECTION,
    [models.PointStruct(id=i, vector=image_vecs[i], payload={"image": Path(d.page_content).name}) for i, d in enumerate(image_docs)],
)
```
- **What it demonstrates**: Full index-building pipeline — normalize embeddings, create per-modality Qdrant collections, upload points with traceable metadata payload.

```python
# Text -> Image query
q_vec = mm_embed.get_text_embedding(query)
res = client.query_points(collection_name=IMAGE_COLLECTION, query=q_vec, using="image", with_payload=["image"], limit=1)

# Image -> Text query
img_vec = mm_embed.get_image_embedding("temp_input_image.jpg")
res = client.query_points(collection_name=TEXT_COLLECTION, query=img_vec, using="text", with_payload=["source"], limit=1)
```
- **What it demonstrates**: The bidirectional query pattern — embed in one modality, search the *other* modality's collection, since both were embedded into the same CLIP latent space.

## Reference Tables
| Output classification | Input(s) | Mechanism | Example |
|---|---|---|---|
| Text-to-image | Text | Text encoder → generative decoder | DALL-E, Imagen, Stable Diffusion |
| Image-to-text | Image | Visual backbone → language decoder | Captioning, VQA |
| Text+image-to-image | Text + image | Fused embeddings → conditional generator | Sketch + prompt → photorealistic image |
| Text-to-specs+image | Text | NLP + symbolic reasoning → spec sheet + image | "Design a chair with dimensions X" |

| Library | Role in this chapter's stack |
|---|---|
| Streamlit | Frontend UI for text/image query input and result display |
| QdrantClient | Vector DB client — insert/query multimodal embeddings |
| LlamaIndex | RAG framework; HuggingFaceEmbedding wrapper for CLIP |
| LangChain / LangChain Community | Document loaders, text splitting, orchestration |
| Sentence Transformers | Alternative text embedding backend |
| Transformers (HuggingFace) | Underlying CLIP/BERT/ViT model implementations |
| Pillow | Image file handling/preprocessing |
| pypdf | PDF text extraction |
| scikit-learn / NumPy | Vector math, normalization, evaluation utilities |

## Worked Example
A Streamlit app (`"🔍 Multimodal Search Demo (Text ↔ Image)"`) offers two modes via radio button: "Text → Image" and "Image → Text". On startup, `build_vectorstores()` (cached via `@st.cache_resource`) loads PDFs/text from `data/documents` and images from `data/images`, embeds both with a shared CLIP model, L2-normalizes vectors, and populates two Qdrant collections (`vdr_text`, `vdr_images`). For a text query, the app embeds the text and searches `IMAGE_COLLECTION`, displaying the top-1 matching image. For an uploaded image, the app embeds it and searches `TEXT_COLLECTION`, displaying the top-1 matching text snippet. The folder structure separates concerns cleanly: `rag/loaders.py` (ingestion), `rag/embedding_utils.py` (CLIP embedder), `rag/index_builder.py` (vector store construction).

## Key Takeaways
1. A shared bi-encoder (CLIP) is the enabling technology for bidirectional cross-modal retrieval — one embedding model, two independently-queryable collections.
2. L2 normalization is mandatory when using cosine similarity for ranking — never skip it.
3. Separating text and image embeddings into distinct collections (dual-index) is the standard pattern, enabling modality-specific tuning and optional late-fusion.
4. This chapter deliberately stops at retrieval — it does not yet generate natural language answers; that is the explicit scope of the next chapter (multimodal RAG).
5. Metadata payload (source filename/text) is what makes vector search results usable — plan payload schema before indexing.
6. Extending retrieval to full RAG requires either captioning retrieved images into text or adopting a native multimodal LLM (GPT-4V, LLaVA, Kosmos-2).

## Connects To
- **Ch 2**: Directly implements the multimodal vector embedding and multimodal vector database concepts (Qdrant collections, points, payload) introduced there.
- **Ch 1 & 6**: Uses CLIP explicitly as a "no interaction" bi-encoder, applying the interaction taxonomy from Ch1/Ch6 to a concrete multimodal system.
- **Ch 8**: The explicit "to do" at this chapter's end — adding an LLM generation layer on top of this retriever — is exactly what the multimodal RAG chapter builds next.
- **Ch 16**: Image-to-text retrieval here is a lighter-weight alternative to full OCR/text-extraction covered later.
