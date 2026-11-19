# Chapter 2: Deep Dive into Multimodal Systems

## Core Idea
Vision-language models (VLMs) are end-to-end unified architectures for vision+text tasks, while multimodal GenAI systems are modular, retrieval-augmented pipelines that can span arbitrary modalities (text/image/audio/tables) and output types — the choice between them is an architecture decision, not a terminology detail.

## Frameworks Introduced
- **VLM categorization**: Retrieval-focused (CLIP, ALIGN, DeCLIP), VQA/captioning (LXMERT, UNITER, VisualBERT), Generative (DALL-E 2, Imagen, Parti), Multimodal reasoning (Flamingo, Kosmos-1), Instruction-tuned (BLIP, BLIP-2, MiniGPT-4).
  - When to use: pick by core objective — alignment/retrieval vs. description vs. synthesis vs. reasoning vs. zero-shot generalization.
- **Cross-encoder vs. bi-encoder VLM architectures**: Cross-encoders concatenate image+text tokens into one transformer with full cross-attention (rich interaction, slow — e-commerce verification, VQA, document understanding). Bi-encoders encode image/text independently with no cross-attention, compared later via cosine/dot-product/Euclidean (fast, scalable — CLIP-style retrieval).
- **Fusion mechanisms**: Early fusion (combine at input), Late fusion (combine at output), Intermediate/cross-modal fusion (combine mid-pipeline via cross-attention layers).
  - How: cross-attention lets one modality's tokens attend to the other's — the general mechanism beneath most fusion strategies.
- **Multimodal RAG system**: Separate retrievers (text + image embedding models) fetch relevant multimodal content from a shared/aligned vector space; an LLM generates grounded output from retrieved multimodal context. Avoids training a VLM from scratch.
  - How: user query (text/image/both) → embed via modality-specific models → search multimodal vector DB → consolidate retrieved text+image results → LLM generates grounded response → return.
- **VLM vs. Multimodal GenAI system decision**: Use VLM for tightly-coupled vision-text tasks needing low-latency single-model inference (captioning, VQA, retrieval, edge/mobile). Use multimodal GenAI system when combining modalities beyond vision+text, needing retrieval-augmented reasoning, tool use/agentic orchestration, or a dynamic/updatable knowledge base.
- **Output-based classification of multimodal systems**: Text-to-image, Image-to-text, Text+image-to-image, Text-to-specs+image, Text-to-SQL, Text-to-code — a design axis for choosing models/architecture by desired output type.

## Key Concepts
- **Multimodal vector embedding**: Each modality (text, image, audio) embedded by its own encoder (CLIP for images, Sentence Transformers for text, Whisper for audio) into a shared or aligned latent space so cross-modal similarity is computable.
- **Multimodal vector database**: Stores embeddings from multiple modalities with metadata (e.g., "type": "text"/"image"), enabling filtered ANN retrieval across modalities; examples — Qdrant, Weaviate, Pinecone, Milvus, Chroma.
- **Qdrant collection**: Labeled group of points sharing fixed vector dimensionality and similarity metric; supports "named vectors" for multiple vector types per point.
- **Point / Point ID / Payload**: A point = unique ID + one or more vectors + optional JSON payload (metadata) enabling filtered/hybrid search.
- **Segments (appendable vs. non-appendable)**: Appendable segments optimize for fast writes; non-appendable optimize for static, read-heavy data.
- **Storage models**: In-memory (max speed, RAM-resident) vs. memory-mapped (disk-backed via OS page cache, balances speed/memory).
- **Vector index vs. payload index**: Vector index (e.g., HNSW) accelerates similarity search; payload index accelerates metadata filtering — combined, they enable hybrid search.
- **Hybrid search**: Retrieval combining vector similarity with structured payload filters (e.g., semantic match + color="blue").

## Mental Models
- Think of a VLM as one fused brain for vision+language; think of a multimodal GenAI system as a modular factory line of specialized retrievers/generators coordinated by an orchestrator.
- Use cross-encoder-style thinking ("let modalities talk to each other directly") when precision matters more than speed; use bi-encoder-style thinking ("embed once, compare later") when scale and latency matter.
- The single-collection-with-payload-filter vs. multiple-collections-with-global-indexing trade-off is the same simplicity-vs-accuracy trade-off seen everywhere in retrieval systems: fewer moving parts (easier ops, ~50% ANN match rate) vs. more moving parts (harder ops, up to ~98% match rate).
- Output-type classification (text-to-image, image-to-text, etc.) is a useful "what does this system produce" lens that complements the "what does this system take in" lens from Chapter 1.

## Anti-patterns
- **Assuming "VLM" and "multimodal GenAI system" are interchangeable**: VLMs are single end-to-end models for vision+text; multimodal GenAI systems are modular pipelines spanning arbitrary modalities with retrieval/orchestration — conflating them leads to wrong architecture choices.
- **Training a VLM from scratch for a product need**: requires millions/billions of aligned image-text pairs and massive GPU/TPU budgets; multimodal RAG (retrieval + existing pretrained embedding models) is almost always more practical.
- **Using a single payload-partitioned collection when near-exact recall is required**: ANN accuracy can drop to ~50% match rate vs. exact KNN when payload filtering isn't backed by localized/global indexing — unacceptable for high-stakes recommendation/analytics use cases.
- **Ignoring modality imbalance**: textual signal often dominates training, underutilizing visual information and degrading fine-grained visual tasks (object grounding, scene description).
- **Deploying cross-attention-heavy VLMs (UNITER, LXMERT) in low-latency/edge environments**: these architectures scale poorly at inference time.

## Code Examples
This chapter is architectural/conceptual with no runnable code; it defines Qdrant data-model concepts referenced in later implementation chapters:
```text
Point = { id: <unique>, vector(s): { "text_vec": [...], "image_vec": [...] }, payload: {"type": "image", "category": "electronics", "date": "2025-05-09"} }
Collection config: vector size + distance metric (cosine | dot | euclidean) per named vector
```
- **What it demonstrates**: The core Qdrant primitives (collection, point, vector, payload) that later chapters' multimodal RAG implementations build on directly.

## Reference Tables
| Feature | VLMs | Multimodal GenAI systems |
|---|---|---|
| Modalities supported | Vision + text only | Any modality (text, image, audio, video, tables) |
| Model structure | End-to-end unified transformer | Modular pipeline: separate retrievers + generators |
| Typical use case | Captioning, VQA, retrieval | Multimodal chat, document analysis, RAG, complex workflows |
| Data sources | Pretrained image-text pairs | Integrates DBs, APIs, tools, memory |
| Retrieval layer | Not always present | Integral part of architecture |
| Flexibility | Moderate | High |
| Orchestration (LangChain etc.) | Rare | Common |
| Scalability | Limited by model size | Scalable via retrieval + modularity |

| Output type | Inputs required | Output generated | Use case category |
|---|---|---|---|
| Text-to-image | Text | Image | Design, marketing, creative AI |
| Image-to-text | Image | Text | Accessibility, search, indexing |
| Text+image-to-image | Text + image | Image | Guided editing, personalization |
| Text-to-specs+image | Text | Structured output + image | Design automation, engineering |
| Text-to-SQL | Text | SQL query | Analytics, BI, data search |
| Text-to-code | Text | Code snippet | Development, automation |

| Strategy | ANN accuracy vs. exact KNN | Operational cost |
|---|---|---|
| Single collection, payload-partitioned | ~50% match rate | Low (one collection) |
| Multiple collections, global indexing per collection | ~98% match rate | High (many collections to maintain) |

## Worked Example
Real-world comparison: answering a question from a product manual with text + figures. A VLM can caption an image or answer a simple visual question ("what color is the switch?"). A multimodal GenAI system instead: parses the PDF, extracts diagrams, runs OCR on embedded images, retrieves relevant product specs from a vector DB, and uses an LLM to synthesize a full answer ("To activate power-saving mode, press the green switch shown in Figure 2.3. The switch controls the secondary circuit.") — demonstrating the multimodal GenAI system's end-to-end capability across text, image, layout, and logic that a single VLM cannot match.

## Key Takeaways
1. VLMs excel at tightly-coupled, single-model vision-language tasks; multimodal GenAI systems excel at open-ended, multi-modality, tool-using, retrieval-grounded workflows.
2. Multimodal RAG (separate embedding models per modality + shared vector DB + LLM generation) is the practical alternative to training a VLM from scratch.
3. Vector database design choices (single collection + payload filter vs. multiple collections + global index) trade operational simplicity against retrieval accuracy — choose based on accuracy requirements.
4. Classifying a system by its output type (text-to-image, image-to-text, text-to-SQL, etc.) clarifies required inputs, model choice, and architecture pattern before building.
5. Cross-encoder fusion (joint attention) and bi-encoder fusion (independent embed-then-compare) represent the same speed/accuracy trade-off seen in Chapter 1's reranking discussion, now applied to vision-language.

## Connects To
- **Ch 1**: Extends the bi-encoder/cross-encoder and vector database concepts from Chapter 1 into the vision-language and multimodal embedding domain.
- **Ch 3-4**: The VLM vs. multimodal GenAI system distinction directly motivates the choice between implementing local vs. API-based unimodal systems next.
- **Ch 8**: The multimodal RAG architecture introduced here (separate text/image retrievers + shared vector DB + LLM generation) is implemented fully in the multimodal RAG build chapter.
- **Ch 14-15**: The text-to-SQL output classification here is the conceptual seed for the dedicated text-to-SQL and agentic text-to-SQL chapters.
- **Ch 16**: Image-to-text classification connects directly to the OCR/text-extraction-from-images chapter.
