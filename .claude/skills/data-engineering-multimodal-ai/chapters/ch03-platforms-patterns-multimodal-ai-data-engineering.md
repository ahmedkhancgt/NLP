# Chapter 3: Platforms and Patterns for Multimodal AI Data Engineering

## Core Idea
Multimodal systems can be built on three enduring platform archetypes — managed AI platforms, composable data architectures, and real-time streaming infrastructure — but regardless of platform choice, every production system must implement the same three core patterns: representation, temporal state assembly, and selective context retrieval, illustrated through the fictional investment platform Revault.

## Frameworks Introduced
- **Multimodal Context Intelligence (a.k.a. Context Engineering)**: the data architecture that transforms unstructured/structured data into synchronized, retrievable, auditable session state models can reason over in real time.
  - When to use: any agentic/multimodal system needing a consistent, aligned, provenance-preserving view across modalities.
  - How: transform signals into consistent representation → align by time/entity → retrieve only relevant state for a decision → preserve structure/provenance for audit.
- **Three Platform Archetypes**: Managed AI Platforms (e.g., Amazon Bedrock, Vertex AI/Gemini Enterprise, Azure AI Foundry) abstract complexity behind a unified API; Composable Data Architectures (e.g., Databricks Lakehouse AI) expose every pipeline stage for full control; Real-Time Streaming Infrastructure (Kafka/Flink ecosystem) handles temporal alignment/state materialization alongside either.
  - When to use: choose based on "where should complexity live?" — managed = speed over control; composable = control/flexibility over speed; streaming = needed whenever in-session, real-time decisions are required regardless of the other choice.
- **Cost and Complexity Framework (Table 3-2)**: cost distributes differently per archetype — managed = pay-per-call (predictable at low volume, spikes at high throughput); composable = self-managed, explicit cost levers (compute/storage/embedding/indexing priced separately); streaming = always-on cost (continuous resource consumption for low latency).
  - When to use: pick architecture based on volume, latency, and governance requirements, not technical preference alone.
- **Three Core System Patterns**: (1) Representation — transform raw artifacts into typed, schema-conformant objects with provenance; (2) Temporal State Assembly — incrementally update session state as asynchronous events arrive, never recompute from scratch; (3) Selective Context Retrieval — construct a bounded, reproducible "decision bundle" from structured queries, semantic search, and metadata filters.
  - When to use: as the universal checklist for any production multimodal context layer, independent of platform archetype.

## Key Concepts
- **Representation Contract**: the schema every typed object (e.g., IdentityObject, TranscriptFeatures, RiskObject) must conform to regardless of source modality, enabling downstream joins.
- **Hot/Cold Context Store Split**: hot store serves low-latency in-session decisions; cold store preserves full timeline/provenance for audit and replay.
- **Event-Time State Materialization**: an append-only, session_id-keyed event log that lets outcomes be recomputed/replayed if policy or models change.
- **Decision Bundle**: the bounded, reproducible, policy-safe slice of session state (authoritative state + discrepancy flags + ranked evidence + audit trace + constraints like token budget/PII mode) handed to a reasoning model.
- **Bundle Hash**: a guarantee that identical inputs always produce the same decision context, making retrieval auditable.
- **Context Budgeting**: explicitly constraining the assembled bundle to fit model token limits and latency targets.
- **Watermark Tracking / Late-Arrival Policy**: mechanisms for handling out-of-order or lagging events in temporal assembly.
- **Context Coherence Across Agent Turns**: ensuring agent-orchestrated multistep reasoning doesn't operate on stale or contradictory state as tool calls update the context layer.

## Mental Models
- The central question when picking a platform archetype is "**where should complexity live**" — it never disappears, it's either absorbed by the platform, owned by engineering, or paid for continuously by streaming infra.
- **"Multimodal systems fail not because models are weak but because representation is incoherent"** — always diagnose representation quality before blaming retrieval or the model.
- Temporal assembly: think of session state as **continuity, not recomputation** — each event expands existing state rather than triggering a full rebuild; recomputing from scratch introduces latency, race conditions, and policy drift.
- Selective retrieval: think of the full session state as **too large to ever hand to a model whole** — always carve a bounded, ranked, redacted, reproducible bundle instead of passing everything.
- "**The bottleneck is almost never retrieval; it is almost always representation**" — invest in parsing/extraction quality before optimizing vector infrastructure.

## Anti-patterns
- **Recomputing session state from scratch on every new event** instead of incrementally updating it — introduces latency, race conditions, and policy drift.
- **Passing the entire session history into a reasoning model**: overwhelms the model, increases latency, dilutes signal with noise, and in regulated environments can produce coherent-looking but incorrect decisions.
- **Treating vector search as the differentiator**: ANN engine choice is commoditizing; what separates production systems from demos is hybrid retrieval (structured + semantic + metadata) with governance, not the index technology.
- **Skipping cost modeling until production**: embedding/indexing/query costs compound quietly (per-token, per-document, per-query) and re-embedding on model refresh multiplies cost — must be visible from the playbook stage.
- **Letting a parser silently drop or truncate data** (table structure, handwritten fields, multipage documents) — no embedding model downstream can recover semantics lost at extraction time.

## Code Examples
```json
// Stage 1 output: structured representation from a parsed KYC document
{
  "text": "extracted document text...",
  "entities": [
    {"type": "person_name", "text": "Laura Russo", "confidence": 0.98},
    {"type": "address", "text": "742 Elm Street...", "confidence": 0.95}
  ],
  "pages": 2,
  "mime_type": "application/pdf",
  "document_uri": "gs://bucket/path/to/original/document"
}
```
- **What it demonstrates**: the representation layer converting a raw binary artifact into a typed object with explicit confidence scores (uncertainty made explicit, not implicit).

```json
// Stage 2 output: embedding with metadata for indexing (create_embeddings.py)
{
  "id": "AHDL_jpeg_processed.json",
  "embedding": [0.021, -0.013],
  "metadata": {
    "document_type": "ID",
    "document_text": "ILLINOIS DRIVER'S LICENSE...",
    "person_name": "April Hensley"
  }
}
```
- **What it demonstrates**: pairing a 768-dim semantic vector with structured metadata so hybrid retrieval (semantic + metadata filter) is possible later.

```bash
# Playbook pipeline scripts
python create_embeddings.py        # Stage 2: text -> dense vectors with metadata
python vector_search_setup.py      # Stage 3: load vectors into ANN index
python search_documents.py         # Stage 4: NL query -> embedding -> top-k nearest neighbors
python test_person_search.py       # smoke test across all sample individuals
```
- **What it demonstrates**: the end-to-end parse → embed → index → retrieve pipeline, with a smoke test to catch upstream drift (e.g., if "James Williams's tax documents" doesn't surface his forms, something upstream broke).

## Reference Tables
**Three Platform Archetypes**
| Archetype | Example platforms | Trade-off |
|---|---|---|
| Managed AI Platform | Amazon Bedrock, Vertex AI/Gemini Enterprise, Azure AI Foundry | Simplicity at API boundary; surrender control inside the box; pay-per-call pricing |
| Composable Architecture | Databricks Lakehouse AI | Full control over parsing/embedding/retrieval/orchestration; owns reliability, cost modeling, governance |
| Real-Time Streaming Infra | Apache Kafka / Apache Flink ecosystem | Always-on cost; delivers continuously updated, replayable session state |

**Table 3-1 (partial, reconstructed): Composable architecture ownership**
| Layer | Engineering responsibility | Critical design decisions | Example (Databricks) |
|---|---|---|---|
| Ingestion | Land and track multimodal artifacts | Batch vs. streaming, metadata capture, idempotency | Auto Loader, Delta Bronze, Unity Catalog |

**Table 3-2 (reconstructed): Cost distribution across archetypes**
| Cost component | Managed platform | Composable architecture | Streaming infrastructure |
|---|---|---|---|
| Compute | Pay-per-call | Self-managed | Always-on processing |

## Worked Example
**Revault Serverless Document Retrieval Pipeline (playbook)**: Four-stage pipeline for selective KYC document retrieval on a managed stack (Google Cloud Document AI + Vertex AI, though AWS/Azure equivalents work identically). Stage 1 — layout-aware parsers (Document AI / Textract / Azure Document Intelligence) convert JPEGs (driver's licenses) and PDFs (tax forms, bank statements) for fictional individuals (April Hensley, Laura Russo, James Williams, Taylor Cox, Mitchell Allen) into structured JSON with typed entities and confidence scores. Stage 2 — a managed embedding service (Vertex AI text-embeddings, Bedrock Titan, Azure OpenAI) converts extracted text into 768-dim vectors with person/document-type metadata. Stage 3 — vectors are indexed into ANN infrastructure (Vertex Vector Search, OpenSearch Serverless, Azure AI Search, or Pinecone/Weaviate/Qdrant). Stage 4 — a natural-language query ("Laura's KYC documents") is embedded with the same model and searched via cosine similarity, returning ranked top-k results (e.g., Laura's driver's license 0.94, utility bill 0.89, Form 1040 0.87). The retrieval is purely semantic — no exact field matches or document ID filters — demonstrating that representation quality (Stage 1) determines whether embeddings (Stage 2) carry meaningful content for similarity search (Stage 4).

## Key Takeaways
1. Platform choice (managed/composable/streaming) is an economic decision as much as a technical one — match it to volume, latency, and governance requirements.
2. Every production multimodal context layer must implement representation, temporal state assembly, and selective context retrieval regardless of underlying platform.
3. Session state should be incrementally updated as events arrive (temporal coherence), never recomputed from scratch — this avoids race conditions and policy drift.
4. The decision bundle (bounded, ranked, redacted, reproducible) is what separates auditable multimodal systems from ones that silently feed too much or the wrong context to a reasoning model.
5. Representation quality is the ceiling on the whole system — invest there before optimizing vector infrastructure.
6. Observability must span all three patterns: extraction confidence (representation), event lag/watermarks (temporal assembly), and retrieval precision/token budget (selective retrieval).
7. Agentic orchestration introduces a feedback loop between context layer and agent tool calls that must be explicitly engineered for coherence across turns.

## Connects To
- **Ch 2**: Builds on vector-native storage and embedding/retrieval patterns, applying them within specific platform archetypes.
- **Ch 4**: Extends representation/parsing concerns into full ingestion and transformation pipeline design.
- **Ch 11**: The governance, audit trail, and compliance (GDPR/CCPA/AML/Bank Secrecy Act) concerns raised here are formalized further.
- **MCP / Agent frameworks**: the agentic orchestration and context flow section connects directly to agent-to-tool protocols introduced in Chapter 1.
