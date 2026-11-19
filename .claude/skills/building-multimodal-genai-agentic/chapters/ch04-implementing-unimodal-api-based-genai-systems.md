# Chapter 4: Implementing Unimodal API-based GenAI Systems

## Core Idea
Moving from local (Ollama) to cloud API-based (OpenAI) GenAI unlocks more powerful models and OpenAI's agentic ecosystem (Responses API, Agents SDK, Operator, Codex), and a modular multi-document RAG system adds metadata tagging and topic-filtered hybrid retrieval to scale beyond single-document QA.

## Frameworks Introduced
- **OpenAI model selection framework**: Choose by performance need (GPT-4/o-series for high reasoning), budget (GPT-3.5/GPT-4o-mini for cost), and use case (GPT family for text, o-series for reasoning, DALL-E for images, Whisper for transcription, Sora for video).
- **OpenAI agentic ecosystem**: Responses API (single-call reasoning + tool invocation + stateful context, successor to Assistants API), Agents SDK (agents/tools/workflows/guardrails/handoffs primitives in Python/JS), Operator (autonomous browser-task agent), Codex (agentic software-development system).
  - When to use: Responses API for integrated reasoning+action loops; Agents SDK for multi-agent orchestration with guardrails; Operator for GUI/web-based task automation; Codex for code generation/debugging/testing agents.
- **Metadata-filtered hybrid RAG architecture**: Tag each chunk with source metadata at ingestion; at query time, filter candidate chunks by topic/metadata *before* building the BM25+vector ensemble retriever, so multi-document queries stay source-specific.
  - How: `filter_chunks_by_topic()` narrows chunks by keyword-to-source mapping → build `BM25Retriever` + vector retriever on filtered set → combine via `EnsembleRetriever` with weights.
- **Modular RAG codebase pattern**: Separate files per responsibility — config.py (constants/keys), embedder.py (embedding model), db_handler.py (vector store persistence), metadata_schema.py (tagging), pdf_parser.py (load+chunk), hybrid_search.py (retriever), generate.py (LLM), react_prompt.py (prompt), conversation_buffer.py (memory), rag_chain.py (orchestration), main.py (UI loop).

## Key Concepts
- **OpenAI company/model history**: Founded 2015 (non-profit → capped-profit); GPT-2 (2019) → GPT-3 (2020) → GPT-3.5/GPT-4 → GPT-4o/4.1 family → o-series (reasoning-focused: o1, o3, o4-mini).
- **Core OpenAI API endpoints**: `/v1/chat/completions` (text gen), `/v1/embeddings`, `/v1/images/generations` (DALL-E), `/v1/audio/transcriptions` (Whisper), `/v1/moderations`, `/v1/fine-tunes`, `/v1/models`.
- **Single-tenant vs. multi-tenant RAG**: Single-tenant = one shared vector store/index for all users. Multi-tenant = per-tenant isolated vector store or namespace, with tenant identity routing the correct store/memory at query time.
- **FAISS local persistence pattern**: Check for existing `index.faiss`/`index.pkl` on disk; load if present (`FAISS.load_local`), else build fresh (`FAISS.from_documents`) and `save_local`.
- **EnsembleRetriever weighting**: Combines multiple retrievers (e.g., BM25 + vector) with explicit weights (e.g., `[0.5, 0.5]`) rather than LangChain's generic HybridRetriever.

## Mental Models
- Think of the Responses API as a merge of "chat" and "tool execution" into one stateful call — a successor pattern to the older Assistants API's thread-based model.
- Treat metadata tagging as the lever that turns a single shared vector store into an effective multi-document, source-aware system without needing separate vector stores per document.
- Single-tenant vs. multi-tenant is the same "shared vs. isolated resource" trade-off seen in the single-collection vs. multiple-collection vector DB decision from Chapter 2 — isolation costs operational complexity but buys correctness/security guarantees.
- API-based (OpenAI) vs. local (Ollama, Ch3) is a direct architecture fork: same RAG skeleton (chunk → embed → retrieve → generate), different trust/cost/control trade-offs.

## Anti-patterns
- **Treating all documents as one undifferentiated corpus in multi-document RAG**: without metadata tagging and topic-based filtering, a hybrid retriever can surface chunks from the wrong source document, degrading answer specificity.
- **Using the legacy Assistants API for new agentic development**: OpenAI is phasing it out in favor of the Responses API and Agents SDK; building new agentic systems on the legacy API accrues migration debt.
- **Deploying a single-tenant RAG architecture for multi-customer/enterprise use**: without per-tenant vector store isolation, one tenant's data can leak into another's retrieval results — a security/compliance risk.
- **Hardcoding API keys directly in scripts**: `config.py` centralizing `OPENAI_API_KEY` is shown for clarity but production systems should use environment variables/secrets managers instead.

## Code Examples
```python
#config.py
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = "your-api-key"
VECTOR_DB_PATH = "db"
SOURCE_DOCS = [
    "data/source_docs/ai_education_article.pdf",
    "data/source_docs/how_blockchain_is_transforming_financial_services.pdf"
]
```
- **What it demonstrates**: Centralized configuration for a modular, multi-document RAG system.

```python
#db_handler.py
def get_vectorstore(documents):
    embedding_model = get_embedding_model()
    index_file = Path(VECTOR_DB_PATH) / "index.faiss"
    store_file = Path(VECTOR_DB_PATH) / "index.pkl"
    if index_file.exists() and store_file.exists():
        return FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    vectorstore = FAISS.from_documents(documents, embedding=embedding_model)
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore
```
- **What it demonstrates**: FAISS persistence pattern — load existing index or build and save a new one.

```python
#hybrid_search.py
def filter_chunks_by_topic(chunks, topic):
    topic = topic.lower()
    if "blockchain" in topic or "crypto" in topic:
        return [c for c in chunks if "blockchain" in c.metadata.get("source", "").lower()]
    elif "education" in topic or "ai" in topic or "artificial intelligence" in topic:
        return [c for c in chunks if "education" in c.metadata.get("source", "").lower()]
    else:
        return chunks

def get_hybrid_retriever(chunks, vectorstore, topic=None):
    filtered_chunks = filter_chunks_by_topic(chunks, topic)
    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = 4
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
```
- **What it demonstrates**: Topic-aware metadata filtering applied *before* hybrid retrieval, enabling source-specific multi-document answers.

```python
from agents import Agent, Runner
agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion")
print(result.final_output)
```
- **What it demonstrates**: Minimal OpenAI Agents SDK usage — instantiate an agent and run it synchronously.

## Reference Tables
| Category | Endpoint(s) | Description |
|---|---|---|
| Text generation | /v1/completions, /v1/chat/completions | Generate/continue text |
| Editing | /v1/edits | Modify existing text or code |
| Embeddings | /v1/embeddings | Generate vector representations |
| Image generation | /v1/images/generations, /edits, /variations | Create/edit images via DALL-E |
| Audio processing | /v1/audio/transcriptions, /translations | Speech-to-text, translation |
| Moderation | /v1/moderations | Detect harmful/sensitive content |
| File handling | /v1/files | Upload/manage fine-tuning files |
| Fine-tuning | /v1/fine-tunes | Train custom models |
| Model listing | /v1/models | List available models |

| Agentic component | Purpose |
|---|---|
| Responses API | Single-call reasoning + tool invocation + stateful context (successor to Assistants API) |
| Agents SDK | Agent/tool/workflow/guardrail/handoff primitives (Python/JS) |
| Operator | Autonomous browser/GUI task execution |
| Codex | Agentic code generation, debugging, testing |

## Worked Example
Building a metadata-aware multi-document RAG system: generate 5 synthetic PDFs via local Ollama (reused from Ch3) across different topics (renewable energy, AGI, blockchain, mental health, climate) → modular pipeline loads/chunks all PDFs and tags each chunk with its source filename → builds/persists a FAISS vector store using OpenAI `text-embedding-3-small` → at query time, `get_hybrid_retriever` filters chunks by topic keyword match before combining BM25 + vector retrieval → `ConversationalRetrievalChain` with GPT-4o and a ReAct prompt generates the answer. Result: asking "how does blockchain improve payments?" pulls only from the blockchain PDF, while "how can AI personalize learning?" pulls only from the AI/education PDF — demonstrating correct source isolation via metadata filtering, even within a single shared vector store.

## Key Takeaways
1. API-based (OpenAI) systems trade data-locality/privacy for access to more powerful models and a maturing agentic ecosystem (Responses API, Agents SDK, Operator, Codex).
2. Multi-document RAG requires explicit metadata tagging and topic-aware filtering before retrieval — otherwise a shared vector store blends unrelated sources.
3. FAISS persistence (check-then-load-or-build) avoids redundant re-embedding across runs.
4. OpenAI is consolidating its agentic stack around the Responses API and Agents SDK, deprecating the legacy Assistants API.
5. Single-tenant RAG architectures need explicit redesign (vector store separation, memory isolation, request-level tenant routing) before being safe for multi-tenant/enterprise deployment.

## Connects To
- **Ch 3**: Directly parallels the local Ollama+Chroma RAG build, swapping in OpenAI API models (GPT-4o, text-embedding-3-small) and FAISS, while reusing the ReAct prompt and hybrid-retrieval pattern.
- **Ch 5**: The OpenAI agentic ecosystem (Responses API, Agents SDK) introduced here is the toolset used to build the HITL agentic GenAI systems next.
- **Ch 6**: Multi-document querying here is a precursor to the two/multi-stage GenAI system architectures.
- **Ch 9-10**: The hybrid/topic-filtered retrieval pattern here is refined further by dedicated reranking and retrieval-optimization chapters.
