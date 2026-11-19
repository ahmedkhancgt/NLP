# Chapter 3: Implementing Unimodal Local GenAI System

## Core Idea
A fully private, offline RAG system can be built with local tools (Ollama for LLM inference, Chroma/Faiss for vector storage, LangChain for orchestration), and most real-world RAG failures trace back to seven specific, named failure points that each have a concrete mitigation.

## Frameworks Introduced
- **GPU-need decision framework**: GPU needed for large LLMs (billions of params), low-latency real-time apps, batch/high-throughput, fine-tuning, embedding large document sets. CPU acceptable for small/quantized models (TinyLlama, DistilGPT2), non-real-time batch jobs, edge/offline hardware-constrained deployments, prototyping.
- **Local vs. cloud GPU economics**: Cloud GPUs (AWS/GCP/Azure, A100/H100) bill per-hour and escalate fast for continuous workloads; local GPU has higher upfront cost but lower marginal cost (electricity/maintenance) and better data governance for privacy-sensitive/regulated environments.
- **Seven RAG Failure Points** (from "Seven Failure Points When Engineering a RAG System"): Missing content, Missed top-ranked documents, Not in context, Not extracted, Wrong format, Incorrect specificity, Incomplete answers — each with a specific engineering mitigation.
  - When to use: as a runtime-validation checklist; RAG failures cannot be fully predicted at design time and must be diagnosed and patched post-deployment.
- **Hybrid retrieval (BM25 + vector)**: Combine sparse keyword matching (BM25 — exact terms, proper nouns, rare words) with dense semantic vector search via a HybridRetriever/EnsembleRetriever to improve recall and precision together.
- **ReAct prompting**: Structure the prompt to force explicit reasoning steps before the final answer, improving transparency, traceability, and synthesis quality from retrieved context.

## Key Concepts
- **Ollama**: Local LLM runtime/manager (pull, run, serve models like Llama3, Mistral) exposing a REST API at `http://localhost:11434`; alternatives — LM Studio, GPT4All, Text Generation Web UI, Unsloth, AutoGPTQ+transformers.
- **Quantization**: Reducing model precision (4-bit/8-bit) to shrink memory footprint and enable CPU-only or lower-VRAM inference.
- **Chunking**: Splitting documents into overlapping segments (`chunk_size`, `chunk_overlap`) for retrieval granularity; strategies — RecursiveCharacterTextSplitter (default), CharacterTextSplitter, TokenTextSplitter, SentenceTransformersTextSplitter, NLTKTextSplitter, SpacyTextSplitter, MarkdownHeaderTextSplitter.
- **ConversationBufferMemory**: LangChain memory module storing full chat history as a string buffer so the LLM can resolve follow-up/multi-turn references.
- **ConversationalRetrievalChain**: LangChain abstraction unifying LLM + retriever + memory + prompt into one multi-turn RAG chain that also returns source documents for citation.
- **Prompt size overflow**: Combined prompt (query + context + instructions + memory) exceeds the model's token limit, causing truncation/errors; mitigated via smaller chunks, memory truncation, token-aware splitters, or compression retrievers.
- **Retriever zoo**: ContextualCompressionRetriever (LLM-summarizes before passing on), MultiQueryRetriever (LLM rephrases query into variants), ParentDocumentRetriever (small chunks for search, larger parent returned), SelfQueryRetriever (LLM-generated metadata filters), TimeWeightedVectorStoreRetriever (recency-biased), EnsembleRetriever (weighted combination of retrievers).

## Mental Models
- Treat local GPU vs. cloud GPU as a build-vs-rent decision: rent (cloud) for spiky/short-term workloads, build (local) for continuous, privacy-sensitive, long-term workloads.
- Think of the seven RAG failure points as a diagnostic checklist to run against any underperforming RAG system — each failure point maps to one pipeline stage (retrieval breadth, context packing, extraction, formatting, specificity, completeness).
- Use hybrid search as "two independent judges voting" — BM25 catches exact/rare-term matches that embeddings miss; vectors catch paraphrase/semantic matches that BM25 misses.
- A RAG pipeline's folder structure (source_docs/, embeddings/, vectorstore/, retriever/, llm/, orchestrator/) mirrors its logical pipeline stages — modularity at the code level enables swapping any one stage without rewriting the rest.

## Anti-patterns
- **Fabricated answers with no fallback**: without an explicit "say I don't know" instruction and a confidence/similarity threshold, an LLM will hallucinate plausible-sounding answers when context is insufficient (Missing content failure).
- **Fixed low top-k with no reformulation**: relevant documents can be missed at low k; relying on a single query embedding without MultiQueryRetriever or increased k misses relevant content (Missed top-ranked documents failure).
- **Ignoring prompt/token budget**: stuffing too many/too-large chunks into context causes prompt overflow, truncation, or silently dropped relevant chunks (Not in context failure).
- **No formatting instructions when structured output is needed**: LLMs will ignore implicit formatting expectations unless explicitly instructed or passed through output parsers (Wrong format failure).
- **Over-provisioning GPU for small/quantized models**: Ollama can run quantized 7B models well on CPU; buying/renting GPU for this is wasted cost.

## Code Examples
```python
loader = PyPDFLoader("data/source_docs/ai_education_article.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```
- **What it demonstrates**: Standard PDF loading + recursive chunking setup for a local RAG pipeline.

```python
embedding_model = OllamaEmbeddings(model="mistral")
vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="db")
```
- **What it demonstrates**: Generating embeddings locally via Ollama and persisting them to a Chroma vector store for reuse.

```python
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
hybrid_retriever = HybridRetriever(vectorstore=vectorstore, bm25_retriever=bm25_retriever)
```
- **What it demonstrates**: Combining sparse (BM25) and dense (vector) retrieval into one hybrid retriever.

```python
react_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant using the ReAct (Reasoning + Acting) technique.
Break down the user query into reasoning steps and retrieve relevant information accordingly.
Question: {question}
Relevant Context:
{context}
First, list your reasoning steps clearly.
Then, provide a final answer based on those steps and the retrieved context.
Reasoning Steps:
1.
"""
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=hybrid_retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": react_prompt}
)
```
- **What it demonstrates**: Wiring a ReAct-style reasoning prompt into a full conversational RAG chain with memory and source-document citation.

## Reference Tables
| Model size | CPU RAM (quantized) | GPU VRAM (full precision) | Example models |
|---|---|---|---|
| 3-7B | 8-16 GB | 6-8 GB | Mistral 7B, Llama 2 7B |
| 13B | 16-24 GB | 12-16 GB | Llama 2 13B |
| 30B+ | 32-64 GB | 24+ GB | Llama 2 33B, Mixtral 8x7B |

| Failure point | Problem | Mitigation |
|---|---|---|
| Missing content | LLM fabricates answer when info absent | "I don't know" fallback + similarity confidence threshold |
| Missed top-ranked documents | Relevant docs exist but not in top-k | MultiQueryRetriever, raise k |
| Not in context | Relevant chunks retrieved but dropped by token limit | ContextualCompressionRetriever, summarize context |
| Not extracted | LLM fails to extract answer from context | Clearer/step-by-step prompts, fine-tuned extractor |
| Wrong format | Output ignores formatting instructions | Explicit format cues, LangChain Output Parsers |
| Incorrect specificity | Output too generic/too detailed | User-settable detail_level/response_style prompt variable |
| Incomplete answers | Answer omits key facts from multiple sources | Answer-merging logic, checklist-style reasoning |

## Worked Example
End-to-end build: (1) generate a synthetic source document by prompting a local `llama3.2:3b-instruct-fp16` model via Ollama's REST API and saving to PDF with reportlab; (2) load and chunk that PDF (`chunk_size=500`, `chunk_overlap=50`); (3) embed chunks with `OllamaEmbeddings(model="mistral")` into a persisted Chroma store; (4) build a `hybrid_retriever` combining `BM25Retriever` and the Chroma vector retriever; (5) attach `ConversationBufferMemory` and a ReAct `PromptTemplate`; (6) assemble a `ConversationalRetrievalChain` with the local Mistral LLM (`temperature=0.2`); (7) run an interactive terminal chat loop that prints answers plus source citations. The result is a fully offline, privacy-preserving document-QA bot.

## Key Takeaways
1. Local deployment (Ollama + Chroma/Faiss + LangChain) gives full data privacy and cost control, at the expense of needing to manage hardware/quantization trade-offs yourself.
2. GPU need is workload-dependent, not automatic — quantized small models run fine on CPU for non-real-time use cases.
3. Hybrid retrieval (BM25 + dense vectors) outperforms either method alone across varied query types.
4. The seven RAG failure points give a concrete, checklist-style way to debug a RAG system rather than treating failures as a monolithic "it's not working."
5. ReAct-style prompting plus returning source documents builds transparency and citation support directly into the generation step.
6. Chunking strategy choice (recursive vs. token-aware vs. semantic vs. markdown-header) materially affects retrieval quality and should match document structure.

## Connects To
- **Ch 1**: Implements the RAG architecture, vector database, and reranking/retrieval concepts introduced conceptually there.
- **Ch 4**: Directly contrasted next — same RAG architecture rebuilt using API-based (OpenAI) models instead of local Ollama models.
- **Ch 9-10**: The seven failure points and hybrid retrieval here are the baseline that reranking (Ch9) and retrieval optimization (Ch10) chapters build on and refine.
- **Ch 18**: RAG failure diagnosis here is a precursor to the formal LLMOps/evaluation techniques covered later.
