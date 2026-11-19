# Implementation Patterns

## Chunking Pattern
**When to use**: Before embedding or feeding long documents into an LLM or vector database — raw long text is inefficient and misses natural semantic boundaries.
**How**: Choose fixed-size (simple, fast, ignores boundaries), sentence-splitting (preserves sentence meaning, up to a max-token budget), recursive (hierarchical halving until under a token threshold, good for long/complex documents), or hybrid (fixed-size + rule-based boundary awareness). Use `RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...)` in LangChain as the default production choice. Tune chunk_size (800-4000 tokens seen across chapters) and overlap (100-120 tokens typical) to the task.
**Trade-offs**: More overlap improves cross-boundary context capture but increases compute cost and storage. Fixed-size is fastest but can split mid-sentence; recursive/hybrid give better semantic boundaries at higher implementation complexity.

## Embedding Generation Pattern
**When to use**: Converting text into numeric vectors for similarity search, clustering, classification, or as LLM-adjacent context retrieval.
**How**: Select an embedding model (BERT/RoBERTa/XLNet/T5 via `transformers`, or domain-tuned Instructor embeddings via `HuggingFaceInstructEmbeddings`) → tokenize text → pass through model → extract `last_hidden_state` → reshape to a usable vector (1D or 2D depending on downstream use). For production RAG, prefer `sentence-transformers` or Instructor embeddings over raw BERT pooling.
**Trade-offs**: Context-aware embeddings (BERT, USE) are better for sentiment/QA but costlier than word-level embeddings (word2vec, GloVe, FastText), which are faster but miss polysemy/context.

## Embedding Similarity / Retrieval Pattern
**When to use**: Finding the most semantically similar documents/passages to a query (semantic search, RAG retrieval, recommendation).
**How**: Embed query with the same model used for the corpus → normalize vectors if using L2/Euclidean distance → compute similarity (cosine similarity most common for NLP; Euclidean/L2 for Faiss `IndexFlatL2`; dot product for raw alignment) → retrieve top-k.
**Trade-offs**: Cosine similarity ignores magnitude (good for text where length varies); Euclidean distance is sensitive to magnitude and requires normalization; dot product is cheapest to compute but conflates magnitude and direction.

## Vector Store Indexing and Query Pattern
**When to use**: Persisting embeddings for repeated retrieval without re-embedding on every query; building the retrieval half of a RAG system.
**How**: Build/load a vector store (Faiss `IndexFlatL2` for in-memory/simple exact search; Chroma `PersistentClient` + collection for metadata-aware persistent storage) → add documents + embeddings (+ optional metadata) → query with top-k (`k` parameter) → optionally filter by metadata.
**Trade-offs**: Faiss is lightweight and fast but lacks built-in metadata filtering/persistence ergonomics; Chroma adds metadata filtering and persistence at the cost of an extra dependency/service layer; cloud vector DBs (Pinecone, Weaviate, Milvus) add scalability and managed infrastructure at higher operational cost.

## RAG (Retrieval-Augmented Generation) Pattern
**When to use**: Grounding LLM responses in an external, current, or proprietary knowledge base rather than relying solely on parametric (trained-in) knowledge.
**How**: (1) Embed and index a document corpus in a vector store. (2) On query: embed the query, retrieve top-k relevant chunks. (3) Inject retrieved chunks as context into a prompt template. (4) Pass to LLM via a chain (e.g., LangChain `RetrievalQA.from_chain_type(llm=..., chain_type='stuff', retriever=vectordb.as_retriever(search_kwargs={'k': N}))`).
**Trade-offs**: Retrieval quality (chunking, embedding choice, k) bounds generation quality — a well-grounded prompt still needs good retrieval. Higher k gives more context but risks diluting relevance and exceeding context window.

## LangChain Chain Composition Pattern (LCEL)
**When to use**: Building multi-step LLM workflows (prompt → model → parse) that need to scale from prototype to production without rewrites.
**How**: Compose `prompt | model | output_parser` declaratively; swap LLM providers without changing chain structure (model-agnostic). For document-heavy tasks exceeding context windows, use Map-Reduce (`MapReduceDocumentsChain` + `ReduceDocumentsChain` + `StuffDocumentsChain`): map step summarizes/processes each chunk, reduce step combines results.
**Trade-offs**: LCEL's declarative style eases production scaling (streaming, parallelism) but adds a learning curve; deeply nested chains become hard to debug — keep chains modular.

## Conversational Retrieval Pattern
**When to use**: Multi-turn chat applications where follow-up questions depend on prior conversation context.
**How**: Maintain chat history → before retrieval, rephrase the follow-up question into a standalone question using a `CONDENSE_QUESTION_PROMPT` (chat_history + question → standalone question) → run standard RAG retrieval on the standalone question.
**Trade-offs**: Adds an extra LLM call (question condensation) per turn, increasing latency/cost, but significantly improves retrieval accuracy for pronoun-heavy or context-dependent follow-ups.

## PEFT / Parameter-Efficient Fine-Tuning Pattern
**When to use**: Adapting a large pretrained LLM to a specific task/domain when full fine-tuning is too expensive (compute, storage, risk of overfitting).
**How**: Choose LoRA (freeze base weights, train small low-rank adapter matrices injected into attention layers) or QLoRA (LoRA + 4-bit quantization of the base model) depending on available GPU memory. Use `bitsandbytes` + `BitsAndBytesConfig` (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`) for quantized loading via `transformers`.
**Trade-offs**: PEFT dramatically cuts trainable parameters (up to 10,000x) and GPU memory (up to 3x) vs. full fine-tuning, with only slight performance trade-offs; QLoRA pushes memory savings further at a small additional precision cost.

## Prompt Engineering Selection Pattern
**When to use**: Steering LLM output without retraining — the fastest, cheapest adaptation lever.
**How**: Match technique to task — explicit prompting for precise/constrained output; few-shot for format consistency; chain-of-thought for multi-step reasoning; self-consistency sampling (multiple CoT paths, pick most frequent answer) for higher-stakes reasoning tasks; tree-of-thought for tasks needing exploration/backtracking (e.g., constraint satisfaction); sequential prompting for naturally staged tasks (e.g., multi-step itinerary planning).
**Trade-offs**: More sophisticated techniques (ToT, self-consistency) improve accuracy on hard reasoning tasks but multiply inference calls/cost; simple explicit/few-shot prompting is cheap but may not suffice for complex multi-step reasoning.

## Hugging Face Model Serving / Deployment Pattern
**When to use**: Serving an open-source LLM (Llama, Falcon, Mistral, GPT-NeoX, etc.) as a hosted inference endpoint rather than running locally.
**How**: Get the TGI image URI (`get_huggingface_llm_image_uri(backend="huggingface", region=region)`) → configure `HuggingFaceModel` with `hf_model_id`, `hf_task`, `SM_NUM_GPUS` (tensor parallelism for large models), optionally `HF_MODEL_QUANTIZE=true` → `model.deploy(instance_type=..., initial_instance_count=...)` → `predictor.predict({"inputs": ..., "parameters": {...}})`.
**Trade-offs**: Managed hosting (SageMaker/Azure ML/Vertex AI) trades some cost/control for operational simplicity, autoscaling, and integration with cloud storage/compute; self-hosting (local or on-prem) gives more control/transparency but requires managing GPU infrastructure directly.

## Secure Embedding / Data Privacy Pattern
**When to use**: Embedding or training on data containing PII or sensitive information.
**How**: Pre-process (anonymize PII, encrypt, tokenize) → choose embedding method mindful of exposure (embed locally for sensitive data, apply differential privacy, consider federated learning) → restrict access during training (access control, audit logging) → secure storage/deployment (encryption, limited deployment scope, compliance with GDPR/HIPAA).
**Trade-offs**: Stronger privacy techniques (differential privacy, federated learning, local-only embedding) add engineering complexity and can slightly reduce model utility/accuracy in exchange for materially lower privacy/legal risk.
