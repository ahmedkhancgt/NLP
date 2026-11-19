# Architecture & Implementation Patterns

## Two-Stage RAG (Retrieve + Rerank)
**When to use**: Whenever raw bi-encoder retrieval is noisy and generation quality is sensitive to irrelevant context.
**How**: Bi-encoder retrieves top-50/100 candidates (fast, no interaction) → cross-encoder or late-interaction model reranks (deep, token-level) → top-5/10 passed to LLM for generation.
**Trade-offs**: Adds latency and compute for the rerank stage; dramatically improves precision and reduces hallucination vs. single-stage retrieval. (Ch 1, 6, 9)

## Multimodal RAG Pipeline
**When to use**: Building systems that must retrieve and generate across text + image (or audio, tables) jointly.
**How**: Modality-specific embedding models (CLIP for image, text-embedding model for text) project into a shared/aligned latent space → unified or dual-collection vector DB → retrieve across modalities → LLM synthesizes grounded response from multimodal context.
**Trade-offs**: Avoids costly end-to-end VLM training; requires careful embedding alignment and dual-index management. (Ch 2, 7, 8)

## Bidirectional Multimodal Retrieval
**When to use**: Need text→image and image→text search from the same system.
**How**: Shared bi-encoder (CLIP) embeds both modalities into one latent space; separate per-modality collections (dual-index); query in one modality, search the other collection; L2-normalize all vectors.
**Trade-offs**: Simple and fast, but retrieval-only — needs an added generation layer for full RAG. (Ch 7)

## HITL Gating for Agentic GenAI
**When to use**: Safety-critical domains (legal, healthcare, compliance) where autonomous output needs human validation.
**How**: Agent produces output → explicit approval function (`human_approval_required()`) halts the pipeline → human approves/rejects → bounded retry loop (e.g., max 3 attempts) → graceful fallback message if still rejected.
**Trade-offs**: Adds latency and requires a human in the loop; essential safety valve against ungrounded/hallucinated high-stakes output. (Ch 5)

## Multi-Agent Pattern Composition (19-pattern catalog)
**When to use**: Any agentic system — pick patterns by task shape (Parallel for independent subtasks, Sequential/Loop for staged refinement, Router for conditional dispatch, Hierarchical/Supervisor-subordinate for delegated authority, Critic/validator for quality gating, Watchdog/recovery for fault tolerance).
**How**: Compose 2+ patterns per system (e.g., Sequential + HITL + Loop) via an orchestration framework (LangGraph StateGraph) with named nodes and conditional edges.
**Trade-offs**: More patterns = more architectural clarity but more coordination overhead; don't over-engineer a single-task system with unnecessary patterns. (Ch 5)

## Multi-Stage RAG with Grading
**When to use**: High-stakes domains where factual accuracy of retrieval and generation must be validated before surfacing to the user.
**How**: Insert LLM-based graders at checkpoints — retrieval relevance grader (binary_score yes/no) → generation → hallucination grader (groundedness check against facts) → answer quality grader (completeness/utility) — using a *different* LLM from the one generating, to avoid self-evaluation bias.
**Trade-offs**: Adds inference cost (multiple LLM calls) and latency; substantially reduces hallucination risk and enables feedback loops. (Ch 6, 8)

## Interaction-Depth-Matched Reranking (No/Late/Full)
**When to use**: Choosing a reranking strategy for a given candidate-set size and accuracy requirement.
**How**: No interaction (bi-encoder) for first-stage retrieval at scale → Late interaction (ColBERT/MaxSim) for a scalable middle ground with token-level precision → Full interaction (cross-encoder) reserved for final reranking of a small (≤100) candidate set.
**Trade-offs**: Full interaction is most accurate but cannot scale to large corpora; late interaction needs more storage (token-level vectors) but avoids full cross-attention cost. (Ch 1, 6, 9)

## Multi-Index + Vector Fusion for Hybrid Queries
**When to use**: Queries combining multiple modalities (image+text) or multiple retrieval sources (BM25+dense) that must be merged into one ranking.
**How**: Build separate modality/source-specific indexes → fuse query embeddings (average or concatenation) or fuse result scores (min-max/z-score normalization + Reciprocal Rank Fusion) → rerank the merged candidate set.
**Trade-offs**: Must keep fusion strategy identical between indexing and query time; weighted fusion (vs. naive averaging) lets you bias toward the more reliable modality/source. (Ch 9, 10)

## Adaptive Index Refresh
**When to use**: Any production retrieval system with a changing document corpus or periodically-upgraded embedding model.
**How**: Scheduled or event-driven reindexing function (`refresh_all_indexes()`) deletes and rebuilds collections from current folder/data-source contents; triggered manually, on a schedule, or by detected recall@k degradation.
**Trade-offs**: Without it, index staleness silently degrades both recall and precision; re-embedding on model upgrade avoids embedding-space incompatibility. (Ch 10)

## Voice-Bracketed RAG (STT/TTS Wrapper)
**When to use**: Adding voice input/output to an existing text-based RAG system.
**How**: STT (SpeechRecognition) transcribes voice to text at the entry point → unchanged RAG core (retrieve → generate) → TTS (pyttsx3) converts the final answer back to speech at the exit point.
**Trade-offs**: Keeps the RAG core modality-agnostic; voice introduces new failure modes (transcription errors) needing their own error handling. (Ch 11)

## Agentic Text-to-SQL with Architecture Decision-Making
**When to use**: Enterprise multi-database environments needing natural-language access without SQL expertise.
**How**: Single LangChain ReAct agent for schema interpretation (flexible reasoning) + plain task functions for SQL generation/grading/aggregation (deterministic, well-scoped) → optional Federated Query Engine (Trino) for cross-database execution → dual grading (SQL grade + summary grade, scored separately).
**Trade-offs**: Reserve agentic behavior for steps needing flexible tool selection; over-agentifying every stage adds unnecessary latency/unpredictability. (Ch 14, 15)

## CoT-Based Reranking (Reasoning-Augmented Reranker)
**When to use**: Multimodal or complex reranking where pure vector similarity misses semantic nuance.
**How**: LLM reasons step-by-step about query-candidate alignment, outputs a reasoning trace + numeric score; fuse with vector-similarity score via weighted average (`combined = α·vector_score + (1-α)·cot_score`).
**Trade-offs**: Adds LLM inference cost per candidate (so apply only after prefiltering); gains interpretability (logged reasoning) over pure vector scoring. (Ch 13)

## GenAI-OCR Pipeline (Wrapped OCR + LLM)
**When to use**: Extracting and acting on text from product images, receipts, or documents in a modular, locally-deployable way.
**How**: OCR engine (EasyOCR) extracts raw text → LLM converts raw text to structured intent (search query, JSON fields) → downstream action (search URL construction, database insertion, summarization).
**Trade-offs**: Most modular/interpretable OCR approach but requires separate tuning of OCR and LLM stages; OCR foundation models or fine-tuned multimodal LLMs trade modularity for integration/accuracy on noisy inputs. (Ch 16)

## Traditional ML Model as LLM-Callable Tool
**When to use**: Enterprises with existing validated ML models (classifiers, regressors, CNNs, forecasters) wanting conversational/agentic access.
**How**: Wrap the model as a REST microservice (FastAPI) → expose as a LangChain `Tool` object with name/func/description → LLM agent parses NL query into structured features, invokes the tool, interprets numeric output into natural language explanation.
**Trade-offs**: Preserves model accuracy and avoids costly retraining; requires careful feature-extraction prompting and `handle_parsing_errors=True` for robustness. (Ch 17)

## RAGOps Observability Stack
**When to use**: Any production RAG/agentic system needing reliability, traceability, and continuous quality assurance.
**How**: Development-time identification (per-stage failure points) + benchmarking (gold-standard dataset, version-tied baselines) → post-deployment continuous monitoring (retrieval/generation/system-level metrics) + span-based tracing (process) + metric-based evaluation (outcome, LLM-as-judge) via platforms like MLflow/Langfuse/Arize Phoenix.
**Trade-offs**: Significant instrumentation overhead, but without it, production failures are undiagnosable and silent regressions accumulate. (Ch 18)
