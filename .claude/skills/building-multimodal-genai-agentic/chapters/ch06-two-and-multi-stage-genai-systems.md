# Chapter 6: Two and Multi-stage GenAI Systems

## Core Idea
Interaction depth (no/late/full interaction) between query and document representations determines the retrieval-vs-precision trade-off at each RAG stage, and production RAG systems increasingly add explicit grading mechanisms (relevance, hallucination, answer-quality graders) as a deliberative evaluation layer rather than treating generation as a single blind pass.

## Frameworks Introduced
- **Interaction taxonomy for dense retrieval**: No interaction (bi-encoder: independent encoding, fast, shallow), Full interaction (cross-encoder: joint encoding with full cross-attention, accurate but expensive, used for reranking small sets), Late interaction (ColBERT/ColPali/ColQwen: independent encoding but token-level embeddings retained, compared via MaxSim — middle ground of scalability + precision).
  - When to use: no interaction for first-stage retrieval at scale; full interaction for reranking small top-k shortlists; late interaction when you need token-level precision without full cross-encoder cost, including multimodal (ColPali) and LLM-integrated (ColQwen) variants.
- **Multi-vector representation (infrastructure layer, e.g., Qdrant)**: Store multiple (token/phrase-level) vectors per document alongside a single dense vector; dense vector used for first-pass ANN retrieval (HNSW-indexed), multi-vector field used for MaxSim-based reranking (HNSW indexing disabled to save memory/compute).
  - Distinct from late interaction models: multi-vector support is a storage/serving mechanism; late interaction is a model-architecture/training-time technique. They're complementary — late interaction models generate the embeddings, multi-vector infra stores/serves them efficiently.
- **Two-stage RAG architecture**: Stage 1 = dense retrieval (no interaction, broad recall, top 50-100 candidates). Stage 2 = reranking (late or full interaction, deep precision) before generation.
- **Multi-stage RAG variants** (the chapter's taxonomy): Simple RAG, Simple RAG with memory, Branched RAG (multi-hop), HyDE (Hypothetical Document Embedding), Adaptive RAG, Corrective RAG (CRAG), Self-RAG, Agentic RAG.
  - When to use each: Simple RAG for straightforward QA; +memory for conversational continuity; Branched for multi-hop/complex synthesis; HyDE when queries are abstract/underspecified; Adaptive when query complexity/type varies; CRAG when factual accuracy is critical and self-verification is needed; Self-RAG for self-reflective iterative improvement; Agentic RAG for tool-using, multi-step, decision-making tasks.
- **Grading mechanisms**: LLM-based binary classifiers (and explanation-producing graders) inserted at pipeline checkpoints to evaluate retrieval relevance, hallucination/groundedness, and answer quality/completeness before surfacing output.

## Key Concepts
- **MaxSim**: Similarity operation computing, for each query token, the maximum cosine similarity against all document token embeddings, then aggregating (sum/average) across query tokens — core scoring mechanism of late interaction and multi-vector reranking.
- **HyDE (Hypothetical Document Embedding)**: Generate a hypothetical "ideal answer" document from the query, embed *that*, and use it to search the knowledge base — helps when queries are abstract or have no exact lexical match.
- **CRAG (Corrective RAG)**: Adds a validation loop — cross-verify generated output against retrieved content; on discrepancy, trigger additional retrieval or corrective regeneration.
- **Self-RAG**: Model performs self-reflection on its own generated answer, identifies weaknesses, and retrieves/regenerates to improve — a self-improvement cycle.
- **Retrieval relevance grader**: LLM-based binary classifier (`binary_score`: yes/no) checking whether a retrieved document is semantically relevant to the query — gate before generation.
- **Hallucination detection grader**: LLM-based grader comparing generated answer ("STUDENT ANSWER") against retrieved facts ("FACTS"), returning `binary_score` + `explanation` — checks groundedness, not just relevance.
- **Answer quality grader**: LLM-based grader assessing whether the generated answer meaningfully/sufficiently addresses the original question (semantic utility, not just factual grounding).
- **Token utilization in multi-stage RAG**: Each added stage (retrieval, reranking, validation, generation) consumes context-window budget; mitigations include selective truncation, summarization/compression, and tiered routing that invests tokens only where warranted.

## Mental Models
- Think of the interaction spectrum (no → late → full) as a dial between speed and precision — move the dial right (toward full interaction) only for the smaller candidate set you can afford to score expensively.
- Treat grading as "turning generation into a deliberative, evaluative pipeline" — instead of one blind generation pass, insert checkpoints (grader) exactly where a specific failure mode (irrelevant retrieval, hallucination, incomplete answer) could occur.
- Multi-stage RAG variants (HyDE, CRAG, Self-RAG, Agentic RAG, Adaptive RAG) are not mutually exclusive — think of them as optional plug-in modules addressing one specific weakness of simple two-stage RAG (query-document mismatch, factual drift, static strategy, lack of autonomy).
- ColBERT-style late interaction retrieval already performs an implicit reranking — adding an explicit second-stage reranker on top is about marginal gains (ranking fidelity, generation-alignment, ensemble robustness), not a strict necessity.

## Anti-patterns
- **Using full-interaction (cross-encoder) models for first-stage retrieval over a large corpus**: computationally prohibitive — reserve for reranking small shortlists only.
- **Naively indexing all token-level vectors in HNSW for multi-vector reranking**: causes high RAM usage, slow inserts, and redundant compute — disable HNSW indexing for the multi-vector field and reserve it for reranking only.
- **Treating two-stage RAG as sufficient for high-stakes domains without validation**: without a hallucination grader or CRAG-style corrective loop, grounded-looking answers can still contain fabricated content.
- **Ignoring token budget in multi-stage pipelines**: expanded context (branched retrieval, intermediate scoring passes, long-form generation) silently escalates token usage/cost unless actively managed via truncation, summarization, or tiered routing.
- **Conflating multi-vector representations (infrastructure) with late interaction models (architecture)**: they operate at different layers and solve different problems — multi-vector support serves late-interaction-style models efficiently, but is not itself a modeling technique.

## Code Examples
```python
### Retrieval Grader
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""
result = llm_json_mode.invoke(
    [SystemMessage(content=doc_grader_instructions)]
    + [HumanMessage(content=doc_grader_prompt_formatted)]
)
json.loads(result.content)
```
- **What it demonstrates**: LLM-as-binary-classifier grader pattern applied to retrieval relevance checking.

```python
### Hallucination Grader
hallucination_grader_instructions = """
You are a teacher grading a quiz.
You will be given FACTS and a STUDENT ANSWER.
Ensure the STUDENT ANSWER is grounded in the FACTS.
Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
..."""
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.
Return JSON with two keys, binary_score ('yes'/'no') and explanation."""
result = llm_json_mode.invoke(
    [SystemMessage(content=hallucination_grader_instructions)]
    + [HumanMessage(content=hallucination_grader_prompt_formatted)]
)
json.loads(result.content)
```
- **What it demonstrates**: Explanation-producing grader that checks groundedness of generation against retrieved facts, the core mechanism behind CRAG-style correction loops.

## Reference Tables
| Dimension | Multi-vector representations (Qdrant) | Late interaction architectures (e.g., ColBERT) |
|---|---|---|
| Definition | Storage/querying mechanism for multiple vectors/doc | Model architecture performing fine-grained similarity |
| Purpose | Infrastructure optimization | Semantic modeling at embedding level |
| Similarity function | MaxSim or configurable | Typically fixed MaxSim |
| Indexing | Selective disabling possible | External; stored for matching |
| Model dependency | Any model with multi-embedding output | Requires specific architecture (e.g., ColBERT layers) |
| Usage context | Production vector DB infra | Algorithmic design (train+inference) |

| Multi-stage RAG variant | Key mechanism | Best for |
|---|---|---|
| Simple RAG | Retrieve → generate | Straightforward QA |
| Simple RAG + memory | + conversational memory | Multi-turn dialogue, co-reference |
| Branched RAG | Multi-hop iterative retrieval | Complex synthesis across sources |
| HyDE | Embed a hypothetical ideal answer document | Abstract/underspecified queries |
| Adaptive RAG | Dynamically choose retrieval/generation strategy | Varying query complexity/domain |
| CRAG | Validate generation against retrieval, correct | High-stakes factual accuracy (legal, medical) |
| Self-RAG | Self-reflect, re-retrieve, improve | Self-correcting iterative quality |
| Agentic RAG | Plan + tool use + multi-step execution | Workflow orchestration, decision-making |

| Grader type | Purpose | When to use |
|---|---|---|
| Retrieval relevance grader | Document relevant to query? | After retrieval, before generation |
| Hallucination grader | Answer grounded in retrieved facts? | After generation, before final output |
| Answer quality grader | Answer meaningfully addresses query? | Final evaluation before display |
| Faithfulness grader | Logical alignment with sources | LLM may infer unstated conclusions |
| Completeness grader | Covers all sub-parts of a compound question | Multi-part questions |
| Coherence grader | Logically/grammatically well-formed | Readability checks |
| Toxicity/bias grader | Harmful/biased content detection | Safety, pre-deployment |
| Conciseness grader | Not verbose/off-topic | Summary or mobile use cases |
| Consistency grader | Similar questions get aligned answers | Multi-turn/batch evaluation |
| Instruction-following grader | Adherence to custom constraints | Prompts with explicit formatting rules |
| Evidence attribution grader | Proper source citation | Knowledge-intensive/academic QA |

## Worked Example
Multi-stage RAG with routing (Chapter 6 notebook): domain documents (blog posts on agents/adversarial attacks) embedded via NomicEmbeddings into an SKLearnVectorStore. Incoming questions pass through a JSON-mode router model that decides vector-store retrieval vs. live web search (Tavily API). Retrieved documents pass a retrieval relevance grader; if relevant, a RAG generator (local Llama 3.2 via Ollama) produces an answer. The answer is then checked by a hallucination grader (groundedness) and an answer-quality grader (completeness), all orchestrated as a LangGraph state machine with conditional routing, retries, and fallback to web search when vector-store content is insufficient.

## Key Takeaways
1. Interaction depth (no/late/full) is the central lever balancing retrieval speed against semantic precision — pick based on corpus size and accuracy requirements per stage.
2. Multi-vector representations (infra) and late interaction models (architecture) are complementary, not identical — know which layer you are optimizing.
3. Two-stage RAG (retrieve broad, rerank precise) is the baseline; multi-stage variants (HyDE, CRAG, Self-RAG, Adaptive RAG, Agentic RAG) each target one specific weakness of the baseline.
4. Grading mechanisms turn RAG from a single blind generation pass into a deliberative, checkpointed pipeline — essential for hallucination reduction in high-stakes domains.
5. Token budget must be actively managed across multi-stage pipelines; more stages = more token cost unless mitigated via truncation/summarization/tiered routing.
6. Routing (vector-store vs. web search, or strategy-selection in Adaptive RAG) lets systems dynamically match retrieval source/strategy to query characteristics.

## Connects To
- **Ch 1**: Extends the bi-encoder/cross-encoder and single-/two-stage RAG framing introduced there into the full no/late/full interaction taxonomy.
- **Ch 5**: The HITL critic/validator pattern is formalized here into specific grader types (relevance, hallucination, answer-quality).
- **Ch 9**: Reranking (late/full interaction, MaxSim) introduced here is implemented in depth in the dedicated reranking chapter.
- **Ch 10**: Multi-vector representations and token-budget management here directly inform retrieval optimization techniques.
- **Ch 18**: Grading mechanisms here are precursors to the formal LLMOps/GenAI evaluation techniques in the final chapter.
