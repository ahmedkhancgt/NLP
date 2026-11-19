# Chapter 18: LLM Operations and GenAI Evaluation Techniques

## Core Idea
Production GenAI systems need two distinct, complementary evaluation layers (LLM evaluation for the model in isolation, RAG evaluation for the full retrieval+generation pipeline) wrapped in an operational discipline — RAGOps — spanning development-time benchmarking and post-deployment continuous monitoring, instrumented via observability platforms like MLflow, Langfuse, and Arize Phoenix.

## Frameworks Introduced
- **LLM evaluation vs. RAG evaluation**: LLM evaluation assesses the model in isolation (fluency, coherence, factual accuracy, relevance) via BLEU, ROUGE, METEOR, BERTScore, human evaluation. RAG evaluation assesses the full pipeline — retrieval quality (Recall@k/Precision@k, embedding drift, coverage/diversity), generation groundedness (faithfulness to source, context usage), and pipeline-level metrics (EM/F1, context-aware BERTScore, LLM-based graders).
  - Why both matter: a perfect LLM paired with poor retrieval still fails; great retrieval paired with a misaligned LLM still hallucinates — evaluate both independently and jointly.
- **RAGOps lifecycle framework**: Development phase (identification of failure points per stage + benchmarking against gold-standard datasets with version-tied baselines) → Post-development phase (continuous monitoring, drift detection, real-time tracing, feedback loop collection).
  - Identification-then-benchmarking sequence: first identify what to track (decompose pipeline into embedding/retrieval/reranking/prompt/generation stages, enumerate failure modes per stage), then establish quantitative baselines/thresholds tied to specific component versions.
- **Continuous monitoring framework**: Retrieval-level (recall approximation, latency, top-k similarity/diversity, embedding drift), Generation-level (hallucination risk, token truncation, coherence/fluency, failure modes), System-level (end-to-end latency, model/prompt version logging, query volume/failure rates, tool execution failures) — feeding threshold-based alerting, dashboards, and self-healing feedback loops.
- **DevOps vs. MLOps vs. RAGOps**: DevOps automates software CI/CD/infrastructure; MLOps extends this to ML model training/versioning/deployment/monitoring; RAGOps extends further to LLM-specific challenges — grounding quality, hallucination rate, retrieval faithfulness, and traceability across retrieval/reranking/generation stages.
- **MLflow dual instrumentation approaches**: Approach 1 — span-based tracing (patch `ollama.chat()` with `@trace` decorator to log every LLM call as an inspectable span: inputs, model, outputs) for tool-auditing/debugging. Approach 2 — metric-based evaluation (`mlflow.metrics.genai.faithfulness()`/`relevance()`, LLM-as-judge) for end-to-end output quality monitoring. Complementary, not exclusive.

## Key Concepts
- **Gold-standard evaluation dataset**: Manually curated or synthetically generated queries paired with known relevant documents and expected outputs — a fixed, repeatable reference for benchmarking that live production data cannot provide (due to drift and lack of ground truth).
- **Embedding drift**: Degradation in how well current embeddings represent an evolving knowledge base — tracked via cosine similarity over time; a key post-deployment monitoring signal.
- **Faithfulness vs. relevance (as separate metrics)**: Faithfulness = does the generated answer reflect retrieved/database facts (groundedness)? Relevance = does the answer align with the user's query intent? These can fail independently and must be scored separately.
- **Span-based tracing vs. metric-based evaluation**: Tracing captures the *process* (every intermediate LLM call, inputs/outputs) for debugging; metric-based evaluation captures the *outcome* (final answer quality) for monitoring — both needed for full-stack observability.
- **Benchmark slicing and shadow evaluation**: Post-development adaptive benchmarking — slicing benchmark subsets by user segment/query type for granular diagnostics; shadow-evaluating new system variants on benchmark queries in parallel without affecting live output.
- **MLflow local tracking directory structure**: `mlruns/<experiment_id>/<run_id>/` containing `artifacts/`, `metrics/`, `params/`, `tags/`, and `meta.yaml` — understanding this layout is essential for troubleshooting missing runs/metrics in file-backed MLflow deployments.
- **Service-Level Objectives (SLOs) for GenAI**: Concrete operational commitments (e.g., "hallucination rate below 7%", "recall@5 above 70%") validated through systematic benchmarking (scheduled/nightly runs, CI/CD gates), not inferred from live traffic alone.

## Mental Models
- Think of LLM evaluation and RAG evaluation as testing "the engine alone" vs. "the whole car on the road" — both are necessary, and a fast engine in a car with bad steering (retrieval) still crashes.
- Treat RAGOps' identification→benchmarking sequence as the same two-step diagnostic logic used throughout the book: first find where things can break (Ch3's seven failure points, Ch10's six retrieval drawbacks), then quantify acceptable thresholds for each.
- Continuous monitoring is "benchmarking's real-time sibling" — benchmarking gives a stable, periodic reference; monitoring gives live, moment-to-moment visibility; together they form the full observability loop.
- Span-based tracing (process) + metric-based evaluation (outcome) is the same "mechanism vs. communication" grading split seen in Chapters 6/8/15 (SQL grade vs. summary grade; retrieval relevance grader vs. hallucination grader) — track both what happened and whether the result was good.

## Anti-patterns
- **Relying solely on periodic benchmarks without continuous monitoring**: benchmarks are static/periodic; production query distributions, knowledge bases, and upstream APIs change continuously — without real-time monitoring, regressions go undetected between benchmark runs.
- **Evaluating only the LLM or only the retrieval pipeline, never both**: a model that aces standalone LLM benchmarks can still produce hallucinated/off-topic answers in production if retrieval quality is poor, and vice versa.
- **Skipping version-tying in benchmark results**: a benchmark score with no linkage to the specific embedding model, vector index, prompt template, or reranker version in use makes regression root-cause analysis impossible.
- **Using accuracy-only metrics for imbalanced or nuanced generative tasks**: faithfulness and relevance must be tracked as separate scalar metrics — conflating them hides which failure mode (groundedness vs. intent-alignment) actually occurred.
- **Instrumenting only final-output metrics without process-level tracing (or vice versa)**: metric-only evaluation misses *why* a bad output happened (which tool/step failed); tracing-only misses whether outcomes are actually good — use both.
- **Treating RAGOps as a one-time setup rather than a lifecycle practice**: identification and benchmarking must recur as the system, data, and models evolve — a static RAGOps setup from launch day stales quickly.

## Code Examples
```python
# Approach 1: span-based tracing
from mlflow.tracing import trace, SpanType

@trace(name="ollama.chat", span_type=SpanType.CHAT_MODEL)
def trace_ollama_chat(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        span.set_inputs({"messages": kwargs["messages"], "model": model})
        response = func(*args, **kwargs)
        span.set_outputs(response)
        return response
    return wrapper

ollama.chat = trace_ollama_chat(ollama.chat)
```
- **What it demonstrates**: Decorator-based patching that makes every LLM call automatically traceable as an MLflow span, without changing calling code.

```python
# Approach 2: metric-based evaluation (LLM-as-judge)
from mlflow.metrics.genai import faithfulness, relevance

faith_score = faithfulness(model="ollama:/mistral")(
    predictions=[final_answer], inputs=[user_question], context=[json.dumps(query_results)]
).scores[0]

mlflow.log_metric("faithfulness", faith_score)
mlflow.log_metric("relevance", rel_score)
```
- **What it demonstrates**: Using a separate LLM-as-judge evaluator to score faithfulness and relevance of a final generated answer, logged as scalar MLflow metrics.

```python
with mlflow.start_run(run_name="CypherTest_Run1") as run:
    mlflow.log_param("question", user_question)
    mlflow.log_param("cypher_query", cypher_query or "EMPTY")
    mlflow.log_text(json.dumps(query_results, indent=2), "neo4j_context.json")
    mlflow.log_text(final_answer or "EMPTY", "final_answer.txt")
    faith_score = evaluate_faithfulness_with_ollama(...)
    rel_score = evaluate_relevance_with_ollama(...)
    mlflow.log_metric("faithfulness", faith_score)
    mlflow.log_metric("relevance", rel_score)
```
- **What it demonstrates**: A complete MLflow run logging parameters, artifacts, and evaluation metrics for a graph-based (Text2Cypher) recommendation pipeline, combining both span-tracing and metric-evaluation observability.

## Reference Tables
| Stage | Key failure points | What to track | Metrics/Tools |
|---|---|---|---|
| Data ingestion/embedding | Low-quality/outdated embeddings, missing docs | Embedding quality, coverage, drift | Embedding similarity, coverage %, Faiss index stats |
| Retrieval | Irrelevant top-k, latency, poor recall | Recall@k, precision@k, latency | Langfuse/Arize Phoenix logs |
| Reranking | Incorrect ranking, noisy scoring | Score divergence, rank stability | Reranker score variance |
| Prompt construction | Overlong prompts, truncation | Prompt length, template consistency | Token length logs |
| Generation | Hallucinations, incoherence | Groundedness, fluency, hallucination rate | BERTScore, WhyLabs, Langfuse traces |
| Evaluation | Subjective quality, no feedback | Human ratings, faithfulness | BLEU, ROUGE, BERTScore, Ragas |

| RAG system type | Key failure points | Metrics |
|---|---|---|
| Single-stage RAG | Low-quality embeddings, hallucination | Recall@k, BERTScore, hallucination rate |
| Two-stage RAG | Weak retrieval, poor reranking | Recall@k pre/post rerank, faithfulness |
| Multi-stage RAG | Error propagation, reranker conflict | Stage-wise recall, ensemble agreement |
| Multimodal RAG | Modality misalignment, poor fusion | CLIP similarity, VQAScore, cross-modal BERTScore |
| Agentic RAG | Planning loops, invalid toolchains | Plan validity, action-observation alignment |
| Graph-based RAG | Sparse/irrelevant graph, traversal errors | Graph coverage, node centrality |
| Text-to-SQL RAG | Wrong schema, invalid SQL | SQL validity rate, execution accuracy |
| OCR-based RAG | OCR inaccuracy, layout misclassification | OCR confidence score, extraction accuracy |

| Observability platform | Role |
|---|---|
| Langfuse | Open-source trace logging, prompt management, cost/latency metrics |
| Arize Phoenix | Pipeline tracing, cluster analysis, RAG diagnostics |
| WhyLabs | Drift monitoring, grounding metrics, security (PII/injection) monitoring |
| MLflow | End-to-end tracing, LLM-as-judge evaluation, prompt registry, deployment governance |
| Ragas | Synthetic test-data generation, reference-free faithfulness/hallucination scoring |
| LlamaIndex observability | Built-in instrumentation for LlamaIndex-structured RAG pipelines |

| Ops layer | Scope |
|---|---|
| DevOps | Software CI/CD, infrastructure, testing |
| MLOps | ML model training/versioning/deployment/monitoring |
| RAGOps | LLM/RAG-specific: grounding, hallucination, retrieval faithfulness, cross-stage traceability |

## Worked Example
A graph-enhanced RAG recommendation system (movie recommendations): product/preference data converted to text → chunked → embedded via `Ollama-all-miniLM-L6-v2` → a NetworkX graph captures product relationships. A LangChain ReAct agent orchestrates three retrieval tools (Faiss vector search, NetworkX graph search, hybrid fusion), reranks with `ms-marco-MiniLM-L6-v2` cross-encoder, and wraps the final result in natural language via Ollama. A second variant uses LangGraph to convert user questions into Cypher queries executed on Neo4j, summarized by Mistral. Both are instrumented with MLflow: `mlflow_ollama_patch.py` traces every `ollama.chat()` call as a span (Approach 1), while `main.py` separately computes `faithfulness` and `relevance` scores via MLflow's GenAI metrics module using an LLM-as-judge (Approach 2). Each run is logged with parameters (question, Cypher query), artifacts (Neo4j results, final answer), and metrics, viewable in the MLflow dashboard (`waitress-serve --host 127.0.0.1 --port 5000 mlflow.server:app`) for cross-run comparison.

## Key Takeaways
1. LLM evaluation and RAG evaluation are distinct and both necessary — never assume a good standalone model guarantees a good end-to-end pipeline.
2. RAGOps follows identification (what can fail, where) then benchmarking (quantitative baselines tied to specific component versions) during development, then continuous monitoring and feedback loops post-deployment.
3. Faithfulness (groundedness) and relevance (intent-alignment) must be tracked as separate metrics since they can fail independently.
4. Span-based tracing (process visibility) and metric-based evaluation (outcome visibility) are complementary observability techniques — use both for full-stack GenAI observability.
5. DevOps, MLOps, and RAGOps address different layers (infrastructure, model, LLM/RAG-specific) and should be orchestrated together, not treated as substitutes for one another.
6. Gold-standard datasets must remain static for repeatable comparison, but can be incrementally augmented with human-verified live examples for hybrid benchmarking.
7. Operational discipline (versioning, traceability, rollback mechanisms, SLO-driven benchmarking) is what separates a GenAI prototype from a production-ready system.

## Connects To
- **Ch 3, 6, 10**: Directly builds on and formalizes the failure-point/mitigation frameworks from the local RAG (seven failure points), grading mechanisms, and retrieval-optimization chapters into a comprehensive operational discipline.
- **Ch 8 & 15**: The dual-metric evaluation pattern (faithfulness/relevance, SQL-grade/summary-grade) recurs here as the capstone evaluation framework for the whole book.
- **Ch 9-10**: The graph-enhanced recommendation system's hybrid retrieval (vector + graph + fusion) directly reuses the multi-index embedding and hybrid retrieval patterns from the reranking and retrieval-optimization chapters.
- **Whole book**: As the final chapter, this closes the loop from Chapter 1's foundational RAG/agentic concepts through to the operational rigor (RAGOps, LLMOps, observability) needed to run these systems reliably in production.
