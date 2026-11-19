# Chapter 15: Agentic Text-to-SQL Systems and Architecture Decision-Making

## Core Idea
A production agentic text-to-SQL system for federated, multi-database enterprise environments uses a single LangChain ReAct agent for schema interpretation (not for everything), surrounded by discrete, independently-invoked task modules (schema matching, SQL generation, grading, summarization, aggregation) orchestrated by a plain Python entry point — agentic orchestration where it adds value, direct function calls everywhere else.

## Frameworks Introduced
- **14-step agentic text-to-SQL pipeline** (the chapter's central architecture — see Reference Tables): User input → query embedding (Sentence Transformers) → global index lookup (ChromaDB) → LangChain orchestration → schema matcher → global index cross-reference → pre_filter(query_embedding) → semantic search on summarized data → SQL generation (ReAct agent + CoT) → SQL produced → SQL graded (LLM) → SQL execution (optionally via FQE) → response to LangChain → response to user.
  - When to use: for enterprise environments with multiple heterogeneous, siloed databases (PostgreSQL, MySQL, MongoDB, etc.) where non-technical users need self-serve natural language access without SQL expertise.
- **Single-agent, multi-tool architecture decision**: Only one active LangChain ReAct agent (`summarization_schema_agent`) is used, invoking two registered tools (schema matcher, schema summarizer); SQL generation, grading, and aggregation are implemented as *plain task functions* called directly from `main.py`, not as additional agents.
  - Architecture rationale: not every pipeline stage needs agentic (LLM-driven, tool-invoking) reasoning — schema interpretation benefits from agentic flexibility, while SQL generation/grading/aggregation are well-scoped enough to be deterministic function calls with LLM calls inside them.
- **Federated Query Engine (FQE) integration (e.g., Trino)**: Optional execution layer allowing a single generated SQL query to be executed across multiple heterogeneous structured data sources and aggregated in real time, rather than requiring all data to live in one database.
- **Pre-filtering + global index pattern**: `pre_filter(query_embedding)` narrows the vector search space before semantic search over summarized schema/data — reduces retrieval latency and improves relevance before SQL construction, directly reusing the prefiltering-threshold concept from Chapter 10.

## Key Concepts
- **Global index**: A ChromaDB-backed vector store of schema summaries and/or historical query patterns, used for similarity-based retrieval to ground schema matching across multiple databases.
- **Schema matcher**: Compares query intent against available database schemas using embedding similarity + heuristics to identify semantically aligned tables/columns — the critical disambiguation step for multi-database querying.
- **Aggregator**: Consolidates partial results/summaries from multiple subtasks or data sources into one unified summary, resolving duplicate entities across databases (e.g., same customer appearing in both PostgreSQL and MySQL).
- **SQL grader**: LLM-based evaluator scoring generated SQL on correctness, relevance, and execution efficiency, with justification — adds explainability and a feedback signal for query refinement.
- **Summary grader**: Separately evaluates the natural-language summary (not the SQL) for accuracy, clarity, and comprehensiveness, catching duplication or omission issues independent of SQL correctness.
- **Modular Python package layout**: `agents/` (LangChain ReAct agents), `core/` (embeddings, ChromaDB interface, LLM interface, SQL executor, multi-SQLite reader, cache, utils — backend infrastructure), `tasks/` (aggregator, sql_generator, grader, schema_matcher, summarizer — discrete LLM-powered functions), `setup/` (index population scripts), `frontend/` (Streamlit UI).
- **In-memory, non-persisted execution**: The pipeline computes everything in memory per query; no intermediate results are persisted to disk/database beyond the seeded SQLite source data and the ChromaDB index.

## Mental Models
- Think of this chapter's architecture decision ("one agent, many plain tools") as the practical answer to "when should a pipeline stage be agentic?" — use an LLM-driven ReAct agent when a step requires flexible, context-dependent tool selection (schema interpretation across unpredictable databases); use a plain function when the step's logic is well-scoped even if it internally calls an LLM (SQL generation, grading).
- Treat the global index + schema matcher + pre-filter trio as the retrieval-optimization toolkit from Chapter 10 (multi-index embedding, modality/source routing, prefiltering thresholds) applied specifically to database schema retrieval instead of document/image retrieval.
- The dual-grading design (SQL grade + summary grade, evaluated separately) mirrors the generative-response-grader vs. retrieval-relevance-grader split from Chapter 6/8 — grade the *mechanism* (SQL correctness) and the *communication* (summary clarity) independently, since they can fail independently.
- Federated execution (via Trino/FQE) is the data-infrastructure analog of multi-index embedding — instead of routing a query to one modality's vector index, route (or fan out) the SQL execution across multiple heterogeneous databases and merge results.

## Anti-patterns
- **Making every pipeline stage its own LLM agent**: unnecessary agentic overhead for well-scoped, deterministic tasks (SQL generation from already-matched schema, grading against a fixed rubric) adds latency and unpredictability without benefit — this chapter deliberately keeps only one agent active.
- **Skipping pre-filtering before semantic search at multi-database scale**: without `pre_filter(query_embedding)`, every query searches the full global index, increasing latency unnecessarily as the number of federated databases grows.
- **Treating SQL correctness and summary quality as one combined score**: they can fail independently (correct SQL, confusing summary; or vice versa) — grade them separately for actionable feedback.
- **Building a text-to-SQL system to replace data engineers entirely**: the chapter explicitly frames this as augmentation (removing bottlenecks on routine queries) while preserving data engineers' role in data modeling, pipeline optimization, and governance — overclaiming full replacement undersells the real risk/oversight trade-offs.
- **Leaving an unused secondary agent (`sql_agent.py`) wired into documentation as if active**: the chapter explicitly notes this file exists but is not invoked in the main execution path — a reminder to keep architecture docs honest about what's actually running.

## Code Examples
```python
# main.py orchestration (plain function, not an agent)
schema_results = summarization_schema_agent.invoke({"input": query})
aggregated_result = aggregate_summarized_data(query)
sql_query = generate_sql(...)
sql_grade = grade_sql(sql_query)
summary_grade = grade_summary(aggregated_result["final_summary"])
```
- **What it demonstrates**: The hybrid architecture — one agentic call (`summarization_schema_agent.invoke`) feeding into a sequence of plain, directly-called task functions (aggregate, generate, grade) rather than further agent delegation.

```python
# Folder-level architecture (conceptual)
agents/summarization_schema_agent.py   # single active ReAct agent: schema matcher tool + schema summarizer tool
agents/sql_agent.py                    # defined but NOT invoked in main pipeline
tasks/sql_generator.py                 # CoT-based SQL construction, called directly
tasks/grader.py                        # LLM-based SQL + summary grading, called directly
tasks/aggregator.py                    # consolidates multi-source summaries, called directly
core/chroma_index.py                   # ChromaDB interface (global index)
core/sqlite_multi_reader.py            # federated access across multiple SQLite files
```
- **What it demonstrates**: The modular package layout separating agentic reasoning (agents/) from deterministic task execution (tasks/) and backend infrastructure (core/).

## Reference Tables
| Pipeline step | Function |
|---|---|
| 1. User input | NL query via Streamlit/API |
| 2. Query embedding | Sentence Transformer vectorizes query |
| 3. Global index lookup | ChromaDB similarity search for schema/query patterns |
| 4. LangChain orchestration | Invokes reasoning/tool-use for summarization/matching |
| 5. Schema matcher | Aligns query intent with schema tables/columns |
| 6. Global index cross-reference | Validates/refines schema selection |
| 7. pre_filter(query_embedding) | Narrows vector search space |
| 8. Semantic search | Over summarized schema/data with pre-filtered embedding |
| 9-10. SQL generation | LangChain ReAct agent + CoT prompt produces SQL |
| 11. SQL graded | LLM evaluates syntactic/semantic correctness |
| 12. SQL execution (+ optional FQE) | Executes, optionally across multiple DBs via Trino |
| 13. Response to LangChain | Post-processing/formatting |
| 14. Response to user | SQL + data + optional summary returned via UI |

| Module category | Examples | Role |
|---|---|---|
| agents/ | summarization_schema_agent.py (active), sql_agent.py (unused) | Agentic schema interpretation |
| core/ | embeddings.py, chroma_index.py, llm.py, sql_executor.py, sqlite_multi_reader.py, cache.py | Backend infrastructure |
| tasks/ | aggregator.py, sql_generator.py, grader.py, schema_matcher.py, summarizer.py | Discrete LLM-powered functions |
| setup/ | populate_chroma.py | One-time index population |
| frontend/ | app.py (Streamlit) | User interaction |

## Worked Example
Real-time retail intelligence scenario: customer data in PostgreSQL, orders in MySQL, marketing logs in MongoDB, inventory in yet another system (~245GB+ total) — too large/siloed for ad hoc manual SQL. A business user submits a natural language query via Streamlit. The pipeline embeds the query, retrieves relevant schema summaries from the ChromaDB global index, matches schema elements via `schema_matcher.py`, pre-filters and semantically searches summarized data, then the `summarization_schema_agent` (the one active ReAct agent) and `sql_generator.py` produce a CoT-reasoned SQL query. `grader.py` scores the SQL for correctness/relevance/efficiency and separately scores the generated summary for clarity/completeness. The final output — aggregated summary, detailed per-database entity breakdown, generated SQL, SQL grade, and summary grade — resolves duplicate customer entries across databases and surfaces actionable patterns (city-wise distribution, etc.) without the user writing any SQL.

## Key Takeaways
1. Not every pipeline stage needs to be an LLM agent — reserve agentic (ReAct, tool-invoking) behavior for steps needing flexible, context-dependent reasoning (schema interpretation), and use plain functions (that may internally call an LLM) for well-scoped steps (SQL generation, grading).
2. Federated Query Engines (e.g., Trino) let a single generated SQL query span multiple heterogeneous databases, directly solving the data-silo problem.
3. Pre-filtering before semantic search is essential for multi-database scale — don't search the full global index on every query.
4. Grade SQL correctness and summary quality separately — they are independent failure modes.
5. A well-modularized package layout (agents/core/tasks/setup/frontend) keeps agentic reasoning, infrastructure, and deterministic logic cleanly separated and independently testable.
6. Text-to-SQL systems augment, not replace, data engineers — they remove bottlenecks on routine queries while preserving engineering oversight of modeling, pipelines, and governance.

## Connects To
- **Ch 14**: Directly extends the single-database text-to-SQL foundations into a multi-database, agentic, federated architecture.
- **Ch 5**: The single-ReAct-agent-with-tools pattern reuses the agent/tool/reasoning primitives first introduced in the HITL agentic chapter.
- **Ch 10**: Reuses the pre-filtering-threshold and multi-index retrieval optimization techniques, applied to schema/query retrieval instead of document retrieval.
- **Ch 6**: The dual SQL-grade/summary-grade design directly extends the grading-mechanism framework (retrieval relevance grader, hallucination grader, answer quality grader).
- **Ch 16**: Explicitly previewed as the next chapter — integrating OCR with GenAI to convert images into actionable search insights.
