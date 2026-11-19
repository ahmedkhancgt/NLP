# Chapter 14: Building Text-to-SQL Systems

## Core Idea
Text-to-SQL is a structurally hard problem (schema linking, data readiness, domain generalization, execution safety) that requires a full pipeline — prompt engineering, schema/metadata integration, intermediate planning, output validation, fallback/retry, observability, and governance — not just a single LLM call, and evaluation needs multiple complementary metrics since SQL has no single "correct" surface form.

## Frameworks Introduced
- **Text-to-SQL core components**: Natural language understanding (intent/entity/aggregation detection) → Schema linking (explicit/implicit/fuzzy mapping of phrases to tables/columns) → Semantic parsing (logical form/AST) → SQL generation (clauses, joins, aggregations, filters).
- **End-to-end text-to-SQL pipeline architecture** (13-part practical guide — the chapter's central framework; see Reference Tables): prompting strategy selection → schema/metadata integration → intermediate planning/decomposition → row/table summarization → SQL validation/safety → multimodal/tool-augmented extensions → system integration → query clarification → governance/compliance → fallback/retry → deployment strategy → observability → feedback loop/correction interface.
- **Fallback and retry mechanism**: Layered strategies — natural language clarification, retry with prompt refinement, fallback to simplified query, fallback to search/documentation, default query templates — triggered by ambiguous queries, schema mismatches, hallucinated references, or execution errors.
  - How: wrap `generate_sql()` + `execute_sql()` in a try/except cascade that regenerates with schema guidance, retries with synonym mapping, or returns a graceful "could you rephrase?" message.
- **Entity extraction multi-agent architecture (LangGraph)**: Five agents — `ColumnNameAgent` (parses column descriptions) → `ChainCreationAgent` (builds LLM extraction chain + JSON parser) → `EntityExtractionAgent` (applies NER-style extraction over review text) → `DataCombinationAgent` (merges structured + extracted data via pandas) → `DatabaseAgent` (persists merged data to in-memory SQLite via SQLAlchemy).
- **Performance metrics framework**: Exact match accuracy, Execution accuracy, Component-level accuracy, Query execution success rate, Semantic equivalence/canonicalization, Human evaluation, Latency/throughput — each capturing a different dimension of correctness/robustness that no single metric covers alone.

## Key Concepts
- **Schema linking**: Mapping natural language phrases (synonyms, abbreviations, business terminology) to exact schema elements — explicit (direct match), implicit (inferred from context), fuzzy (vague/ambiguous references).
- **GenAI readiness**: Whether an organization's database has clean documentation, consistent naming, and annotated metadata — without it, even powerful LLMs produce invalid/inaccurate SQL.
- **Domain generalization gap**: Models fine-tuned on benchmarks (Spider, WikiSQL) perform well academically but degrade on real-world schemas with unseen names, industry terminology, and normalized relational structures.
- **Row summarization vs. table summarization**: Row summarization narrates a single retrieved record (e.g., "Smartwatch X was the best-selling product in April..."); table summarization narrates aggregate trends/outliers across a full result set (e.g., "Sales increased steadily... peaking in Q4...").
- **Static, schema-aware, and logical validation**: Static analysis (SQL parser syntax check) → schema-aware validation (do referenced tables/columns exist) → logical validation (sandbox/read-only execution to catch semantically-wrong-but-syntactically-valid queries).
- **Multi-turn dialogue context**: Follow-up queries ("now break that down by region," "exclude products with returns over 10%") require maintaining evolving query state/filters across turns — a memory-aware, not just single-shot, problem.
- **Governance controls**: Row/column-level access restriction, PII/financial field masking, query audit logging — required before enterprise deployment, independent of model accuracy.

## Mental Models
- Treat text-to-SQL as "NL-to-structured-intent-to-SQL" — the hard part is rarely SQL syntax generation itself, it's correctly resolving ambiguous human language into the *exact* structured intent the schema can satisfy.
- Think of the 13-part pipeline as a checklist for production-readiness — a text-to-SQL demo that only does prompt→SQL generation is missing validation, fallback, governance, and observability, all of which are necessary for enterprise deployment.
- Use execution accuracy as the default production metric (does it return the right *result*?) and exact-match accuracy as the default research/benchmark metric (does it match the *reference query* syntactically?) — they answer different questions and are not interchangeable.
- The fallback cascade (clarify → retry-with-refinement → simplify → redirect-to-docs → template-default) mirrors a customer-support escalation ladder — always have a next, less-ambitious response ready rather than a single hard failure.

## Anti-patterns
- **Assuming a benchmark-tuned model will generalize to enterprise schemas unchanged**: domain-specific terminology, naming conventions, and normalized relational designs routinely break out-of-the-box performance — domain adaptation (RAG, schema pre-embedding, few-shot domain examples) is required.
- **Executing LLM-generated SQL directly against production without validation**: risks performance degradation, data corruption (on write operations), or privacy violations — always validate (static + schema-aware + logical) and prefer read-only sandboxes first.
- **Using exact-match accuracy as the sole evaluation metric**: it over-penalizes semantically correct SQL with different structure (different join order, aliasing) — combine with execution accuracy and component-level accuracy.
- **Ignoring multi-turn context in conversational analytics interfaces**: treating each follow-up as an independent query breaks pronoun/ellipsis resolution ("which of them had declining sales?") — requires dialogue-state tracking.
- **Skipping governance controls for convenience**: generating SQL that bypasses row/column-level access restrictions or fails to mask PII is a compliance and security failure, not just a quality one.

## Code Examples
```python
try:
    sql = generate_sql(natural_query)
    result = execute_sql(sql)
except SQLValidationError:
    sql = regenerate_with_schema_guidance(natural_query)
    result = execute_sql(sql)
except TableOrColumnNotFound:
    sql = retry_with_synonym_mapping(natural_query)
    result = execute_sql(sql)
except Exception:
    return "Sorry, I couldn't find what you're asking for. Could you rephrase?"
```
- **What it demonstrates**: A layered fallback/retry engine handling validation errors, schema mismatches, and unrecoverable failures with progressively more conservative strategies.

```python
template = """You need to act as a Named Entity Recognizer.
Extract the following column names from the review text:
{column_names}
...
STRICTLY respond in JSON format like: {"column_1": "<value 1>", ... }
"""
```
- **What it demonstrates**: A role-specific, structured-output prompt used by `ChainCreationAgent` to extract named entities from unstructured review text before merging into a SQL-queryable table.

```python
class GraphState(TypedDict):
    question: str
    ...
# ColumnNameAgent -> ChainCreationAgent -> EntityExtractionAgent -> DataCombinationAgent -> DatabaseAgent
workflow = StateGraph(GraphState)
graph = workflow.compile()
```
- **What it demonstrates**: A five-agent LangGraph pipeline transforming unstructured review text into an SQL-accessible in-memory database, with conditional entry points and retry branches.

## Reference Tables
| Pipeline stage | Purpose |
|---|---|
| Prompting strategy (zero/few-shot/CoT) | Condition the LLM to translate NL into SQL correctly |
| Schema/metadata integration | Flat listing, structured encoding (JSON/graph), or semantic mapping |
| Intermediate planning/decomposition | Generate NL plan before SQL, enabling stage-wise validation |
| Row/table summarization | Narrate single-record or aggregate-trend results in natural language |
| SQL validation/safety | Static, schema-aware, and logical (sandboxed) checks |
| Tool-augmented extensions | Function-calling for schema lookup, vector search for ambiguous columns |
| System integration | Cloud API vs. self-hosted model; caching; modularity |
| Query clarification | Present disambiguation choices to user when multiple interpretations exist |
| Governance/compliance | Field masking, row/column access control, IAM integration |
| Fallback/retry | Clarification, prompt refinement, simplified query, doc redirect, default templates |
| Deployment | Embedded API, self-hosted (e.g., SQLCoder), hybrid agentic (LangChain) |
| Observability | Query logs, confidence scores, failure patterns, latency telemetry |
| Feedback loop | Correction interface, logging/analytics for retraining |

| Metric | What it measures | Best for |
|---|---|---|
| Exact match accuracy | String-identical match to reference SQL | Benchmarks with fixed schema (Spider) |
| Execution accuracy | Same result set as reference, regardless of structure | Production systems (result correctness) |
| Component-level accuracy | Per-clause correctness (SELECT/WHERE/JOIN/etc.) | Debugging, iterative model improvement |
| Query execution success rate | % of queries that run without error | Production health monitoring |
| Semantic equivalence/canonicalization | Functional identity despite syntactic difference | Advanced eval when execution testing impractical |
| Human evaluation | Expert-reviewed correctness/clarity/relevance | Pilot deployments, ambiguous-case resolution |
| Latency/throughput | Response time, queries/sec | UX optimization, SLA compliance |

## Worked Example
Entity-extraction-to-SQL pipeline: unstructured product review text is processed by a 5-agent LangGraph workflow. `ColumnNameAgent` parses a user-supplied column description string into a semantic dictionary (`{"Name": "<Name of the customer>"}`). `ChainCreationAgent` builds a LangChain prompt+LLM(llama3.2 via Ollama)+JSON-parser chain instructed to act as a Named Entity Recognizer and return strict JSON. `EntityExtractionAgent` applies this chain row-by-row over the `ReviewText` column, producing structured dictionaries. `DataCombinationAgent` merges these extracted fields with the original structured table (`df1`) via pandas join on inferred keys. `DatabaseAgent` writes the merged result into an in-memory SQLite database (via SQLAlchemy `StaticPool`), making it immediately SQL-queryable. The graph supports conditional entry points (based on column-description availability) and retry branches for robustness.

## Key Takeaways
1. Text-to-SQL's hardest problems are schema linking and domain generalization, not raw SQL syntax generation.
2. A production text-to-SQL system needs the full 13-stage pipeline (prompting → schema integration → validation → fallback → governance → observability), not just prompt-to-SQL translation.
3. Always validate generated SQL in three layers (static, schema-aware, logical/sandboxed) before executing against real data.
4. Execution accuracy, not exact-match accuracy, is the right default metric for production correctness; use multiple metrics together for a full picture.
5. Layered fallback/retry (clarify → refine → simplify → redirect → template) keeps the system usable when the primary generation path fails.
6. Entity extraction (via LLM-based NER) is a practical technique for turning unstructured text into SQL-queryable structured tables, bridging unstructured and structured data worlds.
7. Text-to-SQL is as much an organizational/governance challenge (access control, PII masking, audit logging) as a modeling challenge.

## Connects To
- **Ch 2**: Directly implements the "text-to-SQL" output classification introduced in the multimodal-systems chapter.
- **Ch 6**: The fallback/retry cascade and validation layers echo the grading-mechanism and multi-stage RAG failure-handling patterns.
- **Ch 12-13**: CoT-based intermediate planning (generate NL plan before SQL) directly reuses the reasoning-scaffolding techniques from those chapters.
- **Ch 15**: Explicitly previewed as the next chapter — agentic, multi-query text-to-SQL systems with multi-turn dialogue, join reasoning, and query decomposition.
