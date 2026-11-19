# Patterns

## Prompt Engineering with Base Model
**When to use**: Exploring feasibility quickly; task explainable with few examples; info fits context window; task doesn't need external/proprietary data; FM already pre-trained on needed language/skill (e.g., classification, sentiment analysis, entity extraction).
**How**: Structure prompt with Task, Context, Input, Examples (optional, few-shot), Output format (optional). Apply techniques as needed: zero-shot → few-shot → Chain of Thought (CoT, for reasoning) → ReAct (for tool-use + reduced hallucination) → Self-consistency (multiple reasoning paths, pick most consistent) → Tree of Thought (multi-branch reasoning for hardest problems).
**Trade-offs**: Lowest TCO and fastest time-to-market, but highest hallucination risk and lowest response quality for domain-specific tasks; prompts must be rewritten per model change.

## Retrieval Augmented Generation (RAG)
**When to use**: Q&A over proprietary/internal documents not in training data; answers localized in a few document parts; data changes frequently; answers need source verification. (Employee onboarding, contact-center assist, enterprise search, FAQ chatbots, financial report analysis.)
**How**: Data ingestion workflow — chunk documents (fixed-size or context-aware) → embed each chunk → store in vector DB (ChromaDB, Pinecone, OpenSearch, Weaviate, Faiss, Chroma). Response generation workflow — embed user query with same embedding model → ANN similarity search (cosine/Euclidean/dot product; HNSW/LSH/product quantization) → merge retrieved context with query → prompt FM → generate grounded answer. Improve with: larger embedding dimensions, semantic chunking, metadata filters, query transformation, re-ranking.
**Trade-offs**: Higher quality/lower hallucination than prompt engineering, but higher TCO and operational maintenance from many moving parts (embedding model, vector store, chunking strategy all need coordinated updates).

## Agentic Workflows
**When to use**: Task needs external function/API/tool calls + interpreting responses; task decomposes into sub-tasks as tools/agents; task needs self-reflection/CoT to decide or improve actions (KYC automation, credit assessment, ESG analysis, investment research, fraud analytics).
**How**: Agent loop = Plan (build action plan via CoT/ReAct/ToT) → Execute (call tools: DB, API, vector store, calculator, web search) → Memorize (retain task/steps/outcomes via buffered memory or persistent store). Equip with multiple tools and clear natural-language tool descriptions encoding priority/fallback logic (e.g., "try vector store first, then database").
**Trade-offs**: Enables full business-process automation with limited human oversight, but non-deterministic (same input can execute differently across runs), compounds error rate across long agent chains, and requires significant tool-access engineering effort in regulated financial environments.

## Fine-Tuning (Full / PEFT via LoRA, QLoRA, DoRA)
**When to use**: Adapting to new domain language (financial, legal); adding new/rare skills; aligning to preferences via RLHF/DPO; optimizing inference cost with a smaller tuned model (marketing copy, product ideation, summarization, compliance reporting, domain adaptation).
**How**: Unsupervised (no labels, next-token prediction on raw domain corpus — e.g., SEC 10-K filings) or Supervised/instruction fine-tuning (labeled input/output pairs for a specific task). Use PEFT (freeze base weights, train small added parameter set) via LoRA (rank-decomposition matrices), QLoRA (adds 4-bit quantization), or DoRA (decomposes weights into frozen general-knowledge + tuned task-specific components) instead of full fine-tuning to cut cost/time ~10,000x fewer trainable params.
**Trade-offs**: Higher quality/domain fluency than RAG, better data privacy (no context exposure), but requires strong ML skills, high TCO, slow iteration, and risks catastrophic forgetting (full fine-tuning) or per-task-only quality gains (requiring separate tuned variants per task).

## Training a Foundation Model from Scratch
**When to use**: Organization wants to become its own model provider; needs an industry-specific base model; government/confidential-services isolation requirements (rare, high-ROI-justification-only scenarios, e.g., BloombergGPT).
**How**: Full generative AI lifecycle — data collection/curation (hundreds of billions to trillions of tokens) → tokenization → training (massive GPU clusters, weeks to months) → evaluation against benchmark datasets.
**Trade-offs**: Highest achievable quality/control, but highest TCO (~millions of dollars), slowest time-to-market (months), highest operational maintenance, and cannot update knowledge without producing an entirely new model version.

## Hybrid Approach (Fine-tune then RAG/Prompt Engineer)
**When to use**: Want domain fluency plus flexibility/freshness across many downstream applications; mature GenAI programs with budget to amortize fine-tuning cost across many use-cases.
**How**: Fine-tune a base model once on organizational data; use that fine-tuned model as the new "base" for prompt-engineering and RAG applications built on top.
**Trade-offs**: Outperforms any single approach in isolation on quality, at the added one-time (and ongoing refresh) cost of fine-tuning.

## Guardrails (Prompt Engineering / External Tools / HITL)
**When to use**: Every production GenAI application needs input AND output filtering against responsible AI policy (bias, toxicity, PII, brand language, security).
**How**: Prompt engineering guardrails (safety-focused FM like Llama Guard classifies safe/unsafe before/after main FM call) for flexibility; external tools (NeMo-Guardrails, Guardrails AI, Bedrock Guardrails, Azure AI Content Safety) for consistency; HITL review for low-confidence decisions (both false positives and negatives) — not for real-time use-cases.
**Trade-offs**: Combine all three in layers; prompt-only guardrails are flexible but inconsistent/noisy; external tools add setup overhead but reliability; HITL adds latency unsuitable for live interactions.

## Multi-Tenancy Generative AI Platform
**When to use**: Need centralized governance, cost tracking/FinOps, usage metering, throttling, and protection against LoBs adopting risky/unvetted models.
**How**: Build a central GenAI API/gateway exposing a curated set of vetted FMs to all LoBs as tenants; implement isolation, cost tracking per tenant, usage metering, throttling (RPM/TPM quotas), and centralized model monitoring/governance.
**Trade-offs**: Gains governance/cost control but faces the "lowest common denominator" problem (can only expose features all supported models share), scaling bottlenecks at high tenant counts, and requires specialized platform-engineering skills that are scarce.

## Single-Tenancy (LoB-Owned) Generative AI Stack
**When to use**: Need LoB autonomy, fast independent time-to-market, use-case-specific monitoring tuning.
**How**: Each LoB independently deploys and manages its own GenAI application stack and model access, with only baseline security/auth/compute governed centrally.
**Trade-offs**: Faster, more autonomous delivery per team, but leads to duplicated components, fragmented governance, and "shadow IT" as LoBs rebuild IT infrastructure independently.

## Anti-Patterns Reframed as "What Not to Do"
- **Don't** force a single model or single implementation approach onto every use-case (central-IT one-model mandate) — evaluate per use-case against the 9-criterion scorecard (Ch12).
- **Don't** apply GenAI to deterministic, rule-based problems (regulatory reporting, accounting) where consistent non-probabilistic output is required — use traditional programming.
- **Don't** apply GenAI where traditional ML already excels (forecasting, time-series, credit-risk prediction needing explainability) — use ARIMA/Prophet/RNN/LSTM or explainable models instead.
- **Don't** apply a single access-control pattern to all data regardless of velocity (transactional vs. reporting) — use ABAC mapped to use-case and velocity.
- **Don't** treat prior model outputs or incomplete data sources as ground truth (e.g., credit risk from statement history alone) — rate sources by trustworthiness/determinism and curate accordingly.
- **Don't** underestimate TCO by looking only at model-invocation cost — account for compute, storage, human, and tooling costs across the full lifecycle.
- **Don't** let unchecked vector-store embeddings bypass source-document access permissions — re-attach permissions as chunk/embedding metadata and enforce at retrieval time.
- **Don't** mimic DevOps processes wholesale for FMOps — GenAI-specific needs (data/model versioning, bias/toxicity checks, prompt catalogs) require adaptation, not direct reuse.
