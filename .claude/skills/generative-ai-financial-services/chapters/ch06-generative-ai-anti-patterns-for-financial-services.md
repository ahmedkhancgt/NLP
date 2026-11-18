# Chapter 6: Generative AI Anti-Patterns for Financial Services

## Core Idea
Five recurring anti-patterns derail GenAI projects in financial services: weak use-case qualification, chasing "the best model" instead of the right-fit model, choosing the wrong implementation approach, lacking data governance, and performing incomplete cost analysis — all are failures of working backward from business value and full-lifecycle thinking.

## Frameworks Introduced
- **Use-case qualification test**: before building, ask (1) Is this even a good fit for AI/ML? (deterministic/rule-based problems like regulatory transaction reporting or accounting belong in traditional programs, not ML); (2) Is it a good fit for ML but *not* GenAI? (forecasting/time-series/churn prediction and credit-risk prediction — needing explainability/reproducibility — are better served by ARIMA, Prophet, RNN/LSTM, or explainable models than by GenAI).
  - When to use: as the very first gate on any proposed GenAI project.
- **Technical vs. business success metrics**: define both before building.
  - Technical: Accuracy, Precision/Recall/F1, Confusion matrix, MAE/RMSE, ROUGE, BERTScore, Class Imbalance (CI), Difference in Positive Proportion in Predicted Labels (DPPL).
  - Business: ROI, customer satisfaction, increased revenue, operational efficiency, cost reduction, time to market, employee satisfaction.
  - How: work backward from business goals to find the metric that explains value to stakeholders, then derive technical objectives from it.
- **Approach-selection decision framework** (expands Ch3's spectrum from prompt engineering → RAG → agentic → fine-tuning → full training): explicit when-to-use / when-to-avoid criteria per approach (see Reference Tables below).
- **Data governance lifecycle framework**: Data ingestion & storage → Data preparation → Model deployment → Monitoring & updates, each with concrete controls (catalog/metadata, masking/anonymization, ABAC access, Guardrails for compliance checks, drift/hallucination monitoring).
- **Holistic cost analysis**: must span data prep, model training/fine-tuning, infrastructure, ongoing monitoring/evaluation/retraining/user training — not just the initial model acquisition cost (TCO, detailed further in Ch14).

## Key Concepts
- **Anti-pattern**: a common response to a recurring problem that is ineffective/counterproductive and causes more issues than it solves.
- **Compound AI system**: modern AI applications are chains of multiple LLM calls, RAG systems, and agents (via LangChain/LlamaIndex) rather than one monolithic model — so "one best model for everything" is itself a flawed premise.
- **Attribute-Based Access Control (ABAC)**: access control mapped to data's use-case and velocity (vs. uniform role-based lockdown), allowing open access to static reporting data while still restricting fast-moving transactional data.
- **Data producer constraints**: source-system limitations in data freshness, privacy enforcement (masking at source), and metadata capture that — if ignored — pollute training sets or cause hallucination on stale data.
- **TCO (Total Cost of Ownership)**: the full cost footprint of a GenAI system beyond initial acquisition (detailed in Ch14).

## Mental Models
- "A hammer will not work when you need a screwdriver — one LLM does not rule them all": resist central-IT mandates to standardize on a single model for every use-case; different components of a compound system need different models.
- Treat the approach spectrum (prompt engineering → RAG → agentic → fine-tuning → full training) as a ladder you climb only when the previous rung's limitations are proven, not a menu you pick from based on hype.
- Rate data sources by trustworthiness/determinism rather than treating any single prior output or incomplete dataset as ground truth — especially in credit risk, where statement history alone doesn't capture full financial health.
- Think about cost across the full lifecycle (build → scale-to-production → operate) — a cost analysis that stops at the POC stage will be wrong by the time the system is in production at scale.

## Anti-patterns
- **Weak use-case qualification**: building GenAI for deterministic, rule-based problems (e.g., regulatory transaction reporting, accounting) where consistent, accurate, non-probabilistic output is required — ML/GenAI's inherent randomness makes it unsuitable; also applies GenAI where traditional ML (ARIMA, Prophet, RNN/LSTM) or explainable models already perform better (forecasting, churn, credit-risk prediction).
- **The best model (chasing SOTA)**: over-indexing on picking "the best" model rather than the right-fit model for the business problem — ignores that (a) smaller/cheaper models often suffice for simple tasks (e.g., complaint classification with Llama 2 7B/13B), (b) public benchmarks don't reflect your specific data, (c) "best" changes every few months, (d) fine-tuned smaller models can beat large generic ones, and (e) compound systems need different models per step, not one model for all.
- **Central-IT one-model mandate**: forcing every business problem/use-case onto a single approved model restricts innovation and kills projects via high cost or poor fit — the single biggest named anti-pattern in this section.
- **Not choosing the right implementation approach**: using prompt engineering when the task needs search over large document sets (should use RAG); using RAG when data doesn't fit/is expensive in context window (should fine-tune); using agentic workflows for simple deterministic tasks (should use prompt engineering/RAG, or traditional workflow tools, since agent non-determinism compounds errors across chains); using fine-tuning when a pre-trained FM already performs well, or when there isn't enough quality labeled data, or when cost doesn't justify the investment.
- **Implementing a single access pattern** (data anti-pattern): applying the same strict access lockdown meant for transactional data to static reporting data, unnecessarily restricting model training and missing insights.
- **Not identifying data producer constraints** (data anti-pattern): ignoring source-system limits on data freshness, privacy masking, and metadata capture — leads to junk data in training sets or hallucination on stale information.
- **Over-reliance on incomplete data** (data anti-pattern): treating prior model outputs, non-deterministic processes, or partial data (e.g., credit risk from statement history alone) as ground truth — reinforces existing bias/inaccuracy instead of curating training data by trustworthiness/determinism.
- **Not looking at holistic costs** (cost anti-pattern): underestimating TCO by ignoring data prep, training, infra, and maintenance costs beyond initial acquisition.
- **Not appreciating the scale of production** (cost anti-pattern): POC-stage costs that balloon unpredictably once deployed at production scale/demand.
- **Not selecting a cost-optimal LLM** (cost anti-pattern): ignoring compute/memory/licensing cost differences across LLM choices, leading to unnecessary spend.
- **Overlooking operating costs** (cost anti-pattern): underestimating ongoing monitoring, evaluation, retraining, and user-training costs after deployment.

## Code Examples
No code examples in this chapter — it is a pattern-recognition/decision-framework chapter.

## Reference Tables
**When to use vs. avoid each implementation approach**

| Approach | Use when | Avoid when |
|---|---|---|
| Prompt engineering | Exploring feasibility; task explainable in a few examples; info fits context window; task doesn't need external/proprietary info; FM already pre-trained on needed language/skill | Info lives in a large document set needing search (→ RAG); prompt growing too large/expensive (→ RAG/fine-tuning); need new domain language or skill (→ fine-tuning) |
| RAG | Q&A over proprietary/internal docs not in training data; answers live in a few localized document parts; data changes frequently; answers verifiable against source | Learning a wholly new language/skill not describable in a few docs (→ fine-tuning); info doesn't fit context window; context growing too large (latency/cost); non-text/image modalities |
| Agentic workflows | Task needs external function/API/tool calls + interpreting responses; task decomposes into sub-tasks as agents/tools; task needs self-reflection/CoT to decide/improve actions | Task is simple enough for prompt engineering/RAG; task needs predictable, repeatable chains of reasoning (agent non-determinism is a liability); long agent chains/graphs compound error rates |
| Fine-tuning | Adapting to new domain language (legal, financial); adding new/rare skills; aligning to preferences via RLHF/DPO; optimizing inference cost with a smaller tuned model | A pre-trained FM already performs well; insufficient/low-quality labeled data for the task/domain; cost/scale doesn't justify investment |

**Technical vs. business success metrics** — see Key Concepts above for the full enumerated lists; use technical metrics to validate the model, business metrics to justify the investment.

## Worked Example
**Central-IT single-model mandate failure**: An FSI's central IT team mandates one approved LLM for every GenAI use-case to simplify governance/control. A complaint-classification use-case that would run fine and cheaply on a 7B-13B model is forced onto the mandated large general-purpose model, inflating cost; meanwhile a use-case needing deep financial-domain fluency (where a fine-tuned smaller model would outperform) is stuck with a generic model that underperforms. Both projects either get cancelled for cost/performance reasons or deliver poor ROI — illustrating why approach/model choice must be made per use-case, working backward from business metrics, not centrally standardized.

## Key Takeaways
1. Qualify every use-case against the "is this even right for AI/ML, and specifically for GenAI" test before scoping work.
2. Define both technical and business success metrics up front, working backward from business goals.
3. Never mandate a single model across all use-cases — pick per-component, per-use-case based on evaluation on your own data.
4. Follow the approach ladder (prompt engineering → RAG → agentic → fine-tuning → full training) and only escalate when the current rung's documented limitations actually bite.
5. Build a full-lifecycle data governance framework (ingestion/storage → preparation → deployment → monitoring) using ABAC and source-constraint awareness, rather than uniform access rules or blind trust in prior outputs.
6. Do holistic cost analysis (TCO) spanning data prep, training, infra, and ongoing operations before committing budget — and revisit it as the system scales to production.

## Connects To
- **Ch3**: This chapter directly extends the implementation-approaches spectrum with concrete when-to-use/avoid criteria.
- **Ch5**: Many anti-patterns here are failure responses to the challenges named in Ch5 (reliability, data privacy, legacy integration).
- **Ch7**: Architectural considerations chapter addresses how to design systems that avoid these anti-patterns structurally.
- **Ch12**: Formalizes "choosing the right implementation approach" into a complete decision framework.
- **Ch14**: TCO/cost analysis anti-pattern is deep-dived in FMOps practices.
