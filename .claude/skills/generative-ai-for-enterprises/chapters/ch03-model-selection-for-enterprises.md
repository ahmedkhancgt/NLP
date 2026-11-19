# Chapter 3: Model Selection for Enterprises

## Core Idea
Model selection is a strategic, multi-criteria funnel decision — not a technical afterthought — that must weigh regulatory compliance, cost, performance, governance, and sustainability simultaneously, and it continues as an ongoing cycle of evaluation after initial selection.

## Frameworks Introduced
- **The Model Selection Funnel**: Many candidate LLMs enter as input; a structured set of decision points (Ds) filters them down to one or more models that fit enterprise needs. Decision outputs can yield more than one surviving model.
  - When to use: Whenever comparing multiple candidate LLMs/SLMs for an enterprise application.
  - How: Run every candidate model through the full checklist of ~20 questions below; models failing hard constraints (e.g., regulatory compliance) are eliminated first, then remaining models are ranked on soft criteria (cost, accuracy, ease of collaboration).
- **21-Question Selection Criteria Checklist**: Regulatory compliance, time to market, CapEx vs. OpEx, latency/throughput, cost per 1000 tokens, ease of governance, scalability, data proximity, security, managed vs. unmanaged, automation, availability, resiliency, hosting, virtualization, sustainability, hardware, software licensing, ease of collaboration, inference task fit, accuracy, upgrade path, technical debt, technical skills, open vs. closed-source, decommissioning, indemnity, limits (context length/rate limits), LLM vs. SLM.
  - When to use: As the operational due-diligence tool during vendor/model evaluation.
  - How: Answer each question per candidate model; questions map to regulatory, economic, technical-governance, and sustainability dimensions.
- **Post-Selection Lifecycle (Integration → Optimization → Governance → Continuous Evolution)**: Selection is the start, not the end, of the journey.
  - When to use: Immediately after model selection, to plan the adoption roadmap.
  - How: (1) Integrate with legacy systems via APIs/middleware; (2) optimize via fine-tuning/evaluation balancing accuracy/speed/cost; (3) establish governance/compliance monitoring; (4) continuously reassess as new model versions and business needs emerge.

## Key Concepts
- **Model selection**: Identifying the right model that fits enterprise purpose.
- **Selection criteria**: The structured set of questions/decision points used to evaluate candidate models.
- **Funneling of models**: Taking multiple candidate models and narrowing to one or more final fits.
- **LLM vs. SLM**: Choice between a general large language model and a small language model trained for a specific task.
- **Indemnity**: Legal protection question — whether the organization is protected for model output completions.
- **Stakeholder management**: The deliberate engagement of IT, data governance, compliance, and business stakeholders early in the selection process to secure buy-in.

## Mental Models
- Think of selection as a funnel, not a single filter: hard constraints (regulatory, security) eliminate candidates first; soft constraints (cost, UX, collaboration ease) rank survivors.
- Use "selection is a journey, not a destination" — treat the chosen model as a living asset requiring continuous evaluation, not a final purchase decision.
- Communication and stakeholder management are the "social infrastructure" layer underneath the technical selection criteria — without it, even a technically correct model choice can fail on adoption.

## Anti-patterns
- **Optimizing purely for accuracy/capability while ignoring regulatory compliance or data security**: Creates downstream legal exposure that can void the entire initiative.
- **Treating model selection as a one-time decision**: Fails to account for the fast pace of model improvements and evolving business needs.
- **Skipping stakeholder engagement**: Technically sound model choices can fail to gain organizational adoption without early buy-in from IT, compliance, and business units.
- **Ignoring sustainability and technical debt questions**: Short-term model choices that ignore GPU carbon footprint or upgrade paths create costly rework later.

## Worked Example
The chapter frames selection as a funnel: multiple candidate LLMs enter, and the 21-question checklist (spanning regulatory, economic, governance, sustainability, and technical criteria) acts as sequential filters. A healthcare enterprise, for instance, would immediately filter on regulatory compliance (HIPAA) and data proximity (co-locating data with the model), which might eliminate several globally-hosted closed-source options; from the remaining candidates, cost-per-1000-tokens and context-length limits would further narrow the field, and ease-of-governance/automation tooling would determine the final pick(s) — potentially leaving more than one viable model for different use cases (e.g., one for customer-facing chat, another for internal document summarization).

## Key Takeaways
1. Treat model selection as a funnel of ~20 weighted criteria spanning regulatory, economic, technical, and sustainability dimensions — not a single "best model" search.
2. Regulatory compliance and data security are typically hard constraints that should filter candidates before cost/performance comparisons begin.
3. CapEx vs. OpEx trade-offs and time-to-market considerations are as strategically important as raw model accuracy.
4. More than one model may legitimately survive the funnel — different models can serve different enterprise needs simultaneously.
5. Selection is the start of an ongoing cycle: integration, optimization, governance, and continuous re-evaluation follow and matter as much as the initial pick.
6. Communication and stakeholder management are not optional extras — they determine whether a technically sound model selection actually gets adopted.

## Connects To
- **Ch 2**: Directly expands the "model sourcing" and "model selection" dimensions introduced in the eleven-dimension production framework.
- **Ch 4**: The selected model's deployment pattern (cloud/on-premises/hybrid) is the next decision explored in depth.
- **Ch 5**: Data proximity and co-location criteria here are elaborated into the full Operating System demarcation model.
- **Ch 11**: CapEx/OpEx and cost-per-token criteria here feed directly into the dedicated Cost Optimization Strategies chapter.
