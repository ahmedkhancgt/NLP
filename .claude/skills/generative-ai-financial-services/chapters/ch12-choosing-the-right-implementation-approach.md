# Chapter 12: Choosing the Right Implementation Approach

## Core Idea
No single generative AI implementation approach (prompt engineering, RAG, agentic workflows, fine-tuning, training from scratch) wins on every criterion — each approach trades off differently across 9 evaluation criteria (4 general + 5 financial-services-specific), so the right choice is use-case-dependent, and a hybrid approach (fine-tune, then layer RAG/prompt engineering on top) often beats any single approach in isolation.

## Frameworks Introduced
- **9-criterion qualitative evaluation framework (low/medium/high, comparative not absolute)**:
  - *General criteria*: Technical complexity/skills needed, Quality of responses, Total Cost of Ownership (TCO), Time to market, Operational maintenance.
  - *Financial-services-specific criteria*: Probability of hallucination, Toxicity and bias tolerance, Data privacy and security, Compliance effort.
  - When to use: as a structured scorecard to compare approaches for a specific use-case rather than relying on hype or generic "best practice" claims.
- **Hybrid approach**: fine-tune a base model on organizational data first, then apply prompt engineering and/or RAG on top of that fine-tuned model.
  - When to use: when you want the domain fluency of fine-tuning plus the flexibility/freshness of RAG/prompt engineering — trades an added fine-tuning maintenance overhead for better overall performance than any single approach alone.
- **Indemnity** (noted as a cross-cutting but excluded-from-scoring criterion): legal protection a model provider offers against toxic/leaked outputs — varies by provider (e.g., Azure/AWS/Google Cloud offer it for some models) and is hard to measure qualitatively, so the book leaves it out of the main comparison.

## Key Concepts
- **Comparative (not absolute) rating scale**: a "low" rating for an approach on a criterion means low *relative to the other four approaches*, not low in absolute/universal terms.
- **TCO (full-solution cost)**: measures the entire solution's cost, not just the foundation model's inference cost.
- **Compliance effort**: legal/compliance review of a model provider's EULA against jurisdiction-specific regulations — scales with the number of distinct models/providers involved (e.g., RAG needs two EULAs reviewed: embedding model + foundation model).

## Mental Models
- Treat approach selection as a 9-dimension trade-off matrix, not a ranking — pick the approach whose strengths align with your use-case's most important criteria (e.g., hallucination tolerance matters enormously for customer-facing financial advice, less for internal brainstorming).
- "There is no clear winner" — resist the urge to find one universally best approach; the right question is "which criteria matter most for *this* use-case."
- Use the hybrid pattern as the default assumption for mature financial-services GenAI deployments: fine-tune once for domain fluency, then reuse that fine-tuned model as the base for many RAG/prompt-engineering applications — amortizing the fine-tuning cost across many use-cases.

## Anti-patterns
- Assuming one approach is universally superior and forcing every use-case through it (echoes Ch6's "central IT one-model mandate" anti-pattern, now generalized to "one-approach mandate").
- Ignoring that **changing the embedding model in a RAG system requires re-embedding/replacing the entire vector store** — an operational-maintenance trap if not planned for.
- Ignoring that **fine-tuning a model for one task degrades its quality on other tasks** — requiring separate fine-tuned versions per task, which multiplies operational maintenance burden.
- Treating agentic workflows as low-risk just because they "restrict the model to given tools" — multi-agentic workflows can become very hard to maintain as each agent gets its own lifecycle, and changing the underlying model requires extensive regression testing of the whole business process.
- Assuming training from scratch solves domain-knowledge-freshness problems — it is the *worst* approach for use-cases needing frequently updated information, since updating knowledge requires producing and publishing an entirely new model version.

## Code Examples
No code in this chapter — it is a comparative decision-framework chapter (ratings expressed qualitatively, not quantitatively).

## Reference Tables
**Full comparative evaluation across approaches (synthesized from Ch12's per-approach analysis)**

| Criterion | Prompt engineering | RAG | Agentic | Fine-tuning | Train from scratch |
|---|---|---|---|---|---|
| Technical complexity | Low | Medium-High | High (production) | High | Highest |
| Quality of responses | Low-Medium | Medium-High | ~Same as RAG | High | Highest |
| TCO | Low | Medium-High | Medium-High (tool orchestration offsets automation savings) | High | Highest |
| Time to market | Low | Medium-High | Medium-High (non-technical approval delays) | Medium-High | Highest |
| Operational maintenance | Low | Medium-High | High (tool lifecycle management) | High | Highest |
| Probability of hallucination | High | Low | ~Same as RAG (CoT adds some non-determinism) | Low-Medium | Lower (but never zero) |
| Toxicity/bias tolerance | Low | Medium-High | Medium-High | High | Highest (achievable) |
| Data privacy/security | Low (context-exposed) | Medium-High (vector-store ACLs) | High (tool-restricted) | High (no context exposure) | Highest |
| Compliance effort | Low-Medium | Medium (2 models' EULAs) | Medium-High (human-review mandates for critical decisions) | Low-Medium (same as prompt engineering) | High |

**When to choose each approach (condensed from Tables 12.1-12.5)**

| Approach | Best for | Key challenge |
|---|---|---|
| Prompt engineering | Sentiment analysis, text classification, entity extraction | High hallucination risk, low response quality, prompts must be rewritten if model changes |
| RAG | Employee onboarding, contact-center assist, enterprise search, HR/policy FAQ, customer FAQ, SOP creation, financial report analysis | High TCO/operational maintenance from many moving parts; hard to compare competing RAG architectures |
| Agentic workflows | KYC automation, credit assessment, contact-center automation, SDLC automation, investment research reports, ESG analysis, product recommendations, fraud analytics | Fast-moving/immature tech, hard to add human-in-the-loop, few models excel at agentic reasoning, costly model-change regression testing |
| Fine-tuning | Marketing campaigns, product ideation, summarization, SOP/compliance reporting, any domain-adaptation need | Requires strong ML skills, high TCO, hard to update knowledge without re-tuning, degrades on tasks outside the tuned domain |
| Train from scratch | Organization-wide base model, industry-specific models, government/confidential-services use-cases | Highest TCO (~millions of dollars) and time (~months), highest operational maintenance, cannot update knowledge without a new model version |

## Worked Example
**Hybrid approach pattern**: A financial organization first fine-tunes a base model (e.g., Llama 3) on its own curated financial-domain corpus (similar to Ch10's SEC-filing fine-tuning) — this becomes the organization's new "base model." It then builds multiple downstream applications on top of this fine-tuned base using prompt engineering (for simple tasks) and RAG (for document-grounded Q&A), rather than building each application on the generic off-the-shelf FM. The result: every RAG/prompt-engineering application built this way inherits the fine-tuned model's domain fluency, improving quality of responses across the board, at the one-time added cost of fine-tuning and keeping that fine-tuned model current.

## Key Takeaways
1. Evaluate any candidate approach against the same 9-criterion scorecard (4 general + 5 FSI-specific) rather than deciding based on a single dimension like "quality" or "cost" alone.
2. Prompt engineering wins on speed/cost but loses badly on hallucination risk and response quality for financial-domain questions — save it for simple classification-style tasks.
3. RAG offers the best cost/quality balance for most financial-services information-retrieval use-cases, at the price of operational complexity from many moving components.
4. Fine-tuning is the right call specifically for domain-adaptation needs, but multiplies maintenance burden if you need many task-specific fine-tuned variants.
5. Training from scratch should be reserved for organizations that effectively want to become their own model provider (or need government/confidential-use isolation) — it is the most expensive, slowest option by far on every cost/time criterion.
6. A hybrid approach (fine-tune once, then RAG/prompt-engineer on top) frequently outperforms any single approach used in isolation and is the pattern to default toward as GenAI maturity grows.
7. Indemnity (provider legal protection) is a real but hard-to-quantify cross-cutting consideration worth checking per model provider, even though it's excluded from the formal scorecard here.

## Connects To
- **Ch3**: This chapter directly re-evaluates the five approaches introduced there with concrete, scored criteria.
- **Ch6**: Generalizes the "don't force one model/approach on every use-case" anti-pattern into a formal comparative framework.
- **Ch8, Ch9, Ch10, Ch11**: The four hands-on use-case chapters (RAG, prompt-engineering, fine-tuning, agentic) map directly onto the four non-training-from-scratch rows of this chapter's comparison table.
- **Ch13**: Hallucination probability and toxicity/bias tolerance criteria here are deep-dived into concrete trustworthiness techniques next.
- **Ch14**: TCO and operational maintenance criteria connect to the full FMOps cost/lifecycle framework.
