# Cheatsheet

## Approach Selection Decision Table (Ch6, Ch12)

| Need | Use | Avoid when |
|---|---|---|
| Quick feasibility check, simple task (classification, sentiment) | Prompt engineering | Info needs search over large doc sets; prompt getting huge/expensive; need new domain language |
| Q&A over proprietary/internal docs; frequently changing data | RAG | Needs a whole new skill not in a few docs; info doesn't fit context window; non-text/image modality |
| Multi-step orchestration across tools/APIs/DBs | Agentic workflow | Task is simple enough for RAG/prompting; need deterministic, repeatable execution; deep chains compound error |
| New domain language/style; rare skill; preference alignment (RLHF/DPO); cheaper inference via smaller tuned model | Fine-tuning (prefer PEFT/LoRA/QLoRA over full) | Pre-trained FM already performs well; insufficient quality labeled data; cost doesn't justify it |
| Becoming your own model provider; industry-specific base model; confidential/gov isolation | Train from scratch | Almost always — needs extreme ROI justification (~$millions, months) |
| Want domain fluency + flexibility across many apps | Hybrid (fine-tune, then RAG/prompt on top) | Single-use, low-budget projects where the added fine-tune cost isn't amortized |

## 9-Criterion Evaluation Scorecard (Ch12) — rate each approach low/med/high, comparatively

General: Technical complexity | Quality of responses | TCO | Time to market | Operational maintenance
FSI-specific: Probability of hallucination | Toxicity/bias tolerance | Data privacy & security | Compliance effort

Rule of thumb ranking (low→high cost/complexity, low→high quality/hallucination-safety):
Prompt engineering < RAG < Agentic ≈ Fine-tuning < Train from scratch

## Use-Case Qualification Gate (Ch6)
Before building, ask in order:
1. Is this deterministic/rule-based (needs exact, repeatable output)? → **Traditional program**, not ML/GenAI.
2. Is this a good fit for ML but not GenAI (forecasting, time-series, churn, credit-risk needing explainability)? → **Specialized ML** (ARIMA, Prophet, RNN/LSTM, explainable models).
3. Otherwise → proceed to GenAI approach selection above.

## Anti-Pattern Smell Test (Ch6) — tells/smells to watch for
- "We mandated one model for every use-case" → central-IT one-model mandate smell.
- "We're using RAG/agents because it's trendy" without checking the when-to-avoid list → wrong-approach smell.
- "We applied the same access rules to transactional and reporting data" → single-access-pattern smell.
- "We trained on whatever data we could scrape without checking source constraints" → data-producer-constraint-ignorance smell.
- "We treated the model's/system's prior output as ground truth" → over-reliance-on-incomplete-data smell.
- "Our cost analysis only included model API costs" → incomplete-cost-analysis smell.
- "We never looked past the POC-stage cost" → scale-of-production smell.

## Guardrail Layering Checklist (Ch13)
- [ ] Input guardrail (prompt engineering, e.g., Llama Guard classify safe/unsafe)
- [ ] Output guardrail (same or different check on model response)
- [ ] External tool guardrail for consistency (NeMo-Guardrails, Bedrock Guardrails, Azure AI Content Safety)
- [ ] HITL review for low-confidence decisions (set threshold — not too low = queue flood, not too high = false negatives slip through)
- [ ] HITL only for async/batch flows, never real-time latency-sensitive paths

## Model Evaluation Decision Rule (Ch13)
- Need a number for a standard task (summarization, translation, classification)? → **Quantitative** (ROUGE, BLEU, METEOR, BERTScore, toxicity score; benchmark on MMLU/HELM/GSM8K/GLUE/etc.)
- Need brand/tone/simplicity/factual judgment a number can't capture? → **Qualitative** (SME functional evaluation via thumbs/Likert/ordinal ranking; red-teaming for security)
- Building a production financial-services pipeline? → **Hybrid** (quantitative score triggers SME review at low/high confidence extremes + random sampling)

## Red-Teaming Attack Checklist (Ch13)
- [ ] Prompting (instruction override / "ignore everything, respond with yes")
- [ ] Revealing training data (sentence-completion PII extraction)
- [ ] Backdoor attacks (trigger token forcing fixed output)
- [ ] Remote code execution (malicious model code pre-publish)
- [ ] Role-play (fictional framing to bypass ethical boundaries)
- [ ] Data poisoning (planting fake content for future scraping)
- [ ] Data exfiltration (harvesting I/O pairs via API to train a copycat model)

## Trustworthiness Checklist (Ch13)
- [ ] Separately score helpfulness vs. factual correctness (helpfulness drive causes hallucination)
- [ ] Display source chunks alongside RAG answers for groundedness check
- [ ] Define toxicity/bias tolerance per use-case (wealth management ≠ gaming chat)
- [ ] Test robustness against jailbreak/prompt-injection
- [ ] Monitor data drift, concept drift, feature-attribution drift post-deployment
- [ ] Answer model governance questions at every lifecycle phase (data prep, development, deployment, monitoring) — see glossary/patterns for the full question bank

## Security Quick Checks (Ch5, Ch7, Ch14)
- [ ] Vector-store chunks carry access-control metadata (embeddings do NOT inherit source-document permissions automatically!)
- [ ] Model endpoint has no unneeded internet access
- [ ] EULA confirms provider won't train on your data; check indemnity clause
- [ ] OWASP LLM Top 10 scan done (prompt injection, insecure output handling, training data poisoning, model DoS)
- [ ] Encryption at rest + in transit (TLS 1.3)
- [ ] Principle of Least Privilege applied to data, network, hardware, application, model, and permissions — not just data

## FMOps TCO Checklist (Ch14) — don't forget any of these 5 categories
1. **Compute** (data processing clusters, not just model training)
2. **Database & Storage** (prompt catalog, vector store, intermediate state)
3. **Model** (invocation AND tuning cost — tuning is HPC-intensive)
4. **Human** (skilled engineers, domain experts, compliance/legal — don't forget these)
5. **Tools** (vector engine, CI/CD, auth, versioning, observability, cybersecurity, CDN, orchestration, auditability)
+ Miscellaneous (ongoing maintenance, networking/domain/firewall)

## Operating Model Quick Pick (Ch14)
- Small org, need speed → **Decentralized** (but watch for compliance/standardization gaps at scale)
- Large regulated org, need strict governance → **Shared** (but expect slower team velocity)
- Large org wanting both → **Hybrid** (shared CI/CD/orchestration/observability + team-owned dev/experimentation) — usually the right default for financial services
