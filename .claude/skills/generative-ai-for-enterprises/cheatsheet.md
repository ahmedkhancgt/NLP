# Cheatsheet

## Decision Rule: Prompt Engineering vs. Fine-Tuning vs. RAG

| Need | Use | Why |
|---|---|---|
| Model needs current/enterprise-specific facts | RAG | Grounds output in retrieved, up-to-date data instead of static pre-trained knowledge |
| Model needs better output format/focus, low cost | Prompt tuning/engineering | Adjusts input only; no weight changes, cheapest option |
| Model needs deep domain/style adaptation | Fine-tuning (ideally LoRA/PEFT) | Adjusts weights for lasting specialization; higher cost than prompting |
| Hallucination risk is high-stakes | RAG + prompt tuning + fine-tuning together | Each addresses a different failure mode; combined is most robust |
| Narrow enterprise dataset, cost-sensitive | LoRA/PEFT fine-tuning, not full retrain | Avoids overfitting and high compute cost |

## Decision Rule: Deployment Pattern

| Situation | Favor |
|---|---|
| Highly regulated (healthcare, finance), strict data control needed | On-premises (own or open model) |
| Need agility, fast time-to-market, variable demand | Cloud service LLM pattern |
| Mixed sensitivity data | Hybrid (sensitive on-prem, rest in cloud) |
| Latency-critical, real-time | Co-locate model + data via Kubernetes; stay on one side of demarcation |

## Decision Rule: MaaS Consumption

| Environment / Demand | Choose |
|---|---|
| Dev/Test, variable/low volume | Pay-as-you-go (PAYG) |
| Production, extremely high + forecasted/consistent volume | Real-time endpoint |
| Multi-region resilience / avoiding lock-in | Multi-MaaS + containerized, provider-agnostic design |

## Six Ethical Dimensions Checklist (Ch 9)

- [ ] **Accountability** — Is there a named responsible party for this AI system's outputs, with authority to take corrective action?
- [ ] **Fairness** — Has the model been tested on diverse datasets for bias? Is adversarial training or similar mitigation in place?
- [ ] **Transparency** — Are decision processes, data sources, and algorithms documented and explainable (XAI)? Are audits scheduled?
- [ ] **Human-orientation** — Does the system augment human judgment under human oversight, rather than operate as an unchecked replacement?
- [ ] **Security** — Are there protections against unauthorized access, data breaches, tampering, and prompt injection?
- [ ] **Lawfulness** — Does the system comply with applicable data protection, IP, and regional laws, with a process to track regulatory change?

## Cost Optimization Levers Checklist (Ch 11)

**Inside periphery (training your own model)**
- [ ] Parameters/model size reduced via pruning/quantization where possible
- [ ] Data quality validated/cleaned (reduces retraining cycles)
- [ ] Architecture choice fits the task (don't default to largest transformer variant)
- [ ] Modern GPUs with mixed-precision training support selected
- [ ] Skilled personnel in place to avoid costly mistakes
- [ ] Batch inference used where real-time isn't required (off-peak scheduling)

**Outside periphery (consuming a model)**
- [ ] Model size/capability matched to actual accuracy need (not oversized)
- [ ] Prompt tuning and/or fine-tuning applied before accepting higher-cost options
- [ ] Input/output tokens minimized per interaction
- [ ] Discounted/spot/reserved compute used where interruption-tolerant
- [ ] Auto-scaling configured (no fixed over-provisioning)
- [ ] Temperature tuned low for deterministic, token-efficient tasks
- [ ] Application caching used for repeated/duplicate queries

## Governance/Deployment Risk Smells

| Smell | Risk |
|---|---|
| No named owner for an AI system's outputs | Accountability gap — issues can't be corrected or traced |
| Model never tested on diverse data for bias | Latent fairness/discrimination risk, especially in hiring/lending/healthcare |
| Decision logic undocumented, no XAI tooling | Transparency failure — can't be audited or contested |
| Data-in-use left unencrypted on shared/cloud infra | Confidential AI gap — regulatory (GDPR/HIPAA) and breach exposure |
| Mixing cloud and on-prem for one latency-sensitive workload | Crosses demarcation line — unpredictable latency/service quality |
| Full-model retraining for a narrow, small dataset | Wasted compute + overfitting risk vs. LoRA/PEFT |
| Fixed-size compute with no auto-scaling | Cost waste during idle periods, bottlenecks during spikes |
| No LLMOps strategy before production MaaS deployment | Unmanaged model versioning/validation risk |
| Single cloud-provider dependency with no mitigation | Vendor lock-in and outage/regional-disruption exposure |
| Agent system with no human oversight checkpoint | Violates human-orientation ethical principle |
| Latency blamed on "the model" without profiling | Misdiagnosis — bottleneck often network/integration/hardware, not inference |

## Latency Factor Quick Reference (Ch 15)

| Factor | Typical Fix |
|---|---|
| Input network | Edge computing, CDN, faster protocols, ISP peering |
| Hardware | Faster GPU/RAM/SSD, better cooling/power, PCIe bandwidth |
| Model inference | Pruning, quantization, distillation, batching |
| Integration point | Co-location, caching, fewer API calls, lean middleware |
| Output network | Efficient post-processing (compression/encryption), robust return paths |

## Model Selection Hard Constraints (Check First, Ch 3)

1. Regulatory compliance (GDPR, HIPAA, sector-specific)
2. Data security / indemnity
3. Data proximity / residency requirements
4. Context length, rate limits, token quotas sufficient for use case

Only after these pass: compare cost, accuracy, ease of governance, automation, collaboration fit.
