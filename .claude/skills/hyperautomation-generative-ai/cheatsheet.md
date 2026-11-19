# Cheatsheet

## Decision Rule: RPA alone vs. Hyperautomation

| Condition | Choose |
|---|---|
| Process is rule-based, structured data, no judgment required | RPA alone (cheaper, faster to implement) |
| Process requires decision-making, pattern recognition, or unstructured-data handling | Hyperautomation (RPA + AI/ML/NLP) |
| Process flow doesn't depend on any intelligence (human or artificial) | Skip Hyperautomation — it adds cost with no benefit |
| Need end-to-end automation spanning multiple systems/departments | Hyperautomation (orchestration layer required) |
| Need to generate new content/designs/insights, not just execute/decide | Add Generative AI to the Hyperautomation stack |

## Decision Rule: Which Generative Model Family

| Priority | Model family | Trade-off |
|---|---|---|
| Speed + domain-specific quality | GAN | Lower diversity |
| Fast sampling, lower detail acceptable | VAE | Lower output detail than diffusion |
| Highest quality + diversity, latency tolerable | Diffusion (DDPM) | Slower training/generation |
| General-purpose text/language tasks | Transformer-based LLM (GPT, LaMDA, LLaMA) | Needs prompt engineering; fine-tuning is costly |
| Text-to-image | DALL-E / Stable Diffusion | Can produce nonsensical elements |
| Dialogue-specific | LaMDA | Narrower general-purpose use |
| Efficiency-constrained deployment | LLaMA | Lower parameter count trade-off |

## MLOps Level Selection

| Situation | MLOps Level |
|---|---|
| Model rarely changes/retrains | Level 0 (manual) |
| Model must adapt to changing data (Continuous Training) | Level 1 (ML pipeline automation) |
| Daily/hourly retraining across many production models | Level 2 (full CI/CD automation) |

## Automation Project Pre-Mortem Checklist (why projects fail)

| Failure category | Red flag | Mitigation |
|---|---|---|
| Poor process selection | Process is unstable, not rule-based, inputs unstructured | Apply the "repetitive, stable, rule-based, logic-driven" test before automating |
| Weak strategy | No clear business case, no CEO/stakeholder buy-in | Secure executive sponsorship and ROI case before building |
| People gaps | Partners/staff lack automation skills; human-in-the-loop not planned | Train staff; design exception-handling paths for humans |
| Technology/scalability gaps | Tool chosen without considering integration, legacy systems, or data privacy | Run a technology assessment before committing to a platform |

## Operationalization Checklist (Ch9)

- [ ] Inventory existing automated processes
- [ ] Analyze performance (cycle time, AHT, cost savings, error rates)
- [ ] Assess automation maturity (governance/CoE presence)
- [ ] Assess technology (integration, scalability gaps)
- [ ] Prioritize candidate processes by business impact
- [ ] Build an automation roadmap
- [ ] Engage stakeholders early
- [ ] Define KPIs/success metrics before building
- [ ] Pilot before scaling
- [ ] Establish change management/training plan
- [ ] Monitor and optimize post-deployment
- [ ] Plan for scale/expansion

## Hyperautomation Initiative Risk Tells ("smells")

| Smell | What it signals |
|---|---|
| "We'll automate everything, humans won't be needed" | Misunderstands Hyperautomation's human-complementary design intent; risks adoption resistance |
| No Process Design Document (PDD) before automating | High risk of brittle, unmaintainable automation |
| Scaling RPA past 10 bots without a CoE | Matches the book's cited pattern of stalled RPA programs (Forrester: 52% struggle to scale) |
| ML model deployed with no monitoring/retraining plan | High risk of undetected model drift (ModelOps gap) |
| Generative AI content distributed without human review | Risk of mass-distributing biased/inaccurate/nonsensical output |
| No data governance plan before integrating data sources | Risk of perpetuating data fragmentation/silos instead of resolving them |
| Automating a process before optimizing/standardizing it | Locks in and scales existing inefficiency (Ch1's "automation magnifies inefficiency") |
| Expecting bots to be ~3x more effective than humans | Unrealistic; real-world ratio is ~1.8:1–2:1 per Ch10 case study |

## Responsible/Trustworthy AI Quick Checklist (Ch7)

- [ ] Transparent: decision logic visible/understandable
- [ ] Explainable: clear reasoning provided for decisions
- [ ] Configurable: adaptable to context/user needs
- [ ] Fair: training data audited for bias
- [ ] Accountable: clear ownership for AI outcomes
- [ ] Safe/secure: designed for safe failure modes
- [ ] Private: personal data protected, regulation-compliant (e.g., GDPR)
- [ ] Robust: performs across varied/unexpected conditions
- [ ] Reproducible: outcome provenance maintained for auditing

## Quick Reference: Core Technology Stack by Automation Tier

| Tier | Core technologies |
|---|---|
| Automation | RPA bots, rule engines |
| Orchestration | iBPMS, iPaaS, workflow orchestration tools |
| Optimization | Process mining, advanced analytics, ML-driven tuning |
| Intelligence add-on | AI/ML, NLP, Computer Vision, OCR |
| Creation add-on (GenAI) | GANs, VAEs, Diffusion models, Transformers/LLMs |
