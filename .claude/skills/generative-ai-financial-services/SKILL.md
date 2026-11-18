---
name: generative-ai-financial-services
description: "Knowledge base from \"Generative AI for Financial Services\" by Vikesh Pandey and Arnav Khare (BPB Publications, 2025). Use when implementing generative AI in financial services — foundation model selection, RAG/fine-tuning approaches, architecture decisions, known anti-patterns and challenges, customer experience/contact-center/ESG-analysis use cases, trustworthiness, or Foundation Model Operations (FMOps) practices."
---

# Generative AI for Financial Services
**Authors**: Vikesh Pandey, Arnav Khare | **Chapters**: 14 | **Generated**: 2026-11-18

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or use case; I find and read the relevant chapter
- **With a chapter** — ask for `ch09`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Generative AI vs. Predictive AI**: predictive AI trains a new model per use-case on labeled data; generative AI trains a foundation model (FM) once and customizes it per use-case via an increasing-cost ladder: **prompt engineering → RAG → agentic workflows → fine-tuning → training from scratch**. Climb the ladder only when the current rung's documented limitations actually bite (Ch1, Ch3, Ch6).

**Implementation approach decision framework** (Ch12, the book's central tool): evaluate every candidate approach on 9 criteria — General (technical complexity, quality of responses, TCO, time to market, operational maintenance) + Financial-services-specific (probability of hallucination, toxicity/bias tolerance, data privacy/security, compliance effort). No approach wins on everything; pick based on which criteria matter most for the use-case. A **hybrid approach** (fine-tune once on org data, then layer RAG/prompt engineering on top) often beats any single approach.

**The book's 5 central anti-patterns (Ch6)**: (1) Weak use-case qualification — not every problem fits AI/ML, and not every ML problem fits GenAI specifically (deterministic/rule-based tasks need traditional programs; forecasting/credit-risk needing explainability need specialized ML, not GenAI). (2) Chasing "the best model" instead of the right-fit model — "one LLM does not rule them all"; central-IT one-model mandates kill projects. (3) Not choosing the right implementation approach — see decision framework above. (4) Lack of data governance — single access patterns, ignoring data-producer constraints, over-reliance on incomplete data as ground truth. (5) Incomplete cost analysis — ignoring TCO beyond model-invocation cost, production-scale costs, and ongoing operating costs.

**RAG architecture (5 components)**: Document indexing (extract → chunk → embed → store in vector DB) → Chat UI → Orchestration layer → Retrieval layer (embed query → ANN search → top-k chunks) → Response layer (build prompt from chunks+question → call FM). Improve with: larger embedding dimensions, semantic/context-aware chunking, metadata filters, query transformation, re-ranking (Ch3, Ch8).

**Agent framework**: Plan (CoT/ReAct/ToT reasoning to build action plan) → Execute (call tools: DB, API, vector store, calculator, web search) → Memorize (retain task/steps/outcomes). Use agents when the task needs dynamic tool selection or multi-step orchestration; avoid them for simple/deterministic tasks since agent reasoning is non-deterministic and errors compound across chains (Ch3, Ch11, Ch12).

**Fine-tuning taxonomy**: Parameter-update axis (Full fine-tuning vs. PEFT: LoRA/QLoRA/DoRA) × Methodology axis (Unsupervised domain-adaptation, Supervised/instruction, RLHF with PPO, DPO). Default to PEFT over full fine-tuning — comparable quality at a fraction of compute cost (Ch3, Ch10).

**Architectural considerations (Ch7)**: every GenAI app = 5 components (web interface, prompt, model, storage, database) × 4 considerations (security/privacy, cost, tenancy, performance/scalability). **Single tenancy** (LoB autonomy, duplicated effort) vs. **Multi-tenancy** (central governance/cost tracking, "lowest common denominator" feature limits) is the central tenancy trade-off.

**Trustworthiness = Guardrails + Evaluation + Governance (Ch13)**: Guardrails (prompt-engineering like Llama Guard + external tools like NeMo-Guardrails + HITL for low-confidence cases) filter input AND output. Evaluation should be hybrid (quantitative: ROUGE/BLEU/METEOR/BERTScore/toxicity score + qualitative: SME ratings via thumbs/Likert/ordinal ranking + red-teaming). Governance = answering a standard question bank at every lifecycle phase (data prep → development → deployment → monitoring).

**FMOps (Ch14)**: extends MLOps (technical + business process + culture pillars) with FM-specific tools/processes/personas across 6 lifecycle components (data prep, model selection, model tuning, deployment/monitoring, application layer, foundational components). Choose an **operating model**: Shared (central, auditable, slow) / Decentralized (fast, poor standardization) / Hybrid (shared platform + team-owned dev — usually best for financial services). TCO spans 5 categories: Compute, Database/Storage, Model, Human, Tools — not just model-invocation cost.

**Challenges unique to financial services GenAI (Ch5)**: hallucination risk is severity-dependent (money-moving actions need a stricter bar); specialized domain knowledge gap requires fine-tuning; multi-jurisdictional regulatory compliance (EU AI Act/GDPR, US CFPB/NIST, China, Singapore); data bias vs. model bias; new LLM-specific attack surfaces (prompt injection, insecure output handling, training data poisoning, model DoS) beyond traditional ML; legacy system data extraction difficulty; complex third-party integrations; manual-process automation opportunity.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-introduction-to-generative-ai.md) | Introduction to Generative AI | AI taxonomy, predictive vs. generative AI, generative AI lifecycle |
| [ch02](chapters/ch02-types-of-foundation-models.md) | Types of Foundation Models | Transformer/MoE/GAN architectures, FM training lifecycle, modality/context-window/licensing/parameters/language attributes |
| [ch03](chapters/ch03-generative-ai-implementation-approaches.md) | Generative AI Implementation Approaches | Prompt engineering techniques (zero/few-shot, CoT, ReAct, self-consistency, ToT), RAG, agentic workflows, fine-tuning (LoRA/QLoRA/DoRA/RLHF/DPO), training from scratch (BloombergGPT) |
| [ch04](chapters/ch04-generative-ai-use-cases-in-financial-services.md) | Generative AI Use-cases in Financial Services | Financial institution types, contact-center/investment-research/underwriting/marketing workflows |
| [ch05](chapters/ch05-generative-ai-challenges-for-financial-services.md) | Generative AI Challenges for Financial Services | 8-challenge framework: reliability, specialized knowledge, compliance, ethics/bias, privacy/security, legacy systems, integrations, manual processes |
| [ch06](chapters/ch06-generative-ai-anti-patterns-for-financial-services.md) | Generative AI Anti-Patterns for Financial Services | Use-case qualification test, "best model" fallacy, approach-selection criteria, data governance anti-patterns, cost-analysis anti-patterns |
| [ch07](chapters/ch07-architectural-considerations-in-generative-ai.md) | Architectural Considerations in Generative AI | Security vs. privacy, single/multi-tenancy, TCO, performance/scalability, 5-component application anatomy |
| [ch08](chapters/ch08-improving-customer-experience-with-generative-ai.md) | Use-case: Improving Customer Experience with Generative AI | RAG implementation walkthrough (LangChain + ChromaDB + Bedrock), advanced RAG techniques |
| [ch09](chapters/ch09-contact-center-insights-with-generative-ai.md) | Use-case: Contact Center Insights with Generative AI | Post-call prompt-engineering pipeline: summarization, case notes, email, QA scoring, Q&A |
| [ch10](chapters/ch10-fine-tuning-for-financial-services.md) | Use-case: Fine-tuning for Financial Services | Unsupervised domain-adaptation fine-tuning on SEC filings with PEFT/LoRA/quantization on SageMaker |
| [ch11](chapters/ch11-automating-esg-analysis-using-agents.md) | Use-case: Automating ESG Analysis Using Agents | ReAct agent with vector store + SQL DB tools, LangChain agent implementation |
| [ch12](chapters/ch12-choosing-the-right-implementation-approach.md) | Choosing the Right Implementation Approach | 9-criterion evaluation framework, comparative scoring of all 5 approaches, hybrid approach |
| [ch13](chapters/ch13-increasing-trustworthiness-in-generative-ai.md) | Increasing Trustworthiness in Generative AI | Guardrails (Llama Guard, NeMo-Guardrails, HITL), quantitative/qualitative/hybrid evaluation, model governance question bank |
| [ch14](chapters/ch14-building-foundation-model-operations-practices.md) | Building Foundation Model Operations Practices | FMOps vs. MLOps, 6-component lifecycle, operating models, TCO framework, best practices/challenges |

## Topic Index
- **Agents / Agentic workflows**: ch03, ch04, ch06, ch11, ch12
- **Anti-patterns**: ch06 (also ch05, ch12, ch14 for related failure modes)
- **Architecture / Tenancy**: ch07
- **Compliance / Regulation**: ch05, ch13
- **Contact center**: ch04, ch08, ch09
- **Cost / TCO**: ch06, ch07, ch12, ch14
- **ESG**: ch11
- **Evaluation / Metrics (ROUGE, BLEU, BERTScore)**: ch06, ch13
- **Fine-tuning (LoRA, QLoRA, DoRA, RLHF, DPO)**: ch03, ch06, ch10, ch12
- **Foundation models (architectures, attributes)**: ch01, ch02
- **Guardrails**: ch07, ch13, ch14
- **Hallucination**: ch05, ch12, ch13
- **Model governance**: ch13
- **Model selection**: ch02, ch06, ch12, ch14
- **Prompt engineering**: ch03, ch04, ch09, ch13
- **RAG**: ch03, ch06, ch08, ch12
- **Trustworthiness / Responsible AI**: ch13

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — techniques, use-case patterns, and anti-patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 14 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive, as are any recurring publisher promotional asides. For topics beyond this book's scope, check other skills or ask directly.
