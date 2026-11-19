---
name: generative-ai-for-enterprises
description: "Knowledge base from \"Generative AI for Enterprises\" by Vishal Anand (BPB Publications, 2024). Use when making enterprise generative AI decisions — model selection/deployment, prompt engineering and fine-tuning at scale, workflow orchestration, ethical dimensions, target operating models, cost optimization, RAG, Model-as-a-Service, confidential AI, latency, or multimodal multi-agentic assistant frameworks."
---

# Generative AI for Enterprises
**Author**: Vishal Anand | **Chapters**: 16 | **Generated**: 2026-11-18

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or decision; I find and read the relevant chapter
- **With a chapter** — ask for `ch09`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Foundations (Ch1-5): the decision stack before anything is built.** Generative AI adoption follows an input→model→output→business-integration philosophy (Ch1), but production scaling requires managing eleven interlocking dimensions — business objectives, model sourcing, data management, selection, evaluation, optimization, fine-tuning, orchestration, observability, governance (Ch2). Model selection is a *funnel*: ~21 criteria (regulatory compliance, CapEx/OpEx, latency, cost/token, governance, scalability, data proximity, security, LLM vs. SLM, etc.) filter many candidates down to one or more fits (Ch3). Deployment then picks among five patterns — cloud service, open/own on-premises, open/own cloud IaaS — positioned by regulation-vs-agility trade-offs, and shipped via Blue/Green, Canary, or A/B strategies (Ch4). Underneath it all, the "operating system" (technical ecosystem — distinct from the "operating model," Ch10) requires committing fully to one side of a cloud/on-premises demarcation line and co-locating platform+data+apps via Kubernetes for latency (Ch5).

**Enterprise technique toolkit (Ch6-8, Ch12): how to make models useful.** Prompt engineering scales from manual single-input crafting to automated batch pipelines evaluated on accuracy/duration/token-count (Ch6). Fine-tuning follows a prepare→train→evaluate→integrate loop, made cost-viable at scale via LoRA/PEFT (low-rank weight updates instead of full retraining) (Ch7). An orchestration platform acts as the "nerve center" coordinating data, model lifecycle, and compute — most effective when integrated with a landing zone that bakes in compliance/security by default (Ch8). RAG (Ch12) replaces static Q&A with dynamic retrieval grounding: at scale, combine hybrid retrieval (vector + keyword search) with semantic (L2) re-ranking, and layer RAG + prompt tuning + fine-tuning together to minimize hallucinations — no single technique alone is sufficient.

**Governance & economics (Ch9-11): the decision-maker's control layer.** Six interconnected ethical dimensions — accountability, fairness, transparency, human-orientation, security, lawfulness — must be evaluated together, not as independent checkboxes; violating one undermines enforcement of the others (Ch9). The holistic *operating model* (organizational, not technical) has seven layers — actor, requirements, consumption, platform, execution, landing zone, applicability — unified by a continuous feedback loop (Ch10). Cost optimization splits into "inside periphery" (training-time levers: parameters, data quality, architecture, hardware) and "outside periphery" (consumption-time levers: model selection, prompt/fine-tuning, token minimization, auto-scaling, temperature, caching) (Ch11).

**Advanced deployment concerns (Ch13-16): operating at scale and into the future.** Model-as-a-Service (MaaS) trades CapEx for OpEx; choose pay-as-you-go for variable/dev-test demand, real-time endpoints for high-volume predictable production demand, and consider multi-MaaS for resilience against vendor lock-in (Ch13). Confidential AI closes the "data in use" encryption gap (data at rest/in transit is typically secured, but active CPU/GPU processing often isn't) via VM TEE + GPU TEE trusted execution environments — essential for regulated industries (Ch14). Latency is not one bottleneck but five independent factors — input network, hardware, model inference, integration point, output network — each needing separate diagnosis and mitigation (Ch15). The forward-looking culmination is multi-modal, multi-agentic assistant frameworks: specialized autonomous agents (each defined by toolset+data+function+instruction) collaborating across text/voice/image/video modalities, connected via an API layer, always designed to augment rather than replace human judgment (Ch16).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-the-rise-of-generative-ai-in-enterprises.md) | The Rise of Generative AI in Enterprises | Input-output philosophy, use-case mapping by inference task, architectural overview |
| [ch02](chapters/ch02-complex-needs-of-production.md) | Complex Needs of Production | Eleven dimensions of model-at-scale, ROI/TCO analysis, sourcing matrix, LLMOps |
| [ch03](chapters/ch03-model-selection-for-enterprises.md) | Model Selection for Enterprises | Model selection funnel, 21-question criteria checklist, post-selection lifecycle |
| [ch04](chapters/ch04-model-deployment-for-enterprises.md) | Model Deployment for Enterprises | Five deployment patterns, regulation-vs-cost positioning, Blue/Green/Canary/A/B |
| [ch05](chapters/ch05-operating-system-for-enterprises.md) | Operating System for Enterprises | Cloud/on-prem demarcation, Kubernetes co-location, Day 1/Day 2 ops |
| [ch06](chapters/ch06-prompt-engineering-for-enterprises.md) | Prompt Engineering for Enterprises | Single-input vs. batch pipelines, accuracy/duration/token-count evaluation |
| [ch07](chapters/ch07-fine-tuning-for-enterprises.md) | Fine-tuning for Enterprises | Four-step fine-tuning loop, LoRA/PEFT, auto-scaling fine-tuning infra |
| [ch08](chapters/ch08-orchestration-of-generative-ai-workflows.md) | Orchestration of Generative AI Workflows | Orchestration platform component stack, platform+landing-zone integration |
| [ch09](chapters/ch09-six-ethical-dimensions-for-enterprises.md) | Six Ethical Dimensions for Enterprises | Accountability, fairness, transparency, human-orientation, security, lawfulness |
| [ch10](chapters/ch10-designing-a-target-operating-model.md) | Designing a Target Operating Model | Seven-layer operating model, four core infusion principles, feedback loop |
| [ch11](chapters/ch11-cost-optimization-strategies.md) | Cost Optimization Strategies | Inside/outside periphery cost levers, pruning/quantization, token/temperature discipline |
| [ch12](chapters/ch12-retrieval-augmented-generation-for-enterprises.md) | Retrieval-augmented Generation for Enterprises | RAG vs. traditional Q&A, hybrid retrieval + L2 semantic ranking, hallucination-reduction triad |
| [ch13](chapters/ch13-model-as-a-service-for-enterprises.md) | Model as a Service for Enterprises | PAYG vs. real-time endpoint decision quadrants, multi-MaaS strategy |
| [ch14](chapters/ch14-confidential-ai.md) | Confidential AI | Three-state data protection model, VM TEE + GPU TEE, multi-layered access risk |
| [ch15](chapters/ch15-latency-in-generative-ai-solutions.md) | Latency in Generative AI Solutions | Five latency factors, edge computing, model-size-reduction techniques |
| [ch16](chapters/ch16-multi-modal-multi-agentic-assistant-framework-for-enterprises.md) | Multi-modal Multi-agentic Assistant Framework for Enterprises | AI agent definition, four-component agentic architecture, multi-agentic collaboration |

## Topic Index

- **Accountability / fairness / transparency / lawfulness** → ch09
- **API economy** → ch10
- **Auto-scaling (fine-tuning / compute)** → ch07, ch11
- **Blue/Green, Canary, A/B deployment** → ch04
- **Chunking (RAG)** → ch12
- **CapEx vs. OpEx** → ch03, ch13
- **Cloud vs. on-premises deployment** → ch04, ch05
- **Co-location (latency)** → ch05, ch15
- **Confidential AI / data-in-use encryption** → ch14
- **Cost optimization (inside/outside periphery)** → ch11
- **Demarcation (operating system)** → ch05
- **Fine-tuning / LoRA / PEFT** → ch02, ch07
- **Governance (model, ethics, organization-wide)** → ch02, ch09, ch10
- **Hallucination reduction** → ch12
- **Hybrid retrieval / semantic ranking** → ch12
- **Latency (five factors)** → ch15
- **LLMOps** → ch02
- **Model as a Service (MaaS)** → ch13
- **Model drift (data/task/topic/retrieval)** → ch02
- **Model selection criteria** → ch02, ch03
- **Multi-agentic / multi-modal assistants** → ch16
- **Operating model (seven layers)** → ch10
- **Operating system (technical ecosystem)** → ch05
- **Orchestration platform / landing zone** → ch02, ch05, ch08, ch10
- **Prompt engineering (single/batch)** → ch06
- **RAG (retrieval-augmented generation)** → ch12
- **Six ethical dimensions** → ch09
- **TEE (VM TEE, GPU TEE)** → ch14
- **Token/temperature cost discipline** → ch11

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — decision/implementation patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 16 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive, as are any recurring publisher promotional asides. For topics beyond this book's scope, check other skills or ask directly.
