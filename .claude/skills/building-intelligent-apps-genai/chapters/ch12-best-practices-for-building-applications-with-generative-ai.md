# Chapter 12: Best Practices for Building Applications with Generative AI

## Core Idea
Building production-grade generative AI applications requires deliberate model selection (size, knowledge cutoff, fine-tuning support, cost, ethics), a conscious open-source-vs-proprietary trade-off, disciplined LLMOps (a specialized evolution of MLOps), and explicit data privacy/security guidelines for both developers and end users.

## Frameworks Introduced
- **Model-selection question framework**: before choosing an LLM, ask (1) what is your primary objective? (2) what specific tasks must it perform? (3) what is the anticipated scale (low traffic vs. enterprise)? (4) what computational constraints exist?
  - When to use: at the very start of any generative AI project, before evaluating specific models.
- **Model evaluation criteria**: model size/capabilities (parameter count trade-offs), pre-training data/knowledge cutoff, fine-tuning/customizability, accessibility/cost, ethical/responsible AI considerations.
  - When to use: comparing candidate models (GPT-4, Llama, Gemini, etc.) against your specific use case.
- **Open-source vs. proprietary decision framework**: weigh premium models' higher quality/speed-to-market/minimal infrastructure against open-source's data privacy/transparency/cost-effectiveness/no vendor lock-in — pick premium under time constraints, open-source for data control and long-term cost predictability.
  - When to use: a foundational architecture decision made early, revisited as scale/budget/compliance needs evolve.
- **LLM development-to-production workflow (6 steps)**: Select LLM model → Prompt engineering → Evaluate results → Fine-tuning → Deployment to production (canary or blue-green) → Monitoring and tracking metrics (plus a continuous feedback loop).
  - When to use: as the standard lifecycle template for taking any LLM application from prototype to production.
- **LLMOps vs. MLOps differentiators**: computational resources (GPUs, model compression/distillation), transfer learning (fine-tune a foundation model rather than train from scratch), human feedback (RLHF), hyper-parameter tuning (also optimizing cost/compute, not just accuracy), performance metrics (BLEU for translation, ROUGE for summarization, vs. traditional accuracy/AUC/F1), prompt engineering (mitigates hallucination and injection), and LLM chains/pipelines (LangChain/LlamaIndex stringing together calls + external systems).
  - When to use: adapting existing MLOps practices/teams to LLM-specific projects.

## Key Concepts
- **LLMOps**: the set of practices for developing, deploying, and managing LLMs in production — encompasses EDA, data prep/prompt engineering, fine-tuning, model review/governance, inference/serving, and monitoring with human feedback.
- **Canary deployment**: rolling out a new version to a small subset of users/environment before full rollout, to catch problems early.
- **Blue-green deployment**: deploying a new version to a separate environment, then switching traffic over, for zero-downtime releases.
- **RLHF (Reinforcement Learning via Human Feedback)**: a key LLM training advancement where end-user feedback is incorporated into model refinement, especially important given LLM tasks are often open-ended.
- **BLEU score**: evaluates machine translation quality (higher = better translation).
- **ROUGE score**: evaluates text summarization quality (higher = better capture of original main ideas).
- **LLMOps platforms**: Agenta (open-source), Deepset AI, Fine-tuner AI, Hugging Face, Lamini AI, MosaicML, Nemo (Nvidia), Snorkel AI, Valohai — plus integrated offerings from Google, Amazon, and Microsoft Azure.

## Mental Models
- Think of model selection as answering "what, how much, how fast, how constrained" before "which model" — the four framing questions precede any specific model comparison.
- Use "start small and scale up" as the default posture for any new generative AI initiative — validate fundamentals on a simple application before investing in enterprise-scale deployment.
- Treat the open-source-vs-proprietary decision as a spectrum, not a binary: time-constrained launches lean premium; data-sensitive, cost-predictable, long-horizon projects lean open-source; libraries exist to ease switching between them later.

## Anti-patterns
- **Choosing a model based on hype rather than the four framing questions**: skipping objective/task/scale/resource-constraint analysis risks mismatched model size, cost, or capability for the actual use case.
- **Deploying to production without thorough testing**: the chapter explicitly calls out testing for accuracy, bias, and security vulnerabilities as a required pre-deployment step, not optional.
- **Ignoring knowledge cutoff for time-sensitive applications**: selecting a model whose training data predates your application's need for current information produces stale/incorrect answers.
- **Treating LLMOps as identical to MLOps**: assuming traditional ML metrics (accuracy, AUC, F1) suffice for LLM evaluation, instead of adopting LLM-specific metrics (BLEU, ROUGE) and addressing GPU/compute, transfer learning, and RLHF considerations unique to LLMs.
- **Skipping anonymization and access controls on training data**: directly named as a developer guideline — using non-anonymized data or weak access controls on the model/training data risks privacy violations.
- **Trusting generative AI outputs from unverified sources, or oversharing data with generative tools**: user-facing guideline — only use reputable/verified generative AI applications, and only share information you're comfortable making public.

## Code Examples
No code examples in this chapter — content is organized as best-practice checklists, decision frameworks, and a workflow diagram (Figure 12.1: Select LLM → Prompt engineering → Evaluate results → Fine-tuning → Deployment → Monitoring).

## Reference Tables
| Premium (paid) LLM | Pros | Cons |
|---|---|---|
| e.g., GPT-4, Claude, Gemini | High-quality/bias-tested data; cost-effective at scale (e.g., $0.93-$42 per 1,000 pages with GPT-4); faster time-to-market (API-ready); minimal staff specialization needed | Data exposure risk (third-party vendor); high usage costs at scale (e.g., $7,000-$28,000 for 500,000 pages with GPT-4); vendor lock-in; fluctuating pricing; costly fine-tuning |

| Open-source LLM | Pros | Cons |
|---|---|---|
| e.g., Llama 2, Mistral, Falcon | Full data privacy/control; transparency (inspectable architecture/training data); cost-effective (no licensing fees); free fine-tuning; predictable long-term costs; no vendor lock-in; community support; more enterprise control | Upfront rollout/infrastructure investment; typically lower baseline quality (improving rapidly); requires specialized staff; license compliance burden |

| LLMOps vs. MLOps factor | LLM-specific consideration |
|---|---|
| Computational resources | GPUs, model compression/distillation for inference cost |
| Transfer learning | Fine-tune a foundation model rather than train from scratch |
| Human feedback | RLHF — critical given open-ended LLM tasks |
| Hyper-parameter tuning | Also targets cost/compute reduction, not just accuracy |
| Performance metrics | BLEU (translation), ROUGE (summarization) vs. accuracy/AUC/F1 |
| Prompt engineering | Mitigates hallucination, injection, data leaks, unauthorized access |
| LLM chains/pipelines | LangChain/LlamaIndex stringing together calls + vector DBs/web search |

| Recommended open-source commercial licenses | Rank |
|---|---|
| Apache 2.0 | Preferred |
| MIT | Second choice |

## Worked Example
No single running worked example; the chapter instead presents the development-to-production LLMOps workflow as its structural backbone: (1) select a foundational LLM; (2) iterate on prompt engineering until outputs meet quality bar; (3) evaluate results across varied inputs; (4) fine-tune on domain-specific data if evaluation succeeds; (5) deploy via canary (small subset first) or blue-green (parallel environment, then traffic switch) strategy; (6) monitor production metrics (accuracy, latency, error rates) — optionally with a tool like LangSmith (introduced in Chapter 10) — and feed user feedback back into the cycle.

## Key Takeaways
1. Answer the four model-selection framing questions (objective, tasks, scale, resource constraints) before comparing specific models.
2. Weigh premium vs. open-source models along quality, cost curve (per-volume pricing), data control, and time-to-market — there is no universally correct choice.
3. Treat LLMOps as MLOps-plus: it inherits MLOps discipline but adds LLM-specific concerns (RLHF, prompt engineering, BLEU/ROUGE, chain/pipeline construction).
4. Use canary or blue-green deployment strategies to de-risk production LLM rollouts.
5. Apply developer-side privacy practices (anonymized training data, strict access controls, bias monitoring, injection-attack defenses, transparency) and user-side practices (verified sources only, minimal data sharing, bias awareness) as a dual responsibility.
6. Favor Apache 2.0 (then MIT) licensed open-source models for commercial use to minimize legal liability.

## Connects To
- **Ch5**: Builds directly on the fine-tuning discussion from the emerging-trends chapter.
- **Ch4/Ch10**: Revisits the proprietary-vs-open-source comparison (GPT-4/Claude/Gemini vs. Llama 2) with a formal cost/benefit framework.
- **Ch10**: LangSmith, introduced there as a hands-on tool, is referenced here as the recommended production-monitoring solution.
- **Ch6**: Extends the moderation/injection-prevention guidance from the ChatGPT API chapter into formal developer security guidelines.
- **Ch13**: Directly sets up the next chapter's deeper treatment of ethical considerations (bias, deepfakes, misinformation) only briefly flagged here.
