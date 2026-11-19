# Chapter 1: Introduction to Agentic Systems

## Core Idea
Agency is not a vague buzzword but a precise, testable capability: **"the ability of a system to autonomously perceive, recollect, plan, decide, act, and adapt."** A reasoning engine (an LLM) is not itself an agent — it becomes one only when wrapped in layers of environmental perception, action, and adaptation.

## Frameworks Introduced

### The Six-Verb Ontology of Agency
Agency decomposes into six capabilities a system must exhibit (not necessarily in strict sequence — they can iterate or be skipped):
- **Perceive**: sense the environment, build an internal world-state representation (full or partial observability).
- **Recollect**: memorize/recall prior perceptions, decisions, actions, states.
- **Plan**: choose a goal among possibilities, postulate a step-by-step path to it.
- **Decide**: reason over state/perception/plan to pick specific actions via a policy.
- **Act**: execute actions that change the real-world state.
- **Adapt**: learn from the gap between plan and effect; tune the decision policy for long-term reward.
  - When to use: as a diagnostic checklist — if a system is missing any one of these six verbs, it is not yet agentic, just reactive or scripted.
  - How: an agent need not implement all six perfectly in one pass; over time, across interactions, it should visibly exhibit all six.

### Agency-as-System-Property (not entity property)
The definition deliberately avoids tying agency to one physical/logical entity — it's a **capability of a system**, and can be an emergent property of a network of collaborating entities (the book's analogy: human consciousness as an emergent property of a neuron network, not any single neuron).
- When to use: evaluating multi-agent architectures — don't look for agency in any one agent; look for it in the collaborating network as a whole.

## Key Concepts
- **Agency**: "the ability to drive purposeful change in the environment" — purposeful/intentional, not merely causal (a person cutting a tree shows agency; a tornado wrecking a house does not).
- **Agentic AI**: the use of AI to imbue software with agency as defined above.
- **Cognitive core**: the reasoning engine (typically an LLM) at an agent's center — necessary but not sufficient for agency on its own.
- **Turing Test**: historically the benchmark for machine intelligence; the book notes leading researchers now agree current LLMs pass it, shifting the open question from "intelligence" to "agency" and eventually "consciousness."
- **RAG (Retrieval Augmented Generation)**: introduced here as the technique behind agentic search — recall a broad set of semantically similar documents (vector/keyword/graph search), then synthesize an answer by feeding their content to an LLM. Contrast with classical search, which ranks and surfaces documents for the user to read themselves.
- **A2A (Agent-to-Agent) Protocol** and **MCP (Model Context Protocol)**: named here as the two leading standardization efforts for agentic interoperability (Google and Anthropic respectively) — detailed later in Ch 3 and Ch 6.

## Mental Models
- Treat "agentic" as a spectrum defined by the six verbs, not a binary label — ask "which of perceive/recollect/plan/decide/act/adapt is this system actually doing?" before accepting a product's "agentic" marketing claim.
- Think of the cognitive core vs. the agent the way you'd think of a CPU vs. a robot: the CPU (LLM) computes, but perception/action/adaptation loops are what turn computation into agency.
- Use the historical arc (Turing → cybernetics/Wiener → information theory/Shannon → dynamic programming/Bellman → Kalman filtering → STRIPS → SHRDLU → MYCIN → Deep Blue → Watson → AlphaGo/AlphaFold → OpenAI Five → ChatGPT) as a reminder that agentic AI is a convergence of control theory, optimization, and language modeling — not a single lineage.

## Anti-patterns
- **Calling a bare LLM call an "agent"**: no matter how large or sophisticated the model, without perception/memory/planning/action/adaptation wrapped around it, it's a cognitive core, not an agent.
- **Conflating conversational fluency with agency**: passing the Turing Test (conversational indistinguishability) is a different, lower bar than exhibiting the six-verb ontology of agency.
- **Assuming agency requires full environmental observability**: the book explicitly notes partial observability is "also very useful" — don't wait for perfect sensing before building the perceive/act loop.

## Reference Tables

**Table 1.1: Point-in-time summary of major conversational-assistant players (Jan 2026)**

| Company | Model family | Chat assistant | Notes |
|---|---|---|---|
| OpenAI | GPT series + o-series (reasoning) | ChatGPT | GPT-5 default since Aug 2025 |
| Google | Gemini | Gemini (formerly Bard) | Gemini 2.5 Flash free default |
| Microsoft | Uses OpenAI + Anthropic models | Microsoft Copilot | GPT-5 via Smart Mode routing |
| Anthropic | Claude | Claude (claude.ai, app) | Opus 4.5 most capable |
| xAI | Grok | Grok | Grok 4 Premium+ |
| Meta | Llama (open weights) | Meta AI | Llama 4 Maverick powers 40+ countries |
| Alibaba | Qwen (通义千问) | Qwen Chat | Qwen3 series |
| DeepSeek | DeepSeek | DeepSeek | V3.1 general, R1 reasoning mode |

Open-source model families (Llama, Mixtral, Gemma) and open agentic frameworks (LangGraph, AutoGen, CrewAI) exist alongside the closed-source majors; standardization efforts (A2A from Google, MCP from Anthropic) aim to unify how agents built on any of these interoperate.

## Worked Example

**Application taxonomy (Ch 1's closing survey), with a concrete exemplar per bucket:**
1. **AI search/research/summarization** — RAG-based conversational search (Google Search, Bing, Perplexity) replacing keyword lookup with synthesized, cited answers.
2. **Work/research assistants and copilots** — domain-specific proprietary-data assistants (Moody's Research Assistant for financial analysis, Truveta Intelligence for healthcare research on EHR data).
3. **Code generation** — autonomous coding agents (Claude Code, Devin, Replit AI, GitLab Duo) that execute multi-step tasks (migrations, bug fixes, feature work) via agentic codebase search and direct file/command/git actions, not just autocomplete.
4. **Customer support** — high-volume deflection bots (Klarna's AI assistant handles ~2/3 of customer chats; Airbnb's US bot serves ~50% of US users; Salesforce Agentforce for always-on specialized support agents).
5. **Productivity tools / RPA** — Microsoft Copilot across Office, plus UiPath/Automation Anywhere/Adept for structured process automation.

Lesson: nearly every product category is being re-architected around the six-verb agency loop — search, support, and productivity tools now perceive (user intent/context), plan (multi-step task decomposition), and act (execute, not just suggest).

## Key Takeaways
1. Agency = perceive + recollect + plan + decide + act + adapt — use this as your working test for "is this actually agentic."
2. An LLM alone is a cognitive core, not an agent; agency requires the surrounding perception/action/adaptation layers.
3. Agency is a system-level property that can emerge from networks of collaborating entities, not just single monolithic agents.
4. The field's theoretical roots (control theory, information theory, dynamic programming, adaptive filtering) predate generative AI by 70+ years — the generative AI revolution supplied the missing cognitive-core capability, not a wholly new paradigm.
5. RAG, A2A, and MCP are the three technical threads (grounding, agent interop, tool/context interop) that recur throughout the rest of the book.

## Connects To
- **Ch 2 (Theoretical Frameworks)**: formalizes the "decide" and "plan" verbs via decision theory, Bayesian reasoning, and expected-utility maximization.
- **Ch 3 (The Agentic Ecosystem)**: expands on A2A and MCP, introduced here only by name.
- **Ch 4 (LLMs as Cognitive Cores)**: elaborates exactly what a cognitive core is and its limits, referenced here as necessary-but-not-sufficient for agency.
- **Ch 6 (Tools and Interfaces)** and **Ch 7 (Memory Systems)**: implement the "act" and "recollect" verbs concretely.
