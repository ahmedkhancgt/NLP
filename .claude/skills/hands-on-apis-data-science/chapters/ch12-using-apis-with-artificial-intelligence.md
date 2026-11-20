# Chapter 12: Using APIs with Artificial Intelligence

## Core Idea
AI and APIs are deeply intertwined — APIs train models, serve models, and increasingly are *called by* generative AI applications (RAG and agentic patterns) — so API designers should apply specific design tips to make APIs "AI-ready," and this chapter launches the Part III portfolio project (ML deployment, LangChain, ChatGPT).

## Frameworks Introduced
- **AI/API Overlap Taxonomy**: four distinct ways AI and APIs intersect.
  - When to use: scoping which "AI meets API" pattern a project actually needs.
  - How: (1) APIs as training data sources for models; (2) trained models deployed and served via REST APIs (inference endpoints, Ch. 13); (3) RAG — an application calls APIs/data sources and feeds retrieved context to an LLM alongside the user prompt, to overcome the model's training-data knowledge gap; (4) agentic applications — the LLM itself decides which API endpoints to call, by interpreting OAS files, code, or documentation (Ch. 14, Ch. 15).
- **AI-Readiness Design Checklist for APIs**: concrete tips for making an API usable by LLM-driven consumers.
  - When to use: before exposing an API to be called by an agentic AI application or LLM tool-use framework.
  - How: limit result size (token cost + accuracy — return only critical fields, exclude nested child collections into separate endpoints, add filters/pagination); keep data structures consistent (reuse schemas, published in the OAS); provide an SDK scoped to AI-appropriate calls with detailed parameter explanations; customize the OAS with clear descriptions and unique `operationId`s; provide dedicated summary/count endpoints (LLMs behave erratically trying to infer counts themselves); provide a name/language-based search endpoint, since LLMs are more comfortable with language than raw numeric IDs.
- **Safeguard-before-autonomy framework**: decide whether an API/endpoint should be exposed to an agentic LLM at all before optimizing its design.
  - When to use: any time you're considering letting an LLM autonomously call a real API (especially one with side effects).
  - How: apply safeguards such as human approval before executing LLM-recommended actions, multi-agent review before execution, input/output filtering, logging/audit review, and — critically — restrict the permissions granted to any system that includes an LLM, applying standard API security/management practice on top.

## Key Concepts
- **Token**: the unit of text LLM providers charge for processing; larger API responses cost more and can degrade model accuracy on calculations.
- **Retrieval Augmented Generation (RAG)**: pattern where retrieved data (via API/database/file) is injected into the LLM's context alongside the user's prompt, addressing the model's fixed "knowledge gap" from training.
- **Agentic application**: a generative AI application in which the LLM itself decides which API endpoint(s) to call, inferred from OAS files, code, or documentation, rather than a human-coded call sequence.
- **Agent**: software that controls application flow using an LLM; the more autonomously the LLM drives decisions, the more "agentic" the system.
- **Arazzo Specification**: an emerging OpenAPI-companion standard defining "sequences of calls and their dependencies" so multi-step business processes using multiple endpoints can become deterministic, reusable building blocks for AI applications — addresses the risk that nondeterministic LLMs might call related endpoints in the wrong order or combination.
- **Knowledge gap**: the limitation that an LLM only "knows" what was in its training data as of a cutoff point — RAG's primary justification.
- **Expert systems**: older, decades-old rules-based AI systems, distinguished from modern machine learning (where models are trained rather than explicitly programmed).
- **Machine learning vs. generative AI**: machine learning is the broader modern AI paradigm (train, don't hand-code, rules); generative AI (via LLMs) is one major application of it, producing text/music/video from prompts.
- **MCP (Model Context Protocol)** and **OpenTelemetry**: two standards flagged (by Samuel Colvin) as increasingly important — MCP for giving LLMs structured context, OpenTelemetry for observability/logging of AI applications.

## Mental Models
- **"More AI means more APIs"** (Frank Kilcommins epigraph): rather than AI displacing APIs, AI adoption is *driving demand* for well-designed APIs, since AI systems need data/tool access just like traditional software.
- **APIs as the alternative to "robotic process automation via screen scraping"**: per Doerrfield, if APIs aren't made accessible/well-defined for LLMs, agentic AI will route around them via scraping — making API design proactively AI-friendly is a competitive/defensive move, not just a nice-to-have.
- **Nondeterminism as the central design challenge**: unlike traditional software, LLM-driven agents don't reliably call the "correct" sequence of endpoints — hence the push toward deterministic building blocks (Arazzo) and strong fundamentals (per Colvin, "rock solid" unit tests for the deterministic parts of a nondeterministic system).
- **AI terminology nesting** (per Figure 12-1): AI (broadest) → machine learning (a major modern approach within AI, distinct from older rules-based expert systems) → generative AI/LLMs (one prominent application of machine learning).

## Anti-patterns
- **Returning overly large API payloads to an LLM-driven consumer**: increases token cost and measurably reduces calculation accuracy; avoid by trimming fields, excluding nested collections, and using pagination/filters.
- **Exposing count/summary questions only through full-record scans**: without a dedicated summary endpoint, LLMs behave erratically — scanning every record, guessing from IDs, or improvising unpredictably.
- **Forcing ID-based lookups as the only search path**: LLMs are "more comfortable using language than numbers," so APIs intended for AI consumption should support name/language-based search, not just numeric-ID lookups.
- **Letting an LLM call high-risk or side-effecting endpoints without safeguards**: providers themselves warn against this (Anthropic: not "on their own in high-risk situations"; Google: "may sometimes provide inaccurate information"); always apply permission restriction and review mechanisms.
- **Skipping unit tests because "it's just GenAI code"**: Colvin explicitly warns against this — deterministic parts of a nondeterministic system need even stronger, not weaker, test discipline.

## Code Examples
No executable code in this chapter — it is conceptual/strategic, setting up the Part III portfolio project (repository cloning and Codespace setup, following the same pattern as Parts I and II).

## Reference Tables
**AI/API overlap patterns**

| Pattern | Direction | Example in this book |
|---|---|---|
| Training data source | API → model | (General context) |
| Inference endpoint | Model → API | Ch. 13 |
| RAG | API/data → LLM context | Ch. 14 (LangChain) |
| Agentic | LLM → decides which API to call | Ch. 14, Ch. 15 |

**Table 12-1. AI agent frameworks**

| Software | Languages supported |
|---|---|
| Autogen | Python, .NET |
| LangChain/LangGraph | Python |
| LlamaIndex | Python, TypeScript |
| PydanticAI | Python |
| Vercel AI SDK | TypeScript |

**AI-readiness design tips summary**

| Tip | Why |
|---|---|
| Limit result size | Token cost; calculation accuracy degrades on large datasets |
| Consistent data structures (reused schemas/OAS) | Predictability improves LLM accuracy |
| Provide an SDK | Scopes calls appropriately, adds explanatory detail |
| Customize OAS (clear descriptions, unique operationIds) | Helps LLMs infer correct usage |
| Dedicated summary/count endpoint | Avoids erratic full-scan behavior |
| Name/language-based search endpoint | LLMs prefer language over numeric IDs |

## Worked Example
The chapter interviews Bill Doerrfield (Nordic APIs) on the AI/API intersection's trajectory (agents becoming more API-aware, eventual access to monetized/proprietary functionality, the risk of RAG-based scraping bypassing APIs entirely) and Samuel Colvin (Pydantic/PydanticAI founder) on the role of agent frameworks, the importance of fundamentals/testing in nondeterministic systems, and standards to watch (MCP, OpenTelemetry). It defines the AI/ML/generative-AI terminology hierarchy and previews the Arazzo Specification as a mechanism for making multi-step, multi-endpoint API workflows more deterministic for AI consumption. The chapter closes by setting up the Part III repository (`api-book-part-three` cloned as `ai-project`) and Codespace, mirroring the Part I/II setup pattern, in preparation for building a deployed ML API (Ch. 13), LangChain/LangGraph integration (Ch. 14), and a ChatGPT custom GPT/action (Ch. 15).

## Key Takeaways
1. AI and APIs overlap in (at least) four ways: training data, inference serving, RAG, and agentic tool-use — know which pattern your project needs.
2. Not every API/endpoint should be exposed to autonomous LLM agents without safeguards (human approval, permission restriction, logging/review) — provider-level warnings about LLM reliability apply directly to agentic API use.
3. Design APIs for AI consumption by limiting payload size, keeping schemas consistent and OAS-documented, providing a scoped SDK, writing clear OAS descriptions/operationIds, and offering dedicated summary and language-based search endpoints.
4. LLM-driven agents are nondeterministic; standards like the Arazzo Specification aim to make multi-endpoint business workflows more reliable/deterministic building blocks for AI use.
5. As AI adoption grows, well-designed, accessible APIs become more valuable, not less — the alternative is being bypassed via scraping-like techniques.
6. Fundamentals (unit testing, good API/security practice) matter *more*, not less, when LLMs are introduced into a system, because of the nondeterminism they add.

## Connects To
- **Ch 1**: AI developers were explicitly named as a target user persona back in Chapter 1's user-story work; this chapter operationalizes designing for them.
- **Ch 4/Ch 5**: The Pydantic-schema-consistency and OAS-description/operationId tips directly extend techniques already built into the SWC API.
- **Ch 13**: Builds the inference-endpoint pattern (model deployed as REST API) previewed here.
- **Ch 14**: Builds the RAG/agentic pattern using LangChain/LangGraph and the `swcpy` SDK.
- **Ch 15**: Builds a ChatGPT custom GPT/action using the OAS file, applying the AI-readiness design tips directly.
- **External**: Blobr's "Is Your API AI-ready? Our Guidelines and Best Practices"; IBM's "What Is Artificial Intelligence (AI)?"; the Arazzo Specification (OpenAPI Initiative); Kade Halabuza's "Syntax Sunday: Custom API Wrapper for GPTs"; Peter Schroeder's "AI + APIs — What 12 Experts Think The Future Holds."
