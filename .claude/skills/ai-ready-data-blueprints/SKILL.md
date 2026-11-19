---
name: ai-ready-data-blueprints
description: "Knowledge base from \"AI-Ready Data Blueprints\". Use when designing data foundations for GenAI/agentic AI applications — data frameworks, data wrangling/preparation, data governance/security/compliance/orchestration for GenAI, knowledge bases and vector databases, or optimizing AI applications for production readiness."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# AI-Ready Data Blueprints
**Authors**: Srikanth Sopirala and Harsha Tadiparthi (both Principal AI Specialists at AWS) | **Chapters**: 6 | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch04`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Five Emerging GenAI Architectural Patterns** (the book's master framework, Ch 1): Knowledge graphs for contextual intelligence, Event-driven foundations for real-time AI, Lakehouse as unified AI platform, Semantic search as the new query layer, Agent-ready digital properties. Not mutually exclusive — successful organizations combine patterns, starting with the one matching their highest-value use case, proving value in 3–6 months before expanding.

**The AI-Ready Data Framework (8 pillars, Ch 2)**: Capturing business logic/context, Ensuring data quality/consistency, Managing complexity/diversity, Maintaining security/compliance/privacy, Enabling information sharing/collaboration, Supporting scale/performance, Managing data as a strategic product, Empowering users with documentation/guidance. Use as a comprehensive gap-assessment checklist — strength in one pillar never implies readiness in another.

**Semantic Layer Architecture (5 layers, Ch 3–5)**: Data Foundation (lakehouse) → Metadata/Ontology Management (business glossaries + knowledge graphs) → Transformation/Enrichment (entity extraction, NLP/CV) → Vectorization/Indexing (embeddings + vector DBs) → APIs/Reasoning (knowledge APIs, context management). Governance is embedded throughout all five layers, not a separate layer. Build incrementally — never try to perfect the schema up front.

**GenAI Data Preparation Maturity Model (Ch 3)**: Foundation building (0–25%) → Capability development (26–50%) → Optimization (51–75%) → Innovation (76–100%). Most organizations start at 0–25% and need 12–18 months to reach basic readiness — use this to calibrate expectations before committing to an architecture.

**Knowledge Base / Vector DB / RAG as One System (Ch 5)**: these are not three independently selectable technologies — the knowledge base coordinates sync of enterprise sources into vector databases, and RAG is the consumption pattern that grounds LLM responses. Design holistically.

**The 5–6x Rule (Ch 5, critical mental model)**: data preparation quality — especially chunking strategy — has far more leverage on RAG performance than model selection. Empirically: chunking strategy swings performance ~35%, embedding model choice ~27%, but switching the LLM only ~6%. When RAG underperforms, fix data preparation before reaching for a bigger model.

**GraphRAG (Ch 5)**: combine vector search (semantic similarity) with graph traversal (explicit, typed relationships) whenever a question requires multihop reasoning ("why are these connected," not just "what's similar") — vector embeddings alone cannot represent relationship type or direction.

**Responsible AI — 6 Principles (Ch 4)**: Fairness, Transparency, Accountability, Privacy/Security, Reliability, Human Oversight. For each, pair a detection step (bias tools, drift monitors) with a mitigation step (guardrails, prompt engineering, human override) — detection alone is insufficient.

**Intelligent Semantic Metadata Layer (Ch 6)**: a knowledge-graph layer (analogous to a Hive Metastore) that lets AI agents discover/interpret/reason over raw enterprise files in place, without requiring migration into a vector store — essential once data scale (petabytes, millions-to-quadrillions of files) makes wholesale embedding impractical, too costly, or non-compliant.

**Seven Capability Areas for Agentic Production Readiness (Ch 6)**: Core runtime/deployment, Framework/protocol interoperability (MCP, A2A), Tool integration/execution, Memory/knowledge management, Security/compliance, Observability/QA, Multi-agent/interaction. A menu for deliberate investment decisions, not a mandate to maximize every dimension.

**Business Outcome Drives Data Quality (Ch 6)**: there is no universal definition of "clean" data — what is noise for one use case (email headers, for intent classification) is the primary signal for another (interaction-frequency analysis). Always define quality relative to the specific business question.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Introduction to AI-Ready Data Foundation | Four Stages of GenAI Evolution; Five Architectural Patterns; ML vs. GenAI Development Cycle; Data Readiness Checklist |
| ch02 | Data Framework for GenAI and Agentic AI Applications | Eight-Pillar AI-Ready Data Framework; Data Framework Evolution Timeline; Data Product Thinking; Four Implementation Blueprints |
| ch03 | Data Wrangling and Data Preparation for GenAI and Agentic AI Applications | Traditional ML vs. GenAI Paradigm Shift; Maturity Model; Semantic Layer Architecture; Governance Framework (5 areas) |
| ch04 | Data Governance, Security, Compliance, and Orchestration for GenAI | AI Data Governance Operating Model (9 components); Responsible AI Principles (6); HITL Lifecycle Matrix; Orchestration Patterns (linear/modular/agentic) |
| ch05 | Knowledge Bases and Vector Databases | KB/VectorDB/RAG as One Architecture; GraphRAG; Seven-Stage Data Transformation; RAG Architecture Evolution (6 tiers); Vector DB Selection Framework; Chunking Strategy Selection |
| ch06 | AI Application Optimization for Production Readiness | Twin Pillars of AI Optimization; Modular Vector Store Design; Three-Step Large-Context Handling; Automated Reasoning Taxonomy; Intelligent Semantic Metadata Layer; Seven Capability Areas |

## Topic Index

- **Agentic AI orchestration** → ch04, ch06
- **AI data governance operating model** → ch04
- **Automated reasoning (CoT/ToT/GoT)** → ch06
- **AWS reference architecture** → ch03
- **Business glossary** → ch02, ch04
- **Chunking strategy** → ch03, ch05
- **Context window management** → ch06
- **Data mesh** → ch01
- **Data product thinking** → ch02
- **Data quality dimensions (traditional + GenAI-specific)** → ch02, ch03, ch05
- **Embedding dimensionality / drift** → ch05
- **Event-driven architecture** → ch01, ch02
- **Five architectural patterns** → ch01
- **GraphRAG / knowledge graphs** → ch01, ch02, ch03, ch05, ch06
- **HITL (human-in-the-loop)** → ch04
- **Intelligent semantic metadata layer** → ch06
- **Knowledge base fundamentals** → ch05
- **Lakehouse architecture** → ch01, ch03
- **Maturity model (data preparation)** → ch03
- **Orchestration patterns (RAG, agentic)** → ch04
- **Production readiness capabilities** → ch06
- **RAG architecture evolution** → ch05
- **Reasoning tokens** → ch06
- **Responsible AI (fairness, transparency, etc.)** → ch04
- **Semantic layer architecture** → ch02, ch03, ch05
- **Vector database selection** → ch05

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — techniques and architectures with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 6 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
