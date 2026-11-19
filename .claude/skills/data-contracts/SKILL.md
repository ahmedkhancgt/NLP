---
name: data-contracts
description: "Knowledge base from \"Data Contracts\". Use when designing or implementing data contracts — data quality, detection and prevention of data issues, data contract components, organizational change management, shift-left data culture, or measuring the impact of data contract adoption."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Contracts
**Author**: Not stated in source | **Chapters**: 12 | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch07`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Data Contract**: an agreement between data producers and consumers, established, updated, and enforced via an API/CI-CD pipeline, defining schema and semantic expectations for a data asset. Treat it as the "pull request for data" — the missing change-management layer software engineering has (PRs, code review) that data never had.

**Data Quality ≠ pristine data.** Defined as "an organization's ability to understand the degree of correctness of its data assets, and the trade-offs of operationalizing such data at various degrees of correctness... as it pertains to being fit for use by the data consumer." Pitch data quality investment on business/revenue risk, never on abstract purity.

**The Four Architecture Components**: Data Assets → Contract Definition → Detection → Prevention. Assets define scope (transactional DB, analytical DB, event streams, first-party-on-third-party-platform); Contract Definition codifies expectations (spec + business logic + schema registry/catalog); Detection finds violations (code-level SCA, data-in-motion, historical); Prevention operationalizes blocking/alerting (version control, CI/CD, monitoring).

**The Data Contract Workflow (7 steps)**: consumer identifies a constraint → requests a contract → producer confirms viability → contract confirmed as code → producer opens a PR → CI/CD checks for violation → on failure, owners notified (hard or soft failure per spec) / on pass, asset updates flow downstream.

**Place contracts upstream first.** Default starting point: the transactional-database → analytical-database boundary. It's the most upstream point your org actually controls, and prevents the most downstream damage. (Full placement decision tree in cheatsheet.md.)

**Data Contracts vs. Data Observability**: contracts *prevent* known issues before a change ships (targeted, CI/CD-embedded); observability *detects* unknown issues after they occur (broad, whole-stack). Flashlight (observability) vs. laser pointer (contracts) — use both, never one instead of the other.

**Data debt compounds faster than tech debt** because data teams are tightly coupled through shared business entities (unlike independently deployable microservices). GIGO (garbage-in, garbage-out) can only be fixed retrospectively — prevention beats detection.

**Dunbar's Number (~150) + Conway's Law**: past ~150 people, organic communication breaks down; and a system's structure mirrors its organization's communication structure. Data contracts work by narrowing the combinatorial web of relationships down to just the relevant producer/consumer link — this is *why* contracts scale where ad hoc Slack messages don't.

**Shift Left Data**: moving data management responsibility upstream to where data is generated, directly modeled on DevSecOps's 2012–2014 shift left security movement. The core lesson: integrate into tools producers already use (Git, CI/CD) rather than introducing a separate system they must learn in isolation.

**Perception gap** (software vs. data teams) is the #1 adoption blocker — not technology. Use first-principles "going deeper" questioning to find the real root cause before building anything. Software engineers must *own* the adoption process, not just tolerate it; find "hidden influencers," not just org-chart leaders.

**Airplane projects vs. airline projects**: an airplane project proves technical viability (most ETL tools); an airline project solves adoption/incentive/friction problems (data catalogs, data contracts). Most failed contract rollouts fail because they were built as airplane projects when they needed to be airline projects. "If we build it, they will come" never happens — design explicitly against friction.

**Change management is the actual problem data contracts solve** — not tooling, not standards. GitHub itself is best understood as a "federated change management platform," not a DevOps tool. Leaders invest in solving strategic problems, not in technology for its own sake.

**Steel Threads**: a data product qualifies as a steel thread if its failure would cause immediate, measurable business disruption. Select your first pilot via a weighted decision matrix (business criticality, engagement, volatility, DQ issues, dependency complexity, ease of implementation, visibility) — never by convenience.

**Kotter's 8-step change model** for scaling beyond the first pilot: urgency → guiding coalition → vision/strategy → communicate vision → empower action → short-term wins → consolidate gains → anchor in culture. Needs ≥75% of management convinced the status quo is untenable.

**Measurement is persuasion infrastructure, not just reporting.** Build a Minimum Viable Metric Tree connecting your contract to a company-visible North Star metric. Metrics alone don't change minds — pair them with monomyth-structured storytelling (Departure/Transformation/Return).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Why the Industry Now Needs Data Contracts | Data Debt, GIGO Cycle, Data-Centric vs. Model-Centric AI, Shift Left Data |
| ch02 | Data Quality Isn't About Pristine Data | Data Quality Definition, Leading/Lagging Indicators, Violated Expectations Taxonomy |
| ch03 | The Challenges of Scaling Data Infrastructure | Data Discovery, Design Surface Requirements, Concept Drift, Dunbar's Number + Conway's Law |
| ch04 | An Introduction to Data Contracts | Data Contract Workflow, Maturity Curve, 4 Architecture Components, Contracts vs. Observability |
| ch05 | Data Contract Components: Data Assets and Contract Definition | Data Asset Categories, Contract Spec Design Criteria, Business Logic Lifecycle, Schema Registry vs. Catalog |
| ch06 | Data Contract Components: Detection and Prevention | 3 Detection Categories, DAMA 9 Dimensions, WAP Pattern, 5 Ws/1 H Alerting |
| ch07 | Implementing Data Contracts | Reference Implementation, Metadata-Only Detection, Violation Resolution Loop |
| ch08 | Real-World Case Studies of Data Contracts in Production | Convoy/Glassdoor/Adevinta case studies, Push vs. Pull Architecture, Semantic vs. CDC Events |
| ch09 | Shift Left: The Cultural Change Needed for Data Contracts | First-Principles Diagnostic, 1-10-100 Rule, Perception Gap, DevSecOps Parallel |
| ch10 | Change Management: The Crux of People, Process, and Technology | Airplane vs. Airline Projects, Friction, Hypothesis-Led Strategy (McKinsey/BCG) |
| ch11 | Creating Your First Wins with Data Contracts | Steel Threads, Weighted Decision Matrix, Influence/Interest Matrix, Kotter's 8 Steps |
| ch12 | Measuring the Impact of Data Contracts | POSIWID, Hubbard's Clarification Chain, MVMT, Monomyth Storytelling |

## Topic Index
- **ACID / Invariants** → ch05
- **Alert design (5 Ws/1 H)** → ch06
- **Business logic lifecycle** → ch05, ch06, ch09
- **Change data capture (CDC)** → ch04, ch05, ch06, ch08
- **Concept/data drift** → ch03
- **Contract spec (JSON Schema example)** → ch05, ch07
- **Conway's Law** → ch03, ch09, ch10
- **Convoy / Glassdoor / Adevinta Spain case studies** → ch08
- **Data catalog vs. Schema Registry** → ch05
- **Data debt** → ch01, ch02
- **Data discovery** → ch03
- **Data product definition** → ch04, ch11
- **DevSecOps parallel** → ch09
- **Dunbar's Number** → ch03, ch09, ch10, ch11
- **Friction (adoption)** → ch10
- **Kotter's 8-step model** → ch11
- **Maturity curve (awareness/collaboration/ownership)** → ch04
- **Metric trees (MVMT)** → ch12
- **OLTP vs. OLAP** → ch02
- **Perception gap (software vs. data teams)** → ch09
- **Shift left data** → ch01, ch09
- **Static code analysis (SCA)** → ch06, ch08
- **Steel threads / weighted decision matrix** → ch11
- **Storytelling (monomyth)** → ch12
- **Write-Audit-Publish (WAP)** → ch06, ch08

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — techniques and organizational patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 12 chapters, synthesized as structure, not verbatim text. The book's own back-of-book Index is excluded as non-substantive.
