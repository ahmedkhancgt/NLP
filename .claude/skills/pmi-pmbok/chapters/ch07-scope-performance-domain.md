# Chapter 7: Scope Performance Domain

## Core Idea
Scope is the domain where a project's expected value actually lives: the Scope performance domain's job is to ensure the project does *all* the required work and *no* unnecessary work, while treating quality as an integral feature of scope (not a separate bolt-on) — directly optimizing cost and schedule to maximize value.

## Frameworks Introduced
- **Value Breakdown Structure (VBS)**: a hierarchical structure connecting project scope → intended value → product scope. Top-level items are major deliverables, each assigned a value estimate (a dollar figure, a count like "students taught to read," or a % of total project value — 100% if mandatory). Decomposes further into subdeliverables, eventually meeting the WBS.
  - When to use: prioritizing deliverables/work and estimating the "drag cost" of a critical-path item — i.e., quantifying what delay on this item actually costs in value, not just in days.
  - How: assign a value estimate to every top-level deliverable before decomposing it; use that to rank what gets protected when trade-offs arise.
- **Quality as a feature of scope**: functional AND nonfunctional requirements are both part of scope — e.g., a bridge-construction project's scope includes not just "a bridge" but target thresholds for sturdiness, longevity, and maintainability.
  - When to use: any time scope is being defined — explicitly ask what the nonfunctional/quality thresholds are, not just the functional deliverable list.
- **Scope measurement formulas** (Table 2-6): Scope Definition Accuracy (%) = Planned Scope Items Correctly Delivered / Total Planned Scope Items × 100; Scope Creep (%) = Unplanned Deliverables / Total Deliverables × 100; Requirements Stability (%) = Unchanged Requirements / Total Requirements × 100.
  - When to use: quantifying scope health objectively instead of relying on a qualitative "scope feels okay" check.

## Key Concepts
- **Business case**: documented economic feasibility study establishing the validity of benefits a project/program/portfolio component will deliver — includes costs, benefits, value-creation approach, and success criteria.
- **Project scope**: all the work performed to deliver the product/service/result with specified features/functions — "the most important component of any project's baseline" since it encapsulates expected value.
- **Requirement**: a condition/capability necessary in a product/service/result to satisfy a business need.
- **Scope baseline**: in predictive environments, the approved formal scope documents (WBS + scope statement + WBS dictionary), changeable only via formal change control, forming part of the Performance Measurement Baseline (PMB) alongside schedule and cost baselines. In adaptive environments, defined per-iteration from prioritized requirements, with the **product owner** dynamically approving changes — no formal change control procedure.
- **Work Breakdown Structure (WBS)**: hierarchical decomposition of total scope into manageable work packages (predictive/hybrid); its agile analog is the **product backlog** decomposed into epics/features/user stories.
- **WBS dictionary**: elaborates each WBS component (scope description, milestones, responsible parties, resource needs, acceptance criteria) for complex/interdependent deliverables.
- **Product scope**: description of the features/functions/characteristics of the deliverable itself (the "what," vs. project scope's "the work to get there").
- **Product backlog**: dynamic, prioritized list of work items/features — the adaptive-environment framework for managing scope via continuous prioritization.
- **Definition of Done (DoD)**: checklist of criteria a deliverable (or user story/feature/increment, in adaptive work) must meet to be considered complete and ready for release/customer use.
- **Validate Scope**: has two objectives — checking that processes meet quality standards, AND formalizing stakeholder acceptance of deliverables. Both matter; it's not just a sign-off step.

## Mental Models
- Scope management's objective (value delivery) doesn't change between predictive and adaptive — only the *process* does: predictive is formal/document-driven (WBS, formal change control); adaptive is iterative/collaborative (backlog, product-owner-driven).
- Use the VBS alongside the WBS, not instead of it: the WBS tells you *what work exists*; the VBS tells you *what that work is worth* — use both together for trade-off decisions.
- Treat "no unnecessary work" as a scope-quality check, not just a schedule-efficiency one — unnecessary work directly erodes the value-per-investment ratio this whole domain exists to protect.
- Scope doesn't exist in isolation — Schedule and Finance are the most directly affected by scope change, but Risk and Stakeholders are significantly connected too; always route scope changes through governance since they can trigger budget/resource consequences.

## Anti-patterns
- **Treating quality as separate from scope**: the standard explicitly frames quality (functional + nonfunctional requirements) as integral to scope — splitting them out risks under-specifying nonfunctional thresholds.
- **Using a formal WBS change-control mindset on an adaptive project**: adaptive scope changes flow through product-owner-driven backlog reprioritization, not a change-control board; applying predictive rigor here adds bureaucracy without the model fitting the approach.
- **Ignoring sustainability in scope**: the WBS or backlog should explicitly include activities to manage sustainability impact (e.g., CO2 emissions, local biodiversity) when the project has a significant environmental footprint — this is a named Check Results outcome, not an optional add-on.

## Reference Tables

**Scope Performance Domain — 6 Processes**

| Process | Key Output | Predictive form | Adaptive form |
|---|---|---|---|
| Plan Scope Management | Scope management plan | Formal document | More iterative/collaborative |
| Elicit and Analyze Requirements | Requirements documentation | Requirements docs | User stories in a backlog |
| Define Scope | Project scope statement | Full description up front | High-level, via product roadmap/releases |
| Develop Scope Structure | WBS + WBS dictionary / product backlog | WBS | Product backlog (epics → user stories) |
| Monitor and Control Scope | Quality reports, verified deliverables, change requests | Formal variance/trend/root-cause analysis vs. baseline | — |
| Validate Scope | Accepted deliverables, change requests | Formal acceptance process | — |

**Tailoring Considerations** (2.2.3): dependency on external partners (harmonize contractual scope commitments across suppliers/subcontractors); environmental dynamics (high market/tech volatility needs flexible, iterative-feedback scope management); design phase intensity (pharma/construction invest heavily up front to avoid costly later changes); adaptive/hybrid life cycles (extra tailoring needed when overall baselines coexist with subteam-level backlog iteration).

**Table 2-6 — Check Outcomes, Scope Performance Domain (condensed)**

| Outcome | How to check |
|---|---|
| Effective change management | Predictive: change log shows impact across Scope/Schedule/Finance/Stakeholders/Resources/Risk. Adaptive: backlog shows rate of scope completion, rate of scope addition, and stakeholder-feedback prioritization. |
| Clear understanding of requirements | Predictive: few changes to initial requirements. Adaptive: each iteration gives a clear short-term understanding, enabling rolling-wave refinement of longer-term requirements. |
| Aligned with business objectives/strategy | Business case + strategic plan + authorizing documents demonstrate alignment. |
| Stakeholders accept/satisfied with deliverables | Interviews, observation, end-user feedback; complaint/return rates as a proxy. |
| Scope items clearly defined | Scope Definition Accuracy (%) = Planned Scope Items Correctly Delivered / Total Planned Scope Items × 100 |
| Scope creep measured | Scope Creep (%) = Unplanned Deliverables / Total Deliverables × 100 |
| Requirements stability measured | Requirements Stability (%) = Unchanged Requirements / Total Requirements × 100 |
| Sustainability considered in scope | WBS/backlog includes activities to manage sustainability impact (e.g., CO2 emissions, biodiversity). |

## Worked Example

**VBS in practice**: For a literacy-nonprofit project, the top-level VBS deliverable "reading curriculum rollout" might be assigned a value of "500 students taught to read" (not a dollar figure). That deliverable decomposes into subdeliverables (teacher training materials, assessment tools, classroom kits), each inheriting a share of that value — e.g., teacher training might be tagged as contributing 40% of the outcome because untrained teachers block delivery entirely. This lets the team see that delays to teacher training carry more "drag cost" than delays to classroom kits, even if the kit work looks bigger on the WBS.

**Tailoring example (2.2.3.1)**: A city-park project has a tight timeline and multiple stakeholders giving requirements/feedback. Rather than rushing to construction, the project manager invests extra time in the design phase, producing multiple mock-ups to evaluate before committing resources — trading a slower start for avoiding costly rescoping mid-construction.

## Key Takeaways
1. Scope baseline mechanics differ sharply by approach: predictive = formal WBS + change control; adaptive = per-iteration backlog + product-owner approval, no formal CCB.
2. Always ask for nonfunctional/quality thresholds as part of scope definition — don't let "quality" become an unstated assumption.
3. Use a VBS to assign explicit value to deliverables before decomposing into a WBS, so trade-off and prioritization decisions are value-driven, not just task-driven.
4. A WBS dictionary is worth creating whenever deliverables are complex or interdependent — it's what keeps the WBS itself from becoming ambiguous.
5. Validate Scope has two distinct objectives — checking process-quality-standard adherence AND formalizing stakeholder acceptance — treat both as required, not just deliverable sign-off.
6. Use the three scope formulas (Definition Accuracy, Scope Creep, Requirements Stability) as objective, trackable health checks, not just gut-feel status.
7. Route every scope change through governance — it has knock-on effects on Schedule, Finance, Risk, and Stakeholders that a local scope-only decision will miss.

## Connects To
- **Ch 6 (Governance Performance Domain)**: Table 2-4 (in ch06) names Governance–Scope as potentially "the most important domain interaction" since scope is where project value lives.
- **Ch 3 (Principles — Embed Quality Into Processes and Deliverables)**: this chapter's "quality as a feature of scope" concept directly operationalizes that principle.
- **Ch 8 (Schedule)** and **Ch 9 (Finance)**: named as the domains most directly impacted by any scope change.
- **Ch 12 (Risk)**: rescoping is a named strategy to mitigate, avoid, or transfer risk.
