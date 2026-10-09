# Chapter 7: Scope Performance Domain

> **Coverage note**: This chapter is **partial**. The source extraction cuts off mid-sentence inside the Validate Scope process description (the final captured words are "...the key benefit of this process is that the scope is validated through an objective process to assure value and quality in the product, service, or result delivered. This—"). Tailoring Considerations, Interactions With Other Domains, and Check Results for this domain were **not captured** — see Scope & Limits.

## Core Idea
Scope is the domain where a project's expected value actually lives: the Scope performance domain's job is to ensure the project does *all* the required work and *no* unnecessary work, while treating quality as an integral feature of scope (not a separate bolt-on) — directly optimizing cost and schedule to maximize value.

## Frameworks Introduced
- **Value Breakdown Structure (VBS)**: a hierarchical structure connecting project scope → intended value → product scope. Top-level items are major deliverables, each assigned a value estimate (a dollar figure, a count like "students taught to read," or a % of total project value — 100% if mandatory). Decomposes further into subdeliverables, eventually meeting the WBS.
  - When to use: prioritizing deliverables/work and estimating the "drag cost" of a critical-path item — i.e., quantifying what delay on this item actually costs in value, not just in days.
  - How: assign a value estimate to every top-level deliverable before decomposing it; use that to rank what gets protected when trade-offs arise.
- **Quality as a feature of scope**: functional AND nonfunctional requirements are both part of scope — e.g., a bridge-construction project's scope includes not just "a bridge" but target thresholds for sturdiness, longevity, and maintainability.
  - When to use: any time scope is being defined — explicitly ask what the nonfunctional/quality thresholds are, not just the functional deliverable list.

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

## Mental Models
- Scope management's objective (value delivery) doesn't change between predictive and adaptive — only the *process* does: predictive is formal/document-driven (WBS, formal change control); adaptive is iterative/collaborative (backlog, product-owner-driven).
- Use the VBS alongside the WBS, not instead of it: the WBS tells you *what work exists*; the VBS tells you *what that work is worth* — use both together for trade-off decisions.
- Treat "no unnecessary work" as a scope-quality check, not just a schedule-efficiency one — unnecessary work directly erodes the value-per-investment ratio this whole domain exists to protect.

## Anti-patterns
- **Treating quality as separate from scope**: the standard explicitly frames quality (functional + nonfunctional requirements) as integral to scope — splitting them out risks under-specifying nonfunctional thresholds.
- **Using a formal WBS change-control mindset on an adaptive project**: adaptive scope changes flow through product-owner-driven backlog reprioritization, not a change-control board; applying predictive rigor here adds bureaucracy without the model fitting the approach.

## Reference Tables

**Scope Performance Domain — 6 Processes (as far as captured)**

| Process | Key Output | Predictive form | Adaptive form |
|---|---|---|---|
| Plan Scope Management | Scope management plan | Formal document | More iterative/collaborative |
| Elicit and Analyze Requirements | Requirements documentation | Requirements docs | User stories in a backlog |
| Define Scope | Project scope statement | Full description up front | High-level, via product roadmap/releases |
| Develop Scope Structure | WBS + WBS dictionary / product backlog | WBS | Product backlog (epics → user stories) |
| Monitor and Control Scope | Quality reports, verified deliverables, change requests | Formal variance/trend/root-cause analysis vs. baseline | — |
| Validate Scope *(partial — see coverage note)* | Accepted deliverables, change requests | Formal acceptance process | — |

## Worked Example

**VBS in practice**: For a literacy-nonprofit project, the top-level VBS deliverable "reading curriculum rollout" might be assigned a value of "500 students taught to read" (not a dollar figure). That deliverable decomposes into subdeliverables (teacher training materials, assessment tools, classroom kits), each inheriting a share of that value — e.g., teacher training might be tagged as contributing 40% of the outcome because untrained teachers block delivery entirely. This lets the team see that delays to teacher training carry more "drag cost" than delays to classroom kits, even if the kit work looks bigger on the WBS.

## Key Takeaways
1. Scope baseline mechanics differ sharply by approach: predictive = formal WBS + change control; adaptive = per-iteration backlog + product-owner approval, no formal CCB.
2. Always ask for nonfunctional/quality thresholds as part of scope definition — don't let "quality" become an unstated assumption.
3. Use a VBS to assign explicit value to deliverables before decomposing into a WBS, so trade-off and prioritization decisions are value-driven, not just task-driven.
4. A WBS dictionary is worth creating whenever deliverables are complex or interdependent — it's what keeps the WBS itself from becoming ambiguous.
5. Validate Scope has two distinct objectives — checking process-quality-standard adherence AND formalizing stakeholder acceptance — treat both as required, not just deliverable sign-off.

## Connects To
- **Ch 6 (Governance Performance Domain)**: Table 2-4 (in ch06) names Governance–Scope as potentially "the most important domain interaction" since scope is where project value lives.
- **Ch 3 (Principles — Embed Quality Into Processes and Deliverables)**: this chapter's "quality as a feature of scope" concept directly operationalizes that principle.
- **Scope & Limits**: this domain's Tailoring Considerations, Interactions With Other Domains, and Check Results subsections, plus the remainder of Validate Scope, Schedule/Finance/Stakeholders/Resources/Risk Performance Domains, Tailoring (Guide Ch 3), Inputs and Outputs (Guide Ch 4), Tools and Techniques (Guide Ch 5), and all appendices/glossary were **not captured** in this extraction.
