# Chapter 13: Tailoring

## Core Idea
Tailoring is the deliberate adaptation of approach, governance, and processes to the project's environment and objectives — "there is no single approach that can be applied to all projects all of the time," and tailoring is not a one-time exercise but a continuous, four-step improvement loop led by project stakeholders within organizational guardrails.

## Frameworks Introduced
- **The Four-Step Tailoring Process** (Figure 3-1/3-3): (1) Select Initial Development Approach → (2) Tailor for the Organization → (3) Tailor for the Project → (4) Implement Ongoing Improvement. Not a one-time sequence — Step 4 feeds back continuously via review points, phase gates, and retrospectives.
  - When to use: as the master checklist any time a project (or an organization standardizing project practices) needs to decide how much process rigor to apply.
- **Suitability filter**: not a rigid procedure, but a decision-making tool combining culture, team dynamics, and project-factor criteria to help teams discuss and decide the initial development approach (predictive/adaptive/hybrid) for their specific circumstances.
- **Tailoring at Organizational vs. Project level** (Figure 3-2): a general PM methodology gets tailored first into organization-level variants (e.g., "for Small Projects" vs. "for Large/Sensitive Projects"), which then get tailored again into individual project-level methodologies (Project A, B, C...). Two tailoring passes, not one.
- **Five process-tailoring actions**: Add (address unique conditions/increase rigor), Modify (adjust inputs/outputs/tools to fit), Remove (reduce unnecessary cost/effort), Blend (combine elements for added value), Align (ensure consistency in definition/application).
  - When to use: as the specific verb-set for any process-tailoring decision — frame every tailoring choice as one of these five actions.

## Key Concepts
- **Three things that can be tailored** (3.3): Life Cycle and Development Approach Selection; Processes; Engagement (people, empowerment, integration of diverse contributors into one team).
- **Three tailoring attributes at the project level** (3.4.3): Product/Deliverable (standards compliance, tangibility, industry/market, technology stability, timeframe, requirements stability, security classification, incremental/iterative feasibility), Project Team (size, geography, organizational distribution, experience, customer access, diversity), Culture (buy-in, trust, empowerment, organizational-culture alignment).
- **PMO's role in tailoring**: internal-only tailoring usually just needs PM approval; tailoring affecting external groups may need PMO approval. PMOs also provide ideas/solutions drawn from other projects (see Appendix X2).
- **Multi-organization tailoring**: when multiple organizations collaborate, each brings its own processes/governance — tailoring must harmonize these through integration planning, not just pick one organization's approach by default.

## Mental Models
- A single project can contain multiple experienced development approaches at once: a data-center build might use predictive for physical construction and adaptive for computing-capability work — at the *project* level that's "hybrid," but each subteam may only ever experience their own single approach.
- Too little process tailoring (too few processes) causes ineffective management; too much (more than needed) is costly/wasteful — tailoring is explicitly a balance, not a one-directional "add more rigor" exercise.
- Treat tailoring as validated change, not just change: "tailoring is not just about making changes, but also about verifying and validating that the tailored approaches are effective."

## Anti-patterns
- **Treating tailoring as a one-time setup step**: the standard explicitly frames it as continuous — review points, phase gates, and retrospectives should all trigger re-tailoring consideration.
- **Ignoring organizational mandates when tailoring**: some organizational policies or contracts mandate a specific approach — tailoring freedom isn't unlimited; check constraints first.
- **Picking one organization's process wholesale in a multi-org project**: this skips the harmonization work integration planning is supposed to do.

## Reference Tables

**Table 3-1 — Common Situations and Tailoring Suggestions (condensed)**

| Situation | Tailoring suggestion |
|---|---|
| Deliverables are poor quality | Root-cause analysis + targeted feedback/QA measures; focus on process improvement, not just more verification steps. |
| Team members unsure how to proceed | Assess specific needs; apply guidance, mentorship, training, knowledge-sharing. |
| Team members uncooperative, working in silos | Team-building activities + individual meetings to understand cause; strengthen bonds and coordination. |
| Long delays waiting for approvals | Streamline approvals — authorize people to decide up to certain value thresholds. |
| Too much work-in-progress / waste | Value stream mapping, Kanban boards to visualize work and find solutions. |
| Stakeholders disengaged or giving negative feedback | Check timeliness/relevance of information; tailor communication method (simplicity vs. depth). |
| Lack of transparency/understanding of progress | Verify right data is collected/analyzed/shared/discussed; validate agreement on measures with team+stakeholders. |
| Unprepared team keeps reacting to surfacing issues/risks | Explore root causes for gaps in process or activities. |

## Worked Example

**Hybrid-by-subteam (3.3.1)**: Building a new data center uses a predictive approach for the physical building construction/finishing, and an adaptive approach for establishing the needed computing capabilities. Viewed project-wide, this is a hybrid approach — but the construction team experiences pure predictive, and the computing team experiences pure adaptive. Lesson: "hybrid" is often a project-level label that doesn't mean every subteam personally works hybrid — tailor at the level each team actually experiences, not just at the label level.

## Key Takeaways
1. Run the four-step tailoring loop (select → tailor for org → tailor for project → improve continuously) as an ongoing cycle, not a kickoff-only task.
2. Use the suitability filter (culture + team dynamics + project factors) as a structured discussion tool when choosing predictive/adaptive/hybrid — don't default by habit.
3. Frame every process tailoring decision as Add/Modify/Remove/Blend/Align — this makes tailoring decisions explicit and auditable.
4. Check organizational/contractual mandates before assuming full tailoring freedom.
5. In multi-organization projects, harmonize via integration planning rather than imposing one org's process.
6. Use Table 3-1 as a first-response playbook when a specific team dysfunction (poor quality, silos, approval delays, disengaged stakeholders) shows up — match the suggestion to the actual root cause, not just the symptom.

## Connects To
- **Ch 3 (Principles)**: Figure 3-4 shows the six principles (grouped as Proactive/Ownership/Value-Driven mindset) sitting above all seven performance domains, with the instruction to "tailor to fit the project context" — principles guide *how* every domain gets tailored.
- **Ch 4 (Project Life Cycles)**: development-approach selection (predictive/adaptive/hybrid) is tailoring's first step — this chapter cross-references that decision directly.
- **Ch 6–ch12 (all Performance Domains)**: each domain's own "Tailoring Considerations" subsection is this chapter's general framework applied domain-by-domain (Section 2's individual guidance, as this chapter notes explicitly).
- **Appendix X2 (PMOs)**: referenced directly for the PMO's role in reviewing/approving tailored approaches.
