# Chapter 4: Project Life Cycles

## Core Idea
Every project moves through phases via some development approach (predictive, adaptive, or hybrid) and some delivery cadence, and regardless of approach, project work always falls into five universal Project Management Focus Areas (Initiating, Planning, Executing, Monitoring & Controlling, Closing) that overlap and iterate rather than run strictly in sequence.

## Frameworks Introduced
- **The Spectrum of Development Approaches (Figure 4-2)**: Predictive ↔ Hybrid ↔ Adaptive, ordered by "increasingly iterative and incremental."
  - When to use: as the first question when scoping a new project — where does this sit on the spectrum, and why?
  - How: check the Section 4.3 decision factors (deliverables, project, organization) below.
- **The Inverted Triangle / Constraint Options (Figure 4-3)**: in adaptive approaches, pick which constraint is fixed and which flex — (a) fix budget+schedule, vary scope, or (b) fix scope, vary budget/schedule.
  - When to use: negotiating a project charter under an adaptive approach — explicitly name which constraint absorbs unexpected change.
- **Five Project Management Focus Areas** (formerly "Process Groups"): Initiating → Planning → Executing → Monitoring & Controlling → Closing. Not phases — they recur and overlap within every phase and across the whole project, regardless of development approach.
  - How: use as a checklist per phase/iteration — "did we initiate, plan, execute, monitor/control, and close this increment?" — not as a one-time linear sequence.
- **Hybrid Levels (PMI Disciplined Agile)**: Level 1 (predictive-dominant, adaptive elements reduce pain points) → Level 2 (both contribute significantly) → Level 3 (adaptive-dominant, predictive elements satisfy business constraints).
  - When to use: describing precisely *how* hybrid a project is, instead of just saying "hybrid."

## Key Concepts
- **Project phase**: a collection of logically related activities culminating in one or more deliverables/outcomes; phases can be sequential, overlapping, or transitional.
- **Phase gate** (stage gate / gate review / iteration review / decision point review): a review at phase end (traditionally) or start ("pre-phase gate") producing a go/no-go-type decision: continue, continue with modification, end, remain, repeat, or park.
- **Predictive approach** (waterfall/plan-driven/traditional): scope/schedule/cost/quality/risk defined early and expected to stay stable; heavy up-front planning; integrated baseline = schedule+cost+scope baselines together.
- **Incremental approach**: scope well-understood up front, but delivered in sequential/overlapping increments (within a predictive framework) — e.g., Figure 4-5's Increment 1/2/3 with Concept→Plan→Design→Build repeated.
- **Adaptive approach** (change-driven/agile): high uncertainty/volatility; iterative (refine through repeated cycles) and incremental (deliver in usable segments) by nature; often uses flow-based scheduling (Kanban, Theory of Constraints).
- **Delivery cadence**: single delivery (all outcomes at project end), multiple deliveries (components at different times), periodic deliveries (fixed regular schedule — common in adaptive), continuous delivery (production-ready increments shipped continuously).
- **Progressive elaboration**: ongoing refinement of the project management plan as more information becomes available — predictive approaches front-load planning; adaptive approaches "roadmap" lightly up front then replan continuously.

## Mental Models
- Treat "development approach" and "development phase" as distinct terms — approach = predictive/adaptive/hybrid method; phase = a stage within the life cycle where creation/testing happens. Don't conflate them.
- Think of risk/uncertainty and stakeholder influence-on-change as both highest at project start and decreasing over time (Figure 4-1) — plan your highest-leverage stakeholder engagement and risk responses early.
- A hybrid project is not "a little of both everywhere" — identify which of the four Section 4.2.3 patterns applies (adaptive-then-predictive, simultaneous-by-stream, small-adaptive-in-predictive, small-predictive-in-adaptive) to know where control mechanisms differ.

## Anti-patterns
- **Defaulting to predictive because it's familiar, when scope is unstable**: the standard explicitly ties predictive's suitability to requirements being "sufficiently clear and stable early" — using it despite instability invites costly rework.
- **Confusing Focus Areas with phases**: treating "Planning" as a one-time phase rather than a recurring activity within every phase/iteration misses that Monitoring & Controlling runs in parallel with everything else, not as a separate stage.
- **Picking "hybrid" without specifying the pattern or level**: vague hybrid framing (Section 4.2.3) hides which constraints are actually fixed vs. flexible for governance purposes.

## Reference Tables

**Development approach selection factors (condensed from Sections 4.3.1–4.3.3)**

| Category | Favors Predictive | Favors Adaptive |
|---|---|---|
| Requirements certainty | Well known, stable | Unclear, evolving |
| Degree of innovation | Low | High |
| Ease of change | Difficult/costly to change | Easy to incorporate change |
| Safety/regulatory requirements | High (up-front planning for compliance) | Low |
| Value of frequent feedback | Low | High |
| Schedule constraint | Fixed end date | Value in early partial delivery |
| Financing uncertainty | Low | High |
| Org structure | Fixed functional/hierarchical | Network-oriented |
| Org culture | Managing/directing, baseline-driven | Embraces uncertainty, self-managed teams |
| Team size | Larger, scope-driven | Smaller (often 3–9 members) |

## Worked Example

**Multi-deliverable hybrid pattern**: A project with two deliverables — software developed adaptively, and a new data center it must install into, built predictively. Rather than forcing one approach project-wide, the team runs each deliverable stream under its best-fit approach and coordinates at integration points. This maps to the "simultaneous, by stream" hybrid pattern (Figure 4-9) rather than a sequential adaptive→predictive handoff (Figure 4-8).

**Phase gate decision (Section 4.1)**: At a gate, the project team compares performance/progress to project and business documents and issues a decision: continue; continue with modification; end the project/phase; remain in the phase; repeat the phase/elements; or park temporarily for a more urgent business initiative. Practical use: when writing a stage-gate checklist, use these six outcomes explicitly rather than a binary go/no-go — "park" and "repeat" are easy to forget.

## Key Takeaways
1. Place every project on the predictive↔adaptive spectrum deliberately, using the deliverables/project/organization factor checklist — don't default by habit.
2. In adaptive/hybrid work, explicitly state which constraint (scope vs. budget/schedule) is fixed and which absorbs change (the inverted triangle).
3. The five Focus Areas are not phases and are not strictly sequential — Monitoring & Controlling runs continuously in parallel with the others.
4. Match delivery cadence (single/multiple/periodic/continuous) to where stakeholder value realization actually matters, not just to development approach by default.
5. Name the specific hybrid pattern and level (1/2/3) in use — it changes which governance/control mechanisms apply where.
6. Use all six phase-gate outcomes (not just go/no-go) when designing stage-gate reviews.

## Connects To
- **Ch 3 (Principles)**: Adopt a Holistic View and Focus on Value principles directly inform development-approach selection trade-offs.
- **Guide to PMBOK — Tailoring (Ch 3, not captured in this extraction)**: would normally detail how to tailor these life-cycle choices further — see Scope & Limits.
- **Guide to PMBOK — Schedule, Risk Performance Domains**: delivery cadence and risk/uncertainty curves (Figure 4-1) connect directly to those domains' mechanics.
