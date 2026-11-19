# Chapter 10: Change Management: The Crux of People, Process, and Technology

## Core Idea
Data contracts are fundamentally a change management solution, not a technology; leadership invests in data contracts only when framed as solving the strategic problem of managing change across people and process, and getting adoption right requires treating the initiative as an "airline project" (incentives, friction, adoption) rather than an "airplane project" (pure technical capability).

## Frameworks Introduced
- **Airplane vs. Airline Projects**: airplane projects prove technical viability (building the capability); airline projects solve for business/adoption viability (getting people to actually use it).
  - When to use: diagnosing why a technically sound data contract (or catalog, or any tool) implementation isn't getting adopted.
  - How: ask the airline-project questions (why would engineers use this, where would they resist, is it useful to consumers, are violations addressed in time) before or alongside building the tooling — technology alone ("if we build it, they will come") never works.
- **Forward-Looking vs. Backward-Looking Problems**: forward-looking = handling new/changing data as it's produced; backward-looking = handling existing/legacy data not yet under contract.
  - When to use: scoping what your contract system needs to cover, beyond just the "happy path" new-data case.
  - How: explicitly plan for both — what happens to old, uncontracted data, and how consumers discover what is/isn't under contract — not just the flow for newly onboarded data.
- **Friction as the Adoption Lever** (via Peep Laja/CXL conversion-rate framing): adoption is driven by minimizing blockers to a desired action, not by willpower or mandate.
  - When to use: designing any step that asks producers/consumers to do something they aren't naturally inclined to do.
  - How: identify every friction point (unfamiliar systems, unclear ownership, slowed shipping, unknown consumers) and systematically remove it — make the right thing the easy thing.
- **Hypothesis-Led Strategy Approach** (McKinsey/BCG, 7 steps): define the problem → structure it (issue tree) → prioritize issues (impact/ease matrix) → develop a work plan → conduct analyses → synthesize findings (Pyramid Principle) → develop recommendations (SCQA).
  - When to use: building an internal strategy and pitch for introducing data contracts (or any initiative) to leadership.
  - How: "go deeper" iteratively on the problem statement until you hit one with real traction (the authors' own iteration ended at "how do we create the most effective change management workflow for important data"); structure with an issue tree; prioritize with an impact/ease matrix; synthesize with the Pyramid Principle (decision → arguments → data); present with SCQA (situation, complication, question, answer).

## Key Concepts
- **Change management**: the discipline of handling change to codebases, business logic, data users, expectations, producers, and consumers — the thing data contracts actually operationalize (GitHub itself is reframed as "a federated change management platform," not just a DevOps tool).
- **Data supply chain**: mapping data roles onto physical supply chain roles — producers≈farmers/miners, data engineers≈manufacturers, data warehouse≈distributor, BI tools≈retailers, analysts/scientists≈end consumers.
- **Issue tree**: McKinsey technique for decomposing a problem into hierarchical subcomponents to identify prerequisite issues.
- **Pyramid Principle**: structuring persuasive communication as decision → supporting arguments → supporting data, top-down.
- **SCQA / SCR framework**: Situation, Complication, Question, Answer — a four-part structure for pitching a recommendation.
- **Policy**: a centrally defined requirement (regulatory, access-control, querying) that contracts must reference/enforce, ideally at a higher abstraction level than the underlying tech so governance teams don't need to become engineers.
- **Context evolution / semantic drift**: changes to the *meaning* of a field (e.g., `promo_code` only valid if `promo_expiry_date` is in the future) that schema checks alone cannot catch — requires explicit semantic/business-logic detection.

## Mental Models
- Think of GitHub not as a DevOps tool but as a change management platform — branching/merging = parallelizing change, PRs/code review = human-in-the-loop change review, subscribers/changelogs = change notification systems. Data contracts aim to be the equivalent for data.
- Use "lateral thinking with withered technology" (Gunpei Yokoi) to look outside software engineering (to supply chains, which have done change management for centuries) rather than assuming software engineering is the most mature source of best practices.
- Treat standards debates (Protobuf, Avro, Iceberg, ODCS, etc.) as largely irrelevant distractions — winners emerge naturally from solved use cases, not from choosing the spec first; focus energy on adoption, ownership, visibility, and change management instead.
- Apply "going deeper" (iteratively re-stating the problem) until you reach a level where you actually get traction — stopping too early (e.g., at "we need automatic enforcement") may still fail once implementation widens beyond the data team.

## Anti-patterns
- **Treating an airline project as an airplane project**: most failed data catalog and data contract rollouts fail not because the tech doesn't work, but because adoption incentives/friction were never addressed — "if we build it, they will come" never happens.
- **Funneling all change communication through an overburdened central data platform team**: recreates the same bottleneck as a centralized DevOps team approving every PR; works only for small companies/low change volume, not at scale.
- **Chasing a "standard" contract spec as the primary problem**: distracts from the actually hard problems (adoption, ownership, visibility, change management) that determine success regardless of spec choice.
- **Relying on schema checks alone to catch business-logic/semantic changes**: misses cases like a promo code's validity rule changing without any type or column change.
- **Skipping policy centralization**: "We have lots of policies. The problem is enforcement" (quote from a bank's distinguished engineer) — policies without a tie to an enforcement mechanism in the data contract system are not effective.

## Code Examples
*(omitted — this chapter has no code/config examples; it is a strategy/change-management chapter built on consulting frameworks and narrative analogies)*

## Reference Tables
| | Airplane project | Airline project |
|---|---|---|
| Focus | Technical viability | Business/adoption viability |
| Example | Most ETL/ELT tools, cost-optimization tools | Data catalogs, data contracts |
| Risk if misapplied | N/A (works as-is) | Built but unused — "why isn't anyone flying with us?" |

| Supply chain role | Data equivalent |
|---|---|
| Farmer/miner | Data producer |
| Transport truck | Data pipeline |
| Manufacturer | Data engineer |
| Distributor/warehouse | Data warehouse |
| Retailer (Tableau, Power BI, SageMaker) | BI/analytics tooling |
| Consumer | Data scientist/analyst/product manager |

| Hypothesis-led step | Change-management application |
|---|---|
| Define the problem | Iteratively "go deeper" to reach the real root cause |
| Structure | Issue tree breaking the problem into subcomponents |
| Prioritize | Impact × ease-of-implementation matrix |
| Work plan / analyses | Scope existing tooling, map teams/owners along the supply chain |
| Synthesize | Pyramid Principle (decision → arguments → data) |
| Recommend | SCQA pitch structure |

## Worked Example
**The authors' own "going deeper" chain (10 steps)**: (1) data quality is hard despite specialization → (2) governance exists but enforcement is limited → (3) enforcement is hard because breaking changes come from upstream teams data doesn't own → (4) teaching upstream teams best practices fails due to inertia and misaligned incentives → (5) need automatic enforcement (data contracts) → (6) inertia persists without clear "what's in it for them" → (7) CI/CD-embedded, programmatic enforcement fits the existing developer workflow → (8) scope enforcement to only the most important (revenue-driving/risk-mitigating) assets to make the case legible → (9) the real ask of upstream teams is fundamentally a change management process → (10) final problem statement: "How do we create the most effective change management workflow with respect to changes that impact the data that's important to the organization?" The authors note they once stopped at step 5, assuming automatic enforcement alone would solve the issue, but found this insufficient once they tried to scale beyond data teams — illustrating that "how deep is deep enough" is answered empirically by where you lose traction.

**SCQA pitch example (change management business case)**: Situation — breaking changes over 3 months cost ~8,000 person-hours and exposed a $10M regulatory fine risk. Complication — federated team structure caused a 45% increase in delays on cross-team projects, traced to change management gaps. Question — can this be fixed without reversing the federated-teams decision? Answer — yes, via data contracts, which would have prevented the recent SEV-0 that risked the $10M fine within a 24-hour resolution window.

## Key Takeaways
1. Leadership invests in solving strategic problems, not in technology for its own sake — frame data contracts as a change management solution, not a tooling upgrade.
2. Treat data contract rollout as an airline project: explicitly design for adoption incentives and minimize friction, not just technical capability.
3. Plan for both forward-looking (new/changing data) and backward-looking (legacy, uncontracted data) problems — most initial implementations only solve the forward case.
4. Don't centralize all change communication and contract gatekeeping in an overloaded data platform team at scale — this just recreates a single-approver bottleneck.
5. Don't get distracted chasing a "standard" spec — adoption, ownership, visibility, and change management determine success, regardless of spec choice.
6. Use the hypothesis-led consulting approach (define → structure → prioritize → plan → analyze → synthesize → recommend) to build a credible internal pitch for data contracts.
7. Keep "going deeper" on your problem statement until you find the level where you actually get traction — don't settle for the first appealing-sounding root cause.

## Connects To
- **Ch 3**: Explicitly ties back to Dunbar's Number / Conway's Law as why larger, more federated organizations need change management most.
- **Ch 9**: Builds directly on the perception-gap and incentive-alignment themes from the shift left chapter.
- **Ch 11**: Explicitly sets up the next chapter's focus on getting the first concrete win with data contracts.
- **DevOps/supply chain management**: Reframes GitHub and physical supply chains as change-management precedents data contracts should learn from.
