# Chapter 11: Creating Your First Wins with Data Contracts

## Core Idea
Winning data contract adoption requires a three-phase, people-first journey: (1) rigorously select a single high-value "steel thread" data product for your pilot using a weighted decision matrix, (2) build a leadership-ready business case through stakeholder mapping and a lean pitch deck, and (3) scale beyond the pilot using Kotter's eight-step organizational change framework.

## Frameworks Introduced
- **Data Product (minimum viable definition)**: "A packaged workflow and/or process that consists of a data asset, a means to operationalize this data, and a method of serving the data to a consumer... reusable, domain-owned, and tied to a verifiable high-value outcome."
  - When to use: filtering hundreds/thousands of data assets down to a manageable pilot consideration set.
  - How: apply as a first-pass filter; don't over-engineer the definition — iterative progress beats a perfect taxonomy.
- **Steel Threads** (borrowed from suspension-bridge engineering / software architecture): a data product qualifies if its failure would cause immediate, measurable disruption to business operations, decisions, or customer experience.
  - When to use: narrowing Tier-1 candidates further, after the data-product filter.
  - How: ask whether failure of this asset would be immediately and measurably disruptive — if yes, it's a steel thread candidate.
- **Weighted Decision Matrix**: score candidate steel threads (1–5) against weighted criteria (1–10) — business criticality, producer/consumer engagement, low data volatility, existing DQ issues, complexity of dependencies, ease of implementation, organizational visibility — then sum weight×score.
  - When to use: final selection among several compelling steel-thread candidates.
  - How: define criteria, assign weights reflecting your org's priorities, score each candidate, multiply and sum; highest total wins.
- **Four-Step Business Case Process**: (1) gather information and plan to engage, (2) map stakeholders onto the data lifecycle (value stream mapping), (3) determine stakeholder influence/interest and risk (2×2 matrices), (4) gain leadership alignment/buy-in.
  - When to use: turning a selected steel thread into an approved pilot.
  - How: three of the four steps are about people, not technology — treat stakeholder work as the main task.
- **Influence/Interest 2×2 Matrix**: classify stakeholders as champions (high/high), watch-for-blockers (high influence/low interest), amplify-voice (low influence/high interest), or baseline-courtesy (low/low).
  - When to use: managing stakeholder engagement intensity appropriately rather than uniformly.
  - How: plot each stakeholder; tailor communication cadence and depth per quadrant.
- **Probability/Impact 2×2 Matrix**: classify risks as immediate-attention (high/high), small-safeguards (high probability/low impact), contingency-plan (low probability/high impact), or watch-list (low/low).
  - When to use: within the high-influence/high-interest stakeholder group, to triage risks before presenting the pilot proposal.
- **Kotter's Eight-Step Change Model**: (1) establish urgency, (2) create guiding coalition, (3) develop vision/strategy, (4) communicate the vision, (5) empower broad-based action, (6) generate short-term wins, (7) consolidate gains/produce more change, (8) anchor new approaches in culture.
  - When to use: scaling beyond the first pilot to org-wide shift left adoption.
  - How: treat as adaptable guidance, not a rigid checklist — tailor each step to your organization's specific blockers and culture.

## Key Concepts
- **Tracer bullets** (Pragmatic Programmer): using fast, real-world feedback to iteratively adjust aim rather than assuming you'll hit the target on the first try — applied to picking and refining a pilot.
- **Value stream mapping**: visualizing the end-to-end steps required to deliver a data product to a consumer, to connect technical work to business meaning and identify domain owners.
- **Guiding coalition**: a strategically diverse group (senior executives for budget/policy authority + influential mid-level leaders who understand frontline realities), not just technical experts.
- **Removal of complacency**: Kotter's framing of urgency — the organization must recognize the status quo as truly untenable, which the authors warn should not be artificially attached to another large, competing project (e.g., a big migration) since it will get deprioritized when that project runs over budget.
- **Build-measure-learn mindset**: the bottom-up discovery pattern (borrowed from DevOps's own 2008 shift-left history) needed once adoption moves past top-down pilot-driven direction.
- **Target operating model (people, process, technology)**: the framework referenced for anchoring shift left/data contracts as permanent organizational practice (full treatment out of scope for this book).

## Mental Models
- Think of a pilot candidate selection as iterative filtering: data product definition → steel thread qualification → weighted decision matrix — each stage narrows the field with increasing rigor.
- Use "leaders invest in solutions, not technology" (carried over from Ch. 10) to frame every pitch element around business outcomes, never pure technical capability.
- Treat urgency as something you cultivate and manage, not something that exists passively — and recognize that in large orgs, "urgency cannot be everywhere or felt by everyone all at once," so expect uneven adoption.
- View change as experiential, not just understandable — qualitative signals (fewer incidents, calmer postmortems, unprompted producer-consumer conversations about downstream impact) indicate real cultural traction better than any single formal metric.

## Anti-patterns
- **Perfecting a universal "data product" definition before starting**: wastes time; a minimum-viable filtering definition is sufficient to begin iterating.
- **Attaching data contract urgency to another large competing initiative (e.g., a big migration)**: the migration's inevitable overrun gets prioritized and the data contract work is deprioritized — build data contracts as a gradual buildout, not riding on someone else's megaproject's momentum.
- **Building a coalition of technical experts only**: misses the budgetary/policy authority and frontline-realities perspective needed to drive org-wide change.
- **Jargon-heavy, team-specific messaging**: e.g., "poor data quality within the logistics cost ledger data asset is resulting in high error rates" — buries urgency and ownership; concise, strategically framed messaging ("reduce waste... to counteract the rising cost of steel") resonates far better across the org.
- **Treating territorial leadership resistance as purely irrational**: often rooted in fear that data contracts will make a team's existing work seem obsolete — address by connecting their existing work to the shift left vision and giving public credit, not just pushing through.
- **Expecting uniform, simultaneous org-wide adoption**: natural for some teams to lead and others to lag; visibility should take precedence over uniformity.

## Code Examples
*(omitted — this chapter has no code/config examples; it is entirely a change-management/stakeholder-strategy chapter)*

## Reference Tables
| Weighted decision matrix criterion | Why it matters |
|---|---|
| Business criticality | Drives buy-in urgency |
| Producer/consumer engagement | More feedback available for iteration |
| Low data volatility | Lower pilot risk (but volatility is also the reason contracts help) |
| Existing data quality issues | Could be a hazard or exactly the problem to prove value against |
| Complexity of dependencies | Fewer/simpler dependencies = lower pilot risk |
| Ease of implementation | Engineering effort required; common source of SWE pushback |
| Organizational visibility | More teams relying on it = clearer demonstrated benefit |

| Kotter step | Common blocker addressed |
|---|---|
| 1. Urgency | Organizational complacency |
| 2. Guiding coalition | Lack of cross-functional authority |
| 3. Vision/strategy | Misaligned or short-term-only incentives |
| 4. Communicate vision | Jargon, team-siloed messaging |
| 5. Empower action | Unsupported tech, skeptical SWEs, poor onboarding, territorial leadership |
| 6. Short-term wins | Loss of momentum / credibility gaps |
| 7. Consolidate gains | Overreliance on original pilot team |
| 8. Anchor in culture | Reversion once leadership attention moves on |

## Worked Example
**Automotive logistics "steel thread" (running example)**: a weekly dashboard integrating production, inventory, and supply-chain data feeds a report suppliers rely on to plan their own schedules. If the data becomes unreliable (schema change, ingestion issue) and no one notices, suppliers make misinformed adjustments and production disruptions cascade — with manufacturing stoppages historically costing ~$22,000/minute in profit in the auto industry. Applying the weighted decision matrix, the "Logistics Cost Ledger" data product scores highest (weighted total 159) on criteria including business criticality (weight 10, score 4→40) and producer/consumer engagement (weight 7, score 4→28), making it the selected pilot. Communication example: a jargon-heavy, team-blaming pitch ("poor data quality within the logistics cost ledger data asset is resulting in high error rates...") is replaced with a concise, leadership-aligned message ("It's paramount that we reduce waste within our supply chains to counteract the rising cost of steel") that implies urgency and ties to a company-wide strategic outcome.

**Mark's "removal of complacency" story**: a batch replication pipeline kept failing, and the data team had normalized working around it (providing results anyway with stale/incomplete data), so the business never registered true urgency. Once leadership approved simply withholding deliverables unless data was adequately fresh and complete, the business felt the pain directly within a month, and a software engineer was finally allocated to fix the pipeline — illustrating that eliminating workaround-driven complacency, not just raising awareness, is what creates real urgency.

## Key Takeaways
1. Select your first pilot through a disciplined three-stage filter: data product definition → steel thread qualification → weighted decision matrix — don't pick based on convenience or whoever asks first.
2. Three of the four steps in building your business case are about people (stakeholder mapping, influence/interest, risk) — treat relationship work as the main deliverable, not a side task.
3. Pitch decks for leadership should lead with the problem/business alignment, position outcomes as solutions, and make ROI explicit and comparable to competing proposals.
4. Don't tie data contract urgency to another large, competing initiative's momentum — it will get deprioritized when that initiative overruns.
5. Build a genuinely cross-functional guiding coalition (budget authority + frontline insight), not a technical-experts-only team.
6. Messaging must be concise and tied to a company-wide strategic outcome, not jargon-heavy and siloed to one team's pain.
7. Expect and plan for common SWE-adoption blockers: unsupported legacy tech, skeptical engineers (who need incentive alignment, not persuasion about the solution itself), poor onboarding experience, and territorial leadership.
8. Treat organizational change as experiential — track qualitative signals (fewer incidents, calmer postmortems, unprompted cross-team conversations about downstream impact) as evidence adoption is truly taking hold, not just formal milestones.

## Connects To
- **Ch 3**: Explicitly ties enterprise scale and Dunbar's Number/Conway's Law to why large organizations most need this structured adoption approach.
- **Ch 8**: Revisits Convoy, Glassdoor, and Adevinta Spain as concrete examples of what "success" looked like qualitatively during adoption.
- **Ch 10**: Builds directly on the change-management and strategy-development frameworks (hypothesis-led approach) introduced there.
- **Ch 12**: Explicitly sets up the next chapter's focus on quantitative metrics for measuring data contract impact.
