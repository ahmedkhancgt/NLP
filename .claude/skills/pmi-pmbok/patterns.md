# Patterns & Techniques — PMBOK Guide, 8th Edition (as captured)

## Development Approach Selection
**When to use**: Starting any new project or scoping a new deliverable within one.
**How**: Score the deliverable/project/organization against the Section 4.3 factor checklist (requirements certainty, degree of innovation, ease of change, safety/regulatory needs, feedback value, schedule constraint, financing uncertainty, org structure/culture, team size) — see ch04's Reference Table. Predictive-leaning factors push toward waterfall; adaptive-leaning factors push toward agile/iterative; mixed signals point to hybrid.
**Trade-offs**: Predictive gives early cost/schedule certainty but is expensive to change later. Adaptive absorbs uncertainty well but sacrifices up-front predictability. Hybrid requires explicitly naming which pattern (Figures 4-8–4-11) and level (1/2/3) applies, or governance gets confused about which rules apply where.

## The Inverted Triangle (Constraint Trade-off)
**When to use**: Chartering any adaptive or hybrid project, to pre-negotiate how unexpected change will be absorbed.
**How**: Explicitly choose one of two configurations — (a) fix budget + schedule, let scope flex; or (b) fix scope, let budget/schedule flex. Document which one in the charter/plan.
**Trade-offs**: Fixing scope under cost/schedule pressure risks quality erosion or burnout; fixing budget/schedule under scope pressure risks descoping value the stakeholders actually wanted. Pick deliberately, not by default.

## Value Breakdown Structure (VBS) for Prioritization
**When to use**: Any project with multiple deliverables competing for limited time/resources.
**How**: List top-level deliverables; assign each an explicit value (dollar figure, a countable outcome, or % of total project value). Decompose into subdeliverables inheriting a value share. Use this to calculate "drag cost" — what a delay to this item actually costs in value, not just days.
**Trade-offs**: Requires stakeholders to agree on value estimates up front, which can be contentious — but the alternative (prioritizing by gut feel or by "biggest WBS box") routinely misallocates effort toward low-value, high-visibility work.

## Leading/Lagging Indicator Pairing
**When to use**: Designing any project dashboard or governance reporting cadence.
**How**: For every lagging indicator (schedule variance, cost variance, deliverables completed), pair it with at least one leading indicator that would have predicted it (backlog growth rate, stakeholder engagement/availability, undefined success criteria, risk register staleness).
**Trade-offs**: Leading indicators are often harder to quantify and more subjective — resist dropping them just because they're less crisp than lagging numbers; they're what let you act before damage is done.

## SMART-Checking Every Metric
**When to use**: Before adopting any new governance metric or project KPI.
**How**: Run it through Specific / Measurable / Achievable / Realistic / Time-bound. If it fails any letter, redefine it before tracking it.
**Trade-offs**: Takes a small amount of up-front discipline; the payoff is avoiding the Measurement Pitfalls (vanity metrics, demoralization, misuse) documented in ch06.

## Quality Assurance vs. Quality Control Split
**When to use**: Structuring any quality management plan.
**How**: Build two parallel tracks — QA (process audits, compliance checks, failure-analysis reviews aimed at the *process*) and QC (testing, inspection, defect tracking aimed at the *deliverable*). Don't merge them into one undifferentiated "quality" bucket.
**Trade-offs**: Running both tracks costs more coordination overhead than a single informal "check quality" step, but catches process-level risk (will future work be trustworthy?) that pure deliverable testing misses.

## Tacit → Explicit Knowledge Conversion
**When to use**: Any project with specialist/expert knowledge concentrated in a few people, or ongoing lessons-learned capture.
**How**: Use retrospectives, after-action reviews, storytelling, mentoring relationships, and face-to-face interaction (or AI interview bots) specifically to surface tacit knowledge, then document it as explicit knowledge (registers, databases, procedures).
**Trade-offs**: Codified knowledge loses some context and nuance versus the original tacit form — pair written artifacts with ongoing mentoring rather than assuming documentation alone transfers expertise.

## Structured vs. Self-Governance Model Choice
**When to use**: Setting up project governance at kickoff.
**How**: Use structured governance (sponsor + PMO + governance board + PM with cross-domain oversight) for predictive/large/regulated projects. Use self-governance (distributed accountability across the team) for adaptive projects — but only with clear, measurable common objectives, leading indicators, and feedback mechanisms in place to prevent fragmented decision-making.
**Trade-offs**: Structured governance adds process overhead but reduces fragmentation risk; self-governance moves faster but needs strong team discipline to avoid conflicting decisions without a clear owner.

## Predictive vs. Adaptive Change Management
**When to use**: Designing how scope/plan changes get assessed and approved.
**How**: Predictive — route every change through a Change Control Board with formal impact analysis across Scope/Finance/Schedule/Resources/Stakeholders/Risk, ending in Approved/Rejected/Deferred/More-Information-Needed. Adaptive — treat the backlog itself as the change mechanism: a new item gets added and prioritized (or not); lower priority = effectively deferred; removal = effectively rejected.
**Trade-offs**: Formal CCB process adds traceability and auditability (valuable for regulated/contractual work) at the cost of speed; backlog-driven change is fast and lightweight but leaves a thinner audit trail — mismatch the pattern to the project's compliance needs at your own risk.

## Earned Value Management as a Status Check
**When to use**: Any project wanting an objective (not subjective-color) answer to "are we on budget and schedule."
**How**: Track PV/EV/AC per work package. Compute CV (EV−AC) and SV (EV−PV) for current status; compute CPI (EV/AC) and SPI (EV/PV) for efficiency ratios; use EAC/ETC/TCPI to forecast completion (full formulas in cheatsheet.md).
**Trade-offs**: Requires disciplined, consistent PV baselining up front — EVM is only as good as the baseline it's measured against; garbage-in baseline produces confidently-wrong status.

## Critical Path vs. Critical Chain Scheduling
**When to use**: Any predictive/hybrid schedule needing systematic float analysis.
**How**: Use CPM (forward/backward pass, zero-float = critical path) as the default. Switch to CCPM when resources are genuinely constrained and protecting the due date matters more than tracking baseline variance — replace task-level buffers with one project buffer and track the Buffer Protection Index.
**Trade-offs**: CCPM requires more organizational buy-in to adopt (it changes how buffers are owned/perceived) but directly protects the due date; CPM is simpler and universally understood but can hide risk inside many small task-level buffers.

## Schedule Compression: Crash vs. Fast-Track
**When to use**: Schedule needs to shrink without cutting scope, and only critical-path activities qualify.
**How**: Crash (add resources/pay for expedited delivery) when the constraint is capacity and budget can absorb it. Fast-track (run sequential activities in parallel) when the constraint is calendar time and some rework risk is acceptable.
**Trade-offs**: Crashing costs money with limited risk; fast-tracking costs risk (and potential rework cost) with limited direct spend — never apply either off the critical path, since it buys nothing.

## Resource Optimization: Level vs. Smooth
**When to use**: Resource demand exceeds supply at some point in the schedule.
**How**: Level (adjust start/finish dates to balance demand/supply) when the end date can move. Smooth (adjust only within existing float) when the end date and critical path are fixed and cannot move.
**Trade-offs**: Leveling can delay the project; smoothing protects the date but may not fully resolve over-allocation — know which constraint (date or resource conflict) you're actually protecting before choosing.

## Decision Tree + Expected Monetary Value (EMV)
**When to use**: A major decision (capital investment, build-vs-upgrade, vendor selection) with quantifiable probabilities and payoffs.
**How**: Draw decision nodes (choices) and chance nodes (uncertain outcomes with probabilities); compute EMV = Σ(probability × payoff) per branch minus upfront cost; pick the highest-EMV branch (worked example in ch14/cheatsheet.md).
**Trade-offs**: Forces explicit probability estimates, which can be uncomfortable/contested — but the alternative (implicit gut-feel comparison) hides the same assumptions without surfacing them for debate.

## Risk Response Strategy Selection
**When to use**: Any identified individual risk (threat or opportunity) or overall project risk needing a planned response.
**How**: For threats, pick from Escalate/Avoid/Transfer/Mitigate/Accept. For opportunities, pick from Escalate/Exploit/Share/Enhance/Accept. For overall project risk, pick from Avoid/Exploit/Transfer-Share/Mitigate-Enhance/Accept. Match strategy to priority and whether probability, impact, or both need to change (full table in cheatsheet.md).
**Trade-offs**: Escalating too much under-uses the project team's own authority; escalating too little means risks outside the PM's authority go unmanaged — calibrate against the project's actual risk thresholds (ch12).

## PMO Model Selection
**When to use**: Designing or evaluating a PMO.
**How**: Don't pick one pure archetype (directive/supportive/agile) — assess what the PMO's actual "customers" (executives, PMs, teams) value, and blend characteristics from multiple models to fit that specific organizational context.
**Trade-offs**: A hybrid PMO is harder to describe crisply in a one-line mandate, but a "pure" model chased for its own sake is explicitly flagged as a path to decreased value perception over time.

## Procurement Decision Chain
**When to use**: Any work being considered for outsourcing.
**How**: (1) Make-or-buy analysis (ROI/IRR/NPV/payback) → (2) Procurement strategy (delivery method + contract type) → (3) Source selection method (least cost / qualifications-only / quality-based / quality-and-cost-based / single-source / fixed-budget, matched to complexity/risk) → (4) Weighted source selection criteria.
**Trade-offs**: Skipping straight to a contract type without working through strategy and selection-method steps skips real risk-allocation decisions — each step constrains the next.

## AI Task Classification Before Use
**When to use**: Before assigning any task to an AI tool on a project.
**How**: Classify as Automation (low complexity, minimal review), Assistance (iterative, never trust first output), or Augmentation (strategic, use AI as a brainstorming partner through multiple iterations) — then apply the review rigor that tier demands. Run the seven-point AI ethics check (bias/privacy/accountability/reliability/safety/transparency/copyright/sustainability — ch15) before scaling usage.
**Trade-offs**: Treating an Augmentation-tier output with Automation-tier trust is the single biggest AI misuse risk named in the Guide — match scrutiny to tier every time.
