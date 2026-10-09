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
