# Chapter 14: Inputs and Outputs / Tools and Techniques (Reference Sections)

## Core Idea
Guide Sections 4 and 5 are alphabetical reference glossaries — ~90 input/output artifacts and ~150 tools/techniques, explicitly "illustrative but not comprehensive" and not required for any given project. This chapter curates the highest decision-value material; exhaustive term-by-term definitions live in `glossary.md`, and technique patterns live in `patterns.md`/`cheatsheet.md`.

## Frameworks Introduced

### Earned Value Management (EVM) — the full formula set
EVM integrates scope+cost+schedule baselines into one Performance Measurement Baseline (PMB), tracked via three core measures: **PV** (Planned Value — budget for scheduled work), **EV** (Earned Value — budget value of work actually completed), **AC** (Actual Cost — real cost incurred). Everything else derives from these three. See cheatsheet.md for the full formula table (CV, SV, CPI, SPI, EAC, ETC, VAC, TCPI).
- When to use: any time you need an objective, comparable measure of "are we on budget/schedule" instead of a subjective status color.
- Adaptive variant: PV/EV are expressed in story points instead of currency — PV = story points planned by a date, EV = story points done by that date, AC = actual cost/hours for the iteration.

### Critical Path Method (CPM) vs. Critical Chain Project Management (CCPM)
CPM: forward pass (ES/EF) + backward pass (LS/LF) through the network determines total float per activity; the sequence of zero-float activities is the critical path.
CCPM: uses the same resource-loaded critical path logic, but replaces task-level buffers with ONE project buffer, tracked via the **Buffer Protection Index (BPI)** — the ratio of remaining buffer to remaining critical-chain length — which protects the due date directly, unlike SPI which just measures baseline deviation.
- When to use CPM: any predictive schedule needing float/flexibility analysis.
- When to use CCPM: resource-constrained environments where protecting the due date matters more than tracking baseline adherence.

### Decision Tree Analysis with Expected Monetary Value (EMV)
Model alternative decisions (decision nodes) against uncertain outcomes (chance nodes, each with a probability and payoff); compute EMV = Σ(probability × payoff) − cost for each branch; choose the branch with the best EMV (see cheatsheet.md for the worked $120M-vs-$50M plant example).
- When to use: any major capital/strategic decision with quantifiable probabilities and payoffs — makes risk-adjusted comparison explicit instead of gut-feel.

### Risk Response Strategies (full set, both polarities)
Threats: Escalate, Avoid, Transfer, Mitigate, Accept. Opportunities: Escalate, Exploit, Share, Enhance, Accept. Overall project risk: Avoid, Exploit, Transfer/Share, Mitigate/Enhance, Accept. See cheatsheet.md for the full decision table with examples.

### Motivation & Leadership Theories (named models, each with a distinct lens)
- **Herzberg (Hygiene vs. Motivational factors)**: hygiene factors (pay, policies) prevent dissatisfaction but don't create satisfaction; motivational factors (achievement, growth) do.
- **Pink (Autonomy, Mastery, Purpose)**: intrinsic motivators that outlast extrinsic rewards once pay is "fair."
- **McClelland (Theory of Needs)**: people driven by Achievement, Power, or Affiliation in varying mixes.
- **McGregor/Ouchi (Theory X/Y/Z)**: X = income-only motivation, hands-on top-down management; Y = intrinsically motivated, coaching management; Z = self-realization/meaning, long-term "job for life" culture.
- **Tuckman Ladder**: team development stages — Forming → Storming → Norming → Performing → Adjourning (can stall or regress at any stage).
- **Emotional Intelligence (4 quadrants)**: Self-Awareness, Self-Management, Social Awareness, Social Skills.
- When to use: diagnosing a team-motivation or team-development problem — pick the model whose lens matches the symptom (e.g., stuck-in-storming → Tuckman; "paid fairly but still disengaged" → Pink).

### Stakeholder Mapping Models (pick by complexity)
Power/Interest grid (simple projects) → Stakeholder Cube (3D refinement) → Salience Model (power/urgency/legitimacy, or power/urgency/proximity — for large complex stakeholder networks) → Directions of Influence (upward/downward/outward/sideward). Escalate model sophistication with stakeholder-community complexity, not by default.

### Schedule Compression (two techniques, different risk profiles)
**Crashing**: add resources to shorten critical-path activities — costs money, may not always work. **Fast tracking**: run sequential activities in parallel — costs risk/rework, not necessarily money. Both only work on critical-path activities.

### Resource Optimization (two techniques, different trade-offs)
**Resource leveling**: adjusts dates to balance resource demand/supply — can change the critical path (and therefore the end date). **Resource smoothing**: adjusts within existing float only — never changes the critical path or end date, but may not fully resolve over-allocation.

## Key Concepts (selected, high-frequency artifacts — full list in glossary.md)
- **Project charter vs. project scope statement**: charter = high-level authorization (purpose, objectives, milestones, risk, stakeholder list); scope statement = detailed description (scope description, deliverables, acceptance criteria, exclusions). Charter comes first, authorizes the PM; scope statement elaborates it.
- **Risk Breakdown Structure (RBS)**: hierarchical risk-source categorization (Technical/Management/Commercial/External at Level 1) — distinct from the WBS, which organizes deliverables, not risk sources.
- **Contingency reserve vs. management reserve** (recap from ch09/ch12): contingency = known-unknowns, inside baseline; management = unknown-unknowns, outside baseline, senior-leadership-released.
- **RACI matrix**: Responsible/Accountable/Consulted/Informed — a responsibility assignment matrix ensuring exactly one person is Accountable per task.
- **VRIO framework** (resource-based view): Value, Rarity, Imitability, Organization — evaluates whether a resource/capability is a genuine source of competitive advantage.

## Mental Models
- Every ITTO figure in the Guide is a *sample*, not a checklist — "Etc." appears at the end of every list deliberately.
- When picking a prioritization method (MoSCoW, 100-point, cost of delay, Kano), match it to decision type: MoSCoW for binary must/should/could/won't calls; cost of delay when time-value trade-offs matter; Kano when customer delight vs. basic expectation distinctions matter.
- PESTLE / TECOP / VUCA are prompt lists for identifying *sources of overall project risk* — use them to check you haven't missed a risk category, not as a report template.

## Anti-patterns
- **Using crashing on a non-critical-path activity**: wastes money with zero schedule benefit — crashing only works on the critical path.
- **Confusing resource leveling with resource smoothing**: leveling can push out your end date; smoothing cannot — know which trade-off you're accepting.
- **Treating TCPI > 1 as acceptable without judgment**: a TCPI meaningfully above 1.0 means the remaining work must be done more efficiently than everything accomplished so far — flag this as a feasibility risk, not just a number.

## Key Takeaways
1. Learn the three EVM primitives (PV/EV/AC) cold — every other EVM formula (CV, SV, CPI, SPI, EAC, ETC, VAC, TCPI) is derived from them (full table in cheatsheet.md).
2. Use decision trees + EMV for major uncertain decisions — it forces explicit probabilities instead of implicit gut-feel.
3. Match risk response strategy to risk polarity and whether it's individual or overall project risk — 5 strategies each, not interchangeable.
4. Pick a motivation/leadership model by the specific symptom you're diagnosing, not out of habit.
5. Scale stakeholder-mapping sophistication to stakeholder-community complexity.
6. Crashing costs money; fast tracking costs risk — pick deliberately, and only ever on the critical path.

## Connects To
- **Ch 6 (Governance)**: Monitor and Control Project Performance draws directly on EVM, leading/lagging indicators, and work performance reporting defined here.
- **Ch 8 (Schedule)**: CPM, CCPM, schedule compression, and resource optimization are the mechanical techniques behind that domain's processes.
- **Ch 9 (Finance)**: EVM is the primary toolkit for Monitor and Control Finances.
- **Ch 10 (Stakeholders)**: stakeholder mapping models elaborate that domain's engagement-planning process.
- **Ch 12 (Risk)**: the five-strategy response sets elaborate Plan Risk Responses.
- **glossary.md, patterns.md, cheatsheet.md**: carry the full term list and formula/decision tables this chapter curates from.
