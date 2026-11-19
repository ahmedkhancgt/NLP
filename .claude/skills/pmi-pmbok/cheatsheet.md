# Cheatsheet — PMBOK Guide, 8th Edition (as captured)

## Decision Rules

- **When requirements are clear and stable early → use predictive.** When they're unclear/volatile → use adaptive. When some deliverables are stable and others aren't → use hybrid, and name which Section 4.2.3 pattern applies.
- **When budget+schedule are organizationally fixed → let scope be the flex variable.** When scope is fixed (e.g., contractual/regulatory) → let budget/schedule absorb change. Never leave this unstated going into execution.
- **When a metric can't be made SMART → don't track it yet.** Fix the definition first; tracking a non-SMART metric invites the Measurement Pitfalls below.
- **When judging a project "success/failure" → always report outcome success and process success separately.** A process failure (Sydney Opera House: 14x over budget) can still be an outcome success; a process success (Montreal overpass) can still be an outcome failure.
- **When defining scope → always ask the nonfunctional/quality question explicitly.** "Quality" is part of scope, not an afterthought; failing to ask invites unstated requirements.
- **When choosing insource vs. outsource → check which side of Table 2-3 dominates for this deliverable**, not organizational habit.
- **When a change request arrives → predictive: route to CCB with full impact analysis. Adaptive: add/prioritize/deprioritize in the backlog.** Don't apply CCB formality to a backlog-driven team or vice versa.

## Decision Tree — Which Governance Model?

```
Is the project environment adaptive, small-team, and comfortable with uncertainty?
├── Yes → Self-governance
│         ├── Did you establish clear, measurable common objectives + leading
│         │   indicators + feedback mechanisms?
│         │   ├── Yes → proceed
│         │   └── No → fix this first; fragmented decision-making is the #1 self-governance risk
└── No (predictive/large/regulated/multi-org) → Structured governance
          ├── Sponsor + PMO + governance board + PM with cross-domain oversight
          └── Add escalation + investment control if hierarchical org / formal funding steward needed
```

## Decision Tree — Hybrid Pattern Selection

```
Does the project have deliverables with fundamentally different certainty profiles?
├── One deliverable stable, built after another uncertain one settles
│      → Adaptive-development-then-predictive-rollout (Fig 4-8)
├── Multiple parallel streams, each with its own best-fit approach
│      → Simultaneous hybrid, by stream (Fig 4-9)
├── Mostly predictive project, with one small uncertain component
│      → Largely-predictive-with-adaptive-component (Fig 4-10)
└── Mostly adaptive project, with one small fixed/regulated component
       → Largely-adaptive-with-predictive-component (Fig 4-11)
```

## Trade-off Matrix — Insourcing vs. Outsourcing

| Dimension | Insource/Make wins when... | Outsource/Buy wins when... |
|---|---|---|
| Expertise | Already have it internally | Missing internally |
| Cost | Heavy new innovation required | Goods/services are commoditized |
| Strategic fit | Tight integration with core value prop needed | Freeing capacity for core strengths matters more |
| Control | Oversight/control of deliverable is critical | Transferring delivery risk is more valuable |

## Trade-off Matrix — Development Approach by Factor

| Factor | Predictive | Adaptive |
|---|---|---|
| Requirements certainty | High | Low |
| Innovation need | Low | High |
| Ease of change | Hard/costly | Easy |
| Safety/regulatory load | High | Low |
| Feedback value | Low | High |
| Schedule type | Fixed end date | Early partial delivery valuable |
| Org structure | Fixed/hierarchical | Network-oriented |
| Team size | Larger | Smaller (often 3–9) |

## Thresholds & Defaults

- **Adaptive iteration length**: commonly 1–4 weeks per iteration/sprint.
- **Adaptive team size**: commonly 3–9 members per the cited frameworks.
- **Hybrid levels** (PMI Disciplined Agile): Level 1 = predictive-dominant; Level 2 = balanced; Level 3 = adaptive-dominant.
- **Phase gate decision set**: exactly six outcomes — continue / continue with modification / end / remain in phase / repeat phase / park temporarily. Never fewer than this when designing a stage-gate review.
- **Three mandatory governance components**: target metrics + signaling mechanism + feedback mechanism — missing any one is incomplete governance.

## Tells & Smells

- **"We're hybrid"** with no named pattern or level → probably means nobody has actually decided which constraints are fixed where. Press for specifics.
- **A dashboard full of lagging indicators only** → high risk of being blindsided; ask "what would have warned us about this earlier?"
- **A metric nobody can tie to a decision** → likely a vanity metric; cut it or redefine it.
- **Team hitting every target but morale dropping** → check for demoralization from unrealistic stretch goals, not a performance problem.
- **"Behind schedule AND over budget" being treated as one problem** → classic correlation/causation trap; look for a shared root cause (poor estimating, weak risk/change management) instead of assuming one caused the other.
- **Scope defined with no mention of quality thresholds** → nonfunctional requirements are being skipped; this is a scope gap, not a later "quality phase" fix.
- **Change requests handled informally on a predictive/regulated project** → missing CCB discipline; audit trail risk.
- **Formal change-request paperwork on a fast adaptive team** → mismatched process; should be backlog-driven instead.

## Quick Reference — Six Principles → Mindset Dimension

| Principle | Mindset Dimension |
|---|---|
| Adopt a Holistic View | Proactive |
| Embed Quality Into Processes and Deliverables | Proactive |
| Be an Accountable Leader | Ownership |
| Build an Empowered Culture | Ownership |
| Focus on Value | Value-Driven |
| Integrate Sustainability Within All Project Areas | Value-Driven |

## Quick Reference — Sustainability Pyramid (most → least desirable)

1. Avoid negative outcomes altogether
2. Minimize negative outcomes generated
3. Restore impacts of negative outcomes
4. Compensate/offset for negative outcomes generated

## Formula Reference — Earned Value Management

| Abbr. | Name | Equation | Reading |
|---|---|---|---|
| PV | Planned Value | (budget for scheduled work) | — |
| EV | Earned Value | (budget value of work done) | — |
| AC | Actual Cost | (real cost incurred) | — |
| BAC | Budget at Completion | Sum of all budgets | Total planned cost |
| CV | Cost Variance | EV − AC | +under budget / −over budget |
| SV | Schedule Variance | EV − PV | +ahead / −behind schedule |
| CPI | Cost Performance Index | EV / AC | >1 under budget, <1 over budget |
| SPI | Schedule Performance Index | EV / PV | >1 ahead, <1 behind schedule |
| VAC | Variance at Completion | BAC − EAC | +under / −over at completion |
| EAC | Estimate at Completion | BAC/CPI (typical); AC+(BAC−EV) (planned rate); AC+Bottom-up ETC (replan); AC+[(BAC−EV)/(CPI×SPI)] (both influence) | Projected total cost |
| ETC | Estimate to Complete | EAC − AC, or bottom-up re-estimate | Cost left to finish |
| TCPI | To-Complete Performance Index | (BAC−EV)/(BAC−AC) or (BAC−EV)/(EAC−AC) | >1 harder to finish, <1 easier |

**Three-point (PERT) estimate**: tE = (tO + 4tM + tP) / 6 (beta); or simple triangular tE = (tO + tM + tP) / 3, where tO=optimistic, tM=most likely, tP=pessimistic.

**Critical path forward/backward pass**: EF = ES + Duration − 1 (forward pass); LS = LF − Duration + 1 (backward pass). Total float = LS − ES (or LF − EF); zero-float activities form the critical path.

## Decision Table — Risk Response Strategies

| Polarity | Strategies (priority-ordered by typical use) |
|---|---|
| Threats | Escalate (outside PM authority) → Avoid (eliminate, high-priority) → Transfer (shift to 3rd party, pay premium) → Mitigate (reduce probability/impact) → Accept (low-priority, active=reserve or passive=monitor) |
| Opportunities | Escalate (outside PM authority) → Exploit (make it happen, 100% probability) → Share (transfer to a party better able to capture it) → Enhance (increase probability/impact) → Accept (active=reserve or passive=monitor) |
| Overall project risk | Avoid (remove high-risk scope, or cancel if unacceptable) → Exploit (add high-benefit scope, or adjust thresholds) → Transfer/Share (3rd party involvement) → Mitigate/Enhance (replan, change scope/priority/resources) → Accept (active=overall contingency reserve or passive=monitor) |

## Decision Tree Worked Example — Expected Monetary Value (EMV)

Build new plant ($120M invest): 60% strong demand → $200M revenue (net $80M); 40% weak demand → $90M revenue (net −$30M). EMV = .6(80) + .4(−30) = **$36M**.
Upgrade plant ($50M invest): 60% strong demand → $120M revenue (net $70M); 40% weak demand → $60M revenue (net $10M). EMV = .6(70) + .4(10) = **$46M**.
→ Upgrade wins on EMV ($46M > $36M) *and* avoids the worst-case $30M loss — the lower-investment option can beat the bigger bet on both expected value and downside risk.

## Quick Reference — Motivation & Leadership Models

| Model | Core claim | Use when diagnosing... |
|---|---|---|
| Herzberg (Hygiene/Motivational) | Hygiene factors (pay, policy) prevent dissatisfaction; motivational factors (achievement, growth) create satisfaction — they don't substitute for each other | "Paid well but still unhappy" |
| Pink (Autonomy/Mastery/Purpose) | Intrinsic motivators outlast extrinsic ones once pay is fair | Disengagement despite fair compensation |
| McClelland (Theory of Needs) | People driven by Achievement, Power, or Affiliation in varying mix | Mismatched task assignment to what drives someone |
| McGregor/Ouchi (Theory X/Y/Z) | X = income-only/top-down; Y = intrinsically motivated/coaching; Z = meaning-driven/long-term culture | Choosing a management style for a given team culture |
| Tuckman Ladder | Forming→Storming→Norming→Performing→Adjourning (can stall/regress) | Team stuck in conflict or underperforming |
| Emotional Intelligence | Self-Awareness, Self-Management, Social Awareness, Social Skills | PM/team interpersonal friction |

## Quick Reference — Conflict Resolution Modes

| Mode | Approach | Typical outcome |
|---|---|---|
| Withdraw/Avoid | Retreat, postpone | Unresolved, deferred |
| Smooth/Accommodate | Concede to preserve harmony | One-sided |
| Compromise/Reconcile | Partial satisfaction all around | Lose-lose (sometimes) |
| Force/Direct | Push viewpoint via power position | Win-lose |
| Collaborate/Problem-solve | Incorporate multiple viewpoints, open dialogue | Win-win |

## Quick Reference — Schedule Compression & Resource Optimization

| Technique | Mechanism | Cost | Risk | Can change critical path/end date? |
|---|---|---|---|---|
| Crashing | Add resources to critical-path activities | Money | Low-moderate | No (shortens it) |
| Fast tracking | Run sequential activities in parallel | Low direct cost | Rework/quality risk | No (shortens it) |
| Resource leveling | Shift dates to balance resource supply/demand | Schedule delay | Low | Yes, can change |
| Resource smoothing | Adjust only within existing float | None (by design) | May not fully resolve overallocation | No, never changes |

## Quick Reference — Contract Type Risk Allocation

| Contract type | Cost risk mainly on | Best fit |
|---|---|---|
| Fixed-price | Seller | Well-defined, accurately estimable scope |
| Cost-reimbursable | Buyer | Uncertain scope, high-risk/R&D |
| Time & Materials (T&M) | Shared | Small projects, undefined scope |
| Target-cost | Shared (gain/loss formula) | Encourage efficiency, retain flexibility |

## Quick Reference — AI Ethics Checklist (run before scaling AI use)

Bias (diversify data, test periodically) · Privacy (secure sensitive data, clear policy) · Accountability (a human always owns the decision) · Reliability (validate output — may be wrong) · Safety (design/test/monitor properly) · Transparency (share how data/algorithms/decisions work) · Copyright (ownership of AI output is unsettled) · Sustainability (every request consumes real resources)

## Decision Tree — ADR Escalation Ladder

```
Dispute arises
├── Try Negotiation first (direct, no third party)
│      └── Unresolved → Mediation (neutral facilitator)
│             └── Unresolved → Arbitration (binding 3rd-party decision)
│                    or → Dispute Review Board (if pre-established panel exists)
│                    or → Expert Determination (narrow technical/financial question)
│                           └── Still unresolved → Litigation (last resort only)
```

## Tells & Smells (additional)

- **Management reserve being used to cover a known, already-identified risk** → should have drawn from contingency reserve instead; this blurs accountability and depletes the true emergency buffer.
- **TCPI meaningfully above 1.0 with no plan change** → remaining work must be done more efficiently than everything so far; flag as a feasibility risk, not just a status number.
- **Crashing a non-critical-path activity** → wastes money for zero schedule benefit.
- **An AI augmentation-tier output (e.g., portfolio trade-off analysis) accepted without iteration/review** → treating it with automation-tier trust; the single most common AI misuse pattern named in the Guide.
- **Jumping straight to a contract type without a make-or-buy analysis or source-selection method** → skips real risk-allocation decisions upstream.
- **A PMO chasing the "ideal" model** (directive/supportive/agile) and switching types repeatedly → explicitly flagged as decreasing value perception, not improving it.
