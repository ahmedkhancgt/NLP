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
