# Chapter 8: Schedule Performance Domain

## Core Idea
The schedule is a communication/expectation-management tool as much as a plan: it represents how and when scope becomes delivered value, provides the basis for performance reporting, and proactively surfaces delays/risks early enough to act on them.

## Frameworks Introduced
- **Four Steps to Developing the Schedule** (Figure 2-23): (1) Define Activities → (2) Determine Sequence → (3) Estimate Effort and Duration → (4) Adjust. A cyclical loop, not strictly linear — "Adjust" feeds back into revisiting earlier steps.
  - When to use: as the skeleton for any schedule-building effort, predictive or not.
  - How: at Step 2, every activity (except first/last) needs at least one predecessor and successor with an appropriate logical relationship (start-to-start, finish-to-finish, start-to-finish, finish-to-start); use lead/lag time to keep it realistic.
- **Rolling wave planning**: outline broad milestones/key activities first, then progressively refine/expand detail as more information becomes available — used even in predictive projects when timelines are too compressed for full up-front detail.
- **Location-Based Scheduling (LBS)**: allocates quantities to defined physical locations, factoring production rates, crew sizing, task logic, and splitting/buffering needs — common in construction-style, location-driven work.
- **Lean scheduling**: based on lean/pull principles — work is pulled into the process when capacity exists, rather than pushed/assigned; steps are master scheduling → phase scheduling → look-ahead planning; goal is minimizing waste, maximizing value, limiting queues.

## Key Concepts
- **Project schedule**: output of a schedule model — linked activities with planned dates, durations, milestones, resources.
- **Estimate**: quantitative assessment of a likely amount/outcome (effort or duration); *effort* = labor units needed; *duration* = work periods needed given estimated resources.
- **Schedule baseline**: approved schedule model, changeable only via formal change control — more rigid under predictive, more flexible/nonexistent under adaptive.
- **Schedule forecasts**: predictions of future schedule conditions based on current progress/performance trends — updated continuously as the project executes.
- **Velocity**: rate at which deliverables are produced/validated/accepted per iteration (typically 2 weeks–1 month) — the adaptive-approach analog of progress tracking.
- **Project schedule network diagram**: graphical representation of logical dependencies among activities.

## Mental Models
- Treat schedule flexibility as a design goal, not a failure mode — it "enhances team morale, supports incremental delivery, and helps ensure higher-quality outcomes."
- Match your tailoring choice to development approach: predictive → up-front comprehensive timeline with critical path; adaptive → short timeboxed horizons; hybrid → predictive skeleton (key milestones) with adaptive sprints inside specific streams.
- When duration estimates look suspicious, check for cognitive bias (planning fallacy, "end-of-story illusion," Hofstadter's Law) before assuming the data is simply wrong.

## Anti-patterns — Duration-Estimating Pitfalls
- **Law of diminishing returns**: increasing one factor (e.g., a resource) eventually yields progressively smaller output gains — don't assume linear scaling.
- **Doubling resources ≠ halving time**: extra resources can *increase* duration via knowledge transfer overhead, learning curves, and added coordination.
- **Ignoring documentation of assumptions**: every duration estimate should record the data/assumptions behind it — otherwise future re-estimation has nothing to check against.
- **Low stakeholder involvement in schedule development**: per the Check Results table, this is explicitly called out as a cause of unrealistic or poorly developed schedules.
- **No buffers for known-unknowns**: skipping float, contingency reserves, or secondary-plan strategies for known risks sets the schedule up to break on first disruption.

## Reference Tables

**Schedule Performance Domain — 3 Processes**

| Process | Key Output | Focus Area |
|---|---|---|
| Plan Schedule Management | Schedule management plan (development approach, release/iteration length, accuracy level, control thresholds, reporting formats) | Planning |
| Develop Schedule | Schedule baseline, project schedule, schedule data, project calendars | Planning |
| Monitor and Control Schedule | Work performance information, schedule forecasts, change requests | Monitoring & Controlling |

**Monitor and Control Schedule — predictive vs. adaptive focus**

| Predictive concerns | Adaptive concerns |
|---|---|
| Determine schedule status | Compare delivered/accepted work vs. estimates for the elapsed cycle |
| Influence factors causing schedule change | Conduct retrospectives to correct/improve process |
| Reconsider schedule reserves | Reprioritize remaining backlog |
| Determine if overall schedule changed | Determine velocity (rate of delivery per iteration) |
| Manage changes via Assess and Implement Changes (integrated change control) | Manage changes as they occur |

**Estimating techniques** (pick based on context, not habit): expert judgment, Delphi technique, analogous estimating, parametric estimating, PERT, bottom-up estimating, planning poker, story points, T-shirt sizing.

**Table 2-7 — Check Outcomes, Schedule Performance Domain (condensed)**

| Outcome | How to check |
|---|---|
| Scheduling approach matches deliverables | Schedule aligns with the development approach (predictive/adaptive/hybrid) used for those deliverables. |
| Life cycle phases connect value delivery start-to-end | Project work represented in phases with appropriate exit criteria, launch to close. |
| Holistic delivery approach | Schedule shows no gaps or misalignment across the whole delivery. |
| Schedule documentation complete | Includes dependencies, durations, resource allocations, milestones per the chosen approach. |
| Scheduling tools/techniques used | Team used appropriate tools (PM software) and techniques (e.g., critical path method). |
| Stakeholders involved in schedule development | Check participation level of key stakeholders — low involvement risks an unrealistic schedule. |
| Sufficient buffers for known-unknowns | Major constraints/known risks factored in via float, contingency reserves, or secondary plans. |

## Worked Example

**Hybrid scheduling in practice**: A high-level project schedule is planned predictively to fix key milestones and deliverables. Inside that frame, subordinate streams with high requirements-volatility run on adaptive sprints/iterations (frequent reprioritization), while other, more stable components use Kanban or the critical path method. This lets the overall project stay on-track against fixed milestones while letting the volatile parts adapt quickly — the same hybrid-by-stream pattern introduced in ch04, applied specifically to scheduling.

## Key Takeaways
1. Use the four-step schedule-development loop (Define → Sequence → Estimate → Adjust) as a repeatable cycle, not a one-time pass.
2. Rolling wave planning is legitimate even in predictive projects when full up-front detail isn't feasible — don't treat it as an adaptive-only technique.
3. Watch for the duration-estimating pitfalls (diminishing returns, resource-doubling fallacy, cognitive biases) before trusting an estimate.
4. Tailor scheduling to life cycle/approach, product/deliverable attributes, team attributes, culture, environment, and chosen scheduling method — six distinct tailoring dimensions, not just "predictive vs. adaptive."
5. Low stakeholder involvement during schedule development is a named red flag for an unrealistic schedule — treat it as an early-warning leading indicator (ch06).
6. Always build in buffers for known-unknowns — float, contingency reserves, or secondary plans — this is an explicit Check Results criterion, not optional hygiene.

## Connects To
- **Ch 6 (Governance)**: all domains, including Schedule, "function under" Governance to maintain balance across Scope/Finance/Stakeholders/Resources/Risk.
- **Ch 7 (Scope)**: Scope-Schedule-Finance are named as the three most tightly linked domains — a change in one will likely impact the others.
- **Ch 9 (Finance)**, **Ch 11 (Resources)**, **Ch 12 (Risk)**: duration estimating draws directly on resource estimates/calendars (Resources) and risk-driven buffers (Risk); schedule changes ripple into cost (Finance).
