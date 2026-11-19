# Chapter 9: Finance Performance Domain

## Core Idea
Finance is about monetary resource allocation and value maximization, not just cost control — and the real value of financial measurement isn't the data collection itself, but the decisions and actions it enables through conversations in *other* performance domains.

## Frameworks Introduced
- **Two Budget Buildup Scenarios** (Figure 2-25): "Reserves Managed Implicitly" (initial budget = Work Cost Estimates + Cost Baseline + Contingency Reserve + Management Reserve, all bundled) vs. "Reserves Managed Explicitly" (reserves tracked and released separately, outside the initial budget). Neither is universally correct — organizational preference and external factors decide which applies.
  - When to use: before building any budget — explicitly decide and document which scenario this project follows, since it changes what "cost baseline" even means.
- **Contingency Reserve vs. Management Reserve**: contingency = time/money for **known risks with active response strategies** (known-unknowns), usually inside the initial project budget, released per the risk response plan; management reserve = time/money for **unforeseen work within scope** (unknown-unknowns), held outside the baseline, released at senior leadership's (or sometimes the PM's) discretion.
  - When to use: anytime someone says "we need more budget" — first ask whether this is a *known risk materializing* (draw from contingency) or a *genuine unknown-unknown* (draw from management reserve, requires escalation).
- **CapEx vs. OpEx**: CapEx = funds to acquire/upgrade/maintain physical assets (property, equipment, tech); OpEx = funds for ongoing day-to-day operations (advertising, admin, wages, rent, utilities). Different budget strategies and funding rules often attach to each.

## Key Concepts
- **Value definition**: value can be tangible (ROI, IRR, payback period, ROA, ROACE) or intangible (customer satisfaction, innovation, social/environmental impact) — "value may be more than just profit."
- **Value maximization**: Finance's job is not just cost management but ensuring maximum organizational value — aligning with strategy and using ROI/IRR alongside social-impact and innovation indicators.
- **Funding**: can come from internal budgets, customer contracts, grants, or crowdfunding; PMs are often asked to lead or support funding activities.
- **Financial constraints**: budget is the primary one, but not the only one — funding type restrictions (CapEx vs. OpEx), fiscal-year allocation rules, or quarterly agile-budget revisions all constrain differently.
- **Cost baseline**: approved time-phased project budget (reserve treatment varies per the two buildup scenarios above), changeable only through formal change control.
- **Cost measurement timing**: different stakeholders may measure "cost" at acquisition decision, order placement, delivery, or actual-cost-recorded — know which one your organization uses before comparing numbers.

## Mental Models
- Ask "is this cost decision trading schedule/quality for money, or money for schedule/quality?" — e.g., limiting design reviews cuts project cost but can raise the product's lifetime operating cost; cost decisions ripple into other baselines.
- Treat financial measurement as a trigger for conversation, not an end product — a dashboard number only has value once it changes what people in Scope, Schedule, Resources, or Stakeholders actually do next.
- Iterative/adaptive development approaches flatten the spending curve over time (vs. predictive's typical front/mid-loaded curve) — budget and controls need to be time-phased accordingly, often with quarterly re-assessment instead of a single annual allocation.

## Anti-patterns
- **Conflating contingency and management reserves**: using management reserve (meant for true unknown-unknowns, senior-leadership discretion) to cover a known, already-identified risk that should draw from contingency instead — this blurs accountability and depletes the true emergency buffer.
- **Treating financial reporting as the deliverable**: per Key Concepts, "the value of financial measurements is not in the collection and dissemination of the data" — a report nobody acts on isn't doing its job.
- **Ignoring resource-availability cost drivers**: forgetting that human/physical/virtual resource constraints (higher wages, training costs, expedited shipping, rental surcharges, late-delivery penalties) directly inflate actual and forecasted costs.

## Reference Tables

**Finance Performance Domain — 4 Processes**

| Process | Key Output | Focus Area |
|---|---|---|
| Plan Financial Management | Financial management plan, funding strategy | Planning |
| Estimate Costs | Cost estimates, basis of estimates | Planning |
| Develop Budget | Cost baseline, project funding requirements | Planning |
| Monitor and Control Finances | Work performance info, revenue/cost forecasts, change requests, funding proposals | Monitoring & Controlling |

**Tailoring Considerations** (2.4.3): product/regulatory load (SOX, GDPR compliance in finance/pharma raises formality); development approach (iterative flattens the spend curve — needs time-phased budgeting, continuous funding assessment); procurement strategy (make-or-buy decisions drive contract-type choice — fixed-price vs. T&M — which shifts cost-overrun risk); value definition (varies by organization); resource availability (constraints inflate actual/forecasted costs).

**Table 2-8 — Check Outcomes, Finance Performance Domain (condensed)**

| Outcome | How to check |
|---|---|
| Contributes to business objectives/strategy (value maximization) | ROI, NPV, IRR, cost-benefit analysis, KPIs, OKRs, CapEx, OpEx. |
| Completeness within/below budget | Variance analysis: Cost Variance (CV), Cost Performance Index (CPI); vendor contract target accomplishment. |
| Deliverables validated per plan | Earned Value Management (EVM) + other org-defined metrics. |
| Value created (investment or other) | Metrics matched to the project's own success criteria (e.g., future investment/potential value). |
| Financial visibility achieved | Trend analysis, graphical analysis, forecasting. |

## Worked Example

**Government-sector tailoring (2.4.3.1, Example 3)**: A government project faces higher fiscal-accountability and risk-aversion demands due to public accountability and regulation. Response: evaluate budget reserves with a *greater* buffer than a typical commercial project would use, specifically because the sponsor/stakeholders prioritize absorbing unforeseen financial challenges over minimizing reserve size — the "right" reserve size is itself a tailoring decision driven by sector, not a fixed percentage rule.

## Key Takeaways
1. Before building a budget, explicitly decide: implicit or explicit reserve management (Figure 2-25) — this changes what "cost baseline" means for the whole project.
2. Separate contingency reserve (known risks, response-plan-driven) from management reserve (unknown-unknowns, senior-leadership discretion) — don't let one substitute for the other.
3. Value is multidimensional — track ROI/IRR alongside social/environmental/innovation indicators when the organization's value definition includes them.
4. Tailor financial controls to regulatory load, development approach, procurement strategy, and resource constraints — a single generic finance process won't fit a pharma predictive megaproject and a small adaptive startup project equally.
5. Use CV/CPI/EVM as the standard variance toolkit for "are we within budget," but also track forward-looking financial visibility (trend/forecast analysis), not just backward-looking variance.

## Connects To
- **Ch 6 (Governance)**, **Ch 7 (Scope)**, **Ch 8 (Schedule)**: Finance has direct impact with these three; decisions in any one domain ripple into the others.
- **Ch 11 (Resources)**: resource availability/constraints are a named driver of actual and forecasted cost.
- **Ch 12 (Risk)**: contingency reserve is explicitly tied to known risks with active response strategies — see Risk domain for how those response strategies are built.
