# Cheatsheet: Data Quality Fundamentals

## Core Formulas
- **Data Downtime**: `DDT = N × (TTD + TTR)` — incidents × (time to detection + time to resolution)
- **Cost of Data Downtime**: `(TTD hrs + TTR hrs) × Downtime hourly cost`
- **Annual Cost of Broken Data**: `Labor cost + Compliance risk + Opportunity cost`
  - Labor cost ≈ # data engineers × avg salary × 30%
  - Compliance risk ≈ 4% of annual revenue (GDPR-style estimate)
- **Annual Data Downtime Labor Cost**: `# data engineers × 1,804 (avg annual work hrs) × $62 (avg hourly salary) × % time on quality issues`
  - 50% low maturity / 35% average maturity / 20% high maturity
- **Downtime hours estimate**: `(Tables / 15) × (TTD + TTR)`, capped at 8,760 hrs/year (1 in 15 tables affected by an incident/year, per Monte Carlo data)
- **Precision** = TP / (TP + FP); **Recall** = TP / (TP + FN); **Fβ** weights recall as β× precision

## Decision Rules

**Should I invest in a data mesh?** Score: data sources + team size + domains + engineering-bottleneck severity (1-10) + governance priority (1-10).
- 1–15: probably don't need one yet
- 15–30: adopt mesh best practices now to ease future migration
- 30+: you're in the sweet spot — go for it

**When should I invest in data quality?** (any one is a strong signal)
- Recently migrated to the cloud
- Scaling past ~50 tables/sources (or fewer tables but high incident severity)
- Growing data team (risk of tacit-knowledge silos)
- ≥30% of team time spent firefighting
- More data consumers than a year ago
- Moving to self-service analytics
- Data is part of the customer-facing value proposition

**Build vs. buy?** Buy by default; build only if:
- Sensitive/regulated data can't be shared with vendors
- Niche customizations a vendor won't prioritize
- Strategic/competitive advantage to owning it

**Warehouse vs. lake vs. lakehouse?**
- Need strict schema, fast SQL BI, less governance overhead → warehouse
- Need flexibility, ML/non-SQL workloads, large unstructured volumes → lake (watch for swampification)
- Need both → lakehouse (Delta Lake, Redshift Spectrum, etc.)

**Fβ choice for anomaly detection**:
- β > 1 (favor recall) when missed incidents are costly/dangerous (e.g., safety alerts)
- β < 1 (favor precision) when false alarms burn trust/time (alert fatigue risk)
- β = 1 (F1) when both error types matter equally

## Incident Severity Triage (Tell-tale Signs)
| Signal | Likely Severity |
|---|---|
| Rows drop 10,000 → 1,000 | High — investigate immediately |
| Expected/documented schema change | Low — tag "expected," no action |
| Null rate jumps from baseline but table is deprecated/unused ("phantom data") | Low — verify usage before investing time |
| Affects executive/financial reporting | High — escalate via on-call/PagerDuty |
| Isolated to one rarely-used dashboard | Low-Medium — fix on normal cadence |

## Data Reliability Maturity Quick Check
| Stage | Tell |
|---|---|
| Reactive | Majority of time on fire drills; no proactive checks |
| Proactive | Manual QA queries, Slack/email alerts, ad hoc row-count checks |
| Automated | Scheduled validation, health dashboards, schema enforcement at ingestion |
| Scalable | DevOps-style staging, reusable validation, anomaly detection across all key metrics |

## Five Pillars of Data Observability (always check all five)
1. **Freshness** — Is it recent? When last updated?
2. **Distribution** — Values within range? Nulls/zeros normal?
3. **Volume** — Did all the data arrive?
4. **Schema** — What changed, by whom, why?
5. **Lineage** — Upstream sources / downstream consumers?

## RCA Quick Sequence
1. Lineage (find most upstream broken node)
2. Code (what logic/query last touched it, any recent changes?)
3. Data (slice by segment/time/schema for patterns)
4. Operational environment (job logs, permissions, infra, schedule changes)
5. Peers (who owns/uses it, past similar incidents?)

## Default SLA-Setting Process
1. Define: inventory data, baseline historical performance, interview consumers
2. Measure: pick SLIs (incident count, update frequency, expected distribution)
3. Track: set SLO ranges, build dashboards, revisit regularly

## Tells and Smells (Watch For)
- "V6_Final_RealFinal" style duplicate models/tables → no clear data ownership
- Team pinging "the one engineer who knows everything" → missing documentation/lineage
- Alerts being ignored → alert fatigue; retune or split channels
- >20% of engineering time on firefighting → reactive-stage maturity, invest now
- New field silently breaks another team's pipeline → missing cross-team schema contracts
- Dashboard "looks off" is the first detection method → TTD is far too high, invest in monitoring
