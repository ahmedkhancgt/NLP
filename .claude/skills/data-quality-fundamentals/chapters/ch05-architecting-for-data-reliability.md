# Chapter 5: Architecting for Data Reliability

## Core Idea
Data reliability must be architected intentionally across the whole data platform — ingestion, storage, transformation, BI, discovery/governance, and observability — and governed by DevOps-inspired SLAs/SLOs/SLIs, with ROI quantified via time-to-detection (TTD) and time-to-resolution (TTR).

## Frameworks Introduced
- **Six-Layer Data Platform Model**: ingestion, storage and processing, transformation and modeling, business intelligence and analytics, discovery and governance, quality and observability — interconnected (not strictly sequential) layers every data platform needs.
  - When to use: as an architecture checklist when building or auditing a data stack.
  - How: invest in tooling/process for each layer; observability threads through all other layers rather than sitting "on top."
- **Data Observability (5 Pillars)**: freshness, distribution, volume, schema, lineage — the data analogue of application observability's 3 pillars (metrics, logs, traces).
  - When to use: as the measurement framework for data health in the pipeline, paralleling SRE practices.
  - How: instrument each pillar with automated monitoring, alerting, and triaging (see Ch 4 for implementation).
- **SLA / SLO / SLI framework for data**: SLA = the customer promise and consequence of failure; SLO = the target value for an SLI; SLI = the specific measured number (e.g., % of 200 responses, hours since last table update).
  - When to use: formalizing data reliability commitments between data teams and downstream consumers.
  - How: 3-step process — (1) Define reliability via stakeholder interviews and historical baselining; (2) Measure via SLIs (incident count, update frequency, expected distribution); (3) Track via SLOs (acceptable ranges) and dashboards.
- **Cost of Data Downtime formula**: `(TTD hours + TTR hours) × Downtime hourly cost = Cost of data downtime`.
  - When to use: quantifying ROI of data quality investment for executive buy-in.
  - How: estimate downtime hourly cost from engineering time (e.g., ~$14.75/hr per engineer at 25% of an hour spent firefighting) plus idle analyst time (e.g., 4 analysts × $75/hr × 60% impact factor ≈ $300/hr); annualize across monthly downtime hours.
- **Annual Cost of Broken Data formula**: `Labor cost + Compliance risk + Opportunity cost = Annual cost of broken data`, where labor cost ≈ (# data engineers × avg salary × 30%), compliance risk ≈ 4% of annual revenue (GDPR-style), and opportunity cost = foregone revenue from delayed releases/acquisition.

## Key Concepts
- **Data reliability**: an organization's ability to deliver high data availability and health throughout the entire data life cycle — the outcome of sustained high data quality.
- **Time to Detection (TTD)**: how long it takes to surface a data quality issue (often days/weeks/months when detection relies on downstream consumers noticing something "looks off").
- **Time to Resolution (TTR)**: how long it takes to fix a data incident once detected; depends on lineage maturity, catalog robustness, and available resources.
- **Data storytelling**: communicating the "why" behind data trends via narrative, beyond raw visualization — the payoff of a working BI layer.
- **DAMA UK's Six Dimensions of Data Quality**: completeness, timeliness, validity, accuracy, consistency, uniqueness — the traditional data-governance measurement framework (noted as not always directly applicable to data engineering practice, e.g. "accuracy").
- **Data enrichment**: merging first- or third-party data into existing data sets to fill gaps and increase value/reliability.
- **Unit / Functional / Integration testing (data)**: unit tests validate small code/SQL snippets; functional tests validate large data sets across validation/integrity/ingestion/processing/storage/ETL stages; integration tests validate the whole pipeline against fake data before production use.

## Mental Models
- Application observability (metrics, logs, traces) maps directly onto data observability (freshness, distribution, volume, schema, lineage) — borrow SRE culture wholesale rather than reinventing it.
- "Garbage in, garbage out": decision quality is bounded by input data quality, so catching issues at ingestion is cheaper than catching them downstream.
- SLAs/SLOs/SLIs only work with upfront stakeholder alignment — "setting SLAs for the sake of setting them" without accountability is worse than not setting them.
- Treat the six platform layers as interconnected, not hierarchical — the same tool can span 2-3 layers at once.

## Anti-patterns
- **Setting SLAs without stakeholder buy-in**: SLAs/SLOs are meaningless without alignment from data producers, engineers, analysts, and consumers on what matters and why.
- **Relying on downstream consumers to detect breakage**: if TTD is measured in weeks because a dashboard "looked off," every decision made on bad data in that window must be revalidated — a costly default state.
- **Treating data transformation and data modeling as synonyms**: they are distinct processes (data prep vs. encoding business logic into table relationships), and conflating them leads to unclear ownership.
- **Measuring data quality only via DAMA's traditional six dimensions**: dimensions like "accuracy" aren't always directly actionable for data engineers who don't see the business context — favor engineering-native metrics (null rate, timeliness, duplication rate).
- **Setting SLOs without executing against them**: teams get caught up in defining metrics but fail to prioritize and resource the work needed to actually hit them.

## Reference Tables
| Application Observability Pillar | Data Observability Pillar (5) |
|---|---|
| Metrics | Freshness, Distribution, Volume (quantifiable health signals) |
| Logs | Schema (structural change history) |
| Traces | Lineage (causally related up/downstream dependencies) |

| SLx | Definition | Example |
|---|---|---|
| SLA | Customer promise + penalty for missing it | Slack: 99.99% uptime/quarter or service credits |
| SLO | Target value for an SLI | 99% success rate, 95% of the time |
| SLI | The actual measured number | % of HTTP 200 responses from servers |

| Data Platform Layer | Core Question |
|---|---|
| Ingestion | How does data enter the system (ETL vs. ELT)? |
| Storage & Processing | Warehouse, lake, or lakehouse? |
| Transformation & Modeling | How is raw data shaped into business logic? |
| BI & Analytics | How is data made actionable for humans? |
| Discovery & Governance | Where is data, who owns it, who can access it? |
| Quality & Observability | Is the data healthy at every stage? (threads through all layers) |

## Worked Example
**Blinkist** (16M+ users, ebook subscription service): in 2020, COVID-19 invalidated historical behavioral data just as Blinkist was investing heavily in auto-optimizing paid marketing (Facebook/Google) on a 40% growth target. Lack of real-time data tracking caused marketing spend misallocation, and Director of Engineering Gopi Krishnamurthy's team spent an estimated 50% of their working hours firefighting data downtime, repeatedly explaining failures to executives on Monday calls instead of fixing root causes. The fix: adopting "data reliability engineering" as a first-class discipline (explicitly modeled on DevOps/SRE), implementing systematic testing and observability, and setting concrete data SLAs/SLIs aligned with company leadership. Outcome: ~120 hours/week saved across a 6-person data engineering team, and improved marketing targeting attributed directly to the new observability and SLA discipline — illustrating that setting SLOs alone doesn't help; the team's success came from prioritizing and executing against the metrics they chose.

## Key Takeaways
1. Architect for data reliability across all six platform layers simultaneously — observability threads through, rather than sits atop, the other five.
2. Quantify the cost of data downtime using TTD/TTR and a downtime hourly cost estimate; this is essential to win budget and executive buy-in.
3. Borrow the SLA/SLO/SLI framework directly from SRE, but customize SLIs to what your specific data consumers actually need — a wrong SLI invalidates your whole reliability strategy.
4. Setting SLAs requires three disciplined steps: define (stakeholder interviews + baseline), measure (choose SLIs), track (set SLO ranges and dashboards) — skipping stakeholder alignment undermines the whole effort.
5. Catch issues as early (upstream) as possible — ingestion-time testing is cheaper than downstream detection, but testing at ingestion doesn't guarantee reliability later in the pipeline.
6. Data quality transformation is organizational, not just technical — plan for a gradual, company-wide culture shift, not a single tooling purchase.

## Connects To
- **Ch 4**: Supplies the concrete anomaly-detection techniques that operationalize the five observability pillars introduced here at a strategic level.
- **Ch 6**: Picks up immediately with incident management and resolution — the operational process for responding once SLAs/observability flag an issue.
- **Site Reliability Engineering (SRE) / DevOps**: the entire SLA/SLO/SLI and TTD/TTR framework is explicitly borrowed from this external discipline.
