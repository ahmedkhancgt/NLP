# Chapter 8: Real-World Case Studies of Data Contracts in Production

## Core Idea
Three independent companies (Convoy, Glassdoor, Adevinta Spain) — different industries, different geographies, different tech stacks — converged on the same diagnosis: their data problems were not fundamentally technical (lineage tools and documentation didn't fix them) but a lack of accountability/alignment between producers and consumers, solved only once they implemented enforceable, automated data contracts as the upstream "handshake."

## Frameworks Introduced
- **"You produce it, you own it" (Adevinta Spain's 4 guiding principles)**: 1) You produce it, you own it (push architecture, data as a maintained product); 2) Data-driven teams without data analysts (no manual validation dependency); 3) Data governance by design (embedded in code, not manual policy); 4) Data people working as software engineers (testing, version control, SWE discipline applied to data).
  - When to use: framing the cultural shift needed alongside any technical contract rollout.
  - How: move from a pull architecture (consumers accept whatever they get) to a push architecture (producers decide who receives data and how).
- **Convoy's Diagnosis Pattern**: tooling alone (better lineage, better documentation) fails to fix a root accountability problem — only an enforceable agreement defining "what correct means" closes the gap.
  - When to use: when lineage/observability/documentation investments plateau without resolving recurring data incidents.
  - How: check whether anyone owns the *full lifecycle* of a dataset; if not, no amount of visibility tooling will substitute for an explicit contract.
- **Adevinta's Two-Stage Implementation**: Definition time (schema → GenAI-drafted contract proposal → producer/consumer Slack review → validation → auto-created Airflow DAG → S3 upload) then Runtime ingestion (DAG pulls contract + Kafka data into Databricks/Spark, applies schema, quarantines violations, splits/deletes PII for GDPR, stores in analytical DB).
  - When to use: designing an automated, low-friction contract creation and enforcement pipeline at scale (thousands of tables).
  - How: minimize manual contract-authoring effort via GenAI-assisted proposal generation from existing schemas; automate migration of legacy tables into contract coverage rather than requiring manual backfill.

## Key Concepts
- **Semantic events vs. CDC events** (Convoy/Kafka): CDC events describe *what* changed (state before/after) but not *why*; semantic events explicitly describe the real-world business event (e.g., "shipment canceled") — reduces error-prone reverse-engineering of intent from raw change logs.
- **Push vs. pull architecture**: pull = consumers accept whatever data arrives; push = producers define and control what's exposed and to whom (Adevinta's shift).
- **Write-Audit-Publish (WAP)** (Glassdoor): batch data reviewed in staging before promotion to production (same pattern as Ch. 6).
- **"Vertical by vertical" rollout** (Convoy/Adrian Kreuziger's advice): don't put everything under contract at once; attack by business vertical or asset importance.
- **Data product orientation** (Adevinta): treating ingested data as an actively maintained product with a lifecycle and defined users, not passive storage exhaust.
- **Delta Lake time-travel**: used by Adevinta for atomic, rollback-safe ingestion writes.

## Mental Models
- Think of recurring "garbage in, garbage out" incidents not as isolated bugs but as *symptoms of missing accountability* — the fix is organizational (an enforceable agreement), not just another monitoring tool.
- Use the "perfect storm" conditions (Glassdoor) as a readiness checklist for contract adoption: (1) shared understanding of which data assets matter to the business, (2) felt pain on both producer and consumer sides, (3) leadership visibly connects the pain to revenue.
- Treat contract rollout as requiring both *technical* automation and *psychological* trust-building — Glassdoor explicitly frames proactive data quality as "instilling confidence," not just "imposing rules."
- Expect heterogeneous team readiness within one org (Adevinta's real estate/motor/misc teams had very different baseline engineering maturity) — plan for differentiated onboarding, not a single uniform mandate.

## Anti-patterns
- **Trying to fix accountability gaps with visibility tools alone**: Convoy invested in better lineage and then better documentation first; both helped marginally but didn't solve the root cause, because the problem was ownership, not information.
- **Treating a bad dashboard number as a business problem by default**: Convoy's leadership assumed a drop in "auction engagement" was a marketing problem and spent "hundreds of thousands of dollars" on a campaign before discovering it was a data bug — a costly lesson in checking data quality before acting on a metric.
- **Rolling out contracts to everything at once**: explicitly warned against by Adrian (Convoy) — attack vertical by vertical or only where it makes sense.
- **Requiring full manual contract authoring at scale**: Adevinta's early version failed to get traction when producers had to manually write contracts for thousands of existing tables; the fix was automating migration and GenAI-assisted proposal drafting.
- **Ignoring user experience in adoption tooling**: both Convoy and Adevinta leads independently stress that automation and minimal friction (not mandates) are what make adoption stick ("otherwise people won't engage with your project").

## Code Examples
```json
{
  "contract_name": "MyNewEvent",
  "contract_version": "1",
  "description": "Event published from microservice",
  "start_date": "2024-10-01T00:00:00+00:00",
  "schema": { "source": "url", "version": "1", "location": { "url": "https://schema.…/events/…/MyNewEvent-Event.json/1.json" }, "format": "jsonSchema" },
  "landing_source": { "kafka_topic": "pub.mytopic" },
  "pii_fields": [],
  "source": "ms-mysource",
  "owner": "team-myteam",
  "contact_support": "<FILL>",
  "data_periodicity": "<daily/hourly>",
  "retention": "<FILL>",
  "provider_ids": ["mymarketplace"]
}
```
- **What it demonstrates**: Adevinta Spain's real production contract spec template (built before standardized specs existed), notably including `pii_fields`, `data_periodicity`, and `retention` for GDPR compliance — fields not emphasized in the book's own Ch. 5/7 example.

## Reference Tables
| Company | Industry | Trigger for adoption | Key technical lever |
|---|---|---|---|
| Convoy | Freight marketplace | Silent schema change caused false "marketing problem" diagnosis costing hundreds of thousands of dollars | Kafka + semantic events (vs. raw CDC) in CI/CD |
| Glassdoor | Job listings/reviews | Revenue-driving brand ad impressions flowing through many system handoffs | WAP + static code analysis + LLM-assisted business-context surfacing |
| Adevinta Spain | Online marketplaces (jobs/cars/real estate) | Month of silently missing data discovered by chance; GDPR compliance pressure | GenAI-drafted contracts + Airflow DAG automation + Delta Lake |

| Adevinta measured outcomes | Before → After |
|---|---|
| Engineer time on data quality issues | 2 days/week → 0.5 day/week |
| PII automatically identified/governed | ~0% → 40% in months |
| Producers understanding analytical value of their data | 0% → 65% |

## Worked Example
**Convoy's marketing misdiagnosis**: A dashboard showed dropping "auction engagement." Leadership concluded it was a marketing problem and funded a costly campaign to boost participation. Weeks and hundreds of thousands of dollars later, the real cause was found: a bug in the underlying data table, not a market/behavior problem. This incident — "the system hadn't failed, the data had" — was the turning point that led Convoy to try lineage tools, then documentation, before realizing the root cause was a lack of lifecycle ownership, which led to piloting enforceable data contracts in CI/CD with semantic Kafka events replacing raw CDC events.

**Adevinta Spain's missing-data discovery**: an analyst discovered a full month of missing ingested data with no alert, no communication, and no awareness it had happened — described by Sergio as a probable tipping point. Combined with a dual-table (JSON + Parquet) architecture where schema mismatches silently broke downstream queries, and a pull-based Kafka consumer architecture where a single failing table could stall the whole pipeline, this pushed the team to build a two-stage (definition-time + runtime) automated contract system with GenAI-assisted contract drafting and automatic migration of legacy tables into contract coverage.

## Key Takeaways
1. Across three unrelated companies, the root cause of chronic data quality failure was the same: no one owned the full lifecycle of a dataset, so no mechanism enforced producer/consumer expectations.
2. Lineage tools and documentation alone are necessary but insufficient — they add visibility, not accountability or enforcement.
3. A single high-visibility, high-cost incident tied to revenue (Convoy's marketing misdiagnosis, Glassdoor's ad impressions, Adevinta's missing month of data) is often what finally secures leadership buy-in.
4. Roll out contracts vertical-by-vertical or asset-by-asset, not as an all-at-once mandate.
5. Automate contract creation itself (schema reuse, GenAI-drafted proposals, automated legacy-table migration) — manual authoring at scale kills adoption.
6. Shift from CDC-style "what changed" events to semantic "why it changed" events where possible, to reduce fragile reverse-engineering of business meaning.
7. User experience and minimal developer friction are repeatedly cited (by Convoy's and Adevinta's leads independently) as the actual determinant of adoption success, not the sophistication of the technology.

## Connects To
- **Ch 3**: Explicitly references Dunbar's Number/Conway's Law as the reason alignment alone (without automation) didn't scale at Glassdoor.
- **Ch 6**: Reuses WAP, static code analysis, and Kafka/CDC concepts introduced there, now shown in production.
- **Ch 9–12**: This chapter is explicitly framed as the bridge into Part III's focus on leadership buy-in, first wins, and measuring impact.
