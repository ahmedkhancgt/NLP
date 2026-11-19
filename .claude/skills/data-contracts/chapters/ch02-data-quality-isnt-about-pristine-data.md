# Chapter 2: Data Quality Isn't About Pristine Data

## Core Idea
Data quality is not about achieving pristine/perfect data — it's about an organization's ability to understand the degree of correctness of its data and consciously trade off effort against the needs of data consumers ("fit for use"); OLTP/OLAP silos and differing stakeholder expectations are the root cause of most data quality issues.

## Frameworks Introduced
- **Data Quality Definition (authors')**: "an organization's ability to understand the degree of correctness of its data assets, and the trade-offs of operationalizing such data at various degrees of correctness throughout the data lifecycle, as it pertains to being fit for use by the data consumer."
  - When to use: framing any internal pitch for data quality investment.
  - How: shift language from "desired state of correctness" to "desired process for understanding correctness" — data is always decaying, so build a monitoring/iteration process rather than chasing perfection.
- **Leading vs. Lagging (Trailing) Indicators for Data Quality**: mirrors software quality measurement.
  - When to use: building a data quality measurement program.
  - How: track leading indicators (data debt, trustworthiness, ownership) to predict problems; track lagging indicators (data downtime, incident count) to measure realized damage.
- **Violated Expectations Taxonomy**: Schema, Semantics, SLAs, PII — the four forms a consumer's expectation of upstream data can take.
  - When to use: defining exactly what a data contract needs to specify (foundational to Ch. 4+).
  - How: log violations centrally by type, owner, source, and downstream impact; track the trend quarter over quarter.

## Key Concepts
- **OLTP (online transaction processing)**: optimized for fast CRUD transactions, ACID compliance, third-normal-form (3NF) modeling; owned by software engineers.
- **OLAP (online analytical processing)**: optimized for large historical scans, denormalized data, iterative business-logic exploration; owned by analysts/data scientists.
- **Data debt (as a metric)**: measured via documentation coverage, median dependencies per dataset, backfill job count, average filters per query.
- **Trustworthiness**: a leading indicator measured via survey (quarterly) and via the rate of redundant/replicated datasets.
- **Ownership (metric)**: percentage of upstream data sources with explicit, registered owners; predicts speed of incident resolution.
- **Data downtime**: time critical data is inaccessible/wrong; tracked via MTTD (mean time to detect) and MTTR (mean time to resolve).
- **Data incident**: any event compromising integrity, availability, or confidentiality of data with real-world business impact (not every glitch counts).
- **PII (personally identifiable information)**: data that can identify a person, requiring masking/access control as part of expectations.

## Mental Models
- Think of data as "descriptive, not functional" — unlike software (which must behave correctly), data's only value is in accurately reflecting reality; incorrect data has zero or negative operational value.
- Use "fit for use" rather than "pristine" as the target state — data quality is a continuous process, not a finished state, because data is always decaying.
- Think of OLTP and OLAP as two different "worldviews": the same business fact (e.g., average star rating) can be legitimately computed differently depending on whether you're optimizing for product simplicity (OLTP) or analytical nuance (OLAP) — the disagreement itself is a sign a contract is needed to codify the trade-off.
- Use the persona lens (data engineer, data scientist, analyst, software engineer, business team) to anticipate who feels a data quality failure and how, before pitching a fix.

## Anti-patterns
- **Selling data quality on "pristine data" alone**: fails because non-practitioners don't care about data in the abstract, only what they can do with it — pitches must be anchored to business/revenue risk.
- **Treating every outage as an "incident"**: dilutes the signal; only report incidents with real-world business impact (lost revenue, bad decisions, legal exposure).
- **Blaming data engineers for all data issues**: they often lack the business-logic context to resolve issues quickly, and most root causes originate upstream with software engineers.
- **Silent "throw it over the fence" schema changes**: software engineers announce changes on public channels but get no feedback until after breaking changes ship (e.g., the item_id barcode recall story below).

## Code Examples
*(omitted — this chapter has no code/config examples; it uses narrative case studies and diagrams instead)*

## Reference Tables
| Expectation type | Definition | Example |
|---|---|---|
| Schema | Structure, data types, column names | `customer_ID` always a 6-character string |
| Semantics | Underlying business logic/meaning | email field always contains "@" |
| SLA | Latency and volume requirements | minimum 1,000 events/hour from a source |
| PII | Identifiable personal information | `customer_name` masked to all consumers |

| Persona | How data quality issues hit them |
|---|---|
| Data engineers | Blamed for all data issues regardless of root cause; invisible when things work |
| Data scientists | ML models silently degrade; models can be wrong "by an order of magnitude" before detection |
| Analysts | Business logic disarray (e.g., a units change silently doubling values); blamed for changes they can't control |
| Software engineers | Usually the root cause via upstream changes, but get little pre-change feedback |
| Business teams | No agency to diagnose; can only flag symptoms to analysts |

## Worked Example
**Kelp aquarium review site**: engineering lets users submit reviews; OLTP stores reviews in 3NF, and the V1 feature simply averages all stars for simplicity (a product decision). The OLAP-side analyst builds a denormalized wide table and discovers multiple valid ways to calculate "average stars," plus open questions about session-duration logic (how to merge near-simultaneous sessions, duplicate accounts). Both the OLTP and OLAP computations are "correct" relative to their own constraints — the disagreement is a trade-off, not a bug, and is exactly what a data contract should codify explicitly rather than leave implicit.

**Marketing misfire during COVID**: an analyst saw sign-ups down >25% and marketing increased spend to compensate, only to later discover the drop was caused by an engineering change to how the sign-up event was recorded — not real user behavior. **Barcode recall**: a production engineer added a character to `item_id` without realizing downstream consequences; hundreds of mis-printed barcodes shipped to customers, requiring costly physical recalls. **COVID session-duration bug**: a vendor instrumentation bug caused a 25% drop in both active sessions and average time-on-page simultaneously; marketing wrongly attributed it to pandemic churn and overspent before an analyst traced it to a bug.

## Key Takeaways
1. Pitch data quality investment on business/revenue risk, not on the abstract value of "clean data."
2. Define data quality as an ongoing process of understanding correctness, not a finished state — data constantly decays.
3. OLTP and OLAP represent genuinely different, equally valid worldviews; miscommunication between them (not malice or incompetence) is the typical source of data quality bugs.
4. Measure data quality with both leading indicators (data debt, trust, ownership) and lagging indicators (downtime, incident count) — leading indicators let you intervene before damage occurs.
5. Centrally log violated expectations (schema, semantics, SLA, PII) to build an evidence base for governance investment.
6. Understand which persona is affected and how before proposing a fix — their pain points and leverage differ substantially.

## Connects To
- **Ch 1**: Builds on the data debt and GIGO concepts, converting them into measurable leading/lagging indicators.
- **Ch 3**: Will explain why scaling data infrastructure amplifies these OLTP/OLAP translation issues.
- **Ch 5/6**: The "violated expectations" taxonomy (schema, semantics, SLA, PII) becomes the literal structure of data contract components.
