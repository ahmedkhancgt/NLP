# Cheatsheet

## Decision: Where in the data lifecycle should I place my first contract?
1. Third-party data → Transactional DB: usually infeasible directly on the third party; place the contract between staging and ingestion instead.
2. Product interface → Transactional DB: good target only if YOUR team owns the transactional DB (often doesn't, if you're on the data team).
3. **Transactional DB → Analytical DB: START HERE for most orgs.** Both software and data teams are stakeholders; highest leverage to prevent issues before they poison everything downstream.
4. Third-party data → Analytical DB: feasible mainly for CRM/ERP platforms (Salesforce, HubSpot) where you control the underlying data model even though the platform is third-party.
5. Analytical DB → Data products: good as a *second* step after upstream contracts exist; too late to prevent root issues if used first.
6. Analytical DB → Transactional DB (reverse ETL / ML batch predictions): less common; relevant mainly for ML output feeding customer-facing surfaces.
7. Data products → Business users: too downstream to prevent anything, but useful for tiering trust/expectations on what's "under contract."

## Decision: Hard failure or soft failure?
- **Hard failure (block the merge)**: default for most violations — the contract exists to prevent exactly this.
- **Soft failure (alert only, allow through)**: reserve for legitimate emergencies (e.g., a war-room hotfix) where blocking would be worse than the violation; the spec should state explicitly when this exception applies.

## Decision: Which detection method fits my asset type?
| Asset | Best detection methods |
|---|---|
| Transactional DB | Code-level (SCA), data lineage, data quarantining, CDC |
| Analytical DB | Same as above, plus live data monitoring |
| Streaming data | Stream processing, CDC, live monitoring, data lineage |
| First-party/third-party platform | Data quarantining, CDC, live monitoring, data lineage (no SCA — no code access) |

Rule of thumb: code-level detection (SCA) is most preventative but hardest to build; start with schema/metadata detection (cheap) and add SCA once that's working.

## Tells your org needs data contracts (signs of the problem)
- Dashboards fuel more questions than answers; "these numbers look off" is a recurring Slack message.
- A metric drop gets misdiagnosed as a business problem (marketing, ops) when it's actually a data bug — and real money gets spent chasing the wrong cause.
- Data engineers are "the face of failure" for issues they didn't cause.
- No one can say who owns a given dataset's full lifecycle.
- Software engineers announce breaking changes in Slack/email and get silence, then get blamed after the fact.
- Replicated/duplicate datasets proliferate because no one trusts the "official" one.
- A month of missing/bad data goes unnoticed until someone stumbles on it by accident.

## Tells your rollout is failing (signs of trouble)
- "We built the technology, but we're struggling with onboarding" (airplane-project trap).
- Teams ask "What's in it for me?" long after the pilot phase should have ended.
- Measurement efforts were standardized too early and now feel rigid/resisted (Phase 2 anti-pattern).
- Alert volume is high but nobody acts on it (alert fatigue — check the five Ws/one H for each alert type).
- A data catalog or contract spec debate is consuming more energy than adoption/ownership work (the "trap of standards").
- Urgency was borrowed from another big project (a migration) and faded when that project slipped.
- Territorial leadership resistance that seems irrational — usually rooted in fear that contracts make their team's work look obsolete.

## Trade-off matrix: Data Contracts vs. Data Observability
| | Contracts | Observability |
|---|---|---|
| Timing | Before change (prevention) | After change (detection) |
| Scope | Targeted (tier-one assets only) | Broad (whole stack) |
| CI/CD relationship | Embedded in the workflow | Complements the workflow |
| Input | Informed by business logic | Reflects business logic in the data |

Use both: observability informs which constraints belong in a contract; contracts prevent what observability would otherwise just report.

## Defaults and thresholds mentioned in the book
- Dunbar's Number: ~150 people before organic communication breaks down.
- 1-10-100 Rule: $1 (design) → $10 (QA) → $100 (post-release) cost of a bug.
- Kotter's urgency threshold: convince ≥75% of management the status quo is untenable.
- Median data-to-software engineer ratio: ~1:4 at top European tech companies (Dengsøe).
- Manufacturing stoppage cost reference: ~$22,000/minute in auto industry (historical).

## Quick reasoning: "Is this a data contract problem or something else?"
- If the fix is "write better documentation" or "build a better catalog/lineage tool" alone → likely necessary but insufficient; check whether an *accountability* gap (no owner, no enforced agreement) is the real root cause.
- If the issue is a semantic/business-logic change with no schema change (e.g., a field's valid conditions changed) → schema-only contracts won't catch it; you need semantic constraints or code-level (SCA) detection.
- If the issue is adoption stalling despite solid tech → it's an airline project, not an airplane project; diagnose friction and incentives, not the tooling.
