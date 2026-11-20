# Chapter 3: Ingredients of Data Governance: People and Processes

## Core Idea
Tools alone cannot deliver governance: the *people* (who wear multiple overlapping "hats" rather than clean single roles) and the *process* (which must be tailored to company type — legacy, cloud-native, retail, regulated, small, large) determine whether a governance strategy succeeds.

## Frameworks Introduced
- **"Hats" vs. "Roles" model**
  - When to use: whenever analyzing who does governance work in a real organization (as opposed to idealized frameworks).
  - How: define *tasks*, not job titles. One person often wears legal + privacy-tsar + data-owner hats simultaneously because of headcount limits; map responsibility to tasks/hats, not org-chart roles.
  - Hats: Legal (ancillary), Privacy tsar (governor), Data owner (approver/governor), Data steward (governor), Data analyst/scientist (user), Business analyst (user), Customer support specialist (user/ancillary), C-suite (ancillary), External auditor (ancillary).

- **Company-Type Governance Needs Matrix**
  - When to use: diagnosing why a generic governance framework doesn't fit your org.
  - How: classify your company as Legacy / Cloud-native / Retail / Legacy+Retail combo / Highly regulated / Small (<1000 employees) / Large (>1000 employees), then apply the specific considerations for that type (below).

- **Data Segregation Strategies** (three observed patterns)
  1. On-prem (uncurated) vs. cloud (curated) split — reduces blast radius but blocks cross-storage analytics.
  2. Tiered zones within one cloud environment (e.g., "insights zone" fully curated/accessible vs. restricted raw zone) — easier to manage, still carries public-cloud leak risk.
  3. Segregation by line of business — clear ownership/accountability but risks siloing and duplicated pipelines for cross-company analytics.

## Key Concepts
- **Hat**: a set of governance-related tasks performed, independent of job title; one person may wear several.
- **Privacy tsar**: internal Google term for the governor who oversees the overall governance process and ensures legal-approved regulations are followed; not necessarily technical.
- **Data owner**: implements the privacy tsar's strategy — architecture, pipelines, tooling, monitoring ("owns the data" technically).
- **Data steward**: performs categorization/classification/labeling — described as "highly manual," "soul-sucking," and the most commonly *incomplete* task in practice.
- **Tribal knowledge**: undocumented, word-of-mouth knowledge of "which dataset is good" — fragile, doesn't scale, breaks when people leave (especially post-acquisition).
- **Use-case-aware policy**: access decisions that depend not just on data class + role but on the *purpose* of access (e.g., shipping address usable for delivery, not marketing, without consent).
- **Data iceberg**: in large companies, a small "known/enriched/curated" top layer sits above a much larger "unknown/ungoverned" mass — the ungoverned mass is unusable AND risky.

## Mental Models
- **Responsibility clarity beats role completeness**: fuzzy ownership (hats vs. roles) causes miscommunication and incomplete stewardship even when "someone" nominally owns a task.
- **Segregation trades analytics power for blast-radius control**: every segregation strategy (on-prem/cloud split, zones, line-of-business) is a dial between cross-domain analytics capability and containment of leak impact — pick deliberately.
- **Culture as the final integrating layer**: tools + people + process alone are each necessary but insufficient; a "culture of privacy and security" is what makes the other three durable (full treatment in Ch. 9).

## Anti-patterns
- **Assuming a textbook governance framework (owner vs. custodian split, etc.) maps cleanly onto your headcount**: it rarely does; in practice technical staff absorb both "owner" and "custodian" tasks.
- **Pure role-based access control (RBAC) without use-case parameter**: fails when the same employee needs data for one purpose (fulfillment) but not another (marketing) — see retail email example.
- **Governing only "known" data pipelines and ignoring ad hoc ones**: leaves large companies with a growing, un-enriched data iceberg that is both wasted value and latent risk.
- **Line-of-business data segregation with pipelines hard-wired to a single silo**: blocks the very cross-company analytics (e.g., air shipping vs. retail sales patterns) that justified collecting the data.
- **Views-based sanitization as a long-term strategy**: manually re-created clean views don't scale — they require constant re-creation as new data arrives and proliferate stale datasets.

## Code Examples
Not applicable — this is a people/process chapter with no code, config, or policy syntax examples.

## Reference Tables
- **Table 3-1 (hats)**: Hat × Category (governor/approver/user/ancillary) × Key tasks — see Key Concepts above for the full hat list.
- **Table 3-2 ("data lake" tiers)**: Insights zone (known/enriched/curated/cleaned, governance controls applied, highest access) down to raw/uncurated zone (few users) — a tiered-access model within one cloud environment.
- **Table 3-3 (dataset "views")**: Plain-text / Hashed / Redacted versions of the same column (e.g., customer name) — illustrating the classic "create a clean view" strategy and its cost (manual effort, staleness, proliferation).

## Worked Example
**Google's COVID-19 privacy work (privacy tsar in action)**:
1. *Community Mobility Reports*: aggregated, opt-in-only (location history must be explicitly enabled) mobility data; differential privacy added statistical noise and excluded small outlier groups entirely, enabling health-authority insight without individual tracking.
2. *Exposure Notifications*: phones exchange random, frequently-rotating Bluetooth identifiers; matches against positive-diagnosis reports are surfaced to the exposed user only, with no location data collected, no identity disclosed, and results shared only with public health authorities (not Google/Apple). Shows how a privacy tsar resolves a genuine privacy-vs-public-good tension via opt-in design, data minimization, and restricted sharing scope — not by avoiding the use case.

**Large US retailer (350,000+ employees) segregation-by-line-of-business**: Split data into air shipping / ground shipping / retail store sales / marketing silos to speed CCPA compliance per line. Benefit: faster enrichment, clear accountability. Cost: pipelines fed only one silo each, so cross-line analytics (e.g., correlating air shipping with retail sales patterns) became impossible without duplicating data across silos.

**Washington, DC IMPACT teacher-ranking system (2009) — gaming-of-metrics case**: An algorithm scoring teachers purely on standardized test score improvement (no social/familial context, no administrator feedback loop) caused teachers to optimize for test-passing rather than teaching; a 2018 investigation found 1 in 3 2017 graduates received diplomas despite missing classes. Lesson for governance process design: pilot new metrics/processes transparently, solicit affected-party feedback *before* full rollout, and be willing to pivot.

## Key Takeaways
1. Map governance work to tasks ("hats"), not idealized role titles — most orgs can't staff every textbook role.
2. Data stewardship (classification/labeling) is the single most under-resourced, most critical task — expect it to be incomplete and plan mitigations (prioritize by category, automate where possible).
3. Tailor your governance process to your company type (legacy/cloud-native/retail/regulated/small/large) rather than adopting a generic framework wholesale.
4. Use-case, not just role, must be a parameter in access policy wherever the same data serves multiple purposes with different consent/appropriateness.
5. Every data-segregation strategy is a deliberate tradeoff between containment and cross-domain analytics capability — choose consciously and document the tradeoff.
6. Introduce new governance metrics/processes as a transparent pilot with stakeholder feedback loops to avoid unintended gaming (DC IMPACT lesson).
7. Acquisitions compound governance debt: the acquired company's tribal knowledge, enterprise dictionary (or lack thereof), and culture must be actively reconciled or data sits unused.

## Connects To
- **Ch 1**: Expands the governors/approvers/users taxonomy into the full hats model.
- **Ch 4**: RACI matrices and roles-and-responsibilities sections build directly on this chapter's hat definitions.
- **Ch 9**: "Culture of privacy and security," introduced briefly here, is the dedicated topic of the final chapter.
- **GDPR / CCPA**: referenced as drivers forcing companies (regulated or not) to formalize use-case-aware policy and data findability (right to be forgotten).
