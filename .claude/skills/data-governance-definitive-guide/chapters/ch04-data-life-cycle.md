# Chapter 4: Data Governance over a Data Life Cycle

## Core Idea
Operationalizing governance requires mapping it onto the data life cycle (creation → processing → storage → usage → archiving → destruction) and codifying it in a living **data governance policy**, built through a structured, step-by-step program.

## Frameworks Introduced
- **Data Life Cycle (6 phases)**
  - When to use: structuring *where* in the data's journey each governance control applies.
  - How: Creation/Capture (acquisition/entry/capture) → Processing (ETL, lineage, quality, classification) → Storage (encryption at rest, backup) → Usage (access mgmt, catalog, audits) → Archiving (retention controls, removed from active prod) → Destruction (purge, compliance-driven timing). Phases are *logical dependencies, not literal data flows* — data can skip phases or loop back.

- **Data Life Cycle Management (DLM)**
  - When to use: when you need a repeatable, policy-based *process* (not just a conceptual life cycle) governing data flow end-to-end.
  - How: formalize each life-cycle phase into repeatable organizational steps.

- **Data Management Plan (DMP) — 5 guidances**
  - When to use: as a lightweight starting template for a governance policy, borrowed from research-data-management practice.
  - How: (1) identify data to be captured (types/sources/volume), (2) define how data will be organized (warehouse vs. lake, tools), (3) document storage/preservation strategy (backup, retention), (4) define data policies (licensing, legal/ethical restrictions), (5) define roles and responsibilities.

- **Data Governance Policy Structure** (document template)
  - Sections: Vision/mission, Policy purpose, Policy scope, Definitions/terms, Policy principles (access/usage/integration/integrity rules), Program structure (RACI), Policy review cadence, Further-assistance contacts.

- **Step-by-Step Operationalization Process**
  1. Build the business case (identify pain points, risks, success metrics; start small, aim for quick wins).
  2. Document guiding principles (consult DGI, DAMA, DGPO, Enterprise Data Management Council resources).
  3. Get management buy-in (tie to KPIs; present business case + principles to C-level).
  4. Develop an operating model (define roles, council/steward processes).
  5. Develop a framework for accountability (custodianship per data domain, visible data owners).
  6. Develop taxonomies and ontologies (categories for organization, classifications for sensitivity).
  7. Assemble the right technology stack (map infra/architecture/tools to the policy).
  8. Establish education and training (business-language materials, recurring sessions).

## Key Concepts
- **Data acquisition / data entry / data capture**: three ways data is created — from a third party, from manual human/device entry, or from automated device/sensor capture — each with distinct governance checks (e.g., acquisition needs usage-rights contracts).
- **Data archiving**: removal from active production but retained for possible future reactivation; governed by retention-length policy.
- **Data destruction/purging**: permanent removal of every copy; must be timed and provable (regulatory + cost reasons).
- **RACI chart**: Responsible, Accountable, Consulted, Informed — maps roles to governance tasks/decisions.
- **Transactional vs. analytical systems**: transactional systems optimize for concurrent operational writes; analytical systems (data warehouse/lake) optimize for historical querying — most governance in this chapter targets the analytics life cycle.
- **Data governance charter template**: a kickoff document to socialize vision/mission/goals and secure stakeholder buy-in before drafting the full policy.

## Mental Models
- **Life cycle as decomposition tool, not literal pipeline**: break the overwhelming task of "governing all our data" into per-phase sub-problems ("Rome wasn't built in a day").
- **Metadata management spans the whole life cycle, not one phase**: curation/discovery should start at creation and persist through storage — one explicit discrepancy the authors flag in their own framework diagram (Fig. 4-2).
- **Restrictive-regulation-cascades-everywhere strategy** (recommended): assume today's most restrictive regulation (e.g., CCPA) will eventually apply broadly, and build compliance proactively rather than reactively per-jurisdiction.

## Anti-patterns
- **Treating the data life cycle diagram as a literal left-to-right flow**: most data moves non-linearly between phases; rigid adherence misses real pipeline behavior.
- **Complying only with regulations required right now**: reactive-only compliance strategy creates repeated scramble and retrofit costs as new regulations (GDPR, CCPA, future ones) land.
- **Ignoring organizational culture when centralizing governance**: in blame-averse cultures, early-warning signals (like the two NASA engineers who flagged the metric-system discrepancy) get ignored — governance fails silently until disaster.
- **Skipping an audit/RACI structure for accountability**: "custodianship" without visible ownership leads to the same diffusion-of-responsibility problem described in Ch. 3/9.

## Code Examples
No code; the chapter gives a **13-step reference pipeline scenario** instead (policy/process, not syntax):
1. Configure ingestion pipeline (batch/stream) with staged buckets: Ingest (heavily restricted) → Released (processed) → Admin quarantine (needs review).
2. Scan/classify for sensitive info (PII).
3. Redact/obfuscate/anonymize as needed; capture resulting metadata (e.g., tokenization keys used).
4. Tag data with PII labels.
5. Assess data quality (missing values, key format).
6. Begin capturing provenance/lineage.
7. Encrypt data in transit between services.
8. Store in warehouse/lake, encrypted at rest; set up backup/recovery.
9. Attach business + technical metadata; catalog for discovery.
10. Capture audit trails throughout.
11. Enforce IAM-based access at every step.
12. Run analytics/visualization with added privacy/de-identification/anonymization tooling.
13. Archive when no longer needed in production; destroy at end of useful life.
- **What it demonstrates**: a concrete, ordered checklist translating the 6-phase life cycle into an actual cloud ingestion-to-destruction workflow.

## Reference Tables
No formal comparison table in this chapter; the **Data Governance Policy Structure** (list above) functions as the chapter's structured checklist/template.

## Worked Example
**NASA Mars Climate Orbiter (1999)**: After ~10 months of travel, the $125M orbiter was lost (burned up ~37 miles from Mars) because NASA used metric units while a partner contractor used Imperial (SI vs. non-SI) units — an inconsistency undetected until landing. Two engineers had actually noticed and reported the discrepancy, but management ignored the reports. Demonstrates both (a) the need for a documented data governance/compliance policy catching unit/schema inconsistencies early, and (b) how organizational culture (willingness to escalate and act on reported anomalies) is inseparable from technical governance.

**Nike Vaporfly 4% claim validated via Strava data**: Nike's own sponsored study (small dataset) was met with skepticism. The New York Times used publicly available, consented Strava data (280,000 marathon + 215,000 half-marathon results) as a "natural experiment," comparing same-runner/different-shoe and same-race/different-runner conditions, curating out outliers (weather, course difficulty) to approximate a controlled study. Confirmed the claim independently. Demonstrates proper information life cycle: open, consented data (Strava lets users delete/opt out) made available under a controlled environment enabled large-scale, credible secondary analysis — an example of "data as a product" governed well.

**OpenStreetMap (OSM)**: Crowdsourced (2M+ contributors) but succeeded because it established data standards early and enforced contributor adherence — demonstrating that even community-is-the-governance-body models work if standards exist and are enforced from the outset.

## Key Takeaways
1. Treat the 6-phase data life cycle as a decomposition framework, not a literal pipeline — tailor to your org, and not every dataset passes through every phase.
2. A written, living data governance policy (not just tools) is what makes governance auditable, consistent, and durable — it is the organization's "governance bible."
3. Start the operationalization process with a business case and quick wins; don't try to implement the entire framework at once.
4. Metadata management should begin at data creation and persist through the whole life cycle, not get bolted on at a single phase.
5. Revisit data-destruction guidelines every 12–18 months since retention/compliance rules change frequently.
6. Culture (willingness to report anomalies, and management's willingness to act on them) is as critical to avoiding disaster as the technical pipeline itself.
7. Favor proactive compliance with the most restrictive plausible future regulation over reactive, jurisdiction-by-jurisdiction compliance.

## Connects To
- **Ch 2**: Tools (lineage, classification, encryption, catalog) are the mechanics plugged into each life-cycle phase here.
- **Ch 3**: Roles/hats and RACI are reused directly for the policy's "Program structure" section.
- **Ch 5**: Data quality during the "Data Processing" phase is expanded into its own chapter.
- **Ch 6 / Ch 7 / Ch 8**: data-in-flight, protection, and monitoring chapters each deepen a specific life-cycle concern raised here.
- **GDPR / CCPA**: both are cited repeatedly as the regulatory forces shaping retention/destruction timing and policy scope.
