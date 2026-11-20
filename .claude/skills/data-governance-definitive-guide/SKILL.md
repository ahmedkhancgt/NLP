---
name: data-governance-definitive-guide
description: "Knowledge base from \"Data Governance: The Definitive Guide\" by Evren Eryurek, Uri Gilad, Valliappa Lakshmanan, Anita Kibunguchy-Grant, and Jessi Ashdown. Use when designing or implementing data governance — governance tools, people/process ingredients, data lifecycle governance, data quality, governance of data in flight, data protection, monitoring, or building a culture of data privacy and security."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Governance: The Definitive Guide
**Authors**: Evren Eryurek, Uri Gilad, Valliappa Lakshmanan, Anita Kibunguchy-Grant, Jessi Ashdown | **Pages**: ~254 | **Chapters**: 9 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch05`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Three Pillars of Trust in Data** (Ch 1): Discoverability, Security, Accountability. Discoverability requires metadata, lineage, and a business glossary. Security covers regulatory compliance and sensitive-data protection. Accountability (an operating model for data ownership) matters once the first two are in place — only then can data be treated "as a product."

**Governance vs. Enablement vs. Security** (Ch 1): Governance = making data accessible/discoverable + a workflow for acquiring access. Enablement = tooling to analyze data once accessible. Security = mechanics preventing unauthorized access. Governance relies on security but extends into provable policy compliance.

**Enterprise Dictionary → Data Class → Policy** (Ch 2): Infotypes (single-meaning fields like "email") group into data classes (PII, PHI, financial). Attach policy to the *class*, not individual containers — this is what lets governance scale and stay human-auditable.

**"Hats" vs. "Roles"** (Ch 3): Map governance work to tasks performed (hats: privacy tsar, data owner, data steward, business analyst, etc.), not idealized job titles — most organizations can't staff every textbook role separately, and one person wears several hats.

**Data Life Cycle (6 phases)** (Ch 4): Creation/Capture → Processing → Storage → Usage → Archiving → Destruction. Phases are logical dependencies, not a literal linear flow — not all data passes through every phase. A written, living **data governance policy** operationalizes this.

**Quality Beats Quantity** (Ch 5): Data quality = Accuracy + Completeness + Timeliness, always evaluated relative to the specific business use case. Fix quality issues as far upstream as possible — cost of fixing increases, sometimes to impossibility, the further downstream an error is caught.

**Lineage as Policy-Propagation Mechanism** (Ch 6): Column-level lineage lets a data-class policy (e.g., "PII restricted") automatically inherit to derivative tables/dashboards without manual re-tagging at every transformation step — this is the core technique for governing "data in flight."

**Defense in Depth** (Ch 7): No single layer (perimeter, encryption, or IAM alone) suffices. Combine physical security, network/transit encryption, VM/compute isolation, code-trust (binary authorization), data-level encryption/masking, zero-trust identity-based access, and continuous monitoring. Choose protection level via cost × likelihood of breach per asset class.

**Five Monitoring Domains** (Ch 8): Data Quality, Data Lineage, Compliance, Program Performance, Security — each needs its own baseline, thresholds, and alerting (with acknowledgment/escalation). Monitoring systems must run independently of what they monitor.

**Data Culture as the Integrating Layer** (Ch 9): Tools + people + process alone leave gaps; a data culture (explicit tenets, ongoing training/communication, top-down buy-in, named accountability, internal+external transparency) is what sustains governance durably and gives it regulatory agility.

**Cross-cutting principle — Regulatory Agility**: Assume today's most restrictive regulation (GDPR/CCPA-style) will eventually apply more broadly, and build structure/metadata proactively rather than waiting to retrofit after a law lands (Ch 4, Ch 9).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | What Is Data Governance? | Three pillars of trust (discoverability/security/accountability); classification & access-control model; governance vs. enablement vs. security |
| ch02 | Ingredients of Data Governance: Tools | Enterprise dictionary → data class → policy; partial-access spectrum; envelope encryption (DEK/KEK); authentication factor model |
| ch03 | Ingredients of Data Governance: People and Processes | Hats vs. roles; company-type governance matrix; data segregation strategies (on-prem/cloud, zoned, line-of-business) |
| ch04 | Data Governance over a Data Life Cycle | 6-phase data life cycle; Data Management Plan (5 guidances); data governance policy structure; step-by-step operationalization |
| ch05 | Improving Data Quality | Accuracy/completeness/timeliness model; prioritization/annotation/profiling; data quality scorecard |
| ch06 | Governance of Data in Flight | ETL/ELT stages; lineage granularity levels; lineage-driven policy inheritance |
| ch07 | Data Protection | Defense in depth; risk-based protection level selection; zero-trust model; IAM triad; differential privacy (k-anonymity, noise) |
| ch08 | Monitoring | Four monitoring functions (alert/account/audit/comply); five monitoring domains; performance management framework |
| ch09 | Building a Culture of Data Privacy and Security | Data culture definition; top-down buy-in chain; data culture intentionality process; agility framework; incident-handling accountability model |

## Topic Index

- **Access control (fine-grained/column-level)** → ch02, ch06, ch07
- **Agility (regulatory)** → ch04, ch09
- **Audit logging** → ch01, ch07, ch08
- **Binary authorization / code trust** → ch07
- **CCPA / GDPR compliance** → ch01, ch02, ch04, ch09
- **Crypto-shredding** → ch07
- **Data catalog** → ch02
- **Data classes / classification** → ch01, ch02, ch07
- **Data culture** → ch03, ch09
- **Data governance policy** → ch04
- **Data life cycle** → ch04
- **Data quality** → ch02, ch05, ch08
- **Data segregation strategies** → ch03
- **Differential privacy** → ch03 (COVID case), ch07
- **Encryption / key management (DEK/KEK, CMEK)** → ch02, ch07
- **Hats (roles)** → ch01, ch03
- **IAM / zero-trust** → ch02, ch07
- **Lineage / provenance** → ch02, ch05, ch06, ch07, ch08
- **Monitoring (quality/lineage/compliance/security/program)** → ch08
- **VPC-SC / data exfiltration** → ch07

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — governance techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 9 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive. Appendix A (Google's internal data governance) and Appendix B (additional resources), while present in the source, are referenced only informally within ch09's "Connects To" section and are not separately indexed as chapters.
