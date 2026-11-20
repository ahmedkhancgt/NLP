# Chapter 1: What Is Data Governance?

## Core Idea
Data governance is a data management function ensuring quality, integrity, security, and usability of data across its full life cycle, with the ultimate purpose of enhancing **trust** in data so stakeholders can use it for decision making.

## Frameworks Introduced
- **Three Pillars of Trust in Data**: Discoverability, Security, Accountability
  - When to use: as the top-level lens for designing any governance strategy.
  - How: Discoverability — technical metadata, lineage, business glossary, master data management. Security — regulatory compliance, sensitive data management, exfiltration prevention. Accountability — operating model for ownership across data domains, applied once discoverability+security are in place (then data can be treated "as a product").

- **Classification and Access Control Model**: public → external → internal → restricted
  - When to use: structuring any data class hierarchy (example: HR data).
  - How: define classification tiers; attach access-control policy (create/read/update/delete) per tier; policies can be static (role-based) or dynamic (manager sees only direct reports' salary).

- **Governance vs. Enablement vs. Security** (distinguishing framework)
  - Governance = making data accessible/discoverable + workflow for acquiring access.
  - Enablement = tooling to analyze/process data once accessible (requires metadata: lineage, owner, value proposition).
  - Security = mechanics preventing unauthorized access; governance relies on it but extends into policy and provable compliance.

- **Regulatory Compliance Checklist**: fine-grained access control, data retention/deletion, audit logging, sensitive data classes
  - How (fine-grained access): right-size container (table not whole dataset), right level of access (read/write/append/edit/delete, or redact/mask/tokenize), right duration (time-bound, justified access).

## Key Concepts
- **Governors**: group accountable for data/policy (e.g., HR dept.) — sets the governance policy.
- **Approvers / data stewards**: implement/operate the policy (e.g., IT), grant access changes.
- **Users / knowledge workers**: those whose access is governed.
- **Data democratization**: making data broadly accessible; governance enables this safely via metadata catalog + least-access + audit trail, rather than conflicting with it.
- **Blast radius**: scope of damage from a breach; minimized by narrow data collection and immutable cloud audit logs.
- **Data swamp**: a disorganized, un-curated data lake resulting from uncontrolled accumulation.
- **Reduced surface area**: governance benefit of cloud EDWs where data marts are views (no copies), so deletion/retention rules enforce only at the EDW.
- **Hybrid cloud**: systems spanning on-prem + cloud, or multiple clouds, requiring unified tooling for consistent governance.

## Mental Models
- **Risk/agility tradeoff**: governance is the practitioner's continuous trade-off between locking data down (security) and making it available (agility/insight generation).
- **Holistic governance**: tools + people + process is a *combination*, not alternatives — missing any one undermines the strategy (seeded here, expanded Ch. 2–3).
- **Default open vs. default closed**: organizations choose a baseline posture for business-data access; neither is intrinsically correct — it's a governance-board decision.

## Anti-patterns
- **Treating governance as purely a control/compliance exercise**: misses the business-value upside (innovation, new revenue, trust) — governance should be viewed as a strategic enabler.
- **Collecting more data than needed "just in case"**: increases blast radius (Capital One case) and retention/deletion liability.
- **Relying on perimeter/on-prem opacity as "security"**: unlogged on-prem systems let attackers cover tracks; cloud's immutable, non-modifiable logs are what let investigators catch the Capital One attacker.
- **Treating all enterprise data the same regardless of use case**: a furniture company's "deliver this couch" data vs. "market to this person" data need different policies even though it's the same person/record (foreshadows Ch. 2 use-case policies).

## Reference Tables
None (narrative chapter); classification table concept appears in Figure 1-3 (public/external/internal/restricted) — reproduced above under Key Concepts.

## Worked Example
**Capital One breach (2019)**: misconfigured web application firewall → attacker obtained temporary credentials → exfiltrated PII (names, SSNs, bank accounts, addresses) for 100M+ applicants, but not login credentials (data minimization luck). Attacker caught because files lived in a public-cloud storage bucket with immutable, unmodifiable access logs — on-prem, an attacker could typically erase the trail. Four lessons: (1) collect only what's purposeful and store the narrowest slice; (2) turn on org-level audit logging; (3) periodically audit open ports; (4) apply an extra layer (masking/tokenization via PII-detection AI) to sensitive fields even within authorized stores.

**US Coast Guard AVIS (Authoritative Vessel Identification Service)**: vessel-tracking data from AIS often mismatched (wrong names, duplicate MMSI IDs like "1" or "123456789") versus the amalgamated "ground truth" from other USCG systems. A human-review UI surfaced unresolved mismatches; over the pilot, 863 of 866 problem vessels were corrected by Sept. 2011 — eliminating nearly 100% of bad broadcasts out of a ~50,000-vessel global fleet. Demonstrates governance-driven data-quality improvement directly enabling the mission (maritime domain awareness, search-and-rescue).

## Key Takeaways
1. Governance's purpose is enhancing trust, not merely control — trust funds all downstream business value (Spotify's Discover Weekly only existed because EU users trusted Spotify's data handling).
2. Build governance around discoverability + security + accountability, in that order of dependency.
3. Public cloud structurally eases governance: data locality/sovereignty compliance, reduced-surface-area EDWs, ephemeral/serverless compute, immutable audit logs, granular labeling — most of which are cost-prohibitive on-prem.
4. Data democratization and governance are not opposed: a metadata catalog + least-access acquisition workflow + audit trail lets governance be the enabler of safe democratization.
5. Regulatory drivers (GDPR, CCPA, HIPAA) increasingly apply governance obligations outside traditionally regulated industries.
6. Manage three categories of data risk explicitly: theft, misuse (including purpose-drift), and corruption — each has distinct mitigations (trust-building, declared purpose/retention limits, and lineage/confidence tracking respectively).

## Connects To
- **Ch 2**: Tools (enterprise dictionary, data classes, IAM) operationalize the classification/access-control model introduced here.
- **Ch 3**: People/roles (governors, approvers/stewards, users) are expanded into the full "hats" taxonomy.
- **Ch 4**: The data life cycle chapter builds directly on "data governance from creation to destruction" stated here.
- **GDPR / CCPA**: external regulatory frameworks that motivate much of the chapter's urgency around new regulation.
