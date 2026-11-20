# Chapter 7: Data Protection

## Core Idea
Data protection requires **defense in depth** across physical, network, VM, application, and data layers simultaneously — no single perimeter is sufficient — combined with risk-calibrated classification, encryption/key management, zero-trust IAM, and continuous agility against evolving threats.

## Frameworks Introduced
- **Defense in Depth Layering**
  - When to use: as the top-level checklist for any protection plan.
  - How: physical data-center security → network security (in-transit encryption, no reliance on perimeter alone) → virtual-machine/compute isolation → application/code trust (binary authorization) → data-level encryption/masking → identity-based access control (zero trust) → monitoring/detection.

- **Risk-Based Protection Level Selection**
  - When to use: deciding how much protection (and cost) to apply to a given data asset.
  - How: catalog potential breach types for the asset → estimate cost (direct + indirect: downtime, legal exposure, goodwill, stakeholder impact) and likelihood per protection level → choose protection level via cost/risk tradeoff (can simplify to high/medium/low when granular estimates are impractical).

- **Zero-Trust Model**
  - When to use: replacing perimeter-based security, especially as mobile/cloud erode the traditional network boundary.
  - How: only enterprise-managed, uniquely certificated devices access corporate apps; centralized user+group database tightly integrated with HR; 2FA via centralized auth portal; an "unprivileged network" resembling external network for managed devices; apps exposed via encrypting access proxy; access-control manager cross-checks device+user context continuously (endpoint verification).

- **Identity and Access Management (IAM) Triad**: Authentication (who) → Authorization (what) → Auditing (what happened)
  - Authentication: API keys (app-only, no user identity) < OAuth 2.0 access tokens (2FA, user-delegated) < service-account keys (both identity + credential).
  - Authorization: roles = sets of permissions; prefer groups over per-user grants; use custom roles to *subtract* permissions from over-broad predefined roles (least privilege); hierarchical policies propagate from org → project → resource, with lower levels unable to override higher-level denies.
  - Context-Aware Access: combines Identity-Aware Proxy (IAP) + Cloud IAM + Access Context Manager + Endpoint Verification to grant access based on device/network/patch-state context, not just identity.

- **Differential Privacy Techniques**: k-anonymity, statistical noise addition (l-diversity, t-distance)
  - When to use: sharing aggregate data derived from highly personal/sensitive underlying records without re-identification risk.
  - How: k-anonymity ensures aggregates represent ≥k individuals; noise-addition skews small-category aggregates just enough to prevent precise re-identification while preserving overall utility.

## Key Concepts
- **Multi-tenancy risk**: moving from on-prem (physical/network perimeter control) to shared cloud infrastructure removes the traditional perimeter assumption — don't assume isolation; apply cloud-native security, not lift-and-shift on-prem assumptions.
- **Data exfiltration**: authorized party extracts and shares/moves data to unauthorized destinations — may be malicious, accidental, or via a compromised account.
- **VPC Service Controls (VPC-SC)**: creates perimeters around specified cloud services/data so resources are inaccessible from outside even with valid credentials — directly mitigates exfiltration from cloud-native data lakes/warehouses; supports dry-run mode and restricted VIPs.
- **Binary authorization / attestations**: a signer (human or automated build/test system) digitally signs a container image descriptor, creating an attestation; deployment requires verified attestations before promotion.
- **Crypto-shredding**: assigning a unique encryption key per user ID; "deleting" a user's data is accomplished simply by deleting their key, instantly making all their records unreadable everywhere (including backups) without touching the records themselves.
- **Access Transparency vs. Cloud Audit Logs**: Cloud Audit Logs record actions by your own org's members; Access Transparency records actions taken by the *cloud provider's* personnel accessing your content — both are needed together for a complete picture.
- **Benford's Law**: statistical check on leading-digit distribution in numeric data, useful for detecting fraud/corruption at the aggregate level.

## Mental Models
- **"Security by obscurity" is not a strategy**: cloud's increased attack-surface visibility (more publicized breaches) is offset by dedicated world-class security teams (e.g., Google's Project Zero) — don't equate visibility with increased actual risk.
- **Blast radius applies to both malicious attacks and honest mistakes**: employee error (e.g., accidentally sharing data publicly) carries the same exfiltration-style risk profile as a deliberate attacker and should be mitigated the same way (compartmentalization, fine-grained ACLs, time-bound access).
- **Protection level must match aggregate/join sensitivity, not just raw data**: when joining two protected datasets, the resulting access level is typically the *intersection* of the two datasets' authorization permissions — joins and aggregates can both increase or alter sensitivity.

## Anti-patterns
- **Lift-and-shift of on-prem security tooling to the cloud**: gives up elasticity, democratization, and lower operating cost benefits that cloud-native security policies would otherwise provide.
- **Perimeter-only network security**: no perimeter is 100% safe, and insider/compromised-application risk exists regardless — must be paired with encryption, masking, and deletion-on-schedule.
- **Role/permission creep**: granting broad predefined roles "to be safe" leads to massive unused-permission surface (e.g., a user granted BigQuery Admin but using only 5 of 31 granted permissions) — periodic Security Health Analytics scans should catch and correct this.
- **Treating physical/paper records as out of scope for data protection**: the 2011 Texas SAIC/TRICARE tape-theft case shows un-digitized data needs equally rigorous physical safeguards (locked transport, encryption even on backup tapes).
- **Running outdated/unpatched medical device OS software**: healthcare's prevalence of legacy OS devices (long device life cycles) makes it an outsized hacker target; failing to patch is a direct, well-documented root cause.

## Code Examples
No executable code; the chapter documents a **security architecture pattern** and an **envelope-encryption key-wrapping protocol** (detailed, extending Ch. 2):
- Every CMEK-protected table has a wrapped key in its metadata; access requests go to Cloud KMS to unwrap the table key, which is then used to unwrap per-record/file keys; the master key never leaves KMS; deleting the KMS key makes all wrapped keys permanently unwrappable (supports crypto-shredding).
- **What it demonstrates**: layered key-wrapping minimizes blast radius if any single wrapped key leaks (an unwrapped file key can't read other files; an unwrapped table key still requires passing access-control checks to unwrap further file keys).

## Reference Tables
- **Table 7-1 (Data Exfiltration Vectors and Mitigations)**: e.g., Vector = business email/mobile transmission to untrusted third parties; Data at risk = org emails, calendars, databases, planning docs, source code; Mitigation = limit transmission volume/frequency, audit email metadata, scan content for threats, alert on insecure channel attempts.
- **Recommended Org-Level Exfiltration Mitigations** (checklist): minimize blast radius via compartmentalization; fine-grained, time-bound ACLs; simulated/tokenized data for dev teams; immutable logging; restrict/monitor ingress-egress via IAM + bastion hosts; baseline normal data flows to detect anomalies.

## Worked Example
**Equifax breach (2017)**: Cybercriminals accessed personal data of ~145.5M customers (names, SSNs, birth dates, addresses, driver's license numbers); at least 209,000 customers' credit card credentials were also taken. Total resolution cost estimated at $32B+; stock fell >30% after disclosure, and even *unaffected* adjacent competitor firms saw ~9% stock drops from industrywide confidence loss — illustrating that breach costs extend beyond the breached company itself.

**NHS WannaCry ransomware attack (2017)**: Affected 60+ UK NHS trusts, spreading to 200,000+ computer systems across 150 countries — illustrates systemic risk from outdated OS/legacy medical device software, a healthcare-specific vulnerability pattern discussed at length.

**SAIC/TRICARE physical tape theft (Texas, 2011)**: Backup tapes with electronic health records for 4.6M active/retired military personnel were stolen from a contractor's car. Financial data was separately stored and not compromised (a data-minimization/segmentation win), but medical record data (SSNs, diagnoses, lab reports) was exposed. Demonstrates that physical transport/custody of data (not just digital storage/transit) must be governed with the same rigor — encryption, locked custody, chain-of-custody tracking.

## Key Takeaways
1. Apply defense in depth: no single layer (perimeter, encryption, or IAM alone) is sufficient; combine physical, network, compute, code-trust, data, and identity controls.
2. Classify assets and calculate cost×likelihood per protection level before choosing a protection tier — don't apply uniform protection regardless of risk.
3. Adopt zero-trust (device + user + context-based access) rather than relying on network perimeter, especially in multi-tenant cloud environments.
4. Use VPC-SC (or equivalent perimeter-around-services controls) specifically to mitigate cloud data exfiltration even from compromised credentials.
5. Use crypto-shredding (per-user encryption keys) to make "right to be forgotten" deletion requests fast and provably complete across all tables/backups.
6. Periodically run security health scans to catch and trim over-broad role grants (permission creep).
7. Physical data custody (paper, tapes, portable devices) needs the same governance rigor as digital data-at-rest/in-transit — encrypt portable devices and enforce strict BYOD/MDM policy.
8. Data protection must be agile — revisit policies continually as threats, business processes, and regulations evolve; it is never "done."

## Connects To
- **Ch 2**: Envelope encryption (DEK/KEK) introduced there is extended into full CMEK/crypto-shredding mechanics here.
- **Ch 5 / Ch 6**: Lineage and quality checks (e.g., Benford's Law) are reused here as fraud/breach detection tools.
- **Ch 8**: Security Health Analytics, Event Threat Detection, and lineage monitoring are the ongoing operational counterparts to this chapter's planning framework.
- **HIPAA / GDPR / CCPA / NIST SP 800-61**: referenced directly for healthcare, EU personal data, California consumer privacy, and incident-management guidance respectively.
