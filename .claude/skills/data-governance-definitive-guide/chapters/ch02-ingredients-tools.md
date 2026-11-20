# Chapter 2: Ingredients of Data Governance: Tools

## Core Idea
Governance is operationalized through a specific toolchain: an enterprise dictionary/policy book defining data classes, automated classification, a data catalog, quality/lineage tracking, encryption/key management, retention controls, and IAM-based access workflows.

## Frameworks Introduced
- **Enterprise Dictionary → Data Classes → Policies hierarchy**
  - When to use: foundational step before any policy automation.
  - How: leaf nodes = infotypes (e.g., "email," "salary"); root nodes = data classes/sensitivity classification (e.g., PII, PHI, "restricted"). Policies attach to the *class*, not individual containers ("Analysts cannot access PII" vs. "Analysts cannot access column #3").

- **Enterprise Policy Book**
  - When to use: documenting what data classes exist, how each may be processed, and proving compliance to regulators.
  - How: specify who can access a class, retention duration, residency/locality rules, allowed processing (analytics/ML OK or not), and other org-specific considerations.

- **Partial-Access Spectrum**: zero access → format-only → digit-count-only → tokenized → full access
  - When to use: designing graduated access instead of binary yes/no.
  - How: redact (last 4 digits only), coarsen (GPS → city/country), tokenize (symmetric/reversible encryption preserving uniqueness without exposing value).

- **Envelope Encryption (DEK/KEK)**
  - When to use: any cloud-native encryption-at-rest implementation.
  - How: Data Encryption Key (DEK) encrypts the data directly, stored wrapped in table metadata. Key Encryption Key (KEK) lives only in the KMS and unwraps the DEK on request; DEK exists only in memory, never persisted; KEK never leaves the KMS vault.

- **Authentication Factor Model**: something you know / something you have / something you are / additional context
  - When to use: designing IAM authentication strength.
  - How: password (know) + device/token (have) + biometric (are) + contextual signals (corporate hardware only, working-hours only, VPN-only).

## Key Concepts
- **Infotype**: a single-meaning data element (e.g., "email address," "salary amount") — the atomic unit of the enterprise dictionary.
- **Data class**: a group of infotypes sharing common policy treatment (e.g., PII groups phone, IMEI, IP address).
- **Data catalog**: cross-storage-system metadata index enabling faceted search ("type:table AND class:PII AND env:production") even when the underlying data itself is access-restricted.
- **Data lineage**: the traceable origin + transformation history of data; needed for debugging, tainted-data tracking, and explainability of automated decisions.
- **Data encryption key (DEK) vs. key encryption key (KEK)**: see envelope encryption above.
- **CMEK / CSEK**: customer-managed / customer-supplied encryption keys — customer retains control over keys versus default cloud-managed encryption.
- **IAM (Identity and Access Management)**: role-based, scalable (millions of users/sec) system binding identities to permissions with contextual signals (IP, device, time, use case).
- **Data acquisition workflow**: analyst searches catalog → identifies dataset → requests access with justification → steward reviews/approves → access provisioned via native warehouse controls.

## Mental Models
- **Policy attaches to meaning, not location**: governing by data class scales; governing by individual container/column does not.
- **Metadata has its own, often *weaker or different*, governance boundary than the underlying data** — e.g., you may be allowed to know a table exists (and request access) even if you can't read it; conversely, a call-metadata list can be *more* sensitive than the call content itself.
- **Automation vs. manual stewardship tradeoff**: classification can trigger on ingest (preferred, efficient) or run periodically as batch jobs (necessary for legacy/federated data) — but outlier/anomaly cleanup still needs human judgment to avoid introducing bias.

## Anti-patterns
- **Column-level ("Analysts cannot access column #3") instead of class-level policy**: doesn't scale and is unreadable by humans reviewing compliance.
- **Automated outlier elimination without business-context review**: risks silently introducing bias or discarding legitimate rare-but-real signals.
- **Dangling access grants with no expiration/justification**: regulators specifically probe "who has access to what" — ungoverned duration is a red flag.
- **Storing the data encryption key persistently/on disk**: defeats the purpose of envelope encryption; DEK should exist only transiently in memory.

## Code Examples
No executable code; the chapter instead specifies a step-by-step **key management protocol**:
1. User/process requests data, supplies the wrapped-key ID to the warehouse (e.g., BigQuery).
2. Warehouse retrieves the protected (wrapped) DEK from table metadata and sends it to the KMS.
3. KMS unwraps the DEK using the KEK (KEK never leaves the KMS vault).
4. Warehouse uses the unwrapped DEK to decrypt/access data, then discards it (never persisted).
- **What it demonstrates**: separation of duties so that no single compromised component (disk, warehouse, or KMS alone) exposes plaintext data.

## Reference Tables
- **Data class hierarchy example** (Fig 2-2 concept): PII {IMEI, phone number, IP address} + PHI → grouped under "restricted data" → stricter policies apply to the whole "restricted" bucket.
- **Sample access-policy dimensions** (from "User Authorization"): read / edit metadata / update content / copy-export / ETL-transform — each independently grantable.

## Worked Example
**Danish taxi company (GDPR Article 5 case)**: Company anonymized ride records after 2 years by deleting passenger *name* only — but retained geolocation (ride start/end) and passenger *phone number*. Regulators found that phone number functioned as a de facto unique identifier, making "anonymization" ineffective; the company was fined for non-compliance with GDPR Art. 5(1)(e) (no longer than necessary retention) despite having a stated retention policy. Lesson: a retention/anonymization policy must be validated against its *stated business goal* — partial field deletion ≠ true anonymization if other fields are re-identifying.

**Toy Story 2 deletion incident (Pixar, 1998)**: A 10-character erroneous command deleted ~90% of the film's assets; about a week of work was permanently lost. Illustrates that data-retention/backup governance isn't only about sensitive-data compliance — it also protects against catastrophic accidental loss of *any* valuable data class.

## Key Takeaways
1. Build a central enterprise dictionary before building policy — you cannot govern data you haven't named/classified.
2. Attach policy to the data *class*, not individual tables/columns, for human-understandable and scalable governance.
3. Prefer ingest-time automated classification over periodic batch scans where feasible.
4. Design access as a spectrum (redact/mask/tokenize/full), not binary, to balance analytics utility with protection.
5. Use native cloud KMS/envelope encryption (DEK+KEK split) rather than rolling your own key handling.
6. A data retention/deletion policy must be tested against ALL identifying fields retained, not just the obviously named ones (taxi case).
7. Treat accidental/catastrophic data loss as a governance concern equal to breach/leak risk — back up broadly, not just "sensitive" classes.

## Connects To
- **Ch 1**: Extends the classification/access-control model with concrete tooling.
- **Ch 5**: Data quality/profiling tools referenced here (Dataprep, Stitch) are expanded.
- **Ch 7**: Encryption/KMS mechanics here are deepened into full data-protection architecture.
- **GDPR Article 5**: directly invoked in the Danish taxi case study (data minimization, storage limitation).
