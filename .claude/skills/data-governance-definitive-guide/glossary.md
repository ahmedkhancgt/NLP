# Glossary

**Access Transparency** — Cloud-provider logs of actions taken by the *provider's own personnel* on your content, distinct from Cloud Audit Logs which cover your own org's users (Ch 7).

**Accuracy** — Whether captured data is actually correct (no entry errors, duplicates) (Ch 5).

**ADH (Ads Data Hub)** — Google tool joining advertiser data with Google's own data via aggregation thresholds, static checks, and differential-privacy checks, without exposing individual users (Appendix A / Ch 9 context).

**Annotation (data quality)** — Standardized attachment of quality signals ("vetted"/"not vetted") to datasets, optionally crowdsourced (Ch 5).

**Approver / data steward** — Role implementing governance policy changes (e.g., adding users to permitted groups) (Ch 1, Ch 3).

**Benford's Law** — Statistical check on leading-digit distribution used to detect fraud/corruption in aggregated numeric data (Ch 7).

**Binary authorization / attestation** — Policy model requiring signed, verified container-image attestations before deployment to sensitive systems (Ch 7).

**Blast radius** — Scope of potential damage from a data leak or breach; minimized via narrow collection, compartmentalization, and immutable logs (Ch 1, Ch 7).

**Column-level (granular) access control** — Locking down specific sensitive columns while leaving the rest of a table accessible (Ch 6).

**Completeness** — Whether all expected fields/records are present with no missing values (Ch 5).

**CMEK / CSEK** — Customer-managed / customer-supplied encryption keys, giving the customer control over encryption keys beyond default provider-managed encryption (Ch 2, Ch 7).

**Context-Aware Access** — Fine-grained access control based on device, network, and user context (combines IAP, Cloud IAM, Access Context Manager, Endpoint Verification) (Ch 7).

**Crypto-shredding** — Deleting a user's unique encryption key to instantly and provably render all their records unreadable everywhere, including backups (Ch 7).

**Data acquisition chain** — Gathering → landing-zone normalization → analytics workload; errors can compound at each step (Ch 5).

**Data catalog** — Cross-storage-system metadata index enabling faceted search even when the underlying data is access-restricted (Ch 2).

**Data class** — A group of infotypes sharing common policy treatment (e.g., PII groups phone, IMEI, IP address) (Ch 2).

**Data culture** — The set of values, goals, attitudes, and practices around data within an organization; the integrating layer beyond tools/people/process (Ch 9).

**Data exfiltration** — Authorized party extracting data and sharing/moving it to unauthorized destinations, maliciously, accidentally, or via compromised credentials (Ch 7).

**Data governance policy** — A living, documented set of guidelines (quality, access, security, privacy, usage rules + roles) for managing an org's data assets (Ch 4).

**Data iceberg** — In large companies, a small known/curated data layer sits above a much larger unknown/ungoverned mass (Ch 3).

**Data lake tier / zone** — Layered cloud storage areas separating curated ("insights zone," broad access) from uncurated raw data (restricted access) (Ch 3).

**Data lineage (provenance)** — Recorded path data takes through creation, transformation, and import; answers "why does this dataset exist" and "where did it come from" (Ch 2, Ch 6).

**Data Life Cycle Management (DLM)** — Comprehensive, policy-based approach managing data flow from creation to obsolescence/purge (Ch 4).

**Data Management Plan (DMP)** — Document defining how data will be managed, described, stored, and protected across its life cycle (Ch 4).

**Data minimization** — GDPR Article 5(1)(c) principle: collect/retain only data necessary for the stated purpose (Ch 2).

**Data steward** — Hat performing data categorization/classification/labeling; described as highly manual and the most commonly incomplete governance task (Ch 3).

**Data swamp** — A disorganized, un-curated data lake resulting from uncontrolled accumulation (Ch 1).

**Defense in depth** — Layered protection across physical, network, VM, code-trust, data, and identity levels; no single layer suffices (Ch 7).

**Differential privacy** — Techniques (k-anonymity, statistical noise) for sharing aggregate data derived from sensitive records without re-identification (Ch 7).

**DEK / KEK (Data/Key Encryption Key)** — Envelope encryption: DEK encrypts data directly; KEK (kept only in the KMS) wraps/unwraps the DEK (Ch 2, Ch 7).

**Discoverability, Security, Accountability** — The three pillars required to build trust in data (Ch 1).

**Enterprise dictionary** — Agreed-upon repository of infotypes (singular-meaning data elements) used across an organization (Ch 2).

**Enterprise policy book** — Document specifying data classes, their allowed processing, and proof-of-compliance evidence for regulators (Ch 2).

**Envelope encryption** — See DEK/KEK.

**Governor** — Role/group accountable for setting data policy (e.g., HR department for HR data) (Ch 1).

**Hat (vs. role)** — A set of governance tasks performed, independent of job title; individuals commonly wear several hats (Ch 3).

**Infotype** — A single-meaning data element (e.g., "email address," "salary amount") forming the leaf node of a data class hierarchy (Ch 2).

**k-anonymity** — Differential-privacy technique ensuring aggregates represent at least k individuals (Ch 7).

**Lineage granularity** — Table-level, column-level, row-level, or dataset-level provenance tracking, chosen based on governance use case (Ch 6).

**Natural experiment** — A dataset where treatment/control groups arise organically rather than by researcher assignment, enabling valid comparison without a controlled study (Ch 4, Ch 5).

**Partial-access spectrum** — Graduated access model from zero access to full access, via redaction, coarsening, and tokenization (Ch 2).

**Privacy tsar** — Google-internal term for the governor overseeing overall governance strategy and regulatory compliance (Ch 3).

**Quality signal** — A monitored metric (completeness, accuracy, duplicates, conformity, anomalies) that triggers an alert when it crosses a governance-committee-defined threshold (Ch 8).

**RACI chart** — Responsible, Accountable, Consulted, Informed — maps roles to governance tasks (Ch 4).

**Reduced surface area** — Cloud governance benefit where data marts are views (not copies) of an EDW, so deletion/retention rules need enforcing only at the EDW (Ch 1).

**Scorecard (data quality)** — Documented origin, accuracy, completeness, timeliness, and ownership info for a data source, guiding pipeline design decisions (Ch 5).

**Tribal knowledge** — Undocumented, word-of-mouth knowledge about dataset trustworthiness; fragile and non-scalable (Ch 3, Ch 5).

**Use-case-aware policy** — Access policy parameterized by *purpose* of use, not just role/data class (e.g., shipping vs. marketing use of an address) (Ch 2, Ch 3).

**VPC Service Controls (VPC-SC)** — Perimeter around specified cloud services/data, blocking access from outside even with valid credentials, mitigating exfiltration (Ch 7).

**Zero-trust model** — Security model where access depends on device/user credentials and context rather than network perimeter/location (Ch 7).
