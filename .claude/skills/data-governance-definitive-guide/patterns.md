# Governance Patterns

## Enterprise Dictionary + Data Class Hierarchy
**When to use**: Foundational first step for any governance program; before writing any access/retention policy.
**How**: Catalog infotypes (single-meaning fields) → group into data classes (e.g., PII, PHI, financial) → attach policies (access, retention, locality) to the class, not individual containers.
**Trade-offs**: Scales well and is human-auditable, but requires sustained manual curation effort (data stewardship) that most orgs under-resource.

## Use-Case-Aware Access Policy
**When to use**: When the same data/role combination is appropriate for one purpose but not another (e.g., shipping address for delivery vs. marketing).
**How**: Overlay a declared "purpose" parameter on top of role-based access; require purpose justification at request time; route through steward approval.
**Trade-offs**: More precise and compliant, but adds friction/complexity to the access workflow versus flat RBAC.

## Column/Attribute-Level (Label-Based) Access Control
**When to use**: A table mixes broadly useful analytics fields with a small number of sensitive columns (e.g., transaction logs with customer name).
**How**: Tag sensitive columns; apply redaction/hashing/masking at the column level so most users see the full table minus sensitive fields; grant unredacted access only to authorized roles.
**Trade-offs**: Avoids both full-access overexposure and the headcount burden of manually rewriting redacted table copies; requires upfront column classification effort.

## Lineage-Driven Policy Inheritance
**When to use**: Propagating a data-class policy (e.g., "PII access restricted") automatically to derivative tables/dashboards.
**How**: Tag source columns at origin; track column-level lineage through joins/transforms; inherit identical policy when a column is copied unchanged, different/no policy when nullified/transformed.
**Trade-offs**: Scales governance without manual re-tagging at every step, but requires investment in automated (not manual/after-the-fact) lineage capture to be reliable.

## Data Segregation by Storage Tier (On-Prem/Cloud or Zoned Lake)
**When to use**: Limiting blast radius of a leak while still enabling analytics.
**How**: Separate curated (cloud, broad access) from uncurated (on-prem or restricted zone, narrow access) data; or create tiered zones (insights zone vs. raw zone) within one cloud environment.
**Trade-offs**: Reduces breach impact and simplifies compliance for the curated layer, but on-prem/cloud split blocks cross-storage analytics; zoned-single-cloud approach is easier to manage but still carries cloud leak exposure.

## Data Segregation by Line of Business
**When to use**: Establishing clear accountability/ownership for governance tasks in large organizations.
**How**: Assign dedicated data owner/steward/business-analyst hats per line of business; each line manages its own pipelines, enrichment, and access controls.
**Trade-offs**: Clear accountability and faster enrichment per domain, but risks siloing data and blocking cross-line analytics unless pipelines are deliberately designed for cross-feed.

## Envelope Encryption (DEK/KEK Key Wrapping)
**When to use**: Any cloud-native encryption-at-rest implementation, especially where customer-managed keys (CMEK) are required for compliance.
**How**: Data Encryption Key (DEK) encrypts data directly and is stored wrapped in table metadata; Key Encryption Key (KEK) lives only in a KMS vault and unwraps the DEK on authorized request; DEK exists only transiently in memory.
**Trade-offs**: Minimizes blast radius of a single leaked key and supports crypto-shredding, but adds KMS dependency and latency to every decrypt operation.

## Crypto-Shredding
**When to use**: Fulfilling "right to be forgotten" / deletion requests quickly and provably across all tables, including backups.
**How**: Assign a unique encryption key per user ID at write time; encrypt all that user's sensitive fields with it; delete the key (not the records) to render all their data permanently unreadable everywhere.
**Trade-offs**: Fast, provable, and backup-safe deletion, but requires per-user key management infrastructure designed in from the start — hard to retrofit.

## Zero-Trust Access Model
**When to use**: Replacing perimeter-based security, especially as mobile/cloud access erodes the traditional network boundary.
**How**: Require managed+certificated devices, centralized 2FA user/group auth, an "unprivileged network" for managed devices, encrypting access proxies, and continuous device/user context checks (endpoint verification).
**Trade-offs**: Much stronger against insider/compromised-perimeter threats, but requires significant IAM/device-management infrastructure investment.

## VPC Service Controls (Perimeter Around Cloud Services)
**When to use**: Mitigating data exfiltration from cloud-native data lakes/warehouses, even against compromised valid credentials.
**How**: Define a perimeter around an explicit set of services/data; restrict access to clients within authorized VPC networks; use dry-run mode to monitor without blocking first.
**Trade-offs**: Strong exfiltration containment, but must be paired with Cloud IAM for metadata-level protection (VPC-SC primarily controls data movement, not all metadata access).

## Differential Privacy (k-anonymity + Noise Addition)
**When to use**: Sharing aggregate statistics derived from highly sensitive underlying individual records without re-identification risk.
**How**: Ensure aggregates represent ≥k individuals (k-anonymity); add statistically insignificant noise to discrete-value fields (age, gender) to prevent precise reconstruction via cross-referencing multiple aggregates.
**Trade-offs**: Preserves aggregate utility while protecting individuals, but requires careful tuning (too much noise destroys utility; too little allows re-identification via cross-matching).

## Data Quality Scorecard + Source Ranking
**When to use**: Before incorporating any new data source into an analytics pipeline, and when merging/joining datasets of differing trust levels.
**How**: Document origin, accuracy, completeness, timeliness, and owner per source; assign a trust rank to each source to deterministically resolve conflicting field values on merge.
**Trade-offs**: Prevents low-quality sources from silently degrading high-quality aggregates, but requires upfront profiling investment per source.

## Upstream-First Data Quality Remediation
**When to use**: Any data pipeline where cleanup cost increases the further downstream an error is caught.
**How**: Validate at extraction (while source context is still available); profile for min/max/missing/outliers before transformation; propagate corrections back to the original source record, not just derivatives.
**Trade-offs**: Much cheaper and more reliable than late-stage cleanup, but requires pipeline design discipline and may require revisiting/re-architecting legacy ETL that defers validation.

## Culture-First Regulatory Agility
**When to use**: Preparing for regulatory changes (e.g., CCPA-style residency-based deletion rights) before they're mandated.
**How**: Proactively structure metadata/tags to capture dimensions regulators might later require (e.g., customer residence, not just transaction location); reinforce via culture (tenets, training) rather than one-off technical patches.
**Trade-offs**: Significantly cheaper than retrofitting after a law lands, but requires anticipatory investment with no immediate ROI, which is hard to justify without strong leadership buy-in.

## Named Accountability for Incident Response
**When to use**: Designing who is responsible when a governance failure (e.g., improper PII access) occurs.
**How**: Explicitly name an accountable person/group per governance domain in role descriptions, with defined consequences for failure; avoid "everyone is responsible."
**Trade-offs**: Clarifies ownership and improves response speed, but requires organizational commitment to enforce consequences, which can create friction if not culturally supported.
