# Cheatsheet

## Choosing a Protection Level
1. Catalog potential breach types for the asset (read, bulk read, modify, delete).
2. Estimate cost (direct + indirect: downtime, legal, goodwill, stakeholder impact) × likelihood for each protection option.
3. Pick protection level via cost/risk tradeoff; use high/medium/low if granular estimates are impractical.
4. Default: narrowest container + minimal duration + least privilege, expand only on justified request.

## Access Control Decision Tree
- Need full plaintext for a business task? → grant narrow, time-bound, justified role-based access.
- Need aggregate/statistical use only? → use k-anonymity / noise / coarsening, no direct table grant.
- Need to distinguish records without seeing values? → tokenize (reversible, preserves uniqueness).
- Need partial visibility (last 4 digits, city/country)? → redact/coarsen at the column level.
- Mixed sensitive + non-sensitive columns in one table? → column-level/label-based security, not a full rewrite.

## Data-in-Flight vs. Data-at-Rest Controls
| Concern | In Flight (Ch 6) | At Rest (Ch 2, Ch 7) |
|---|---|---|
| Primary control | Lineage tracking, validation at extraction | Encryption (DEK/KEK), access control |
| Policy propagation | Column-level lineage inheritance | Static classification/tagging |
| Granularity | Table/column/row/dataset-level lineage | Table/column-level encryption + masking |
| Key risk | Information loss during transform/aggregation | Breach of stored plaintext |
| Primary tool | API/job audit logs, lineage graphs | KMS, CMEK, crypto-shredding |

## On-Prem vs. Cloud Governance Tradeoffs
| Factor | On-Prem | Public Cloud |
|---|---|---|
| Audit log tamper-resistance | Attacker can often erase trail | Logs typically immutable |
| Data locality/sovereignty | Requires building regional data centers (costly) | Native multi-region controls |
| Compute/storage separation | Rare, expensive | Native (ephemeral compute, serverless) |
| Perceived vs. actual risk | "Security by obscurity" (false comfort) | More visible breaches, but dedicated security teams |
| Blast radius if leaked | Internal only | Potentially public — mitigate with VPC-SC |

## Company-Type Quick Reference (Ch 3)
- **Legacy**: fragmented enterprise dictionaries per system/line of business; prioritize centralizing before migrating.
- **Cloud-native**: better baseline governance but still risks multi-cloud/multi-storage siloing.
- **Retail**: third-party data ingestion + use-case-aware policy (delivery vs. marketing) is the key complication.
- **Highly regulated** (finance/healthcare): most mature processes, but slow to adopt new/beta tools (compliance gate).
- **Small (<1000 employees)**: lower risk surface, easier access control, less data proliferation.
- **Large (>1000 employees)**: data iceberg risk, acquisition-driven silo sprawl, RBAC alone insufficient.

## Data Quality Red Flags (Tells & Smells)
- Sudden spike at the edge of an otherwise normal bell curve → investigate, don't auto-discard.
- Bimodal distribution where one is expected → likely two distinct populations, not noise.
- "Null" vs. "0" used interchangeably across merged sources → reconcile explicitly before joining.
- A dataset relies on "tribal knowledge" for trust assessment → replace with scorecard/annotation before institutional memory is lost (esp. pre-acquisition).
- Model performance degrades after deployment → check for positive feedback loop (predictions becoming future training inputs).

## Monitoring Thresholds & Defaults
- Establish a **quality baseline** before setting any alert threshold — never alert against an undefined "normal."
- Alerts must have: acknowledgment requirement + escalation/re-notification if unactioned within SLA.
- Revisit data-destruction/retention guidelines every **12–18 months** (regulations change).
- Run **Security Health Analytics** scans periodically to catch permission creep (e.g., a role granting 31 permissions but only 5 used).
- Monitoring systems must run independently of the production systems they monitor (own failover).

## Regulatory Posture Decision
- **Reactive** (comply only when required): lower upfront cost, high retrofit risk when new regulation lands.
- **Proactive** (assume strictest regulation will eventually apply everywhere): higher upfront cost, much lower retrofit/compliance-scramble risk — **recommended default** per the authors.

## Choosing Data Segregation Strategy
- Need max analytics power across domains? → single zoned cloud environment (tiered access), not full on-prem/cloud split.
- Need max blast-radius containment and don't need cross-domain analytics? → on-prem (uncurated) / cloud (curated) split.
- Need clear accountability per business unit? → segregate by line of business — but design pipelines for cross-feed from the start, or cross-analytics becomes impossible later.

## Governance Program Accountability Check
Ask of every domain: "If this fails, who specifically is held accountable?" If the honest answer is "everyone" or "no one in particular," the program has an accountability gap (Ch 9) — name a person/group explicitly in role descriptions.
