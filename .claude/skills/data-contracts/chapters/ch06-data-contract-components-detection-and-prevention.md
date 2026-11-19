# Chapter 6: The Data Contract Components: Detection and Prevention

## Core Idea
Detection (code-level, data-in-motion, historical) finds where data diverges from contract expectations, while prevention (version control, CI/CD, violation monitoring/alerts) operationalizes that detection into blocked merges and targeted, trustworthy alerts — and the earlier (more "shift left," i.e. code-level) the detection, the more truly preventative the contract.

## Frameworks Introduced
- **Three Detection Categories**: Code-level, Data-in-motion, Historical.
  - When to use: choosing where/how to detect a contract violation for a given asset type.
  - How: code-level (static code analysis on PRs) catches issues before code ships; data-in-motion (WAP pattern, CDC, stream processing) catches issues in live data between source/target; historical surfaces pre-existing violations when first putting a legacy asset under contract.
- **DAMA's Nine Data Quality Dimensions**: Validity, Completeness, Consistency, Integrity, Timeliness, Currency, Reasonableness, Uniqueness, Accuracy.
  - When to use: deciding exactly which dimension(s) a given contract constraint should enforce.
  - How: match the business risk to the right dimension (e.g., Currency for freshness SLAs, Accuracy for business-logic-dependent correctness like legal drinking age).
- **Write-Audit-Publish (WAP) Pattern**: write to staging → audit against contract spec → publish to production if passed / route to dead-letter-queue if failed.
  - When to use: preventing bad streaming/live data from propagating downstream.
  - How: apply software engineering's staging/testing-before-production discipline to data itself.
- **Five Ws and One H for Violation Alerts** (Who/What/When/Where/Why/How): a journalism framework repurposed for alert design.
  - When to use: designing any violation notification so it's actionable rather than noise.
  - How: for each stakeholder type (software dev, data dev, business stakeholder, leadership, external customer) specify exactly who is notified, what info they get, when, where, why they should care, and how to resolve it.
- **End-to-End Lineage** = Source Code Flow Graph + Data Lineage.
  - When to use: tracing a data quality issue all the way from a dashboard back to the exact function/commit that caused it.
  - How: use data lineage (table/asset graph) for steps near the database, and source code flow graphs (function-call graph) for steps inside the application codebase; combine both for a complete trace.

## Key Concepts
- **Data quarantining ("dead letter queue")**: tagging/separating violating data instead of blocking it outright, used when you can't engage the producer directly (e.g., third-party data brokers).
- **Change Data Capture (CDC)**: monitors CRUD ops on an upstream DB and propagates changes downstream in real time.
- **Stream processing**: processes/analyzes ingested data in real time at creation.
- **Live/real-time data monitoring**: observes and analyzes real-time data for quality without necessarily acting on it.
- **Static Code Analysis (SCA)**: reasoning about program behavior without executing it; used to build source code flow graphs and detect data-impacting changes before deployment (e.g., Meta's Pysa, Infer).
- **Source code flow graph**: directional graph of function/object references within a codebase.
- **Data lineage**: directional graph of how data is sourced/transformed/moved across tables.
- **SEV-0**: the highest-severity incident tier, used in the alerting framework as the trigger for leadership notification.

## Mental Models
- Think of code as "the earliest point you can programmatically prevent a data quality violation" — code-level detection (SCA) is the most shift-left, most preventative option, even though it's the hardest to build.
- Use "the more metadata types you connect, the more robust your contract" — end-to-end lineage (code graph + data graph) turns a days-to-weeks manual trace into an automatically surfaced, targeted alert.
- Treat alerting like journalism: an alert without a clear who/what/when/where/why/how is noise, and noise causes alert fatigue that erodes trust in the entire system (cf. the Epic Sepsis Model failure).
- Translate vocabulary across audiences: "data lineage" means nothing to many software engineers — say "code flow graph" instead when talking to them.
- View detection/prevention as the developer's first impression of data contracts — make the required effort for developers as close to zero as possible, since their job constraints don't include data by default.

## Anti-patterns
- **Relying only on schema/semantic detection, ignoring code**: misses the root cause, since code is the actual mechanism that turns business logic into data.
- **Pushing bad data forward indiscriminately**: wrong for high-stakes cases (e.g., health insurance eligibility, where no data is safer than wrong data) but can be *useful* elsewhere (e.g., telemetry, where "bad" data signals faulty hardware) — the choice must be use-case-specific, not a blanket rule.
- **Over-alerting without targeting ("alert fatigue")**: the Epic Sepsis Model required clinicians to evaluate 109 patients to find one true sepsis case per alert cycle — an extreme illustration of how non-actionable alerts destroy trust and get ignored.
- **Manual end-to-end tracing as a standing process**: Mark's 10-step manual lineage+code trace took days to weeks — a clear signal that this should be automated, not repeated by hand indefinitely.
- **Expecting data lineage tools alone to solve root-cause analysis**: most lineage tools only see the analytical database post-replication, missing the transactional/application-code origin of the issue.

## Code Examples
*(omitted — this chapter illustrates concepts via workflow diagrams (WAP pattern, CI/CD figure-eight, alert workflow) and a narrative frontend-event-renaming example rather than literal code/config snippets)*

## Reference Tables
| Detection method | Transactional DB | Analytical DB | Streaming data | 3rd-party-platform 1st-party data |
|---|---|---|---|---|
| Data quarantining | X | X | | X |
| Change data capture | X | X | X | X |
| Stream processing | | | X | |
| Source code flow graphs | X | X | X | |
| Data lineage | X | X | X | X |
| Static code analysis | X | X | X | |
| Live data monitoring | X | X | X | X |

| DAMA dimension | Example |
|---|---|
| Validity | Timestamp occurs during business hours |
| Completeness | Birth date has day/month/year |
| Consistency | Column values structured uniformly |
| Integrity | City/country combination is plausible |
| Timeliness | Data refresh arrives within SLA window |
| Currency | Last refresh not older than SLA allows |
| Reasonableness | A "month" column has ≤12 distinct values |
| Uniqueness | All user_ids are distinct |
| Accuracy | Age meets jurisdiction's legal minimum |

| Alert "Who" | Alert "When" |
|---|---|
| Software developer making the change | When code change is committed to PR branch in CI/CD |
| Data developer owning impacted asset | When a depended-on asset is impacted |
| Business stakeholder | When blocked from a critical workflow |
| Leadership | SEV-0 incidents threatening the business |
| External customer | When actively impacted at time of event |

## Worked Example
**Health insurance eligibility ingestion (Mark's early job)**: incoming files were critical (affecting whether patients got care covered) and went through a staging ingestion check; files failing requirements (e.g., missing date of birth) were quarantined for manual review by an entire operations team over weeks, sometimes causing backlog spikes that forced the whole team to drop other work. The business rule here was strict: no data was better than bad data, because wrong eligibility data caused direct patient harm or billing errors. Data contracts would improve this by generating more descriptive failures tied to the contract spec's domain context, automating communication to external data producers, and communicating quarantine scope to internal downstream consumers.

**Glassdoor's static code analysis implementation** (referenced from their engineering blog, "Data Quality at Petabyte Scale"): used SCA as one of the earliest steps in its data-quality-check pipeline, embedding detection directly in the CI/CD code-review step to catch data-impacting code changes (e.g., a frontend event changing from separate `first_name`/`last_name` fields to a single `full_name` field) before merge.

**Epic Sepsis Model alert fatigue** (JAMA-published external validation study): the model alerted in 18% of hospitalizations; even with alert-minimization, clinicians had to evaluate 8 patients per true sepsis case, rising to 109 patients per case under reactive re-evaluation — a cautionary real-world example of how poorly targeted alerts erode trust and defeat the entire point of a detection/alerting system, used here as a warning for data contract alert design.

## Key Takeaways
1. Detection should happen as far upstream (code-level) as possible — it's the most preventative but hardest to build; data-in-motion and historical detection fill the gaps where code-level isn't feasible.
2. Match each contract constraint to a specific DAMA data quality dimension to clarify exactly what business risk it mitigates.
3. Use the WAP pattern (write-audit-publish) to apply staging/testing discipline to live/streaming data before it reaches production.
4. Combine data lineage (table/asset graph) and source code flow graphs (function-call graph) into "end-to-end lineage" — most existing lineage tools stop at the analytical database and miss the application-code origin.
5. CI/CD is the most critical prevention mechanism: without it, a data contract is just "a fragile handshake agreement," not an enforced one.
6. Design every violation alert against the five Ws and one H, tailored per stakeholder type — generic or excessive alerts cause alert fatigue and destroy trust (see Epic Sepsis Model).
7. Treat detection/prevention UX as developers' first impression of data contracts; minimize the extra work required of them.

## Connects To
- **Ch 5**: Builds directly on the data assets and contract definition covered there — detection/prevention act on that metadata.
- **Ch 3**: Alert fatigue and the "right time, right people" communication principles directly extend the design-surface requirements from Ch. 3.
- **Ch 7**: Promises a hands-on coding tutorial implementing detection and CI/CD-based prevention end to end.
- **Ch 8**: References Glassdoor's real-world SCA implementation as a detailed case study.
