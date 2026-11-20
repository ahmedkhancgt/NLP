# Chapter 6: Governance of Data in Flight

## Core Idea
Governance must be preserved continuously as data is extracted, transformed, and loaded (ETL/ELT) — this is achieved primarily through **lineage tracking**, which lets policies (access, retention, classification) automatically propagate to derivative datasets rather than requiring manual re-tagging at every step.

## Frameworks Introduced
- **ETL / ELT Pipeline Stages**
  - When to use: understanding where governance controls (validation, classification) should be inserted in a data movement pipeline.
  - How: Extract (retrieve from source; validate here while source context is still available) → Transform (normalize, join, aggregate, split — but this *removes* information, so re-anticipate future use cases before discarding) → Load (into warehouse, e.g., BigQuery/Snowflake/Redshift). Modern warehouses often invert the order to ELT (transform inside the warehouse after loading).

- **Lineage Granularity Levels**: table/file-level → column/field-level → row-level → dataset-level
  - When to use: choosing how fine-grained your lineage tracking needs to be for a given governance use case.
  - How: table-level answers "what produced this table"; column-level enables tracking a specific infotype (e.g., PII) through joins into derivative tables; row-level supports transaction-specific tracing; dataset-level gives coarse source information.

- **Lineage-Driven Policy Inheritance**
  - When to use: automatically propagating a data-class policy (e.g., "PII access restricted to full-time employees") to all derivative tables/dashboards without manual re-tagging.
  - How: tag source columns with infotype/class at origin; use lineage graph to detect when a column is copied unchanged (inherit identical policy) vs. transformed/nullified (apply a different or no policy).

## Key Concepts
- **Lineage (provenance)**: the recorded "path" data takes through creation, transformation, and import — answers "why does this dataset exist" and "where did the data come from."
- **Granular/column-level access control**: locking down a single sensitive column (e.g., customer name in a retail transaction table) while leaving the rest of the table accessible — avoids the binary choice of full access vs. full rewrite.
- **The Fourth Dimension (temporal lineage)**: lineage is a *point-in-time* state — a table's "current" lineage may differ from its historical lineage; versioning information must be tracked to reconstruct past states.
- **Egress controls / trusted-system enforcement**: using lineage to detect and block data entering from unapproved/untrusted sources.
- **Change-impact analysis via lineage**: before changing an access policy or deleting a table, use lineage to trace forward to all affected dashboards/end products and warn affected users in advance.

## Mental Models
- **Lineage as a force multiplier for quality and sensitivity propagation**: once you know a source's quality/sensitivity, lineage lets you infer the same property for every downstream derivative without re-assessing each one independently.
- **Normalization removes information you may need later**: early-stage cleaning (joins, aggregation, outlier removal) can discard context relevant to an unforeseen future use case — weigh business context before transforming.
- **Technical lineage vs. business lineage**: present full technical detail (intermediate tables) to analysts, but a simplified "business view" (e.g., "this dashboard traces to trusted, high-quality sources") to business stakeholders — both representations matter for different audiences.

## Anti-patterns
- **Forcing users to manually register/annotate every reshaped dataset**: adds friction and causes governance to be skipped under deadline pressure; lineage automation is meant to prevent this.
- **Relying only on "current state" lineage**: ignores that a table's provenance may have changed over time — critical investigations need the temporal/versioned view.
- **Choosing to grant full table access or a full rewritten/redacted copy as the only two options**: column-level/attribute-based granular access controls avoid both the overexposure and the headcount burden of constant table-rewriting.
- **Assuming a job-log-based ("after the fact") lineage approach is as effective as just-in-time recording**: it requires backtracking/batch processing and is a fallback, not a preferred primary method.

## Code Examples
No executable code; the chapter illustrates with lineage-graph diagrams (Figures 6-1 through 6-4) rather than code. Key procedural pattern:
- **How to collect lineage**: (1) prefer a built-in warehouse/catalog lineage facility capturing lineage end-to-end automatically; (2) failing that, connect to the data warehouse's API/job audit log (captures SQL jobs + programmatic pipelines) and backtrack table-creation statements to predecessors.
- **What it demonstrates**: automation is strongly preferred over manual curation as data volume grows, since human bottlenecks block governance at scale and broken lineage chains compound distrust in derivative products.

## Reference Tables
No formal table; three recurring CIO/CISO-level governance questions function as a practical checklist:
1. "Show me all sensitive data in our warehouse and what systems contain it" — answered via source classification + lineage.
2. "I want certain systems marked trusted, and data must not leak into untrusted systems without oversight" — answered via lineage-based egress controls.
3. "I need to report/audit all systems processing PII" (post-GDPR) — answered via PII-tagged-source lineage tracing.

## Worked Example
**Column-level PII propagation (Fig. 6-4 scenario)**: An external data source and Tables A/B feed into Table D and Table C respectively; these merge into a dashboard. If any one source (say Table B) contains a PII column, lineage tracking at the column level reveals that the PII may propagate into the final dashboard — triggering the dashboard itself to inherit an access policy such as "available only to full-time employees," even though the PII column was several transformation-hops upstream.

**Oracle label-based (attribute-based) security for telecom retail transactions**: A telecom retail-store transaction log contains item/price/date/location (broadly useful for analytics) alongside the purchaser's name/account (sensitive, needed only if the purchaser is an account holder). Instead of granting full table access or manually rewriting a redacted copy, label-based security lets most analysts see the full table minus the sensitive column (redacted/hashed/removed), while only authorized individuals see the unredacted column — achieving both data democratization and sensitive-field protection without constant manual rewriting (a direct answer to the headcount constraint raised in Ch. 3).

## Key Takeaways
1. Lineage tracking is the core enabling mechanism for governing "data in flight" — it lets sensitivity/quality signals and policies propagate automatically to derivatives.
2. Favor column-level (not just table-level) lineage and access control wherever a table mixes sensitive and non-sensitive fields.
3. Validate data as early as possible during extraction, while source context is still available — don't defer validation past the point where context is lost.
4. Prefer automated, just-in-time lineage capture over manual curation or after-the-fact log backtracking as data volume scales.
5. Use lineage for proactive change-impact analysis (warn affected dashboard users before changing an access policy) as well as reactive debugging.
6. Maintain a temporal/versioned view of lineage — "lineage now" is not the same as "lineage as of the decision being audited."

## Connects To
- **Ch 2**: Lineage tracking and data cataloging tools introduced there are deepened into full governance mechanics here.
- **Ch 5**: Data quality's reliance on lineage (for prioritization) is the same mechanism applied here to sensitivity/policy propagation.
- **Ch 7**: Column-level access control and classification here directly feed into the data-protection planning process.
- **Ch 8**: Lineage monitoring (as an ongoing operational concern) is covered as its own monitoring category.
- **GDPR**: the "report all systems processing PII" need is explicitly framed as a post-GDPR requirement.
