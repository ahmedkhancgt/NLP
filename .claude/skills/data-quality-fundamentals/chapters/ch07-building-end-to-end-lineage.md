# Chapter 7: Building End-to-End Lineage

## Core Idea
Field-level, end-to-end data lineage — a map of exactly which source fields produce which destination fields — is essential for fast root cause analysis and impact analysis, but is technically hard to build (parsing arbitrary SQL complexity) and must be surfaced through a thoughtfully designed, abstracted UI to be actually useful.

## Frameworks Introduced
- **Field-Level Lineage** (vs. table-level lineage): tracks relationships down to individual columns/fields, not just whole tables.
  - When to use: whenever table-level granularity is too coarse to pinpoint root cause quickly (the common case in root cause analysis).
  - How: parse SQL queries (e.g., via ANTLR) to extract column-level SELECT and non-SELECT relationships; recursively resolve aliases/subqueries back to true source tables/columns.
- **Five Key Requirements for Building Lineage**: fast time to value (field-level, not table-level), secure architecture (metadata-only access, data stays in customer environment), automation (vs. manual upkeep), integration with popular data tools (warehouses, transformation tools, BI tools), and extraction of column-level information.
  - When to use: as a build-vs-buy / scoping checklist before investing engineering time in a lineage system.
- **SELECT vs. Non-SELECT Lineage**: SELECT clause lineage captures direct field-to-field relationships (an upstream field change directly changes a downstream field); non-SELECT lineage (e.g., WHERE clauses) captures field-to-table relationships where downstream fields are shaped by upstream filtering/ordering logic rather than direct substitution.
  - When to use: distinguishing "this field literally becomes that field" from "this field determines which rows appear downstream" when debugging.
- **"Controlled Freedom" model** (Fox Networks): a centralized data team controls ingestion, security, and standard metric definitions, while granting decentralized stakeholders free rein over ad hoc analytics and data discovery within that trusted "optimized layer."
  - When to use: balancing self-service analytics against data governance/trust requirements at scale.
  - How: wrap the full stack in QA/validation/alerting (observability) so stakeholders can trust data without needing centralized gatekeeping for every use case.

## Key Concepts
- **Data lineage**: a record/map of how data got from point A (ingestion) to point B (analytics layer), tracing upstream sources and downstream dependencies.
- **Table_mcon ID**: a logical data model identifier (Monte Carlo's "mcon") used with a hashed field-level lineage object to capture multiple distinct queries updating the same destination table.
- **Selected vs. nonselected fields**: selected fields define the actual result table's field values; nonselected fields affect which rows are fetched (filtering) but don't directly contribute to output values — distinguishing these declutters the lineage UI.
- **STM (Source to Target Mapping)**: a specification (used at Fox) that lets analysts define business requirements so engineers can build pipelines from a well-defined playbook, without engineers needing to sit through business requirement-gathering meetings.
- **Three-layer cake architecture** (Fox): raw layer (data as ingested) → optimized layer (sorted/sliced/reformatted for speed and usability — the self-serve layer) → published layer (defined data models/sets for broad consumption).
- **Data debt reduction via lineage**: using field-level lineage to safely identify and deprecate unused/outdated columns without breaking downstream consumers.

## Mental Models
- Lineage is to data what `git blame` is to code — a way to trace exactly where and why something broke in a larger system.
- "The most effective lineage doesn't just surface information — it surfaces the right information at the right time": prioritize UI design that abstracts away a "spider web" of relationships down to a clean tree of relevant nodes.
- Data teams care most about the two extremes of the lineage chain: the most downstream layer (BI dashboards/reports — what to tell consumers not to trust) and the most upstream layer (the source table — where to actually fix the bug); the middle layers matter less for most RCA.
- Trust compounds: transparency about data health (even when something breaks) builds the trust needed for self-serve data culture; hiding issues destroys it.

## Anti-patterns
- **Relying on manual, encyclopedic lineage knowledge**: historically lineage was parsed manually, requiring deep tacit knowledge of the data environment — unsustainable as pipelines scale.
- **Stopping at table-level lineage**: too coarse-grained for the root cause analysis teams actually need; doesn't explain *why* or *how* a pipeline broke, only *that* something downstream is connected.
- **Letting lineage access raw user data/PII directly**: a security anti-pattern — lineage should operate on metadata, logs, and queries, keeping underlying data in the customer's environment.
- **Treating self-service analytics as "no governance needed"**: Fox's "controlled freedom" explicitly rejects pure free-for-all; centralized control over ingestion, security, and standard definitions must remain in place for decentralized freedom to be trustworthy.
- **Chasing "shiny new toys"**: per Alex Tverdohleb, pursuing the latest technology for its own sake rather than optimizing the existing stack to meet actual business needs is a common, costly mistake.
- **Letting developers run the business-requirements process**: having engineers spend significant time in business meetings instead of letting analysts do requirements-gathering (via STM) wastes specialized engineering time.

## Code Examples
```json
// Index schema for a field-level lineage edge (destination table <- source tables)
{
  "edge_id": "37d65dc5c943cab124398b2c43f0d8f2c0ff5e76a2ba3052",
  "destination_table_mcon": "",
  "source_table_mcons": ["", ""],
  "sources": [{"table_mcon": "", "field_name": ""}],
  "destination_field": "new field name",
  "parsed_query": ""
}
```
- **What it demonstrates**: a denormalized lineage data model capturing edges between destination fields and their upstream source tables/columns, keyed by a hashed combination of destination table ID and parsed query, so multiple queries updating the same table can each be tracked independently.

## Reference Tables
| Lineage Layer | Primary Use |
|---|---|
| Most downstream (BI reports/dashboards) | Tell consumers which outputs are currently untrustworthy |
| Middle layers (transformations) | Less critical for most RCA; useful for deep tracing |
| Most upstream (source table/field) | Locate and fix the actual root-cause bug |

| Lineage Requirement | Why It Matters |
|---|---|
| Fast time to value | Field-level granularity for quick remediation |
| Secure architecture | Metadata-only access; no PII exposure |
| Automation | Scales with data growth; avoids manual upkeep |
| Tool integration | Must span warehouse/lake, transform, and BI layers |
| Column-level extraction | Enables anomaly root-causing, not just table mapping |

## Worked Example
**Fox Networks** (VP of Data Services Alex Tverdohleb): Fox ingests data from 200+ sources multiple times a day, processing ~10,000 schemas and tens of billions of records per week — a scale where manual validation is impossible. Fox built a lakehouse architecture organized as a "three-layer cake" (raw → optimized → published) and gave internal stakeholders free, self-service access to the "optimized" layer under a "controlled freedom" model: the centralized data team guarantees ingestion is trustworthy, data is secure, and standard metrics/definitions are consistent for executive reporting, while everything else (ad hoc analytics, discovery) is left open for consumers to use as they see fit. This is underpinned by continuous observability (using tools like Monte Carlo) and automated lineage, enabling the team to catch issues before production, reverse-engineer impact scope when something does break, and stop problems in-flight before they cascade downstream — all while five specialized sub-teams (tagging/collections, engineering, analytics, science, architecture) collaborate via a Source-to-Target-Mapping (STM) spec that cleanly separates analyst-driven requirements gathering from engineer-driven pipeline building.

## Key Takeaways
1. Invest in field-level (not just table-level) lineage if your team's primary bottleneck is root-cause analysis speed — table-level lineage alone leaves too much ambiguity.
2. Expect SQL-parsing complexity to be the hardest technical obstacle; budget for incremental coverage (the authors' first prototype covered only ~70% of real-world SQL clause combinations).
3. Design lineage architecture around a flexible, hashable data model (destination table ID + query hash) so multiple queries touching the same table can be tracked distinctly.
4. Keep lineage tooling metadata-only for security; never pipe raw PII through the lineage layer itself.
5. Build the lineage UI to highlight the two most actionable layers — most-upstream (where to fix) and most-downstream (who/what to warn) — rather than exposing the full dependency "spider web."
6. Pair self-service/decentralized data access with strong centralized guarantees on ingestion, security, and standard definitions ("controlled freedom") rather than either pure centralization or pure free-for-all.
7. Ship lineage/product features iteratively with early customer prototypes rather than waiting for a "perfect" first version.

## Connects To
- **Ch 6**: Lineage was the foundation of RCA Step 1 there; this chapter details how to actually build that lineage system.
- **Ch 2**: Extends the data discovery/catalog concepts, since lineage is explicitly called out there as a feature of modern, discovery-oriented catalogs.
- **Ch 8**: Previews the organizational/cultural work of democratizing data quality, building on the technical trust that lineage and observability establish here.
- **git blame / software engineering**: the explicit analogy the authors use to frame why lineage matters for debugging distributed data systems.
