# Chapter 5: The Data Contract Components: Data Assets and Contract Definition

## Core Idea
Data contract architecture is built from four components (data assets, contract definition, detection, prevention); this chapter covers the first two — identifying which data assets (analytical, transactional, event, first-party-on-third-party-platform) to put under contract, and how to define the contract itself (spec, business logic, schema registry/data catalog) as version-controlled, technology-agnostic code.

## Frameworks Introduced
- **Four Data Contract Architecture Components**: Data assets, Contract definition, Detection, Prevention.
  - When to use: structuring any implementation plan.
  - How: this chapter covers assets + definition; Ch. 6 covers detection + prevention.
- **Data Contract Spec Design Criteria**: technology-agnostic, human-readable, covers schema AND semantics, semantically versioned, accounts for typing heterogeneity across languages.
  - When to use: building or evaluating any contract spec format (custom or open source).
  - How: use JSON Schema + YAML as the recommended base language; don't chase a "standard" spec — the spec is a tool, not the contract itself.
- **Business Logic Lifecycle**: (1) business domains establish logic/SOPs → (2) upstream engineers operationalize it into the transactional database → (3) replication pipelines move it into the analytical database → (4) downstream data teams analyze and recommend iteration back to the business.
  - When to use: diagnosing where business logic has drifted from the data representing it.
  - How: trace the four stages to find where institutional knowledge was lost or where the data stopped reflecting the current logic.
- **Schema Registry vs. Data Catalog choice**: Schema Registry = real-time, record-level enforcement for streaming; Data Catalog = batch, asset-level enforcement plus broader metadata (lineage, governance, glossary).
  - When to use: choosing the metadata source of truth for CI/CD contract checks.
  - How: use Schema Registry for streaming/event assets; use data catalogs once you need lineage, governance, or batch-level enforcement; use both where architecture warrants it.

## Key Concepts
- **Data asset categories**: Analytics databases, Transactional databases, Event sourcing/streams, First-party data on third-party platforms (CRM/ERP like Salesforce/HubSpot/SAP).
- **Invariant**: a true condition about your data that the application (not the database) is responsible for preserving (ACID's "consistency," per Kleppmann) — e.g., "what defines a customer."
- **Event sourcing**: an immutable, appended log of every event (publish-subscribe); enables full state reconstruction and supports multiple consumer views.
- **Event streaming (windowing)**: aggregating high-volume data into fixed time intervals rather than logging every event, to avoid memory/cost blowup.
- **Contract management fields**: `spec-version`, `name`, `namespace`, `dataAssetResourceName`, `doc`, `owner` — the metadata that identifies and governs a contract, independent of schema/semantics.
- **Data schema (in spec)**: standard JSON Schema types (string, number, integer, object, array, boolean, null) plus optional/union typing for nuanced cases (e.g., nullable fields).
- **Data semantics (in spec)**: constraints on actual data values (charLength, isNull, isNullThreshold, max, regex-like conditions) via schema composition/subschemas — goes beyond structure into business meaning.
- **Prevention window**: the gap between staging and execution of a batch replication job (for third-party-platform data) where a contract check can still block bad data from reaching the target database.

## Mental Models
- Think of the data contract spec the way you'd think of a database choice (Snowflake vs. BigQuery vs. Databricks) — the specific format matters far less than that it exists and is enforced; don't get stuck chasing standardization.
- Use "crawl, walk, run": start with schema enforcement, then semantic constraints, then advanced metadata-driven use cases (PII enforcement, lineage-based impact analysis, profiling thresholds) — don't try to implement everything at once.
- Think of transactional databases as "the most upstream data source your organization actually controls" — prioritize contracts there over analytical databases, which are usually just replications.
- Treat first-party-data-on-third-party-platforms (Salesforce, HubSpot) as the one case where contracts ARE enforceable on "third-party" data, because you control the underlying data model even though you don't control the platform.

## Anti-patterns
- **Chasing a single standardized contract spec format**: premature — the authors explicitly decline to recommend one; what matters is the underlying mechanism (schema + semantics + management metadata), not the file format.
- **Treating analytical databases as a dumping ground via unchecked ELT replication**: creates a "needle in a haystack" of mostly-unused data and brittle downstream dbt transformation layers that break on any upstream schema change.
- **Assuming ACID "consistency" is a database guarantee**: it's an application responsibility — the database can't stop you from writing data that violates your own business invariants (only specific checks like FK/uniqueness constraints help).
- **Full replacement-style batch loads at scale**: as data volume grows, full-table replacement batch jobs cause hours-long backfills and stale data on failure — a sign to migrate toward event sourcing/CDC.
- **Trying to enforce contracts directly on genuine third-party data**: rarely works without leverage; instead, place contracts at the ingestion/staging boundary before data enters your transactional database.

## Code Examples
```json
{
    "spec-version": "1.0.0",
    "name": "object-images-contract-spec",
    "namespace": "met-museum-data",
    "dataAssetResourceName": "postgresql://postgres:5432/postgres.object_images",
    "doc": "Data contract for the object_images table...",
    "owner": { "name": "Data Engineering Team", "email": "data-eng@museum.org" },
    "schema": {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "title": "Object Images Table Schema",
      "table_catalog": "postgres",
      "table_schema": "public",
      "table_name": "object_images",
      "properties": {
        "object_id": {
          "description": "Identifying number for each artwork (unique, can be used as key field)",
          "examples": [437133],
          "constraints": { "primaryKey": true, "data_type": "integer", "is_nullable": false, "is_updatable": true }
        }
      }
    }
}
```
- **What it demonstrates**: a real JSON-Schema-based data contract spec (Postgres museum DB) separating contract-management metadata (spec-version/name/namespace/owner) from schema definition.

```yaml
schema:
  - name: exhibit_id
    doc: (Optional) The id of an exhibit within a specific aquarium.
    type: union
    types: ['null', 'string32']
  - name: exhibit_status
    type: enum
    symbols: ['OPEN', 'CLOSED', 'MAINTENANCE', 'FEEDING']
```
- **What it demonstrates**: optional/union typing and enums in a YAML schema definition to handle nullable and constrained-value fields.

```yaml
schema:
  - name: aquarium_id
    constraints:
      - charLength: 32
      - isNull: FALSE
  - name: exhibit_id
    constraints:
      - isNullThreshold: 0.8
  - name: last_exhibit_update_time
    constraints:
      - isNull: FALSE
      - max: today
```
- **What it demonstrates**: semantic constraints (beyond schema) — null thresholds, char length, max-date bounds — enforced via schema composition/subschemas.

## Reference Tables
| Data asset category | Example | Who typically controls it |
|---|---|---|
| Analytics databases | Snowflake, BigQuery, data lakes | Data teams |
| Transactional databases | Postgres/MySQL production DBs | Software engineers |
| Event sourcing / streams | Kafka, Redpanda | Data/platform engineers |
| First-party data, third-party platform | Salesforce, HubSpot, SAP | Business ops, replicated via ELT |

| | Event sourcing | Event streaming |
|---|---|---|
| Use case | Immutable log, every event appended | High-volume data aggregated into time windows |

| | Schema Registry | Data Catalog |
|---|---|---|
| Processing | Stream, record-level | Batch, asset-level |
| Scope | Schema/serialization consistency | Schema + lineage + glossary + governance |

## Worked Example
**B2C-to-B2B invariant break**: a consumer app company assumed for ~10 years that "each customer/user is a one-to-one relationship" (B2C only), baking this into architecture, databases, and docs. When it expanded to B2B, the invariant broke (e.g., a freelancer joining a company for a 3-month contract), causing billing failures and revenue loss. The company used data contracts during its refactor to define the desired end-state invariants and iteratively find and fix every violation in the transactional database.

**Mark's HR tech third-party ingestion**: enterprise customers sent employee data via CSV/SFTP/API on inconsistent cadences (daily/monthly/quarterly), causing missing files, format changes, and broken org-chart/manager-status logic. A contract at the staging-to-transactional-database boundary would document the differential ingestion patterns, block violating data from propagating into the app, and log violation patterns to show customers the cost of their poor-quality submissions.

**Mark's Salesforce-to-BigQuery ARR optimization**: replicated Salesforce data (first-party data, third-party platform) via Fivetran into BigQuery for staffing-optimization reports; a one-week window existed where bad Salesforce data could silently corrupt revenue-related business decisions before anyone noticed — illustrating the "prevention window" pattern (contract check between staging and execution of the batch replication job) instead of a CI/CD PR check.

## Key Takeaways
1. The four data asset categories (analytical, transactional, event, first-party-on-third-party) each have different ownership and different feasible points of contract enforcement.
2. Transactional databases are the best default starting point because they are the most upstream asset your org actually controls.
3. The contract spec itself is just a tool (JSON Schema + YAML recommended) — don't over-invest in finding "the standard"; focus on the underlying capabilities it must support.
4. A contract spec has three logical parts: contract management metadata (name/namespace/owner/version), schema definition, and semantic constraints — build in that order (crawl, walk, run).
5. Business logic has its own lifecycle (establish → operationalize → replicate → analyze/iterate) and typically lives in institutional knowledge (Slack messages, tribal memory) unless captured in the contract's `doc` and `owner` fields.
6. Use Schema Registry for real-time/streaming enforcement and data catalogs for batch-level enforcement plus lineage/governance; use both when architecture warrants it.
7. Even "third-party" data can sometimes be contracted if you control the underlying data model (CRM/ERP platforms) — enforce at the staging/replication boundary instead of CI/CD.

## Connects To
- **Ch 4**: Implements the "where to implement data contracts" placement guidance (points A–G) with concrete asset categories.
- **Ch 3**: The ACID/invariant discussion extends Ch. 3's point that data debt stems from assumptions, not just code.
- **Ch 6**: Will cover the remaining two components — detection and prevention — that act on the assets/definitions covered here.
- **Ch 8**: Promises a full end-to-end implementation using the museum (Postgres) contract spec shown here.
