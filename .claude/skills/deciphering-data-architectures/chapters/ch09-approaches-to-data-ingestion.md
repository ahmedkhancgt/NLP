# Chapter 9: Approaches to Data Ingestion

## Core Idea
Data ingestion decisions — ETL vs. ELT, reverse ETL for operationalizing analytics, batch vs. real-time processing, and data governance — determine how well an organization can trust and act on its data, and the "right" choice depends on data volume/type, latency tolerance, and target system, not a single universal method.

## Frameworks Introduced
- **ETL vs. ELT**: where transformation happens relative to loading.
  - When to use: ETL when source and destination are both relational, datasets are smaller, and tight data-quality/security control before loading matters; ELT when dealing with high-volume unstructured/semi-structured data destined for a data lake, or when transformation needs to be iterated without re-extracting.
  - How: ETL = extract → transform/clean in-flight → load into destination. ELT = extract (optionally filtering unneeded rows/columns) → load raw into destination → transform in place, leveraging destination compute and batch-style transformation.
- **Reverse ETL**: moving analytically-prepared data back out of the DW/lake into operational (often SaaS) systems.
  - When to use: turning analytical insights (lifetime value, propensity scores, churn risk) into real-time, operational action inside tools teams already use (Salesforce, CRM, customer support systems, production apps).
  - How: clean/master data via ELT in the DW/lake, compute metrics in SQL, then push those metrics/scores back into SaaS or production systems to drive personalized experiences or sales workflows.
- **Batch vs. Real-Time Processing**: when/how often to extract.
  - When to use: batch for large volumes with high latency tolerance (reporting that can wait hours/overnight); real-time for low-latency-tolerance needs (fraud detection, traffic apps, IoT streams).
  - How: batch groups transactions over a period and runs a scheduled job (usually off-peak); real-time uses an event-based architecture that reacts within seconds/milliseconds to new records/files.
- **Data Governance Maturity Model**: a framework (the author's own) for assessing an organization's governance readiness.
  - When to use: before building any DW/lake solution — governance should be established upfront, not retrofitted.
  - How: establish a Center of Excellence (CoE) defining governance policies, roles, responsibilities, and decision-making processes for data quality, compliance, security, and transformation standards.

## Key Concepts
- **ETL mnemonic**: "Early Transformation Leads" — transform happens early, right after extraction, before loading.
- **ELT mnemonic**: "Every Load Transforms" — raw data loads first, then is transformed in the destination.
- **Operational analytics**: using DW/lake-derived insights to improve day-to-day operational decisions (vs. traditional long-term strategic BI).
- **Lifetime value, product qualified lead, propensity score**: example metrics computed in the DW and pushed via reverse ETL into operational tools to drive customer engagement.
- **Latency tolerance**: how much delay a business process can accept — the central variable in choosing batch vs. real-time.
- **Data governance Center of Excellence (CoE)**: a dedicated organizational hub for defining and enforcing governance policy, roles, and standards.

## Mental Models
- Electric bill = batch processing (data collected over a period, processed as a whole at a scheduled time); Waze traffic updates = real-time processing (continuous, event-driven, decision depends on freshness).
- "The wrong ingestion strategy is a business risk, not just a technical one" — poor data quality, cost overruns, performance issues, and regulatory breaches all trace back to ingestion-strategy choices.
- Reverse ETL closes the loop: analytical insight (DW/lake) → operationalized action (SaaS/production systems) → better day-to-day decisions, completing what traditional ETL/ELT only half-finishes (sourcing data in, but never pushing insight back out).

## Anti-patterns
- **Using ETL for very large or highly unstructured/semi-structured datasets**: traditional ETL tools have limited processing power and transform record-by-record (slow), and some support limited data types — a poor fit for big-data/unstructured workloads better served by ELT.
- **Re-extracting from the source system every time a transformation bug needs fixing (ETL drawback)**: repeated extraction risks degrading source-system performance; ELT avoids this because raw data is already landed and only the transformation step needs rerunning.
- **Building a DW/lake solution before establishing data governance**: the chapter explicitly warns that "too often, projects fail because no one has paid enough attention to data governance" — governance must be designed upfront, in parallel with or before the architecture, not bolted on after.
- **Choosing real-time processing without the resources to support it**: real-time systems carry higher system-resource demands, higher failure risk, harder error recovery, and higher cost — inappropriate when latency tolerance is actually high and batch would suffice.

## Reference Tables
| Dimension | ETL | ELT |
|---|---|---|
| Transform timing | Before load (in-flight) | After load (in destination) |
| Best for | Smaller datasets, relational-to-relational | High-volume, unstructured/semi-structured, data lakes |
| Data quality control | Strong (clean before load) | Transform iteratively after load |
| Performance on large data | Weaker (record-by-record, resource-limited tools) | Stronger (leverages lake compute, batch transforms) |
| Rerun cost if bug found | Must re-extract from source | Re-transform only; no re-extraction needed |

| Dimension | Batch Processing | Real-Time Processing |
|---|---|---|
| Latency | Minutes to hours/overnight | Seconds to milliseconds |
| Resource use | Lower, scheduled off-peak | Higher, continuous |
| Failure risk | Lower (retriable) | Higher, complex recovery |
| Best for | Large volumes, high latency tolerance | Fraud detection, IoT, traffic, trading |
| Typical store | RDW (traditionally) | Data lake (handles millions of events/sec) |

| Pipeline Type | Direction | Purpose |
|---|---|---|
| ETL | Source → DW/lake (transform before load) | Analytical readiness |
| ELT | Source → DW/lake (transform after load) | Analytical readiness, big-data scale |
| Reverse ETL | DW/lake → operational/SaaS systems | Operationalize insights into daily workflows |

## Worked Example
A company ingests and masters customer data in its DW via ELT, computes a propensity score with SQL, and uses reverse ETL to push that score into a production database — enabling real-time, personalized in-app experiences (tailored offers/recommendations) based on how likely a customer is to buy. The same mastered customer data is also pushed into Salesforce and a customer support system via reverse ETL, giving every operational team a consistent view of each customer, and usage data surfaced in the CRM helps identify customers at risk of churning.

## Key Takeaways
1. Choose ETL for smaller, relational-to-relational pipelines prioritizing pre-load data quality/security; choose ELT for large-scale, unstructured/semi-structured data lake ingestion.
2. Use reverse ETL to operationalize DW/lake insights (lifetime value, propensity scores, churn risk) back into the SaaS/production tools teams use daily — this is a distinct, increasingly common pipeline direction, not just ETL/ELT in reverse.
3. Match batch vs. real-time processing to the business process's actual latency tolerance, not to technology hype — most data still doesn't need real-time.
4. Build and staff a data governance Center of Excellence before building the DW/lake solution; retrofitted governance is a common cause of project failure.
5. The ETL/ELT/batch/real-time choice is a business risk decision (cost, compliance, quality) as much as a technical one.

## Connects To
- **Ch 4**: ETL is the classic RDW population mechanism (Ch 4's extraction methods — full/incremental, timestamps/CDC — are the mechanics underlying ETL/ELT here).
- **Ch 5**: ELT's preference for data lakes directly follows from Ch 5's schema-on-read/raw-storage model.
- **Ch 10**: The modern data warehouse architecture (next Part) operationalizes ETL/ELT/reverse ETL together across lake + warehouse.
- **Ch 6**: MDM (customer mastering) is the prerequisite step before reverse ETL pushes consistent customer data to SaaS systems.
- **External concepts**: Salesforce and other SaaS CRM/support systems as reverse ETL targets.
