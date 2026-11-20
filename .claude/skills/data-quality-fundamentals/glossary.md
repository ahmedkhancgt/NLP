# Glossary

**Analytical data** — Data used for decision making (churn, clickthrough rates); the book's primary focus, as distinct from operational data. (Ch 2)

**Anomaly detection** — Identifying events/observations that deviate from the norm, e.g., fraud or a technical glitch. (Ch 4)

**Blameless postmortem** — A meeting and document after an incident, framed around the system (not individuals) being at fault, used to prevent recurrence. (Ch 6)

**Blast radius** — The extent of downstream impact/downtime experienced by stakeholders when a data asset breaks. (Ch 8)

**Blind ETL** — Transforming lake data while assuming an unenforced schema; dangerous because transformations can silently fail on upstream changes. (Ch 2)

**Central limit theorem** — The statistical principle that independently generated random samples approach a normal distribution as sample size grows; a flawed basis for naive anomaly detection on correlated business data. (Ch 4)

**Circuit breaker (data)** — A pattern that halts a pipeline ("circuit open") when data fails quality thresholds, preventing bad data from propagating downstream. (Ch 3)

**Data catalog** — An inventory of metadata giving stakeholders information about data accessibility, health, and location. (Ch 2)

**Data certification** — The process by which data assets are formally approved for org-wide use after meeting agreed SLAs for quality, observability, ownership, and communication. (Ch 8)

**Data discovery** — A dynamic, domain-specific, real-time alternative/successor to static data catalogs, suited to distributed architectures. (Ch 2, Ch 9)

**Data downtime** — Periods when data is missing, inaccurate, or otherwise erroneous; the book's core metric, analogous to software application downtime. (Ch 1)

**Data enrichment** — Merging first- or third-party data into existing data sets to fill gaps and add value. (Ch 5)

**Data lakehouse** — A convergence architecture adding warehouse-style schema/SQL/ACID features to data lakes (or lake-style flexibility to warehouses). (Ch 1, Ch 2)

**Data literacy** — The ability to read, write, and communicate about data in a way that drives organizational value. (Ch 8)

**Data mesh** — A domain-oriented, self-serve data architecture with federated governance, pioneered by Zhamak Dehghani; the data-platform analogue of microservices. (Ch 1, Ch 9)

**Data mesh score** — A heuristic (data sources + team size + domains + engineering bottleneck severity + governance priority) for deciding whether to invest in a data mesh. (Ch 9)

**Data observability** — An organization's ability to fully understand the health of data in its system at every stage via five pillars: freshness, volume, distribution, schema, lineage. (Ch 4, Ch 5)

**Data product manager** — A role dedicated to data democratization, cross-functional platform management, and increasing time-to-value for data (pioneered at Uber). (Ch 8, Ch 10)

**Data reliability** — An organization's ability to deliver high data availability and health throughout the entire data life cycle; the outcome of sustained high data quality. (Ch 5)

**Data reliability engineer** — A predicted emerging role focused on building resilient data stacks via observability and testing, importing DevOps practices. (Ch 10)

**Data reliability life cycle** — The data analogue of the DevOps life cycle: Detect → Respond → Root Cause → Resolve → Prevent. (Ch 6)

**Data reliability maturity curve** — Four stages of organizational data quality practice: Reactive, Proactive, Automated, Scalable. (Ch 8)

**Data virtualization** — Exposing transactional/operational data as-is for analytical use; explicitly distinct from (and a poor substitute for) a true data mesh. (Ch 9)

**Data warehouse** — A structured, schema-on-write data repository optimized for high-throughput analytical querying. (Ch 1, Ch 2)

**Data lake** — A schema-on-read repository storing raw structured, semi-structured, or unstructured data with high flexibility but higher governance risk. (Ch 1, Ch 2)

**DataOps** — The process of improving data reliability/performance through automation, reducing silos and fostering fault-tolerant analytics; data's analogue to DevOps. (Ch 1)

**Decentralized data architecture** — A central platform team plus distributed analytics/data science teams across business domains (distinct from the data mesh). (Ch 1)

**Entrypoint** — The initial point of contact where outside-world data enters a pipeline (application logs, API responses, sensor data). (Ch 3)

**ETL / ELT** — Extract-Transform-Load (transform before loading, safer) vs. Extract-Load-Transform (load first, faster but riskier without strong monitoring). (Ch 3)

**False positive / False negative** — Alerting when nothing is wrong ("crying wolf") vs. missing a real issue ("sleeping guard dog"); the two failure modes of any anomaly detector. (Ch 4)

**Field-level lineage** — Lineage tracked down to individual columns/fields rather than whole tables, enabling fast, precise root cause analysis. (Ch 7)

**Five pillars of data observability** — Freshness, distribution, volume, schema, lineage: the complete measurement framework for data health. (Ch 4, Ch 5)

**F-score (Fβ)** — A weighted harmonic mean of precision and recall; β>1 favors recall, β<1 favors precision. (Ch 4)

**Hub-and-spoke model** — A team structure where a centralized data platform team owns infrastructure/quality while decentralized embedded analysts own semantic layers and business application. (Ch 8)

**Incident commander** — A designated role responsible for flagging incidents, coordinating response, and assessing severity during data downtime events. (Ch 6)

**Known unknowns / Unknown unknowns** — Predictable data quality issues coverable by tests vs. unpredictable issues that evade testing and require monitoring. (Ch 4)

**Lineage** — A map/record of a data set's journey from ingestion to analysis, tracing upstream sources and downstream dependencies. (Ch 2, Ch 7)

**Normalization (data)** — The first transformation layer resolving heterogeneous entrypoint data into a consistent format (time zones, types, schema). (Ch 3)

**Operational data** — Data produced by day-to-day business operations (inventory, transactions); runs the business, as distinct from analytical data. (Ch 2)

**Phantom data** — Deprecated or unused data whose breakage triggers wasted firefighting effort because it no longer matters to the business. (Ch 6)

**Precision / Recall** — Precision = TP/(TP+FP) (of alerts raised, how many were real); Recall = TP/(TP+FN) (of real issues, how many were caught). (Ch 4)

**RACI matrix** — Responsible/Accountable/Consulted/Informed framework for assigning data quality ownership across data personas. (Ch 8)

**Root cause analysis (RCA)** — The five-step process (lineage → code → data → operational environment → peers) for diagnosing a data incident's underlying cause. (Ch 6)

**Runbook / Playbook** — A runbook gives step-by-step instructions for using services/handling known issues; a playbook provides broader incident-category response processes. (Ch 6)

**Schema change** — Any alteration to data's structural blueprint (added/removed columns, type changes); a major, hard-to-manually-track source of pipeline breakage. (Ch 4)

**Seasonality** — Predictable periodic fluctuation in a time series (e.g., weekly or yearly) that naive anomaly thresholds must account for. (Ch 4)

**SLA / SLO / SLI** — Service Level Agreement (customer promise + penalty for failure), Service Level Objective (target value), Service Level Indicator (the specific measured number). (Ch 5)

**STM (Source to Target Mapping)** — A spec (used at Fox Networks) letting analysts define requirements so engineers can build pipelines from a well-defined playbook. (Ch 7)

**Swampification** — The tendency of data lakes to accumulate technical debt and tacit knowledge until no one can navigate them ("data swamp"). (Ch 2)

**Table sprawl** — Uncontrolled proliferation of tables/connectors (e.g., from automated ingestion tools) that outpaces manual tracking. (Ch 9)

**Throughput vs. latency trade-off** — The inherent trade-off, for any system with fixed computational power, between data volume processed per unit time and delay before processing. (Ch 2)

**Time to Detection (TTD) / Time to Resolution (TTR)** — How long it takes to surface a data quality issue, and how long it takes to fix it once detected. (Ch 5)
