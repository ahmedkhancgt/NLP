# Chapter 10: The Modern Data Warehouse

## Core Idea
The Modern Data Warehouse (MDW) combines an RDW's governed, top-down structure with a data lake's flexible, bottom-up staging, with data moving through five defined stages (ingestion, storage, transformation, modeling, visualization); organizations typically reach a full MDW via one of three interim "stepping stone" architectures rather than building it all at once.

## Frameworks Introduced
- **MDW Five-Stage Data Journey**: the canonical data flow through an MDW.
  - When to use: as the reference architecture for any MDW build or evaluation.
  - How: (1) Ingestion — any data type/size/speed, batch or streaming; (2) Storage — lands in the layered data lake (Ch 5); (3) Transformation — compute engines move data raw → conformed → cleansed → presentation within the lake; (4) Modeling — copy all/some presentation-layer data into an RDW, usually in 3NF or a star schema; (5) Visualization — business users report/dashboard from the RDW's easy-to-consume model.
- **Stepping-Stone Architectures**: interim architectures organizations use en route to a full MDW.
  - When to use: when a full MDW is too large an undertaking to build at once, or legacy constraints (on-prem EDW, limited time/budget) require a phased migration.
  - How: choose among EDW Augmentation (add a cloud data lake alongside an unchanged on-prem EDW for big-data use cases), Temporary Data Lake + EDW (use the lake solely as a transient staging/refining area, offloading transformation from the EDW's maintenance window; all querying stays in the EDW), or All-in-One (skip the RDW entirely, query/report straight from the data lake — closely related to the lakehouse, Ch 12).
- **Source-bypass decision**: when to skip the data lake and load structured source data straight into the RDW.
  - When to use: for relational, non-streaming reference/dimension-table data, or migrating existing ETL packages quickly with minimal rework.
  - How: evaluate the trade-off — bypassing the lake saves rework but loses the lake's backup/rerun benefit, "single source of truth" role, and risks extra cleansing burden on the RDW.

## Key Concepts
- **MDW defining feature**: at least some data must be duplicated/replicated from the lake to an RDW — without that duplication, the architecture is a data lakehouse (Ch 12), not an MDW.
- **Separation of storage and compute**: the MDW's core architectural advantage — storing data in the lake and processing it with independently chosen compute tools, enabling flexibility (any cloud/third-party compute engine can attach).
- **Vendor lock-in**: a named con of cloud-based MDW services (Azure Synapse Analytics, Redshift, BigQuery, Snowflake) — dependency on one vendor's ecosystem limiting future flexibility.
- **Self-service BI via MDW**: as data moves through the lake into the RDW, it becomes progressively easier to use, enabling drag-and-drop reporting without joins.
- **~10TB threshold**: the book's rough dividing line — below it, MDW remains popular and sufficient; above it, data fabric/lakehouse architectures (Ch 11, 12) become more attractive.

## Mental Models
- "Blend of both worlds": RDW = structure/governance (top-down, schema-on-write); lake = flexibility/exploration (bottom-up, schema-on-read) — MDW is explicitly the synthesis, not a compromise.
- Stepping-stone architectures as a maturity ramp: EDW Augmentation (add lake, EDW unchanged) → Temporary Data Lake + EDW (offload transformation only) → full MDW (lake + RDW fully integrated) — each stage removes one EDW limitation at a time.
- Decide "does this specific data need the lake at all?" per data source rather than applying the same ingestion path to everything.

## Anti-patterns
- **Trying to use the data lake for all use cases and bypassing the RDW entirely (mid-2010s failure pattern)**: this approach failed broadly and directly motivated the rise of the MDW as the "best of both worlds" correction.
- **Copying relational reference/dimension data through the lake unnecessarily**: doing so loses valuable metadata (data types, constraints, foreign keys) during the round-trip through the lake's less-structured format, for data that didn't need cleaning in the first place.
- **Choosing an expensive MPP RDW for a company with persistently small, well-bounded data volumes**: an SMP solution can be far more cost-effective (see Wilson & Gunkerk case study) when data growth is genuinely modest and well understood.
- **Letting stepping-stone architectures become permanent without planning the next step**: EDW Augmentation in particular "does not help you offload any of the EDW's workload" — if left in place indefinitely, you keep the legacy EDW's limitations while adding lake complexity.

## Reference Tables
| Stepping Stone | Lake Role | Who Queries | Offloads EDW Workload? | Good Fit For |
|---|---|---|---|---|
| EDW Augmentation | Full lake alongside unchanged EDW | Lake (new data) + EDW (legacy) | No | On-prem EDW hitting storage/scale limits |
| Temporary Data Lake + EDW | Transient staging/refining only | EDW only | Yes (offloads transformation) | Reducing EDW maintenance window |
| All-in-One | Sole repository, no RDW | Lake directly | N/A (no EDW) | Startups, technical-expert-only user base, quick prototyping |

| MDW Pros | MDW Cons |
|---|---|
| Multi-source integration | Complexity |
| Scalability | Cost (storage + pipeline duplication) |
| Real-time analysis | Skill requirements |
| Improved performance | Potential data silos |
| Flexibility | Compliance challenges |
| Enhanced security | Vendor dependency |

| Component | Audience | Strengths |
|---|---|---|
| Data lake | Data scientists, power users | Batch + real-time landing, cheap compute flexibility, ELT, backup/sandbox, exploration |
| RDW | Business users, non-technical | Low latency (esp. MPP), complex joins, row/column security, millisecond dashboards, concurrency |

## Worked Example
Wilson & Gunkerk, a fictitious midsized pharmaceutical company with data volume under 1TB and projected growth to only ~10TB, chose a cloud MDW with an SMP (not MPP) RDW rather than a full data lakehouse or MPP solution — minimizing migration disruption from its on-prem system while gaining predictive analytics (e.g., forecasting drug efficacy from patient demographics/genetics) at lower cost, illustrating that MDW remains the right choice for organizations with modest, well-bounded data growth even in 2020s architecture discussions.

## Key Takeaways
1. An MDW's defining trait is deliberate data duplication between lake and RDW — remove that duplication and you have a lakehouse instead.
2. Follow the five-stage data journey (ingest → store → transform → model → visualize) as the default MDW blueprint, but allow justified exceptions (e.g., bypassing the lake for clean relational reference data).
3. Use stepping-stone architectures (EDW Augmentation, Temporary Data Lake + EDW, All-in-One) as deliberate phases toward a full MDW, not permanent end states, unless the business genuinely doesn't need more.
4. MDW remains the right architecture below roughly 10TB of data or where simplicity/cost outweighs the benefits of data fabric or lakehouse.
5. Match RDW compute technology (SMP vs. MPP) to genuine, well-understood data growth rather than defaulting to the most powerful (and expensive) option.

## Connects To
- **Ch 4, 5**: MDW is explicitly the combination of RDW (Ch 4) and data lake (Ch 5) concepts.
- **Ch 7**: SMP vs. MPP choice for the RDW component is a direct application of Ch 7's framework.
- **Ch 11**: Data fabric is described as the natural evolution once an MDW needs more technology (real-time, access policies, metadata catalog) — picked up next chapter.
- **Ch 12**: Removing the RDW's data duplication from an MDW yields the data lakehouse architecture.
- **External concepts**: Azure Synapse Analytics, Amazon Redshift, Google BigQuery, Snowflake — leading MDW platform providers.
