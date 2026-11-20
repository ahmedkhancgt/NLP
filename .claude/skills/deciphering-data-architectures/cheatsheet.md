# Cheatsheet

## Architecture Decision Tree
1. **Small data (<10TB), team comfortable with RDWs, gentle transition needed?** → Modern Data Warehouse (MDW)
2. **Growing/diverse multi-source data (varying size/speed/type), need seamless integration, large volume (>10TB)?** → Data Fabric (add ≥3 of: access policies, metadata catalog, MDM, virtualization, real-time, APIs, services, products)
3. **Want a flexible single-repository default; comfortable trading some RDW features for simplicity/cost?** → Data Lakehouse (use until you hit its limits; then selectively copy specific datasets to an RDW)
4. **Very large, domain-oriented enterprise, genuine organizational/technical scaling crisis, resources/time for multi-year transformation?** → Data Mesh (via gradual hub-and-spoke adoption — never a one-shot conversion)
5. **None of these fit perfectly?** → Combine features from multiple architectures; most real solutions are mosaics, not pure forms.

## RDW vs. Data Lake vs. Lakehouse — Quick Trade-off Matrix
| Need | RDW | Data Lake | Lakehouse |
|---|---|---|---|
| Structured, governed, fast self-service BI | Best | Weak | Good (via serving layer) |
| Raw/unstructured/streaming data | Poor | Best | Good |
| Historical "as of" accuracy (SCD Type 2) | Best | Poor | Moderate |
| Millisecond dashboard latency | Best (MPP) | Poor | Moderate (copy hot data to RDW if needed) |
| Row/column security, encryption, masking | Native | None | None natively — add externally |
| Multi-table ACID | Native | None | Single-table only |
| Cost at small scale | Moderate-High | Low | Low (serverless, pay-per-query) |

## ETL vs. ELT — When to Use Which
- **ETL**: source + destination both relational, smaller datasets, need strict pre-load data quality/security control.
- **ELT**: large-scale, unstructured/semi-structured data destined for a lake; need to iterate transformations without re-extracting.

## Batch vs. Real-Time — Latency Tolerance Test
- High latency tolerance (hours/overnight OK) → **Batch**
- Low latency tolerance (seconds/milliseconds required — fraud detection, IoT, trading, traffic) → **Real-time**
- Default assumption: most data does NOT need real-time. Don't over-provision for latency you don't need.

## SMP vs. MPP
- Small, well-bounded, non-growing data volume → **SMP** (cheaper, "scale up")
- Large/growing analytical workloads → **MPP** (scale out, e.g. Teradata, Netezza, cloud MPP)

## Data Mesh Readiness Red Flags (signs you're NOT ready / it's not really a mesh)
- You've only reorganized data by department/acquisition with no domain team building its own analytical solution → **data federation, not a mesh** (principle #1 only).
- IT (not domains) is still doing the data cleaning → **missing principle #2**, forfeiting the mesh's core benefits.
- Connecting lakes/warehouses with data virtualization alone → **not a mesh**, satisfies none of the 4 principles.
- Each domain building its own infrastructure from scratch with no central platform team → **violates principle #3** intent (duplication, inconsistency, delay).
- No central team setting global governance rules → **violates principle #4** (inconsistent security/quality standards across domains).
- Most of your "responsibility dots" (Fig 14-4 style) sit on the centralized-IT side → more accurately an **enterprise data fabric or lakehouse**, not a mesh.
- Less than ~1% of organizations meet Dehghani's full readiness bar (all criteria somewhat/highly applicable) — assume you probably aren't ready unless proven otherwise.

## Data Mesh Myths (reject these claims)
1. "Mesh is a silver bullet" → False; it takes LONGER than any other architecture here.
2. "Mesh replaces your lake/warehouse" → False; you get MORE lakes/warehouses, decentralized.
3. "DW projects fail because they can't scale, so mesh fixes that" → False; failures are people/process issues that would sink a mesh too.
4. "Mesh decentralizes everything" → False; principles #3 and #4 are explicitly centralized functions.
5. "Data virtualization alone = a data mesh" → False; satisfies none of the four principles (it's data federation).

## Architecture Design Session (ADS) Red Flags / Smells
- Jumping to a product demo before discovery questions are complete.
- Running the session with slides instead of a whiteboard.
- No clear deliverable (blueprint + action plan) defined upfront.
- Skipping scheduled breaks or letting one person dominate the discussion.
- No pain points/goals list tracked and checked off by session end.

## Project Failure Smells (People & Process)
- Executives assume BI/data projects are "easy" → set unrealistic timelines.
- Choosing tech by familiarity ("hammer looking for a nail") instead of evaluating fit.
- Requirements phase dragging on for many months with no development in parallel.
- Shipping a report/dashboard without validating totals against known-good production data first.
- Consultants who show senior experts in the sales pitch, then staff junior/offshore teams.
- Mid-project budget cuts hitting staff, training, or hardware instead of scope.
- Fixed end-date-backward planning compressing discovery and quality work.
- Data warehouse modeled after source-system shape instead of business needs/KPIs.
- No project champion (executive sponsor) or team utilization far below ~80%.

## Thresholds & Defaults (from the book)
- **~10TB**: rough dividing line below which MDW remains sufficient; above which data fabric/lakehouse become more attractive.
- **≥3 of 8 features**: author's heuristic for when an MDW becomes a "data fabric."
- **~1% of companies**: author's estimate meeting Dehghani's full data mesh readiness bar.
- **80% team utilization**: target project-management efficiency heuristic (20% reserved for overhead/leave/learning).
- **87%**: cited statistic — data science projects that never reach production (VentureBeat, 2019).
- **3NF**: default target normal form for OLTP/RDW relational models.
- **Mesh Type 1** (shared tech, one physical lake, logically separated): the most commonly adopted mesh topology in practice; Mesh Type 3 (fully independent per-domain tech) the author predicts will never be widely adopted.

## Multi-Cloud Quick Test
- Genuine regional/data-sovereignty requirement one CSP can't meet → justified.
- Hoping for negotiating leverage → mostly a myth (volume discounts reward single-CSP consolidation).
- Hoping for better redundancy/DR → use cross-region failover within ONE CSP instead.
