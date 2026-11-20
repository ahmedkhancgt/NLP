# Patterns

## Architecture-First, Technology-Second
**When to use**: Starting any new data project, before evaluating vendors or tools.
**How**: Finalize the architecture blueprint (what, why, when — trade-offs, reversibility) before selecting specific technologies (the how).
**Trade-offs**: Slower initial start than "just pick a tool," but avoids costly rework and bear-trap lock-in later.

## Reversible Decisions (Two-Way Doors)
**When to use**: Any major technology or structural decision, especially under uncertainty.
**How**: Default to choices that can be cheaply undone; reserve irreversible ("one-way door") commitments for rare, carefully vetted cases.
**Trade-offs**: May sacrifice some performance/integration benefits of a fully committed, specialized solution in exchange for long-term flexibility.

## Data Maturity Calibration
**When to use**: Deciding what a data team should prioritize at a given company stage (starting/scaling/leading).
**How**: Stage 1 — buy-in + minimal architecture + off-the-shelf tools; Stage 2 — formalize DataOps + scalable architecture; Stage 3 — automation + custom tools only where competitive advantage exists.
**Trade-offs**: Over-building for your stage wastes resources; under-building creates technical debt that compounds.

## Type A / Type B Engineering Posture
**When to use**: Build-vs-buy decisions for any new capability.
**How**: Default to Type A (abstraction/managed services); invest in Type B (custom build) only where it's a genuine competitive differentiator.
**Trade-offs**: Type A loses some customization/control; Type B costs more engineering time and carries more maintenance burden.

## Log-Based Continuous CDC
**When to use**: Need near-real-time replication or full change history with minimal source-system impact.
**How**: Tail the database's write-ahead log (or use a managed CDC trigger) rather than polling timestamp fields.
**Trade-offs**: More complex to set up than batch CDC; batch/timestamp-based CDC is simpler but misses intra-window changes.

## Insert-Only / Append-Only History
**When to use**: Historical state matters (audits, address history, slowly changing records).
**How**: Insert new versioned rows with timestamps instead of updating in place; query finds the latest version by key.
**Trade-offs**: Table grows unbounded without purging; "latest state" lookups can be expensive (`MAX(timestamp)` scans).

## Hot/Warm/Cold Data Tiering
**When to use**: Managing cost vs. access-latency trade-offs for growing datasets.
**How**: Place frequently accessed data in fast/costly tiers (SSD/memory), infrequently accessed data in slow/cheap tiers (HDD/object storage/archival); automate migration via lifecycle policies.
**Trade-offs**: Retrieval from cold storage can be slow and/or costly if access patterns are misjudged.

## Separation of Compute and Storage (with Hybrid Caching)
**When to use**: Large-scale batch/analytical architectures needing elasticity and cost efficiency.
**How**: Default to separated compute/storage (ephemeral clusters reading from object storage); reintroduce colocation via multitier caching or hybrid object storage where raw performance is critical.
**Trade-offs**: Pure separation sacrifices some colocated performance; pure colocation sacrifices elasticity and cost efficiency.

## Schema-on-Write vs. Schema-on-Read
**When to use**: Deciding how strict to be about structure before storage vs. at query time.
**How**: Schema-on-write (warehouse pattern) for consistency and ease of future consumption; schema-on-read (lake pattern, self-describing formats like Parquet/JSON) for ingestion flexibility.
**Trade-offs**: Schema-on-write slows ingestion of novel/evolving data; schema-on-read defers pain to consumption time and risks data-swamp dynamics without discipline.

## Dead-Letter Queue for Streaming Ingestion
**When to use**: Any streaming pipeline with heterogeneous or evolving event payloads.
**How**: Route events that fail ingestion (bad schema, oversized, expired TTL) to a separate queue for diagnosis/reprocessing rather than blocking the main stream.
**Trade-offs**: Adds operational surface (another queue to monitor); without active triage, DLQs become their own silent data-loss risk.

## Idempotent Stream Consumers
**When to use**: Any distributed message/event processing where at-least-once or out-of-order delivery is possible (i.e., almost always).
**How**: Design consumers so reprocessing the same event produces the same end state (e.g., upsert by key rather than blind insert/append).
**Trade-offs**: Adds design complexity; naive append-only consumers are simpler but fragile under duplicate delivery.

## Modeling at the Lowest Grain
**When to use**: Any new data model for batch analytical data.
**How**: Store data at the finest practical resolution (e.g., order-line-item, not daily summary); aggregate on demand downstream.
**Trade-offs**: More storage and potentially slower raw queries; but aggregation is always possible later, de-aggregation is not.

## Conformed Dimensions (Kimball)
**When to use**: Multiple star schemas share overlapping reference entities (customer, date, product).
**How**: Build and reuse a single shared dimension table across fact tables rather than duplicating it per star schema.
**Trade-offs**: Requires upfront coordination across teams/domains; prevents drifting, inconsistent business definitions.

## Semantic / Metrics Layer for Business Logic
**When to use**: A metric or business rule (e.g., "customer," "net profit") must be computed consistently across many teams/tools.
**How**: Define the logic once in a semantic/metrics layer (e.g., LookML, dbt) and generate queries from it everywhere, instead of duplicating logic across ETL scripts.
**Trade-offs**: Adds a layer of indirection and tooling dependency; but prevents costly definitional drift across reports.

## Strangler Pattern for Brownfield Migration
**When to use**: Migrating/replacing an existing legacy architecture.
**How**: Incrementally replace one component at a time, validating impact before moving to the next, rather than a risky big-bang rewrite.
**Trade-offs**: Slower than a full rewrite; requires maintaining both old and new systems temporarily.

## Managed Connectors Over Custom API Code
**When to use**: Ingesting from well-supported sources (common SaaS platforms, standard databases).
**How**: Use managed connector platforms (Fivetran, Airbyte, Matillion) rather than writing/maintaining custom connector code.
**Trade-offs**: Less control over edge-case handling; vendor/platform lock-in risk; but saves substantial engineering time on undifferentiated work.

## Broken-Glass Access for Sensitive Data
**When to use**: Data that must remain available for rare emergencies but shouldn't be routinely accessible.
**How**: Require two-person/time-boxed approval to unlock access, then auto-revoke immediately after the task.
**Trade-offs**: Adds friction/delay during genuine emergencies; necessary trade-off for protecting sensitive data otherwise.

## Active/Negative-Thinking Security Review
**When to use**: Designing any new pipeline, storage system, or access policy.
**How**: Proactively imagine attack/leak scenarios and avoid collecting data with no genuine downstream need ("data cannot leak if it is never collected").
**Trade-offs**: Takes deliberate time upfront; the alternative (reactive security) is far costlier after a breach.

## Data Contracts + SLA/SLO with Upstream Teams
**When to use**: Any production dependency on a source system or data provider outside your direct control.
**How**: Formalize what data, extraction method, cadence, and contacts are agreed upon; define measurable SLOs (e.g., 99% uptime) against a stated SLA.
**Trade-offs**: Requires organizational buy-in and maintenance overhead; absence of contracts leads to silent breakage and trust erosion.
