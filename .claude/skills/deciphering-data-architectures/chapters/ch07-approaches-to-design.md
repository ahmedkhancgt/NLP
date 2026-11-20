# Chapter 7: Approaches to Design

## Core Idea
Data design (deciding where data lives, flows, and is processed — "building the city") is distinct from data modeling (Ch 8, "designing the buildings"); this chapter surveys OLTP vs. OLAP, SMP vs. MPP, Lambda vs. Kappa architecture, and polyglot persistence as complementary design strategies to combine based on need, not a single "best" answer.

## Frameworks Introduced
- **OLTP vs. OLAP**: the transactional/analytical processing split.
  - When to use: OLTP for real-time CRUD operational systems (POS, ecommerce, banking); OLAP for fast multi-dimensional analytical queries/reporting.
  - How: OLTP uses relational DBMSs optimized for concurrency/low latency; OLAP uses pre-aggregated cubes (multidimensional, via star/snowflake schema) or tabular models for fast slice-and-dice without long-running queries; OLTP sources typically feed a DW which feeds an OLAP database.
- **SMP vs. MPP**: scale-up vs. scale-out database processing designs.
  - When to use: SMP ("scale up," add processors/memory to one server) suits OLTP; MPP ("scale out," add more servers each with own memory/disk) suits DW-scale analytical workloads with huge data volumes.
  - How: MPP splits queries at a control/name node into subqueries sent to compute/worker nodes, each querying its own data shard, with results merged back.
- **Lambda Architecture**: a dual batch+real-time processing design.
  - When to use: need both the trusted, comprehensive "single source of truth" (batch) and low-latency "I want it now" views (stream) — e.g., recommendation engines, fraud detection.
  - How: three principles — dual data model (separate batch layer and stream/speed layer), single unified presentation/serving layer merging both, and decoupled processing layers (scaled/developed independently). Presentation layer defaults to the batch layer (trusted) but falls back to the stream layer for up-to-the-second needs.
- **Kappa Architecture**: a real-time-only simplification of Lambda.
  - When to use: need real-time processing at scale without a batch layer — e.g., streaming platforms, financial trading systems.
  - How: three principles — real-time processing (no batch delay), single event stream (all data flows through one stream for easy scaling/fault tolerance), stateless processing (each event processed independently).
- **Polyglot Persistence / Polyglot Data Stores**: using the right storage technology per data type.
  - When to use: an application or organization handles multiple distinct data shapes/use cases (e.g., shopping carts, orders, inventory, social graphs) that don't fit one storage paradigm well.
  - How: polyglot persistence picks different store types (key-value, document, relational, graph) within one application; polyglot data store applies the same idea at the org level (different projects/business units use different optimized stores).

## Key Concepts
- **CRUD vs. "write-once, read-many"**: OLTP is CRUD-heavy; OLAP is optimized almost purely for reads.
- **OLAP cube**: pre-aggregated, summarized data organized by dimensions (time, location, product) for fast multi-angle analysis without long queries.
- **Tabular data model**: a newer, simpler alternative to multidimensional OLAP cubes using a relational-table-like structure; the book calls multidimensional models "OLAP cubes" and tabular models "tabular cubes," though "cube" is used interchangeably in industry.
- **Operational data vs. analytical data**: operational data is real-time, high-volume, used for day-to-day snapshots (via OLTP); analytical data is historical, lower-volume, consolidated/aggregated, batch-ingested, used for trends/ML (via OLAP/DW).
- **Semantic layer**: the abstraction a multidimensional/tabular model provides over underlying DW schemas.
- **Push-down/stateless processing**: Kappa's stateless design processes each event independently with no memory of prior events — a deliberate scalability trade-off.

## Mental Models
- "Building a city vs. designing a building": data design (this chapter) is urban planning (where things go, how they connect); data modeling (Ch 8) is architecture of individual structures.
- The card-search analogy for SMP vs. MPP: one faster searcher (Fiona → Max) = scale-up/SMP (diminishing returns); splitting the deck among 26 searchers = scale-out/MPP (dramatic speedup through parallelism).
- Stateless vs. stateful recommendation failure: a Kappa-style stateless system might recommend items already in a customer's cart or miss contextual follow-ups (shoes → socks → shoe polish should suggest shoelaces) because it has no memory of the event sequence.

## Anti-patterns
- **Forcing non-relational data into a relational store (anti-polyglot)**: significantly slows both development and performance — e.g., storing graph-shaped social-network data in a non-graph store.
- **Choosing Lambda when you need pure low-latency stateful real-time processing**: Lambda's stateless design and batch/stream duality add complexity without delivering Kappa's lower latency or state-aware processing.
- **Choosing Kappa when you need robust ad hoc historical queries or batch processing**: Kappa is not designed for large historical batch analysis or ad hoc queries over history — Lambda (or a DW) fits better there.
- **Treating Lambda/Kappa as architectures you must pick exclusively**: they are design patterns implementable within any Part III architecture (MDW, fabric, lakehouse, mesh) depending on whether the solution supports batch+real-time (Lambda) or real-time only (Kappa).

## Reference Tables
| Dimension | OLTP | OLAP / Tabular |
|---|---|---|
| Processing type | Transactional | Analytical |
| Operations | CRUD | Write-once, read-many |
| Optimized for | Concurrency, low latency | Fast multi-dimensional query |
| Typical DBMS | SQL Server, Oracle | Cube / tabular engines |

| Dimension | SMP | MPP |
|---|---|---|
| Scaling direction | Up (more CPU/memory, one server) | Out (more servers, each own memory/disk) |
| Best for | OLTP | DW-scale analytical workloads |
| Examples | SQL Server, Oracle | Teradata, Netezza |

| Dimension | Lambda | Kappa |
|---|---|---|
| Data handled | Batch + real-time | Real-time only |
| Layers | Batch layer + stream/speed layer + presentation layer | Single event stream + presentation layer (no batch layer) |
| State | Mixed | Stateless |
| Good for | Recommendation engines, fraud detection | Streaming platforms, financial trading |
| Weakness | Complexity, limited stateful processing | No batch processing, weak ad hoc historical query support |

## Worked Example
An ecommerce platform applies polyglot persistence: a key-value store for shopping cart/session data (fast retrieval), a document store for completed orders (flexible, fast read/write), an RDBMS for inventory/item prices (structured, relational by nature), and a graph store for customer social graphs (natively suited to graph data) — each data type lands in the store best suited to it rather than forcing everything into one database.

## Key Takeaways
1. Separate "data design" (where/how data flows) from "data modeling" (Ch 8, how individual datasets are structured) conceptually.
2. Match OLTP to operational/transactional needs and OLAP/tabular to analytical/reporting needs; most architectures chain OLTP → DW → OLAP.
3. Choose MPP (scale-out) over SMP (scale-up) once data volumes exceed what single-server scaling can handle, especially for DW workloads.
4. Pick Lambda when you need both a trusted historical view and near-real-time views reconciled in one presentation layer; pick Kappa when you only need real-time, stateless, highly scalable stream processing.
5. Apply polyglot persistence/data stores deliberately — accept the complexity cost of multiple technologies in exchange for using the right tool per data shape.

## Connects To
- **Ch 2**: CRUD and OLTP are introduced there and formalized here.
- **Ch 4**: DW architecture is the typical bridge between OLTP sources and OLAP cubes.
- **Ch 8**: Star/snowflake schemas underlie OLAP cube construction; data modeling is the next chapter's focus.
- **Ch 10–14**: Lambda/Kappa and polyglot persistence are design patterns applicable within MDW, data fabric, lakehouse, and data mesh architectures.
- **External concepts**: Teradata, Netezza (MPP); SQL Server, Oracle (SMP/OLTP).
