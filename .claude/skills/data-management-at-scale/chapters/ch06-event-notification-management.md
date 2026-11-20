# Chapter 6: Event and Notification Management

## Core Idea
Event-driven architectures enable real-time, loosely-coupled, asynchronous data distribution that complements (not replaces) batch data products and APIs, but they introduce significant complexity (consistency models, ordering, retention, governance) that must be deliberately managed with the same domain-ownership and segregation principles used elsewhere in the book.

## Frameworks Introduced
- **Event Governance Model** (author's framework): Mirrors data product/API governance — strict segregation between internal-application event streams (technical, consumable only by the producing domain) and domain-to-domain business streams (read-optimized, versioned, ubiquitous-language-based, cataloged).
  - When to use: Any time an event broker or streaming platform is shared across domains.
  - How: Segregate via separate services/landing zones, naming conventions, dedicated event hubs, or consumer groups; register all streams (ownership, schema, versioning) in an event/schema registry.
- **EDA Reference Architecture (progressive layering)**: A staged design — (1) event brokers decoupling producers/consumers with stateless routing functions, (2) + data product architecture for persistence, (3) + event stores for queryable long-term state, (4) + streaming analytics components for CEP.
  - When to use: Building out an event-driven architecture incrementally rather than deploying everything at once.
  - How: Start with brokers + functions for decoupling and validation; add persistence/data-product ingestion; add event stores (e.g., NoSQL with change feed) for queryable state; add CEP/analytics components last, only for domain-agnostic generic needs.
- **Consistency & Delivery Guarantee Framework**: Classifies streaming correctness trade-offs along two axes — consistency (eventual vs. strong) and delivery (at-least-once, exactly-once, at-most-once).
  - When to use: Whenever designing SLAs for an event pipeline.
  - How: Default to eventual consistency across domain boundaries; use exactly-once only when business-critical; reconcile with CDC/extra validation services as needed.

## Key Concepts
- **Event-carried state transfer**: The event itself contains full state so the consumer never needs to call back the producer (vs. a bare notification that only signals "something changed").
- **Message queue vs. event broker**: Queues are for transactional, exactly-once/ordered, point-to-point delivery (with dead-letter queues); event brokers provide direct multi-destination streaming/routing (topics/event hubs/streams) and can validate, store, route.
- **Simple (SEP) / Stream (ESP) / Complex (CEP) event processing**: Increasing levels of sophistication — SEP reacts to single discrete events; ESP considers event history/sequences; CEP finds patterns across multiple streams/time windows (CEP is a subset of ESP).
- **Application-generated vs. database-generated events**: Events can originate from app/API instrumentation (fire-and-forget, e.g. UI clicks) or from the database layer via polling, triggers, or Change Data Capture (CDC).
- **Event store**: A database (often schemaless NoSQL) retaining all events for replay/rewind, lookup, or enrichment — distinct from a message queue because messages are not deleted after consumption.
- **Backpressure**: The failure mode when a system receives events faster than it can process them; mitigated by scaling resources or setting queue limits/overflow handling.
- **Dead letter queue**: Holding area for undeliverable/malformed messages requiring manual developer investigation.
- **Command sourcing**: Persisting commands before processing them, enabling full system-state replay and auditability.

## Mental Models
- **SOA vs. EDA distinction**: SOA emphasizes commands and avoids inconsistency through service isolation (loose coupling, limited shared-resource dependency); EDA emphasizes events and explicitly tolerates data inconsistency/replication in exchange for responsiveness and decoupling.
- **Listen instead of talk**: Event-driven thinking inverts the natural human expectation of immediate response (synchronous request/response) — architects and consumers must adopt a reactive mindset where "someone will contact you."
- **Progressive architecture layering**: Don't deploy the full streaming analytics stack on day one — add brokers, then persistence, then event stores, then analytics, each layer justified by a concrete requirement, matching the book's broader "start small, standardize gradually" philosophy.

## Anti-patterns
- **Using the streaming platform as a primary application backend** (e.g., Kafka as the system of record): Creates tight coupling between enterprise data distribution and application-internal concerns, and risks becoming an "integration database" by another name.
- **Streams derived from streams without control**: Allowing arbitrary re-ingestion of consumer-created streams back into the platform creates a "chaotic web of event streams on top of event streams."
- **Treating event-driven as free of trade-offs**: Not every problem needs real-time streaming — if there's no genuine need for speed, the added complexity isn't justified; validate business value before adopting EDA.
- **Ignoring retention/compliance limits on long-term event stores**: Infinite retention without governance risks runaway cost and GDPR/data-lifecycle violations.
- **Mixing internal (technical) and domain-to-domain (business) event streams** without segregation: Risks other domains consuming raw, unstable internal communication channels.

## Code Examples
```json
// Event-carried state transfer (full state in the event)
{
  "id": "321125215",
  "firstName": "Adolfo",
  "lastName": "Prosacco",
  "address": { "city": "Baldwin", "country": "United States" },
  "phoneNumber": [{ "type": "mobile", "number": "(202) 555-0154" }]
}

// Bare notification (reference only — consumer must call back for details)
{ "id": "321125215" }
```
- **What it demonstrates**: The two fundamental event payload shapes — carried-state events avoid a callback but risk payload bloat/staleness, while notification events are minimal but create a dependency on a follow-up call to the producer.

## Reference Tables
| Consistency Model | Guarantee | Best For |
|---|---|---|
| Eventual consistency | Data converges "eventually," weak guarantee | Cross-domain distribution, non-critical data |
| Strong consistency | All replicas updated/locked in order before confirming | Business-critical, low-velocity use cases only |

| Delivery Model | Guarantee |
|---|---|
| At-least-once | No message loss; duplicates possible |
| Exactly-once | No loss, no duplicates (requires extra engineering) |
| At-most-once | No guarantees; loss or duplication both possible |

| Event Processing Style | Scope |
|---|---|
| Simple (SEP) | Single, isolated event → isolated reaction |
| Stream (ESP) | Sequence of events, history-influenced | 
| Complex (CEP) | Multi-stream pattern/correlation detection (subset of ESP) |

## Worked Example
A progressive Azure-based EDA (Event Hubs + Functions): Stage 1 — two stream-processing services with stateless functions decouple producers from consumers, enabling domains to evolve/break schemas without breaking all consumers and allowing regional brokers for distributed consumers. Stage 2 — a data product architecture is added so events can be persisted (either captured raw at ingestion for debugging/ad hoc analysis, or tapped at the consumption layer for near-real-time data product updates, e.g., hourly instead of daily). Stage 3 — Cosmos DB (NoSQL) event stores are added on producing and/or consuming sides for queryable lookups, enrichment, and replay/rewind (repair-then-replay pattern). Stage 4 — streaming analytics components (e.g., fraud-detection scoring, geospatial processing via Databricks) are layered in for CEP use cases like real-time dashboards or anomaly alerts.

## Key Takeaways
1. Apply the same domain-ownership and segregation principles to events as to data products and APIs: internal streams stay internal; domain-to-domain streams are governed, versioned, and cataloged.
2. Build EDA incrementally — brokers first, then persistence, then event stores, then analytics — rather than deploying a maximal stack up front.
3. Default to eventual consistency and at-least-once delivery across domain boundaries; reserve strong consistency/exactly-once for cases that truly require it.
4. Never let the streaming platform become an unintended application backend or integration database.
5. Events are immutable by definition (they represent things that already happened) — don't try to "edit" them; use replay/rewind with corrective new events instead.
6. Validate that real business value (genuine need for speed) justifies the added complexity before adopting event-driven architecture for a given use case.
7. Standardize schemas and adopt a contract-first approach (schema/event registry, AsyncAPI) early, before stream counts grow into the hundreds or thousands.

## Connects To
- **Ch 1**: Directly implements the operational/analytical integration trend and real-time decision-making needs raised there.
- **Ch 4**: Event-carried state transfer and CQRS read-stores connect directly to data product architecture; event stores can double as data product stores.
- **Ch 5**: EDA is explicitly positioned as the asynchronous counterpart to synchronous API/SOA communication, sharing the same governance philosophy.
- **Ch 7**: The next chapter zooms out to connect all three integration patterns (data products, APIs, events) into one coherent enterprise architecture view.
