# Patterns

## Knowledge Graph for Contextual Intelligence
**When to use**: Agents need to reason across organizational boundaries/siloed systems (e.g., connecting customer, order, product, and support data to answer "why did this customer return three orders?").
**How**: Deploy a graph database (Neo4j, Amazon Neptune); build entity resolution pipelines to merge duplicate entities; maintain ontology management for consistent relationship definitions; continuously sync from source systems. Evolve toward federated, domain-specific graphs (data mesh) as scale grows.
**Trade-offs**: High setup cost (ontology design, entity resolution); but compounding returns — each new source integrated enriches context for every existing node, and multihop reasoning becomes possible that pure embeddings cannot provide.

## Event-Driven Architecture for Real-Time AI
**When to use**: Agents need fresh, non-stale context (inventory, pricing, policy changes) rather than batch-job snapshots.
**How**: Replace nightly ETL with event streaming (Kafka, Kinesis); implement change data capture (CDC); use stream processing (Flink, Spark Streaming) for transformation; push low-latency updates into vector database indexes.
**Trade-offs**: Higher infrastructure complexity and operational overhead than batch, but prevents compliance exposure and stale-answer risk from outdated information.

## Lakehouse as Unified AI Platform
**When to use**: Need both structured analytics and unstructured RAG data joined seamlessly in one platform.
**How**: Adopt open table formats (Apache Iceberg, Delta Lake, Hudi) on object storage; add multi-engine access (Spark, Trino, Athena); apply unified governance (Lake Formation, Unity Catalog); extend to multimodal (text/image/audio/video) over time.
**Trade-offs**: Requires migration/consolidation effort from legacy warehouse/lake split architectures, but eliminates the structured-vs-unstructured choice GenAI can't tolerate.

## Semantic Search as Query Layer
**When to use**: Keyword search misses conceptually related content (e.g., "GDPR compliance" vs. "EU data protection regulations").
**How**: Generate vector embeddings via an embedding model; store/index in a vector database (Pinecone, OpenSearch, pgvector); implement hybrid search combining semantic + keyword + metadata filtering.
**Trade-offs**: Embedding generation/storage costs scale with corpus size and dimensionality; hybrid search adds implementation complexity but is needed for exact-terminology precision.

## Agent-Ready Digital Properties
**When to use**: Business needs to remain discoverable/actionable by external AI agents (shopping agents, research agents), not just human visitors.
**How**: Add structured data markup (Schema.org, JSON-LD); build API-first design (OpenAPI/GraphQL); write machine-readable (not just human-readable) documentation; add auth/rate-limiting and agent-usage analytics.
**Trade-offs**: Requires rethinking content/product data architecture from the ground up; delaying this risks invisibility as "agent optimization" replaces SEO.

## Business Context Intelligence Engine (Blueprint)
**When to use**: Organization needs to systematically capture business logic/decision rationale, not just outcomes, for AI consumption.
**How**: Phase 1 (months 1–2): deploy semantic layer + knowledge graph (e.g., Neptune) + business glossary + taxonomy. Phase 2 (months 3–4): add orchestration (LangGraph/CrewAI/Strands), MCP for agent state, vector DBs, RAG pipelines.
**Trade-offs**: Significant upfront glossary/ontology-development labor (though generative AI can reduce this by 60–70%), but is the prerequisite for agents correctly interpreting any enterprise data field.

## Adaptive Data Quality Orchestration (Blueprint)
**When to use**: Need enterprise-grade data integrity sustained through the full AI lifecycle rather than one-time cleanup.
**How**: Phase 1: deploy validation (Great Expectations), observability (Monte Carlo), orchestration (Airflow) against explicit accuracy/completeness/consistency/timeliness/uniqueness targets. Phase 2: deploy AI agents for automated cleansing, predictive quality monitoring, dynamic validation-rule updates.
**Trade-offs**: Requires continuous monitoring investment; the payoff is a documented 80%+ reduction in manual data-quality intervention in the chapter's benchmarks.

## Security-First AI Data Platform (Blueprint)
**When to use**: Any AI system handling regulated or sensitive data (PII, PHI, financial) that must pass compliance review.
**How**: Phase 1: zero-trust architecture — identity-first RBAC, AES-256 encryption at rest/in transit, AI-aware DLP, PII/PHI tokenization. Phase 2: automated compliance (TrustArc/OneTrust), immutable audit logging, differential privacy, federated learning, homomorphic encryption where needed.
**Trade-offs**: High initial engineering investment, but avoids disqualification in regulated-industry vendor/architecture reviews and reduces compliance reporting time (40% in the chapter's financial-services case study).

## Modular Vector Store Design
**When to use**: Multiple business use cases need different "views" of the same source content (e.g., email headers vs. body) without storing redundant copies.
**How**: Parse source content into separable components → index each component separately (e.g., headers index, body index) → tag each chunk with a shared surrogate-key identifier at ingestion → join results across indexes at query time via postprocessing.
**Trade-offs**: Adds a join step at query time, but yields storage efficiency, faster focused-index retrieval, and independent-component maintainability.

## GraphRAG (Vector + Graph Hybrid Retrieval)
**When to use**: Queries require connecting multiple entities across a relationship chain (e.g., "what compliance requirements affect our derivatives trading desk?") that pure vector similarity would miss.
**How**: Run vector search for semantically relevant documents in parallel with graph traversal for explicit relationship discovery (regulations, products, org units); merge both result sets into a single grounded context for the LLM.
**Trade-offs**: Requires maintaining both a vector index and a graph database in sync; delivers retrieval that neither approach alone can match for multihop, relationship-rich questions.

## Hybrid Search (Dense + Sparse Retrieval)
**When to use**: Enterprise RAG where queries mix semantic concepts and exact terminology (e.g., "Basel III capital adequacy requirements").
**How**: Combine dense vector embedding similarity with sparse keyword matching (BM25) via weighted fusion; demonstrated 35%+ gains over single-method retrieval on broad QA benchmarks.
**Trade-offs**: More implementation complexity (two retrieval paths to tune and merge) than single-method retrieval, but single-method approaches consistently underperform on mixed query types.

## Chunking Strategy Selection by Content Type
**When to use**: Any RAG system — this is the single highest-leverage optimization lever (up to 35% performance swing, 5–6x more impactful than model choice).
**How**: Match strategy to document structure — hierarchical chunking for legal/financial documents (preserve clause/section structure); semantic chunking for medical texts (preserve diagnosis/treatment relationships); QA-pair atomic chunking for FAQs; sentence/turn-pair chunking for conversational logs. Use the content-type-to-strategy mapping rather than a single default.
**Trade-offs**: More sophisticated strategies (semantic, hierarchical, AI-driven dynamic) cost more in preprocessing compute than fixed-size chunking, but the accuracy gain dwarfs model-upgrade gains.

## Query Drift Compensation (QDC)
**When to use**: An embedding model upgrade is needed but full corpus re-embedding is too costly/slow to do immediately.
**How**: Train a lightweight linear projection matrix mapping new-model embeddings into the old model's vector space using a representative sample embedded by both models; apply this transformation to incoming queries at inference time, avoiding re-embedding the existing index.
**Trade-offs**: Computationally cheap (simple matrix multiplication, negligible added latency) but introduces a small accuracy trade-off since the mapping is approximate, not exact — a stopgap before eventual full reindexing.

## Intelligent Semantic Metadata Layer (Raw-Files-in-Place Pattern)
**When to use**: Enterprise unstructured data is too vast (terabytes to petabytes, millions to quadrillions of files), heterogeneous, or sensitive for wholesale vector-store migration, but agents still need to discover/interpret/reason over it.
**How**: Build a knowledge graph (Neptune, Neo4j) mapping entities/relationships/lineage from raw files without migrating the underlying content; agents perform schema discovery first, then traverse relationships to locate and verify source documents, downloading only what's needed; many queries resolve directly from graph metadata without touching source files at all.
**Trade-offs**: Requires ongoing metadata-extraction pipeline investment (LLM-guided entity/relationship extraction) and iterative schema refinement, but avoids embedding-at-scale cost/risk and preserves full source traceability/audit trails.

## Supervisor Agent with Specialized Agents
**When to use**: Multiple distinct task domains (technical support, billing, product info, order status) need routing to specialized handlers under one coordinated interface.
**How**: A central supervisor agent classifies incoming requests (via LLM intent detection or keyword routing) and dispatches to the matching specialist agent, each with its own tools; supervisor consolidates and returns results.
**Trade-offs**: Enables specialization and independent scaling of each agent, but the router becomes a single point of failure/complexity bottleneck as task variety grows — monitor router decision quality as the system expands.

## Large-Context Handling (Segmentation → Inference → Organization)
**When to use**: Input data exceeds a model's context window, or contains enough noise within the window to degrade output quality.
**How**: Segment (chunk) the large input → run inference per segment, parallelized where possible → combine segment outputs via a dedicated combiner process/agent.
**Trade-offs**: Adds pipeline complexity (combiner logic, parallel orchestration) but is the only way to reason over datasets that structurally exceed model capacity, and enables in-context search (no external storage) for security-sensitive content like proprietary codebases.
