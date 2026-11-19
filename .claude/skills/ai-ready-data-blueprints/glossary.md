# Glossary

**Agent optimization (AO)** — The successor discipline to SEO: structuring data as machine-readable, verifiable attributes so AI shopping/research agents (not humans) can discover and select an offering. (Ch 1)

**Agentic AI** — The fourth stage of GenAI evolution: systems that autonomously plan and execute complex tasks with minimal human supervision, requiring permission frameworks, monitoring, feedback loops, and trust verification. (Ch 1)

**Agentic RAG** — The most sophisticated RAG architecture tier, embedding autonomous planning, reflection, and tool-use to orchestrate retrieval and generation dynamically. (Ch 5)

**ANN (Approximate Nearest Neighbor)** — Search algorithms that trade a small amount of recall for dramatically faster similarity search vs. brute-force exact search (10–100x speedup, 90–99% recall). (Ch 5)

**Automated reasoning tokens** — Explicit tokens/phrases (e.g., "think step by step") that activate deliberate, systematic reasoning in a model rather than an immediate answer. (Ch 6)

**Business glossary** — Standardized definitions, usage examples, lineage, and category tags mapping business concepts to data fields so humans and AI share meaning. Must be verbose and non-overlapping for AI agents to use reliably. (Ch 2, Ch 4)

**Business autopilot** — Autonomous execution of core business processes (supply chain, marketing, customer service, finance, product dev) with minimal human intervention. (Ch 1)

**Chain-of-thought (CoT)** — Reasoning framework breaking a problem into a logical sequence of intermediate steps; useful for multistep/math tasks, wasteful overhead for simple tasks. (Ch 6)

**Chunking (semantic)** — Splitting documents by coherent meaning boundaries rather than arbitrary size; the single highest-leverage RAG optimization (up to 35% performance swing). (Ch 3, Ch 5)

**Context engineering** — Using RAG to provide proprietary data as context to foundation models, requiring low-latency semantic search with real-time updates. (Ch 1)

**Context window** — The fixed-length span of tokens a model can attend to at once; drives the need for retrieval infrastructure to surface relevant context within that limit. (Ch 1, Ch 6)

**Data lifecycle management** — End-to-end management of data from creation through storage, usage, archival, and deletion, ensuring quality, security, and compliance throughout. (Ch 4)

**Data mesh** — Federated, domain-owned knowledge graphs/data products (e.g., customer graph, product graph) that agents discover and query autonomously, avoiding centralized bottlenecks. (Ch 1)

**Data product thinking** — Managing each dataset as a product with explicit ownership, defined consumers, measurable quality standards, and lifecycle — not as a byproduct of operations. (Ch 2)

**Data sovereignty** — Jurisdiction-specific rules (GDPR, LGPD, APAC frameworks) on where AI-processed data may be stored/transferred; compliance responsibility stays with the organization even on cloud infrastructure. (Ch 2, Ch 4)

**Data stewardship** — Assignment of responsibility to individuals for overseeing definition, quality, accessibility, and integrity of data; in AI contexts, stewards must build verbose, agent-usable business glossaries. (Ch 4)

**Differential privacy** — Privacy-preserving ML technique injecting calibrated noise (Laplace/Gaussian) to prevent individual re-identification in training data/outputs. (Ch 2)

**Dimensionality (embedding)** — The number of values in a vector embedding; higher dimensions improve recall/nuance but cost linearly more in storage, memory, and latency. (Ch 5)

**Event-driven architecture** — Architectural pattern replacing batch processing with streaming pipelines (Kafka, Kinesis) so agents always work with current, non-stale data. (Ch 1)

**Federated learning** — Training approach keeping personal data on-device (e.g., Apple's keyboard prediction) while still allowing the global model to learn, preserving privacy. (Ch 2)

**Fine-tuning** — Adapting a pretrained foundation model with domain-specific labeled datasets; requires versioned data with lineage tracking. (Ch 1)

**GraphRAG** — Retrieval architecture combining vector indexes (semantic similarity) with graph traversal (explicit relationships) in one system, surfacing connections pure embedding search would miss. (Ch 5)

**Graph-of-thought (GoT)** — Reasoning framework using network-based relationships where concepts/steps influence each other non-linearly (e.g., medical diagnosis combining symptoms, history, tests). (Ch 6)

**Guardrails** — Enterprise-grade constraint layer (e.g., Amazon Bedrock Guardrails) applied before/after an LLM to restrict topics, filter profanity, and redact PII, scaling better than embedding all rules in a prompt. (Ch 4)

**HITL (Human-in-the-loop)** — Mandatory or conditional human review embedded at specific stages of an AI lifecycle (always required for data labeling and evaluation/validation). (Ch 4)

**HyDE (Hypothetical Document Embeddings)** — Query transformation technique that generates a hypothetical ideal answer first, then searches for documents similar to it; demonstrated 38% relative NDCG@10 gain in the original study. (Ch 5)

**Intelligent semantic metadata layer** — A knowledge-graph-based layer (analogous to a Hive Metastore) that tells AI agents what enterprise data means and how it relates, enabling queries on raw files in place without migration to a vector store. (Ch 6)

**Knowledge graph** — A network of entities (nodes) connected by typed relationships (edges), enabling multihop reasoning and inference that vector embeddings alone cannot represent. (Ch 1, Ch 2, Ch 3, Ch 5)

**Lakehouse** — Architecture converging data lakes and data warehouses via open table formats (Iceberg, Delta Lake, Hudi), supporting ACID transactions, schema evolution, and both structured and unstructured data in one platform. (Ch 1, Ch 3)

**LLMOps** — Practices, tools, and methodologies for managing the full lifecycle of LLM applications: development, deployment, and ongoing operations, with orchestration as its backbone. (Ch 4)

**Master data management (MDM)** — Processes/tools for maintaining a single, consistent, accurate view of critical business entities (customers, products) across the organization. (Ch 4)

**Matryoshka Representation Learning (MRL)** — Embedding training technique where meaningful representations exist at multiple dimensionalities within the same vector, allowing truncation at query time without recomputation. (Ch 5)

**MCP (Model Context Protocol)** — Infrastructure/standard for managing agent state, session memory, and reasoning strategies in real time, and for tool integration across agent frameworks. (Ch 2, Ch 6)

**Ontology** — Formalized rules, constraints, and entity relationships (e.g., drug-disease-pathway) enabling AI to reason about complex business scenarios like eligibility or compliance; should be mutually exclusive and collectively exhaustive (MECE). (Ch 1, Ch 2, Ch 3, Ch 5)

**Query drift compensation (QDC)** — Projecting new query embeddings back into an older embedding space via a lightweight transformation matrix, avoiding full corpus re-embedding during model transitions. (Ch 5)

**RAG (Retrieval-Augmented Generation)** — Architectural pattern combining LLM generation with information retrieval to ground responses in factual, up-to-date enterprise knowledge, reducing hallucinations. Introduced by Meta in 2020. (Ch 1, Ch 5)

**Reasoning tokens** — Explicit token budget allocated for a model's internal step-by-step reasoning; counts toward output limits and must be empirically calibrated against latency/cost. (Ch 6)

**Reranking (cross-encoder)** — A second-pass precision step applying more sophisticated attention-based models to an initial retrieval candidate set, improving relevance ordering at added latency cost. (Ch 5)

**Responsible AI** — Practice of designing/deploying AI ethically, transparently, fairly, and accountably; operationalized via six principles: fairness, transparency, accountability, privacy/security, reliability, human oversight. (Ch 4)

**Self-supervised learning** — Training method where a model teaches itself by predicting masked/missing elements in raw, unlabeled data — the core learning methodology behind foundation models. (Ch 1)

**Semantic chunking** — See Chunking.

**Semantic layer** — A centralized layer (ideally ontology-based, OWL/RDF) holding business logic/definitions as a single source of truth, decoupled from individual applications. (Ch 2, Ch 3)

**Semantic normalization** — Ensuring different terms for the same concept (e.g., "customer"/"client"/"account holder") are recognized as one semantic entity across data sources. (Ch 3)

**Tree-of-thought (ToT)** — Reasoning framework exploring multiple competing approaches/branches before selecting the optimal path (e.g., evaluating agile vs. waterfall vs. hybrid project methodology). (Ch 6)

**Vector database** — Specialized database storing high-dimensional vector embeddings optimized for similarity search at scale; leading examples: Pinecone, Weaviate, Milvus, Chroma, pgvector. (Ch 1, Ch 5)

**Vector embedding** — A high-dimensional numeric representation of content (text, image, audio) capturing semantic meaning, enabling similarity-based rather than keyword-based retrieval. (Ch 1, Ch 3, Ch 5)
