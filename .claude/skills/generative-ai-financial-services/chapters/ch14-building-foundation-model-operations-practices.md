# Chapter 14: Building Foundation Model Operations Practices

## Core Idea
FMOps (Foundation Model Operations) extends MLOps with new technology, business processes, and personas specific to foundation models, organized across six generative AI lifecycle components (data preparation, model selection, model tuning, deployment/monitoring, application layer, foundational components); success depends as much on operating model and culture as on tools, and TCO spans compute, database/storage, model, human, and tooling costs — not just model-invocation cost.

## Frameworks Introduced
- **MLOps definition (3 pillars)**: Technical (automation via orchestration tools), Business processes (reproducibility, access control, regulatory requirements), Culture (people/mindset change — the most important and most ignored pillar). Outcomes: automation, governance, scalability.
- **FMOps definition**: "technical, business and cultural practices pertaining to foundation models which result in shipping generative AI products faster, while keeping compliance and governance in check" — extends MLOps with FM-specific tools (model evaluation libraries, fine-tuning/PEFT frameworks, vector DB management, FM-specific monitoring, guardrails, prompt stores), business processes (data governance for embeddings, uniform model access, bias detection, compliance, risk management), and new personas (prompt engineer, fine-tuner, prompt tester, GenAI app developer, vector/prompt DB admin, GenAI risk/compliance officer).
- **6-component FMOps lifecycle** (each analyzed via Process / Personas / Software-tools):
  1. **Data preparation**: collection → curation → bias/toxicity check → chunking → embedding → vector store (RAG-specific steps) or data lake (non-RAG).
  2. **Model selection**: define criteria → candidate model selection → data synthesis → model evaluation → human review → post-processing.
  3. **Model fine-tuning** (optional): setup environment → prepare data → fine-tuning → metrics evaluation → model bias check → model registration.
  4. **Model deployment and monitoring**: setup deployment environment → deploy models/adaptors (pruning/quantization/distillation + LoRA/QLoRA adaptors) → model endpoint serving (dynamic batching, micro-batching, streaming, ONNX) → model monitoring/guardrails → post-processing.
  5. **Application layer**: the user-facing chat/app layer with conversation history, auth, session management, caching — UX matters more to end users than backend sophistication.
  6. **Foundational components**: code/artifact versioning, observability/auditability, CI/CD, networking/security — IT-operations baseline shared across all applications, not GenAI-specific.
- **3 Operating Models**: Shared (central environment, higher auditability, slower team velocity), Decentralized (independent team environments, faster but poor standardization/compliance), Hybrid (independent experimentation/dev/test/prod per team + shared CI/CD/orchestration/observability/auditability — "best of both," at the cost of ongoing shared-vs-team-owned boundary decisions).
- **TCO (Total Cost of Ownership) framework — 5 cost categories**: Compute (data processing clusters), Database & Storage (prompt catalog, relational/non-relational DBs), Model (invocation cost + tuning cost), Human (skilled engineers, domain experts, compliance/legal), Tools (vector engine, CI/CD, auth, versioning, observability, cybersecurity, CDN, orchestration, auditability), plus Miscellaneous (ongoing maintenance, networking/domain/firewall costs).

## Key Concepts
- **Model registry**: central, versioned catalog of fine-tuned models + metadata, essential for audits and model governance.
- **Model compression techniques** (reprised from Ch7): pruning, quantization, knowledge distillation — reduce hosting cost/complexity and improve inference speed.
- **Adaptors (LoRA/QLoRA)**: can be kept separate from the base model or merged — ongoing research on which is better, trade-off between latency and performance.
- **Dynamic batching / micro-batching / streaming**: model-serving techniques to optimize throughput and perceived latency.
- **ONNX (Open Neural Network Exchange)**: interoperability format across model frameworks.
- **Threat modeling**: process for identifying potential security threats and mitigations, popular in financial-services networking/security practice.
- **Center of Excellence (CoE)**: a central generative AI practice team that experiments with new models/tools and disseminates learnings to LoBs.

## Mental Models
- "You do not need a Ferrari to go to the supermarket" — choose the model that fits the task's latency/cost/accuracy profile, not the flashiest/largest model.
- Treat FMOps maturity as requiring all three pillars (technology, business process, culture) simultaneously — strong tooling without the business/cultural pieces still fails in a regulated industry.
- Use the "one person, at most two personas" rule of thumb when staffing FMOps roles — too many responsibilities per person degrades execution quality.
- View TCO holistically: model-invocation cost is usually the most-discussed but least representative slice of true solution cost — compute, storage, human, and tooling costs typically dominate.
- When choosing between agents and traditional workflow orchestration, use agents only where the execution path is genuinely dynamic; use deterministic workflow tools where you need certainty in execution steps (echoes Ch6's agent anti-pattern).

## Anti-patterns
- **Running FMOps as pure DevOps**: borrowing DevOps processes wholesale without adapting for GenAI-specific needs (data/model versioning, bias/toxicity checks, prompt catalogs) causes the stack to fall short and require re-architecture.
- **Chasing the single "best" model for all tasks** (echoes Ch6/Ch12): most flagship models still struggle at specific task types (e.g., reasoning/coding at time of writing); no model wins on latency, cost, and accuracy simultaneously.
- **Treating vector-store embeddings as inherently access-controlled**: once documents are chunked/embedded into a vector store, the original document-level permissions are NOT automatically preserved — permissions must be re-added as metadata on chunks/embeddings, or RAG can leak sensitive documents to unauthorized users.
- **Ignoring indemnity and EULA data-use terms when picking a model provider**: especially relevant for open-source models, where the deploying organization bears full responsibility for harmful model output.
- **Assuming a single compliance/governance tool can automate everything**: frameworks provide checklists of what to capture, not automatic capture — governance still requires deliberate process design.
- **Applying uniform governance rigor regardless of risk level**: a math-tutoring chatbot and an investment-advice chatbot carry different risk profiles and should not receive identical governance overhead (risk-based governance, not blanket governance).
- **Letting organizational silos block FMOps adoption**: lacking a central platform team before attempting FMOps forces teams to build personas/structure before tools/process, slowing adoption significantly.

## Code Examples
No executable code in this chapter — it is an organizational/operational framework chapter (processes, personas, tooling categories, cost categories, best practices).

## Reference Tables
**FMOps lifecycle components × what each needs**

| Component | Key processes | Key personas | Key tools |
|---|---|---|---|
| Data preparation | Collection, curation, bias/toxicity check, chunking, embedding, vector store | Data sourcing team, data engineers, ethical AI officers, GenAI platform admin, DB admin, domain expert | Data processing libraries, data lake, workflow orchestration, embedding models, vector stores, high-performance compute |
| Model selection | Define criteria, candidate selection, data synthesis, evaluation, human review, post-processing | Model evaluator, data engineer, ethical AI officer, domain expert, GenAI platform admin, DB admin | Data lake, model evaluation libraries (Hugging Face Evaluate, AWS fmeval), data generation libraries (Faker), orchestration, prompt catalog, web app libraries |
| Model fine-tuning | Setup environment, prepare data, fine-tune, metrics evaluation, bias check, model registration | Data scientist, ethical AI officer, ML engineer, GenAI platform admin, domain expert | Data lake, orchestration, model training libraries, debugging/profiling tools, high-performance compute |
| Deployment & monitoring | Setup env, deploy models/adaptors, endpoint serving, monitoring/guardrails, post-processing | ML engineer, domain expert, GenAI platform admin, risk/compliance officer | Data lake, inference libraries (PyTorch, Accelerate, DeepSpeed, FlashAttention), serving libraries (TorchServe, Triton, DJLServing), monitoring/guardrail tools, prompt catalog, HPC |
| Application layer | (integrated into deployment) | — | Conversation history, auth/authz, session mgmt, caching |
| Foundational components | Code/artifact versioning, observability/auditability | Security specialist, infrastructure specialist | CI/CD, networking/security tools |

**Operating model comparison**

| Model | Advantage | Disadvantage |
|---|---|---|
| Shared | Higher auditability/observability (centralized) | Slows teams down; environment gets convoluted with conflicting requirements |
| Decentralized | Team independence on stack/process/tools | Poor standardization → compliance/governance issues; hard to build central observability |
| Hybrid | Fast team iteration + shared platform leverage | Ongoing friction deciding what's shared vs. team-owned for every new component |

**TCO cost categories**: Compute (data processing clusters) → Database & Storage (prompt catalog, relational/non-relational DBs, intermediate state storage) → Model (invocation: token-based or instance-hosted; tuning: HPC-intensive) → Human (skilled engineers, domain experts, compliance/legal) → Tools (vector engine, CI/CD, auth, versioning, observability, cybersecurity, CDN, orchestration, auditability) → Miscellaneous (ongoing maintenance, networking/domain/firewall).

**Best practices by category**: Tooling (prefer managed offerings; use specialized tools when 70-80% better; pick vendors with strong track records), Model (no universal best model; use prompt/semantic caching; use agents only for dynamic paths, workflow tools for deterministic ones; keep app logic loosely coupled to model version), Cost optimization (right-size self-hosted instances; reserve throughput for serverless; use smaller embedding models; prefer serverless vector engines; auto-shutdown idle resources; meter/throttle per-team in centralized platforms), Security (verify EULA data-use + indemnity terms; least-privilege + metadata-based permissions on vector chunks; private network boundary; prompt guardrails against privilege escalation; encryption at rest/in transit; regular vulnerability scans per OWASP LLM checklist), Governance (choose region-compliant models; capture full lineage of every answer — question, embedding, retrieved context, response; risk-based governance tiers; IP-leak testing pipelines), Organization/culture (define personas, max 2 per person; create a GenAI Center of Excellence; invest in training/hackathons; hybrid operating model with embedded central engineers; internal audit teams).

**FMOps challenges**: lack of organizational awareness (~90% of PoCs per Forbes don't reach production), immature/fast-changing tooling (LangChain/LlamaIndex release velocity), scarcity of FMOps experts, lack of reference implementations/unclear ROI, siloed organizational structure, mistakenly running FMOps as plain DevOps, compliance/governance overhead (e.g., EU AI Act), and the overall fast pace of technology evolution requiring active maintenance.

## Worked Example
**Vector-store permission leak risk (Security best practice)**: In a RAG use-case, a user asks a question whose answer lies in a document they do NOT have read access to in the source system. Once that document has been chunked and embedded into the vector store, the original document-level access control is NOT automatically enforced — the vector store will happily retrieve and surface that chunk to any querying user unless permissions are explicitly re-attached as metadata on the chunk/embedding and enforced at retrieval time. This is a concrete illustration of why "add permissions as metadata with chunks and embeddings" is listed as a specific security best practice, not a generic data-security platitude.

## Key Takeaways
1. FMOps requires technology, business process, AND culture together — none alone is sufficient, and culture is the most commonly neglected pillar.
2. Map every FMOps initiative against the 6-component lifecycle (data prep, model selection, tuning, deployment/monitoring, application layer, foundational components) and staff each with the right personas (max ~2 per person).
3. Choose an operating model deliberately: hybrid is usually the best fit for large financial organizations, balancing team velocity with centralized governance.
4. TCO is dominated by compute, storage, human, and tooling costs — not just model invocation cost; build the full TCO picture before committing budget (echoes Ch6's "incomplete cost analysis" anti-pattern).
5. Vector-store embeddings do not inherit source-document access permissions automatically — this must be explicitly engineered via chunk/embedding metadata.
6. Governance should be risk-based, not uniform: match governance rigor to the use-case's actual risk profile.
7. Expect FMOps-specific challenges (tooling immaturity, expert scarcity, unclear ROI, DevOps-mimicry pitfalls) and plan for active, ongoing maintenance rather than a one-time setup.

## Connects To
- **Ch3**: The model-serving and fine-tuning techniques (LoRA/QLoRA adaptors, PEFT) are operationalized here into production deployment pipelines.
- **Ch6**: The TCO/cost-analysis anti-patterns are formally addressed via this chapter's 5-category TCO framework.
- **Ch7**: Model compression techniques and architectural cost/performance trade-offs from Ch7 recur here in the deployment-phase best practices.
- **Ch13**: Model evaluation, guardrails, and governance frameworks from Ch13 are embedded directly into the model-selection and deployment phases here.
- **Ch1**: Closes the loop on the FMOps concept introduced in Ch1's predictive-AI-vs-generative-AI comparison table (MLOps vs. FMOps).
