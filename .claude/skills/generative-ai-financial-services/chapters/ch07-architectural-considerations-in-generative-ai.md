# Chapter 7: Architectural Considerations in Generative AI

## Core Idea
Every generative AI application has five common components (web interface, prompt, model, storage, database) and must be evaluated against four architectural lenses (security & privacy, tenancy, cost, performance & scalability) — in financial services there are no "best" solutions, only trade-offs, and tenancy choice (single vs. multi) is the central organizational trade-off.

## Frameworks Introduced
- **Security vs. Privacy distinction**: Security = protecting systems/data/services from unauthorized access, misuse, malicious attack (confidentiality, integrity, availability). Privacy = protecting personal/financial data from unauthorized disclosure/misuse (data minimization, purpose limitation, consent). Data can be secure but not private (e.g., bank legally shares customer data with marketers under EULA — securely transmitted, but no longer private).
  - When to use: as a diagnostic — a data breach is a security issue; legally sharing data without consent is a privacy issue.
- **Security/privacy surface = 6 components**: Data, Network (private vs. public-facing), Hardware (no cross-request info leakage), Application (MITM/DoS/OWASP Top 10), Model (supply-chain attacks), Permission management (Principle of Least Privilege / PoLP).
- **Tenancy models — Single vs. Multi-tenancy**: Single tenancy = each LoB independently deploys/manages its own GenAI stack and model access (central IT provides only baseline security/auth/compute). Multi-tenancy = a central GenAI platform/API provides uniform, vetted access to multiple FMs and shared components to all LoBs as tenants.
  - When to use single tenancy: need LoB autonomy, fast independent time-to-market, use-case-specific monitoring tuning.
  - When to use multi-tenancy: need centralized governance, cost tracking/FinOps, usage metering, throttling, and protection against LoBs picking risky/vulnerable models.
- **TCO (Total Cost of Ownership) framework**: cost spans more than the model itself (expanded fully in Ch14).
- **Performance & Scalability metrics framework**: RPM (requests/minute), Concurrency, TPM (tokens/minute), Latency, Utilization metrics (CPU/GPU/disk/IOPS). Scalability = Horizontal (scale-out: more, smaller machines — better cost control, less downtime) vs. Vertical (scale-up: bigger machine — may cause downtime); applies across Compute, Storage, Database, Network layers.
- **Generative AI application anatomy**: Web interface → Prompt → Model → Storage (data + metadata) → Databases (prompt cache, prompt store, vector store, application-specific DB).
- **Architectural best-practices matrix**: for each component (web interface, prompt, model, storage, database), apply each consideration (security/privacy, cost, tenancy, performance/scalability) — see Reference Tables.

## Key Concepts
- **Principle of Least Privilege (PoLP)**: actors/resources should only access what they strictly need to perform their task, nothing more.
- **Prompt cache**: stores frequent prompts + answers to avoid redundant FM calls; only useful where answers don't change often.
- **Prompt store**: central catalog of system/reusable prompts across FMs.
- **FinOps**: processes to track/optimize cloud cost per project — centralized naturally under multi-tenancy.
- **Lowest common denominator problem**: a multi-vendor/multi-model platform can only expose features (e.g., function-calling, guardrails) that all supported models share, limiting access to provider-specific advanced capabilities.
- **Shadow IT**: LoBs under single-tenancy end up re-building IT infrastructure (networking, auth, security ops) independently, duplicating central-IT work.
- **Model compression techniques** (cost/performance lever): pruning (remove parameters), quantization (reduce numeric precision), knowledge distillation (train smaller model to mimic larger one), speculative decoding (small model drafts tokens, large model reviews/corrects).

## Mental Models
- "There are no best solutions in the generative AI world, it's all about trade-offs" — apply this explicitly to the tenancy decision and to every cost-vs-performance call.
- Use the component × consideration matrix (5 components × 4 considerations) as a design checklist to run through for any new GenAI architecture, not just an afterthought.
- Treat prompts as stateless — tenancy, performance, and scalability concerns don't apply to the prompt component itself (only security/privacy and cost do).
- When choosing self-hosted vs. serverless, default to serverless for cost/scalability unless a specific isolation or compliance requirement forces dedicated hosting.

## Anti-patterns
- Mandating a uniform multi-tenant feature set that forces every LoB down to the "lowest common denominator" of model capability, discarding valuable model-specific features (function-calling, guardrails) to keep API consistency.
- Treating storage/database security as purely a data concern while ignoring network, hardware, application, model, and permission-management attack surfaces.
- Allowing unrestricted internet access from model endpoints or storage without explicit justification — creates data exfiltration risk.
- Skipping latency/throughput testing before model selection — picking the largest model by default increases latency without validating business need for the extra accuracy.
- Building single-tenancy without governance discipline, leading to shadow IT and duplicated baseline components across LoBs.

## Code Examples
No executable code in this chapter — it is an architecture/best-practices chapter presented as checklists.

## Reference Tables
**Single tenancy vs. multi-tenancy trade-offs**

| | Single tenancy | Multi-tenancy |
|---|---|---|
| Pros | Autonomy, easier maintenance/lifecycle, easier per-use-case monitoring, faster time-to-market | Model choice flexibility, tenant isolation, central cost tracking, usage metering, throttling, central governance, central model monitoring, simpler FinOps |
| Cons | Complex governance (fragmented), duplicate components across LoBs, shadow IT | Scaling complexity at large tenant counts, lowest-common-denominator feature set, skills shortage, hard abstraction-level decisions, operating-model change required |

**Architectural best practices by component (condensed)**

| Component | Security/Privacy | Cost | Tenancy | Performance/Scalability |
|---|---|---|---|---|
| Web interface | HTTPS, auth/authz, user data wipe, GDPR notices, session timeouts, no exposed backend code | Serverless deploy, idle auto-shutdown | Usually multi-tenant with per-tenant IDs in metadata | Auto-scaling compute + load balancer |
| Prompt | TLS 1.3 in transit, encryption at rest, provider EULA guarantees against training on your data, guardrails for toxicity/bias/PII | Be concise, restrict response length, cap input size, prompt cache, prompt summarization | N/A (stateless) | N/A (stateless) |
| Model | Secure endpoint, OWASP LLM Top 10 scan, no supply-chain compromise, no unneeded internet access, restrict binary access to CI/CD only | Prefer open-source where viable, self-host smaller models for high-volume/simple tasks, pruning/quantization/distillation/speculative decoding | Serverless API = inherently multi-tenant; self-hosted = your choice | Avoid largest model by default, latency-test before choosing, auto-scaling endpoints, reserved throughput for serverless |
| Storage | Encrypt at rest, private network, PoLP, no internet access | Serverless storage, inexpensive managed metadata stores | Generally single-tenant for privacy (multi-tenant needs manual row/key isolation) | Serverless scales with traffic; else track CPU/disk/IOPS |
| Database | Same as storage + user/group-level access isolation (e.g., commercial bank vs. risk/fraud team) | Serverless = pay per retrieval, optimize queries; self-hosted = size via iterative testing | Same as storage | NoSQL often more performant for unstructured data; use indexing/aggregation |

## Worked Example
**Multi-tenant GenAI platform scaling bottleneck**: An organization builds a central multi-tenant GenAI API to govern model access across hundreds of LoBs. As onboarding grows, the central platform faces millions of requests per minute aggregated across all tenants — self-hosted model endpoints struggle to autoscale fast enough, and even proprietary API providers must be pushed to increase throughput. Simultaneously, the platform can only expose features common to all onboarded models (the lowest-common-denominator problem), so LoBs needing a provider-specific feature like function-calling may be blocked from using it through the shared API — illustrating the central trade-off multi-tenancy makes against single-tenancy's autonomy.

## Key Takeaways
1. Design against a 5-component × 4-consideration matrix (web interface/prompt/model/storage/database × security-privacy/cost/tenancy/performance-scalability) for any GenAI architecture.
2. Distinguish security from privacy explicitly — they require different controls and can succeed/fail independently.
3. Choose single- vs. multi-tenancy as a conscious trade-off: autonomy and speed vs. governance and cost efficiency — there's no universally correct answer.
4. Prompts are stateless: optimize them for security/privacy and cost, not performance/scalability/tenancy.
5. Prefer open-source/smaller self-hosted models for high-volume, simpler tasks, and use pruning/quantization/distillation/speculative decoding to control model-hosting cost.
6. Apply the Principle of Least Privilege across every component, not just data.

## Connects To
- **Ch3**: Recaps the five implementation approaches before layering architectural considerations on top.
- **Ch5**: Security, privacy, and compliance challenges from Ch5 are operationalized here into concrete architectural controls.
- **Ch13**: Guardrails (toxicity/bias/PII interception) mentioned here are detailed fully in the trustworthiness chapter.
- **Ch14**: TCO and FMOps practices (cost components, model compression techniques) are expanded in full in the FMOps chapter.
