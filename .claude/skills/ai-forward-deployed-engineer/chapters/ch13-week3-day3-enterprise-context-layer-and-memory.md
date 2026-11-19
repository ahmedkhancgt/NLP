# Week 3, Day 3: The Enterprise Context Layer, Integrations, and AgentCore Memory

## Lessons Covered
- Lesson 103: The Enterprise Context Layer - Context Engineering, Atlan and Gartner
- Lesson 104: Context Layer Products, AgentCore Memory Strategies and Namespaces
- Lesson 105: Enterprise Integrations - Systems of Record, Permissions and MCP Gateways
- Lesson 106: Claude Code Adds Gateway Tools, a Context Layer and Memory to UnderCover
- Lesson 107: Deploy the Context Layer as an MCP Server on AWS Lambda with IAM Auth
- Lesson 108: Two Gateway Targets, Cedar Policy, Context Lookups and AgentCore Memory
- Lesson 109: A Returning Customer - Recalling an Earlier Claim with AgentCore Memory

## Core Idea
An enterprise context layer gives agents a holistic, consistent view of what data means, how work is done, and where it came from (permissions live in the gateway, not here); it's built from ordinary data-governance practice, not a new discipline — and long-term agent memory should be added one simple strategy at a time, namespaced so the code (never the model) enforces isolation between customers.

## Frameworks Introduced
- **The Four Parts of the Enterprise Context Layer**: (1) what the data means (data dictionary, glossary, reference data), (2) how the work is done (procedures, business rules), (3) what the agent may see and do (permissions/approvals — this lives in the Gateway's Policy, not the context layer), (4) where it came from (provenance — source, owner, review date).
  - How: UnderCover implements parts 1, 2, and 4 in its own context layer; part 3 was already built into the AgentCore Gateway Policy on Day 2.
  - Origin: traces from Philipp Schmid's 2025 "context engineering" framing ("the right information and tools, in the right format, at the right time") to Atlan's 2026 framing of the context layer as "the system that turns knowledge, expertise, and norms into machine-usable context for AI."
- **Gartner's Context Layer Lens**: three aspects — Semantics (what data means), Provenance (where it came from), Operational State (a golden record per customer/product) — framed around Master Data Management.
  - When to use: as an alternative, vendor-neutral framing to cross-check against vendor pitches like Atlan's broader "context substrate" terminology (take vendor framing with a pinch of salt — it's shaped to fit their product).
- **Four Sources of Agent Data, Each With a Different Owner**: Gateway (security team writes rules for tool calls; agent cannot change it, every call goes through it) → Context layer (business teams write enterprise data under a data governance model; agent reads it as tools through the gateway) → Memory (the agent itself reads and writes its own notes, separate per customer) → Database/system of record (the app and agent both read/write records and checkpoints through the app's API).
  - How: keep these four conceptually and architecturally separate rather than trying to force everything (e.g., memory) into one "context layer" product, even though some vendors pitch memory as belonging there too.
- **AWS's Four Long-Term Memory Extraction Strategies**: Semantic (facts extracted from conversations, vector-searchable), Preference (user-specific likes/behavior), Summary (compressed record of what's happened, like a conversation compact), Episodic (specific past events with actions and outcomes).
  - When to use: start with only one strategy (semantic is the simplest starting point); add a second only when a curated test-question set demonstrates the first isn't enough — avoid over-engineering.
- **Namespace Partitioning for Memory**: organize memories hierarchically like a folder structure (e.g., `/customers/{actorId}/facts/`, `/customers/{actorId}/claims/{sessionId}/`), where the namespace keys (actor ID, session ID) are set **programmatically by code**, never by the model — this is what prevents cross-customer memory leakage.
  - How: always pass the actor/session IDs explicitly into both the write (`store.put`) and the read/search call; never let the model choose or supply its own scoping key.
- **Enterprise Data Architecture Questions**: (1) Systems of record — does each data type have one authoritative source? (2) Read/write access — what can the agent actually change? (3) Data boundaries — where may data go, and what stays segmented? (4) Permissions — whose access/identity does a call use?
  - When to use: run through these four questions when reviewing any enterprise AI architecture, not just UnderCover's.
- **Enterprise MCP Architecture Pattern**: one MCP gateway in front of everything — company APIs (via OpenAPI), the context layer (as its own MCP server behind the gateway), external integrations (e.g., Jira), with memory typically connected directly to the agent (not routed through the gateway) and the database reached via the app's own checkpointer/API.
  - How: the agent gets one consolidated MCP tool list; it doesn't need to know which backend (API, Lambda-hosted context server, or external MCP server) serves each tool.

## Key Concepts
- **Context engineering (Philipp Schmid, 2025)**: "the right information and tools, in the right format, at the right time" — the conceptual root from which the enterprise context layer term grew.
- **Provenance**: the origin metadata (source system, owner, last review date) attached to context-layer data, so an agent/reviewer can judge how much to trust it.
- **Data quality is the hard part, not the technical build**: standing up a context layer's infrastructure is a relatively easy technical challenge; ensuring the underlying data is complete and high-quality is the expensive, time-consuming, usually-underestimated part — budget for it explicitly.
- **Context layer vendors (as of the course's reference date)**: Atlan, Collibra, Alation, DataHub (work across many platforms); Snowflake Horizon Catalog, Databricks Unity Catalog, Google Cloud Knowledge Catalog, Microsoft Fabric IQ, Palantir Ontology (platform-specific) — most cover all four context parts, generally available or in preview.
- **Memory products landscape**: AgentCore Memory (AWS, used this week), Zep, Mem0, Letta, Cognee, LangGraph store with LangMem, Google Cloud Memory Bank — vary in managed-vs-open-source, and in which of facts/preferences/summaries/episodes/knowledge-graph/LangGraph-integration they support.
- **UnderCover's homemade context layer**: rather than installing or paying for a heavyweight commercial product, the course builds its own lightweight MCP server (deployed as a Lambda function) serving seven YAML-based files (data dictionary, glossary, plans, claim rules, reference data, policy wording, claims manual) — same learning value, zero cost, and a direct illustration of how any enterprise context layer plugs into a gateway the same way.
- **Context layer tools exposed**: `lookup_term`, `describe_fields`, `get_plan`, `get_claim_rules`, `search_reference_data`, `search_documents`, `read_section` — seven tools the homemade MCP server offers, joining the app's (now six, down from eight — `search_documents`/`read_section` moved to the context layer) via the Gateway for 13 total tools.
- **Two Gateway targets**: a single Gateway can have multiple targets (the UnderCover app's six API routes, and the context layer's seven MCP tools) unified into one consolidated tool list for the agent.
- **IAM-authenticated MCP server**: the homemade context-layer MCP server requires AWS SigV4 signing to call — an unsigned call returns 403, confirming the server isn't open to the public internet.
- **AgentCore Memory event model**: the agent saves two events per claim (the claim's summary/benefit/date/place/items, and its decision) under an actor (customer) and session (claim) key; AgentCore's strategies then extract facts (semantic) and summaries (summary) from those events in the background, asynchronously.
- **Enforced namespace isolation, proven empirically**: searching customer 3005's memory with query "earlier claims" returns facts and summary; the identical search scoped to customer 3001 returns zero records — proof the isolation is code-enforced, not merely a convention the model might ignore.
- **`memory.remember` / `memory.recall`**: the harness functions (in `memory.py`/`tools.py`) that wrap AgentCore's `store.put`/`store.search` calls, always supplying the current claim's customer ID and claim ID programmatically — never leaving scoping to the model's discretion.

## Mental Models
- Treat the context layer, the gateway, memory, and the database as four separate concerns with four separate owners — resist the temptation (common in vendor pitches) to collapse them all into one "context substrate."
- When evaluating a vendor's framing of a concept like "context layer," cross-check it against a neutral analyst framing (Gartner) to see where the vendor's terminology is pitching its own product versus describing the underlying reality.
- Apply the instructor's two standing rules to memory design specifically: start simple (one strategy), and add complexity only when a curated eval set proves the simple version insufficient — exactly the same "simpler is better, add complexity when it's warranted" principle used throughout the course for architecture decisions.
- Picture namespace isolation as a locked filing cabinet drawer, not a labeling convention: the code decides which drawer to open based on the current customer/claim, so there's no path by which the model's own reasoning could open the wrong drawer.

## Anti-patterns
- **Routing all enterprise information — including memory — through one monolithic context layer product**: vendors like Atlan push this framing, but the instructor recommends keeping memory as its own distinct concern with its own purpose-built tools (e.g., AgentCore Memory), since memory is agent-authored notes, not business-governed data.
- **Building full multi-strategy memory (semantic + preference + summary + episodic) from day one with whiteboards and file-structure diagrams before testing**: over-engineering risks duplicate memories, inconsistent storage, and retrieval failures (information landed in episodic but you search semantic). Start with one strategy, test, then expand only to fix an identified, measured gap.
- **Letting the model set its own memory namespace/scope keys**: this is the one mistake that actually causes cross-customer data leakage; always force actor ID/session ID programmatically in the harness code.
- **Treating a vendor's context-layer article as neutral documentation**: read it, but recognize when terminology is built around selling a specific product (the instructor flags Atlan's broad "context substrate" vocabulary as somewhat self-serving, even while recommending the read).
- **Underestimating the data-quality investment**: assuming the technical infrastructure build is the hard part of a context layer project, when the real cost and time sink is ensuring the underlying data is complete and high-quality.
- **Skipping provenance metadata**: serving context-layer data to an agent without source system, owner, and last-review-date attached removes the agent's (and a reviewer's) ability to judge how much to trust it.

## Code Examples
```python
memory_id = agentcore.create_memory(
    name=MEMORY_NAME,
    memoryStrategies=[
        {"semanticMemoryStrategy": {"name": "customer_facts", "namespaceTemplates": ["/customers/{actorId}/facts/"]}},
        {"summaryMemoryStrategy": {"name": "claim_summaries", "namespaceTemplates": ["/customers/{actorId}/claims/{sessionId}/"]}},
    ],
)
```
- **What it demonstrates**: creating an AgentCore Memory object with exactly two strategies (semantic facts per customer, summary per claim), each with its own namespace template keyed on `actorId`/`sessionId` — the minimal, deliberately-not-over-engineered memory design the lab uses.

```python
def records_in(namespace):
    """The memory records in one namespace."""
    return memory_records.list_memory_records(memoryId=memory_id, namespace=namespace, maxResults=20)["memoryRecordSummaries"]
```
- **What it demonstrates**: reading back memory records scoped strictly to one namespace path — the same pattern the harness uses internally to prove isolation (customer 3005's namespace returns records; customer 3001's returns none).

```python
memory.recall(customer_id, claim_id)
# internally: store.search(namespace=f"/customers/{customer_id}", query=...)
```
- **What it demonstrates**: (paraphrased from the harness walkthrough) the `recall` function always receives the customer ID and claim ID programmatically from the current claim being processed — never from model-generated input — which is the actual mechanism that prevents memory leakage between customers.

## Reference Tables

| Context layer part | What it holds | Where it lives in UnderCover |
|---|---|---|
| What the data means | Data dictionary, glossary, plans, reference data | Context layer (built today) |
| How the work is done | Claims manual, claim rules | Context layer (built today) |
| What the agent may see/do | Permissions, approvals, payout limits | Gateway's Cedar Policy (built Day 2) |
| Where it came from | Source system, owner, review date | Context layer (built today) |

| Memory strategy (AWS) | What it captures |
|---|---|
| Semantic | Facts extracted from conversations, vector-searchable |
| Preference | User-specific likes/behavior |
| Summary | Compressed record of what's happened (like a compact) |
| Episodic | Specific past events: actions taken and outcomes |

| Data source | Who writes it | How the agent accesses it |
|---|---|---|
| Gateway | Security team | Every tool call passes through it; agent cannot change it |
| Context layer | Business teams (data governance model) | Reads it as tools, through the gateway |
| Memory | The agent itself | Reads and writes with its own memory tools |
| Database | The app and the agent | Reads and writes through the app's API / checkpointer |

## Worked Example
The Day 3 product build has Claude Code make five changes per `docs/CONTEXT.md`: (1) the agent becomes a true MCP client of the Gateway for its existing tools (closing a gap left from Day 2, where the agent called the backend API directly rather than through MCP for everything); (2) a brand-new lightweight MCP server — the homemade "enterprise context layer" — is written and packaged as its own Lambda function; (3) the decision record is extended to capture everything the claims manual requires be documented; (4) AgentCore Memory is wired in, keyed per customer; (5) every tool call, lookup, memory access, and refusal is surfaced on the claim page for in-platform observability. The build runs roughly 3.5 hours with the harness's full test suite running after every commit.

In the lab, the new app is redeployed to Lambda with two fewer routes (eight down to six — `search_documents` and `read_section` moved out to the context layer), and the new context-layer MCP server is zipped (10.3MB) and deployed as its own Lambda function requiring SigV4-signed calls (an unsigned call returns 403). Listing its tools returns seven: `lookup_term`, `describe_fields`, `get_plan`, `get_claim_rules`, `search_reference_data`, `search_documents`, `read_section`. The Gateway is recreated with two targets (the app's six routes, the context layer's seven tools) for 13 tools total, and this time the Cedar policy engine is set straight to `ENFORCE` mode from the start (skipping log-only, since the policy pattern was already proven Day 2) — with three active policies: permit app tools, permit context tools, forbid `pay_claim` over $1,000. Spot tests confirm `lookup_term("unattended")` returns the glossary definition with its provenance metadata, `read_section` pulls claims-manual section 14, a reference-data search for "Lisbon Strike" returns a matching event, an over-limit `pay_claim` is refused by the Gateway, and an invalid referral-with-payout call is rejected by the backend's own business-rule check.

An AgentCore Memory object is created with exactly two strategies (semantic facts and summary, each namespaced by customer/claim as shown above). The database is reset to a blank slate, and claim CLM-1005 (Oliver Grant's passport eaten by a giraffe) is processed entirely with cloud-hosted tools, context layer, and memory, while the agent code itself still runs locally on the student's laptop. Result: approved for $243, with the dashboard's observability trace showing the full tool-call history including a final "saved to memory" step. Inspecting the stored memory shows six extracted facts plus an XML-structured summary (topic: "insurance claim CLM-1005, passport destroyed by Giraffe"). A second claim, CLM-1006, is then run for the same customer; this time the agent's observability trace shows it voluntarily calling `search_memory` with a query like "Oliver Grant Kenya Giraffe prior claim," retrieving both the earlier facts and summary, and it approves $425 — demonstrating cross-claim recall within one customer's namespace. A deliberate leakage test — searching customer 3005's memory versus customer 3001's with the same query — returns five records for 3005 and zero for 3001, confirmed by inspecting `harness.py`'s `memory.remember`/`memory.recall` functions, which hard-code the actor/session scoping in code rather than leaving it to the model. The lab ends by tearing down the app, context layer, gateway, and policy engine (keeping only the Lambda function itself, which costs nothing idle), with an assignment to add an Episodic memory strategy and compare what it captures against the Semantic/Summary strategies already in place.

## Key Takeaways
1. Keep the context layer, the gateway, memory, and the database conceptually and architecturally separate — each has a different owner and a different access pattern, even though vendors sometimes pitch one product that does everything.
2. Build (or buy) a context layer around the four genuine parts — meaning, procedure, permissions (really the gateway's job), and provenance — not around a vendor's broader marketing framing.
3. The hard part of a context layer project is data quality, not the technical plumbing; budget time and money for it explicitly.
4. Start agent long-term memory with a single strategy (semantic is the simplest) and add others only when a curated test set proves the gap — avoid pre-emptive multi-strategy over-engineering.
5. Namespace memory hierarchically (e.g., per customer, per session) and set the namespace keys programmatically in your harness code, never via model-generated input — this is the actual mechanism that prevents memory leakage, not a naming convention.
6. A Gateway can unify multiple targets (an app's API routes, a homemade or vendor MCP server acting as a context layer, external integrations) into one consolidated tool list the agent never has to reason about separately.
7. When a policy pattern is already proven (as the $1,000 Cedar rule was from Day 2), it's reasonable to skip log-only mode and go straight to enforce for a repeat deployment.
8. Verify memory isolation empirically — query the same question under two different customer namespaces and confirm zero cross-contamination — rather than trusting the design on paper alone.
9. Cross-check vendor framing (Atlan) against neutral analyst framing (Gartner) when forming your own opinion on an emerging architecture concept, and prefer the simpler, narrower interpretation until complexity is proven necessary.

## Connects To
- **Week 3, Day 2**: Directly extends the Gateway built there — adds a second target (the context layer), reuses the same Cedar policy pattern, and finally makes the agent a genuine MCP client for all its tools instead of calling the backend API directly.
- **Week 3, Day 4**: The remaining gap in the architecture — the agent itself still runs locally on the student's laptop rather than in the cloud — is closed next, when the agent moves onto AgentCore Runtime.
- **AgentCore Memory / MCP specification**: This day is a direct, hands-on application of AWS's published long-term memory strategy guidance and of MCP as the uniform protocol for exposing both business APIs and a bespoke context layer as tools.
