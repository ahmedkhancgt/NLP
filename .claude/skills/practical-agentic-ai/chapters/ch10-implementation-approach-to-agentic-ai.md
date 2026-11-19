# Chapter 10: Implementation Approach to Agentic AI

## Core Idea
TnTMart's three agents (order tracking, proactive reminders, refund status) are all built from the same four-part implementation recipe — a normalized PostgreSQL schema the agent reasons over, an MCP server exposing generic `execute_query` access instead of hand-coded per-task queries, Microsoft Agent Framework tools (`@ai_function`) for actions that need custom logic beyond SQL, and carefully layered system-prompt instructions encoding business rules as explicit conditional logic — and the chapter closes by showing that scaling, memory management, and AgentOps observability all require per-agent-type tuning rather than one-size-fits-all treatment, because each TnTMart agent has a fundamentally different risk/autonomy profile.

## Frameworks Introduced

### Schema-First Agentic Design
A well-normalized data model (customer, address, product, warehouse/warehouse_product junction, orders/order_item, payment, refund, shopping_cart, customer_regular_items) is the prerequisite for accurate agent reasoning and SQL generation — the more descriptive the schema, the more accurately an agent can work out dependencies (e.g., adding a product to a cart requires both a product ID and a customer ID) without those relationships being hand-coded into agent logic.
- When to use: before writing any agent tool or prompt, invest in schema clarity and documentation — this is the Ch 5 data-governance principle applied at the most concrete level: an agent's reasoning quality is bounded by how well-structured and well-described its underlying data is.

### MCP Server as Generic Database Access (vs. Per-Task Queries)
Rather than hand-writing a distinct query/tool for every operation (add to cart, approve refund, check order status), a single **Postgres MCP server** exposes three methods — `connect_db` (startup), `execute_query` (runs arbitrary SQL, returns JSON for SELECT or a row-count success message for INSERT/UPDATE/DELETE, rolls back and returns a JSON error object on failure), `close_db_connection` (shutdown) — and agents autonomously compose the SQL they need against the documented schema.
- **Logging for debuggability**: FastMCP's `Context` object (`ctx.info()`, `ctx.error()`) sends server-side log messages directly to the client, solving the classic client-server debugging blind spot where server logs are invisible to the client application.
- When to use: whenever multiple agents need flexible, varied access to the same backend data store — a generic query-execution tool scales better than a growing library of single-purpose query tools, provided the schema is well-documented enough for the agent to compose correct SQL (the chapter's explicit tradeoff for schema-first design).

### Microsoft Agent Framework Tools for Non-SQL Logic
Custom `@ai_function`-decorated tools handle logic that isn't a database query: `getETA` (static lookup by location type — "Local" → 5 days, else → 10 days), `add_to_cart`/`update_quantity_in_cart` (parameterized SQL execution with business-logic wrapping), `approve_refund`/`reject_refund` (status-changing SQL with a reason parameter). The agent is responsible for collecting the necessary parameters from the customer conversation before invoking the tool — parameter-gathering is not implemented explicitly as separate code.
- When to use: anything beyond "run this SQL" — calculations, external API calls (Google Distance Matrix for live ETA), or business actions that need validation/side effects beyond a raw query.

### Layered System Prompt Design (the "Instructions" Pattern)
Across all three TnTMart agents, instructions are built from the same recurring sections: **general hygiene** (role, scope boundaries, SQL correctness requirements, graceful error handling, empty-response handling) → **schema understanding** (explicit table/column documentation, which plugin to use for which operation) → **customer interaction basics** (what to ask for, in what order, how to present results) → **core business logic** (explicit conditional rules — e.g., refund approval requires ALL of: reason matches refund-table reason, delivery was late relative to expected date, amount matches order amount) → **rejection logic** (explicit disjunctive conditions — ANY one being true triggers rejection) → **personalization/closing** (use the customer's name, end positively).
- When to use: this is the chapter's reusable prompt-engineering template for any agent handling a conditional, policy-governed business process — the key insight is that *business rules belong in the system prompt as explicit, enumerable conditions*, not left to the model's implicit judgment, especially where approval/rejection has financial consequences.

### Scaling Methodology — Four Independent Axes
- **Infrastructure scaling**: containerize agents as microservices, horizontally scale replicas by demand signal (active sessions for refund/order-tracking agents; scheduled triggers for proactive reminders); Kubernetes dynamically scales up for flash-sale spikes and down during idle hours.
- **Database scaling**: vertical (more CPU/memory) or horizontal (read replicas isolating reporting/analytics from OLTP traffic; date/region-based partitioning on `orders`/`order_item`; connection pooling for concurrent agent access).
- **Agent orchestration**: event-driven architectures/message buses (RabbitMQ, Kafka) let agents subscribe to events rather than hold direct dependencies, enabling independent scaling and preventing **agent sprawl** (agents multiplying unchecked into an ungovernable fragmented system).
- **Connection scaling**: the MCP layer itself must scale independently — multiple MCP instances behind a load balancer, connection caching, token-based auth, FastMCP patterns for concurrent tool calls.
- When to use: treat these as four separate scaling problems, not one — a system can be infrastructure-scaled but still bottleneck on database connections, or orchestration-scaled but still agent-sprawl ungoverned; audit all four independently.

### Multi-Tiered Memory Hierarchy
**Ephemeral (session) memory**: short-lived conversation context, discarded when the session expires — stored in-memory/Redis to avoid repeated LLM calls for recall. **Persistent memory**: long-term structured data (order history, preferences, past refund interactions) in PostgreSQL or vector databases. **Semantic memory**: embedded vector representations enabling similarity-based recall across past exchanges (a customer asking "where is my replacement item?" can retrieve context from similar prior conversations). **Procedural memory**: encoded workflows/action sequences (refund steps, recommendation flows) stored as structured patterns (MCP registry entries or YAML configs).
- **Optimization strategies**: **context compression** (a summarization agent periodically compresses old dialogue into semantic summaries, preventing context-window overflow on long-running interactions like multi-day refund escalations); **relevance scoring** (recency + access frequency + engagement value determine what gets pruned in nightly batch jobs); **tiered memory access** (namespaced partitions so each agent only queries its relevant subset — refund agent needs payment/refund data, order-tracking agent needs order/delivery data); **memory sharing via a Shared Context Registry (SCR)** (agents publish/subscribe to context summaries via MCP rather than redundantly re-fetching the same data, avoiding duplication/inconsistency across collaborating agents).
- When to use: this hierarchy is the concrete implementation of the memory-ontology concepts introduced in Ch 2 — map each memory *type* from that chapter's theory onto a specific storage technology and optimization strategy here before building any production agent that needs to "remember."

### AgentOps — Per-Agent-Type Metric Profiles
AgentOps extends DevOps/MLOps to agents' non-deterministic, self-governing behavior across a five-stage lifecycle: **design & implementation** (goals, constraints, HITL override points) → **evaluation** (sandboxed testing — success rate, latency, tool-selection accuracy, token cost) → **deployment** (runtime workflow/resource/scheduling management) → **observability** (continuous session/trace/span-level tracking of every LLM call, tool use, and decision — "what it decided and why") → **feedback loop** (user + developer signals feed back into design, closing the cycle).
- The chapter's key insight: **metrics must match the agent's risk/autonomy profile**, not be applied uniformly:
  - **Order tracking agent** (high-volume, low-risk, read-only): focus on efficiency — Average Task Duration, LLM Cost Per Task (LCPT).
  - **Proactive reminder agent** (moderate autonomy, behavior-triggered): Task Success Rate, Context Utilization Score, LCPT, Redundant Steps, Tool/Action Selection Accuracy, Decision Turn Count, Recovery Rate.
  - **Refund agent** (high agency — initiates financial transactions — but deliberately low operational autonomy due to compliance/irreversibility risk): governance-focused — Tool/Action Selection Accuracy, HITL Escalation Rate and Success, Threshold Breach count, Policy Mismatch count.
- **Robustness-cost tradeoff**: a high Recovery Rate (successful self-correction after failures) requires more LLM re-planning/retry calls, which directly increases token cost — so robustness must be strategically prioritized (worth the cost for the high-stakes refund agent) or deprioritized (cost matters more for the high-volume order-tracking agent), not maximized uniformly everywhere.
- When to use: before instrumenting any agent for production observability, classify it by agency (can it act on/modify the environment?) and required autonomy (should it act without approval?) — the refund agent's "high agency, low autonomy" profile is the chapter's central lesson: granting an agent the ability to do something consequential doesn't mean granting it permission to do it unsupervised.

## Key Concepts
- **Agent sprawl**: uncontrolled proliferation of agents leading to a fragmented, hard-to-govern system — the specific failure mode that event-driven orchestration (pub/sub decoupling) is designed to prevent as agent count grows.
- **LLM Cost Per Task (LCPT)**: total token consumption × model cost for a complete interaction chain — the chapter's standard unit for comparing agent efficiency across use cases.
- **High agency, low autonomy** (the refund agent's defining property): the ability to act on/modify the environment (agency) is architecturally separate from the permission to do so without oversight (autonomy) — a system can deliberately grant the former while withholding the latter for compliance-sensitive actions.
- **Threshold breach / policy mismatch**: specific, countable governance-violation signals (proposed refund exceeds a dollar limit; proposed action falls outside a business rule like the 90-day window) that force HITL intervention — concrete, measurable instances of the "escalation as a first-class branch" principle from Ch 9.
- **Context compression via periodic summarization**: a dedicated summarization agent (not the primary conversational agent) handles the compression step — separating "have the conversation" from "manage the conversation's memory footprint" as distinct responsibilities.

## Mental Models
- Treat "which tool does this agent need" as two separate questions: does this operation vary enough across use cases that a generic query-execution tool (MCP) suffices, or does it need validated, parameterized, custom logic (an `@ai_function`)? The chapter answers both — dynamic lookups go through MCP, calculations/actions with business logic go through dedicated tools.
- When writing an agent's system prompt for a policy-governed process, enumerate approval conditions as an explicit AND (all must hold) and rejection conditions as an explicit OR (any one triggers it) — the refund-agent prompt's structure is the template: ambiguity in approval logic is a production risk when the action is financial and hard to reverse.
- Before setting up observability for an agent, ask "what's the worst thing this agent could autonomously do, and how reversible is it?" — the answer determines whether your metrics should emphasize efficiency (order tracking) or governance (refunds), not a generic dashboard template.
- Scaling an agentic system is scaling four different things (infrastructure, database, orchestration, MCP connections) that happen to run together — don't assume horizontally scaling agent replicas also scales the database connections or the MCP layer behind them.

## Anti-patterns
- **Hand-writing a bespoke query tool for every single database operation**: creates tool-proliferation overhead the MCP generic-execute-query pattern specifically avoids — reserve dedicated tools for logic genuinely beyond "run this SQL."
- **Leaving business-rule conditions implicit in a prompt ("approve reasonable refund requests")**: the chapter's refund-agent prompt makes every approval/rejection condition an explicit, enumerable rule — implicit judgment calls on financial actions are a governance and audit liability.
- **Granting an agent high autonomy just because it has high agency**: the refund agent's design explicitly separates "can initiate a financial transaction" (agency) from "may do so without approval" (autonomy, deliberately kept low) — conflating the two for any high-stakes, hard-to-reverse action is a direct path to ungoverned autonomous financial decisions.
- **Applying uniform AgentOps metrics across all agents regardless of risk profile**: measuring the refund agent primarily on speed/cost (like the order-tracking agent) would miss the governance signals (threshold breaches, policy mismatches, HITL escalation rate) that actually matter for a high-agency, low-autonomy agent.
- **Maximizing Recovery Rate everywhere without considering cost**: higher robustness via more retry/re-planning cycles directly drives up LLM cost — treat the robustness-vs-cost tradeoff as a deliberate per-agent decision, not a universal "more robust is always better" default.
- **Letting ephemeral session memory persist indefinitely, or persistent memory grow unpruned**: the chapter explicitly warns memory growth "can be exponential" without context compression and relevance-scored pruning — unmanaged memory causes context drift, retrieval lag, and storage bloat.

## Code Examples

**MCP server — generic query execution with error handling**:
```python
try:
    with db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query, params)
        if cursor.description:
            res = cursor.fetchall()
            return json.dumps(res, default=str)
        else:
            db_conn.commit()
            return json.dumps({"status": "success", "rows_affected": cursor.rowcount})
except psycopg2.Error as e:
    if db_conn:
        db_conn.rollback()
    return json.dumps({"error": str(e)})
```

**Microsoft Agent Framework tool — cart management**:
```python
@ai_function(description="Adds an item to the shopping cart.", name="add_to_cart")
def add_to_cart(customer_id: int, product_id: int, quantity: int, unit_price: float) -> str:
    query = f"""insert into shopping_cart (customer_id, product_id, quantity, unit_price)
                values ({customer_id}, {product_id}, {quantity}, {unit_price});"""
    cursor = TnTMartTools.connection.cursor()
    cursor.execute(query)
    TnTMartTools.connection.commit()
    return f"Product {product_id} added to cart for customer {customer_id}."
```

**Refund approval/rejection tools**:
```python
@ai_function(description="Approves a refund request.", name="approve_refund")
def approve_refund(refund_id: int) -> str:
    query = f"""update refund set status='Approved' where refund_id={refund_id};"""
    cursor = TnTMartTools.connection.cursor()
    cursor.execute(query)
    TnTMartTools.connection.commit()
    return f"Refund request with ID {refund_id} has been approved."
```

**Agent creation wiring MCP + custom tools + Chainlit UI**:
```python
pgsql_tool = MCPStdioTool(name="PGSQLMCPServer", command="python", args=[str(mcp_server_path)])
await pgsql_tool.connect()

agent = client.create_agent(
    name="TnT Mart Refund Bot",
    instructions="""{SYSTEM_PROMPTS}""",
    tools=[pgsql_tool, TnTMartTools.approve_refund, TnTMartTools.reject_refund],
)
thread = agent.get_new_thread()

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    thread = cl.user_session.get("thread")
    response = await agent.run(message.content, thread=thread)
    await cl.Message(content=response).send()
```

## Reference Tables

**AgentOps metrics by agent type** (Table 10.1):
| Agent | Key Metrics |
|---|---|
| Order Tracking Agent | Average Task Duration, LLM Cost Per Task (LCPT) |
| Proactive Reminder Agent | Task Success Rate (TSR), Context Utilization Score, LCPT, Redundant Steps (RS), Tool/Action Selection Accuracy, Decision Turn Count, Recovery Rate |
| Refund Agent | Tool/Action Selection Accuracy, HITL Escalation Rate and Success, Threshold Breach, Policy Mismatch |

**Refund approval logic** (delay reason — ALL must hold to approve): refund reason matches refund-table reason · order status is "delivered" AND actual delivery date is after expected delivery date · refund amount matches order amount.
**Refund rejection logic** (ANY true triggers rejection): reason mismatch · delivered on/before expected date · order status is "cancelled" · "damaged product" reason with no photo provided · refund amount mismatch.

**Memory hierarchy at a glance**: Ephemeral/session (Redis/in-memory, discarded on session end) · Persistent (PostgreSQL/vector DB, long-term structured data) · Semantic (vector embeddings, similarity-based recall) · Procedural (MCP registry/YAML, encoded workflows).

## Worked Example

**Refund agent's full conditional reasoning**: a customer reports a delay and requests a refund. The agent's system prompt requires it to first gather customer ID, order ID, amount, and reason (asking repeatedly until all are provided), then query the `refund` table via the `refund_status_plugin` to check current status. If status is "Pending" and the stated reason is "delay," the agent checks the `orders` table: it may approve the refund only if the customer-stated reason matches the refund-table reason, the order is marked "delivered" with `actual_delivery_date` after `expected_delivery_date`, and the claimed amount matches the order amount — all three conditions, not just one, must hold. If instead the reason is "damaged product," the agent must first request a photo of the damage (handled by inspecting message elements in the Chainlit UI layer) before any approval condition can be evaluated. Any single violated condition from the explicit rejection list (reason mismatch, on-time delivery, cancelled order, missing damage photo, amount mismatch) causes rejection via the `tnt_mart_manager` plugin's `reject_refund` tool. This demonstrates the chapter's layered-prompt pattern operating end-to-end: schema-grounded data lookup, explicit AND/OR business-rule evaluation, and a tool call that actually changes system state — all orchestrated by the agent's own reasoning rather than a hardcoded script.

## Key Takeaways
1. A well-documented, normalized schema is the prerequisite for accurate agentic SQL generation — schema clarity directly determines reasoning quality, not just reasoning convenience.
2. MCP's generic `execute_query` pattern scales better than per-task query tools for varied database access; reserve dedicated `@ai_function` tools for calculations, external APIs, and actions needing custom validation logic.
3. Business rules for consequential actions (refund approval) belong in the system prompt as explicit, enumerable AND/OR conditions — not left to implicit model judgment.
4. Scaling an agentic system means scaling four distinct things independently: infrastructure (containerized replicas), database (read replicas/partitioning/pooling), agent orchestration (event-driven decoupling to prevent agent sprawl), and the MCP connection layer itself.
5. A four-type memory hierarchy (ephemeral/persistent/semantic/procedural), paired with context compression, relevance scoring, tiered access, and shared-context registries, is the concrete architecture needed to prevent memory-driven context drift and storage bloat at scale.
6. AgentOps metrics must be chosen per agent's agency/autonomy profile — the refund agent's "high agency, deliberately low autonomy" design is the chapter's central lesson: the ability to act consequentially and the permission to act unsupervised are separate design decisions.
7. Robustness (self-correction via retries) and cost are in direct tension — prioritize robustness where failure consequences are severe (refunds), prioritize efficiency where volume and low stakes dominate (order tracking).

## Connects To
- **Ch 9 (Practical Example: Case Study)**: the design-thinking decomposition (data location, state machine, tool list, escalation branch) for all three TnTMart use cases is directly implemented here with real schema, MCP server, and Microsoft Agent Framework code.
- **Ch 6 (Frameworks Part 1)**: the `@ai_function` decorator and Microsoft Agent Framework agent-creation pattern introduced there are applied throughout this chapter's TnTMart tool implementations.
- **Ch 2 (Essential Concepts)**: the memory taxonomy (STM/LTM, contextual/working, episodic/semantic/procedural) introduced there is given a concrete production architecture (ephemeral/persistent/semantic/procedural tiers + optimization strategies) here.
- **Ch 11 (Productionizing Agentic AI)**: picks up exactly where this chapter's scaling/memory/AgentOps sections leave off, with a full treatment of validation, tracing, and performance optimization for production agentic components.
