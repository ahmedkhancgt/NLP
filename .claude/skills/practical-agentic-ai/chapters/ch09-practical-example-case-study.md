# Chapter 9: Practical Example: Case Study

## Core Idea
Designing an agentic application starts from decomposing a concrete business use case into design-thinking questions ("where is the data, how does the agent access it, what tools does it need, how does it handle the failure/delay case") before any framework code — demonstrated through TnTMart, a nationwide e-commerce company whose operational pain points (support overload, repetitive queries, generic recommendations, unreliable ETAs, slow refunds) are mapped one-to-one onto specific agentic solutions (semantic search agent, context-aware recommendation agent, inventory prediction agent, proactive-reminder agent, predictive-ETA agent).

## Frameworks Introduced

### The Design-Thinking Decomposition Method
For each use case, the chapter asks a fixed sequence of questions before design: *Where does the needed data live? How does the agent access it (which tool/plugin)? What are the possible states/stages the entity can be in? How does the agent compute a derived value (ETA, delay) from that state? How does the agent handle the exception case (delay, stock-out, ambiguous request)?*
- When to use: this is a transferable template for scoping any new agentic feature — resist jumping to "which framework" or "which LLM" before answering these structural questions about data location, access, state machine, and exception handling.

### Operational Pain Point → Agentic Solution Mapping (Table 9.2 pattern)
Each pain point is decomposed into **technical cause** → **agentic solution**, not just "add an LLM":
- Keyword search misses synonyms (cilantro ≠ coriander) → **conversational NLP / agentic RAG**: vector-database knowledge base + semantic search agent, enabling natural-language queries ("organic snacks under $200").
- Generic rule-based recommendations → **context-loaded reasoning agent**: full purchase history + dietary tags + seasonality + trending products as context, so the agent reasons about relevance rather than following static rules (gluten-free customer + festive season → proactively suggests festive gluten-free sweets).
- Batch inventory sync causing post-checkout stock-out surprises → **inventory prediction agent**: predicts stock-outs before checkout completes and offers an immediate alternative (organic almond milk predicted to stock out → suggests soy milk with next-day delivery).
- Long support queues from manual escalation → **support agent with a product + case-history knowledge base**: resolves ~90% of repetitive queries instantly, escalates only genuinely ambiguous/complex cases (full context + chat history attached) to a human.
- Carts expiring due to short session persistence at scale → **AI purchase-reminder agent**: detects abandoned carts, recognizes habitual-purchase timing, and nudges with context-aware offers ("left your daily essentials — shall I place a 6 PM order?").
- Unreliable live tracking from batch ETA updates → **predictive ETA agent**: continuous GPS/traffic/route-data fusion, recalculates and explains delays conversationally rather than silently updating a number.
- When to use: this is the chapter's core methodology — never propose "an AI chatbot" as the solution; trace the specific technical cause of each pain point to the specific agentic capability (semantic search vs. context reasoning vs. prediction vs. escalation routing) that actually fixes it.

### Worked Design: "Where Is My Order" (Order Status + ETA Agent)
Decomposition questions: where is order-status data (Postgres `orders` table) → how does the bot access it (a `@ai_function`-decorated tool interfacing with Postgres) → what stage is the order at (an explicit 9-stage state machine: Received → Confirmed → Being processed → Ready to dispatch → Dispatched → Out for delivery → Delivered → Cancelled → Returned) → how is ETA computed (pre-dispatch: static average inter-stage transition times; post-dispatch: Google Distance Matrix API against current delivery-vehicle location) → how are delays handled (pre-emptive customer notification + offer to cancel/reschedule + pre-approved store credit, or escalate to a human for judgment calls).
- **Tool decomposition**: get order status from DB → calculate pre-dispatch ETA (and detect delay) → determine alternate-warehouse delivery options → determine current delivery-partner location → calculate post-dispatch ETA. The chapter is explicit that *which tool to call in what sequence is not hardcoded* — that orchestration judgment is "the autonomous part of agentic AI."
- When to use: any status-lookup-plus-prediction use case (order tracking, shipment tracking, service-ticket status) — the pattern of "explicit state machine + tool-per-data-source + autonomous sequencing + exception path" generalizes directly.

### Worked Design: Proactive Order Reminders (Habitual-Purchase Nudging)
Decomposition questions: where is the regular-customer list (a `customer_category` column: New/Regular/Returning/Churned) → how to find a customer's order history and regularly-ordered items (SQL over Customers/Orders/Order-details tables) → how to run thousands of concurrent nudge conversations → how to pause and resume per-customer since responses arrive asynchronously.
- **Batch cadence trade-off**: daily recalculation is wasteful (habitual purchase patterns shift slowly) vs. monthly recalculation (efficient, but less fresh) → the chapter's resolution is a **hybrid**: monthly baseline computation (identify habitual buyers + persist individualized nudge dates) plus lightweight daily adjustments for sudden behavior shifts — explicitly chosen over either pure-daily or pure-monthly for its cost/freshness/simplicity balance.
- **Two-flow architecture**: Flow A (periodic batch — identifies regular items + nudge dates, writes to customer table) and Flow B (notification batch — sends conversation invites on scheduled dates; accepting the invite spins up a fresh agent instance pre-loaded with the customer's regular-item context).
- When to use: any proactive, large-scale, per-customer outreach scenario — the lesson is that *not every personalization decision needs real-time computation*; matching recompute frequency to how fast the underlying pattern actually changes is itself a design decision.

### Worked Design: Refund Status Agent
Decomposition questions: where are order details (Orders + Order-details tables, with a `status`, `delay`, `delay_duration` column) → what delay threshold triggers automatic refund eligibility (>2 hours) → where do customer-provided quality-issue images get stored (standard cloud storage, linked to the order) → what is the standard refund ETA (2 working days) and pre-approved store-credit amount (10% of order value).
- **Two-reason branching**: for a **delay** reason, the agent verifies the delay was actually reported on that order; for a **quality** reason, it asks the customer for photographic proof; for any **other** reason, it forwards the conversation to a human agent rather than guessing.
- When to use: any claims/dispute-resolution agent — the pattern of (verify against system-of-record data for objectively checkable reasons; request evidence for subjective/quality reasons; escalate everything else) is a general template for bounded autonomous decision-making in a trust-sensitive process.

## Key Concepts
- **State machine as the backbone of a status-tracking agent**: defining the order's explicit stage set (9 named stages) before writing any agent logic is what makes ETA calculation, delay detection, and customer communication all tractable — without it, "where is my order" has no structured answer to compute from.
- **Tool decomposition vs. orchestration logic**: the chapter explicitly separates "what tools exist" (DB lookup, ETA calculator, alternate-warehouse checker, location fetcher) from "in what order/when to call them" — the latter is what makes the system agentic rather than a fixed script, and is deliberately left to the agent's own reasoning.
- **Recompute cadence as a cost/freshness trade-off**: the proactive-reminder use case's daily-vs-monthly-vs-hybrid analysis is a concrete instance of a general agentic-systems principle — not every signal needs to be recalculated at the same frequency it could be.
- **Escalation as a first-class branch, not a fallback**: in both the support-query and refund-status designs, "forward to a human" is an explicit, designed-for branch (triggered by delivery-marked-but-not-received ambiguity, or an unlisted refund reason) — not an afterthought for when the agent "doesn't know what to do."
- **Context-loaded agent instantiation**: in the proactive-reminder flow, a fresh agent instance is created only once a customer accepts the conversation invite, and it's instantiated already knowing the customer's regular-item context — illustrating that agent creation itself can be deferred and parameterized rather than always running continuously.

## Mental Models
- Before writing any agent code, ask the chapter's design-thinking questions in order: where's the data, how does the agent reach it, what are the possible states, how is a derived value computed, how is the exception/delay/ambiguous case handled — skipping straight to "which framework" produces a system without a clear data/state foundation.
- Treat "add an LLM chatbot" as a non-answer to an operational pain point — the chapter's pain-point table insists on naming the specific technical cause (keyword-only search, batch inventory sync, manual escalation) and the specific agentic mechanism that addresses that cause (semantic search, stock-out prediction, bounded auto-resolution with escalation).
- When a use case involves scale (nudging thousands of customers, syncing 200,000 SKUs), ask explicitly whether the computation needs to happen in real time or whether a batched/hybrid cadence achieves the same outcome more cheaply — real-time is not free, and the proactive-reminder case shows deliberately choosing against it.
- Design the escalation branch with the same care as the happy path — every one of this chapter's three worked examples (order status, reminders, refunds) has an explicit "when the agent should not decide alone" branch, and that branch is where trust and safety actually live.

## Anti-patterns
- **Building a single monolithic "chatbot" without decomposing the underlying data/state model first**: the "where is my order" design only becomes implementable once the 9-stage order state machine and per-stage data sources are made explicit — a chatbot without this structure cannot reliably answer the question it exists to answer.
- **Letting agent orchestration logic be hardcoded as a fixed script**: the chapter explicitly calls out that tool-calling sequence must be left to the agent's autonomous reasoning, not pre-scripted — a pre-scripted sequence isn't agentic, it's a conventional workflow wearing an LLM's clothing.
- **Recomputing personalization signals more frequently than the underlying pattern changes**: daily recalculation of habitual-purchase nudge dates "offers minimal benefit but incurs significant computational cost" when buying patterns shift slowly — matching compute cadence to actual signal volatility avoids this waste.
- **Treating every customer request as answerable by the agent**: both the support and refund use cases model explicit "forward to human" branches for genuinely ambiguous or out-of-policy cases — an agent without this branch either guesses (risking a wrong, possibly costly decision) or gets stuck.
- **Ignoring the cost of connecting disparate backend systems**: TnTMart's own list of challenges (ERP/CRM/logistics/catalog synchronization across 200,000 SKUs, a 5,000-entry knowledge base needing daily refresh) is a reminder that an agentic redesign inherits all the integration complexity of the systems it wraps — it doesn't make that complexity disappear.

## Code Examples
This chapter is design-and-decomposition-focused rather than implementation-focused (full code arrives in Ch 10); its only reusable technical references are: Microsoft Agent Framework's `@ai_function` decorator (introduced in Ch 6, reused here to wrap the order-status-lookup tool) and the Google Distance Matrix API (named as the mechanism for post-dispatch ETA calculation from a delivery vehicle's current location).

## Reference Tables

**Order lifecycle stages** (used as the state machine for the "Where is my order" agent): Received → Confirmed → Being processed → Ready to dispatch → Dispatched → Out for delivery → Delivered / Cancelled / Returned.

**TnTMart's top operational/technical challenges** (each later mapped to an agentic solution): high support-query volume (15,000/day, >10 min peak response) · repetitive queries (60%+ of tickets, 40% of agent time) · low recommendation relevance (<3% CTR) · limited support hours (10hrs/day, 25% of queries unanswered until next day) · search accuracy drop (~20%) on typos/natural language · 200,000-SKU multi-system integration latency · 5,000+ FAQ/policy knowledge-base freshness · only 2 of 10 major regional languages supported · 70% of delivery complaints reactive rather than proactively flagged · data privacy/consent for 2M+ registered users.

**Refund-agent business rules**: delay threshold for automatic refund eligibility = >2 hours · standard refund processing ETA = 2 working days · pre-approved store credit = 10% of order value.

## Worked Example

**"Where is my order" — full agent reasoning path**: a customer asks for their order status. The agent's tool-driven reasoning (sequence not hardcoded) works out to: (1) query the Postgres `orders` table for the order's current stage; (2) if pre-dispatch, sum static average inter-stage transition times to produce an ETA and check whether elapsed time already exceeds that estimate (a delay); (3) if post-dispatch, call the Google Distance Matrix API against the delivery partner's current location and the customer's address for a live ETA; (4) if a delay is detected at any point, proactively message the customer with the revised ETA and offer to cancel, reschedule, or accept pre-approved store credit; (5) if the situation is ambiguous or high-stakes, hand off to a human agent for judgment. This single use case threads together the state-machine, tool-decomposition, autonomous-sequencing, and escalation-branch concepts all at once — making it the chapter's canonical example of agentic design thinking in practice.

## Key Takeaways
1. Agentic application design starts with explicit design-thinking questions (data location, access method, state model, derived-value computation, exception handling) before any framework or model choice.
2. Every operational pain point should be traced to its specific technical cause and matched to a specific agentic mechanism — semantic/RAG search for keyword-matching failures, context-loaded reasoning for generic recommendations, prediction for batch-sync-driven surprises, bounded autonomy + escalation for support overload.
3. Tool decomposition (what capabilities exist) is distinct from orchestration (what order to invoke them in) — the latter is deliberately left to the agent's autonomous judgment, which is what makes the system agentic rather than a scripted workflow.
4. Not every signal needs real-time recomputation — matching batch/recompute cadence to how fast the underlying pattern actually changes (the hybrid monthly-baseline-plus-daily-adjustment reminder design) is itself a cost-conscious design decision.
5. Explicit, well-designed escalation-to-human branches (ambiguous refund reasons, "delivered but not received" disputes) are a core part of agentic design, not a fallback bolted on afterward — they are where bounded autonomy meets accountability.
6. A real-world agentic redesign inherits the full integration complexity (disparate systems, large SKU counts, multi-language support, data privacy) of the business it's built for — agentic AI addresses many but not all of these challenges, and does not eliminate the underlying systems-integration burden.

## Connects To
- **Ch 6 (Frameworks Part 1)**: the `@ai_function` decorator and Microsoft Agent Framework patterns introduced there are the concrete implementation substrate this chapter's TnTMart designs are built on.
- **Ch 8 (Value Add with Agentic AI)**: the retail inventory/supply-chain case study there shares the same agent-role-decomposition discipline applied here at the level of individual customer-facing features rather than enterprise-wide supply chain.
- **Ch 4 (Agentic AI Architecture)**: the task decomposition layer and tool-use pattern from that chapter are directly exercised in this chapter's "break the use case into tools" methodology.
- **Ch 10 (Implementation Approach to Agentic AI)**: picks up exactly where this chapter leaves off — implementing the TnTMart assistant bot and its related practical/technical aspects in full.
