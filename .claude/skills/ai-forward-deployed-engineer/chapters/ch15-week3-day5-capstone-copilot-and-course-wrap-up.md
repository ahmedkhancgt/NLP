# Week 3, Day 5: The Capstone Copilot, Handover, and Course Wrap-Up

## Lessons Covered
- Lesson 117: The Capstone Project - An UnderCover Copilot Agent for Claims Handlers
- Lesson 118: Build the Copilot with Claude Code and Give It Memory for Each Handler
- Lesson 119: The Copilot's LangGraph Graph and Two Agents on AgentCore Runtime
- Lesson 120: The Handover Conversation, the Success Criteria and Your Assignment
- Lesson 121: The Final Teardown - Delete UnderCover from AWS and Check Your Costs
- Lesson 122: Your Skills, FDE Job Postings, Next Steps and Completing the Course

## Core Idea
The capstone proves every Week 3 technique at once by building a second agent — a Copilot that acts on a handler's behalf with its own namespaced memory and its own AgentCore Runtime deployment — and the single most important FDE discipline, demonstrated in the closing "handover conversation," is to define success criteria at project inception and then actually measure them at the end, in front of the client, rather than simply declaring victory.

## Frameworks Introduced
- **The UnderCover Copilot Pattern**: a second LangGraph agent, separate from the claim agent, that (1) has durable state (LangGraph checkpoints in Supabase, so any conversation can be picked up later), (2) runs on a production runtime (its own AgentCore Runtime deployment, a session per conversation, keeps working after replying), (3) has namespaced long-term memory (AgentCore Memory, one namespace per handler rather than per customer), and (4) reuses the claim agent's existing tools but invoked under the logged-in handler's own permissions — so it can only do what that handler could do themselves.
  - When to use: whenever you want to give a human operator (not an end customer) an AI assistant that can act with the operator's own authority inside an existing product, rather than building a wholly separate automation.
  - How: give the Copilot its own LangGraph graph (nodes for recall, agent, tool-calling, check_sign_off, confirm, cancel/sign_off, remember), its own AgentCore Runtime deployment alongside the existing agent's, and a second AgentCore Memory object scoped by handler ID instead of customer ID.
- **Second-Opinion Plan Review Across LLM Families**: after Claude Code writes an implementation plan, have a different coding agent from a different LLM family (the instructor uses Codex/OpenAI) independently review that plan and write feedback, then have the original agent (Claude Code) selectively incorporate only the feedback it agrees with before implementing.
  - When to use: for a complex capstone-scale build, as a cheap way to get a second, differently-trained perspective on architecture risk before committing engineering hours.
  - How: (1) ask agent A to write a plan to a file; (2) ask agent B (different family) to review that plan and write feedback to a separate file; (3) ask agent A to review the feedback and update its own plan only where it agrees, explicitly instructing it not to start implementation yet.
- **The Handover Conversation**: a structured, live, five-question conversation with the deployed system (ideally performed with or in front of the client) that proves the delivered product works and meets its agreed success criteria — not a demo of features, but a demonstration of outcomes.
  - How: ask the deployed Copilot (1) to run a real batch of work ("work the queue with 10 agents and tell me what needs me"), (2) to explain why a specific item needs human attention, (3) to take an action on your behalf (send a claim back with a note), (4) to perform a human-in-the-loop sign-off (ask before paying), and (5) to self-report against the previously agreed success criteria — this is the capstone validation of the entire Week 1 Day 1 "success criteria at inception, measured at handover" principle.
- **Three Success Criteria Categories (worked example)**: Straight-through rate (share of claims handled with no handler needed; target ≥70%, achieved 88%), Accuracy (at least as accurate as human handlers; target ≥60%, achieved 88%), Over-the-limit compliance (every payout above the approval limit signed off by a person; target 100%, achieved 1-of-1 = 100%).
  - When to use: as a template for defining multi-dimensional success criteria for any agent-with-autonomy-threshold system — covering throughput/efficiency, quality, and governance/compliance simultaneously.
- **The Final Teardown's Three Checks**: (1) UnderCover's own resources, checked by name/service (both runtimes, Gateway and its Policy, credential providers, both Memories, the app and context-layer Lambda functions with their roles and logs, the CLI's encryption key and CDK bootstrap); (2) everything on the whole account via AWS Resource Explorer, by service and region; (3) what it actually cost, today and over the last 30 days, from the budgets set up on Day 1.
  - When to use: as the standard end-of-engagement checklist before walking away from any cloud deployment — verify by name, verify by account-wide scan, and verify by cost, not just one of the three.

## Key Concepts
- **Two separate AgentCore Runtime deployments, one AWS account**: `agentcore.json`'s `runtimes` section can list multiple entry points (`main.py` for the claim agent, `copilot.py` for the Copilot), each deployed and invoked independently, while sharing the same Gateway, Policy, and context layer.
- **Separate memory namespaces for separate concerns**: the claim agent's memory lives under `/customers/{actorId}/...`; the Copilot's memory lives under `/handlers/{actorId}/...` with its own two strategies (user-preference and semantic) — structurally the same pattern as Day 3's memory design, applied to a different entity (handler instead of customer).
- **The Copilot sidebar**: a UI panel showing each tool call as a card as it happens (e.g., `get_claim`, `list_steps`, `lookup_term`) plus the full answer once the turn completes — in-platform observability extended to a conversational interface, reusing the same "surface every step" principle from Day 3.
- **"It can do only what the handler can do"**: the Copilot's defining permission boundary — it operates under the logged-in handler's own login/authority, not an elevated or separate identity, so a sign-off still genuinely requires the handler's explicit action ("Sign off and pay" button) even when the Copilot drafts the recommendation.
- **Claude Code's divergent interpretation of a simple request**: asked merely to print the LangGraph graph diagram, Claude Code instead spent ~10 minutes and ~50,000 tokens building and publishing an elaborate interactive web artifact — illustrating both the impressive upside and the unpredictable scope-creep risk of giving a coding agent an open-ended instruction.
- **316 job postings / 184 employers analysis**: the instructor's review found AI Engineering mentioned in 88% of FDE-related postings, AI Architecture in 79%, Forward Deployed Engineering specifically in 95%, "solved real-world business problems" in 88%, and "delivered to customers" in 78% — framing the course's three weeks as directly mapped to what the market actually screens for.
- **Compensation context**: FDE-adjacent roles span widely (examples cited: up to ~$400K+ base for individual contributor/management roles at frontier labs like OpenAI/Anthropic; OpenAI FDE role cited at $153K–$325K base with equity, with senior top performers potentially exceeding $1M total compensation).
- **"You don't need to tick every box"**: the instructor cites Anthropic's own job posting language — most applicants won't match every bullet, and that's expected; the point is majority fit, not 100% fit.
- **Final cost outcome**: the instructor's own October AWS spend for the entire course's AgentCore usage was eight cents, with nearly all other account spend attributable to unrelated personal projects — reinforcing that the cloud-cost fear-mongering from Day 1 was about discipline, not actual expense at this scale.

## Mental Models
- Treat the Copilot capstone as proof-by-construction that the whole Week 3 toolkit (gateway, context layer, memory, durable execution, runtime) composes cleanly — it's built almost entirely by reusing existing tools and patterns under a new identity (the handler) rather than inventing new infrastructure.
- Use a second LLM family as a "different pair of eyes" the same way you'd use a second human reviewer — the value isn't necessarily superior intelligence, it's a differently-trained perspective catching different blind spots.
- Treat the handover conversation as the real deliverable of an FDE engagement, not the code — a system is only "delivered" when someone (ideally the client, live) can see it answer "are we meeting the success criteria we agreed?" with evidence.
- Think of the final teardown's three checks as a "trust but verify" triad: trust your own inventory of what you built (check 1), verify nothing else snuck in across the whole account (check 2), and verify the bill agrees with both (check 3).

## Anti-patterns
- **Declaring a project successful without re-measuring against the originally agreed success criteria**: the instructor repeatedly frames this as the single most important FDE discipline and the biggest single risk of AI project failure — the capstone's final Copilot exchange exists specifically to model doing this correctly.
- **Giving a coding agent an open-ended, loosely specified request late in a build** ("make me a diagram of the graph"): can trigger wildly disproportionate scope and token spend (10 minutes, 50K tokens, a whole interactive artifact) — be specific about exactly what you want when you want a quick, cheap answer.
- **Skipping independent review on a complex capstone-scale plan**: relying on a single coding agent's self-review misses the value of a second, differently-trained model's perspective on implementation risk.
- **Walking away from a cloud deployment after only checking the resources you remember creating**: the final teardown's three-check pattern exists precisely because a single "did I delete what I think I made" pass misses stray resources — always also scan the whole account and check the actual bill.
- **Believing you need to match every bullet point in a job posting before applying**: the instructor explicitly counters this, citing Anthropic's own hiring language, as a confidence-sapping misconception worth actively correcting.
- **Treating the Copilot's elevated convenience as elevated authority**: a well-designed copilot for a human operator must still operate strictly within that operator's own existing permissions — convenience of interface should never imply an expansion of what's actually allowed to happen.

## Code Examples
```python
import week3day4_aws as aws
# ...
for name in ["OPENROUTER_API_KEY", "DATABASE_URL", "VITE_SUPABASE_URL", "UNDERCOVER_API_KEY"]:
    print(f"{name}: {'found' if os.getenv(name) else 'missing, please add it to .env'}")
```
- **What it demonstrates**: the Day 5 lab reuses the entire Day 4 deployment module (`week3day4_aws`) unchanged — a concrete illustration that the capstone's "new app" deployment is additive on top of existing infrastructure, not a rebuild, and a simple pre-flight check pattern worth copying into any deployment notebook (verify every required secret is present before spending time on cloud calls).

```python
agentcore_control = session.client("bedrock-agentcore-control")
agentcore_runtime = session.client("bedrock-agentcore")
```
- **What it demonstrates**: the two distinct boto3 clients needed to manage AgentCore — `bedrock-agentcore-control` for administrative operations (creating Gateways, Policies, Memories) versus `bedrock-agentcore` for actually invoking a deployed runtime — a distinction worth remembering when scripting against AgentCore outside the CLI.

## Reference Tables

| Success criterion | Target | Achieved | Met? |
|---|---|---|---|
| Straight-through rate (claims handled with no handler) | ≥70% | 88% | Yes |
| Accuracy (at least as accurate as handlers) | ≥60% | 88% | Yes |
| Over-the-limit compliance (every over-limit payout signed off by a person) | 100% | 1 of 1 (100%) | Yes |

| Agent | Memory namespace pattern | Runtime entry point |
|---|---|---|
| Claim agent | `/customers/{actorId}/...` | `main.py` |
| UnderCover Copilot | `/handlers/{actorId}/...` | `copilot.py` |

| Job-posting coverage area (316 postings, 184 employers) | % mentioning it |
|---|---|
| Forward Deployed Engineering | 95% |
| AI Engineering | 88% |
| Solved real-world business problems | 88% |
| AI Architecture | 79% |
| Delivered to customers | 78% |

| Final teardown check | What it verifies |
|---|---|
| UnderCover's resources, by name | Each service (runtimes, Gateway, Policy, credentials, Memories, Lambda functions, bootstrap) is actually gone |
| AWS Resource Explorer, whole account | Nothing unaccounted-for remains anywhere, in any region |
| Budgets / billing console | Actual spend today and over 30 days matches expectations |

## Worked Example
**Capstone build**: Claude Code is given `labs/week3day5_product_COPILOT.md` (copied to `docs/COPILOT.md`) describing a Copilot sidebar that acts as the handler, can look things up, asks before paying, surfaces its work step by step, remembers per-handler, and can report success criteria. After Claude Code writes an initial plan, the instructor runs an independent cross-check: Codex (a different coding agent, different LLM family) reviews that plan and writes feedback to `docs/REVIEW_PLAN_COPILOT.md`; Claude Code is then told to review that feedback and update its own plan only where it agrees, explicitly without starting implementation yet. The refined plan has five phases (conversation storage in the app; the Copilot's graph with lookups, actions, and pause/resume; success-criteria checking plus per-handler memory; the Copilot as an AgentCore app runnable both locally and on Runtime; the sidebar UI) plus docs and live checks. Implementation then runs roughly 4-5 hours.

**Lab 1 (deployment and the handover conversation)**: The new app (with Copilot routes and sidebar) is redeployed to Lambda; Day 3's pieces (context layer, Gateway with its three policies) are brought back exactly as before; two separate AgentCore Memory objects are created — the familiar claim-agent memory (customer-namespaced, semantic + summary strategies) and a new Copilot memory (handler-namespaced, semantic + user-preference strategies). Both runtimes (`main.py` for the claim agent, `copilot.py` for the Copilot) are deployed via `agentcore deploy`, confirmed healthy via independent health-check invocations (~8.6 seconds for a cold start), and the app's configuration is updated to point at both runtime ARNs. The database is reseeded, and the actual "handover conversation" is performed live in the browser, signed in as handler Dana Whitfield: (1) "Work the queue with 10 agents and tell me what needs me" triggers a real dispatch of 10 claim agents on Runtime, visible as live tool-call cards in the sidebar, finishing with 19 approved, 3 declined, 4 referred, and a list of items needing Dana; (2) "Why does CLM-1011 need me?" gets a correct explanation (proposed $1,500 exceeds the $1,000 approval limit); (3) "Send it back with a note that we will not be covering the black dress shoes" causes the Copilot to call `send_back_claim`, which triggers a new claim-agent session that recalculates and auto-approves at $920 (under the limit once the shoes are excluded) — an emergent, unprogrammed composition of existing tools; (4) "Sign off CLM-1008" has the Copilot present the $2,240 payout for explicit confirmation, and only pays once Dana clicks "Sign off and pay" inside the sidebar, with a cost/timing readout (three model calls, $0.004, 22 seconds); (5) "Are we meeting the success criteria we agreed?" produces the three-row table above, all three criteria met. A side note: asking Claude Code off-script to "diagram the Copilot's graph" produced a 10-minute, ~50,000-token interactive web artifact rather than the simple LangGraph diagram intended — flagged as both impressive and a cautionary example of agent scope creep on loosely specified requests.

**Lab 2 (final teardown)**: `show_everything()` first inventories all tagged and untagged resources under the account (Lambda functions, runtimes, Gateway, Policy engine, credentials, two Memories, CloudFormation stacks, IAM roles, log groups, secrets, S3 buckets, encryption keys) — intentionally broad so stray resources surface. Teardown proceeds in order: local env keys removed, log groups deleted, Day 3's Gateway/Policy/Memories removed, the Copilot's separate Memory explicitly deleted (since it falls outside the Day 3 teardown routine), the app and context-layer Lambda functions and their roles/logs removed, and finally the CDK bootstrap (its S3 deployment bucket and, if needed, its KMS encryption key) torn down. A post-teardown scan for anything tagged `undercover` returns empty; a broader account-wide check leaves only free resources (IAM user/group/policy, budgets). The instructor's actual October AWS bill for the entire course shows eight cents spent on AgentCore, with all other account spend attributed to unrelated personal projects — confirming the Day 1 cost-discipline message held true in practice.

## Key Takeaways
1. A copilot for a human operator should reuse your existing agent tools and infrastructure patterns under that operator's own login/permissions, rather than being built as a separate, more-privileged system.
2. Getting a second opinion from a different LLM family on a complex implementation plan — review, then selectively incorporate agreed feedback — is a cheap, worthwhile step before committing hours of agentic build time.
3. The handover conversation, not the shipped code, is the actual deliverable of an FDE engagement — design it as a live, evidence-based demonstration against pre-agreed success criteria.
4. Define multi-dimensional success criteria up front (throughput/efficiency, quality/accuracy, governance/compliance) and build the capability to self-report against them into the product itself.
5. An open-ended instruction to a coding agent late in a build can produce wildly more (or less) than intended — be specific when you want a quick, bounded answer.
6. Tear down cloud infrastructure with a three-part check — your own named resources, a whole-account scan, and the actual bill — rather than trusting any single verification method alone.
7. Market data (job postings) consistently rewards the exact three-week structure of this course: AI engineering fundamentals, AI architecture/evals, and forward-deployed delivery skills — none of the three alone is sufficient for most FDE roles.
8. You do not need to match every bullet point in a job posting to be a credible candidate; majority fit is the realistic bar, including at frontier labs.
9. Real-world proof — your own deployed assignment, a second project solving someone else's problem, and publicly sharing both — is what converts course completion into job-ready credibility.

## Connects To
- **Week 3, Day 4**: Directly extends the production deployment built there — reuses the same `week3day4_aws` deployment module unchanged, adds a second AgentCore Runtime and a second Memory object alongside the existing claim agent, and performs the same style of teardown with an extra deletion step for the Copilot's own memory.
- **Week 3, Day 1**: The handover conversation is the practical fulfillment of Day 1's abstract "success criteria agreed at inception, measured at handover" principle — this is where that governance concept becomes a literal conversation with a working system.
- **Appendix A (The UnderCover Repository) and Appendix B (The Bench Repository)**: both appendices, referenced throughout the course as the companion projects' full source code, hold the complete implementation underlying everything built across these five chapters, but are out of scope for this skill's chapter content (lines 28987 onward in the source text are excluded from chapter generation).
