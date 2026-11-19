# Week 3, Day 4: Production Deployment to AgentCore Runtime

## Lessons Covered
- Lesson 110: Agents in Production - Configuration, Secrets and Managed Runtimes
- Lesson 111: The AgentCore CLI - Create, Deploy and Invoke a Hello World on AWS
- Lesson 112: Tour a LangGraph Agent's Code and Run It Locally with agentcore dev
- Lesson 113: Deploy to AgentCore Runtime and Resume a Paused Claim in a New Session
- Lesson 114: Claude Code Builds Send Back and Turns the Agent into an AgentCore App
- Lesson 115: Deploy the UnderCover Agent to AgentCore Runtime and Run a Swarm of 25
- Lesson 116: Human in the Loop on AgentCore Runtime - Sign Off and Send Back a Claim

## Core Idea
Deploying an agent to production is mostly housekeeping (settings, secrets, permissions, limits, logging) that a managed runtime like AgentCore absorbs for you; and because the agent is built on LangGraph with durable, database-backed checkpoints, a session can vanish entirely and still be resumed later from exactly where it left off — which is what makes human-in-the-loop sign-off/send-back work across arbitrarily long real-world delays.

## Frameworks Introduced
- **Five Production Configuration Concerns**: settings from the environment (where are the Gateway/Memory/app addresses), secrets in a vault (keys and passwords never hardcoded — AgentCore Identity holds them), who may call it (restrict invocation to a specific IAM role), limits (e.g., a session ends after 60 idle seconds), logs and cost (CloudWatch plus a custom spend-monitor).
  - How: walk this checklist for any agent going to production, whatever runtime you use.
- **Your-Infra vs. Managed-Runtime Tradeoff**: own infrastructure (Lambda/ECS/Cloud Run/Kubernetes/Render) is cheaper/free and more flexible but requires building everything yourself; a managed runtime (AgentCore, Google Agent Runtime, Microsoft Foundry, LangSmith Deployment) gets permissions/security/memory/gateway integration "out of the box" fast, but costs more at scale and is opinionated — doing things its way is easy, deviating is harder.
  - How: AgentCore is framework-agnostic (works with LangChain/LangGraph, Strands, OpenAI Agents SDK, etc.) but is easiest when paired with AWS Bedrock for inference; the course deliberately uses OpenRouter instead and accepts "a couple of hoops" to jump through as the cost of staying model-agnostic.
- **The AgentCore CLI Lifecycle**: `agentcore create --template <template>` (scaffold a project) → `agentcore dev` (run locally against a dev server) → `agentcore deploy` (zip and push to AWS Runtime) → `agentcore invoke --runtime <name> --payload <json>` (call the deployed agent, or call it from Python via boto3) → `agentcore remove all` followed by another `agentcore deploy` (teardown — remove-all alone doesn't fully delete; you must redeploy the now-empty config to actually delete cloud resources).
  - When to use: this is the standard path from a laptop prototype to a cloud-hosted agent on AgentCore.
- **Durable Execution Resuming a Vanished Session**: an AgentCore Runtime session is a disposable microVM that can be destroyed (idle timeout, deliberate stop) without losing anything, because LangGraph's checkpointer persists graph state to Postgres (Supabase) keyed by thread ID — a brand-new session, possibly weeks later, resumes the exact same graph from its last checkpoint by passing the same thread ID with action `resume`.
  - How: always design the agent as a LangGraph graph with a database-backed checkpointer (e.g., PostgresSaver) rather than relying on any in-memory session state, so human-in-the-loop delays of arbitrary length never break continuity.
- **Dispatcher + Swarm Architecture on Runtime**: one AgentCore Runtime session can run a dispatcher (a model-less LangGraph node whose only job is to fan work out) that spins up and manages up to a batch (e.g., 25) of claim-agent sub-executions within that same session/process, while the calling app's own request returns immediately rather than blocking — "reply at once, keep working, let the app poll." AgentCore Runtime sessions can run up to 8 hours for genuinely long-running work.
  - When to use: any workload where a single trigger (one dispatch click) should fan out to many parallel agent instances without making the caller wait synchronously.
- **Sign Off vs. Send Back (Human-in-the-Loop Resolution)**: a handler reviewing a claim above the autonomy threshold has exactly two outcomes — Sign off (the app pays directly; no further agent involvement needed) or Send back (a note is appended and a brand-new session resumes the same LangGraph thread, letting the agent reconsider with the new information).
  - How: build this as a first-class product feature (not an afterthought) — it's the concrete mechanism by which the Day 1 autonomy dial ("approval above a threshold") actually gets implemented in a running system.

## Key Concepts
- **AgentCore project scaffold**: `agentcore create --name <name> --template <template>` generates a folder with `agentcore/agentcore.json` (project settings: name, Python version, entry point, credentials) and an `app/agent/main.py` with an `@app.entrypoint`-decorated `invoke` function — the equivalent of `crewai create` for other frameworks.
- **`agentcore.json`**: holds project name, Python version (pinned to 3.12 here, since AgentCore defaults to 3.14 which breaks some data-science packages), entry point path, and registered credentials (e.g., `OPENROUTER_API_KEY`, `UNDERCOVER_API_KEY`, `DATABASE_URL`), added via `agentcore add credential`.
- **`.env.local`**: the file `agentcore dev` reads automatically for local-run secrets — kept git-ignored, populated from the real `.env`.
- **Cold start vs. warm session**: the first call into a new AgentCore Runtime session takes several seconds (microVM spin-up); a second call to the same session ID is near-instant (a fraction of a second) because the VM is already warm.
- **Thread ID vs. Session ID**: the session ID identifies a particular AgentCore Runtime microVM instance (ephemeral, destroyed after idle timeout); the thread ID identifies a LangGraph conversation/graph state (durable, persisted to Supabase) — these are deliberately decoupled so a new session can continue an old thread.
- **`npm ci` requirement for a pre-created project**: because the course's `undercoverlab`/production agent folders were committed to the repo rather than generated fresh with `agentcore create` on the student's own machine, a `npm ci` (clean install) is needed once before `agentcore dev`/`deploy` will work.
- **The dispatcher node**: a LangGraph node in the UnderCover agent graph that has no LLM at all — its only job is to divide work across up to 25 claim agents and track completion, while the app polls.
- **`invoke_agent_runtime`**: the boto3 call (via the `bedrock-agentcore` client) used to call a deployed AgentCore Runtime agent directly from Python, as an alternative to the `agentcore invoke` CLI command.
- **InvokeAgentRuntime IAM permission**: the specific permission the UnderCover app's Lambda role needs granted before it's allowed to call into the deployed agent runtime — a concrete example of the "who may call it" production configuration concern.
- **26 seeded claims, not 25**: the Day 4 swarm demo seeds 26 claims rather than 25 because it deliberately includes the CLM-1901 prompt-injection claim from Day 2's security lab, to prove the guardrail still catches it when running for real on the cloud.
- **Security check survives the move to production**: on the swarm run, CLM-1901's injected "pre-approved by Dana Whitfield" note is caught by the model-check guardrail with the message "the claim story includes an instruction," demonstrating that even if that guardrail had failed, the Day 2/3 Cedar policy on the Gateway would have been a second, independent backstop.

## Mental Models
- Think of AgentCore Runtime sessions as disposable containers and LangGraph's checkpointer as the actual "memory" of progress — the container can die at any time; the graph's state in Postgres is what survives and lets anything pick the work back up.
- Treat "reply at once, keep working, let the app poll" as the standard pattern for any long-running or fan-out agent workload — never make a synchronous caller block on the full completion of parallel sub-work.
- When deciding your own-infra vs. managed-runtime tradeoff, frame it as "fast and opinionated" vs. "flexible and more work" — pick based on whether the runtime's defaults align with your model/framework choices (the course picks OpenRouter over Bedrock specifically to stay model-agnostic, accepting some extra setup friction as the price).
- View "sign off" and "send back" as the two terminal states of the human-in-the-loop autonomy gate introduced on Day 1 — this day is where that abstract governance concept becomes concrete, working code.

## Anti-patterns
- **Assuming `agentcore remove all` alone deletes cloud resources**: it only declares the configuration empty; you must run `agentcore deploy` again afterward to actually push the (now-empty) deletion to AWS.
- **Relying on in-memory or session-local state for agent progress**: this breaks the moment a session ends (idle timeout, handler going on vacation for a week); always checkpoint to a durable store (Postgres/Supabase) keyed by thread ID instead.
- **Blocking the caller on a swarm's full completion**: the UnderCover app returns immediately when a dispatch is kicked off; the dashboard polls for status rather than holding a synchronous connection open while 25 agents work.
- **Choosing AgentCore's "easiest path" (Bedrock inference) without checking whether it matches your actual model strategy**: the instructor explicitly chooses OpenRouter over Bedrock despite it being more friction, because Bedrock inference setup ("regions and quotas and permissions") is its own sweat-inducing project best avoided unless you specifically want it.
- **Skipping the IAM permission step when wiring an app to a newly deployed agent**: the Lambda role needs an explicit `InvokeAgentRuntime` grant — without it, the app cannot call the agent even though both are deployed and healthy independently.
- **Treating "nothing visibly changed" after a cloud migration as underwhelming rather than as success**: the instructor explicitly frames the anticlimactic feel of the swarm/sign-off/send-back demos working identically to the laptop version as the entire point — a successful production migration should be invisible to the end user.

## Code Examples
```python
def package(name, group, folders):
    """Zip folders of the repo with a dependency group's packages for Lambda."""
    folder = BUILD / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    requirements = BUILD / f"{name}-requirements.txt"
    export = ["uv", "export", "--only-group", group, "--no-hashes", "--no-emit-project", "--quiet", "--output-file", str(requirements)]
    subprocess.run(export, cwd=REPO, check=True)
    subprocess.run(["uv", "pip", "install", "--quiet", "--target", str(folder), *LINUX, "-r", str(requirements)], cwd=REPO, check=True)
    for source, target in folders.items():
        shutil.copytree(REPO / source, folder / target, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    zip_path = Path(shutil.make_archive(str(BUILD / name), "zip", folder))
    print(f"{zip_path.name}: {zip_path.stat().st_size / 1_000_000:.1f} MB (Lambda takes up to 50 MB)")
    return zip_path
```
- **What it demonstrates**: a reusable Lambda-packaging helper (used for both the app and the context layer) that exports a dependency group with `uv`, pip-installs it for the Linux/x86_64 Lambda target, copies in repo source folders, and zips the result — the same boilerplate pattern used repeatedly across Days 2-4 for deploying Python to Lambda.

```python
@app.entrypoint
def invoke(payload):
    """Handle a chat, a health check, or starting/resuming a claim."""
    # dispatches on payload["action"]: "health" -> {"healthy": True}
    # "chat" -> chat(payload["prompt"])
    # "start"/"resume" -> work_claim(action, payload["claim_id"], thread_id=payload.get("thread_id"))
```
- **What it demonstrates**: (paraphrased structure, per the lesson's walkthrough) the pattern for an AgentCore app's single entry point — one function that branches on a lightweight `action` field in the payload, letting the same deployed artifact serve health checks, ad hoc chat, and durable claim-graph start/resume calls.

## Reference Tables

| Hosting choice | Pros | Cons |
|---|---|---|
| Your own infrastructure (Lambda, ECS, Cloud Run, Kubernetes, Render) | Cheaper or free; much more flexible | More setup work; state/scaling are yours to build |
| Managed runtime (AgentCore, Google Agent Runtime, Microsoft Foundry, LangSmith Deployment) | Permissions, security, memory, gateway integration built in; fast to get running | Costs more at scale; opinionated — easiest on its own rails, harder off them |

| AgentCore CLI command | Purpose |
|---|---|
| `agentcore create --template <template>` | Scaffold a new agent project |
| `agentcore dev` | Run a local dev server for the agent |
| `agentcore deploy` | Zip and push the agent to AgentCore Runtime |
| `agentcore invoke --runtime <name> --payload <json>` | Call the deployed agent |
| `agentcore remove all` then `agentcore deploy` | Tear down — remove-all declares nothing, the follow-up deploy actually deletes |

| Human handler action | Result |
|---|---|
| Sign off | The app pays immediately; no agent call needed |
| Send back (with a note) | A new session resumes the same LangGraph thread; the agent reconsiders with the new information |

## Worked Example
**Lab 1 (AgentCore Runtime fundamentals)**: A trivial `agent-python-minimal` template project (`hellolab`) is created, its `invoke` function edited to return "Hello, Forward Deployed Engineer," and deployed with a single `agentcore deploy`. Invoking it with an empty payload via `agentcore invoke` returns the custom message from a cold start in AWS us-east-1. It's torn down with `agentcore remove all` followed by `agentcore deploy` (the second step is required to actually delete the ~30 underlying AWS resources that were created). Next, a pre-built LangGraph project (`undercoverlab`, from the `agent-python-langchain` template) is toured: its `main.py` defines a `chat` function (a simple `create_agent` with an add-numbers tool) and a `work_claim` function that builds and invokes a small claim-approval LangGraph graph (nodes: propose, approval, revise) using a Postgres-backed checkpointer, dispatching on a `thread_id` and a start/resume action. Run locally with `agentcore dev`, a chat request ("What is 1,180 minus 240?") correctly returns 940 via the local HTTP server. Deployed to the real cloud with `agentcore deploy`, the identical chat call now runs in AWS and returns the same correct answer, first in ~2.5 seconds (cold start) then ~0.2 seconds (warm). The durable-execution centerpiece: a claim (9001) is started with a new thread ID, the agent proposes paying $1,180 (pending clarification on duplicate taxi receipts), the session is then deliberately killed; a direct Supabase query confirms three checkpoints persisted under that thread ID; a brand-new session, given the same thread ID and action `resume` plus a handler's note, picks up exactly where the dead session left off and produces a consistent follow-up answer — proving durable state survives total session loss. Everything is torn down and verified against the `monitor.py` spend tracker and the AWS Billing console (a fraction of a cent total).

**Lab 2 (full production deployment of the real UnderCover agent)**: Claude Code first implements `docs/RUNTIME.md`'s four changes (claim approvals over the limit wait for a handler; handlers can sign off or send back with a note, resuming the agent; the agent becomes a proper AgentCore app runnable identically on a laptop or on Runtime; the app enforces one run at a time locally while Runtime can run two sessions concurrently) — a ~4-hour Claude Code run with full end-to-end testing. In the lab: Day 3's pieces (app, context-layer MCP server, Gateway with both targets and the enforced Cedar policy, AgentCore Memory with semantic+summary strategies) are redeployed exactly as before. The real UnderCover agent is then deployed as an AgentCore app via `agentcore deploy` (after an `npm ci` since the project folder came pre-populated in the repo rather than freshly scaffolded), configured with six settings (OpenRouter key, UnderCover API key, database URL, and the app/Gateway/Memory URLs) in `.env.local`. A health-check call confirms the deployed agent responds `{"healthy": true}` in about 4 seconds. The UnderCover app's Lambda IAM role is granted `InvokeAgentRuntime` permission and its environment variables updated to point at the new agent. The database is reseeded with 26 claims (25 ordinary plus the CLM-1901 prompt-injection claim from Day 2). From the browser dashboard — now fully backed by AWS end to end — a dispatch is kicked off as a swarm of agents; the app's call returns essentially instantly while the dispatch runs entirely on AgentCore Runtime with zero local CPU usage, settling moments later at 23 decided, 3 waiting for sign-off. One claim (Dorothy Gale's) is signed off directly with no agent involvement; a second (Owen Gallagher's, a lost-baggage claim with a possible double-counted $350 reimbursement) is sent back with a handler's clarifying note, and a new AgentCore session resumes the same thread, confirms the $350 was correctly deducted, and the handler then signs it off. Finally, CLM-1901 (the prompt-injection claim) is inspected in the dashboard and shown to have been caught by the Day 2 security-check guardrail running live on the cloud — "the claim story includes an instruction" — with the Gateway's Cedar policy standing as an unused second line of defense. Everything is torn down (`agentcore remove all` + `agentcore deploy`, `git restore agentcore.json`, removal of local env vars and log groups, teardown of Day 3's app/context/gateway/memory), leaving only free-idle Lambda functions and IAM roles.

## Key Takeaways
1. Deploying to production is mostly a checklist of housekeeping — settings, secrets, caller permissions, limits, logging/cost — and a managed runtime like AgentCore absorbs most of it, at the cost of being opinionated.
2. Choose your own infrastructure vs. a managed runtime based on whether the runtime's easy defaults (e.g., AgentCore's smooth path to Bedrock) actually match your architecture; accept some extra setup friction (as with OpenRouter) when staying model-agnostic matters more.
3. The AgentCore CLI's five commands (create, dev, deploy, invoke, remove all) cover the full laptop-to-cloud lifecycle; remember that teardown requires remove-all followed by another deploy, not remove-all alone.
4. Build durable execution into any agent that might need human-in-the-loop delays: a LangGraph checkpointer persisting to a real database (keyed by thread ID) lets a totally different session, hours or weeks later, resume exactly where a previous one left off.
5. Decouple session identity (ephemeral microVM) from thread identity (durable conversation state) — this is the architectural choice that makes "the handler went on vacation for a week" a non-issue.
6. For fan-out/swarm workloads, always reply to the caller immediately and let the long-running work (and multiple sub-agents) proceed server-side, polled rather than awaited synchronously.
7. "Sign off" and "send back" are the two concrete outcomes that implement the Day 1 autonomy-threshold concept in running code — build both as explicit, first-class product paths.
8. When migrating a working system to production infrastructure, the correct and desired outcome is that nothing visibly changes for the end user — treat that as success, not anticlimax.
9. Defense in depth (guardrail model-check plus Gateway Cedar policy) should be verified on the real production deployment, not assumed to carry over automatically from local testing — the Day 4 swarm run explicitly re-confirms the Day 2 prompt-injection catch works live on the cloud.

## Connects To
- **Week 3, Day 3**: Reuses and redeploys every piece built there (app, context-layer MCP server, Gateway with Cedar policy, AgentCore Memory) exactly as-is — today's only new addition is the agent itself moving from a local notebook onto AgentCore Runtime.
- **Week 3, Day 1**: The risk-based autonomy dial and "approval above a threshold" concept from Day 1 is directly implemented here as the sign-off/send-back mechanism.
- **Week 3, Day 5**: The architecture diagram explicitly teases "Tomorrow?!" with one more piece still to add — the capstone day builds on this fully-deployed system rather than starting fresh.
- **LangGraph durable execution (Week 2)**: This day is the production payoff of the checkpointer/human-in-the-loop patterns introduced earlier in the course — the same PostgresSaver-backed graph now survives not just a pause within one process, but the complete destruction and replacement of the process itself.
