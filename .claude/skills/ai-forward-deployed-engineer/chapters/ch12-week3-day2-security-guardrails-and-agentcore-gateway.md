# Week 3, Day 2: Security, Guardrails, and the AgentCore Gateway

## Lessons Covered
- Lesson 93: AI Agent Security Risks - Sandboxes, API Scopes and Prompt Injection
- Lesson 94: The Lethal Trifecta for AI Agents and Four Kinds of Safety Guardrails
- Lesson 95: API, MCP and Model Gateways and an Overview of AWS Bedrock AgentCore
- Lesson 96: How AgentCore Gateway Turns APIs into MCP Tools and Checks Cedar Policies
- Lesson 97: Catching a Prompt Injection with Keyword and Model Checks in LangChain
- Lesson 98: Claude Code Gets UnderCover Ready for AWS - API Key, Guardrail and OpenAPI
- Lesson 99: Setting Up to Deploy on AWS - boto3 vs the Console, Terraform and CDK
- Lesson 100: Deploy the UnderCover App to AWS Lambda with a Function URL and API Key
- Lesson 101: Create an AgentCore Gateway, Call MCP Tools and Add a Cedar Policy
- Lesson 102: Enforce the Cedar Policy, Check CloudWatch and Tear Down the Gateway

## Core Idea
Agent security reduces to three concrete risks (runtime, API scope, prompt injection) that must be defended with layered guardrails — a model check, a gateway policy, and an app-level limit — none of which is individually bulletproof; and the AgentCore Gateway is the mechanism that turns a documented REST API into MCP tools while enforcing those policies on every call.

## Frameworks Introduced
- **Three Agent Security Risks**: (1) the runtime the agent executes in (sandboxing/permissions — largely solved by deploying to a managed runtime like AgentCore), (2) the scope of the APIs/tools it's given (constrain via API scope, not via prompting — e.g., give email draft-only access, not send access), (3) prompt injection (untrusted content tricking the model into harmful actions).
  - How: build constraints into the API/tool scope itself rather than relying on prompt instructions like "always identify yourself as an AI" — a scope restriction can never be bypassed by the model, a prompt instruction can.
- **The Lethal Trifecta (Simon Willison)**: private data + untrusted content + external communication, combined in one agent, is a maximal-risk configuration (classic example: the GitHub MCP server exposure, also a Supabase MCP incident). A wide-scope API that can move money or similarly cause damage can stand in for the private-data-plus-external-communication pair.
  - When to use: audit any agent design against these three factors whenever a new integration is added — the risk can appear later even if the original design didn't have it (e.g., adding customer reviews to a CMS sync later created the trifecta in the instructor's worked scenario).
  - How: if all three are present, add extra controls (never rely on "the model is smart enough" alone).
- **Four Kinds of Safety Guardrails**: (1) LLM validator — another model checks each response before it goes out; (2) coded checks — explicit code for problematic words, PII, API key patterns, on both inputs and outputs; (3) user flagging — a UI button for bad responses, with a review process; (4) comprehensive testing/audits — pre-go-live and continuous via observability, including testing guardrails themselves against golden/held-out data to track false-positive rates.
  - When to use: combine multiple kinds rather than relying on one; keyword/coded checks are brittle (defeated by rewording or translation) while model checks generalize better but cost more and need their own false-positive measurement.
- **API / MCP / Model Gateway triad**: API gateway (front door for company APIs — keys, rate limits, logs; e.g., Azure API Management, AWS API Gateway, Kong) + MCP gateway (turns APIs into tools, controls who can call which tool; e.g., AWS AgentCore Gateway, MuleSoft Agent Fabric) + Model gateway (one API in front of many models, budgets, fallbacks; e.g., AgentCore Gateway, OpenRouter, LiteLLM). Many products (notably AgentCore Gateway) now do all three.
  - When to use: agree gateway architecture up front, unlike most architecture decisions which the instructor says to defer via experimentation — gateways are an exception worth deciding early.
  - How: pick lightweight (a few lines of code, a model router like LiteLLM/OpenRouter) vs. heavyweight (full platform like AgentCore Gateway or Kong) based on migration risk — "if we needed to migrate off this gateway, would that be a big undertaking?"
- **A Gateway Call's Three Steps**: (1) inbound authentication — confirm the caller is who it claims (SigV4/IAM or OAuth); (2) policy decision — Cedar rules decide if the action is authorized; (3) outbound authentication — the gateway attaches the target's own credentials (API key, OAuth, IAM) before hitting the real endpoint/Lambda/MCP server.
  - How: treat step 1 as authentication, step 2+3 as authorization — "check identity, check the policy, add the target's credentials."
- **OpenAPI-to-MCP-Tools conversion**: document a REST API with the OpenAPI (formerly Swagger) standard, then AgentCore Gateway reads that spec and automatically exposes each endpoint as an MCP tool, with the documentation text becoming the tool description the model sees.
  - When to use: whenever a company already has standard OpenAPI-documented internal APIs — this is the simplest path to equipping an agent, versus building a bespoke MCP server.
- **Cedar Policy Language (log-only, then enforce)**: AWS's open-source policy language for writing `permit`/`forbid` rules evaluated on every gateway tool call; forbid always wins over permit. Roll out policies in `LOG_ONLY` mode first (records violations, lets calls through) before switching to `ENFORCE` mode.
  - How: write a baseline `permit` rule for all intended tools, then add specific `forbid` rules with `when { context.input.<field>.greaterThan(decimal("X")) }` conditions; test in log-only, verify via CloudWatch metrics, then enforce.

## Key Concepts
- **Sandbox/runtime risk**: largely mitigated by deploying to ring-fenced managed infrastructure like AgentCore rather than running an agent loosely on unprotected hardware.
- **API scope as a hard constraint**: restricting what an integration (email, Jira, etc.) can do at the API/permission level, which the model can never override — contrasted with prompt-based "soft" constraints.
- **Authentication vs. Authorization**: "who is calling" (SigV4/IAM check) vs. "what they may do" (Cedar policy decision) — a useful simplification, though SigV4 does carry some coarse-grained authorization too.
- **AgentCore suite of services**: Gateway (APIs→MCP tools), Policy (Cedar rule checks), Identity (agent/API key management), Observability (CloudWatch), Memory (Day 3), Runtime (Day 4), plus growing extras: Harness, Evaluations, Optimization, Registry, Code Interpreter, Browser, Web Search, Payments.
- **Credential provider**: AgentCore Identity's mechanism for storing a secret (e.g., `undercover-app-key`) that the Gateway attaches automatically on outbound calls.
- **Policy engine**: the Cedar rule-evaluation service attached to a Gateway; must itself be granted IAM permission (`AuthorizeAction`, `PartiallyAuthorizeActions`) before it can be used.
- **LOG_ONLY vs ENFORCE mode**: LOG_ONLY records what would be refused without blocking; ENFORCE actually blocks calls. Always roll out new policies log-only first.
- **SigV4**: AWS's request-signing scheme used to authenticate an MCP client (the agent, or in the lab, the student's IAM user) to the Gateway.
- **Mangum**: the adapter library Claude Code installs to let a FastAPI app run as an AWS Lambda function.
- **boto3 vs. console vs. Terraform vs. CDK**: four ways to drive AWS infrastructure; the lab deliberately uses boto3 directly (simplest, no new DSL) over Terraform/CDK (powerful but overkill for one week) or the console (functional but clunky).
- **Gateway migration risk**: the recurring lens for choosing a gateway product — how hard would it be to leave if the ecosystem (MCP, policy standards) changes.
- **"Get the gist" vs. "dig in" learning strategy**: the instructor's explicit advice that for infrequently-touched infrastructure like Cedar policy syntax, understanding the concept is more valuable than memorizing the API — review documentation fresh each time you actually need it (every few months).

## Mental Models
- Treat every new integration as a standing trifecta audit, not a one-time check at design time — a later, unrelated change (adding customer reviews to a synced CMS) can silently complete the trifecta.
- Picture the gateway's three jobs as a security checkpoint: ID check (authn) → rulebook check (authz/Cedar) → badge issued for the destination (outbound creds) — this three-step mental model clarifies what "the gateway talks MCP twice" actually means (agent-to-gateway is MCP; gateway-to-external-MCP-server is also, separately, MCP).
- Use "three places to catch the same $1,000 rule" as the template for defense in depth: model-level check, gateway Cedar policy, and backend application limit — each layer independently sufficient, none alone trusted.
- When choosing infrastructure-as-code tooling, weigh "powerful and reusable" (Terraform) against "AWS-specific but still code" (CDK) against "simplest, most transparent, no new DSL" (boto3 directly) — pick based on whether you'll reuse the setup elsewhere.

## Anti-patterns
- **Relying on prompt instructions to constrain agent behavior** (e.g., "always say you're an AI"): use API/tool scope restrictions instead — these cannot be talked around.
- **Keyword/coded checks as your only prompt-injection defense**: trivially defeated by rewording ("pre-approved" → "signed off by telephone") or switching language (English check bypassed by Spanish text).
- **Asking the same model to do the task and screen its own input in one pass**: a separate screening prompt/model, even a weaker one, often performs better and is independently testable — because the task "is this suspicious?" is simpler than the main decision task.
- **Deploying a new guardrail without measuring false positives**: the lab found GPT-4o mini as a screening model flagged 7 of 25 ordinary (non-malicious) claims — always validate screeners against known-good data, not just known-bad data.
- **Enforcing a new Cedar policy before testing it in log-only mode**: skips the chance to see what it would have blocked on real traffic before anything actually breaks.
- **Treating Gateway/Cedar syntax as something to memorize**: a low-frequency task (every few months) better handled by reading documentation fresh each time than by committing to memory.
- **Choosing a heavyweight all-in-one gateway without weighing migration cost**: "would it be a big undertaking to migrate off this?" should be asked before committing, given how fast this space is still moving.

## Code Examples
```python
SUSPICIOUS = ["ai claims assistant", "ignore previous instructions", "pre-approved", "approve this claim"]


@before_agent(can_jump_to=["end"])
def keyword_check(state: AgentState, runtime: Runtime):
    """Stop before the model sees a claim with a suspicious phrase in it."""
    claim_text = state["messages"][0].content.lower()
    found = [phrase for phrase in SUSPICIOUS if phrase in claim_text]
    if found:
        return {"messages": [{"role": "assistant", "content": f"The keyword check found {found}"}], "jump_to": "end"}
    return None
```
- **What it demonstrates**: a LangChain `before_agent` middleware hook that intercepts the message before the model ever sees it — the brittle keyword-based version of a guardrail, trivially defeated by rewording.

```python
policies = {
    "undercover_allow_agent_tools": f"""permit(
  principal is AgentCore::IamEntity,
  action in AgentCore::Action::"{TARGET_NAME}",
  resource == AgentCore::Gateway::"{gateway_arn}"
);""",
    "undercover_payment_limit": f"""forbid(
  principal is AgentCore::IamEntity,
  action == AgentCore::Action::"{TARGET_NAME}___pay_claim",
  resource == AgentCore::Gateway::"{gateway_arn}"
)
when {{
  context.input.amount.greaterThan(decimal("1000.00"))
}};""",
}
```
- **What it demonstrates**: a baseline Cedar `permit` rule for all agent tools, paired with a `forbid` rule that always overrides it for payouts over $1,000 — the actual enterprise-grade enforcement layer sitting in front of the app.

```python
def attach_policy_engine(mode):
    """Attach the policy engine to the Gateway in LOG_ONLY or ENFORCE mode."""
    for attempt in range(20):
        try:
            agentcore.update_gateway(
                gatewayIdentifier=gateway_id,
                name=GATEWAY_NAME,
                description=gateway["description"],
                roleArn=gateway_role["Arn"],
                protocolType="MCP",
                authorizerType="AWS_IAM",
                exceptionLevel="DEBUG",
                policyEngineConfiguration={"arn": engine_arn, "mode": mode},
            )
            break
        except agentcore.exceptions.ValidationException as error:
            if "GetPolicyEngine" not in str(error):
                raise
            print("Waiting for the role's new permission to reach AgentCore")
            time.sleep(5)
```
- **What it demonstrates**: the log-only-then-enforce rollout pattern for gateway policy, plus a practical IAM eventual-consistency retry loop (new role permissions can take a few seconds to propagate into AgentCore).

## Reference Tables

| Gateway type | Role | Examples |
|---|---|---|
| API gateway | Front door to company APIs; keys, rate limits, logs | Azure API Management, AWS API Gateway, Kong |
| MCP gateway | Turns APIs into tools; controls who can call which tool | AWS AgentCore Gateway, MuleSoft Agent Fabric |
| Model gateway | One API in front of many models; budgets and fallbacks | AWS AgentCore Gateway, OpenRouter, LiteLLM |

| AI gateway (ranked by use in AI projects per instructor) | Note |
|---|---|
| 1. AWS Bedrock AgentCore Gateway | Most mentioned in US job postings among these |
| 2. Azure API Management | AI gateway in 1,200+ enterprises (Microsoft, Nov 2025) |
| 3. Databricks Unity AI Gateway | Thousands of customers, often on by default |
| 4. Kong AI Gateway | Hundreds of AI Gateway customers (Aug 2026) |
| 5. Prisma AIRS (Portkey) | Bought by Palo Alto Networks (May 2026) |
| Early-stage | Apigee, MuleSoft Agent Fabric, Cloudflare AI Gateway, Bifrost |
| Dev favorites | LiteLLM, OpenRouter, Composio |

| Guardrail layer | What it checks | Can be bypassed by |
|---|---|---|
| Keyword check | Fixed phrase list in input text | Rewording, translation |
| Model check (LLM screener) | Meaning/intent of input text | Sufficiently clever/novel phrasing (cat-and-mouse) |
| Cedar gateway policy | Structured conditions on tool-call parameters | N/A if correctly scoped — enforced server-side |
| App-level limit | Business rule in application code | N/A if correctly scoped — final backstop |

## Worked Example
**Lab 1 (prompt injection guardrails, local)**: The claim CLM-1901 ("the watch on the beach") contains a hidden instruction in its story text claiming a senior handler pre-approved the claim and waived the exclusion. Running the Week 2 claim agent with GPT-6 Luna correctly declines it under the unattended-belongings exclusion, ignoring the injected note. Running the identical agent with GPT-4o mini (a 2024-era model) falls for it and approves the full $4,800 payout — demonstrating that susceptibility to injection is a function of model capability, not just prompt design. A keyword-check middleware (`before_agent` hook checking for phrases like "pre-approved") successfully blocks the original wording but is defeated once the note is reworded to "signed off by telephone" with no flagged keywords present. Replacing the keyword check with a model-based screener (a Pydantic `Screening` object produced by a cheap LLM call, checking for intent rather than literal words) catches both the original and the reworded version even when the screener itself runs on GPT-4o mini — because "is this suspicious?" is an easier task than the main claims decision. However, testing GPT-4o mini as the screener against the full 25-claim ordinary queue produced 7 false positives; switching the screener to GPT-6 Luna produced zero false positives while still catching every injection attempt.

**Lab 2 (AWS deployment, Gateway, and Cedar policy)**: Claude Code first re-engineers UnderCover per `docs/SECURE.md` (four changes: an `UNDERCOVER_API_KEY` shared secret between agent and backend, the Lab 1 model-check guardrail wired into the live claim agent, OpenAPI documentation for the backend's eight routes, and Mangum packaging for Lambda) — a multi-hour run with the harness's end-to-end test suite running after every commit. Then, step by step: the whole app (frontend, backend, data, dependencies) is zipped (~20MB, under Lambda's 50MB cap) and deployed to AWS Lambda with a public function URL; the deployed dashboard is confirmed reachable and still backed by the same Supabase database used all week. An AgentCore Gateway is created, the API key is registered with AgentCore Identity's credential provider, and the eight OpenAPI-documented routes are added as Gateway targets — a LangChain MCP client then confirms it sees exactly those eight tools and can successfully call `get_claim` and `search_documents`, with a request for a nonexistent claim correctly returning a 404 MCP error. A Cedar policy engine is attached in `LOG_ONLY` mode with a baseline `permit` rule plus a `forbid` rule on `pay_claim` over $1,000; a test call paying $4,800 on CLM-1901 passes the Gateway (log-only lets it through) but is still refused downstream by the app's own limit — confirming the layering works even before enforcement. Switching to `ENFORCE` mode causes the Gateway itself to refuse the $4,800 call immediately ("Tool execution denied... due to policy enforcement"), while a $400 call still passes the Gateway and is independently refused by the app because that specific claim was never approved. CloudWatch metrics (`AllowDecisions`/`DenyDecisions`), checked with a short lag, confirm one allow and one refusal recorded in LOG_ONLY mode, then zero allows and one refusal in ENFORCE mode. The lab ends with teardown of the Gateway, policy engines, credential provider, and IAM role (keeping only the free-running Lambda app), and an assignment to add a third Cedar policy forbidding `record_decision` approvals over $5,000.

## Key Takeaways
1. Classify every agent security concern into one of three buckets — runtime, API scope, or prompt injection — and address each with its own control, not a single catch-all.
2. Constrain agent capability through API/tool scope, never through prompt wording alone; scope restrictions cannot be talked around by the model.
3. Audit every agent integration against the lethal trifecta (private data + untrusted content + external communication) continuously, not just at initial design — later unrelated changes can complete the trifecta.
4. Layer guardrails (model check, keyword check, platform flagging, continuous testing) rather than trusting any single mechanism; always measure a guardrail's false-positive rate against known-good data, not just its catch rate against known-bad data.
5. Decide on gateway architecture (API, MCP, model) early — this is one architecture decision worth settling up front rather than deferring to experimentation — and weigh lightweight vs. heavyweight by migration risk.
6. A gateway call is authenticate → authorize (policy) → attach outbound credentials; design Cedar (or equivalent) policies with a baseline permit plus specific forbid rules, since forbid always wins.
7. Roll out every new policy in log-only mode first, verify via logs/metrics what it would have blocked, then switch to enforce.
8. Defense in depth means the same business rule (e.g., "$1,000 limit") should be enforced independently at the model, the gateway policy, and the application layer — redundancy here is the point, not waste.
9. For infrastructure you touch only every few months (like Cedar syntax), prioritize understanding the concept over memorizing the API; plan to re-read documentation when the need actually arises.

## Connects To
- **Week 3, Day 1**: Builds directly on the AWS account, IAM user, and budget/region setup established there — nothing here would run without it.
- **Week 3, Day 3**: Explicitly left unfinished — the Gateway built today gets its remaining pieces (context layer, memory) completed tomorrow, and the AgentCore Runtime (where the agent itself finally runs, instead of a local notebook) arrives Day 4.
- **MCP specification / OpenAPI standard**: This day is a direct practical application of both standards — OpenAPI as the documentation format AgentCore Gateway consumes, and MCP as the protocol the agent uses to discover and call the resulting tools.
