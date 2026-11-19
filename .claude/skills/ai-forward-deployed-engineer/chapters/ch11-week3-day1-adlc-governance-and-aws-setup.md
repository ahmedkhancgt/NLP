# Week 3, Day 1: The Agent Development Lifecycle, Governance, and AWS Setup

## Lessons Covered
- Lesson 85: Welcome to Week 3 and the Five Phases of the Agent Development Lifecycle
- Lesson 86: The FDE Through the Lifecycle and Prioritizing AI Use Cases in Discovery
- Lesson 87: Systems Architecture for AI Agents and Three Ways to Host an Agent
- Lesson 88: Governance for AI Projects, Agent Autonomy and Managing Many Agents
- Lesson 89: Responsible AI, Privacy with LLMs, Accountability, Handover and Adoption
- Lesson 90: The Six Steps to Set Up AWS and Your Responsibility for Cloud Costs
- Lesson 91: Open an AWS Account, Turn On MFA, Create a Budget and an IAM User
- Lesson 92: Configure the AWS CLI and Check Your AWS Setup from Python with boto3

## Core Idea
Delivering an AI agent at a client is a five-phase lifecycle (Discovery, R&D, Development, Implementation, Maintenance) in which the FDE is the bridge between business and technical teams at every phase, and governance, autonomy decisions and accountability must be designed in from the start — not bolted on after the model works.

## Frameworks Introduced
- **ADLC (Agent Development Lifecycle), "building agents" sense**: the emerging process for building and deploying AI agents — distinct from the other ADLC sense (Week 1 Day 3's "building code with agents," i.e. Claude Code / AI-driven SDLC). Always clarify which meaning is meant when someone says ADLC.
  - When to use: framing any client AI-agent delivery engagement.
  - How: five phases below, each with its own owner, tollgate and risk profile.
- **The Five Phases of the ADLC**: Discovery → R&D → Development → Implementation → Maintenance, with ownership, governance, evals and change management running throughout.
  - Discovery: sponsors/owner assigned, business case written, evals/baseline/success criteria set, adoption plan drafted.
  - R&D: data curation, experiments with evals, autonomy-level decisions, framework/architecture selection. Hardest phase to estimate — have a plan B and plan C.
  - Development: agent engineering, harness engineering, integrations, tests, observability.
  - Implementation: pilot/UAT, staged rollout, training/adoption, handover, release approval.
  - Maintenance: monitoring, self-improvement, human feedback, drift/bias testing, upgrade/merge/retire decisions, feedback to your own employer.
- **RICE scoring**: Reach, Impact, Confidence, Effort — standard product-management method to rank candidate use cases before applying the AI-specific screen below.
  - When to use: a client has a spreadsheet of candidate AI use cases and no prioritization method.
- **The Three Questions for Use-Case Screening**: (1) Does it avoid the Week 2 traps (is it a real, measurable business problem, not "I want an agent for X")? (2) Does it use proprietary data, connections and evals? (3) Is it automation, acceleration, or differentiation?
  - How: screen every candidate use case with all three before ranking with RICE.
- **Automation → Acceleration → Differentiation ("the bar has risen")**: three successive tiers of ambition for AI use cases — do things cheaper, then faster, then differently/new markets.
  - When to use: pushing back on a client that frames every AI initiative as cost-cutting; steer prioritization toward differentiation.
- **Three Ways to Host an Agent**: (1) your own infrastructure (Lambda, ECS, Cloud Run, Kubernetes, Render) — any framework/model, you own state and scaling; (2) a managed runtime (AWS AgentCore, Google Agent Runtime, Microsoft Foundry, LangSmith Deployment) — any framework/model, identity and memory built in; (3) the model lab's own managed agents (Claude Managed Agents, OpenAI Agents API) — only that lab's models, the lab's own harness.
  - How: default to option 1 for simplicity; move to option 2 when you need the built-in identity/memory/permission machinery (this is why the course uses AgentCore this week). Option 3 is hardest to justify in enterprise contexts.
- **Three-Tier Governance with Tollgates**: Firmwide Leadership (monthly/quarterly — portfolio, policy, risk tolerance) → Project Committee (monthly — scope, risk, releases; sponsor, owner, security, audit, operations) → Working Group (weekly — build/evaluate/run; product, engineering, business, FDE). A sponsor signs off at the end of each of the five ADLC phases (scope/budget → approve build → go/no-go → program success → change management).
  - When to use: at project inception, agree sponsors, roles and the tollgate criteria for each phase before work starts.
- **Risk-Based Autonomy Dial**: per-action classification into Automatic, Approval-required (above a threshold), or Prohibited, set by weighing risk appetite, consequence, verifiability, reversibility, security/privacy/IP sensitivity.
  - How: for every consequential agent action, score it on these factors and assign it to one of the three autonomy buckets; back it with a review screen, an audit trail, and a plan for delays/escalation/no-double-payment.
- **Six Pillars of Responsible AI**: Safety, Fairness/Bias, Security, Privacy, Transparency, Accountability — each needs an identified owner and a sign-off plan, budgeted for both during the project and after go-live.
  - How: walk every pillar at Discovery; assign an owner for each that's relevant; don't let any pillar become an unplanned-for showstopper later.

## Key Concepts
- **Discovery baseline and success criteria**: measured starting point and the agreed target that defines project success, set before build begins.
- **Plan B / Plan C**: alternative delivery strategies prepared during R&D in case the primary approach proves infeasible — core risk management for AI's inherent uncertainty.
- **Agent roster / agent registry**: the (controversial) practice of tracking AI agents like an HR database of staff (e.g., Microsoft Agent 365/Entra Agent ID, Workday Agent System of Record, AWS Agent Registry, OpenAI Frontier). The instructor's recommendation: track agents like software assets against the business problem they solve, not like employees.
- **Privacy as vendor risk, not something unique to LLMs**: sending data to an LLM API should be risk-assessed exactly like sending data to any SOC2-certified third-party vendor (e.g., Workday, Microsoft) — review the contract, the training policy, and minimize personal data sent.
- **The Lethal Trifecta (preview)**: mentioned here, fully covered Day 2 — private data, untrusted content, and external communication combined is the dangerous agent security pattern.
- **Accountability principle**: "You can never blame the LLM." A human/owner is always accountable for outcomes; as a builder you cannot invoke the "it hallucinated" defense available to a mere user of AI.
- **Handover**: the FDE's goal of making themselves unnecessary — ownership/escalation transferred, full knowledge transfer, ongoing measurement program (drift, adoption, bias) left running without you.
- **Adoption**: training, sponsor-driven top-down change management, demonstrated output accuracy, and transparency about usage — necessary because a delivered-but-unused system is not a successful project.
- **Feedback loop to employer**: an FDE's obligation (contract permitting) to bring learnings and reusable components back to their own employer after a client engagement, and to seed the next project's Discovery phase.
- **IAM user vs. root user**: root can do anything including closing the account; the IAM user is a narrower-permission account created for day-to-day/CLI/agent use.
- **us-east-1 as the course's single region**: chosen because every AgentCore feature used this week (Runtime, Gateway, Identity, Policy, Memory, Observability) is available there, including Memory, which six other regions lack.
- **No hard spending ceiling on AWS**: unlike prepaid LLM API credits, AWS (and GCP/Azure) bill on a contractual, uncapped basis — budgets only alert, they don't cut off usage.

## Mental Models
- Think of the two ADLC meanings as a disambiguation checkpoint: always ask "building agents, or building code with agents?" before using the term with a client or in an interview.
- Use the autonomy dial as a continuum, not a binary: a credit-card fraud classifier and something like Claude Code sit at opposite ends; most real systems (like the UnderCover claims agent) sit in the middle with human-in-the-loop thresholds.
- Treat an LLM provider relationship exactly like any other vendor risk review — "if a SOC2-certified vendor is fine, a SOC2-certified LLM provider is fine" neutralizes the irrational "it feels human so it must be learning on us" worry.
- Treat cloud cost responsibility like a seatbelt: it's not that AWS will bankrupt you, it's that you must actively drive the alerting and monitoring yourself because there's no airbag (hard cap).

## Anti-patterns
- **Investing in heavyweight, opinionated internal layers (prompt template repositories, large bespoke context platforms) too early**: the field moves fast enough that today's heavyweight build looks obsolete within a year; keep prompts with the applications and systems architecture light.
- **Framing AI use cases as "I want an agent for X"**: skips the business-problem screen; always re-anchor on the measurable problem the business sponsor cares about.
- **Prioritizing automation-only use cases as the ceiling of ambition**: the bar has risen past automation and acceleration to differentiation; low-ambition automation projects should not top the use-case list.
- **Anthropomorphizing agents into an HR-style roster** (reports-to org charts, retirement ceremonies for agents): useful technical identity/permissioning gets lost once you start thinking of software as staff.
- **Declaring victory at go-live without measuring adoption**: a shipped-but-unused AI system is still a failed project; adoption must be tracked post-implementation, not assumed.
- **Blaming the LLM for a bad outcome as a builder**: valid only for end users of consumer AI products; as an FDE/architect you are accountable for aligning the system to the business objective.
- **Using vanity metrics (e.g., raw token counts) to drive adoption "success"**: incentivizes the wrong behavior; prefer transparent, authentic usage/value reporting and organic champions.
- **Signing up for AWS via the "simplified"/social-login sign-up flow**: gives limited access and won't support AgentCore or this course's needs — always use AWS's advanced sign-up with the Paid plan.
- **Letting a cost budget include credits**: a budget that counts free credits shows $0 spend and never alerts while credits are being silently consumed — exclude Credit/Refund charge types explicitly.

## Code Examples
```python
session = boto3.Session()
identity = session.client("sts").get_caller_identity()

print(f"Account: {identity['Account']}")
print(f"Signed in as: {identity['Arn']}")
if identity["Arn"].endswith(":root"):
    print("These are the root user's keys. Create an access key for your IAM user in Step 5 and run aws configure again.")
```
- **What it demonstrates**: verifying from Python (via boto3) that you're operating as the narrow-permission IAM user, not the all-powerful root account — a basic cloud-hygiene check to run before doing anything else.

```python
def counts_credits(budget):
    """Whether a cost budget counts AWS credits against its limit."""
    excluded = budget.get("FilterExpression", {}).get("Not", {}).get("Dimensions", {})
    if excluded.get("Key") == "RECORD_TYPE":
        return "Credit" not in excluded["Values"]
    return budget.get("CostTypes", {}).get("IncludeCredit", True)
```
- **What it demonstrates**: programmatically auditing whether a budget alert will actually fire while free credits are in play — catching the classic "budget shows $0 spend because credits absorbed it" trap before it costs you (per the instructor's own near-miss story).

```lua
runtimes = session.client("bedrock-agentcore-control").list_agent_runtimes()
print(runtimes["agentRuntimes"])
```
- **What it demonstrates**: confirming the IAM policy already grants AgentCore access (an empty list is success — it means the call was authorized, even though nothing has been deployed yet; AgentCore itself is built starting Day 2).

## Reference Tables

| Hosting option | Examples | What you get |
|---|---|---|
| Your own infrastructure | AWS Lambda, ECS, Cloud Run, Kubernetes, Render | Any framework and model; you own state and scaling |
| A managed runtime | AWS AgentCore Runtime, Google Agent Runtime, Microsoft Foundry, LangSmith Deployment | Any framework and model; memory and identity built in |
| The model lab runs it | Claude Managed Agents, OpenAI Agents API | Only that lab's models; the lab's own harness |

| Governance level | Cadence | Covers | Who |
|---|---|---|---|
| Firmwide Leadership | Monthly/Quarterly | Portfolio, policy, risk tolerance | Executives across the business |
| Project committee | Monthly | Scope, risk, releases for one project | Sponsor, owner, security, audit, operations |
| Working group | Weekly | Build, evaluate, run the agent | Product, engineering, business, FDE |

| AWS setup step | Action |
|---|---|
| 1 | Open AWS account via advanced sign-up, Paid plan |
| 2 | Turn on MFA for root (passkey or authenticator app) |
| 3 | Choose us-east-1 and leave it there |
| 4 | Activate IAM billing access; create a $10/month budget excluding credits, with alerts at 50/80/100% |
| 5 | Create IAM policy (`undercover-course`), group, user (`undercover`), and CLI access key |
| 6 | Install AWS CLI v2, run `aws configure`, verify with `aws sts get-caller-identity` |

## Worked Example
The Day 1 lab (`labs/week3day1.ipynb`) sets up the AWS account that the rest of Week 3 builds on — no AgentCore work yet, purely account hygiene. Steps performed in the AWS console as root: open a Paid-plan account via advanced sign-up; enable MFA on root; set the console region to us-east-1; activate IAM billing access and create a $10/month budget with the Credit/Refund charge-type filter excluded and alert thresholds at 50/80/100%, all emailing the student; create an IAM policy from the repo's `undercover-course-policy.json` (scoped to exactly what the course needs: AgentCore, CloudFormation/CDK resources, Lambda functions and IAM roles named `undercover...`, encryption keys/secrets, logs, and read-only billing/plan access), a matching IAM group, and an IAM user named `undercover` with a CLI access key. Then, in a terminal: install the AWS CLI v2, run `aws configure` with the access key, `us-east-1`, and JSON output, declining the CLI's offer to set up AWS skills/MCP server mid-course.

Verification runs entirely in Python via boto3: confirm the caller identity is the IAM user (not root); confirm the region is us-east-1; check the account plan type and remaining credits (handling the case where older accounts have no `accountPlanType` at all); confirm root MFA is enabled via `get_account_summary()`; inspect each budget and flag any that still counts credits against its limit; and finally call `list_agent_runtimes()` on `bedrock-agentcore-control` to confirm the IAM policy already authorizes AgentCore calls, even though zero runtimes exist yet. Result: a fully verified, narrowly-scoped AWS account with monitored spend and no actual charges incurred — the groundwork for Day 2's Gateway and Policy work. A free alternative path (`week3day1free.ipynb`) sets up a Render account instead, for students who don't want AWS billing exposure; it documents Render's free-tier limits (15-minute idle spin-down, 750 free instance-hours/month, no persistent disk, no shell access, single instance, not for production) rather than running code.

## Key Takeaways
1. Always disambiguate "ADLC" — the agent-building lifecycle (this week) versus the Claude-Code-style agentic SDLC (Week 1 Day 3) — before using the term with anyone.
2. Screen every candidate use case against three questions (avoids Week 2 traps? uses proprietary data/connections/evals? automation vs. acceleration vs. differentiation?) before RICE-ranking it.
3. Default to hosting an agent on your own infrastructure for simplicity; reach for a managed runtime like AgentCore specifically when you need its built-in identity, memory, and permissioning — not merely for scaling.
4. Set up three-tier governance with named sponsors and phase tollgates at project inception, not reactively.
5. Make the autonomy level a deliberate, documented, risk-based decision per action — not a default setting — backed by review screens, audit trails, and delay/escalation plans.
6. Budget and assign an owner for all six responsible-AI pillars up front; don't let any one of them surface as a surprise blocker during Implementation.
7. Treat LLM-provider data sharing as ordinary third-party vendor risk management, not a special new category of fear.
8. Handover and adoption are deliverables in their own right — plan knowledge transfer, ongoing measurement, training and transparent usage reporting as explicitly as you plan the build.
9. On AWS, there is no hard spending ceiling — budgets only alert; actively monitor them, exclude credits from cost budgets, and keep IAM permissions scoped narrowly to what the project needs.

## Connects To
- **Week 3, Day 2**: This day explicitly defers all AgentCore, Gateway and Policy work to Day 2 — the AWS account configured here is the prerequisite for the security/guardrails and gateway deployment work that follows immediately.
- **Week 2**: The "traps from Week 2" use-case screening question and the evals/baseline/success-criteria discipline both build directly on Week 2's architecture and evaluation material.
- **AWS Bedrock AgentCore / IAM**: The IAM policy, user, and group model set up here is the identity foundation that AgentCore's Gateway and Runtime (Days 2 and 4) will rely on and extend.
