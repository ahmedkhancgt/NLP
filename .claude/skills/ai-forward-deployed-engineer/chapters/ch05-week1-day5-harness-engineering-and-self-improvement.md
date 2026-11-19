# Week 1, Day 5: Harness Engineering, Self-Improvement and Hardening the UnderCover Harness

## Lessons Covered
- Lesson 36: Recap of Open-Weight vs Proprietary Models, Distillation and Token Cost
- Lesson 37: Harness Engineering, Context Engineering and Recursive Self-Improvement
- Lesson 38: Build or Buy an Agent Harness - LangChain, Deep Agents or Claude Agent SDK
- Lesson 39: The Claims Desk Lab - Configure the Claude Agent SDK with a Sub-Agent
- Lesson 40: Run the Claude Agent SDK, Then Deep Agents on DeepSeek V4.1 Flash
- Lesson 41: Choose Your Branch and Plan UnderCovers Harness with Claude Code
- Lesson 42: Build and Run UnderCovers Agent Harness with a Human Code Review
- Lesson 43: Refactor the Harness, Improve the Process and Complete Week 1

## Core Idea
Harness engineering — building the scaffolding (workflow, state, sub-agents, permissions, observability, self-improvement) around an LLM so it can take on bigger, higher-value tasks reliably and safely — is the current frontier skill beyond prompt engineering and context engineering; but "batteries-included" frameworks that get you there fast (Claude Agent SDK, LangChain Deep Agents) can still produce code a human reviewer must catch and correct, because looking impressive and working correctly are not the same thing.

## Frameworks Introduced
- **The Skill Evolution: Prompt Engineering → Context Engineering → Harness Engineering**: 2024's prompt engineering (how to word a message) gave way to 2025's context engineering (Phil Schmid's term — constructing the entire input sequence: system prompt, state/history, retrieved info, long-term memory, tools, user prompt, structured output) which has now given way to harness engineering (constructing all the scaffolding around one or more agents).
  - When to use: to place any current AI-engineering skill discussion on a timeline and know what it builds on.
- **Harness Engineering — Two Senses**: (1) as a *user* of an agent product (e.g. extending Claude Code's own harness with more MCP servers, hooks, AGENTS.md structure — Day 3's focus), vs. (2) as a *creator* of agents (building the scaffolding around your own agent, e.g. the UnderCover claims agent — this course's primary focus).
  - When to use: disambiguate immediately whenever "harness engineering" comes up, just as with the two meanings of ADLC.
- **The Seven Components of Harness Engineering**: context engineering (prompt/tools/memory) plus six more — workflow (the loop and its checks, e.g. automated `npm run check` after every turn), state (persistent repo-based memory across sessions — "the repo is memory"), sub-agents (delegating to fresh-context LLM calls), agent teams/swarms/fleets (collaborating peer agents, covered more in Week 2), permissions (controls plus sandboxing), observability (monitoring what the agent actually did), and self-improvement/recursive self-improvement.
  - When to use: as a checklist when assessing how mature or fragile any agent-based system's harness is.
- **Build vs. Configure Spectrum for Harness Frameworks**: a continuum from raw LLM calls (maximum build, minimum configure) through LangChain `create_agent` through LangChain Deep Agents through Claude Agent SDK (minimum build, maximum configure/reuse).
  - When to use: to choose how much engineering effort vs. off-the-shelf capability a project needs.
  - How: raw calls = full control, full boilerplate; `create_agent` = tool-calling/loop boilerplate handled, you still write all tools; Deep Agents = built-in to-do list and file-system tools plus sub-agent support, model-agnostic; Claude Agent SDK = the entire Claude Code agent driven programmatically, Anthropic-model-optimized, mostly final configuration via `ClaudeAgentOptions`.
- **Self-Improvement and Recursive Self-Improvement (RSI)**: self-improvement = an agent reviewing its last run, finding root causes of problems, and updating its own process documents (e.g. `improve.md`) so mistakes don't repeat; RSI = the agent also improving *how it improves* — "get better at getting better."
  - When to use: self-improvement whenever you want a harness that learns from its own mistakes over iterations; treat RSI as a largely theoretical/early-stage idea (no evidence yet of exponential takeoff), currently practiced only as incremental, not runaway, improvement.
  - How (self-improvement in practice): write a markdown recipe (e.g. `improve.md`) with steps — identify what went wrong, find root causes, design a fix strategy, add validation checkpoints — and invoke it at the end of each dev cycle via `PROCESS.md`.
- **Intent-Driven Development with HARNESS.md**: a spec document written from the business/IT perspective stating the *goal* (handle harder claims) and desired capabilities (one run that works the whole queue, a visible to-do list, a sub-agent per claim, manual lookup, automatic problem correction, use DeepSeek V4.1 Flash) without prescribing the technical solution.
  - When to use: as the Day-5 analog of Day 3's `SPEC.md` — stating business intent and letting the coding agent plan+build the implementation.

## Key Concepts
- **Context engineering components** (Phil Schmid's framing): system prompt/instructions, state/history/short-term memory, retrieved information (RAG), long-term memory, tools, user prompt, structured output.
- **"If you're not the model, you're the harness"** (Vivek Trivedy, LangChain): the clean dividing line between what the LLM contributes and everything else that makes an agent work.
- **"Human steer, agents execute"** (OpenAI framing): a proposed division of labor between human oversight and agent execution.
- **Mitchell Hashimoto's harness-engineering insight**: whenever an agent makes a mistake, invest in engineering the scaffolding so that exact mistake becomes structurally impossible in the future, rather than just re-prompting around it.
- **`ClaudeAgentOptions`**: the Claude Agent SDK's central configuration object — sets model, working directory, system prompt, tools/allowed-tools, sub-agents (via `AgentDefinition`), permission mode, and budget.
- **`AgentDefinition`**: how a sub-agent is declared for Claude Agent SDK — includes a description (tells the main agent when to use it), its own system prompt, and its own tool list.
- **`create_deep_agent`** (LangChain Deep Agents): the Deep Agents equivalent of `create_agent`, adding built-in to-do-list middleware and file-system tool access, plus native sub-agent configuration via a dictionary of agent definitions.
- **Over-prescriptive sub-agent instructions (anti-pattern, acknowledged live)**: writing a rigid, step-by-step script for what an agent should do is really just hard-coded workflow logic dressed up as agentic autonomy — a red flag the instructor deliberately leaves in the lab to illustrate, while defending it as "setting up for future autonomy needs."
- **"Pilot" code quality vs. "production" code quality**: code that was acceptable for a fast pilot (Day 3) can carry forward unacceptable shortcuts (hard-coded URLs, spaghetti module dependencies, oversized functions, mixed harness/domain concerns) into what's meant to become production-grade — and a strong model (Opus 5.5) will readily copy that pattern unless explicitly told to raise the bar.
- **`HUMAN_REVIEW_HARNESS.md`**: the artifact used here to formalize a human code review as a structured self-improvement input — list specific issues, demand root-cause investigation, require retesting, and require `PROCESS.md`/`CODE_STANDARDS.md`/`STATUS.md` updates so the same issues can't recur.
- **`/context` and `/compact` as working practice**: checking what fraction of context is used (`/context`) and proactively compacting before a big new instruction (e.g. before handing over a long review document) to keep the session efficient.

## Mental Models
- See the "prompt engineering → context engineering → harness engineering" sequence as each skill *subsuming* rather than replacing the last — harness engineering still requires good context engineering underneath it, plus six more dimensions.
- Treat "the repo is memory" literally: any state you want an agent (or a team) to carry across sessions belongs in a tracked file in the repo, not in an ephemeral chat session.
- When reviewing agent-generated code, assume impressive working functionality does NOT imply good code — the "wow, look at this feature working" reaction is precisely the moment accountability discipline (from Day 3) must kick back in.
- Use the "alarm bell" test for sub-agent prompt instructions: if you're writing a precise numbered sequence of steps for an agent to follow, ask whether that's genuinely a judgment call needing LLM autonomy, or whether it's actually a for-loop you should have written in code.
- Picture RSI on a spectrum from "learns from its own mistakes" (real, achievable, incremental) to "runaway self-amplifying intelligence explosion" (speculative, not currently observed) — don't let the scary framing obscure that today's practice is squarely in the first camp.

## Anti-patterns
- **Carrying pilot-grade shortcuts into a production build without raising the bar explicitly**: hard-coded backend URLs, duplicated request-building logic across modules, a single 50-line function mixing harness orchestration with business logic — all traced back to code copied forward from the Day 3 pilot without a renewed quality mandate.
- **Writing long, prescriptive step-by-step agent instructions**: this converts an "agent" into a thinly-disguised deterministic script; if the steps never vary, write the loop in Python instead and save the LLM calls for the parts that genuinely need judgment.
- **Trusting a single successful run as proof of correctness**: the Claude Agent SDK run scored 5/5 most times but 4/5 once across ~7 runs — a harness's reliability must be assessed across repeated trials, not a single lucky pass.
- **Being "bedazzled" by a working UI into skipping the code review**: a slick, functioning dashboard is not evidence of sound underlying code — the instructor explicitly flags this as the trap he nearly fell into himself.
- **Letting a strong model substitute for process discipline**: Opus 5.5 is a frontier-caliber model and still produced "spaghetti code" when the plan and process didn't explicitly demand higher standards — model strength does not replace harness/process rigor (callback to Day 3's "harness, hardening, humans").

## Code Examples
```python
# Claude Agent SDK: defining a sub-agent and options
claim_handler = AgentDefinition(
    description="Decides one claim and writes a decision file.",
    system_prompt=SUBAGENT_SYSTEM_PROMPT,
    tools=["Read", "Write", "Grep"],  # illustrative; actual tool names per SDK
)

options = ClaudeAgentOptions(
    model="claude-sonnet-5-5",
    cwd=str(desk_path),
    system_prompt=MAIN_SYSTEM_PROMPT,
    tools=[...],
    allowed_tools=[...],
    agents={"claim-handler": claim_handler},
    permission_mode="bypassPermissions",
    max_budget=...,
)

async for message in query(prompt="Please work the queue.", options=options):
    describe(message)
```
- **What it demonstrates**: the Claude Agent SDK pattern — nearly all the "harness engineering" work is configuring `ClaudeAgentOptions` and an `AgentDefinition` for the sub-agent; the agent loop itself is just `async for message in query(...)`.

```python
llm = ChatOpenAI(model="deepseek/deepseek-v4.1-flash", base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)

agent = create_deep_agent(
    model=llm,
    system_prompt=MAIN_SYSTEM_PROMPT,
    subagents={"claim-handler": {"description": "...", "system_prompt": SUBAGENT_SYSTEM_PROMPT}},
    middleware=[todo_list_middleware],
    backend=FileSystemBackend(root=desk_path),
)

for namespace, update in agent.stream({"messages": [{"role": "user", "content": "Please work the queue."}]}, max_messages=200):
    ...
```
- **What it demonstrates**: `create_deep_agent` mirrors the Claude Agent SDK's configuration-heavy approach but stays model-agnostic (here using DeepSeek V4.1 Flash via OpenRouter) and uses `agent.stream` instead of `query`.

## Reference Tables
| Harness framework | Build effort | Model flexibility | Built-in capabilities |
|---|---|---|---|
| Raw LLM calls (Day 1 style) | Maximum | Any model | None — you write tool JSON, loop, everything |
| LangChain `create_agent` | Moderate | Any model | Tool-calling/loop boilerplate handled; you supply tools |
| LangChain Deep Agents (`create_deep_agent`) | Low | Any model | To-do list, file-system tools, sub-agent config out of the box |
| Claude Agent SDK | Minimal (mostly config) | Optimized for Anthropic models (usable with others) | Full Claude Code agent, tools, permissions, sub-agents programmatically driven |

| Lab 1 (Claude Agent SDK, Sonnet 5.5) | Lab 2 (Deep Agents, DeepSeek V4.1 Flash) |
|---|---|
| 5/5 claims correct (typical run; once 4/5 across ~7 runs) | 5/5 claims correct |
| More expensive (paid Anthropic API) | ~4 cents total — much cheaper |
| Produces a handover.md with escalations and a notable qualitative insight (predicted complaint) | Produces a similar handover document with a results table and flagged items |

## Worked Example
Day 5 runs two parallel "Claims Desk" labs on the same task — work a small file-system-based queue of 5 claims autonomously, using a main agent with a to-do list that delegates each claim to a fresh sub-agent, with no custom tools written by the student (the framework's built-in file read/write/grep tools suffice). Lab 1 uses the Claude Agent SDK with Claude Sonnet 5.5: the main agent lists tasks, hands each claim to the `claim-handler` sub-agent, writes decision files, and finishes with a `handover.md` — scoring 5/5 against the held-out ground truth and surfacing a qualitative insight (one declined claim will likely trigger a customer complaint) that a purely scripted pipeline wouldn't have produced. Lab 2 repeats the exact same structure with LangChain's `create_deep_agent` and DeepSeek V4.1 Flash (the Day 4 cost/accuracy winner) via OpenRouter — also scoring 5/5, for roughly 4 cents total, demonstrating that the "batteries-included" harness pattern transfers across both an Anthropic-specific SDK and a model-agnostic framework.

The chapter's second half returns to the real UnderCover product: a `HARNESS.md` intent document (handle harder claims, one queue-wide run, visible to-do list, per-claim sub-agent, manual-lookup capability, automatic problem correction, use DeepSeek V4.1 Flash) is handed to Claude Code, which plans a six-phase `PLAN_HARNESS.md` and then — after 44 minutes — implements it. The resulting dashboard works impressively (to-do list visible live, claims processing correctly, a flagged large claim awaiting sign-off), but a close code review reveals real problems: hard-coded backend URLs duplicated across modules, a tangled import relationship between `main.py` and `agent.tools`, a 50-line `run_queue` function mixing harness orchestration with domain logic, and (separately, inherited from Day 3's unaddressed issue) overly prescriptive step-by-step sub-agent prompts that amount to hard-coded workflow dressed as agent autonomy. The instructor writes `HUMAN_REVIEW_HARNESS.md` listing these issues and demanding root-cause investigation, fixes, retesting, and updates to `PROCESS.md`/`CODE_STANDARDS.md`/`STATUS.md` so the problems can't recur — a concrete self-improvement cycle. One hour and twelve minutes later, Claude Code returns with a cleaner module split (`api`, `harness`, `tools`), updated coding standards with explicit "one way to do each thing" guidance, an expanded `PROCESS.md` review section, and a re-run that again correctly processes 5 claims — now with code quality matching the working functionality, confirmed by spot-checking light mode and a mobile viewport too.

## Key Takeaways
1. Harness engineering has succeeded context engineering as the current frontier skill, but it builds on (doesn't replace) good context engineering fundamentals.
2. Separate "harness engineering as an agent user" from "harness engineering as an agent creator" — this course focuses on the latter.
3. The seven components of a harness (context, workflow, state, sub-agents, agent teams, permissions, observability, self-improvement) form a checklist for assessing any agentic system's maturity.
4. Choosing a harness framework is a build-vs-configure spectrum decision, not a binary — pick based on how much custom control vs. out-of-the-box capability the task needs.
5. Self-improvement (learning from past mistakes via updated process docs) is practical and valuable today; recursive self-improvement (improving the improvement process itself) is a real but still largely theoretical/incremental idea, not an imminent runaway intelligence explosion.
6. Writing rigid, step-by-step sub-agent instructions is a red flag for disguised deterministic logic — prefer giving agents a goal and tools, reserving autonomy for decisions that genuinely need judgment.
7. A working, good-looking product is not evidence of good code — human code review remains mandatory even (especially) when the demo is impressive.
8. Pilot-stage code shortcuts will propagate into "production" builds unless you explicitly raise and communicate the quality bar when the project's purpose changes.
9. A structured human review document (naming specific issues, demanding root-cause fixes and process updates) is an effective, repeatable mechanism for driving genuine self-improvement in an agent-built codebase.

## Connects To
- **Week 1, Day 3**: directly extends the ADLC's "harness, hardening, humans" framework and the UnderCover pilot build process; the over-prescriptive sub-agent prompts issue traces back to a pattern left unaddressed since Day 3.
- **Week 1, Day 4**: DeepSeek V4.1 Flash, identified as the day's cost/accuracy winner, is explicitly carried forward as the model of choice for both the Deep Agents lab and the production harness build.
- **Week 2 (Agent Teams, Swarms, Fleets, Recursive Self-Improvement)**: this day explicitly flags agent teams/swarms/fleets and deeper RSI exploration as Week 2 topics, building on the brief introduction here.
- **LangChain / LangGraph / Claude Agent SDK**: this day is the first hands-on use of both LangChain's heavier Deep Agents framework and Anthropic's Claude Agent SDK, setting up Week 2's deeper architecture and orchestration coverage.
