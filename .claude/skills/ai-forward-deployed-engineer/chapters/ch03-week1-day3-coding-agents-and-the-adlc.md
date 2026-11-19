# Week 1, Day 3: Coding Agents, the ADLC and Building the UnderCover Pilot

## Lessons Covered
- Lesson 18: Using a Coding Agent like Claude Code to Build an Agent
- Lesson 19: Vibe Coding, the Coding Agent Landscape and the Rise of Claude Code
- Lesson 20: Choose and Install a Coding Agent - Claude Code, Codex or OpenCode
- Lesson 21: Be the Boss of Your Coding Agent - AGENTS.md and Permission Modes
- Lesson 22: Claude Code Working Practices, MCP and the Playwright MCP Server
- Lesson 23: The Two Meanings of ADLC and a Tour of the UnderCover Project Brief
- Lesson 24: Hardening the Repo - Linters, Type Checks, Tests and a Pre-Commit Hook
- Lesson 25: Claude Code Plans the UnderCover Pilot in Phases with Success Criteria
- Lesson 26: Testing the Pilot Claude Code Built and Running the Agent on 25 Claims

## Core Idea
A coding agent (Claude Code or equivalent) can build an entire working product — backend, agent, dashboard, tests — from a short set of markdown project-brief files in about half an hour, but only if you've "hardened" the repo with automated controls (linters, type checks, tests, pre-commit hooks) and you personally remain "the boss": accountable, demanding evidence, and reviewing what matters rather than reading every line.

## Frameworks Introduced
- **The Three H's of the ADLC (Harness, Hardening, Humans)**: a mapping of the three dimensions of change (Platform, Process, People) onto the agentic coding world.
  - When to use: as a structure for assessing how mature your own coding-agent setup is — ask separately "is my harness good?", "is my hardening good?", and "do I have the human accountability model right?"
  - How: Harness = the coding agent platform itself (state of the repo, skills/integrations, sub-agents, agent teams). Hardening = automated controls baked into the repo (linters, type checks, test coverage thresholds, pre-commit hooks) that constrain what the agent can get away with. Humans = a named accountable person on every PR, who sets evaluation criteria, exercises judgment, and applies taste.
- **Two Meanings of ADLC**: (1) Agentic Development Lifecycle — the process for building AI *agents* themselves (risk/controls assessment, evals, adoption, observability, retirement) — an emerging standard akin to the SDLC but for agents, covered mainly in Weeks 2–3; (2) AI/Agentic-Driven Development Lifecycle — using coding agents like Claude Code to build *software* via a lifecycle, which is Day 3's actual subject and what AWS's and GitHub Spec Kit's published lifecycles address.
  - When to use: clarify which meaning is intended whenever "ADLC" comes up — the term is ambiguous in industry usage, and the second meaning (building code with agents) is currently more common.
- **Spec-Driven Development / Spec-Driven Intent (SDD)**: driving a coding agent's build from a markdown specification document (`SPEC.md`) written in pure business language, supplemented iteratively rather than treated as a complete, final contract.
  - When to use: at the start of any coding-agent build — write the business requirements and success criteria before asking the agent to build anything.
  - How: write `SPEC.md` (business requirements from the "sponsor"), `BRAND.md` (style), `TECH_STACK.md` (high-level technical choices only — framework names, not implementation detail), `PROCESS.md` (the build process and hard rules, e.g. "never bypass pre-commit checks"), `CODE_STANDARDS.md`, `STATUS.md` (agent-maintained running status); let `AGENTS.md` be the short entry point that points to all of these via progressive disclosure.
- **"Be the Boss" Accountability Model**: treat the coding agent like a very fast junior engineer reporting to you; your job is not to re-review every line but to ask hard questions, demand evidence, and own the outcome.
  - When to use: any time you're tempted to rubber-stamp fast, confident-looking agent output.
  - How: dig into sensitive/high-risk code specifically; ask the agent to justify decisions and name the tradeoffs of alternatives it didn't choose; never say "that wasn't me, that was Claude."
- **Progressive Disclosure (for AGENTS.md and documentation hierarchies)**: keep the root file (`AGENTS.md`) short, and have it point to more detailed files only loaded when relevant, rather than dumping everything into context every time.
  - When to use: whenever your project's documentation would otherwise bloat the agent's context on every single turn.
  - How: `AGENTS.md` names only the files to always read in full plus the folder structure; deeper technical/style docs are referenced by path and loaded on demand by the agent's own exploration.

## Key Concepts
- **Vibe coding**: Andrej Karpathy's February 2025 term for coding by "fully giving in to the vibes" and trusting the agent, without close inspection of the generated code.
- **Coding agent vs. coding agent harness**: the coding agent is harness + model together (e.g. Claude Code's software plus the Claude Opus 5.5 model it calls); the harness alone is just the local software managing tools, permissions and sandboxing.
- **AGENTS.md**: the markdown file all major coding agents (Claude Code, Codex, OpenCode) now read into context at session start and after every context compaction; historically Claude Code used `CLAUDE.md` but now falls back to `AGENTS.md`.
- **Permission modes**: Manual (ask before every file edit/command — now rare, causes "decision fatigue"), Accept Edits (auto-approve file edits only), Plan Mode (now considered unnecessary since models reliably follow a "just plan" instruction without a dedicated mode), Auto Mode (current default — a built-in classifier decides what's safe to auto-approve vs. escalate), Bypass Permissions / "YOLO mode" (`--dangerously-skip-permissions`, mostly used only inside sandboxed environments now).
- **MCP (Model Context Protocol)**: a standard way to plug someone else's tools into your AI application ("the USB-C port for AI applications"); its success comes from near-universal adoption, not technical novelty.
- **Skills**: a simpler, complementary alternative to MCP specifically for coding agents — a markdown file with a lightweight header the agent always reads, and full body content loaded only when the task needs it (progressive disclosure applied to capabilities).
- **Playwright MCP server**: Microsoft's browser-automation library, exposed via MCP so a coding agent can drive a real Chrome browser to test the product it just built — installed with `claude mcp add playwright -- npx -y @playwright/mcp@latest`.
- **Hardening controls**: ruff (Python linting), Prettier/ESLint (TypeScript linting), pyright (Python type checking), tsc strict mode (TypeScript type checking), unit tests with an 80% coverage threshold, Playwright end-to-end tests, and lefthook (a pre-commit hook runner) that blocks commits unless checks pass.
- **Cyclomatic complexity**: a code-smell metric (too-complex functions) enforced by linters; coding agents, unlike senior humans, don't push back when told to simplify — making hard-and-fast complexity rules more practical to enforce.
- **The repo as memory**: the practice of keeping a living catalog of markdown docs (plan, status, process, standards) checked into the repo so that both the coding agent and human teammates share the same persistent context across sessions.
- **UnderCover project brief**: the seeded skeleton repo structure — yellow files (`AGENTS.md` + the six docs: `SPEC.md`, `BRAND.md`, `TECH_STACK.md`, `PROCESS.md`, `STATUS.md`, `CODE_STANDARDS.md`) describing what to build; green folders (`backend`, `agent`, `dashboard`) as an empty skeleton; orange folders (`data`, `labs`) as prior experimentation; blue files as tooling config.

## Mental Models
- Treat the coding agent relationship like a non-technical-but-sharp manager overseeing a fast junior developer: ask for examples, ask "under what circumstances would you have made a different decision," get a second opinion on sensitive parts — don't try to become a superhuman line-by-line reviewer.
- Think "harness, hardening, humans" whenever evaluating any agentic software-delivery pipeline — if any of the three is weak, the whole chain is unreliable regardless of how good the model is.
- Use branch-based checkpoints as your safety net for agent-driven builds: create a dedicated branch before letting the agent build, and rely on git history (not the harness's `/rewind`) to roll back if things go wrong.
- See the plan-with-phases-and-success-criteria pattern as the controllable unit of agent work: never let a coding agent "just build it" — always have it produce a reviewable, checkable plan first, even without a dedicated "plan mode."
- Free/cheap models in a coding-agent harness demand a different working style entirely: tiny incremental steps, frequent `npm run check`, frequent commits, and willingness to throw away broken progress — patience and tenacity substitute for model strength.

## Anti-patterns
- **Rubber-stamping agent output because it "tested okay"**: the single biggest trap identified this day — don't let agent confidence substitute for your own accountability.
- **Writing AGENTS.md as an everything-dump**: stuffing all coding standards, process detail, and context into one giant file bloats every single context load; use progressive disclosure instead.
- **Over-specifying "don't do X" instructions**: negative instructions in markdown project files tend to confuse models more than positive "do Y" instructions — prefer stating desired behavior.
- **Relying on Plan Mode as a safety mechanism**: modern frontier models reliably honor a plain "please plan the pilot, don't write code yet" instruction, so a dedicated mode is now largely theater.
- **Trusting pre-commit hooks as fully foolproof**: an agent could, in principle, bypass hooks with an explicit flag; mitigate by explicitly forbidding this in `PROCESS.md`/`CODE_STANDARDS.md`, but understand it's a strong deterrent, not an absolute guarantee (hooks-based enforcement would be stronger).
- **Fighting free models out of stubbornness**: expecting free/rate-limited models to behave like frontier paid models leads to frustration; either budget for a cheap paid model (e.g. DeepSeek Flash) or accept the slow, incremental, patience-heavy workflow free models demand.

## Code Examples
No exact code snippets are reproduced in this day's source — the lab is markdown-driven (project brief files) and terminal-driven (installing tools, running `uv sync`, `npm install`, `npm run check`, `npm run dev`, `git checkout -b`, and interacting with Claude Code conversationally). The closest artifacts to "code" are the terminal commands and the structure of the markdown project-brief files themselves:

```txt
# Create a branch for your build
git checkout -b week1day3-yourname

# Install Node packages (dashboard, lefthook, Playwright's Chromium)
npm install

# Add the Playwright MCP server so the agent can test its own UI
claude mcp add playwright -- npx -y @playwright/mcp@latest
claude mcp list

# Run the full hardening suite: lint, type-check, format, unit tests, e2e tests
npm run check

# Inside Claude Code: set model and reasoning effort, then plan before building
/model
/effort
# "Please plan the pilot."
# (review docs/PLAN.md)
# "Please implement the plan for the pilot."
```
- **What it demonstrates**: the full day-3 workflow — branch isolation, tool installation, MCP wiring, hardening verification, and the plan-then-implement discipline used with the coding agent.

## Reference Tables
| Permission mode | Behavior | Current status |
|---|---|---|
| Manual | Asks before every edit/command/network call | Rare now — causes decision fatigue |
| Accept Edits | Auto-approves file edits only | Occasionally used |
| Plan Mode | Forces plan-only output, needs explicit approval to proceed | Considered obsolete — models honor plain planning instructions |
| Auto (default) | Built-in classifier auto-approves safe actions, escalates risky ones | Current default, considered safer than Manual due to less decision fatigue |
| Bypass Permissions ("YOLO") | Everything auto-approved, started with `--dangerously-skip-permissions` | Rare; mainly used only in sandboxed environments |

| UnderCover repo file | Author role | Content |
|---|---|---|
| `AGENTS.md` | Entry point | Course context, "start here," folder structure, pointers (progressive disclosure root) |
| `SPEC.md` | Business sponsor | Business requirements, in-scope/out-of-scope, written in non-technical language |
| `BRAND.md` | Marketing/style | Fonts, colors, visual style |
| `TECH_STACK.md` | Technical lead (instructor) | High-level stack choices only (FastAPI + SQLite backend, LangChain agent, React/Vite/TypeScript dashboard, Playwright for e2e, Python 3.12 + uv, Node 26) |
| `PROCESS.md` | Technical lead | Build process, mandatory checks, rule against bypassing hooks |
| `CODE_STANDARDS.md` | Technical lead / team | Keep it simple, minimal comments, no emojis, avoid overly defensive code, always `uv run`, TypeScript strict mode |
| `STATUS.md` | Agent-maintained | Running log of what's built and how to run it |

## Worked Example
The Day 3 lab has the instructor create a new git branch (`week1day3-ed`), start Claude Code on model Opus 5.5 at extra-high reasoning effort, and issue just two instructions. First: "Please plan the pilot" — Claude Code reads `AGENTS.md` and its linked docs, then produces `docs/PLAN.md`: a six-phase plan (claim workflow, the agent, dashboard queue/claim view, agent-at-work/sign-off/instructions view, working the queue, explicit out-of-scope items) each with concrete success criteria as checkboxes, plus a few flagged design decisions (e.g., deliberately excluding customer PII like email/phone/DOB from what's sent to the deciding model) for the human to approve or challenge. The instructor reviews and accepts the plan (no changes needed on this run). Second instruction: "Please implement the plan for the pilot." Claude Code built the complete three-tier product — FastAPI/SQLite backend, LangChain claims agent, React/Vite/TypeScript dashboard — in 32 minutes and 13 seconds, with all six phases' success criteria checked off, 45 passing unit tests at adequate coverage, and 5 passing Playwright end-to-end tests, and committed the work (which required passing the lefthook pre-commit checks, since bypass is disallowed by `PROCESS.md`). Running `uv run -m backend.seed` reset the 25-claim queue to "undecided," then `uv run -m agent` kicked off the LangChain agent against the live backend API: the dashboard updated in real time as claims were approved, referred for sign-off, or (in the "child left on the plane" and "giraffe ate my passport" Easter-egg cases) correctly declined or approved per the seeded ground truth — ending at 22 decided, 3 awaiting sign-off, 0 waiting/in-progress. The dashboard's "Agent at Work" view exposed a full observability trace of each decision, policy lookup, and rationale, demonstrating that a from-spec pilot built entirely by a coding agent in half an hour can already be demo-ready.

## Key Takeaways
1. A coding agent is harness + model; distinguishing the two clarifies which vendor/product owns which part of the "coding agent" you're using.
2. ADLC has two live meanings in industry — building agents (process-and-governance sense) vs. building software with agents (Day 3's sense) — always disambiguate.
3. Auto Mode is now the sensible permission-mode default: it reduces decision fatigue versus Manual mode while still escalating genuinely risky actions.
4. AGENTS.md should be short and use progressive disclosure — point to detail rather than inlining it, so every context load stays lean.
5. Spec-driven development (business-language `SPEC.md` plus supporting docs) lets even non-technical stakeholders drive what gets built, with the technical shape (`TECH_STACK.md`) kept separate and minimal.
6. Hardening (linters, type checks, test coverage thresholds, pre-commit hooks via lefthook) is what makes agent-built code trustworthy enough to commit without full line-by-line review.
7. Always have the agent produce a phased plan with explicit success criteria before implementation — review and challenge it like a manager, not a rubber stamp.
8. "Be the boss" means selective, evidence-demanding review, not exhaustive line reading — and it means never disclaiming accountability for what an agent produced under your name.
9. MCP's value is adoption, not technical novelty — use it as the default way to bolt on externally-written tools (e.g. Playwright for agent-driven UI testing) rather than reinventing tool descriptions yourself.

## Connects To
- **Week 1, Day 2**: the hand-built LangChain claims agent from Day 2's notebook becomes one of the three components (`agent/`) the coding agent is asked to build into the full UnderCover pilot product.
- **Week 1, Day 4-5**: harness engineering (Day 5) revisits the "harness" concept introduced here at a deeper, more formal level; open-source/local models (Day 4) will later be tested as alternative coding-agent backends for cost-sensitive builds.
- **Claude Agent SDK**: mentioned as the more code-driven, Anthropic-only alternative to using Claude Code interactively — covered properly at the end of Week 1.
- **AWS AI-Driven Development Lifecycle / GitHub Spec Kit**: external, named industry standards for the "building software with agents" meaning of ADLC, referenced as points of comparison.
