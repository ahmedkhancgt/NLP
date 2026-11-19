# Week 1, Day 1: The FDE Roadmap and Your First Agent Harness

## Lessons Covered
- Lesson 1: Instant Gratification with OpenRouter and Free LLMs
- Lesson 2: An LLM Solves a Maze - Tool Calling and an Agent Loop from Scratch
- Lesson 3: The 3-Week Roadmap from AI Engineer to AI Architect to AI FDE
- Lesson 4: What Is a Forward Deployed Engineer From Palantir to OpenAI
- Lesson 5: The Skills, Tools and Experience You Need to Become an AI FDE
- Lesson 6: Environment Setup on Windows and Mac - VS Code, Git, Node and uv
- Lesson 7: Clone the UnderCover Repo, Run uv sync and Set Up Your .env File
- Lesson 8: Run the Maze Agent Locally in VS Code and Plan Your API Costs

## Core Idea
An AI agent is nothing mystical — it's `Model + Tool + Loop`: a plain Python `for` loop calling an LLM repeatedly, handing it a tool, and letting the model's own decisions (which tool call, when to stop) drive the behavior. Everything in the course's later enterprise-scale harnesses reduces to this same pattern.

## Frameworks Introduced
- **Agent Harness (first principles)**: the minimal shape of an agent — no framework, just an API client, a tool description in JSON, a `handle_tool_call` function, and a `for` loop up to `max_steps` that re-calls the model with the growing conversation history until it returns plain text with no tool call.
  - When to use: to build intuition before adopting any agent framework (LangChain, LangGraph, Claude Agent SDK) — understand the loop before you let a framework hide it.
  - How: (1) write a plain Python function as the "tool", (2) describe it to the model in OpenAI-style JSON schema, (3) call the model with `tools=[...]`, (4) if `finish_reason == tool_calls`, run the function and append a `{"role": "tool", ...}` message, (5) loop back to step 3 with full history until the model replies with text and no tool call.
- **FDE Role Framework — Three Disciplines + Two Experience Areas**: AI Engineering, AI Architecture, Forward Deployed Engineering (the three skill disciplines, one per course week) plus two kinds of hands-on experience: having solved real-world business problems with LLMs/agents, and having delivered directly to customers (internal or external).
  - When to use: as a checklist for what "FDE-ready" means — a resume/portfolio gap analysis.
  - How: acquire each discipline through the corresponding week, then backfill experience either through a real workplace project or a self-built project (like the course's UnderCover project) posted publicly with personal insight attached.
- **The FDE Definition — Narrow vs Broad**: Narrow (OpenAI's usage): a senior applied AI engineer employed by a frontier lab, embedded on-site with a client, responsible for getting AI solutions into production. Broad (industry usage): any applied AI engineer solving commercial problems by selecting, applying, and deploying LLMs and agents — effectively a modern name for "AI solutions architect."
  - When to use: when scoping a job description or a LinkedIn headline — know which sense is meant.

## Key Concepts
- **Forward Deployed Engineer (FDE)**: a customer-facing AI engineer; term coined ~2006-2007 by Palantir CTO Shyam Sankar, repurposed by OpenAI in 2026 for applied AI engineers embedded with clients.
- **Agent**: an LLM with tools, operating in a loop, pursuing a goal.
- **Tool calling**: the model requests a function call by name and arguments (JSON); your code executes it and returns the result as a new message — the model never runs code itself.
- **Harness**: the surrounding code/infrastructure (loop, tools, guardrails, retries) that turns a raw LLM into an agent.
- **Guardrail (informal)**: in this lab, the `read_web_page` function's check that the URL starts with `maze.edwarddonner.com` — a minimal example of constraining what a tool is allowed to do.
- **OpenRouter**: a unified API gateway to hundreds of LLMs from many providers, including free-tier models, accessed via the OpenAI Python client pointed at a different `base_url`.
- **`openrouter/free`**: a special OpenRouter model string that auto-selects any available free model for the request.
- **`.env` file**: the file (named exactly `.env`, in the project root) holding secrets as environment variables, loaded via `load_dotenv()`; must never be committed (git ignores it by convention).
- **uv**: Astral's Python package/environment manager (now owned by OpenAI) — `uv sync` builds the project's `.venv` and installs the pinned Python version automatically.
- **Kernel (notebook)**: the Python environment a Jupyter notebook executes against; must be explicitly set to the project's `.venv` or you get import/NameErrors.
- **ADLC**: the Agentic Development Lifecycle — introduced here as a label for Week 1's later coding-agent topics (defined fully on Day 3).
- **Context engineering / Harness engineering**: two terms flagged for deep coverage later in the week — managing what's in the prompt/context vs. engineering the full loop (sub-agents, permissions, observability, self-improvement).

## Mental Models
- Think of the three FDE disciplines as stacked, not parallel: AI Engineering (Week 1) is the foundation AI Architecture (Week 2) builds on, which Forward Deployed Engineering (Week 3) builds on again — you can't skip to FDE skills without the engineering fundamentals.
- Use the "theory vs. practice" lens on any role title: ask whether the person using the term "FDE" means the narrow lab-employee definition or the broad applied-AI-engineer definition — they hire and pay very differently.
- Treat "no agent framework required" as the default mental check: before reaching for LangChain/LangGraph/Claude Agent SDK, ask whether a hand-rolled loop (model + tool + loop) would already solve the problem — frameworks add power but also hide mechanics you should understand first.
- The maze is a stand-in for any agentic task: reading a "room" (retrieving state), deciding an action (choosing a tool call), and persisting history (conversation list) is the same shape as a claims-processing agent or a customer-support agent navigating a knowledge base.

## Anti-patterns
- **Treating `.env` casually**: naming it anything other than exactly `.env`, pasting a key through a word processor (which can silently convert hyphens to long-hyphens or inject Unicode), or forgetting to save the file (the "white blob" unsaved indicator) — all produce silent key failures or `NameError`/401 errors downstream.
- **Running notebook cells out of declaration order**: a `NameError` almost always means a prior cell wasn't executed yet, not that the code is broken — execution order (Shift-Enter order), not visual order, is what matters in a notebook.
- **Assuming a job title ("FDE") maps to one fixed definition**: conflating OpenAI's narrow definition with the broad industry usage leads to mismatched expectations about scope, seniority, and employer (lab vs. client-side).
- **Skipping real customer-facing experience**: acquiring only the three technical disciplines without a portfolio of solved business problems or direct customer/user partnership experience leaves the "FDE-ready" claim unsupported — nine in ten job ads in the instructor's survey require at least four of the five areas (engineering, architecture, FDE, solved real business problems, delivered to customers).

## Code Examples
```python
MAZE_URL = "https://maze.edwarddonner.com"

def read_web_page(url):
    """Read a web page in the maze and return the text on the page, without the HTML"""
    print(f"Reading web page: {url}")
    if not url.startswith(MAZE_URL):
        return f"Sorry, this tool can only read pages in the maze. Please ask for a URL that starts with {MAZE_URL}"
    response = requests.get(url, timeout=20)
    page = BeautifulSoup(response.text, "html.parser")
    return page.body.get_text(separator="\n", strip=True)
```
- **What it demonstrates**: a tool is just a plain Python function; the only "guardrail" is a string-prefix check before the function does anything.

```python
read_web_page_json = {
    "type": "function",
    "function": {
        "name": "read_web_page",
        "description": "Read a web page and return the text on the page",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL of the web page to request"},
            },
            "required": ["url"],
        },
    },
}
tools = [read_web_page_json]

def handle_tool_call(tool_call):
    """Run the tool that the model asked for and package the result as a message"""
    try:
        arguments = json.loads(tool_call.function.arguments)
        result = read_web_page(**arguments)
    except Exception as error:
        result = f"The tool call failed: {error}"
    return {"role": "tool", "tool_call_id": tool_call.id, "content": result}
```
- **What it demonstrates**: the model only ever sees a JSON description of the tool, never the implementation; results (including errors) are fed back as `role: "tool"` messages.

```python
def run_agent(task, max_steps=25):
    """Call the model in a loop, running the tool whenever it asks, until it gives a final answer"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for step in range(1, max_steps + 1):
        response = openrouter.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        message = response.choices[0].message
        print(f"Step {step} answered by {response.model}")

        text = (message.content or "").strip()
        if text:
            display(Markdown(text))

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                messages.append(handle_tool_call(tool_call))
        elif text:
            return
        else:
            print("The reply was empty, so we ask again")

    print("The agent ran out of steps")
```
- **What it demonstrates**: the complete agent loop — model call, branch on tool-call vs. final text vs. empty reply, growing message history, hard step cap as a safety limit.

## Reference Tables
| API cost tier | Spend | What you get |
|---|---|---|
| Free route | $0 | Free models throughout; more patience needed with slower/less reliable free models |
| Typical experience | ~$10 | Comfortable use of frontier models for the whole course |
| Full experience | $20+ | Production infra, larger test data sets, heavier experimentation (e.g. AWS deployments) |

| Job-ad survey finding (316 ads, 184 employers) | Share |
|---|---|
| Mentions AI Engineering | 88% |
| Mentions AI Architecture | 79% |
| Mentions Forward Deployed Engineering | 95% |
| Mentions solved real-world business problems | 88% |
| Mentions delivered to customers | 78% |
| Mentions all five areas | 41% |
| Mentions at least four areas | 48% (cumulative ~89%) |

## Worked Example
The Day 1 lab is run twice — once in Google Colab ("instant gratification," zero local setup) and once locally in VS Code against the cloned `undercover` repo — with identical code both times. Steps: (1) create an OpenRouter account, enable the two privacy switches for free endpoints, and generate an API key; (2) store the key as `OPENROUTER_API_KEY` in Colab secrets (cloud run) or in `.env` in the project root (local run); (3) make a trivial "Hi there" call to `openrouter/free`, confirming connectivity (answered by models like NVIDIA Nemotron 3.5 Lightning Free); (4) define `read_web_page` with the maze-URL guardrail and verify it both succeeds (`maze.edwarddonner.com`) and refuses (any other domain); (5) wrap it as a JSON tool description, make one model call, observe `finish_reason == "tool_calls"`, execute the tool, and feed the result back for a one-shot page summary; (6) finally wrap steps 4-5 in `run_agent`'s loop and launch it on the real task: "Have a look at https://maze.edwarddonner.com and do what it says." Observed result: the agent (serviced by a rotating cast of free models — Nemotron, InclusionAI, Cohere models, etc., since `openrouter/free` is non-deterministic about provider) explores rooms, hits dead ends, backtracks, and eventually retrieves the exit message: "Welcome to the start of the course. Prepare for three intensive weeks ahead." A typical run takes 10-15 tool calls, well under the free-tier's 50 requests/day and the loop's `max_steps=25` cap.

## Key Takeaways
1. An agent is just `Model + Tool + Loop` — understand this from-scratch version before trusting any framework's abstraction of it.
2. Tool calling is a protocol, not magic: the model emits structured JSON requesting a function; your code is solely responsible for executing it and returning a result message.
3. A minimal guardrail can be as simple as a string-prefix check inside the tool function itself — constrain the tool, not just the prompt.
4. Treat `.env` setup with real care: exact filename, no accidental character substitution from pasting through other apps, and always save before running cells.
5. "FDE" has two live meanings — a narrow lab-employee definition and a broad applied-AI-engineer definition — and almost all real job postings assume the broad one.
6. Becoming FDE-ready requires three technical disciplines (engineering, architecture, forward-deployed engineering) plus two experience tracks (solved business problems, customer delivery) — technical skill alone is an incomplete portfolio.
7. There is always a $0 path through this course via free models — spend is optional, not required, for learning the material.
8. Notebook execution order (not visual cell order) determines correctness; a `NameError` is almost always a sequencing mistake.

## Connects To
- **Week 1, Day 2**: Day 2 formalizes the mechanics glimpsed here (tokens, context windows, the six GPT→ChatGPT tricks) and introduces the UnderCover claims-processing project that this hand-rolled loop foreshadows.
- **Claude Agent SDK / LangChain `create_agent`**: the frameworks covered later in Week 1 and Week 2 are thicker wrappers around exactly this loop-plus-tool-calling pattern.
- **OpenAI's Chat Completions API / tool-calling spec**: the JSON tool-description format used here (`type: function`, `parameters` as JSON Schema) is the same shape used across OpenAI-compatible APIs, including OpenRouter.
