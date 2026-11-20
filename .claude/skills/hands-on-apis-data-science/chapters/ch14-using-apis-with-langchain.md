# Chapter 14: Using APIs with LangChain

## Core Idea
Build a LangGraph agent that first calls an LLM via API (Anthropic Claude), then is given a custom LangChain toolkit wrapping the Chapter 7 `swcpy` SDK — demonstrating both directions of the AI/API relationship: calling AI via an API, and letting AI call APIs as tools to fill its knowledge gap.

## Frameworks Introduced
- **LangGraph State-Graph Agent Pattern**: model agent control flow as a graph of nodes/edges with state passed between them.
  - When to use: building any agent whose behavior should loop (ask → reason → act → ask again) rather than run once straight through, and where more control than legacy LangChain "agent" abstractions is desired.
  - How: define a `StateGraph(MessagesState)`; add nodes (`workflow.add_node("agent", call_model)`, `workflow.add_node("tools", tool_node)`); wire edges including a **conditional edge** (`should_continue`) that routes to `"tools"` if the model's last message contains tool calls, or to `END` otherwise; close the loop with an edge from `"tools"` back to `"agent"`; compile with a `checkpointer` (e.g., `MemorySaver`) for conversational memory via `thread_id`.
- **Directed Cyclic Graph (vs. Airflow's Acyclic Graph)**: distinguishing orchestration topologies.
  - When to use: recognizing when a workflow needs loops (agent reasoning, possibly multiple tool calls) vs. strictly linear pipelines (Airflow's DAGs, Ch. 10).
  - How: LangGraph explicitly supports cycles — nodes and edges can repeat (e.g., agent ↔ tools) until a termination condition is met, unlike Airflow's one-direction, no-loop DAGs.
- **LangChain Toolkit Pattern (BaseTool/BaseToolkit subclassing)**: wrap existing API/SDK functions as LLM-callable tools.
  - When to use: giving an agent access to a specific, scoped set of API capabilities rather than raw unrestricted access.
  - How: for each capability, define a Pydantic input schema (subclass `BaseModel`, using `Optional`/`Field(description=...)` for parameters) and a tool class (subclass `BaseTool` with `name`, `description`, `args_schema`, and a `_run()` method invoking the underlying SDK call); group tools into a `BaseToolkit` subclass whose `get_tools()` returns instances of each tool; the tool's `description` and input field `description`s are what the model reads to decide *when* and *how* to call it.
- **Binding Tools to a Model + Conditional Routing**: make a model "function-calling" aware and let the graph branch on its decisions.
  - How: call `model.bind_tools(tools)` so the model can emit `tool_calls`; implement `should_continue(state)` to inspect `messages[-1].tool_calls` and return `"tools"` (continue the loop) or `END` (finish); register this function via `workflow.add_conditional_edges("agent", should_continue)`.

## Key Concepts
- **Agent** (per Harrison Chase, LangChain's creator): "a system that uses an LLM to decide the control flow of an application" — not preprogrammed; reasons and decides what happens next, including which tool calls to make.
- **Function-calling model**: a model type that can examine available tools/functions and *suggest* their use — the model itself never directly executes a tool call; the agent (surrounding code) performs the actual call based on the model's suggestion.
- **Node / edge** (graph theory, recap from Ch. 10): a node is a process that updates application state; an edge is a one-way flow between nodes; LangGraph's graphs can be cyclic (loop), unlike Airflow's DAGs.
- **Tool / toolkit**: a tool is code providing an extra "skill" to an agent (e.g., a function calling one API endpoint); a toolkit is a collection of related tools bundled together.
- **MessagesState / checkpointer / thread_id**: LangGraph's state object carrying the conversation's message history; a `checkpointer` (e.g., in-memory `MemorySaver`) persists that state across `invoke()` calls sharing the same `thread_id`, giving the agent conversational memory.
- **Temperature**: a model parameter controlling response randomness/creativity; `temperature=0` yields more deterministic (though never fully deterministic) outputs — used here for predictable agent behavior.
- **Knowledge gap (recap from Ch. 12)**: demonstrated concretely — the base Claude model confidently answers historical NFL trivia (training-data knowledge) but cannot answer anything about the fictional/private SportsWorldCentral platform until given a tool to query the live API.
- **swcpy SDK reuse**: the toolkit wraps the Chapter 7 SDK's client methods (`get_health_check()`, `list_leagues()`, `list_teams()`) rather than calling the raw API directly — chosen as "the most robust way to interact with your API," though simpler alternatives (the Ch. 9 client, or direct httpx calls) are explicitly noted as viable too.

## Mental Models
- **"A system is more agentic the more an LLM decides how the system can behave"** (Harrison Chase epigraph): agentic-ness is a spectrum, not binary — this chapter moves from a non-agentic simple chat loop to a genuinely agentic tool-using loop.
- **Two directions of AI/API integration, demonstrated back to back**: first "calling AI using APIs" (the notebook calls the Anthropic API to get chat completions), then "calling APIs using AI" (the agent decides to call the SWC API through tools) — the same underlying LLM, used in opposite integration roles.
- **The model suggests, the agent executes**: an important, easily-missed distinction — function-calling models never directly invoke anything; they emit a structured suggestion that surrounding orchestration code (the agent/tool node) chooses to act on.
- **Tool descriptions are the API documentation an LLM actually reads**: the `description` fields on `BaseTool` subclasses and their `args_schema` fields function exactly like OAS descriptions (Ch. 5/Ch. 12) — they are what determines whether and how correctly the model uses a tool.

## Anti-patterns
- **Assuming a general-purpose LLM "knows" about your private/internal platform**: the chapter explicitly demonstrates the model failing (and even doubting the platform's existence) on SWC-specific questions — any proprietary or recent data requires retrieval (RAG/tools), not reliance on training data.
- **Treating API keys carelessly**: the chapter advises starting with a small account balance (e.g., $5) and disabling auto-reload to cap potential misuse, and explicitly recommends rotating (deleting and recreating) a key if it may have been exposed.
- **Using high-risk LLM deployments without safeguards (recap)**: reiterates Anthropic's own warning that "models should not be used on their own in high-stakes situations" — this chapter's read-only fantasy football API is deliberately chosen as low-risk.
- **Giving tools vague names/descriptions**: since the model relies entirely on `name`/`description`/`args_schema` text to decide tool usage, poorly described tools will be used incorrectly or not at all — mirrors the "AI-readiness" design tips from Chapter 12.

## Code Examples
```python
# Basic (toolless) LangGraph agent
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```
- **What it demonstrates**: the minimal LangGraph agent shape — a single node calling the model, compiled with a checkpointer for conversational memory.

```python
# Toolkit tool definition (swc_toolkit.py)
class LeaguesInput(BaseModel):
    league_name: Optional[str] = Field(default=None,
        description="league name. Leave blank or None to get all leagues.")

class ListLeaguesTool(BaseTool):
    name: str = "ListLeagues"
    description: str = ("get a list of leagues from SportsWorldCentral. "
                         "Leagues contain teams if they are present.")
    args_schema: Type[LeaguesInput] = LeaguesInput
    return_direct: bool = False

    def _run(self, league_name: Optional[str] = None,
              run_manager=None) -> List[League]:
        return local_swc_client.list_leagues(league_name=league_name)

class SportsWorldCentralToolkit(BaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        return [HealthCheckTool(), ListLeaguesTool(), ListTeamsTool()]
```
- **What it demonstrates**: the Pydantic-input + BaseTool-subclass pattern for wrapping one SDK method as an LLM-usable tool, and bundling tools into a toolkit via `get_tools()`.

```python
# Agentic LangGraph loop with tool node and conditional edge
tool_node = ToolNode(tools)
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0).bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state['messages'][-1]
    return "tools" if last_message.tool_calls else END

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", 'agent')
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```
- **What it demonstrates**: upgrading the basic agent into a tool-using loop — `bind_tools()` makes the model function-calling-aware; `should_continue` implements the conditional edge; the `tools → agent` edge closes the loop so the agent can reason over tool results before answering.

```python
# Chatting with memory via thread_id
final_state = app.invoke(
    {"messages": [HumanMessage(content="What are the leagues in the SportsWorldCentral fantasy football platform?")]},
    config={"configurable": {"thread_id": 99}}
)
display(Markdown(final_state["messages"][-1].content))
```
- **What it demonstrates**: invoking the compiled graph with a conversation thread ID, and extracting/formatting the latest message; with tools bound, this call triggers a real `GET /v0/leagues/` call visible in the API's terminal log.

## Reference Tables
**Table 14-1. Tools used in this chapter**

| Software | Version | Purpose |
|---|---|---|
| LangChain | 0.3 | Create tools/toolkits for agents |
| LangGraph | 0.2 | Create the agent (state graph) |
| Anthropic Claude | 3.5 Sonnet | Reasoning model for the agent |
| Pydantic | 2 | Validation in the toolkit |
| swcpy | N/A | SDK (Ch. 7) wrapped by the toolkit |

**LangChain/LangGraph terminology**

| Term | Meaning |
|---|---|
| Agent | System using an LLM to decide control flow |
| Function-calling model | Model that suggests tool/function usage (doesn't call directly) |
| Model family | Multiple models sharing name/architecture |
| Toolkit | Collection of tools for an agent |
| Tool/function | Code giving an agent one extra capability |
| Node | A process updating application state |
| Edge | One-way flow between nodes (cycles allowed in LangGraph) |

**Toolless vs. tool-equipped agent behavior**

| Question | Toolless agent | Tool-equipped agent |
|---|---|---|
| "What teams did Joe Montana play for?" | Answers correctly (training data) | Same |
| "What are the leagues in SportsWorldCentral?" | Fails / doubts platform exists | Calls `ListLeagues` tool, answers correctly from live API data |

## Worked Example
After signing up for Anthropic, funding a capped ($5) account, creating and securely storing an API key (saved as a Codespaces secret `ANTHROPIC_API_KEY`), and installing LangChain/LangGraph/Anthropic libraries, the chapter builds a basic LangGraph agent in `langgraph_notebook.ipynb` using `ChatAnthropic` with `temperature=0`. Asked about Joe Montana's teams, it answers correctly from training data; asked about SportsWorldCentral's leagues, it fails, exposing the knowledge gap. The chapter then installs the `swcpy` SDK, builds `swc_toolkit.py` defining `HealthCheckTool`, `ListLeaguesTool`, and `ListTeamsTool` (each a `BaseTool` subclass with a Pydantic input schema, wrapping the corresponding `swcpy` client method), bundled into `SportsWorldCentralToolkit`. A copy of the notebook (`langgraph_notebook_with_toolkit.ipynb`) rebuilds the graph: `model.bind_tools(tools)`, a `tools` node (`ToolNode(tools)`), a `should_continue` conditional edge routing to `"tools"` or `END`, and a `tools → agent` loop-back edge. Re-asking the SportsWorldCentral leagues question now succeeds — the agent calls `GET /v0/leagues/?skip=0&limit=100` (visible in the API's terminal log) and returns the correct five league names, demonstrating the agent autonomously deciding to and successfully using a custom API tool.

## Key Takeaways
1. LangGraph models agent behavior as a graph of nodes/edges that can cycle (unlike Airflow's acyclic DAGs), enabling multi-step reasoning/tool-use loops.
2. Function-calling models suggest tool use; the surrounding agent code (a `ToolNode` plus conditional routing) is what actually executes the call.
3. Wrap existing SDK/API functions as LangChain tools via `BaseTool` + Pydantic input schemas; the tool's and parameters' `description` text is what the model relies on to use the tool correctly — treat it like API documentation written for an AI reader.
4. Bundle related tools into a `BaseToolkit` to give an agent a scoped, curated set of capabilities rather than unrestricted API access.
5. A base LLM's training-data knowledge does not include private/internal platform data — demonstrated directly by the model failing on SportsWorldCentral questions until given a live-API tool.
6. Use conversational memory (`checkpointer` + `thread_id`) so an agent's multi-turn interactions build on prior context.
7. Manage API key risk proactively: cap spending, disable auto-reload, and rotate keys immediately if exposure is suspected.

## Connects To
- **Ch 7**: Directly reuses the `swcpy` SDK as the implementation layer behind every LangChain tool in this chapter.
- **Ch 10**: Contrasts LangGraph's cyclic graphs with Apache Airflow's acyclic DAGs, both drawing on the same graph-theory vocabulary (nodes/edges).
- **Ch 12**: Implements the "agentic application" and "knowledge gap"/RAG concepts previewed there, and applies the AI-readiness tool-description guidance in practice.
- **Ch 15**: The next chapter builds the same kind of API-calling agent capability inside ChatGPT via a custom GPT and custom action, using the OAS file instead of a LangChain toolkit.
- **External**: LangChain "Chat Models" provider list; LangChain/LangGraph official documentation; Anthropic's "Building Effective Agents"; Google's Agents Whitepaper; Anthropic's "The Claude 3 Model Family" model card.
