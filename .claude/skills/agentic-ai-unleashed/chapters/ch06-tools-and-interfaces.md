# Chapter 6: Tools and Interfaces

## Core Idea
Agents escape the limits of their frozen training data through **function calling** (raw capability, manual orchestration) and **tool calling** (framework abstraction, automatic orchestration); production-grade systems separate concerns further by hosting **tools behind MCP servers** (reusable, independently deployable) and **agent intelligence behind ACP servers** (orchestration, intent understanding) — "the future of AI systems lies not in monolithic architectures but in modular, protocol-based designs."

## Frameworks Introduced

### Function Calling vs. Tool Calling
- **Function calling**: the model requests execution of a function per a schema you defined; *you* execute it, feed results back, and manage the whole conversation loop. Low-level, maximum control.
- **Tool calling**: a framework wraps function calling with auto-generated schemas (from decorators), automatic execution, and end-to-end conversation management. High-level, rapid development.
- When to use which: function calling for custom conversation flows, building your own agent framework, complex multi-step orchestration, integrating legacy systems needing custom handling, or human-in-the-loop gating before continuing. Tool calling for rapid prototyping and standard conversational/function-to-result workflows — "most developers should use tools unless they need the extra control."
- Analogy: function calling is assembly language; tool calling is Python — a convenience layer over the same raw capability.

### The ACP + MCP Separation of Concerns
- **MCP servers** host **tools** (data/API access) — independently developed, tested, versioned, and discoverable at runtime; reusable across any MCP-compatible agent.
- **ACP servers** host **agent intelligence** — orchestration, intent parsing, multi-step workflows, and result synthesis that *wields* the MCP tools.
- Why it works (four compounding benefits): **rapid development** (tool teams and agent teams work in parallel without blocking each other), **reliability through isolation** (a tool failure or an agent crash doesn't take down the other side), **scalability through standardization** (new tools/agents plug in without modifying existing ones — a network effect), **intelligence through abstraction** (each layer focuses purely on what it does best).
- How: an MCP server exposes tools via `@server.list_tools()` (discovery/schema) and `@server.call_tool()` (execution); an ACP server builds a LangGraph-style multi-node workflow (e.g., intent understanding → tool invocation → analysis/response) that calls into one or more MCP servers via `ClientSession`, with no hard-coded dependency on any specific server implementation.

### Agent Specialization Through Tool Selection
Give each agent a *focused* toolset rather than one agent all possible tools — a weather agent with only `get_weather` responds fast and accurately without distraction; a web agent with search+fetch tools (`DuckDuckGoSearchTool` + `VisitWebpageTool`) can research arbitrary topics. Tool *combination* creates capabilities neither tool provides alone (search discovers sources, fetch extracts their content — the general "search and retrieve" pattern underlying most real-world research tasks).

### Three-Node Intent→Retrieve→Respond Workflow (Census Data Agent)
A LangGraph pattern for turning natural-language questions into tool calls and back into natural answers:
1. **Intent understanding** (LLM): parse the query into structured parameters (demographic_type, geographic_level, analysis_type, geographic_filter, metric_requested) via a JSON-schema prompt with worked examples.
2. **Data retrieval**: dynamically build and execute the MCP tool call from the parsed intent — no hard-coded query logic.
3. **Analysis and response**: post-process retrieved data contextually (percentages/rankings for comparisons, currency formatting for income, absolute+comparative framing for population) into a direct answer.
- Why it works: replacing keyword matching with LLM-based intent parsing handles natural ambiguity ("Which state has the most Black residents?" → structured `{demographic_type: race, analysis_type: highest, specific_demographic: Black or African American}`) that a rules engine would miss.

## Key Concepts
- **Tool (ACP/MCP sense)**: a Python function (custom, via `@tool` decorator) or off-the-shelf capability (e.g., `smolagents`' `DuckDuckGoSearchTool`) that extends an agent beyond its base LLM — the docstring *is* the tool's description to the agent, and type hints convey expected inputs/outputs.
- **CodeAgent** (smolagents): a reasoning agent that analyzes requests, decides which tools to use, chains multiple tool calls, generates code for computational sub-tasks, and synthesizes multi-source results.
- **Dynamic tool discovery**: MCP clients query `list_tools()` at runtime rather than hard-coding which tools exist — enabling an agent to work with *any* MCP server exposing a compatible tool, not just the one it was written against.
- **Dual-format tool output**: returning both a human-readable summary *and* machine-parseable JSON from a tool call lets an agent show immediate insight to users while retaining full data fidelity for further processing.
- **ACP (Agent Communication Protocol)**: standardizes building/deploying agents that use built-in + custom tools and communicate with external systems — distinct from A2A (peer-to-peer agent coordination/negotiation) in that ACP's focus is the agent-hosting/tool-use layer, while A2A is agent-to-agent task delegation.
- **BeeAI**: open-source, Linux Foundation–backed platform providing a unified hub (CLI/UI/SDK) to discover, deploy, test, and orchestrate ACP-compatible agents regardless of underlying framework.

## Mental Models
- Before adding a tool to an agent, ask "is this single-responsibility?" — the weather tool example does exactly one thing (fetch+transform weather for a city) with robust error handling and strong typing; resist bundling unrelated capabilities into one tool.
- When an agent's answers feel unreliable or slow, check whether its toolset is over-broad — specialization (narrow, focused toolsets) is both an accuracy and a latency lever, not just an architecture preference.
- Treat MCP/ACP separation as a parallel-development unblocking mechanism: if tool changes keep requiring agent-code changes (or vice versa), the separation of concerns has leaked and needs re-drawing.

## Anti-patterns
- **Embedding tools directly inside agent code** ("monolithic, difficult-to-maintain codebases") instead of hosting them behind MCP — loses reusability, independent testing/versioning, and runtime discoverability.
- **Hard-coding query-to-tool-parameter translation in the client** (as the chapter's first Census example does, manually mapping variable codes and state FIPS codes) instead of letting an LLM-based intent-understanding node do it — brittle and doesn't generalize to new natural-language phrasings.
- **Returning raw API responses from a tool** instead of transforming them into a consistent, analysis-ready format (e.g., a DataFrame) with graceful error handling for malformed responses or missing API keys.
- **Giving one agent every available tool** "just in case" — defeats the specialization benefits (focus, speed, accuracy) that motivated splitting into multiple agents in the first place.

## Code Examples

```python
# Custom tool creation via the @tool decorator (smolagents)
@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a given city using Open-Meteo API.
    Args:
        city: The name of the city to get weather for
    Returns:
        Weather information as a string
    """
    try:
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_data = requests.get(geocoding_url).json()
        if not geo_data.get('results'):
            return f"Could not find coordinates for {city}"
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        return requests.get(weather_url).json()['current_weather']
    except Exception as e:
        return f"Error getting weather for {city}: {str(e)}"
```
- **What it demonstrates**: the docstring-as-tool-description pattern, chained API calls (geocode → forecast), and graceful error handling — the three hallmarks of production-ready custom tools.

```python
# ACP agent registration: a focused weather agent and a general-purpose web agent
server = Server()

@server.agent()
async def weather_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a CodeAgent which answers questions about weather based on open-meteo."
    agent = CodeAgent(tools=[get_weather], model=model)
    response = agent.run(input[0].parts[0].content)
    yield Message(parts=[MessagePart(content=str(response))])

@server.agent()
async def web_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a CodeAgent which answers questions based on web resources."
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)
    response = agent.run(input[0].parts[0].content)
    yield Message(parts=[MessagePart(content=str(response))])
```
- **What it demonstrates**: agent specialization through selective tool provisioning — two agents, same framework, different focused capabilities.

```python
# MCP server: tool discovery and execution decorators
@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    return [types.Tool(
        name="us_census_tool",
        description="Access US Census Bureau data and return as DataFrame",
        inputSchema={"type": "object", "properties": {
            "dataset": {"type": "string", "enum": list(DATASETS.keys())},
            "year": {"type": "integer", "minimum": 2000, "maximum": 2030},
            "variables": {"type": "array", "items": {"type": "string"}},
            "geography": {"type": "string", "enum": list(GEOGRAPHY_TYPES.keys())},
            "geography_filter": {"type": "object", "additionalProperties": {"type": "string"}}
        }, "required": ["dataset", "year", "variables", "geography"]}
    )]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    if name != "us_census_tool":
        raise ValueError(f"Unknown tool: {name}")
    df = await fetch_census_data(**arguments)
    # ... format as human-readable summary + machine-parseable JSON ...
    return [types.TextContent(type="text", text=result)]
```
- **What it demonstrates**: the `list_tools`/`call_tool` decorator pair that implements MCP's discovery+execution contract — any MCP client can introspect and invoke this server without prior knowledge of its internals.

## Reference Tables

**Table 6.1: Function Calling vs. Tool Calling**

| Aspect | Function calling | Tool calling |
|---|---|---|
| Control level | Manual, fine-grained | Automatic, high-level |
| Implementation | Write schemas + orchestration code | Simple decorators |
| Execution | You handle everything | Framework handles everything |
| Conversation flow | Manual back-and-forth | Automatic management |
| Error handling | Custom implementation | Built-in |

**MCP vs. ACP — division of labor**

| Layer | Hosts | Focuses on |
|---|---|---|
| MCP server | Tools (data/API access) | Reliable execution, data transformation, discovery |
| ACP server | Agent intelligence | Intent understanding, state/orchestration, result synthesis |

## Worked Example

**Census Data Agent, end to end**: a user asks "Which state has the largest proportion of Black residents?" →
1. `understand_intent` node (LLM) parses this into `{demographic_type: "race", geographic_level: "state", analysis_type: "highest", specific_demographic: "Black or African American", geographic_filter: null}` using a prompt with worked examples to disambiguate phrasing.
2. `query_census_data` node opens a `ClientSession` to the MCP census server and calls `us_census_tool` with the derived parameters — no hard-coded mapping from question-phrasing to API parameters.
3. `process_and_answer` node receives the DataFrame, computes the proportion calculation (non-trivial math — raw counts must be converted to percentages of total state population) and returns a direct, human-readable answer.
Contrast with the chapter's earlier *manual* notebook client, which required hard-coding variable codes (`B02001_002E`, etc.) and a FIPS-code-to-name mapping by hand — the agent layer is precisely what converts that brittle, hard-coded integration into a natural-language interface that generalizes to new questions.

## Key Takeaways
1. Default to tool calling (framework-managed) over function calling (manual) unless you specifically need fine-grained control, custom orchestration, or human-in-the-loop gating.
2. Separate tool hosting (MCP) from agent intelligence (ACP) as soon as a project outgrows a single prototype script — the four benefits (parallel dev, isolation, standardization, abstraction) compound as the system grows.
3. Specialize agents by giving each a narrow, focused toolset rather than one agent with every capability — this is both an accuracy and a latency decision.
4. Use LLM-based intent parsing (structured JSON extraction with worked examples) instead of keyword matching or hard-coded client logic to translate natural language into tool parameters.
5. Design tools to return both a human-readable summary and machine-parseable structured data — serves immediate users and downstream processing simultaneously.
6. Dynamic tool discovery (`list_tools()`) rather than hard-coded dependencies is what makes an agent portable across any compatible MCP server, not just the one it was built against.

## Connects To
- **Ch 3 (The Agentic Ecosystem)**: MCP and A2A were introduced there at a protocol-overview level; this chapter shows their concrete implementation, plus ACP as a third complementary standard.
- **Ch 4 (LLMs as Cognitive Cores)**: the "grounding and world model accuracy" limitation named there is directly solved by the tool/MCP grounding techniques here.
- **Ch 5 (Agent Patterns)**: LangGraph's state/graph model, introduced there, is reused directly for the Census Data Agent's three-node workflow.
- **Ch 7 (Memory Systems)**: next chapter's RAG/vector-database memory mechanisms are a sibling grounding technique to the live-data tool access covered here.
- **Ch 13 (Security and Safety)**: MCP's third-party-server trust risk (flagged in Ch 3) applies directly to any production deployment of the patterns shown here.
