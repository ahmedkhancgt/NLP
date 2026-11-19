# Week 2, Day 2: Agentic RAG, MCP Architecture, and Long-Term Memory

## Lessons Covered
- Lesson 54: How RAG Works - Embeddings, Vector Databases and Semantic Search
- Lesson 55: Agentic RAG, Research Sub-Agents and Agent-to-Agent Communication
- Lesson 56: Retrieval Evals and When to Use RAG, Agentic RAG or Fine-Tuning
- Lesson 57: MCP Architecture - Hosts, Clients, Servers, stdio and Streamable HTTP
- Lesson 58: MCP Server Parameters, Remote Servers, Common Uses and Trade-Offs
- Lesson 59: Short-Term and Long-Term Agent Memory and the Even Harder Claims
- Lesson 60: Agentic RAG over MCP - Give an Agent Long-Term Memory with Qdrant
- Lesson 61: Comparing the UnderCover Agent with and without Memory in Langfuse

## Core Idea
Giving an agent long-term memory through agentic RAG — where the agent itself decides when and what to search — turns an agent that confidently gets things wrong into one that gets them right, and MCP makes wiring up that memory (a full vector database and retrieval tool) a matter of a few configuration parameters rather than a week of infrastructure work.

## Frameworks Introduced
- **Traditional RAG pipeline**: When to use: simple, low-latency knowledge lookup where one retrieval pass is enough. How: chunk documents → embed chunks into vectors with an encoder/embedding model → store in a vector database → embed the user's query → semantic search for nearest chunks → inject the retrieved text (never raw vectors) into the prompt of the normal autoregressive LLM.
- **Agentic RAG**: When to use: whenever retrieval needs to be iterative, multi-source, or query-refining, or the first pass might miss relevant content. How: equip the agent with a "retrieve context" tool (semantic search) and let it decide how many times to call it, how to rewrite/split its own queries, and when it has enough information — replacing the "cottage industry" of query rewriting, query expansion, re-ranking, hierarchical RAG, and graph RAG with agent autonomy.
- **Research sub-agent pattern**: When to use: when retrieval needs multiple tools/sources (semantic search, web search, SQL, markdown, business tools) and you want to keep that mess out of the main agent's context. How: wrap a dedicated research agent behind a tool call ("do research for this question"); it uses many tools internally and returns only a distilled research brief to the main agent.
- **Agent-to-agent communication (demystified)**: When to use: your own agents, on your own infrastructure. How: just a tool call — the calling agent's tool description documents what the sub-agent does; no special protocol needed. Reserve A2A-style protocols for genuinely heterogeneous agents built by different teams on different infrastructure (not yet a pressing problem for most practitioners).
- **MCP architecture**: host / client / server roles.
  - When to use: to consume tools (or resources/prompts) built by someone else.
  - How: the **host** is the AI application (Claude Code, ChatGPT, Claude Desktop, your LangChain agent); it runs one **MCP client** per MCP server it wants to use; the **MCP server** is the bundle of tools (written by anyone, any language) that answers exactly two requests: "list tools" and "call tool." Transport is either **stdio** (client spawns the server as a local subprocess, talks over standard input/output — most common for your own agents) or **Streamable HTTP** (client connects to a server running remotely; required for things like ChatGPT connectors, and used by servers whose provider wants to host the infrastructure, e.g. Atlassian, GitHub, Context7). The legacy SSE transport is largely superseded by Streamable HTTP.
- **RAG vs. Agentic RAG vs. Fine-tuning decision framework**:
  - When to use RAG: need is simple, fast, cheap, easy to evaluate (MRR, precision/recall), and one retrieval pass is enough; accept it can miss facts.
  - When to use Agentic RAG: need accurate results across multiple sources, willing to accept higher cost/latency/unpredictability and harder evaluation, in exchange for multi-query refinement and combining vector + SQL + web search.
  - When to use fine-tuning: rare; only when the model needs to *learn a generalizable skill* from your data (not just memorize facts) and the skill isn't already present in a frontier model — otherwise RAG/agentic RAG/tools are cheaper and faster to iterate on, and frontier-model intelligence already generalizes from a few in-context examples.

## Key Concepts
- **Embedding model / encoder**: a non-autoregressive LLM that converts text into a vector (typically 1,000+ dimensions) such that semantically similar text produces nearby vectors.
- **Semantic search**: looking up stored content by meaning (vector distance) rather than exact string match — this is "all the voodoo" in RAG; the generation-time LLM never sees vectors, only retrieved text.
- **Vector database**: a database that indexes text chunks by their embedding vectors (e.g., Qdrant, Chroma) to support nearest-neighbor semantic search.
- **MRR (Mean Reciprocal Rank)**: measures how high up relevant content is surfaced in the retrieval results.
- **Retrieval precision**: of the chunks retrieved, what proportion were actually relevant.
- **Retrieval recall**: of the chunks that should have been retrieved, what proportion were actually found — usually the more important metric for RAG, because an LLM can ignore irrelevant chunks but can't use facts it never received.
- **found_fact metric**: a custom eval used for agentic RAG that tracks whether the agent's searches actually surfaced the specific fact needed to answer correctly (used in place of simple recall, since agentic RAG's multi-source, rewritten retrievals make traditional recall hard to define).
- **MCP host / client / server**: see Frameworks above — the three core roles in the protocol's terminology.
- **MCP resources and prompts**: the two other MCP primitives besides tools; rarely used in practice — tools dominate adoption.
- **uvx / npx**: the "run this package in an isolated environment, no install step" commands used to launch stdio MCP servers (`uvx` for Python/uv packages, `npx` for Node/npm packages).
- **Short-term memory**: the conversation history passed into the agent on each call.
- **Long-term memory**: context that persists *between* conversations, usually accessed via tools (markdown file reads, SQL queries, custom business-query tools, or RAG-style semantic search).
- **Four long-term memory types (AWS Bedrock AgentCore convention)**: semantic (facts across conversations), preference (per-user settings), summary (running summary of prior sessions), episodic (specific past events and outcomes) — introduced but the instructor recommends starting with just one type.
- **Learning from experience vs. fine-tuning**: an agent harness rewriting its own markdown/memories at inference time mimics "training" in effect but never changes model weights — distinct from true fine-tuning/post-training.

## Mental Models
- Think of the embedding/vector step as a "yellow box" that's entirely separate from the "green box" (the actual answer-generating LLM) — the green LLM only ever sees retrieved text, never vectors; confusing the two is the most common RAG misconception.
- Treat agentic RAG as "traditional RAG's cottage-industry techniques (query rewriting, expansion, re-ranking) absorbed into agent autonomy" — the agent can already do all of that itself by choosing how to call its retrieval tool.
- Use "would I trust a stranger's code to run inside my own process?" as the test for whether something needs MCP — if it's your own tool for your own agent, skip MCP and just use `@tool`.
- For memory design, default to one memory type and add more categories only when a measured problem (duplicate info, wrong-store lookups) demands it — mirrors Day 1's "start simple, add complexity only when it measurably helps."
- View MCP's real value as translation, not infrastructure: "it's taking an API and translating it into natural language and data types ... that's the bridge between an LLM generating tokens and calling an API."

## Anti-patterns
- **Believing RAG passes vectors into the generation-time LLM**: it never does; only retrieved text goes into the prompt.
- **Over-engineering traditional RAG with a stack of cottage-industry techniques** (query rewriting, semantic chunking, re-ranking, graph RAG) instead of trying agentic RAG or just evaluating and iterating on the basics.
- **Using MCP for your own tools on your own agent**: adds unnecessary process-boundary/infrastructure overhead when `@tool` + `create_agent` does the job directly.
- **Treating "local MCP server" as meaning it's not someone else's code**: it still is — it's just conveniently run on your machine (via `uvx`/`npx`) instead of a shared remote server.
- **Skipping MCP server vetting**: security and quality vary hugely; check GitHub stars, issue activity, last-updated date, and maintainer reputation before using an unfamiliar server, especially given prompt-injection and local-execution risks.
- **Assuming A2A-style protocols are necessary for your own multi-agent setup**: unnecessary complexity when a tool call or shared markdown files suffice for agents you control.
- **Reaching for fine-tuning to "teach the model facts"**: fine-tuning is for learning generalizable skills, not memorizing a knowledge base — use RAG/tools/context instead.

## Code Examples
```python
parameters = {"transport": "stdio", "command": "uvx", "args": ["--python", "3.12.12", "mcp-server-qdrant@0.8.1"], "env": MEMORY_ENV}
client = MultiServerMCPClient({"memory": parameters})
```
- **What it demonstrates**: the entirety of what's needed to stand up a full vector-database MCP server (Qdrant) as a LangChain MCP client — no manual database install, no custom retrieval code.

```python
read_only_client = MultiServerMCPClient({
    "memory": {"transport": "stdio", "command": "uvx", "args": ["--python", "3.12.12", "mcp-server-qdrant@0.8.1"],
               "env": {**MEMORY_ENV, "QDRANT_READ_ONLY": "true"}}
})
```
- **What it demonstrates**: giving the evaluation agent read-only access to memory (via `QDRANT_READ_ONLY`) so running the same claims dataset twice can't let the agent "cheat" by having written memories during scoring.

```python
search = [memory_tool for memory_tool in memory_tools if memory_tool.name == "qdrant-find"][0]
# ... later, pulling retrieved text out of the trace:
retrieved = "\n".join(message.text for message in result["messages"]
                       if message.type == "tool" and message.name == "qdrant-find")
```
- **What it demonstrates**: selecting a specific MCP tool out of the list returned by the server, and extracting what the agent actually retrieved from a LangChain tool-call message for scoring (`found_fact`).

## Reference Tables
| Approach | Pros | Cons |
|---|---|---|
| Traditional RAG | Simple, fast, cheap, easy to evaluate (MRR, precision/recall) | Least powerful — single retrieval pass, can miss facts |
| Agentic RAG | Multi-query, multi-source (vector + SQL + web), self-correcting, more accurate | More expensive, slower, less predictable, harder to evaluate |
| Fine-tuning | Learns generalizable skills beyond a single prompt's context | Expensive, slow, needs re-training per base model, rarely needed given frontier-model in-context intelligence |

| MCP transport | Where server runs | Typical use |
|---|---|---|
| stdio | Spawned locally (`uvx`/`npx`) | Your own agents, Claude Code, Playwright, Qdrant |
| Streamable HTTP | Remote, provider-hosted | ChatGPT connectors, Atlassian/Jira, GitHub, Context7 |
| SSE (legacy) | Remote | Mostly superseded by Streamable HTTP |

| Metric (UnderCover, 25 "even harder" claims) | Payout tool, no memory | Payout tool + MCP memory |
|---|---|---|
| Decision accuracy | 60% | 100% |
| Accuracy | 56% | 100% |
| Precision | 86% | 100% |
| Recall | 71% | 100% |
| Kappa | 0.30 | 1.0 |
| Cost of mistakes | high (11/25 wrong) | $0 |
| Straight-through rate | — | 92% |
| LLM cost per claim | ~1¢ | ~1.4¢ |

## Worked Example
The Week 2 Day 2 lab (`labs/week2day2.ipynb`) builds an "even harder" 25-claim dataset where correct adjudication requires facts the agent can't possibly know (e.g., that the "Halden Arc" is a pocket projector excluded from an Electronics Plus policy, or when Tropical Storm Marisol was named). Part 1 loads the claims and a `knowledge.json` of 33 facts about products, places, and events. Part 2 stands up a Qdrant MCP server over stdio (`uvx mcp-server-qdrant@0.8.1`) via LangChain's `MultiServerMCPClient`, exposing exactly two tools: `qdrant-store` and `qdrant-find`. Before any memories are written, asking the agent "What kind of device is the Halden Arc?" correctly returns "I don't know" (semantic search over an empty database). Part 3 loops through all 33 facts, calling `qdrant-store` via the agent for each ("please remember this"), after which the same question is answered correctly ("a pocket projector") — confirmed independently by calling `qdrant-find` directly with an unrelated-sounding query ("a gadget for watching films on a wall") and getting the Halden Arc entry back by meaning, not keyword. Part 4 runs two Langfuse experiments on the identical 25-claim dataset: Experiment 1 reuses the Day 1 "payout tool" agent with no memory access (read-only MCP client still attached but empty of relevant facts) — result: 60% decision accuracy, 11/25 claims wrong, kappa 0.30. Experiment 2 gives the same agent the populated, read-only memory via `qdrant-find` — result: a perfect confusion matrix, 100% accuracy/precision/recall, kappa 1.0, cost of mistakes $0, at roughly 40% higher LLM cost (1¢ → 1.4¢) from the extra retrieval tool calls.

## Key Takeaways
1. RAG's only real trick is semantic search (embedding + vector lookup) to supplement a prompt — the generation LLM never sees vectors, only retrieved text.
2. Agentic RAG subsumes most of traditional RAG's "cottage industry" (query rewriting, expansion, re-ranking) simply by letting the agent decide how and how often to search.
3. Recall usually matters more than precision for RAG, because the LLM can ignore noise but can't recover facts it was never given.
4. MCP's entire value is natural-language tool description that bridges an LLM's token generation to an API call — not any deep technical innovation, but near-universal adoption.
5. Use stdio for your own locally-run agents; use Streamable HTTP for provider-hosted remote servers that need OAuth-style authentication (e.g., Jira, GitHub).
6. Never use MCP for tools you wrote yourself for your own agent — `@tool` + `create_agent` is simpler and has no process-boundary overhead.
7. Start long-term memory with one type (here: semantic, via agentic RAG over Qdrant) and only add preference/summary/episodic memory if a measured problem demands it.
8. Agent-to-agent communication for your own agents is just a tool call with a good description — reserve A2A-style protocols for truly cross-infrastructure, cross-team agent ecosystems.
9. Fine-tuning is now rare in applied AI engineering; reach for it only when the model needs a new generalizable skill, not when it needs more facts.

## Connects To
- **Week 2, Day 1**: reuses the exact same evals/Langfuse experiment discipline (confusion matrix, kappa, cost of mistakes) established on Day 1 to prove the value of adding memory, mirroring how Day 1 proved the value of the payout tool.
- **Week 2, Day 3**: long-term memory and MCP servers here are stateless tool calls; Day 3 introduces durable, checkpointed *state* across a whole agent run via LangGraph, including a Supabase MCP server for the durable build.
- **AWS Bedrock AgentCore**: referenced as a more heavyweight, four-category long-term memory implementation to be revisited in Week 3's system design content.
