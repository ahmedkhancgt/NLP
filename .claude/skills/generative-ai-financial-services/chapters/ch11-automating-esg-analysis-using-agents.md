# Chapter 11: Use-case: Automating ESG Analysis Using Agents

## Core Idea
A ReAct agent equipped with two tools — a Pinecone vector store (ESG report text) and a SQLite database (structured ESG metrics) — can autonomously decide which tool to query for a given question, demonstrated by building a working ESG-analysis assistant with LangChain and Claude 3.5 Sonnet.

## Frameworks Introduced
- **Agent definition (recap)**: an autonomous software application with access to an LLM, memory, and a user prompt (with instructions like CoT/ReAct and available tools); it loops — calling the LLM to build reasoning and action — until a final answer is found.
- **Function calling vs. agents distinction**: Function calling = LLM is given one function's signature/inputs/return type and builds the call arguments for just that function (available in a handful of models: Anthropic, OpenAI, etc.) — developer retains more control. Agent = handles multiple tools' inputs/outputs internally, with less developer control but more autonomy and recursive/dynamic tool selection.
  - When to use agents over RAG alone: when the answer may live in a database, require a microservice/external API call, or need recursive multi-tool calls, and the application needs to reason about which source to query dynamically.
  - When to build a custom agent vs. use LangChain/LlamaIndex pre-built agents: only when pre-defined implementations don't fit, and only after weighing the ROI against the ongoing operational effort of maintaining a custom agent.
- **ESG agent architecture**: Agent (LLM-driven) ↔ Tools [Vector Store (ESG report narrative text) + SQL Database (ESG_data metrics table)] → agent autonomously selects and queries the right tool(s) to answer a question.
- **ReAct agent implementation pattern (LangChain)**: bind tools to LLM → pull a ReAct/structured-chat prompt template → create the structured chat agent → wrap in an `AgentExecutor` → invoke with a query, optionally reinforced with an explicit instruction to "use all tools at your disposal."

## Key Concepts
- **ESG (Environmental, Social, Governance)**: framework measuring an organization's environmental sustainability, diversity/equity, employee experience, data privacy/governance, and societal impact; all three pillars carry equal importance.
- **ESG score / metrics**: structured data (esg_score, total_emissions, net_zero_target, renewable_energy_pct, board_diversity_pct) suited to SQL/relational storage, as opposed to narrative ESG report content suited to vector-store retrieval.
- **SQLDatabaseToolkit (LangChain)**: wraps common DB operations (list tables, describe schema, run query) as agent-usable tools, abstracting raw SQL away from the agent's reasoning loop.
- **ORM (Object Relational Mapper, e.g., SQLAlchemy)**: lets developers interact with a database using application code instead of hand-written SQL.
- **AgentExecutor**: the LangChain runtime wrapper that actually drives the agent's reasoning/action loop given its tools and prompt.

## Mental Models
- Treat the agent's tool selection as evidence of "right-tool reasoning": a well-designed agent will try the higher-priority tool first (e.g., vector store, per the tool description's instruction), gracefully recognize when that tool didn't have the answer, and fall back to the next tool (e.g., database) — mirroring how a human analyst would triage information sources.
- Use the "give each tool a clear description, including a priority/fallback instruction" pattern (e.g., "This should be given the first priority... If the information is not found, then the database tools must be used") to shape agent behavior without hardcoding routing logic.
- For financial-services ESG use-cases, think of the two information types as structurally different: quantitative, auditable metrics (→ relational DB) vs. qualitative, narrative disclosures (→ vector store/RAG) — and let the agent bridge both.

## Anti-patterns
- **Building a custom agent framework without ROI justification**: when LangChain/LlamaIndex's pre-built agent implementations fit the need, building a bespoke agent adds significant long-term operational maintenance burden for uncertain benefit.
- **Using agents for simple/deterministic tasks** (echoing Ch6): the book's own MCQ flags "deciding whether credit should be processed or not" as a scenario where an agent might NOT be the right fit (needs explainability/determinism, not autonomous recursive reasoning).
- **Letting agent output flow to production without guardrails**: because agents are autonomous, unchecked responses risk hallucination or inappropriate output — guardrails must intercept every agent response.
- **Scaling agentic chains into deep, complex multi-agent graphs without architectural restraint**: replacing simple tools with more agents creates stateful, complex chains that become very difficult to operate and maintain — keep the architecture lean.
- **Removing human-in-the-loop oversight from agent reasoning**: agents can reason down a wrong path and produce confidently wrong output; knowledge workers should review agent logs/reasoning, not just final answers.

## Code Examples
```python
# --- Database setup (SQLite + SQLAlchemy) ---
esg_data = Table(
    "esg_data", metadata,
    Column("instance_id", Integer, primary_key=True),
    Column("company_name", String(4), nullable=False),
    Column("esg_score", Float, nullable=False),
    Column("total_emmisions", Float, nullable=True),
    Column("net_zero_target", Date, nullable=True),
    Column("renewable_energy_pct", Float, nullable=True),
    Column("board_diversity_pct", Float, nullable=True),
)

db = SQLDatabase(sql_engine)
sql_db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_tools = sql_db_toolkit.get_tools()

# --- Vector store setup (Pinecone) ---
pinecone_client = Pinecone(api_key=pinecone_api_key)
pinecone_client.create_index(
    index_name, dimension=768, metric='cosine',
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
vector_index = pinecone_client.Index(index_name)
vector_store = PineconeVectorStore(index=vector_index, embedding=embedding_model)

def add_document_to_vector_store(document_path):
    with open(document_path) as f:
        esg_report = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=10, length_function=len, is_separator_regex=False,
        )
        texts = text_splitter.split_documents([Document(esg_report)])
        uuids = [str(uuid4()) for _ in range(len(texts))]
        vector_store.add_documents(documents=texts, ids=uuids)

# --- Agent setup ---
vector_store_retriever = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
)
rag_tools = [
    Tool(
        name='Vector Store',
        func=vector_store_retriever.run,
        description="Useful for searching information from the knowledge base. This should "
                     "be given the first priority while searching for information. It has "
                     "information about Company1, Company2 and Company3's ESG reports in "
                     "detail. If the information is not found, then the database tools "
                     "must be used to find the answer")
]
tools = rag_tools + db_tools
llm_with_tools = llm.bind_tools(tools)

prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm_with_tools, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

additional_instructions_for_agent = "Use all the tools at your disposal to find the right answer for the query below: \n\n"
query = additional_instructions_for_agent + "Can you tell me what is the ESG score for Company3?"
agent_executor.invoke({"input": query})
# Agent tries VectorStore first (no numeric score found) -> falls back to sql_db_list_tables
# -> sql_db_schema -> determines ESG score = 300.0 from the esg_data table -> Final Answer

query2 = additional_instructions_for_agent + "What is the favorability rating on work-life balance and flexibility for Company1 in the year 2022?"
agent_executor.invoke({"input": query2})
# Agent tries VectorStore first -> finds "91% favorability rating" directly in the ESG report text -> Final Answer
```
- **What it demonstrates**: a working multi-tool ReAct agent that autonomously picks between a narrative vector store and a structured SQL database based on tool descriptions and the nature of the question — including a visible fallback (vector store fails → agent queries DB schema → finds the answer) and a visible first-try success (vector store directly answers a qualitative question).

## Reference Tables
**Technology stack used in this chapter**

| Layer | Tool | Notes |
|---|---|---|
| Database | SQLite (in-memory) | Lightweight RDBMS for structured ESG metrics |
| ORM | SQLAlchemy | Interacts with DB without raw SQL |
| DB agent tools | LangChain SQLDatabaseToolkit | Wraps list-tables/describe-schema/query as tools |
| Vector store | Pinecone (free serverless, ≤5 indexes) | Stores chunked ESG report narrative text |
| Embedding model | Sentence-transformer (768 dim) on Hugging Face | Captures semantic meaning of ESG report text |
| LLM | Claude 3.5 Sonnet via Amazon Bedrock | Chosen because Llama 3 "struggled with ReAct agents" in internal testing |
| Agent framework | LangChain ReAct / structured-chat-agent | `create_structured_chat_agent` + `AgentExecutor` |

**Productionization checklist**: scalability (multiple LLM/agent instances, DB/vector-store load handling), guardrails on every agent response (hallucination/inappropriate output), enterprise-grade vector store/DBMS (Pinecone enterprise key or alternative; Amazon RDS or similar), human-in-the-loop review of agent reasoning logs, and keeping agent architecture lean (avoid deep multi-agent chains that become unmaintainable).

## Worked Example
See Code Examples above: two live queries ("ESG score for Company3" and "work-life balance favorability rating for Company1 in 2022") show the agent's tool-selection reasoning transcript in full — demonstrating graceful fallback from vector store to database when the first tool doesn't have the answer, and direct success when the vector store does.

## Key Takeaways
1. Agents are the right tool when answering a question requires dynamically choosing between multiple heterogeneous data sources (structured DB vs. unstructured vector store) rather than always querying the same source.
2. Tool descriptions can encode priority/fallback logic in natural language (e.g., "try vector store first, then database") instead of hardcoded routing code.
3. Prefer established agent frameworks (LangChain, LlamaIndex) over custom agent-building unless there's a strong ROI case — operational maintenance of custom agents is expensive long-term.
4. Model choice matters for agent reliability: the book found Claude 3.5 Sonnet outperformed Llama 3 specifically for ReAct-style agent reasoning.
5. Guardrails and human-in-the-loop review are non-negotiable for production agentic systems because of their autonomous, potentially-wrong-path reasoning.
6. Keep agentic architectures lean — avoid deep chains of agents calling agents, which become very hard to operate and maintain.

## Connects To
- **Ch3**: Implements the agent Plan/Execute/Memorize framework and ReAct technique defined there.
- **Ch4**: Extends the investment-research use-case pattern (manual multi-source analysis) into a fully automated agentic solution.
- **Ch6**: The agent-vs-simple-approach and custom-agent-ROI anti-patterns from Ch6 are directly echoed in this chapter's guidance.
- **Ch13**: Guardrails on agent output connect to the trustworthiness practices covered there.
- **Ch12**: This use-case is a concrete input into the "choosing the right implementation approach" decision framework.
