# Chapter 7: Large Language Model Frameworks

## Core Idea
Production LLM applications require an orchestration framework (LangChain, LlamaIndex) layering chains, memory, document loaders, chunking, vector stores, and agents on top of a raw model, and Retrieval-Augmented Generation (RAG) is the dominant enterprise pattern for grounding LLMs in proprietary knowledge without retraining.

## Frameworks Introduced
- **LangChain component stack**: Chain → PromptTemplate/ChatPromptTemplate → Memory → Document Loaders → Splitters/chunking → Vector Stores → Agents.
  - When to use: building any LLM application needing more than a single raw prompt-response call.
  - How: wrap a prompt template + LLM into a `LLMChain`; compose multiple chains via `SequentialChain`; use chain strategies (stuffing, map-reduce, map-rank, refine) depending on whether context exceeds the model's token window.
- **Chain strategy selection for long documents**: stuffing (single most-relevant doc) vs. map-reduce (parallel per-doc then combine) vs. map-rank (per-doc + confidence ranking) vs. refine (sequential, each doc refines prior answer).
  - When to use: match to data shape — stuffing when one chunk suffices, map-reduce for parallelizable independent documents, refine when accuracy from incremental context-building matters more than speed.
- **Memory types**: `ConversationBufferMemory` (all messages), `ConversationBufferWindowMemory` (last k messages), `ConversationTokenBufferMemory` (token-budget-limited), `ConversationSummaryMemory` (rolling summary).
  - When to use: pick by conversation length and context-budget constraints — summary memory for long-running conversations that would otherwise exceed context window.
- **RAG sequence**: retrieve relevant chunks from a vector store → augment the user query with retrieved context → generate the final answer (optionally with cited sources).
  - When to use: grounding a time-static LLM in current/domain-specific/proprietary knowledge without fine-tuning.
  - How: embed documents, store in a vector database, similarity-search against the query embedding, stuff/synthesize retrieved chunks into the LLM prompt.
- **Small-to-big (auto-merging) retrieval**: build a hierarchy of parent chunks (e.g., 1024 tokens) with multiple child chunk sizes (128/256/512) referencing their parent via `IndexNode` links.
  - When to use: when the answer to a query spans multiple document sections and needs broader context than a single small chunk provides.
  - How: retrieve fine-grained child chunks first (precise matching), then expand to parent chunks for fuller context during synthesis (LlamaIndex's `RecursiveRetriever` + `RetrieverQueryEngine`).
- **Agent framework (ReAct pattern)**: LLM + tools (e.g., SQL query execution) where the model decides which tool to invoke and with what arguments.
  - When to use: tasks requiring the LLM to interact with external systems/data sources dynamically (not just static retrieved text).
  - How: define a `Tool` wrapping a function (e.g., `run_sqlite_query`), register it with an `OpenAIFunctionsAgent`/`AgentExecutor`; the LLM receives tool descriptions, decides relevance, generates the call (e.g., SQL query), and the application executes and returns results for final response generation.
- **Property graph extraction pipeline**: PDF ingestion → vector embedding (Pinecone) → LLM-based entity/relationship extraction (Vertex AI Text-Bison) → Cypher query generation → graph database insertion (Neo4j) → visualization (NetworkX/Pyvis).
  - When to use: converting unstructured documents (research journals) into queryable knowledge graphs for relationship-centric analysis.

## Key Concepts
- **Chain**: a LangChain class wrapping a prompt template + LLM into a reusable, composable text-generation pipeline.
- **Vector store**: a database optimized for storing/querying embedding vectors via similarity (e.g., Chroma, Pinecone), enabling contextual retrieval rather than exact-match search.
- **Chunking (fixed-size vs. recursive)**: splitting documents into token-budget-respecting pieces, optionally with overlap to preserve semantic continuity across chunk boundaries.
- **Document loader**: a LangChain abstraction (`PyPDFLoader`, `TextLoader`, `JSONLoader`, `WikipediaLoader`, S3 loaders) that converts various source formats into a uniform `Document` object (page_content + metadata).
- **Agent**: a software component that lets an LLM decide which external tool to invoke and with what parameters, then incorporates the tool's output into its response.
- **ReAct (Reason + Act)**: an agent pattern combining reasoning and action — breaking a query into actionable sub-tasks executed sequentially.
- **IndexNode**: a LlamaIndex object representing a document chunk with a unique ID, used to link child chunks to parent chunks in hierarchical retrieval.
- **Embedding model vs. LLM separation**: the generation model (e.g., GPT-3.5-Turbo) and the embedding model (e.g., BAAI/bge-large-en-v1.5 via FlagEmbedding) are typically distinct, bundled together via a service/context object.

## Mental Models
- Think of **a chain as a function composition pipeline**: each chain step transforms data (prompt → LLM call → output key), and `SequentialChain` composes these like function composition in regular software engineering.
- Use the **"memory is not automatic" model**: LLMs are inherently stateless per call — any conversational continuity is an engineering add-on (buffer, window, token-limited, or summary memory), not a model capability.
- Think of **RAG as "open-book exam" grounding**: instead of trying to make the model memorize everything (fine-tuning/retraining), RAG hands it the relevant "textbook pages" (retrieved chunks) at query time.
- Use the **small-to-big retrieval as a zoom lens**: retrieve with fine granularity (small chunks, precise matching) then zoom out to the parent chunk for synthesis context — avoids the trade-off between precision (small chunks) and sufficient context (large chunks).
- Think of **agents as giving the LLM hands, not just a mouth**: the LLM's native capability is only text generation; agents are the mechanism by which it can trigger real actions (SQL queries, API calls) in the world.

## Anti-patterns
- **Using a single-chunk "stuffing" strategy when an answer spans multiple documents**: truncates relevant context — use map-reduce or small-to-big retrieval instead.
- **Giving an LLM unlimited raw conversation history as context**: wastes tokens and risks exceeding the context window — use windowed, token-limited, or summary memory as conversations grow.
- **Hardcoding table/schema names into agent prompts when the goal is reusability**: the chapter notes that supplying table schema and description alone (without an explicit table name) lets the agent generalize better across databases.
- **Skipping chunk overlap in splitting**: chunks without overlap can sever context at boundaries, losing semantic continuity needed for accurate retrieval.
- **Choosing LangChain for every use case regardless of fit**: LangChain is powerful but has a steeper learning curve for general-purpose chat apps; LlamaIndex/Haystack may be better suited specifically for retrieval-centric (search, summarization) applications.

## Code Examples
```python
# Basic LangChain pipeline: prompt template + LLM wrapped as a reusable chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

llm = OpenAI(openai_api_key=api_key)
category_prompt = PromptTemplate(input_variables=["risk"], template="Categorize the following risk {risk} ")
category_chain = LLMChain(llm=llm, prompt=category_prompt, output_key="category")

rating_prompt = PromptTemplate(
    input_variables=["category", "risk"],
    template="Rate the following risk {category}:\n{risk} between 1-5 with 1 as minimum and 5 as maximum.")
rating_chain = LLMChain(llm=llm, prompt=rating_prompt, output_key="rating")

chain = SequentialChain(chains=[category_chain, rating_chain],
                         input_variables=["risk"], output_variables=["category", "rating"])
result = chain({"risk": "No antivirus software in my system"})
```
```python
# RAG with vector store (Chroma): embed, store, similarity search
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)
chunks = PyPDFLoader("23501-gi0.pdf").load_and_split(text_splitter=splitter)
db = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(openai_api_key=api_key), persist_directory="spec_emb")
results = db.similarity_search("What is social impact of 5G?")
```
```python
# Agent with a custom tool (SQL execution) using OpenAIFunctionsAgent
from langchain.tools import Tool
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor

def run_sqlite_query(query):
    c = conn.cursor(); c.execute(query); return c.fetchall()

run_query_tool = Tool.from_function(name="run_sqlite_query", description="Run a sqlite query.",
                                     func=run_sqlite_query, args_schema=RunQueryArgsSchema)
agent = OpenAIFunctionsAgent(llm=ChatOpenAI(temperature=0), prompt=prompt, tools=[run_query_tool])
agent_executor = AgentExecutor(agent=agent, verbose=True, tools=[run_query_tool])
agent_executor("Get me total number of students")
```
- **What it demonstrates**: (1) composing sequential chains for multi-step LLM reasoning (categorize then rate a risk); (2) the full RAG retrieval pipeline from document to vector store to similarity search; (3) an agent autonomously generating and executing a SQL query in response to a natural-language request.

## Reference Tables
**Chain strategies for long documents**

| Strategy | Mechanism | Best for |
|---|---|---|
| Stuffing | Most relevant chunk(s) inserted directly | Context fits within a few chunks |
| Map-reduce | Per-document answers combined | Parallelizable, independent documents |
| Map-rank | Per-document answers ranked by confidence | Need best single-source answer |
| Refine | Sequential, each doc refines prior answer | Accuracy from incremental context |

**LangChain memory types**

| Memory type | Stores | Best for |
|---|---|---|
| ConversationBufferMemory | All messages | Short conversations |
| ConversationBufferWindowMemory | Last k messages | Bounded-length context |
| ConversationTokenBufferMemory | Messages up to token budget | Token-budget-constrained apps |
| ConversationSummaryMemory | Rolling summary | Long-running conversations |

**LangChain vs. LlamaIndex (as framed in this chapter)**

| Framework | Strength | Trade-off |
|---|---|---|
| LangChain | Extensive, modular, general-purpose (chains, agents, memory) | Steeper learning curve |
| LlamaIndex | Specialized for retrieval (search, summarization), auto-merging retrieval | Narrower feature scope |

## Worked Example
**Property graph creation from a medical research journal**: Upload a PDF research article via Streamlit; hash and preview it; chunk and embed it into Pinecone. Use a prompt template with Google Vertex AI's Text-Bison model to extract entities (authors, keywords, categories, article) and relationships between the article and its references/entities from the raw text. Clean and parse the LLM's extracted entity/relationship text into structured lists. Generate Cypher queries from these structured records and insert them into a Neo4j graph database, creating nodes (article, author, keyword, category) with properties (abstract, DOI, article_type) and labeled relationships between them. Finally, visualize the resulting graph using NetworkX and Pyvis rendered as an HTML page inside the Streamlit app. This demonstrates a complete pipeline: unstructured PDF → vector-grounded LLM extraction → structured graph database → interactive visualization — combining UI (Streamlit), data modules (SQLite, Neo4j, Pinecone), and LLM modules (OpenAI, Vertex AI, LangChain) into one enterprise application.

## Key Takeaways
1. Production LLM applications need an orchestration framework (chains, memory, document loaders, vector stores, agents) — raw model API calls alone are insufficient for anything beyond a single-turn demo.
2. RAG is the preferred enterprise technique for grounding LLMs in proprietary/current knowledge because it avoids the cost, lock-in, and staleness issues of fine-tuning (echoing Ch 6's "fine-tuning as a late resort" takeaway).
3. Chunking strategy and chain strategy (stuffing/map-reduce/map-rank/refine) must be matched to whether the needed context fits in one chunk or spans multiple documents.
4. Small-to-big (auto-merging) retrieval resolves the precision-vs-context trade-off inherent in naive fixed-chunk RAG by retrieving with small chunks and synthesizing with their parent context.
5. Agents extend LLMs from text generators into action-takers by giving them access to tools (SQL execution, APIs) whose outputs feed back into the generation loop.
6. Knowledge graphs (Neo4j) built from LLM-extracted entities/relationships turn unstructured documents into a structured, queryable form — useful for relationship-centric analysis that pure vector search cannot easily express.

## Connects To
- **Ch 5**: RAG was introduced as a QA mitigation technique there; this chapter operationalizes it into a full production pipeline (chunking, embedding, retrieval, synthesis).
- **Ch 6**: the Prompt Playground's LangChain usage previewed there is expanded here into the complete LangChain component stack (chains, memory, agents).
- **Ch 8**: the LLM application components built here (model serving, vector stores, monitoring needs) connect directly to LLMOps' model-serving and inference-optimization practices.
- **Ch 9**: the enterprise RAG/agent pattern here is the concrete technical substrate for the enterprise GenAI use cases discussed later.
