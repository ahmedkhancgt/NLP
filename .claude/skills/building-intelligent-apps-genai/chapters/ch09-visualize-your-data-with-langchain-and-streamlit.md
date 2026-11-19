# Chapter 9: Visualize your Data with LangChain and Streamlit

## Core Idea
A CSV-querying data-visualization chatbot can be built by pairing a LangChain `create_pandas_dataframe_agent` (which reasons over the DataFrame via an LLM) with a strict JSON-shaped prompt (answer/bar/line/table) and a Streamlit UI that pattern-matches the returned JSON key to decide which chart/table/text widget to render.

## Frameworks Introduced
- **LangChain+Streamlit data-viz pattern**: user uploads CSV → `create_pandas_dataframe_agent` wraps the DataFrame with an LLM → `query_agent` sends a structured prompt instructing the model to reply in one of three JSON shapes (`{"bar": {...}}`, `{"line": {...}}`, `{"answer": "..."}`) → Streamlit's `write_response` inspects the returned dict's key and renders `st.bar_chart`, `st.line_chart`, `st.table`, or `st.write` accordingly.
  - When to use: building a natural-language interface over tabular/CSV data where the output should automatically become the right visualization type.
  - How: see Code Examples and Worked Example below for the full prompt and routing logic.
- **Agent pattern (LangChain `create_pandas_dataframe_agent`)**: software that acts independently to achieve a goal by interacting with an LLM and external resources (here, a pandas DataFrame) based on a user query.
  - When to use: whenever a task requires an LLM to reason over structured data rather than free text.
- **Chunking for vector storage (conceptual, reused from Ch5/Ch7)**: breaking documents into bite-sized pieces stored as embeddings in a vector database (Pinecone, Chroma) for later querying — mentioned as LangChain's general document-retrieval mechanism, though this chapter's hands-on build uses the simpler DataFrame agent instead of a vector store.

## Key Concepts
- **LangChain**: an open-source Python/JavaScript framework connecting LLMs to external/custom data sources so responses are grounded in enterprise-specific data rather than general knowledge alone.
- **`create_pandas_dataframe_agent`**: the specific LangChain Python utility used in this chapter to let an LLM (via `langchain.llms.OpenAI`) query a pandas DataFrame loaded from CSV.
- **`decode_response`**: a Streamlit-side helper that parses the agent's string response into a Python dictionary for routing.
- **`write_response`**: the dispatcher function that inspects the decoded dict's key (`answer`, `bar`, `line`, `table`) and calls the matching Streamlit rendering function.
- **Environment file (`.env`) with `environ`**: the book reuses the "never hard-code API keys" practice from Chapter 6, here using the `environ` package (`env = environ.Env(); environ.Env.read_env(); API_KEY = env("apikey")`).

## Mental Models
- Think of the agent's JSON-shaped response as a "routing instruction" to the UI: the LLM isn't just answering — it's also telling Streamlit *how* to display the answer (table vs. chart vs. text).
- Treat the structured prompt (bar/line/answer schema) as a contract between the LLM and the front-end: as long as the model honors the contract, any new chart type can be added by extending both the prompt and `write_response` symmetrically.
- Use "I do not have sufficient information" / "I do not know" as designed fallback behavior, not a bug — the prompt explicitly instructs the model to refuse gracefully on out-of-scope or unknown queries, directly preventing hallucination on tabular Q&A.

## Anti-patterns
- **Uploading very large CSV files**: the chapter notes you may hit OpenAI token-count/usage errors when the data volume exceeds the model's token limit — use smaller datasets and be mindful of how much data a query pulls in.
- **Assuming consistent formatting across runs**: the book's own test shows the model successfully rounded balances in one run but failed to round them in another when asked — treat numeric formatting instructions as not fully reliable and validate output.
- **Skipping the explicit "reply as JSON" prompt contract**: without instructing the model precisely how to format bar/line/table/answer responses, `write_response`'s key-based dispatch has nothing reliable to parse.

## Code Examples
```bash
conda create --name langchainenv python=3.12
conda activate langchainenv
pip install langchain openai streamlit python-environ tabulate
```

```python
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

# Setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")
```

```text
If the query requires creating a bar chart, reply as follows:
   {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

If the query requires creating a line chart, reply as follows:
   {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

If it is just asking a question that requires neither, reply as follows:
   {"answer": "answer"}
```

```python
# Run the prompt through the agent.
response = agent.run(prompt)

# Convert the response to a string.
return response.__str__()
```

```bash
streamlit run interface.py
```
- **What it demonstrates**: setting up a dedicated Conda environment and installing LangChain/Streamlit dependencies, securely loading the OpenAI key via `environ`, the exact JSON-contract prompt that drives chart-type selection, and launching the finished Streamlit app.

## Reference Tables
| Agent JSON response key | Streamlit rendering |
|---|---|
| `answer` | `st.write` (plain text) |
| `bar` | Convert `data`/`columns` to DataFrame → `st.bar_chart` |
| `line` | Convert `data`/`columns` to DataFrame → `st.line_chart` |
| `table` | Convert nested `data`/`columns` to DataFrame → `st.table` |

## Worked Example
The chapter builds "Chat with your CSV": (1) set up `langchainenv` Conda environment and install `langchain openai streamlit python-environ tabulate`; (2) store the OpenAI key in `.env` as `apikey=...`; (3) in `agent.py`, define `create_agent(path)` which builds an `OpenAI` LLM object, reads the CSV into a pandas DataFrame, and returns a `create_pandas_dataframe_agent`; (4) define `query_agent(agent, query)` which builds the three-way JSON-contract prompt (bar/line/answer) around the user's query and calls `agent.run(prompt)`, returning the stringified response; (5) in `interface.py`, define `decode_response` (string→dict) and `write_response` (dict key → matching Streamlit widget); (6) assemble the Streamlit page: `st.title("Chat with your CSV")`, a `st.file_uploader` for the CSV, a `st.text_area` for the query, and a "Submit Query" button that chains `create_agent` → `query_agent` → `decode_response` → `write_response`; (7) run via `streamlit run interface.py`. Testing against a fake bank-customer CSV: "how many records" correctly returns "There are 10000 rows in the file"; an out-of-scope question ("What is the price of Bitcoin") correctly triggers "I do not have sufficient information to provide an answer to your query"; a tabular request (average balances of female customers by country) correctly renders as a table, though a follow-up rounding instruction was inconsistently honored.

## Key Takeaways
1. A structured, JSON-contract prompt (bar/line/table/answer) lets an LLM's output double as a UI-routing instruction, enabling automatic chart-type selection.
2. LangChain's `create_pandas_dataframe_agent` is the right tool when the data source is a flat CSV/DataFrame rather than a document corpus needing vector-store retrieval.
3. Explicitly instruct the model to refuse out-of-scope or unknown queries ("I do not have sufficient information...") to prevent hallucinated answers on tabular data.
4. Watch token limits with large CSVs — use smaller datasets or be selective about how much data a single query pulls into context.
5. Numeric formatting instructions (e.g., rounding) are not reliably honored every run — validate output rather than assuming consistency.

## Connects To
- **Ch6**: Reuses the "never hard-code API keys, use an environment file" practice established there.
- **Ch5/Ch7**: Reuses the chunking/vector-store concept conceptually, though this chapter's actual implementation uses a simpler DataFrame agent instead of full RAG.
- **Ch8**: A parallel "quick Python UI for an LLM-backed task" chapter, contrasting Gradio (ML demo interfaces) with Streamlit (data dashboards) for different use cases.
