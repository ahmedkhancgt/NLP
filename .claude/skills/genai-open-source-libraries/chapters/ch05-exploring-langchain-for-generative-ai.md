# Chapter 5: Exploring LangChain for Generative AI

## Core Idea
LangChain is a modular, model-agnostic framework that composes LLMs, prompts, retrievers, and memory into chains (via LCEL) — turning raw LLM calls into production-ready applications (chatbots, summarizers, RAG systems) without rewriting code per LLM provider.

## Frameworks Introduced
- **LangChain component stack**: LangChain libraries (Python/JS interfaces, chains, agents) + LangChain templates (reference architectures) + LangServe (deploy chains as REST APIs) + LangSmith (debug/test/evaluate/monitor chains).
  - When to use: libraries for building; templates for fast-starting common patterns; LangServe for productionizing as an API; LangSmith for observability.
- **LangChain Expression Language (LCEL)**: declarative way to compose chains — describes *what* should happen, not *how*; supports streaming, parallelism, logging; same code scales from prototype (`prompt | model | output_parser`) to production.
  - How: `prompt = ChatPromptTemplate.from_template(...)`; `model = ChatOpenAI()`; `output_parser = StrOutputParser()`; compose and `.invoke({...})`.
- **Prompt → Model → Output Parser pipeline**: `BasePromptTemplate` takes a dict of variables → produces a `PromptValue` → passed to model (`ChatModel`/LLM/tools/agents) → returns `BaseMessage` → passed to `BaseOutputParser` (e.g., `StrOutputParser`) → final string.
- **Retrieval QA chain pattern**: `RetrievalQA.from_chain_type(llm=..., chain_type='stuff', retriever=vectordb.as_retriever(search_kwargs={'k': 3}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})`.
  - When to use: answering questions grounded in a vector-store-indexed document set; `k` controls how many top-matching chunks are retrieved as context.
- **Conversational retrieval chain**: uses a `CONDENSE_QUESTION_PROMPT` to rephrase a follow-up question (given chat history) into a standalone question before retrieval — preserves multi-turn context.
- **Map-Reduce summarization pattern**: `MapReduceDocumentsChain` (map step: summarize each chunk individually; reduce step: `ReduceDocumentsChain` + `StuffDocumentsChain` combine chunk summaries into one final summary) — handles documents too large for a single context window.
  - When to use: summarizing long documents/articles that exceed the model's context length; enables parallel per-chunk processing.

## Key Concepts
- **Chain**: LangChain's core concept — sequencing LLM tasks/components (LLM chain, router chain, sequential chain, transformation chain, etc.) into a pipeline.
- **PromptTemplate**: predefined recipe (instructions + few-shot examples + context/question placeholders) for generating a model prompt, reusable across LLMs.
- **Retriever**: component that fetches relevant documents from an external source (vector store, database) given a query string; outputs a list of `Document` objects.
- **Memory**: LangChain component that stores conversation history to maintain context across turns.
- **StuffDocumentsChain**: concatenates multiple documents into one string and passes it to the LLM for summarization/analysis.
- **MapReduceDocumentsChain**: applies a chain per-document (map), then combines results (reduce) — for scaling beyond single-context-window limits.
- **Model-agnostic**: LangChain's design lets you swap LLM providers (OpenAI, Cohere, Hugging Face, Anthropic, Bedrock) without rewriting application logic.

## Mental Models
- Think of LangChain as "plumbing" connecting three things: a prompt template, a model endpoint, and an output parser — everything else (retrievers, memory, chains) is built on top of this basic flow.
- Use LCEL's declarative style as the difference between describing *the pipeline's shape* (what connects to what) vs. imperative step-by-step code — this is what lets the same chain scale from prototype to production.
- Map-Reduce for documents mirrors the classic distributed-computing pattern: split work (map: per-chunk summary) → combine (reduce: consolidated summary) — useful whenever a task exceeds a single model call's capacity.

## Anti-patterns
- **Treating LangChain as fully open-source/transparent**: some core backend components are not fully open-source, which can limit transparency and deep customization.
- **Building deeply nested chains without modularity discipline**: "building intricate chains can become daunting" — debugging multi-step chains is a known complexity/learning-curve cost; plan and modularize deliberately.
- **Ignoring context-length limits when summarizing large documents with a single `StuffDocumentsChain`**: will fail or truncate; use `MapReduceDocumentsChain` (map + reduce) instead to chunk and combine.
- **Hardcoding a single LLM provider into application logic**: defeats LangChain's model-agnostic design — use its standardized interfaces to keep provider swaps cheap.

## Code Examples
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Write 2 sentences about this {topic}?")
model = ChatOpenAI()
output_parser = StrOutputParser()
chain = prompt | model | output_parser
chain.invoke({"topic": "Artificial Intelligence"})
```
- **What it demonstrates**: the basic LCEL chain composition — prompt | model | output_parser — requires `OPENAI_API_KEY` env variable.

```python
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompt_template import qa_template

def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template, input_variables=['context', 'question'])
    return prompt

def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return dbqa
```
- **What it demonstrates**: building a retrieval-QA chain over a FAISS vector store with a custom prompt template, retrieving the top-3 (`k=3`) most relevant chunks.

```python
from langchain.prompts.prompt import PromptTemplate
template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question in its original language.
Chat history:
{chat_history}
Follow up input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)
```
- **What it demonstrates**: conversational retrieval — rephrasing a follow-up question into a standalone one using chat history, so retrieval stays accurate across turns.

```python
# chain.py — Map-Reduce summarization with Mistral-7B (CTransformers) + LangChain
import os, time
from langchain.chains import MapReduceDocumentsChain, LLMChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.document_loaders import NewsURLLoader
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

def summarizer(article_url):
    loader = NewsURLLoader([article_url])
    docs = loader.load()
    config = {'max_new_tokens': 4096, 'temperature': 0.3, 'context_length': 4096}
    llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                         model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                         config=config, threads=os.cpu_count())

    map_template = """<s>[INST] The following is a part of an article from a website:
    {docs}
    Based on this, please identify the main theme and key points of it
    Answer:  [/INST] </s>"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """<s>[INST] The following is set of summaries from the article:
    {doc_summaries}
    Take these as a input and condense it into a final, consolidated summary of all the main points.
    Construct it as a well organized summary of the main points and it should be between 2 and 6 paragraphs.
    Answer:  [/INST] </s>"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=True,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    start_time = time.time()
    result = map_reduce_chain.__call__(split_docs, return_only_outputs=True)
    time_taken = time.time() - start_time
    return result['output_text'], time_taken
```
- **What it demonstrates**: a full open-source article-summarizer backend — loads a news URL, chunks the article, runs a map (per-chunk key-point extraction) then reduce (consolidated summary) chain over a locally-run Mistral-7B GGUF model via `CTransformers`.

```python
# frontend.py — Streamlit UI calling the summarizer chain
import streamlit as st
from chain import summarizer

def main():
    st.set_page_config(page_title="Summarizer App", page_icon="📜", layout="wide")
    st.title("Summarizer", anchor=False)
    st.header("Summarize Articles using Mistral", anchor=False)
    url = st.text_input("Enter URL of any article", value="")
    if url:
        with st.status("Processing...", state="running", expanded=True) as status:
            st.write("Summarizing Article...")
            summary, time_taken = summarizer(url)
            status.update(label=f"Finished - Time Taken: {time_taken} seconds", state="complete")
        st.subheader("Summary:", anchor=False)
        st.write(summary)

if __name__ == "__main__":
    main()
```
- **What it demonstrates**: wiring a LangChain summarization backend to a Streamlit front end for a usable web app.

## Reference Tables
### Table 5.1: LangChain components — input/output types
| Component | Input type | Output type |
|---|---|---|
| Prompt | Dictionary | PromptValue |
| ChatModel | Single string, list of chat messages | ChatMessage |
| LLM | Single string, list of chat messages | String |
| OutputParser | The output of an LLM or ChatModel | Depends on parser |
| Retriever | Single string | List of Documents |

## Worked Example
End-to-end "Summarizer App": install `streamlit`, `langchain`, `beautifulsoup4`, `ctransformers`, `transformers`, `newspaper4k` in a conda env (`langchain`, Python 3.11) → `chain.py` loads an article via `NewsURLLoader`, splits it with `RecursiveCharacterTextSplitter` (chunk_size=4000, overlap=100), runs a Map-Reduce chain over a locally-hosted Mistral-7B-Instruct GGUF model (via `CTransformers`, CPU-threaded) to produce a 2-6 paragraph consolidated summary → `frontend.py` (Streamlit) takes a URL input, shows a "Processing..." status, and displays the final summary and time taken.

## Key Takeaways
1. LangChain's value is abstraction: one standardized interface across LLM providers (OpenAI, Cohere, Hugging Face, Anthropic, Bedrock) and data sources.
2. LCEL enables the same chain code to go from prototype to production without rewriting (supports streaming, parallelism, logging natively).
3. RetrievalQA chains (`chain_type='stuff'`, retriever with `k`) are the standard LangChain pattern for RAG-based question answering.
4. Map-Reduce chains (`MapReduceDocumentsChain` + `ReduceDocumentsChain` + `StuffDocumentsChain`) are the standard pattern for summarizing documents larger than a single context window.
5. LangChain's flexibility (modular nodes, conditional branching, custom nodes) comes with a real learning curve and potential debugging complexity in multi-step chains — plan chain architecture deliberately.

## Connects To
- **Ch2**: Expands the `RetrievalQA` pattern first shown informally in the Ch2 chat-with-PDF app into a formal LangChain chain framework.
- **Ch4**: LangChain's retriever abstraction wraps the vector databases (FAISS, Chroma) covered in Chapter 4.
- **Ch6**: The retrieval QA chain pattern shown here is built out into a full PDF-extractor LLM implementation in Chapter 6.
- **Ch7**: Hugging Face models (like Mistral via CTransformers here) are explored in depth as a dedicated implementation chapter.
