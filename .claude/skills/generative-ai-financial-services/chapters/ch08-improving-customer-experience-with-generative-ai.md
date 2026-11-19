# Chapter 8: Use-case: Improving Customer Experience with Generative AI

## Core Idea
A working RAG-based "Insurance CoPilot" can be built from five standard components (document indexing, chat UI, orchestration, retrieval, response) using LangChain + ChromaDB + Bedrock-hosted Llama 2, and its quality can be iteratively improved via five advanced RAG techniques (embedding dimensions, chunking strategy, metadata filters, query transformation, re-ranking).

## Frameworks Introduced
- **5-component RAG architecture** (expands Ch3/Ch7): Document processing & indexing layer → Chat UI → Orchestration layer → Retrieval layer → Response layer.
  - How: Indexing = extract text (PDF/OCR tools) → chunk (fixed-length+overlap or other strategy) → embed each chunk → store chunk+embedding in vector DB. Query time = orchestration receives chat request → retrieval layer embeds the query, searches vector DB for top-k closest chunks → response layer builds a prompt from chunks + question → calls LLM → returns answer via orchestration to UI.
- **Advanced RAG improvement techniques** (5 named techniques):
  - **Optimizing vector embedding dimensions**: larger dimensions (e.g., 256→768) capture more nuance — useful for credit risk profiling with subtle financial behavior differences.
  - **Refining chunking strategies**: semantic/logical chunking (e.g., by legal clause type: definitions, terms, liability) instead of fixed-length, to keep related information together.
  - **Leveraging metadata filters**: weight/prioritize chunks by attributes (e.g., recency + source authority for financial news summarization).
  - **Query transformation**: rephrase/expand/augment the input query with domain jargon synonyms (e.g., "high-yield savings" → CDs, money market accounts, treasury bills) to retrieve more relevant chunks.
  - **Re-ranking**: apply a secondary model to reorder initially retrieved results by use-case-specific factors (e.g., client risk tolerance, investment goals, tax situation for wealth management advice).
  - When to use: not all five apply to every RAG application — the book recommends experimentation/iteration to determine which technique(s) help a given use-case.

## Key Concepts
- **Document indexing layer**: extracts text → chunks → embeds → stores in vector DB; the foundational RAG ingestion pipeline.
- **Orchestration layer**: receives chat requests, calls retrieval then response layers in sequence; the "glue" component of a RAG app.
- **Retrieval layer**: embeds the incoming query with the same embedding model used at indexing time, performs similarity search for top-k chunks.
- **Response layer**: constructs the final prompt from retrieved chunks + question, calls the LLM, returns the answer.
- **ChromaDB**: local open-source vector database used in this chapter's example (alternatives: OpenSearch, pgVector, Pinecone, Weaviate).
- **Sentence Transformers (all-MiniLM-L6-v2)**: the Hugging Face embedding model used in this chapter's example.

## Mental Models
- Treat RAG quality improvement as an experimentation loop across five independent levers (embedding dimension, chunking, metadata filters, query transformation, re-ranking) — tune one at a time and measure against a human-labeled evaluation set.
- Use source-chunk display alongside the generated answer as a cheap, built-in hallucination check: if the answer's claims aren't traceable to the retrieved chunk text, treat the response with suspicion.
- Separate "basic RAG works" from "production RAG works" — production needs testing/evaluation frameworks, scale planning (thousands to millions of documents), latency guarantees (live call setting), multi-modal extraction (tables/images/graphs), and multi-document/multi-section question answering.

## Anti-patterns
- **Trusting LLM output without checking source chunks**: the chapter explicitly recommends displaying the retrieved source chunk alongside the answer to verify the answer is grounded in the document, not hallucinated.
- **Using a single fixed chunking strategy regardless of document structure**: fixed-length chunking can split logically related content (e.g., a legal clause) across chunks, hurting retrieval quality — semantic/structural chunking should be considered per use-case.
- **Ignoring latency requirements in live customer-facing settings**: a contact-center copilot answering live calls must minimize response latency, or both agent and customer suffer.
- **Assuming text extraction suffices for financial documents**: important information in tables, images, or graphs is lost if the document pipeline only extracts plain text.

## Code Examples
```python
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_community.llms import Bedrock
from langchain.embeddings import HuggingFaceEmbeddings
import os
import boto3

DEFAULT_MODEL_ID = "meta.llama2-13b-chat-v1"

def get_llm(model_id=DEFAULT_MODEL_ID, aws_region="us-east-1"):
    """Connect with Bedrock API for accessing LLM"""
    bedrock = boto3.client(service_name='bedrock-runtime', region_name=aws_region)
    llm = Bedrock(
        model_id=model_id,
        region_name=aws_region,
        client=bedrock
    )
    return llm

def get_embedding_model():
    """Connect with Embedding model"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def index_docs(file_path, embeddings, chunk_size=1000):
    # Loading text from local files
    isDirectory = os.path.isdir(file_path)
    if isDirectory:
        loader = DirectoryLoader(
                    file_path,
                    loader_cls=PyPDFLoader,
                    show_progress=True)
    else:
        loader = PyPDFLoader(file_path, encoding='utf8')
        print("Loading documents from path:", file_path)
    documents = loader.load()
    print("Splitting documents. chunk_size=", chunk_size)
    # For splitting by tokens, use TokenTextSplitter class
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print("Calculating embedding and storing in vector db")
    db = Chroma.from_documents(texts, embeddings)
    return db

CHUNK_SIZE = 1000
db = index_docs(
    file_path="files/",
    embeddings=get_embedding_model(),
    chunk_size=CHUNK_SIZE,
)

def search_docs(user_query, db, k=2):
    docs = db.similarity_search_with_score(user_query)[:k]
    docs = [{"content": x[0].page_content, "source": x[0].metadata["source"]} for x in docs]
    return docs

QNA_PROMPT = """
\n\nHuman: You are a financial AI, an artificial intelligence developed to answer questions about insurance policies.
Use the following documents and the information contained therein to answer the following question and provide relevant information: "{question}".
The text of the document is within the <text></text> XML tags: {documents}
Use these documents to formulate your own answer to the question "{question}", as if you were directly answering the question. Ensure that your answer is correct and does not contain any information that cannot be directly taken from the documents. Do not directly cite the document or metadata.
Assistant:"""
HUMAN_PROMPT = "Human:"

def ask_llm(user_query, docs, llm):
    qna_prompt = QNA_PROMPT.format(question=user_query, documents=docs)
    answer = llm.invoke(input=qna_prompt, stop=[HUMAN_PROMPT])
    return answer

# Usage
user_query = "What type of accidental breakage is covered in Home Insurance?"
docs = search_docs(user_query, db)
llm = get_llm()
answer = ask_llm(user_query, docs, llm)
print("Answer: ", answer)

# Hallucination check: inspect retrieved source chunks
for doc in docs:
    print("# Source: " + doc["source"].split("/")[-1])
    print(doc["content"][:2000].replace("#", "") + "...")
```
- **What it demonstrates**: a complete, minimal end-to-end RAG pipeline — PDF ingestion/chunking/embedding into ChromaDB, similarity-search retrieval, prompt construction with explicit anti-hallucination instructions ("does not contain any information that cannot be directly taken from the documents"), and a manual grounding check by inspecting the retrieved source chunks.

## Reference Tables
**5 advanced RAG techniques and their financial-services example**

| Technique | Mechanism | Financial example |
|---|---|---|
| Optimize embedding dimensions | Larger vectors capture more nuance | Credit risk assessment: 256→768 dims improves subtle behavior/history profiling |
| Refine chunking strategy | Semantic/logical vs. fixed-length chunking | Legal document analysis: chunk by clause type (definitions, terms, liability) |
| Leverage metadata filters | Weight by recency/authority/other attributes | Financial news summarization: prioritize last-24h, recognized-agency sources |
| Query transformation | Expand query with domain synonyms/jargon | Personal finance advisory: "high-yield savings" → CDs, money market, T-bills |
| Re-ranking | Secondary model reorders initial results | Wealth management: re-rank by client risk tolerance, goals, tax situation |

**Production RAG considerations beyond the basic flow**: testing/evaluation framework (vs. human-labeled answers), scale (thousands-to-millions of documents), latency (live call constraint), multi-modal extraction (tables/images/graphs), multi-document/multi-section question answering.

## Worked Example
**Insurance CoPilot end-to-end walkthrough**: (1) A sample AXA home insurance policy PDF is indexed via `index_docs()` using TokenTextSplitter (chunk_size=1000, overlap=50) and Sentence Transformers embeddings into ChromaDB. (2) A contact-center agent (via chat UI) asks "What type of accidental breakage is covered in Home Insurance?" (3) `search_docs()` retrieves the top-2 closest chunks by similarity score. (4) `ask_llm()` builds the QNA_PROMPT (explicitly instructing the LLM to answer only from the provided document text, not cite metadata) and calls Llama 2 13B Chat via Amazon Bedrock. (5) The model answers citing accidental damage to fixed glass, ceramic hobs, sanitary ware, etc., with named exclusions (frost, insects, vermin, gates/hedges/fences). (6) The agent (or developer) can verify groundedness by printing the retrieved source chunk text alongside the answer.

## Key Takeaways
1. A minimal RAG pipeline (index → embed → store → retrieve → prompt → generate) can be built with open-source tools (LangChain, ChromaDB, HuggingFace embeddings) and a cloud-hosted LLM (Bedrock/Llama 2) in a few dozen lines of code.
2. Always pair the generated answer with its source chunk for a built-in hallucination sanity check.
3. The five advanced RAG techniques (embedding dimension, chunking, metadata filters, query transformation, re-ranking) are independent levers — apply them selectively based on experimentation, not all at once.
4. Production RAG needs testing/evaluation, scale, latency, multi-modal extraction, and cross-document reasoning considerations that a basic prototype does not address.
5. The prompt itself should explicitly instruct the model to answer only from provided context and not fabricate — a cheap, high-leverage reliability control.

## Connects To
- **Ch3**: Implements the RAG data-ingestion and response-generation workflows defined conceptually there.
- **Ch4**: Implements the contact-center customer-experience use-case introduced there.
- **Ch7**: Follows the RAG architectural component breakdown (web interface/prompt/model/storage/database) from the architecture chapter.
- **Ch9**: Continues the contact-center theme with post-call automation/insights.
- **Ch13**: The hallucination-check pattern here is a precursor to the formal trustworthiness evaluation techniques covered later.
