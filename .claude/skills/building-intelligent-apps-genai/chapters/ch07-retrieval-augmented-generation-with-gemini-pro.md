# Chapter 7: Retrieval Augmented Generation with Gemini Pro

## Core Idea
A working RAG pipeline is built by chunking a document, embedding the chunks into a vector database (Chroma), wiring a LangChain `RetrievalQA` chain to Gemini Pro, and adding a prompt template that explicitly instructs the model to say "I don't know" rather than hallucinate when the retrieved context doesn't answer the question.

## Frameworks Introduced
- **RAG two-phase pipeline (retrieval + generation)**: retrieval phase fetches relevant chunks from an authoritative knowledge base (PDF/Excel/CSV); generation phase uses that retrieved knowledge to produce a grounded response.
  - When to use: whenever an LLM needs access to current, authoritative, or proprietary data without retraining the model.
  - How: user query → embed query → similarity search in vector DB → retrieved chunks + query → LLM → grounded answer.
- **LangChain + Gemini Pro RAG stack**: `ChatGoogleGenerativeAI` (Gemini Pro) + `GoogleGenerativeAIEmbeddings` + `RecursiveCharacterTextSplitter` + Chroma vector store + `RetrievalQA` chain.
  - When to use: this is the book's concrete, named implementation pattern for building a document Q&A app on Google's stack (Gemini-Pro-based RAG pattern).
  - How: split document into chunks → embed chunks → store/index in Chroma → build `RetrievalQA` → query it with natural language.
- **Anti-hallucination prompt template**: explicitly instruct the model via a `PromptTemplate` to answer only from provided `{context}`, and to say "I don't know" rather than fabricate an answer, appending a courteous closing line ("Thanks for asking!").
  - When to use: any RAG deployment where ungrounded hallucination is unacceptable (e.g., enterprise/customer-facing use).

## Key Concepts
- **Vector embedding (building analogy)**: "secret addresses in a high-rise building" — words with similar meaning (happy/joyful) have nearby addresses; opposites (hot/cold) are on different floors.
- **Similarity search**: comparing the query's vector embedding against stored document-chunk embeddings to retrieve the most relevant chunks.
- **Chroma**: the vector database used in this chapter to store and retrieve document embeddings.
- **RecursiveCharacterTextSplitter**: LangChain utility that splits large documents into fixed-size overlapping chunks (book uses 10,000 characters with 1,000 character overlap) so they fit model context limits.
- **RetrievalQA**: a LangChain chain type that combines a retriever (vector DB) and an LLM into a single question-answering pipeline.
- **Temperature**: a model parameter controlling response creativity/randomness; set to 0.2 in this chapter to make Gemini Pro's answers less creative and more deterministic/factual.
- **Google AI Studio / MakerSuite**: the web tool (makersuite.google.com) for prototyping prompts and obtaining a Gemini Pro API key.

## Mental Models
- Think of RAG as "a tireless researcher" who searches enormous libraries and finds exactly what you need, then answers your question using what they found — not from memory alone.
- Think of a vector database as a "knowledge compass" guiding retrieval toward the most relevant information, not just a generic data store.
- Think of a prompt template as "a recipe card" — fixed structure with placeholders (`{context}`, `{question}`) that keeps every generated prompt consistent.

## Anti-patterns
- **Querying a general-purpose LLM about domain-specific documents without retrieval**: the book shows Gemini Pro giving an irrelevant, generic answer to "What are the three steps of career growth?" until the career-growth eBook PDF is actually retrieved and supplied as context.
- **Letting the model answer when it doesn't know**: without the anti-hallucination prompt template, a model faced with an out-of-scope question (e.g., "Can you describe random forest?" against a document that doesn't cover it) will hallucinate instead of admitting ignorance.
- **Trying to embed/process an entire large document in one pass**: ignoring chunking (RecursiveCharacterTextSplitter) risks exceeding context limits or diluting retrieval relevance.

## Code Examples
```python
!pip install -q --upgrade google-generativeai langchain-google-genai chromadb pypdf langchain
```

```python
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY,
             temperature=0.2, convert_system_message_to_human=True)
```

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
```

```python
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
```
- **What it demonstrates**: installing the full RAG toolchain, configuring Gemini Pro with low temperature for factual consistency, chunking a large PDF for embedding, and an anti-hallucination prompt template that forces honest "I don't know" responses when context is insufficient.

## Reference Tables
No formal comparison tables in this chapter; content is a sequential, numbered hands-on walkthrough (register API key → Colab setup → install libraries → configure Gemini Pro → configure LangChain → embed → build vector DB → build RetrievalQA → add prompt template).

## Worked Example
The chapter's single running example: using Andrew Ng's eBook "How to Build a Career in AI" (a PDF) as the knowledge source. Step by step: (1) register a Gemini Pro API key at makersuite.google.com; (2) create a Google Colab notebook, store the key as a Colab Secret (no `.env` file needed); (3) `pip install` google-generativeai, langchain-google-genai, chromadb, pypdf, langchain; (4) configure and sanity-test Gemini Pro directly with a generic query ("What are the use cases of LLMs?"); (5) configure LangChain's `ChatGoogleGenerativeAI` with `temperature=0.2`; (6) ask "What are the three steps of career growth?" without RAG — get a generic, unhelpful answer; (7) upload the PDF, chunk it (10,000 chars / 1,000 overlap), embed chunks with `GoogleGenerativeAIEmbeddings`, and build a Chroma vector index; (8) wire a `RetrievalQA` chain over that index and re-ask the same question — now get the correct, document-grounded answer; (9) add the anti-hallucination `PromptTemplate` and test both an in-document question (answered correctly) and an out-of-document question about "random forest" (correctly answered "I'm sorry, but this document does not contain information about Random Forest...", demonstrating the hallucination-prevention guardrail actually works).

## Key Takeaways
1. RAG lets you update what an LLM effectively "knows" by changing the retrieved documents, with zero model retraining.
2. The standard RAG build order is: chunk → embed → index (vector DB) → retrieval chain → prompt template guardrail.
3. Lower the `temperature` parameter (e.g., 0.2) for factual/retrieval-grounded tasks to reduce creative drift.
4. Always add an explicit "say I don't know" instruction in the prompt template — RAG alone doesn't guarantee the model won't still try to answer from parametric memory when retrieved context is irrelevant.
5. The stack is swappable: Gemini Pro could be replaced with Llama 2, and Chroma with Pinecone, depending on project needs (cost, control, hosting).

## Connects To
- **Ch4/Ch5**: Directly implements the vector-embedding and RAG concepts introduced conceptually in those earlier chapters.
- **Ch6**: Parallels the ChatGPT API chatbot build as another tool-specific (Gemini Pro) hands-on application.
- **Ch10**: Llama 2 is explicitly named here as a swappable open-source alternative LLM for the same RAG pattern.
- **Ch11**: Flowise AI's document chatbot chapter builds a similar RAG document-Q&A experience, but no-code.
