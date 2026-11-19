# Chapter 2: Overview of Foundational Models

## Core Idea
Foundational models (FMs) are large models pre-trained via self-supervised learning on massive unlabeled datasets; they can be used directly, fine-tuned, or customized via RAG/prompt engineering, and this chapter builds the first end-to-end open-source LLM application (chat-with-PDF using Falcon-7B, LangChain, and FAISS).

## Frameworks Introduced
- **Self-supervised learning**: converts an unsupervised problem into a supervised one by auto-generating labels from the data itself (no external annotation).
  - When to use: pre-training FMs on massive unlabeled corpora.
- **FM customization decision framework — RAG vs. Prompt Engineering vs. Fine-tuning**:
  - When to use RAG: need to ground answers in an external knowledge source without retraining; retrieval model fetches relevant passages, generation model conditions on them.
  - When to use Prompt Engineering: limited training data, need low-cost/no-parameter-change steering via clear instructions + examples + domain knowledge.
  - When to use Fine-tuning: have domain-specific training data and computational resources/ML expertise to adjust model parameters for best performance.
  - How to choose: start with prompt engineering or RAG when data/compute is limited; graduate to fine-tuning as data and resources become available.
- **Domain-customization pipeline (finance/manufacturing/customer experience)**: (1) data collection, (2) data preprocessing (tokenization, stemming, lemmatization), (3) RAG/fine-tuning, (4) deployment/integration (APIs, UI, CLI), (5) continuous monitoring and retraining.
- **Chat-with-PDF implementation pattern**: ingest → chunk → embed → store in vector DB → load LLM → retrieve + generate via QA chain.
  - How: `PyPDFLoader` loads PDF → `RecursiveCharacterTextSplitter` chunks text → `HuggingFaceInstructEmbeddings` embeds chunks → `FAISS.from_documents` stores vectors → `RetrievalQA` chain combines retriever + LLM for Q&A.

## Key Concepts
- **Foundational model (FM) / base model**: large model pre-trained on massive unlabeled data (text, image, audio, code), fine-tunable for downstream tasks.
- **RAG (Retrieval-Augmented Generation)**: retrieval model finds relevant passages; generation model uses them as context.
- **Fine-tuning**: adjusts model parameters using task/domain-specific additional training data.
- **Prompt engineering**: crafts input prompts to steer output without modifying model parameters.
- **Virtual environment**: isolated Python dependency environment (via Anaconda `conda`) for reproducible LLM app development.
- **CUDA**: NVIDIA's parallel-computing architecture, used to accelerate model inference/embedding on GPU vs. CPU.
- **Instructor embeddings**: instruction-tuned sentence embedding model family (`hkunlp/instructor-base`) used via `HuggingFaceInstructEmbeddings`.

## Mental Models
- Treat RAG, prompt engineering, and fine-tuning as a cost/data spectrum: prompt engineering (cheapest, least data) → RAG (external knowledge, no retraining) → fine-tuning (most data/compute, most specialized).
- Think of FM customization pipelines (finance, manufacturing, CX) as the same five-step template applied to different data sources — the process is domain-agnostic even though the data isn't.

## Anti-patterns
- **Skipping the C++ build dependency check**: `llama-cpp-python` and similar packages can fail to install without C++ build tools — verify toolchain before troubleshooting Python errors.
- **Choosing fine-tuning by default**: fine-tuning requires more compute/expertise than RAG or prompt engineering — don't reach for it when training data is scarce or a retraining pipeline isn't in place.
- **Reusing another user's conda virtual environment**: environments belong to specific user profiles on shared machines and should be recreated per user.

## Code Examples
```
Requirements.txt
Pandas==2.1.2
PyPDF2
Transformers==4.34.0
langchain==0.0.309
llama-cpp-python==0.2.0
xformers==0.0.21
urllib3==1.26.6
InstructorEmbedding==1.0.1
sentence-transformers==2.2.2
faiss-cpu==1.7.4
huggingface_hub==0.16.4
protobuf==3.20.0
accelerate==0.21.0
bitsandbytes==0.41.1
click==8.1.6
einops==0.7.0
charset-normalizer==2.0.4
epub2txt==0.1.6
```
- **What it demonstrates**: pinned dependency set for an open-source LLM chat-with-PDF stack (LangChain, Transformers, FAISS, Instructor embeddings).

```python
# Ingest.py
import click
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings

def load_documents():
    # To Load the PDF document from source documents directory
    loader = PyPDFLoader('SOURCE_DOCUMENTS/Attention.pdf')
    docs = loader.load()
    return docs

@click.command()
@click.option('--device_type', default='cuda', help='select gpu or cpu for execution')
def main(device_type, ):
    if device_type in ['cpu', 'CPU']:
        device = 'cpu'
    else:
        device = 'cuda'
    # Load documents and split in chunks
    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                       model_kwargs={"device": device})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local('faiss_index')

if __name__ == "__main__":
    main()
```
- **What it demonstrates**: ingestion pipeline — load PDF, chunk with overlap, embed with instructor embeddings, persist to a local FAISS index.

```python
# run_LLM.py
import torch
import click
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_model(device):
    model = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        trust_remote_code=True,
        device_map=device if device == "cpu" else "auto",
        max_length=2048,
        temperature=0,
        top_p=0.90,
        top_k=10,
        repetition_penalty=1.15,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)

@click.command()
@click.option('--device_type', default='cuda', help='select gpu or cpu for execution')
def main(device_type, ):
    device = 'cpu' if device_type in ['cpu', 'CPU'] else 'cuda'
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                       model_kwargs={"device": device})
    database = FAISS.load_local('faiss_index', embeddings)
    retriever = database.as_retriever()
    llm = load_model(device)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                            return_source_documents=True)
    while True:
        query = input("\nEnter the query: ")
        if query == "exit":
            break
        result = qa_chain(query)
        answer, docs = result['result'], result['source_documents']
        print(answer)

if __name__ == "__main__":
    main()
```
- **What it demonstrates**: loading an open-source LLM (Falcon-7B-Instruct) via Hugging Face `transformers`, wrapping it as a LangChain `HuggingFacePipeline`, and building a `RetrievalQA` chain ("stuff" chain type) over a FAISS vector store for interactive PDF Q&A.

## Reference Tables
None in this chapter.

## Worked Example
End-to-end "chat with a PDF" CLI app: set up an Anaconda virtual env named `LLM` (Python 3.11) → install pinned requirements.txt → ingest the "Attention Is All You Need" PDF via `ingest.py` (chunk_size=900, overlap=100, instructor-base embeddings, saved to local FAISS index) → run `run_LLM.py` to download/load Falcon-7B-Instruct, reload the FAISS index, and answer interactive questions (e.g., "how does a transformer work efficiently?") with sourced answers.

## Key Takeaways
1. FMs are self-supervised pre-trained models that must be customized (RAG, fine-tuning, or prompting) to become useful for a specific task or domain.
2. Choose customization strategy by data/compute availability: prompt engineering/RAG first, fine-tuning once sufficient data and resources exist.
3. A minimal open-source LLM app needs four pieces: a document loader, a text splitter, an embedding model, and a vector store — plus an LLM wrapped for LangChain's chain interfaces.
4. GPU (CUDA) vs. CPU is a runtime choice exposed via a `--device_type` flag; GPU is faster but CPU is a valid fallback.
5. Domain customization (finance, manufacturing, CX) follows the same 5-step pipeline: collect → preprocess → RAG/fine-tune → deploy → monitor.

## Connects To
- **Ch3**: The chunking (`RecursiveCharacterTextSplitter`) and embedding (`HuggingFaceInstructEmbeddings`) steps here are explored in depth as text-processing and embeddings fundamentals.
- **Ch4**: The FAISS vector store used here for retrieval is generalized into a full vector-database discussion.
- **Ch5**: The `RetrievalQA` chain previews LangChain's broader chain-composition patterns covered later.
- **Ch7**: Falcon-7B loading via `transformers`/Hugging Face here is expanded into a dedicated Hugging Face implementation chapter.
