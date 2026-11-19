# Chapter 3: Text Processing and Embeddings Fundamentals

## Core Idea
Text chunking and text embeddings are the two foundational preprocessing steps that convert raw text into structured, semantically meaningful numeric vectors that LLMs and downstream NLP systems can consume efficiently.

## Frameworks Introduced
- **Text chunking taxonomy**: shallow chunking (basic NP/VP/PP labeling) vs. deep chunking (nested hierarchical chunk trees); performed via rule-based or statistical/ML methods.
  - When to use: shallow for quick syntactic grouping; deep for detailed hierarchical topics.
- **Four chunking strategies**: fixed-size, sentence splitting, recursive, hybrid.
  - Fixed-size: simplest, splits text into equal-length (e.g., 512/1024 token) chunks; fast but ignores natural boundaries.
  - Sentence splitting: splits on sentence boundaries up to a max token budget; good for sentence-level meaning tasks.
  - Recursive: recursively halves text until each piece is under a max-token threshold; produces hierarchical structure, good for detailed/long documents.
  - Hybrid: combines fixed-size efficiency with rule-based boundary awareness.
  - How to choose: rule-based/hybrid for high precision; fixed-size/sentence-splitting for large-scale or real-time processing. Chunk overlap trades context-capture accuracy against computational cost.
- **Text embedding method family**: TF-IDF, Word2vec (CBOW/Skip-gram), Doc2vec, FastText, GloVe, ELMo, BERT, Universal Sentence Encoder (USE).
  - When to use: context-aware embeddings (BERT, USE) for sentiment/QA; word-level (word2vec, GloVe) for translation/summarization; FastText for rare/out-of-vocabulary words; USE/CLIP for multimodal/sentence-level tasks.
- **Embedding-selection checklist**: task-specificity, dimensionality (100-1000 for word, 512-1024 for subword embeddings), data size/quality, compute budget, domain-specificity, multimodal needs, benchmark consultation (MTEB), evaluation/experimentation.
- **Secure embedding pipeline**: data pre-processing (PII anonymization, encryption, tokenization) → embedding method selection (local embedding, differential privacy, federated learning) → model training (access control, secure training, audit logging) → storage/deployment (access control + encryption, limited deployment scope, compliance with GDPR/HIPAA).

## Key Concepts
- **Text chunking**: dividing text into syntactically/semantically meaningful segments before embedding.
- **Text embedding**: mapping text to dense numeric vectors capturing semantic meaning; similar meanings → nearby vectors.
- **Cosine similarity**: metric (max score 1) used to compare two embedding vectors for semantic similarity.
- **Tokenization**: splitting text into words/subwords as embedding-model input.
- **Transfer learning (via embeddings)**: reusing pre-trained embeddings (word2vec, GloVe, BERT) instead of training from scratch.
- **MTEB (Massive Text Embedding Benchmark)**: standard benchmark for comparing embedding models across tasks.
- **Differential privacy / federated learning**: privacy-preserving techniques for embedding sensitive data without centralizing it.

## Mental Models
- Think of chunking as "giving the embedding model well-formed sentences to read" — the better the chunk boundaries match semantic units, the better the resulting vector captures meaning.
- Use the overlap-vs-efficiency trade-off as a dial: more overlap = better cross-boundary context but higher compute cost.
- Treat embedding selection as an experimentation loop (choose → evaluate → adjust), not a one-time decision — benchmark against MTEB and your own task data.
- Layered defense model for data security: anonymize → encrypt → restrict access → audit → comply with regulation, at every stage from preprocessing to deployment.

## Anti-patterns
- **Feeding raw unchunked long documents directly into embedding models**: inefficient and computationally expensive, and misses natural semantic boundaries.
- **Using word-level embeddings (word2vec/GloVe) for context-sensitive tasks**: they do not disambiguate polysemous words the way contextual embeddings (BERT, USE) do.
- **Skipping PII anonymization/encryption before embedding**: risks leaking personally identifiable information through stored embeddings, violating GDPR/HIPAA.
- **Treating embedding selection as a one-time choice**: performance should be continuously monitored and the embedding approach adjusted as data or task needs evolve.

## Code Examples
```python
# Fixed-size chunking
def fixed_size_chunking(text, chunk_size=45):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
chunks_fixed = fixed_size_chunking(text)
print(chunks_fixed[0])
```
- **What it demonstrates**: simplest chunking strategy — splits text into equal-length substrings regardless of word/sentence boundaries.

```python
# Sentence splitting
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def sentence_splitting(text, max_tokens=30):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence.split()) <= max_tokens:
            current_chunk.append(sentence)
            current_length += len(sentence.split())
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence.split())
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
chunks_sentence = sentence_splitting(text)
```
- **What it demonstrates**: groups whole sentences into chunks up to a token budget, preserving sentence-level meaning.

```python
# Recursive chunking
def recursive_chunking(text, max_tokens=30):
    if len(text.split()) <= max_tokens:
        return [text]
    else:
        midpoint = len(text) // 2
        first_half = text[:midpoint]
        second_half = text[midpoint:]
        return recursive_chunking(first_half, max_tokens) + recursive_chunking(second_half, max_tokens)
chunks_recursive = recursive_chunking(text)
```
- **What it demonstrates**: recursively halves text until each piece is below the token threshold, producing a hierarchical split.

```python
from transformers import AutoTokenizer, AutoModel

# Select pre-trained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to create the text embeddings
def get_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt")
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    tensor = embeddings.detach().numpy()
    nsamples, nx, ny = tensor.shape
    tensor = tensor.reshape((nsamples, nx*ny))
    one_D_embed = np.reshape(tensor, (np.product(tensor.shape),))
    return one_D_embed
```
- **What it demonstrates**: generating BERT embeddings for arbitrary text and flattening the last hidden state into a 1-D vector.

```python
text1 = "The Transformer can be trained significantly faster for translation tasks."
text2 = "The Transformer architecture helps it to excel at translation tasks."
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

tokens1 = tokenizer(text1, return_tensors="pt")
tokens2 = tokenizer(text2, return_tensors="pt")
outputs1 = model(**tokens1)
embeddings1 = outputs1.last_hidden_state
outputs2 = model(**tokens2)
embeddings2 = outputs2.last_hidden_state

tensor1 = embeddings1.detach().numpy()
tensor2 = embeddings2.detach().numpy()
nsamples, nx, ny = tensor1.shape
tensor11 = tensor1.reshape((nsamples, nx*ny))
nsamples, nx, ny = tensor2.shape
tensor22 = tensor2.reshape((nsamples, nx*ny))

similarity_score = cosine_similarity(tensor11, tensor22)
print("Similarity Score between two sentences:", similarity_score[0][0])
```
- **What it demonstrates**: computing cosine similarity (0.78 in the book's example) between two sentence embeddings to quantify semantic similarity.

## Reference Tables
None (this chapter uses figures/screenshots rather than tables).

## Worked Example
BERT-base-uncased is loaded via Hugging Face `transformers`, embeddings are generated for two similar sentences about Transformers, reshaped to 2D, and compared with cosine similarity — yielding a 0.78 similarity score, illustrating that overlapping vocabulary/semantics produces a high (but not perfect) similarity score. The chapter's Appendix notes paid embedding alternatives (OpenAI, Anthropic, Cohere) accessed via API key as an alternative to open-source embeddings like BERT.

## Key Takeaways
1. Chunking method choice (fixed-size, sentence, recursive, hybrid) should match the task's precision vs. throughput requirements.
2. Embeddings convert text into dense vectors enabling similarity, classification, retrieval, and generation tasks — pre-trained embeddings enable transfer learning instead of training from scratch.
3. Selecting embeddings is a multi-factor, iterative decision (task type, dimensionality, data size, compute, domain, multimodality) validated against benchmarks like MTEB.
4. Secure embedding requires a layered approach across preprocessing, model training, and storage/deployment — not just encryption at one stage.
5. Open-source embeddings (BERT, word2vec, GloVe, FastText, USE) are viable alternatives to paid APIs (OpenAI, Anthropic, Cohere) for most tasks.

## Connects To
- **Ch2**: Extends the `HuggingFaceInstructEmbeddings` and chunking (`RecursiveCharacterTextSplitter`) used informally in the Ch2 chat-with-PDF app into formal, named techniques.
- **Ch4**: The embeddings generated here are exactly what gets stored and queried in vector databases, covered next.
- **Ch5**: LangChain and LlamaIndex (mentioned here as providing out-of-the-box chunking) are covered in depth in Chapter 5.
