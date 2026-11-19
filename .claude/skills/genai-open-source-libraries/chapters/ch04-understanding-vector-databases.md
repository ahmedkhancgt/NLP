# Chapter 4: Understanding Vector Databases

## Core Idea
Vector databases store and index high-dimensional embeddings for fast similarity search (via ANN/KNN algorithms), and they form the retrieval half of Retrieval-Augmented Generation (RAG) — the pattern that grounds LLM responses in an external knowledge base.

## Frameworks Introduced
- **Vector database taxonomy**: indexed vector DBs (Pinecone, Milvus, Weaviate, Anyscale), graph vector DBs (Neo4j, Amazon Neptune, OrientDB), spatial DBs (PostGIS, Oracle Spatial), document DBs (MongoDB, CosmosDB), full-text search DBs (Elasticsearch), vector-capable SQL (SingleStoreDB), and standalone vector libraries (Faiss, Annoy, HNSWLib).
  - When to use: indexed vector DBs for pure similarity search/retrieval; graph vector DBs when relationships/connections matter (social networks, knowledge graphs); spatial DBs for geographic data; vector libraries (Faiss) for in-process/offline similarity search without a full DB server.
- **RAG architecture (3 steps)**: (1) create embeddings from a document corpus, (2) store vectors as indexes in a vector database, (3) search by comparing query vector to stored vectors and pass top-k retrieved content to the LLM as context.
  - How: prompt + retrieval → enriched prompt (top-k context appended) → enhanced generation (LLM answers grounded in retrieved context).
- **Similarity metric selection**: dot product (alignment magnitude), cosine similarity (angle-only, magnitude-independent — most common in NLP), Euclidean/L2 distance (straight-line distance, used by Faiss's `IndexFlatL2`).
  - When to use: cosine similarity for comparing text/document embeddings regardless of length/magnitude; Euclidean distance when absolute magnitude differences matter; dot product for raw alignment strength.
- **Indexing algorithm choice**: inverted indexes (text/keyword-oriented, map content→documents), tree-based indexes (k-d trees; efficient for low-dimensional/geospatial data), graph-based indexes (relationship/traversal-oriented).
  - Trade-off: inverted indexes are fast but weaker for high-dimensional vectors; tree/graph indexes scale better for high-dimensional data with different speed/accuracy trade-offs.
- **Vector DB selection checklist**: data type, scale, performance/latency needs, required features (anomaly detection, semantic search, metadata filtering), deployment preference (cloud vs. standalone).

## Key Concepts
- **Vector database**: database specialized for storing/retrieving/searching high-dimensional vector embeddings via similarity rather than exact match.
- **ANN (Approximate Nearest Neighbor)**: indexing technique trading exactness for speed when finding the most similar vectors at scale.
- **KNN (K-Nearest Neighbors)**: finds the k most similar data points to a query using a distance metric.
- **Faiss**: Facebook's open-source library for efficient similarity search/clustering of dense vectors, with an in-RAM index structure (`IndexFlatL2` for exact L2/Euclidean search).
- **Chroma DB**: open-source vector store with collections (analogous to DB tables), built-in embedding (default `all-MiniLM-L6-v2`), metadata storage and filtering.
- **Normalization**: scaling vectors to unit length (L2 normalization) so magnitude doesn't skew distance comparisons.
- **Collection** (Chroma): a named group of embedded documents + metadata, queried via `collection.query()`.

## Mental Models
- Think of a vector database as "a search engine for meaning" rather than for keywords — it indexes by semantic closeness, not exact string match.
- Use the RAG pipeline as: Query → Vectorize → Retrieve top-k by similarity → Inject as LLM context → Generate grounded answer.
- When picking a vector DB, map the use case to the category first (pure similarity search vs. relationship-aware vs. hybrid structured+vector), then narrow by scale/latency/deployment needs.

## Anti-patterns
- **Using a traditional relational database for high-dimensional similarity search**: row/column structures are optimized for exact/scalar matches, not for nearest-neighbor search over dense vectors — this is slow and often infeasible at scale.
- **Choosing inverted indexes for high-dimensional vector data**: fast for text/keyword lookups but inefficient for high-dimensional embedding search; prefer tree-based or graph-based/ANN indexes instead.
- **Ignoring vector normalization before distance-based search**: unnormalized vectors can distort Euclidean/L2 distance comparisons — normalize before indexing and before querying.
- **Treating RAG as "add any context and done"**: retrieval quality (embedding method, chunking, top-k count) directly determines generation quality — poor retrieval still yields irrelevant or hallucinated answers.

## Code Examples
```python
# Faiss: build an L2 index over 768-dim sentence-transformer embeddings and query it
import faiss
# vectors: numpy array of shape (n_docs, 768), already L2-normalized
index = faiss.IndexFlatL2(768)       # L2 (Euclidean) distance index, 768-dim space
index.add(vectors)                   # add normalized document vectors

# Query
query_vector = model.encode(["how good is the quality of the dress?"])
faiss.normalize_L2(query_vector)
k = 5                                 # retrieve top-k nearest vectors
distances, indices = index.search(query_vector, k)
```
- **What it demonstrates**: constructing a Faiss `IndexFlatL2` similarity index from sentence-transformer embeddings and performing an exact nearest-neighbor search, sorted ascending by distance.

```python
# ChromaDB: create a persistent collection, add documents with metadata, and query
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    documents=["doc text 1", "doc text 2", "..."],
    ids=["id1", "id2", "..."],
    metadatas=[{"topic": "..."}, {"topic": "..."}]
)

results = collection.query(
    query_texts=["what are the available modes"],
    n_results=1
)
```
- **What it demonstrates**: Chroma DB's collection-based API — store embeddings + metadata, use cosine similarity (`hnsw:space: cosine`), and retrieve the single most similar document via `collection.query()`.

## Reference Tables
### Table 4.1: Comparison of vector databases
| Aspect | Indexed vector DBs | Graph vector DBs |
|---|---|---|
| Structure | Store vector embeddings in structured format; use KNN indexing, hierarchical data structures, dimensionality reduction | Represent embeddings as graph nodes; use graph-based algorithms leveraging connectivity |
| Use cases | Recommendation systems, anomaly detection, fraud detection, content search | Knowledge graphs, social network analysis, semantic search, relationship-based recommendations |
| Examples | ChromaDB, Milvus, Pinecone, Weaviate | Amazon Neptune, Neo4j, OrientDB Graph Edition, YugaByte DB |

## Worked Example
RAG implementation in Google Colab using ChromaDB + Google's Gemini embedding model (`embedding-001`): obtain a Gemini API key via Google AI Studio → install Google GenAI + Chroma libraries → load 3 sample documents → generate embeddings with `embedding-001` → store in a Chroma vector store → query with "Attention mechanism" to retrieve relevant passages → construct a prompt ("You are a helpful assistant that answers questions using text from the reference passage...") → LLM answers "what is the attention mechanism?" correctly grounded in the retrieved passage.

## Key Takeaways
1. Vector databases solve a problem traditional (row/column) databases can't: efficient similarity search over high-dimensional embeddings.
2. Choose a vector DB category by use case shape — indexed DBs for pure similarity retrieval, graph DBs when relationships matter.
3. RAG's three-step architecture (embed → store/index → retrieve-and-generate) is the standard pattern for grounding LLM output in external data.
4. Cosine similarity is the dominant metric for NLP embedding comparison; Euclidean/L2 and dot product are viable alternatives depending on whether magnitude matters.
5. Faiss (library, in-RAM/on-disk index) and Chroma (full vector store with metadata/collections) represent two different levels of vector-storage tooling — pick based on whether you need persistence, metadata filtering, and a client-server model.

## Connects To
- **Ch2**: Extends the FAISS vector store usage from the Ch2 chat-with-PDF app into a full comparative study of vector database options.
- **Ch3**: The embeddings generated via chunking/embedding techniques in Ch3 are exactly what gets indexed and searched here.
- **Ch5**: LangChain (next chapter) wraps vector stores like FAISS and Chroma into its retriever abstraction for building production RAG chains.
