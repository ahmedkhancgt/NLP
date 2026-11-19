# Chapter 7: Managing Data for Generative AI

## Core Idea
Data is the foundation of any GenAI project — object storage (MinIO) with tagging/lifecycle/compliance rules provides the governed storage layer, Kubernetes PersistentVolumes/Claims decouple storage from pod lifecycle, automated data-quality scanning catches issues before they poison models, and vector databases/embeddings enable similarity search that underpins RAG.

## Frameworks Introduced
- **Data lake → data swamp → lakehouse evolution**: data warehouses (rigid structure) → data lakes (flexible but ungoverned, risk becoming "swamps") → lakehouses (matured governance + flexibility combined).
  - When to use: as a lens for diagnosing where your organization's data platform sits and what governance gaps need closing.
  - How: assess whether you have clear metadata/lineage/access-control (lakehouse maturity) or just raw accumulation (swamp risk).
- **MinIO governance toolkit**: tagging (custom key-value metadata), bucket versioning, lifecycle/tiering rules (auto-transition/expire objects to cheaper/remote tiers), object locking (Governance vs Compliance mode) and legal holds.
  - When to use: any object-storage-based data lake needing cost control, compliance retention, and traceability.
  - How: tag via UI/`mc tag set`; enable versioning via UI/`mc version enable`; create lifecycle rules (Transition/Expire) targeting a configured remote tier (e.g. Azure Blob); set object-lock mode at bucket creation for retention enforcement.
- **PersistentVolume / PersistentVolumeClaim abstraction**: K8s `StorageClass`-provisioned storage that outlives individual pod lifecycles; PVCs bind/claim a volume for a pod's use, similar to how resource requests work for CPU/memory.
  - When to use: any stateful workload (e.g. MinIO, vector DB) needing data to survive pod restarts/redeployments.
  - How: define a `StorageClass` → create a `PersistentVolumeClaim` requesting capacity → mount the PVC in the pod spec; inspect with `kubectl describe pvc`.
- **Vector database / embedding similarity search**: text/images are converted to high-dimensional vectors; semantic similarity is computed as vector-space closeness rather than keyword matching, enabling natural-language search over unstructured data.
  - When to use: whenever you need to search/match on meaning rather than exact keywords (support KBs, recommendations, fraud similarity, RAG retrieval).
  - How: choose an embedding model, store vectors + metadata in a vector DB (e.g. Chroma), query with a new phrase to retrieve nearest-neighbor matches via the DB's index.

## Key Concepts
- **Data gravity**: the tendency for data's size/complexity to make it hard to move/manage as it accumulates.
- **Tagging**: lightweight governance — custom metadata labels on buckets/objects (e.g. cost_center, classification, owner).
- **Lifecycle/Tiering rule**: automated policy to transition or expire objects to another storage tier (e.g. after N days) for cost/compliance management.
- **Object locking (Governance vs Compliance mode)**: Governance allows privileged deletion before retention expiry; Compliance enforces retention for all users with no override.
- **Legal hold**: indefinite lock on a specific object, settable/removable only by privileged users.
- **Data quality dimensions**: duplicates, missing/null values, categorical typos, outliers, bias/fairness issues, erroneous entries — the standard checklist before data reaches a model.
- **Embeddings**: numeric vector representations of text/images capturing semantic relationships (high-dimensional).
- **Vector DB indexes**: structures enabling fast nearest-neighbor similarity search over large embedding collections (analogous to traditional DB indexes, but similarity-based).

## Mental Models
- Think of **"lakes, not swamps"** — flexibility without governance (tagging, lineage, access control) degrades a data lake's value over time regardless of its technical scalability.
- Use **"storage tier ladder"** thinking: high-performance/high-cost tiers for frequently accessed/recent data, automatically aging down to lower-cost/archival tiers via lifecycle rules.
- Treat **embeddings as "meaning compressed into numbers"** — similar meaning clusters spatially close in vector space, which is why "sky blue" and "blue like the sky" can match even without shared keywords.
- **Garbage in, garbage out** still governs GenAI — augmentation/fine-tuning/prompting cannot fully compensate for poor-quality source data.

## Anti-patterns
- **Treating object storage as a dumping ground with no tags/lifecycle policy**: leads to "data swamp" — unclear ownership, uncontrolled cost growth, compliance risk.
- **Skipping automated data-quality scanning**: manual review doesn't scale; missing nulls, duplicates, and bias issues flow straight into fine-tuning/RAG data without detection.
- **Ignoring unstructured data cleansing** (special characters, casing, stopwords) before embedding: dirtier input text degrades embedding/similarity quality just as much as for structured ML.
- **Using relational exact-match search for semantic queries**: traditional DB queries can't capture "blue like the sky" vs "sky blue" equivalence — this is precisely the gap vector databases close.

## Code Examples
```bash
# MinIO CLI governance commands (mc utility)
mc mb local/demo-datalake
mc tag set local/demo-datalake "cost_center=finance&classification=sensitive"
mc version enable local/demo-datalake

# Lifecycle tiering to Azure Blob storage
mc ilm tier add azure local AZUREBACKUP \
  --account-name miniobackup1234 \
  --account-key <Storage Account Key> \
  --bucket archive \
  --endpoint https://miniobackup1234.blob.core.windows.net
```
- **What it demonstrates**: command-line equivalents for bucket creation, tagging, versioning, and remote-tier lifecycle configuration — scriptable governance instead of manual UI steps.

```python
from ydata_quality import DataQuality
from ydata_quality.missings import MissingsProfiler
from ydata_quality.erroneous_data import ErroneousDataIdentifier
import pandas as pd

df = pd.read_csv('..\\demo-data.csv', sep=r'\s*,\s*', header=0, encoding='ascii')
ED_EXTENSIONS = ['Network Eng', 'None', '']
SENSITIVE_FEATURES = ['education']
dq = DataQuality(df=df, label='job_title', ed_extensions=ED_EXTENSIONS,
                  sensitive_features=SENSITIVE_FEATURES, random_state=13)
results = dq.evaluate()
label_warnings = dq.get_warnings(category='Labels')
duplicate_warnings = dq.get_warnings(category='Duplicates')
erroneous_warnings = dq.get_warnings(category='Erroneous Data')
bf_warnings = dq.get_warnings(category='Bias&Fairness')
```
- **What it demonstrates**: automated multi-category data-quality scanning (missing labels, duplicates, erroneous values, bias/fairness) with a few lines of Python using Ydata Quality.

```python
client = chromadb.HttpClient(host=chroma_host, port=chroma_port,
    settings=Settings(allow_reset=True, anonymized_telemetry=False,
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials=chroma_token))
collection = client.get_or_create_collection("art-descriptions")
if collection.count() <= 1:
    for art in artwork_descriptions:
        collection.add(documents=[art.get('description')],
                        metadatas=[{"source": art.get('source')}],
                        ids=[art.get('id')])

# Query
results = collection.query(query_texts=[phrase], n_results=num_return_sequences)
```
- **What it demonstrates**: ChromaDB vector database usage — token-authenticated connection, collection creation, and similarity query, with automatic embedding generation handled by Chroma.

## Reference Tables
| Data Quality Tool | Approach |
|---|---|
| Soda | CLI + Python library; finds invalid/incomplete/unexpected data |
| Great Expectations | Defines "expectations" (expressive unit tests for data); highly customizable |
| Ydata-quality | Simple Python library for scanning quality issues; can integrate Great Expectations definitions |

## Worked Example
**Building a mini lakehouse + similarity search with MinIO and ChromaDB:** Reuse the MLflow minikube cluster (includes MinIO) → port-forward MinIO UI → create a `demo-finance-data` bucket → tag it (`cost_center=finance`, `classification=sensitive`, `owner=...`) → enable versioning → upload `demo-data.csv` twice to see version history → configure an Azure Blob remote tier (`AZUREBACKUP`) and add a lifecycle rule transitioning non-current versions to it after 1 day → run `dq-test.py` (Ydata Quality) against the CSV to surface missing labels, duplicates, erroneous entries, and bias/fairness warnings → separately, install ChromaDB via Helm → deploy a Gradio app (`vector-db.py`) that loads sample artwork descriptions into a `art-descriptions` collection and exposes a similarity-search UI → query phrases like "Dreaming of blue skies" and observe that semantically related but keyword-different descriptions (clouds, sky, butterfly) rank as top matches.

## Key Takeaways
1. Treat your data lake's governance (tagging, lifecycle, versioning, compliance locks) as a first-class design concern, not an afterthought — this is what prevents "swamp" degradation.
2. Kubernetes PersistentVolumes/Claims decouple stateful storage (MinIO, vector DBs) from pod lifecycle, essential for any durable data-layer component.
3. Automate data-quality scanning (duplicates, missing values, bias) with tools like Ydata-quality rather than relying on manual review — this scales with data growth.
4. Embeddings convert text/images into vectors that capture semantic relationships; vector databases index these for fast similarity search, which is the foundation RAG is built on (Ch8).
5. MinIO's S3-compatible API means lessons learned here transfer directly to AWS S3, Azure Blob, or GCP storage in production.

## Connects To
- **Ch5-6**: extends the MinIO storage layer (introduced as MLflow's artifact backend) into a fuller data-management discussion.
- **Ch8**: embeddings and vector database similarity search here are the direct foundation for Retrieval Augmented Generation (RAG).
- **Ch10**: data governance/classification concepts here feed into the broader GenAI program-office governance framework.
