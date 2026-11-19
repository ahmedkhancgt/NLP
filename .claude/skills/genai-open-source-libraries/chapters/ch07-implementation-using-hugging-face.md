# Chapter 7: Implementation Using Hugging Face

## Core Idea
Hugging Face is the collaborative hub (200,000+ models, the `transformers` library, Text Generation Inference/TGI, and cloud partnerships with AWS/Azure/GCP) that democratizes access to open-source LLMs — this chapter builds a Falcon-7B chat-with-PDF app and shows deploying models to AWS SageMaker via TGI.

## Frameworks Introduced
- **Hugging Face Hub ecosystem**: `transformers` library (thousands of pre-trained models, framework-interoperable across PyTorch/TensorFlow/JAX) + Hub (model/dataset repository) + TGI (Text Generation Inference toolkit for serving) + Inference Endpoints.
  - When to use: discovering/reusing pre-trained models instead of training from scratch; benchmarking via the Open LLM Leaderboard.
- **Hugging Face Inference Endpoint deployment flow (AWS SageMaker)**: (1) `get_huggingface_llm_image_uri(backend="huggingface", region=region)` to get the TGI container image, (2) configure `HuggingFaceModel` with `hf_model_id`, `hf_task`, `SM_NUM_GPUS` (tensor parallelism degree) and optionally `HF_MODEL_QUANTIZE`, (3) `model.deploy(...)` to launch the endpoint, (4) `predictor.predict(...)` to query it.
  - When to use: production-grade hosted inference without managing raw infrastructure; `SM_NUM_GPUS` matters when a model's parameter size exceeds single-GPU memory (tensor parallelism splits the model across GPUs).
- **Hugging Face Text Generation client pattern**: `Client(endpoint_url).generate(...)` for single-shot generation, `.generate_stream(...)` for token streaming.

## Key Concepts
- **Hugging Face Hub**: open-source platform/community hosting pre-trained models, datasets, and demos.
- **Transformers library**: Hugging Face's library providing unified access to thousands of pre-trained models for NLP, audio, vision, and multimodal tasks.
- **Text Generation Inference (TGI)**: toolkit for high-performance serving of popular open-source LLMs (Llama, Falcon, StarCoder, BLOOM, GPT-NeoX).
- **Tensor parallelism**: splitting a model's tensors across multiple GPUs when the model is too large for one GPU; configured via `SM_NUM_GPUS`.
- **HF_MODEL_QUANTIZE**: environment variable flag to serve a quantized (lower-memory, lower-precision) version of a hosted model.
- **Framework interoperability**: `transformers` supports PyTorch, TensorFlow, and JAX without rewriting model code.

## Mental Models
- Think of Hugging Face as "GitHub for models" — a collaborative hub where pretrained weights, datasets, and demos are shared and reused rather than rebuilt.
- Treat cloud deployment (SageMaker/Azure ML/Vertex AI) as a thin managed layer over the same core steps: get an image → configure environment variables → deploy → predict — the same four steps appear regardless of provider.
- Quantization at serving time (`HF_MODEL_QUANTIZE=true`) is the same trade-off introduced in Ch6 (memory/speed vs. precision) applied to hosted inference endpoints specifically.

## Anti-patterns
- **Treating Hugging Face as a single all-in-one solution**: it's explicitly a platform/community, not a one-size-fits-all tool — other libraries may suit specific project needs better.
- **Deploying a large model without considering `SM_NUM_GPUS`**: models too large for a single GPU will fail or be unusably slow without tensor parallelism configured correctly for the instance type.
- **Ignoring the output-quality trade-off of `HF_MODEL_QUANTIZE`**: quantized serving reduces memory/cost but can lower output precision — validate output quality before assuming quantization is "free."

## Code Examples
```python
# Query a Hugging Face Inference Endpoint (text generation + streaming)
!pip install text-generation
from text_generation import Client

endpoint_url = ""  # Copy your model endpoint here
client = Client(endpoint_url)
text = client.generate("How does Transformers work?").generated_text
print(text)

# Token streaming
text = ""
for response in client.generate_stream("How does Transformers work?"):
    if not response.token.special:
        text += response.token.text
print(text)

from text_generation.inference_api import deployed_models
print(deployed_models())
```
- **What it demonstrates**: calling a deployed Hugging Face Inference Endpoint for both single-shot and streaming text generation, and listing available deployed models.

```python
# Deploy GPT-NeoX-20B to AWS SageMaker via Hugging Face TGI container
import sagemaker, time
from sagemaker.huggingface import get_huggingface_llm_image_uri, HuggingFaceModel

sagemakersession = sagemaker.Session()
region = sagemakersession.boto_region_name
role = sagemaker.get_execution_role()

imageuri = get_huggingface_llm_image_uri(backend="huggingface", region=region)

modelname = "gpt-neox-20b"
hub = {
    'HF_MODEL_ID': 'EleutherAI/gpt-neox-20b',
    'HF_MODEL_QUANTIZE': 'true',
    'HF_TASK': 'text-generation',
    'SM_NUM_GPUS': '4'
}
model = HuggingFaceModel(name=modelname, role=role, env=hub, image_uri=imageuri)

predictor = model.deploy(initial_instance_count=1, instance_type="ml.g4dn.2xlarge", endpoint_name=modelname)

inputdata = {"inputs": "How does Transformers work?",
             "parameters": {"do_sample": True, "max_new_tokens": 100, "temperature": 0.6}}
predictor.predict(inputdata)
```
- **What it demonstrates**: end-to-end cloud deployment of an open-source LLM (GPT-NeoX-20B) via Hugging Face's TGI container on AWS SageMaker, with 4-GPU tensor parallelism and quantized serving enabled.

```python
# Ingest.py — chunk + embed PDF with Hugging Face Instructor embeddings (Falcon 7B pipeline)
import os, click
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings

def load_documents():
    loader = PyPDFLoader('SOURCE_DOCUMENTS/Attention.pdf')
    return loader.load()

@click.command()
@click.option('--device_type', default='cuda', help='select gpu or cpu for execution')
def main(device_type):
    device = 'cpu' if device_type in ['cpu', 'CPU'] else 'cuda'
    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                                 model_kwargs={"device": device})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local('faiss_index')

if __name__ == "__main__":
    main()
```
- **What it demonstrates**: a slightly tuned chunking config (chunk_size=800, overlap=120, vs. Ch2's 900/100) for a Hugging Face-centric Falcon-7B RAG pipeline.

## Reference Tables
None (this chapter uses figures rather than tables).

## Worked Example
Same chat-with-PDF application pattern as Chapter 2, rebuilt explicitly around Hugging Face tooling: create an `LLM` conda environment → install the same pinned requirements (Transformers, langchain, faiss-cpu, InstructorEmbedding, sentence-transformers) → ingest the "Attention Is All You Need" PDF with `chunk_size=800`/`chunk_overlap=120` and `HuggingFaceInstructEmbeddings` → `run_LLM.py` downloads `tiiuae/falcon-7b-instruct` via `AutoTokenizer`/`AutoModelForCausalLM`, wraps it as a LangChain `HuggingFacePipeline`, and answers interactive questions via `RetrievalQA` with sourced documents — identical architecture to Ch2 but framed explicitly as "using the Hugging Face Hub."

## Key Takeaways
1. Hugging Face's core value is collaboration and democratization: shared pre-trained models/datasets save training time/cost and reduce carbon footprint.
2. TGI (Text Generation Inference) is the standard toolkit for serving popular open-source LLMs at scale, with tensor-parallel GPU splitting for very large models.
3. Cloud deployment of Hugging Face models follows the same four-step pattern across AWS/Azure/GCP: get image → configure model/env vars → deploy → predict.
4. Quantization can be applied at serving time (`HF_MODEL_QUANTIZE`) as well as training time, trading output precision for lower memory/compute cost.
5. Hugging Face is a platform/community, not the only option — evaluate project-specific needs before defaulting to it.

## Connects To
- **Ch2 & Ch6**: Reuses the exact chat-with-PDF architecture from Chapters 2 and 6 (Falcon/Llama), now explicitly framed as "Hugging Face Hub implementation."
- **Ch6**: Builds directly on PEFT/quantization concepts from Chapter 6, applying `HF_MODEL_QUANTIZE` at serving/deployment time rather than training time.
- **Ch9**: The AWS SageMaker deployment pattern here (image URI → model config → deploy → predict) is the direct precursor to the full deployment chapter's infrastructure discussion.
