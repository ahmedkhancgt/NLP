# Cheatsheet: Decision Rules & Quick Reference

## Which LLM-adaptation approach? (Ch2, Ch6)
| Situation | Use |
|---|---|
| No/limited training data, need fast iteration | Prompt engineering |
| Need grounding in external/current/proprietary knowledge | RAG |
| Have labeled domain data + compute budget | Fine-tuning |
| Have data + compute but want to minimize cost/GPU memory | PEFT (LoRA/QLoRA) instead of full fine-tuning |
| Need preference alignment (helpful/harmless) | DPO (simpler) over RLHF (needs separate reward model + RL loop) |

## Which chunking strategy? (Ch3)
| Need | Strategy |
|---|---|
| Fast, simple, large-scale/real-time | Fixed-size chunking |
| Preserve sentence-level meaning | Sentence splitting |
| Long/hierarchical documents | Recursive chunking |
| High precision, can tolerate complexity | Hybrid chunking |
| Cross-boundary context matters | Add overlap (trade-off: higher compute) |

## Which embedding approach? (Ch3)
| Need | Choice |
|---|---|
| Context-sensitive tasks (QA, sentiment) | BERT, Universal Sentence Encoder |
| Word-level tasks (translation, summarization) | word2vec, GloVe |
| Rare/OOV words | FastText |
| Multimodal (text + image) | CLIP |
| Limited training data | Pre-trained embeddings (MUSE, Hugging Face hub) |
| Limited compute | Lightweight models (FastText) or pre-trained, not from-scratch |
| Need benchmark comparison | Consult MTEB |
| Budget allows, want best quality | Paid APIs (OpenAI, Anthropic, Cohere) — else open-source (BERT, Instructor, sentence-transformers) |

## Which vector database / library? (Ch4)
| Need | Choice |
|---|---|
| Pure similarity search / retrieval at scale | Indexed vector DB (Pinecone, Milvus, Weaviate, Chroma) |
| Relationship/connection-aware search | Graph vector DB (Neo4j, Amazon Neptune, OrientDB) |
| Geographic/spatial data | Spatial DB (PostGIS, Oracle Spatial) |
| Lightweight, in-process, no server | Vector library (Faiss, Annoy, HNSWLib) |
| Need metadata filtering + persistence, simple setup | Chroma |
| Existing SQL/NoSQL investment | Vector-capable SQL (SingleStoreDB) or NoSQL (MongoDB) |

## Which similarity metric? (Ch4)
| Metric | Use when |
|---|---|
| Cosine similarity | Comparing text embeddings regardless of length/magnitude (most common for NLP) |
| Euclidean (L2) distance | Magnitude matters; used by Faiss `IndexFlatL2`; requires normalization for comparability |
| Dot product | Need cheap alignment-strength signal; conflates magnitude and direction |

## Which prompting technique? (Ch6)
| Task shape | Technique |
|---|---|
| Need precise, constrained, formatted output | Explicit prompting |
| No examples available | Zero-shot |
| Want consistent format/style | Few-shot |
| Multi-step reasoning (math, logic) | Chain-of-thought (CoT) |
| High-stakes reasoning, can afford more compute | Self-consistency sampling (multiple CoT paths, majority vote) |
| Exploration/backtracking needed (e.g., constraint puzzles) | Tree-of-thought (ToT) |
| Naturally staged sub-tasks | Sequential prompting |
| Need to both include and exclude content | Positive + negative prompting |

## Which cloud deployment service? (Ch7, Ch9)
| Provider | Managed hosting | Container orchestration |
|---|---|---|
| AWS | SageMaker | ECS |
| Azure | Azure Machine Learning (AML) | Azure Kubernetes Service (AKS) |
| GCP | Vertex AI | AI Platform Pipelines / Kubeflow |

## Deployment checklist (Ch9)
- [ ] Model serialized in a deployment format (TensorFlow SavedModel / PyTorch TorchScript)
- [ ] Infrastructure sized for parameter count (GPU/TPU for training+inference; enough RAM to avoid bottlenecks)
- [ ] Storage plan accounts for both training data AND model parameters (consider quantization to cut footprint)
- [ ] API built (typically RESTful) to expose the model
- [ ] Containerized (Docker) for environment consistency
- [ ] Scaling strategy chosen: horizontal (more instances) vs. vertical (more resources per instance)
- [ ] Security: authentication, authorization, monitoring for misuse
- [ ] Monitoring/alerting configured for performance, resource use, errors
- [ ] Tensor parallelism (`SM_NUM_GPUs` or equivalent) configured if model exceeds single-GPU memory
- [ ] Quantization considered if storage/compute is constrained (trade-off: precision)

## Tells and smells for common implementation mistakes
| Smell | Likely problem | Fix |
|---|---|---|
| Model answers confidently but incorrectly | Hallucination; possibly poor retrieval grounding | Verify with RAG grounding; add "say you don't know" instruction |
| RAG answers are off-topic or generic | Poor chunking or embedding mismatch, or k too low/high | Re-tune chunk_size/overlap; check embedding-corpus match; adjust k |
| Fine-tuning is prohibitively expensive/slow | Full fine-tuning on a huge model | Switch to PEFT (LoRA/QLoRA) |
| Model fails on long documents | Context window exceeded | Use Map-Reduce chain or better chunking |
| Chatbot loses context across turns | No conversation history/condensation | Add memory + `CONDENSE_QUESTION_PROMPT` |
| Embeddings leak PII | No anonymization/encryption before embedding | Apply the secure-embedding pipeline (Ch3) |
| Generated images show skewed demographics | Model/training-data bias (documented even in commercial tools) | Apply bias-mitigation stack (data + model + human oversight, Ch10) |
| Inference too slow/expensive at scale | No quantization, wrong instance sizing, no tensor parallelism | Apply quantization; set `SM_NUM_GPUS`; reconsider instance type |
| Vector search misses obviously-relevant results | Vectors not normalized before Euclidean distance search | Normalize vectors before indexing and querying |

## Evaluation metric quick reference (Ch6)
| Task | Metric |
|---|---|
| Machine translation | BLEU (0-1, higher better) |
| Summarization | ROUGE |
| Language modeling quality | Perplexity (lower better) |
| General LLM capability | Benchmarks: GLUE, MMLU, SQuAD 2.0, HELM; Hugging Face Open LLM Leaderboard |
| RAG retriever quality | Cosine/Euclidean similarity search scores; LangChain MMR (maximum marginal relevance) |
| Subjective quality (fluency, relevance) | Human evaluation (expert annotation or crowdsourcing) |
