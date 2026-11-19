# Cheatsheet: Decision Rules & Smells

## Local vs. API-based GenAI
| Signal | Choose Local (Ollama) | Choose API (OpenAI/cloud) |
|---|---|---|
| Data sensitivity | High / regulated | Low, or compliant cloud setup available |
| Model size needed | Small/quantized (7B-ish) OK | Need GPT-4-class reasoning |
| Budget pattern | Continuous, predictable workload | Spiky, short-term, or low-volume |
| Latency need | Can tolerate CPU inference | Need high-throughput, low-latency |
| Need agentic ecosystem (Responses API, Agents SDK) | No | Yes |

## When to add reranking
- Retrieval returns top-k by cosine similarity but downstream generation hallucinates or cites loosely-relevant content → add a reranking stage.
- Candidate pool is large (>50) → bi-encoder retrieve, then rerank only top-N (prefilter first).
- Precision matters more than latency/cost (legal, medical, QA) → use cross-encoder rerank.
- Need scalable token-level precision without full cross-encoder cost → use late interaction (ColBERT/MaxSim).

## Choosing an interaction depth
| Need | Use |
|---|---|
| First-stage retrieval over millions of docs | No interaction (bi-encoder) |
| Scalable precision w/ token-level matching | Late interaction (ColBERT, MaxSim) |
| Max precision on a small (≤100) candidate set | Full interaction (cross-encoder) |

## Retrieval-optimization levers (pick by symptom)
| Symptom | Lever |
|---|---|
| Missing paraphrased/synonym matches | Query expansion |
| Cross-modal mismatch (image vs text) | Modality-based routing / joint embedding (CLIP) |
| Similarity scores biased by vector length | Embedding normalization (L2) |
| Combining BM25 + vector scores unfairly | Score normalization (min-max/z-score/RRF) before fusion |
| Reranking too slow at scale | Prefiltering thresholds (top-N/score cutoff before cross-encoder) |
| Index serving stale/outdated content | Adaptive index refresh (incremental or scheduled) |
| Long documents losing fine-grained info | Multi-index / multi-vector representation |
| Manual tuning of many retrieval params is costly | Genetic algorithm (GA) parameter search |

## When to use which multi-stage RAG variant
| Situation | Variant |
|---|---|
| Simple QA, single-document | Simple RAG |
| Multi-turn conversation | Simple RAG + memory |
| Multi-hop / complex synthesis | Branched RAG |
| Abstract/underspecified query | HyDE |
| Query complexity/type varies | Adaptive RAG |
| Factual accuracy critical (legal/medical) | Corrective RAG (CRAG) |
| Need self-correction/iterative improvement | Self-RAG |
| Tool use, multi-step decision-making | Agentic RAG |

## LLMOps / evaluation metric selection
| What you're evaluating | Use |
|---|---|
| Model fluency/coherence in isolation | BLEU, ROUGE, METEOR, BERTScore |
| Retrieval quality | Recall@k, Precision@k, embedding drift |
| Generation groundedness | Faithfulness score, hallucination rate |
| Full pipeline correctness | EM/F1, context-aware BERTScore, LLM-as-judge |
| Text-to-SQL correctness | Execution accuracy (production), exact match (benchmarks) |
| Process-level debugging | Span-based tracing (MLflow/Langfuse) |
| Outcome-level monitoring | Metric-based evaluation (LLM-as-judge faithfulness/relevance) |

## Agentic/retrieval failure smells
- **"Correct-looking but wrong" answers** → check hallucination grader / faithfulness score, not just relevance.
- **High recall, poor precision** → add reranking stage.
- **High precision, missed relevant docs** → increase top-k, add MultiQueryRetriever, or query expansion.
- **Unbounded agent retry loops** → always cap retries (e.g., 3) with a graceful fallback message.
- **Same LLM generating and grading its own output** → confirmation bias risk; use a separate model/instance for grading.
- **Vector search returns right modality but wrong content, or wrong modality entirely** → modality mismatch; check embedding alignment / joint latent space.
- **Retrieval was fine yesterday, degraded today** → check for index staleness or embedding drift; trigger adaptive refresh.
- **Multi-turn follow-up ("which of them...") breaks** → missing dialogue-state/co-reference tracking.
- **Text-to-SQL references non-existent columns/tables** → schema linking failure; add schema-aware validation before execution.
- **Agent spends excessive tokens on every query regardless of complexity** → missing tiered/prefiltering routing; add complexity-based gating.
- **One pipeline stage is agentic for no clear reason** → check if it should instead be a plain deterministic function calling an LLM once.

## HITL vs. full automation
- Use HITL when: domain is safety-critical (legal/healthcare/finance), cost of a wrong answer is high, or regulatory compliance requires human sign-off.
- Skip HITL when: task is low-stakes, high-volume, and graders/validators already provide adequate automated quality control.

## OCR approach selection
| Situation | Approach |
|---|---|
| Need modular, local, interpretable pipeline | Wrapped OCR (EasyOCR) + separate LLM |
| Noisy/handwritten/unstructured documents | OCR foundation model (Mistral OCR) |
| Need single-step extraction + visual context reasoning | Fine-tuned multimodal LLM (LLaVA, Llama 3.2 Vision) |
