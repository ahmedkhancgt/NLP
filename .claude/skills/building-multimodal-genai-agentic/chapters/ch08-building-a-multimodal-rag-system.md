# Chapter 8: Building a Multimodal RAG System

## Core Idea
Completing the multimodal RAG loop requires a dedicated generator module that turns retrieved text/image context into coherent answers, and trustworthy systems must grade both retrieval relevance and generation quality using a *separate* LLM from the one doing generation (LLM-as-a-judge) to avoid self-evaluation bias.

## Frameworks Introduced
- **Multimodal RAG generation layer**: `init_generator()` builds a reusable LLMChain (prompt + LLM); `generate_response()` concatenates retrieved items into a context string and runs the chain with `{query, context}` — decoupling model initialization from generation logic.
  - How: retrieve top-k text/image results → concatenate into one context string → format prompt ("You are an assistant. Based on the following query and context...") → LLM generates final answer.
- **Dual grading with separate LLMs**: Use one LLM (often local, e.g., Ollama) for generation and a *different* LLM (often cloud, e.g., GPT-3.5/4) for grading — avoids confirmation bias where a generative model favors its own outputs when self-evaluating.
  - When to use: always, for production-grade RAG — generation prompts favor verbosity/creativity while grading prompts need precision/brevity/objectivity, a tension best resolved by separating roles entirely.
- **LLM-as-a-judge**: Using an LLM to evaluate/rank/grade outputs of other AI systems via explicit rubric-based prompting, replacing hard-coded rules or human raters.
  - Three approaches: Single-output scoring (1-5 scale against rubric), Pairwise comparison (pick the better of 2+ candidates — benchmarks like MT-Bench/Chatbot Arena show >80% agreement with human evaluators), Reference-based scoring (compare output to a reference/retrieved context).
- **Multimodal LLM-based recommender systems**: Apply the same embedding-alignment/multimodal-fusion/semantic-understanding capabilities that power RAG to personalized content recommendation (not just QA) — handling cold-start via rich cross-modal signal rather than pure collaborative filtering.

## Key Concepts
- **Generative response grader**: LLM (e.g., `ChatOpenAI(temperature=0.3)`) scores a generated response 1-5 with justification, given {query, context, response}.
- **Retrieval relevance grader**: LLM (e.g., `ChatOpenAI(temperature=0)`) returns a binary_score (yes/no) JSON judging whether a retrieved document is relevant to the question — low temperature for reproducibility.
- **Cloud vs. local LLM for grading**: Cloud LLMs (GPT-3.5/4) offer continuous fine-tuning, broader training data, and infrastructure optimization, yielding more consistent/accurate grading at the cost of money and data egress; local LLMs are GPU-constrained and may have outdated weights, degrading grading fidelity.
- **Cold-start problem**: Traditional recommenders (collaborative filtering, metadata-matching) struggle with new users/items lacking interaction history; multimodal LLMs infer preferences from content signals (trailers, images, text) even with minimal history.
- **MLLM recommender architectures**: MMRec (shared latent space + deep ranking), Molar (sequential, CF-aligned embeddings), HistLLM (compresses user history into one prompt token), Serendipitous MLLM (novelty via hierarchical planning), LLMRec (LLM-driven graph augmentation + noise filtering, model-agnostic with GNN/MF).

## Mental Models
- Think of generation and grading as requiring opposite temperaments: generation wants creativity/verbosity (higher temperature), grading wants objectivity/brevity (near-zero temperature) — one model instance rarely serves both well.
- Treat LLM-as-a-judge as a scalable replacement for human rating panels — the same way reranking replaced manual curation in Chapter 1, judge-LLMs replace manual QA review, with similar speed/consistency trade-offs.
- Multimodal recommendation is "RAG's sibling discipline" — both rely on cross-modal embedding alignment, but RAG retrieves-to-answer while recommendation retrieves-to-suggest; the underlying infrastructure (shared vector space, fusion) is reusable across both.

## Anti-patterns
- **Using the same LLM instance for both generation and grading**: introduces self-evaluation bias (confirmation bias) where the model favors its own generated phrasing/approach — always split these roles across distinct LLM calls/models.
- **Grading image content with a text-only relevance-grading prompt**: the chapter's grader.py only evaluates textual documents; applying it directly to images without captioning first silently produces meaningless/incorrect relevance judgments.
- **Treating grading as optional "nice to have"**: without grading, systems cannot self-assess, degrade silently over time, and risk misinformation/bias in safety-critical domains (healthcare, education, finance).
- **Relying purely on collaborative filtering/metadata for cold-start recommendation**: fails when little/no interaction history exists; multimodal content signals (image, text, trailer) fill this gap.

## Code Examples
```python
def init_generator():
    llm = ChatOllama(model="llama3", temperature=0.5)
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="""You are an assistant. Based on the following query and context, provide a relevant and coherent answer.
Query: {query}
Context:
{context}
Answer:"""
    )
    return LLMChain(llm=llm, prompt=prompt)

def generate_response(llm_chain, query: str, retrieved: list):
    context = "\n".join(retrieved)
    return llm_chain.run({"query": query, "context": context})
```
- **What it demonstrates**: The modular generation layer completing the multimodal RAG loop — local LLM (Ollama) synthesizes an answer from concatenated retrieved context.

```python
def init_grader():
    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["query", "context", "response"],
        template="""Evaluate the quality of the following generated response.
Query: {query}
Context: {context}
Response: {response}
Give a score from 1 to 5 and explain why.
Score and Justification:"""
    )
    return LLMChain(llm=llm, prompt=prompt)

def grade_response(grader_chain, query: str, context: str, response: str):
    return grader_chain.run({"query": query, "context": context, "response": response})
```
- **What it demonstrates**: Generative response grader using a *separate cloud LLM* (GPT-3.5) from the local generation model, implementing the dual-LLM grading pattern.

```python
def init_retrieval_grader():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["question", "document"],
        template="""You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Here is the retrieved document:
{document}
Here is the user question:
{question}
Return a JSON object with a single key, binary_score, that is either 'yes' or 'no'."""
    )
    return LLMChain(llm=llm, prompt=prompt)
```
- **What it demonstrates**: Zero-temperature retrieval relevance grader producing a reproducible binary JSON verdict.

## Reference Tables
| Model name | Core strategy | Modality fusion | Key strength | Cold-start handling |
|---|---|---|---|---|
| MMRec | Multimodal embedding + deep ranking | Text + image | Shared latent space, false-positive control | Moderate |
| Molar | CF alignment + multimodal input | Text + image + behavior | Aligns embeddings with evolving user sequences | High |
| HistLLM | History compression into prompt token | All user interactions | Fast inference without losing context | High |
| Serendipitous MLLM | Intent modeling + hierarchical planning | Text + contextual | Novelty while preserving relevance | High |
| LLMRec | LLM-driven graph augmentation + noise filtering | Text + graph + attributes | Model-agnostic, robust in sparse data | Very high |

| LLM-as-a-judge approach | Mechanism | Use case |
|---|---|---|
| Single-output scoring | Score one response 1-5 against rubric | Quality control of a single answer |
| Pairwise comparison | Pick the better of 2+ candidates | A/B comparing generation strategies |
| Reference-based scoring | Compare output to reference/retrieved context | Factuality/grounding checks |

## Worked Example
Extending the Chapter 7 retriever into a full RAG system: `generator.py`'s `init_generator()` wraps a local Ollama `ChatOllama` model (`temperature=0.5`) in an `LLMChain` with a query+context prompt template. `generate_response()` joins retrieved text/image-caption items with newlines and runs the chain to produce a grounded answer. Separately, `grader.py` defines two OpenAI-backed graders: a generative response grader (`temperature=0.3`, scores 1-5 with justification) and a retrieval relevance grader (`temperature=0`, binary yes/no JSON). The complete pipeline: user query → embed (text/image) → vector search (Qdrant) → retrieval relevance grader filters results → local LLM generates response → cloud LLM grades the response → both grading signals feed a feedback loop that can filter what's shown to the user or flag generations for review. This exemplifies the "local generation, cloud judgment" architectural best practice (Figure 8.5 in the book).

## Key Takeaways
1. Completing RAG requires a decoupled generator module (init + run functions) that treats retrieval and generation as separable concerns.
2. Grading and generation should use different LLMs/model instances to avoid confirmation bias — this is a core responsible-AI design principle, not just an implementation detail.
3. LLM-as-a-judge (single-output, pairwise, or reference-based scoring) is now a validated, scalable substitute for human evaluation panels, with reported >80% agreement with humans on benchmarks like MT-Bench.
4. Multimodal recommendation systems (MMRec, Molar, HistLLM, LLMRec, etc.) reuse RAG's embedding-fusion infrastructure to solve cold-start and personalization problems.
5. Cloud LLMs are generally preferred for grading tasks due to consistency/accuracy, even though local LLMs remain preferred for cost-sensitive/private generation.
6. Extending text-only relevance grading to images requires either a captioning preprocessing step or a native multimodal LLM (GPT-4V, LLaVA, MiniGPT-4).

## Connects To
- **Ch 7**: Directly completes the retrieval-only pipeline built there by adding the missing generation component.
- **Ch 6**: Extends the grading-mechanism framework (retrieval relevance grader, hallucination grader, answer quality grader) introduced there into a dual-LLM, cloud-vs-local grading architecture.
- **Ch 9**: The chapter explicitly previews reranking as the next refinement layer sitting between retrieval and generation.
- **Ch 18**: LLM-as-a-judge here is a direct precursor to the formal LLMOps/GenAI evaluation techniques covered in the final chapter.
