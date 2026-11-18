# Chapter 8: Refining and Improving Results

## Core Idea
Refinement techniques form a cost/time/accuracy trade-off spectrum — from cheapest/fastest (prompt engineering) through RAG (vector search + LLM) to PEFT to full fine-tuning — and should be approached "right to left" (start cheap, escalate only if needed); RAG combines embeddings-based retrieval with LLM generation to ground responses in domain-specific data while using context windows efficiently.

## Frameworks Introduced
- **Refinement trade-off spectrum**: Prompt engineering (cheapest, no weight changes) → RAG (moderate cost, adds retrieval infrastructure) → PEFT (moderate-high cost, partial weight updates) → Full fine-tuning (highest cost/time/expertise) → Training from scratch (almost never justified).
  - When to use: start at prompt engineering; escalate only when the cheaper technique demonstrably fails to meet accuracy/domain needs.
  - How: work backward from Figure 8.2's trade-off diagram — test prompting, then small-scale RAG, then PEFT, before committing to full fine-tuning budget.
- **RAG (Retrieval Augmented Generation) architecture**: user query → embed + similarity search against a vector store → retrieve top-k relevant documents → combine documents + original query into an augmented prompt → send to LLM for a natural-language grounded response.
  - When to use: when the model needs access to domain-specific/proprietary knowledge without the cost of fine-tuning, and when context-window efficiency matters (retrieval narrows what's sent to the LLM).
  - How: embed documents into a vector store (Ch7) → at query time, retrieve top-N matches → construct an "Instruct... only use descriptions from: {relevant_document} ... {user_input}" style prompt → invoke the LLM.
- **Advanced prompting techniques**: Least-to-most prompting (ask model to list sub-problems before solving), Plan-and-solve prompting (ask model to build a plan then execute it step by step), Self-consistency (sample the same prompt multiple times, choose majority/most-common answer), Role prompting (assign a persona, e.g. "Economics Professor"), Agents (automate multi-step prompting with tool/API calls).
  - When to use: Least-to-most/Plan-and-solve for multi-step reasoning failures; Self-consistency for non-deterministic answer reliability; Role prompting for domain-specific style/depth; Agents for automating multi-step workflows.
  - How: append structured instructions (e.g. "What subproblems must be solved before answering?", "Let us first understand the problem... make a complete plan...") to the base prompt; for self-consistency, run N times and take the modal answer (optionally automatable via JSON-formatted outputs and code-based tallying).
- **PEFT via Adapters**: add small "adapter layers" on top of a frozen base model's weights rather than updating all parameters.
  - When to use: domain adaptation when full fine-tuning's cost/time/data requirements are unjustified.
  - How: LoRA (low-rank adapter weights added to base layers, minimizes tracked dimensions) or IA3 (injects learned vectors to amplify/suppress activations between layers, often more efficient than LoRA).

## Key Concepts
- **Context window**: the LLM's "short-term memory" limit on prompt+response tokens; RAG helps use it efficiently by sending only retrieved-relevant content instead of an entire knowledge base.
- **Zero-shot / Few-shot / Chain-of-Thought (CoT)**: baseline prompting maturity levels (recap from Ch4) that advanced techniques (least-to-most, plan-and-solve) build upon.
- **ContextRelevancyEvaluator (Llama-Index)**: scores whether a RAG response's context/answer is actually relevant to the original query.
- **FaithfulnessEvaluator / CorrectnessEvaluator (Llama-Index)**: check for hallucination and evaluate against a reference answer, respectively.
- **Dot Product similarity**: linear-algebra measure of vector closeness used by embedding similarity search (near 0 = near-perpendicular/dissimilar, near 1 = aligned/similar).
- **SentenceTransformers**: the framework ChromaDB uses by default (`all-MiniLM-L6-v2` model) for automatic text embedding.

## Mental Models
- Think of **RAG as "give the LLM a cheat sheet, not the whole textbook"** — retrieval narrows the context window to just what's relevant, saving compute/cost vs. stuffing an entire knowledge base into the prompt.
- Use **"work backward from Figure 8.2"** — always ask whether a cheaper technique (prompting) can solve the problem before reaching for RAG, then PEFT, then full fine-tuning.
- Treat **negative statements ("I don't like clouds") as a stress test for RAG** — pure vector similarity can rank the wrong (topically similar but sentiment-opposite) item first; the LLM generation step is what correctly reinterprets user intent.
- **Self-consistency treats non-determinism as a feature to sample, not a bug to eliminate** — run multiple times and trust the modal answer rather than the first one.

## Anti-patterns
- **Jumping straight to fine-tuning without trying prompt engineering or RAG first**: wastes budget/time on the costliest option when cheaper ones often suffice.
- **Feeding an entire knowledge base into the context window instead of retrieving top-k relevant chunks**: wastes context capacity and compute, and dilutes similarity relevance — especially problematic at real-world scale (tens of thousands of entries).
- **Trusting raw vector similarity ranking without an LLM generation/evaluation step**: pure similarity search can rank semantically-close-but-intent-wrong items first (e.g. matching on "clouds" when the user said they dislike clouds).
- **Skipping evaluation of RAG responses**: without evaluators (ContextRelevancyEvaluator, FaithfulnessEvaluator), hallucinated or off-topic RAG answers (e.g. recommending unrelated items when no good match exists) go undetected.
- **Not cleaning/preparing data before embedding**: spelling errors, typos, extra characters degrade embedding/indexing quality, directly harming RAG retrieval precision.

## Code Examples
```python
augmented_prompt = """Instruct: You are a bot that makes recommendations for art. Answer with the best item and description, but only use descriptions from these: {relevant_document}.
Make sure to use the correct matching item id. Suggest all matching items based on the recommendations and this request: {user_input}
Output: Based on your preferences I recommend item """
augmented_prompt = augmented_prompt.format(user_input=prompt, relevant_document=retrevied_descriptions)

phi_llm = LlamaCpp(
    model_path=model_file,
    temperature=temp,
    max_tokens=max_length,
    n_ctx=context_length,
    n_gpu_layers=0,
    verbose=True
)
results = phi_llm.invoke(augmented_prompt)
```
- **What it demonstrates**: the core RAG prompt-construction pattern — instructions + retrieved documents + user query combined into a single augmented prompt sent to a quantized local LLM (Phi-2 GGUF).

```python
# Least-to-most prompting
# Prompt: "...If I remove one cup from each shelve, how many cups are there in the cupboard?
#   What subproblems must be solved before answering the inquiry?"
# -> model lists sub-steps, which are then fed back into a follow-up prompt to reach the correct answer.
```
- **What it demonstrates**: least-to-most prompting fixing an initially incorrect chain-of-thought math answer by first eliciting sub-problem decomposition.

```python
#Create query engine and run prompt using vector store for additional information
query_engine = index.as_query_engine(text_qa_template=text_qa_template, llm=llm)
response = query_engine.query(prompt)

#Setup Evaluators to judge results
cre_evaluator = ContextRelevancyEvaluator(llm=llm, service_context=service_context)
cre_eval_results = cre_evaluator.evaluate_response(query=prompt, response=response)
```
- **What it demonstrates**: Llama-Index's query engine + built-in evaluator pattern for scoring RAG response relevancy automatically.

## Reference Tables
| Refinement Technique | Relative Cost/Time | Resources/Skill Needed |
|---|---|---|
| Prompt engineering | Lowest | None special |
| RAG | Low-moderate | Vector DB + embedding setup |
| PEFT (LoRA/IA3) | Moderate | ML fine-tuning expertise, GPU |
| Full fine-tuning | High | Significant GPU/data/expertise, days-weeks |
| Training from scratch | Highest | Massive GPU clusters, months, rarely justified |
(Derived from Figure 8.2's conceptual trade-off diagram)

## Worked Example
**Phi-2 RAG art-recommendation app:** Use the quantized Phi-2 GGUF model (Q4_K_M, <2GB) via LlamaCpp with `n_ctx=512`, `max_tokens=160`, `temperature=0.1` → connect to the ChromaDB `art-descriptions` collection from Ch7 → on each user prompt, retrieve top-3 similar descriptions → build an augmented instruct-prompt restricting the model to only use retrieved descriptions → test cases: (1) "I like cars, can you recommend something?" correctly filters car-related items but occasionally hallucinates details (wrong color, invented reviews); (2) "I am looking for a painting with clouds" returns accurate matches; (3) "I love the color cobalt" surfaces a nuanced recommendation beyond pure keyword match; (4) "I don't like clouds, what do you recommend?" demonstrates the LLM correctly reinterpreting negated intent that pure vector similarity ranked wrong. A second version using Llama-Index adds `ContextRelevancyEvaluator`, which correctly flags an out-of-domain query ("I like candy...") as not contextually relevant, and confirms a car-related query as relevant but incomplete.

## Key Takeaways
1. Always evaluate the cheapest viable refinement technique first — prompt engineering, then RAG, before PEFT or full fine-tuning.
2. RAG's core value is context-window efficiency: retrieve only what's relevant rather than feeding an entire knowledge base to the LLM.
3. Advanced prompting (least-to-most, plan-and-solve, self-consistency, role prompting) can resolve multi-step reasoning failures without touching model weights.
4. RAG is not infallible — pure vector similarity can misrank on negation/intent; the LLM's generation step and dedicated evaluators (ContextRelevancyEvaluator, FaithfulnessEvaluator) are necessary safety nets.
5. Real-world RAG at scale requires more rigor than the book's toy example: larger datasets need better chunking, metadata filtering (timestamps, source), and context-window budgeting.
6. Data cleanliness (from Ch7) directly determines RAG retrieval quality — garbage embeddings in, garbage retrieval out.

## Connects To
- **Ch4**: directly extends the zero-shot/few-shot/CoT prompting techniques introduced there into least-to-most, plan-and-solve, and self-consistency.
- **Ch7**: builds the full RAG pipeline on top of the embeddings/vector database foundation established there.
- **Ch10**: RAG's indirect-injection risk (poisoned retrieval data) is explicitly flagged as a security attack vector in the next chapter.
