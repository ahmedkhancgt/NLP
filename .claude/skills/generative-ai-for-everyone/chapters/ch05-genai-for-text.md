# Chapter 5: GenAI for Text

## Core Idea
The transformer architecture (encoder-only, decoder-only, or encoder-decoder, built from tokenization, embeddings, positional encoding, multi-head self-/cross-attention, and a language-modeling head) superseded RNN/LSTM sequence models by processing sequences in parallel rather than recurrently, and underpins every modern LLM and NLP task — classification, summarization, QA, and translation can all be cast as instances of next-token prediction.

## Frameworks Introduced
- **Sequence modeling taxonomy**: one-to-sequence (image captioning), sequence-to-one (sentiment analysis), sequence-to-sequence (translation).
  - When to use: classifying any NLP task by its input/output shape before choosing an architecture.
- **Three transformer architecture types**: encoder-only (bidirectional attention; BERT/RoBERTa/ALBERT — classification, NER, sentiment), decoder-only (causal/autoregressive attention; GPT-3/4, LLaMA, PaLM, Falcon — text generation, completion, QA), encoder-decoder (cross-attention; T5, BART — translation, summarization, image captioning).
  - When to use: pick by task — need a fixed representation of input (encoder-only); need open-ended generation (decoder-only); need to map one sequence to a transformed sequence (encoder-decoder).
  - How: encoder-only attends bidirectionally over the whole input; decoder-only masks future tokens (causal attention) and is trained self-supervised via next-token prediction; encoder-decoder adds cross-attention where decoder queries attend to encoder-produced keys/values.
- **Transformer component pipeline**: tokenization (word/character/sub-word — BPE, WordPiece, SentencePiece) → token IDs → embeddings → positional embeddings (sinusoidal, relative, RoPE) → stacked encoder/decoder blocks (multi-head self-attention + position-wise feedforward, each with residual connection + layer norm) → language modeling head (logits over vocabulary) → decoding strategy.
  - When to use: the universal blueprint for building or debugging any transformer-based NLP system.
  - How: self-attention computes Q/K/V projections per token, scores via scaled dot product + softmax, multi-head splits this across subspaces; decoder adds masked self-attention (no future leakage) and cross-attention (attends to encoder output).
- **Decoding strategy selection**: deterministic (greedy, beam search) vs. stochastic (top-k, top-p/nucleus, temperature sampling).
  - When to use: greedy/beam for precision-critical tasks (translation); top-k/top-p/temperature for creative, diverse generation.
  - How: greedy picks argmax at each step (fast but locally myopic); beam search keeps k candidate sequences; top-k/top-p truncate the vocabulary distribution before random sampling; temperature reshapes the logit distribution's sharpness before sampling.
- **NLP task → modeling pattern**: NSP/MLM for BERT-style pretraining, span classification for extractive QA, RAG for knowledge-grounded QA, auto-regressive cross-entropy for translation, extractive/abstractive/hybrid for summarization.

## Key Concepts
- **NLU vs. NLG**: natural language understanding (extract meaning/intent/entities/sentiment) vs. natural language generation (produce coherent output from structured meaning).
- **Byte-Pair Encoding (BPE) / WordPiece / SentencePiece**: sub-word tokenization algorithms balancing vocabulary size against out-of-vocabulary handling; BPE is used by GPT, WordPiece by BERT, SentencePiece by T5.
- **Positional embedding**: injects token-order information since attention itself is permutation-equivariant; sinusoidal (fixed), relative, or RoPE (rotary, multiplicative, used in RoFormer).
- **Self-attention vs. cross-attention**: self-attention derives Q/K/V from the same sequence (context within one sequence); cross-attention derives K/V from one sequence (encoder output) and Q from another (decoder input).
- **Masked multi-head attention**: blocks a decoder token from attending to future tokens (upper-triangular mask set to −∞ before softmax), enabling autoregressive generation.
- **Masked Language Modeling (MLM) / Next Sentence Prediction (NSP)**: BERT's bidirectional pretraining objectives — predict randomly masked tokens, and predict whether two sentence spans are related, using `[CLS]`/`[SEP]`/`[MASK]` special tokens.
- **Span classification**: extractive-QA modeling pattern that predicts start/end token probabilities for the answer span within a passage.
- **Retrieval-Augmented Generation (RAG)**: grounds an LLM's answer by retrieving relevant documents as context, mitigating hallucination and proprietary-knowledge gaps.
- **Context size / sequence length**: BERT 512 tokens, BART/PEGASUS 1024, LongT5 4096 (up to 16k via sparse attention), most LLMs ~2048+; memory requirement grows sharply (roughly exponentially) past 4096 tokens.

## Mental Models
- Think of **attention as a soft, learned lookup table**: queries ask "what am I looking for," keys answer "what do I contain," and values are "what I'll contribute if selected" — multi-head attention runs several such lookups in parallel subspaces (echoing CNN filters in Ch 3, each capturing a different feature).
- Use the **"everything is next-token prediction" frame**: sentiment analysis, QA, and summarization can all be reframed as conditional probability estimation over the next token(s) given a crafted prompt (e.g., `{Text} TLDR:` for summarization).
- Think of **masking as the only structural difference between encoder and decoder self-attention**: same mechanism, but decoder blocks future information to preserve autoregressive generation validity.
- Use the **decoding-strategy dial (deterministic ↔ stochastic)**: move toward greedy/beam search when you need reliability and reproducibility (translation), and toward top-k/top-p/temperature when you need creativity and diversity (open-ended writing).

## Anti-patterns
- **Using word-level tokenization for production LLMs**: breaks on misspellings, polysemy, and OOV words — sub-word tokenization (BPE/WordPiece/SentencePiece) is the practical default.
- **Relying on greedy decoding for creative generation tasks**: produces repetitive, locally "safe" text and is highly sensitive to early sub-optimal token choices that propagate forward.
- **Expanding context length indefinitely without considering memory cost**: required memory grows sharply past ~4096 tokens; use sparse attention or retrieval (RAG) instead of brute-force context expansion.
- **Trusting a vanilla decoder-only LLM's direct answer to a knowledge-intensive question**: prone to hallucination and lacks proprietary/up-to-date knowledge — ground it with RAG instead.
- **Fine-tuning a general pretrained model only once and expecting task-specific quality**: the chapter's own example shows a two-stage process (domain pretraining on manuals, then task-specific fine-tuning on labeled query→output pairs) is needed for good task performance.

## Code Examples
```python
# Load GPT-2 and generate before any fine-tuning (baseline, often poor on niche domains)
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

generator = pipeline('text-generation', model='gpt2')
generator("Cypher query to create a customer node ", max_length=60, num_return_sequences=3, do_sample=True)
```
```python
# Stage 1: domain-adaptive pretraining on parsed PDF manuals (causal LM objective)
train_dataset = TextDataset(tokenizer=tokenizer, file_path='/content/train_data.txt', block_size=32)
val_dataset = TextDataset(tokenizer=tokenizer, file_path='/content/validation_data.txt', block_size=32)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir='/content/model_checkpoints_fine-tuned_cypher',
    evaluation_strategy="epoch", learning_rate=2e-5,
    per_device_train_batch_size=128, num_train_epochs=20,
    weight_decay=0.01, fp16=True, save_strategy="epoch",
    save_total_limit=2, load_best_model_at_end=True,
)
trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=val_dataset,
                   data_collator=collator, tokenizer=tokenizer)
trainer.train()
```
```python
# Stage 2: task-specific fine-tuning on (natural-language-query -> Cypher-query) pairs
final_prompt = f'{prompt1}English:' + data['Userquery'] + '\n' + prompt2 + '' + data['Cypher'].astype(str)
cypher_data = Dataset.from_pandas(pd.DataFrame({'text': final_prompt})).map(
    lambda d: tokenizer(d['text'], truncation=True), batched=True
).train_test_split(train_size=0.8)
# ... retrain with smaller batch size (8), more epochs (30), same Trainer pattern ...
```
- **What it demonstrates**: a two-stage LLM adaptation workflow — first domain-adaptive pretraining (GPT-2 on raw Cypher manuals) to improve contextual relevance, then supervised task-specific fine-tuning (natural language → Cypher query pairs) to sharpen output format, with measurable improvement in generated response quality at each stage.

## Reference Tables
**Transformer architecture types**

| Type | Attention direction | Example models | Typical tasks |
|---|---|---|---|
| Encoder-only | Bidirectional | BERT, RoBERTa, ALBERT, DistilBERT | Classification, NER, sentiment |
| Decoder-only | Causal/autoregressive | GPT-3/4, LLaMA, PaLM, Falcon, Chinchilla | Text generation, completion, QA |
| Encoder-decoder | Bidirectional (enc) + cross-attention (dec) | T5, BART, vision transformers | Translation, summarization, captioning |

**Tokenization strategies**

| Strategy | Unit | Strength | Weakness | Used by |
|---|---|---|---|---|
| Word | Whole words | Simple | OOV, vocab size, polysemy | Legacy NLP |
| Character | Single chars | No OOV | Long sequences, slow learning | — |
| BPE (sub-word) | Frequent byte pairs | Balances both | — | GPT models |
| WordPiece | Likelihood-merged pieces | Balances both | — | BERT |
| SentencePiece | Language-independent pieces | Balances both, language-agnostic | — | T5 |

**Decoding strategies**

| Strategy | Type | Behavior |
|---|---|---|
| Greedy search | Deterministic | Picks argmax token each step; fast, repetitive |
| Beam search | Deterministic | Tracks k candidate sequences; good for translation |
| Top-k sampling | Stochastic | Samples from k most-likely tokens |
| Top-p (nucleus) | Stochastic | Samples from smallest set covering probability mass p |
| Temperature | Stochastic modifier | Reshapes distribution sharpness before sampling |

**QA task types and evaluation**

| QA type | Description | Metric |
|---|---|---|
| Extractive | Answer is a span in the text | Exact match, F1, MRR |
| Long-form | Answer is a paragraph | ROUGE-like |
| Structured-data QA | Aggregation/counting over tables | Accuracy |
| Multi-hop | Multi-document reasoning | F1, MRR |
| Factual | Short fact extraction | Exact match |

## Worked Example
**GPT-2 domain adaptation for text-to-Cypher generation**: Parse four PDF Cypher-language manuals with PyPDF2 into a text corpus, split 80/20, and continue-pretrain GPT-2 (causal LM objective, `DataCollatorForLanguageModeling(mlm=False)`) for 20 epochs — this measurably improves the model's response to "Cypher query to create a customer node" compared to the vanilla pretrained model. Then collect 95 manually labeled (natural-language query → Cypher query) pairs from a spreadsheet, build a prompt template (`English: {query}\nCypher: {cypher}`), tokenize, and fine-tune the domain-adapted model for 30 more epochs with a small batch size (8, given the small dataset). The final fine-tuned model produces contextually and syntactically correct Cypher output for the same test query, demonstrating the progression: generic LLM → domain-pretrained LLM → task-fine-tuned LLM. A second worked example fine-tunes T5-small on CNN-DailyMail for abstractive summarization, evaluated via ROUGE-1/2/L/Lsum scores, showing measurable score improvement (e.g., ROUGE-1 0.35→0.37) after fine-tuning.

## Key Takeaways
1. Transformers beat RNN/LSTM primarily because of parallelizable training (no sequential recurrence) and superior long-range dependency handling via attention, at the cost of O(n²) attention complexity.
2. The choice among encoder-only, decoder-only, and encoder-decoder architectures should be driven by task shape, not by which model is most popular — representation extraction vs. open-ended generation vs. sequence transformation.
3. Virtually any NLP task (sentiment, QA, summarization) can be reframed as conditional next-token-probability estimation, which is why decoder-only LLMs generalize so broadly.
4. Decoding strategy is a separate lever from model architecture — the same trained model produces very different outputs depending on greedy vs. beam vs. top-k/top-p/temperature sampling.
5. RAG is the standard mitigation for hallucination and knowledge-staleness in generative QA, grounding generation in retrieved documents rather than relying purely on parametric memory.
6. Domain-adaptive pretraining followed by task-specific fine-tuning is a reliable two-stage recipe for adapting a general LLM to a narrow, specialized task with limited labeled data.
7. Context-length scaling has a real memory cost (roughly exponential past ~4096 tokens) — sparse attention and retrieval are the practical alternatives to brute-force context expansion.

## Connects To
- **Ch 2-3**: layer normalization, residual connections, and embeddings introduced in deep learning/CV chapters reappear as direct building blocks of transformer encoder/decoder blocks.
- **Ch 4**: VQ-GAN's transformer-over-codebook and CLIP's text-image contrastive embedding foreshadow the attention and embedding mechanisms detailed fully here.
- **Ch 6**: this chapter's transformer/decoder-only foundation and prompt-based task framing become the basis for ChatGPT's architecture and prompt-engineering practice.
- **Ch 7**: RAG, introduced here as a QA technique, is expanded into a full LangChain-based retrieval-augmented-generation pipeline.
