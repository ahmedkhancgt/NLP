# Chapter 4: Integrating Generative AI Models into Applications

## Core Idea
Building generative AI applications requires choosing an appropriate foundation model/API (OpenAI, Claude, Gemini, Llama 2), understanding token economics, and properly preparing data — the LLM API stack (APIs → vector DBs → orchestration frameworks like LangChain) is the common architecture pattern companies converge on.

## Frameworks Introduced
- **LLM API stack (adoption-ranked)**: LLM APIs (most adopted) → retrieval/vector databases (Pinecone, Qdrant, Milvus) → orchestration frameworks (LangChain) → custom/open-source model training (least adopted).
  - When to use: deciding where to invest engineering effort first when building a GenAI product — start with APIs, add retrieval, then orchestration, only build custom models last.
  - How: per Sequoia Capital survey cited: 65% of companies had GenAI in production; 94% use a foundation model; 88% expect vector DBs to stay core to their stack; 38% use orchestration like LangChain; only 15% train custom models.
- **OpenAI library pattern (Chat, Embeddings, Fine-tuning APIs)**: Chat API for conversational/completion tasks; Embeddings API for vector representations (search/classification/clustering); Fine-tuning API for domain adaptation.
  - When to use: Chat API for building chatbots/assistants and text tasks; Embeddings for similarity/search; Fine-tuning only when prompting has been exhausted.
- **Claude (Anthropic) SDK pattern**: Python/TypeScript SDKs calling Anthropic's API, governed by an "ethical constitution" inspired by the Universal Declaration of Human Rights.
  - When to use: tasks needing a large context window (100K tokens / ~75K words) or where response safety/ethics framing matters.
- **Token accounting framework**: treat tokens (not words) as the true unit of cost, latency, and context-limit — 1 token ≈ 4 characters or ¾ of a word in English.
  - When to use: before designing any prompt or chat loop, to estimate cost and avoid truncation.

## Key Concepts
- **Token**: a word-fragment the API processes internally; tokens can include leading/trailing spaces and sub-words, and the same word tokenizes differently depending on casing/position in a sentence.
- **Context window**: the maximum tokens a model can process in one call (e.g., GPT-4 ≈ 3,000 words vs. Claude ≈ 75,000 words/100K tokens at time of writing; Gemini Pro 1.5 standard 128K tokens, preview up to 1M tokens).
- **Vector embedding**: a numerical array representation of data (words/images) capturing semantic relationships, produced via an Embeddings API.
- **Fine-tuning**: additional training of a pretrained model on task-specific data to improve domain accuracy and reduce required prompt length (available at the time for `gpt-3.5-turbo-0613`, `babbage-002`, `davinci-002`).
- **Mixture of Experts (MoE) architecture**: used by Gemini Pro 1.5 for more efficient training/serving.
- **Software 2.0**: Andrej Karpathy's term for neural-network-driven software development (code "written" by training rather than hand-coded), underpinning tools like GitHub Copilot.
- **Multimodal model**: accepts multiple input types (e.g., GPT-4 accepting image + text to produce text).

## Mental Models
- Think of tokens as a different "dictionary" than words — the same word can map to different token IDs depending on capitalization and sentence position, so token-counting is not simply word-counting.
- Use the "prompt engineering → fine-tuning → pretraining from scratch" ladder as an escalation path: try the cheapest fix first (better prompts), then fine-tune, and only pretrain a new foundation model as a last, costly resort.
- Treat GitHub Copilot/AI pair-programming as augmentation, not replacement — it's meant to remove boilerplate so developers focus on business logic.

## Anti-patterns
- **Ending a prompt with a trailing space**: trailing whitespace is tokenized and can degrade response quality — keep prompts concise and trimmed.
- **Ignoring token limits**: exceeding a model's max token count (shared between input + output) causes truncation or incomplete responses; a 4,097-token model given a 4,000-token prompt only has 97 tokens left for the answer.
- **Jumping straight to fine-tuning or full pretraining**: the book stresses prompt engineering should be exhausted first since pretraining a foundation model from scratch can cost millions of dollars.
- **Treating vector embeddings storage as solved by the embeddings API alone**: OpenAI's Embeddings API generates vectors but provides no built-in storage — a separate vector database (Pinecone, Chroma, FAISS, etc.) is required.

## Code Examples
```text
https://{your-resource-name}.openai.azure.com/openai/deployments/{deployment-id}/completions?api-version={api-version}
```

```text
My favorite color is red.
-> token ids: 3666, 4004, 3124, 318, 2266, 13

My favorite color is Red.
-> token ids: 3666, 4004, 3124, 318, 2297, 13

Red is my favorite color.
-> token ids: 7738, 318, 616, 4004, 3124, 13
```
- **What it demonstrates**: how identical words tokenize differently by case and sentence position (red=2266 mid-sentence lowercase, Red=2297 mid-sentence capitalized, Red=7738 at sentence-start), and the Azure OpenAI Completions API REST endpoint pattern for enterprise deployment.

## Reference Tables
| Model | Key stat (as reported in book) |
|---|---|
| GPT-3 | 175B parameters, 800GB storage, released 2020 |
| GPT-4 | Up to 1 trillion parameters, trained on 825TB data, multimodal (image+text in, text out) |
| Claude (Anthropic) | ~100,000 token context window (~75,000 words); Claude 3 suite = Opus, Sonnet, Haiku |
| Gemini Pro 1.5 | 128,000 token standard context; up to 1M tokens in private preview; MoE architecture; 40+ languages, 230+ countries |
| Llama 2 | 7B–70B parameter range; 40% more training data than Llama 1, double the context length |
| Whisper | Trained on 680,000 hours of multilingual/multitask audio; open-sourced |

| Sequoia Capital survey finding | Stat |
|---|---|
| Companies with GenAI in production | 65% |
| Companies using a foundation model | 94% |
| Believe vector DB will remain core to stack | 88% |
| Interested in LLM orchestration (LangChain) | 38% |
| Built custom models from scratch / used open-source | 15% |

## Worked Example
Using OpenAI's Chat Completion API (GPT-3.5-Turbo) to perform sentiment analysis of a fictitious Amazon product review: a prompt instructs the model to classify sentiment as positive/negative, demonstrating the same underlying API that powers ChatGPT. The chapter also walks through the Anthropic Claude SDK in both Python and TypeScript for calling Claude directly, and shows Google's Generative AI Studio (Vertex AI console) where developers can design/test prompts and optionally fine-tune a model when prompt engineering alone hasn't solved hallucination or accuracy problems.

## Key Takeaways
1. Architect GenAI apps around the adoption-ranked stack: LLM API first, then retrieval/vector DB, then orchestration (LangChain), reserving custom model training for last.
2. Budget for tokens, not words — count tokens (via OpenAI's Tokenizer tool) before finalizing prompt length, since cost/latency/limits are all token-denominated.
3. Escalate model customization in order: better prompting → fine-tuning → full pretraining (most expensive, rarely justified).
4. Choose a foundation model based on context-window needs and modality requirements (Claude's large context vs. GPT-4's multimodal input vs. Gemini's 1M-token preview).
5. Vector embeddings require a separate vector database (Pinecone, Chroma, FAISS, etc.) — the embeddings API alone does not persist them.
6. AI pair-programming tools (GitHub Copilot, CodeWhisperer) integrate into existing IDEs and are positioned to augment, not replace, developers.

## Connects To
- **Ch3**: Builds directly on the OpenAI API setup and prompting guidelines established there.
- **Ch5**: Expands on fine-tuning and vector embeddings introduced here into RAG and multimodal model discussion.
- **Ch6**: The ChatGPT API chapter is the hands-on implementation of the Chat API integration pattern described here.
- **Ch7**: Gemini Pro's RAG capability referenced here is built out fully in Chapter 7.
- **Ch10**: Llama 2's architecture/licensing described here is the subject of the full build chapter.
