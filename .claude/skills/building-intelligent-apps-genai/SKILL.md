---
name: building-intelligent-apps-genai
description: "Knowledge base from \"Building Intelligent Applications with Generative AI\" by Yattish Ramhorry (BPB Publications, 2025). Use when building hands-on generative AI applications — prompt engineering, integrating GenAI models, or building with specific tools like the ChatGPT API, Gemini Pro (RAG), Gradio, LangChain+Streamlit, Llama 2, or Flowise AI, plus best practices and ethical considerations."
---

# Building Intelligent Applications with Generative AI
**Author**: Yattish Ramhorry | **Chapters**: 13 | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a tool or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch08`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Generative AI taxonomy (Ch1-2)**: AI → ML → Deep Learning → Generative AI. A task is generative AI only if the output y is natural language, image, audio, or video — not a number, discrete class, or probability. GANs (generator vs. discriminator, adversarial) and VAEs (latent-space compression) are the two classic generative architectures.

**Two prompting guidelines (Ch3)**: (1) write clear and specific instructions, (2) give the model time to think. Techniques built on these: delimiters (prevent prompt injection), structured output requests, few-shot examples, Chain-of-Thought step-breakdown, persona/system messages, and LangChain `PromptTemplate`s. Follow the 7-step iterative process: define goal → write initial prompt → test → analyze → refine → iterate → implement.

**Model-customization escalation ladder (Ch4-5, Ch12)**: (1) prompt engineering (cheapest) → (2) RAG (add external retrieval for missing domain knowledge) → (3) fine-tuning (needs ~50-100+ examples; improves consistency/format/tone, reduces prompt size up to 90%) → (4) pretrain a new foundation model from scratch (most expensive, rarely justified). Token economics: 1 token ≈ 4 characters/¾ word; both input and output tokens count toward cost and context limits.

**RAG pipeline (Ch5, Ch7)**: chunk documents → embed chunks (vector embeddings) → store/index in a vector database (Chroma, Pinecone, FAISS) → similarity search retrieves relevant chunks for a query → LLM generates a grounded response. Always add an explicit "say I don't know" prompt template to prevent hallucination on out-of-scope queries. RAG ≠ fine-tuning: RAG retrieves at inference time; fine-tuning retrains weights.

**Tool-specific build pattern (Ch6-11)** — each chapter is a named, hands-on application build:
- **Ch6 ChatGPT API**: system/user/assistant message roles + Moderation API + delimiter-based injection defense + prompt chaining + Panel UI. Built a dinosaur-education chatbot.
- **Ch7 Gemini Pro RAG**: LangChain + `ChatGoogleGenerativeAI` + `GoogleGenerativeAIEmbeddings` + Chroma + `RetrievalQA`. Built a career-advice document Q&A app.
- **Ch8 Gradio**: plain Python function → Gradio components → `.launch()`. Built text summarization (DistilBART), image captioning (BLIP), and chat (`gr.ChatInterface`) demos.
- **Ch9 LangChain+Streamlit**: `create_pandas_dataframe_agent` + a bar/line/table/answer JSON-contract prompt + Streamlit dispatch (`st.bar_chart`/`st.line_chart`/`st.table`/`st.write`). Built "Chat with your CSV."
- **Ch10 Llama 2**: local quantized `.gguf` model via `CTransformers` (blog generator with Streamlit) + serverless Code Llama via Together.AI (pair-programming assistant) + LangSmith tracing/evaluation.
- **Ch11 Flowise AI**: no-code drag-and-drop canvas, Marketplace template ("Claude LLM") + PDF upload = instant document chatbot, zero code.

**Open-source vs. proprietary decision (Ch10, Ch12)**: proprietary (GPT-4/Claude/Gemini) wins on time-to-market, baseline quality, minimal infra; open-source (Llama 2/Mistral/Falcon) wins on data privacy, cost predictability, fine-tuning cost, no vendor lock-in. Time-constrained → proprietary; data-sensitive/long-horizon → open-source. Prefer Apache 2.0 then MIT licenses for commercial open-source use.

**LLMOps lifecycle (Ch12)**: select LLM → prompt engineering → evaluate results → fine-tuning → deployment (canary or blue-green) → monitoring (accuracy/latency/error rate, e.g. via LangSmith) → continuous feedback loop. LLMOps differs from MLOps via: GPU/compute needs, transfer learning (fine-tune a foundation model vs. train from scratch), RLHF, cost-aware hyperparameter tuning, LLM-specific metrics (BLEU for translation, ROUGE for summarization), prompt engineering, and chain/pipeline construction (LangChain/LlamaIndex).

**Ethics framework (Ch13)**: embed ethics at every lifecycle stage (never just start/end) via 3 pre-code questions — define the problem, assess impact, evaluate bias. Mitigate bias with diverse data + fairness metrics + synthetic data + audits. Mitigate hallucination across 3 layers: data/prompt, model/architecture, application design. Assess copyright risk via transformative use (new meaning added, more likely fair use) vs. derivative work (substantially based on original, needs consent).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-exploring-the-world-of-generative-ai.md) | Exploring the World of Generative AI | AI/ML/DL/GenAI hierarchy, discriminative vs. generative, GANs, VAEs, transformers |
| [ch02](chapters/ch02-use-cases-for-generative-ai-applications.md) | Use Cases for Generative AI Applications | Content gen (Jasper, Copy.ai), image gen (DALL-E, Midjourney, Leonardo.Ai), conversational agents, VR/gaming |
| [ch03](chapters/ch03-mastering-the-art-of-prompt-engineering.md) | Mastering the Art of Prompt Engineering | Delimiters, few-shot, Chain-of-Thought, personas, prompt templates, 7-step iteration |
| [ch04](chapters/ch04-integrating-generative-ai-models-into-applications.md) | Integrating Generative AI Models into Applications | LLM API stack, tokens, OpenAI/Claude/Gemini/Llama 2 APIs, GitHub Copilot |
| [ch05](chapters/ch05-emerging-trends-and-the-future-of-generative-ai.md) | Emerging Trends and the Future of Generative AI | Autonomous agents, RAG, fine-tuning, vector embeddings, few-shot learning |
| [ch06](chapters/ch06-building-intelligent-applications-with-the-chatgpt-api.md) | Building Intelligent Applications with the ChatGPT API | System/user/assistant roles, moderation, prompt injection defense, function calling |
| [ch07](chapters/ch07-retrieval-augmented-generation-with-gemini-pro.md) | Retrieval Augmented Generation with Gemini Pro | Gemini Pro + LangChain RAG pipeline, Chroma, anti-hallucination prompt template |
| [ch08](chapters/ch08-generative-ai-applications-with-gradio.md) | Generative AI Applications with Gradio | Gradio demo-app pattern, DistilBART summarization, BLIP captioning, chat UI |
| [ch09](chapters/ch09-visualize-your-data-with-langchain-and-streamlit.md) | Visualize your Data with LangChain and Streamlit | Pandas DataFrame agent, JSON-contract prompt, Streamlit chart dispatch |
| [ch10](chapters/ch10-building-llm-applications-with-llama-2.md) | Building LLM Applications with Llama 2 | Local Llama 2 (quantization, CTransformers), serverless Code Llama, LangSmith |
| [ch11](chapters/ch11-building-an-ai-document-chatbot-with-flowise-ai.md) | Building an AI Document Chatbot with Flowise AI | No-code Flowise canvas, Marketplace templates, Claude LLM node |
| [ch12](chapters/ch12-best-practices-for-building-applications-with-generative-ai.md) | Best Practices for Building Applications with Generative AI | Model selection framework, open-source vs. proprietary, LLMOps |
| [ch13](chapters/ch13-ethical-considerations-of-generative-ai.md) | Ethical Considerations of Generative AI | Fairness, bias mitigation, hallucination, copyright, environmental impact |

## Topic Index
- **Autonomous agents** → Ch5
- **Bias mitigation** → Ch13 (strategy stack), Ch12 (checklist)
- **ChatGPT API** → Ch3 (prompting basics), Ch4 (integration), Ch6 (full build)
- **Claude (Anthropic)** → Ch4, Ch11
- **Code Llama** → Ch10
- **Copyright / transformative use** → Ch13
- **Deepfakes** → Ch13
- **Fine-tuning** → Ch4, Ch5, Ch6
- **Flowise AI** → Ch11
- **Function calling (OpenAI)** → Ch6
- **Gemini Pro** → Ch4, Ch7
- **Gradio** → Ch8
- **Hallucination** → Ch1, Ch3, Ch5, Ch7, Ch13
- **LangChain** → Ch3 (templates), Ch7 (RAG), Ch9 (agents), Ch10
- **LangSmith** → Ch10, Ch12
- **Llama 2** → Ch4, Ch10
- **LLMOps** → Ch5, Ch12
- **Moderation API** → Ch6
- **Open-source vs. proprietary LLMs** → Ch10, Ch12
- **Prompt engineering techniques** → Ch3 (primary), Ch6, Ch10
- **Prompt injection** → Ch3, Ch6
- **RAG (Retrieval Augmented Generation)** → Ch5 (concept), Ch7 (Gemini Pro build)
- **Streamlit** → Ch8, Ch9, Ch10
- **Tokens** → Ch4
- **Vector embeddings / vector databases** → Ch4, Ch5, Ch7, Ch9

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — tool-specific application patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 13 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive, as are any recurring publisher promotional asides. For topics beyond this book's scope, check other skills or ask directly.
