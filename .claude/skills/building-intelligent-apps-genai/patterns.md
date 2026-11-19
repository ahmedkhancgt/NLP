# Patterns

## ChatGPT API Integration Pattern
**When to use**: Building a conversational chatbot/virtual assistant backed by OpenAI's Chat Completions API.
**How**: Structure every call with `system`/`user`/`assistant` role messages; store the API key in a `.env` file (never hard-code); run both user input and model output through the Moderation API; wrap untrusted user text in delimiters (triple backticks) to prevent prompt injection; maintain `all_messages` conversation history across turns (prompt chaining) for context; build a UI with Panel or Gradio.
**Trade-offs**: Fast to prototype and production-ready within weeks via prompt engineering alone; requires explicit safety layers (moderation + injection defense) that are not automatic; ongoing per-token API costs.

## Gemini-Pro-Based RAG Pattern
**When to use**: Building a document-grounded Q&A application where answers must come from specific authoritative documents (PDFs, spreadsheets) rather than the model's general knowledge.
**How**: Chunk the source document with `RecursiveCharacterTextSplitter` (e.g., 10,000 chars / 1,000 overlap); embed chunks with `GoogleGenerativeAIEmbeddings`; store/index in a Chroma vector database; configure `ChatGoogleGenerativeAI` with low `temperature` (e.g., 0.2) for factual consistency; wire a LangChain `RetrievalQA` chain over the vector index; add an explicit anti-hallucination `PromptTemplate` instructing the model to say "I don't know" when context is insufficient.
**Trade-offs**: Dramatically reduces hallucination on domain-specific questions and requires no model retraining; adds pipeline complexity (chunking, embedding, indexing) and ongoing vector-DB maintenance; swappable with other LLMs (Llama 2) or vector stores (Pinecone) per project needs.

## Gradio Demo-App Pattern
**When to use**: Quickly exposing an ML model or LLM pipeline (summarization, image captioning, chat) as an interactive, shareable web demo without writing HTML/CSS/backend code.
**How**: Write a plain Python function performing the model inference (tokenize → generate → decode → return consistent dict); wrap it with Gradio input/output components (`Textbox`, `Upload`, `gr.ChatInterface`); call `.launch(share=True)` for a local + public URL; optionally host permanently on Hugging Face Spaces.
**Trade-offs**: Minimal code for a working demo and easy sharing; chat demos built this way lack moderation unless explicitly added; best suited to specialist/open-source models (DistilBART, BLIP) for cost efficiency on narrow tasks.

## LangChain+Streamlit Data-Viz Pattern
**When to use**: Building a natural-language interface over tabular/CSV data that should automatically render as the right chart or table type.
**How**: Wrap the uploaded CSV in a `create_pandas_dataframe_agent`; send a structured prompt instructing the model to reply in one of fixed JSON shapes (`{"bar": {...}}`, `{"line": {...}}`, `{"table": {...}}`, `{"answer": "..."}`); in Streamlit, `decode_response` parses the JSON and `write_response` dispatches on the key to the matching widget (`st.bar_chart`, `st.line_chart`, `st.table`, `st.write`).
**Trade-offs**: Lets the LLM's output double as a UI-routing instruction, enabling zero-custom-chart-logic visualization; relies on the model consistently honoring the JSON contract (not guaranteed for formatting details like rounding); large CSVs risk hitting token limits.

## Local Llama 2 Application Pattern
**When to use**: Fully offline/private LLM applications where data cannot leave the local environment, or cost/control matters more than raw capability.
**How**: Download a quantized `.gguf` Llama 2 model file from Hugging Face; load it with `CTransformers` (`model_type='llama'`, low temperature for consistency); fill a LangChain `PromptTemplate` with user-supplied variables; build a Streamlit UI collecting those variables; generate on demand.
**Trade-offs**: No internet/API dependency after model download, full data control, no per-call API cost; requires local compute (quantization trades some accuracy for the ability to run on modest hardware); smaller parameter models (e.g., 7B) yield noticeably lower quality than larger/proprietary models.

## Serverless Code Llama Pair-Programming Pattern
**When to use**: Code generation, technical documentation, and debugging assistance without hosting model weights yourself.
**How**: Obtain an API key from a serverless host (Together.AI, DeepInfra, Anyscale); store credentials in `.env`; install `langchain-together`; build `ChatPromptTemplate` (system message defining the assistant's role + user message with the specific coding task) → chain → `StrOutputParser`; format output with a markdown helper.
**Trade-offs**: Avoids local hosting of large Code Llama variants while keeping cost low (e.g., $25 free credit); requires detailed, specific prompts to avoid hallucinated code — vague prompts increase hallucination risk significantly.

## Flowise AI No-Code Chatbot Pattern
**When to use**: Rapid prototyping of a document-grounded chatbot by non-technical users or for fast idea validation, with zero hand-written code.
**How**: Install Node.js + Flowise (`npm install -g flowise`); launch with `npx flowise start`; select a Marketplace template (e.g., "Claude LLM") rather than building from a blank canvas; configure the LLM node's API credentials; upload the source PDF to the PDF File node; save; test via the built-in Chat window; optionally expose as an API endpoint for external integration.
**Trade-offs**: Fastest path from zero to a working document chatbot, accessible to non-developers; less flexible/customizable than hand-coded LangChain pipelines; response quality depends heavily on source document structure (clear headings/focused topics improve results; unstructured/unowned PDFs may yield less accurate answers).

## LLMOps Development-to-Production Pipeline
**When to use**: Taking any LLM-powered application from prototype to a monitored, production-grade deployment.
**How**: Select a foundational LLM → iterate on prompt engineering → evaluate results across varied inputs → fine-tune if needed → deploy via canary (small subset first) or blue-green (parallel environment, then switch) strategy → monitor accuracy/latency/error-rate metrics (e.g., via LangSmith) with a continuous feedback loop back into the cycle.
**Trade-offs**: Reduces production risk and surfaces prompt/cost/latency issues before they affect all users; adds process overhead compared to "just ship the prototype," which is appropriate to skip only for throwaway demos.

## Anti-Hallucination Prompt Contract Pattern
**When to use**: Any RAG or document-Q&A application where ungrounded, confidently-wrong answers are unacceptable.
**How**: Build a `PromptTemplate` that explicitly instructs: "Use the following pieces of context to answer the question... If you don't know the answer, just say that you don't know, don't try to make up an answer." Pass retrieved `{context}` and the `{question}` into the template every time.
**Trade-offs**: Measurably reduces hallucination on out-of-scope queries (verified in the book's Gemini Pro example asking about "random forest" against an unrelated document); does not guarantee zero hallucination on in-scope but ambiguous queries — pair with fact-checking/human review for high-stakes use.

## Bias-Mitigation Strategy Stack
**When to use**: Any generative AI project at the data-collection and model-evaluation stages, to reduce demographic/societal bias in generated content.
**How**: Collect diverse/representative data → remove biased data points + normalize → measure with fairness metrics → augment with oversampling/undersampling/synthetic data → stratified sampling for subgroup representation → anonymize sensitive attributes (watch for correlated proxies) → increase transparency/interpretability → conduct regular audits.
**Trade-offs**: Substantially reduces (but does not eliminate) bias risk; fairness metrics are themselves subjective/context-dependent, so this requires ongoing vigilance rather than a one-time fix.
