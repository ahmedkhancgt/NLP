# Glossary

**Agent (LangChain)** — Software that acts independently to achieve a goal by interacting with an LLM and external resources (databases, APIs, files) based on user queries (Ch5, Ch9).

**Anthropic ethical constitution** — A set of guidelines inspired by the Universal Declaration of Human Rights governing Claude's behavior (Ch4).

**Autonomous AI agent** — A system that understands objectives, creates and executes tasks, adapts priorities, and learns from actions until a goal is reached, without per-step human prompting (Ch5).

**BLEU score** — A metric evaluating machine translation quality; higher scores indicate better translation (Ch12).

**Blue-green deployment** — Deploying a new application version to a separate production environment, then switching traffic over, enabling zero-downtime releases (Ch12).

**Canary deployment** — Rolling out a new application version to a small subset of users/environment before full release, to catch problems early (Ch12).

**Chain-of-Thought (CoT) reasoning** — Prompting technique that asks the model to think through a problem in smaller steps before answering, improving accuracy (Ch3).

**Chroma** — An open-source vector database used to store and retrieve document embeddings in RAG pipelines (Ch7, Ch9).

**Chunking** — Breaking large documents/text into smaller pieces for embedding and storage in a vector database (Ch5, Ch7, Ch9).

**Code Llama** — Meta's programming-specialized Llama 2 variants (7B, 13B, 34B, 70B parameters) (Ch4, Ch10).

**Context window** — The maximum number of tokens a model can process in a single call, shared between input and output (Ch4).

**CTransformers** — Python library used to load and run local quantized Llama 2 `.gguf` model files (Ch10).

**Delimiter** — A marker (triple quotes, triple backticks, XML tags, etc.) used to separate distinct parts of a prompt, primarily to prevent prompt injection (Ch3, Ch6).

**Diffusion model** — An image-generation technique that starts with random noise and iteratively refines it toward an image matching a text prompt (Ch1, Ch2).

**Discriminative model** — A model that learns P(y|x) to classify/label data, as opposed to generating new data (Ch1).

**Fine-tuning** — Additional training of a pretrained model on task-specific data to improve domain accuracy, consistency, and reduce required prompt length (Ch4, Ch5, Ch6).

**Flowise AI** — An open-source, no-code drag-and-drop platform for visually building LangChain applications/chatbots (Ch11).

**Foundation model** — A model pretrained on massive unsupervised/unstructured data that transfers across many downstream tasks (Ch1, Ch4).

**Function calling (OpenAI)** — A feature letting an LLM output a structured JSON object matching a developer-defined function signature, for connecting LLMs to external tools/APIs (Ch6).

**GAN (Generative Adversarial Network)** — A generator/discriminator pair trained adversarially; the generator creates data, the discriminator tries to distinguish real from generated data (Ch1, Ch2).

**Gemini Pro** — Google's multimodal LLM supporting text/image input, used in this book's RAG chapter via LangChain (Ch4, Ch7).

**Generative model** — A model that learns the joint distribution P(x,y) to create new data instances resembling training data (Ch1).

**Gradio** — An open-source Python library for quickly building interactive web UIs for ML models and LLM-backed applications (Ch8).

**gr.ChatInterface** — The Gradio component providing a full-featured chat UI with minimal code (Ch8).

**Hallucination** — A model-generated output that is grammatically incorrect, nonsensical, or factually wrong, often from insufficient data, missing context, or missing constraints (Ch1, Ch3, Ch13).

**Khanmigo** — Khan Academy's GPT-4-powered tutoring conversational agent (Ch2, Ch6).

**LangChain** — An open-source Python/JavaScript framework for connecting LLMs to external data sources and building data-aware, chainable LLM applications (Ch3, Ch4, Ch7, Ch9, Ch10, Ch11).

**LangSmith** — LangChain's companion product for visual tracing, monitoring, and evaluation of LangChain application calls (prompts, responses, tokens, latency, cost) (Ch10, Ch12).

**Llama 2** — Meta's open-source LLM family (7B/13B/70B parameters), pretrained on ~2 trillion tokens with a 4,096-token context length (Ch4, Ch10).

**LLMOps** — Large Language Model Operations; practices for developing, deploying, and managing LLMs in production, adapted from MLOps with LLM-specific concerns (Ch5, Ch12).

**Moderation API (OpenAI)** — A free content-classification endpoint flagging harmful categories (hate, self-harm, harassment, violence, sexual content) in user input/model output (Ch6).

**Multimodal language model (MLLM/MMLM)** — A model that processes/generates across multiple modalities (text, image, audio, video) rather than a single modality (Ch1, Ch4, Ch5).

**Non-deterministic** — A property of generative AI outputs where the outcome is not predictable or guaranteed to repeat identically, unlike traditional ML's traceable prediction path (Ch13).

**Panel (pn)** — A Python GUI library used to build interactive chatbot interfaces in a Jupyter Notebook (Ch6, Ch8).

**Prompt chaining** — Using the output of one model completion as input/context for the next prompt, to maintain conversational context across turns (Ch6).

**Prompt engineering** — The practice of crafting clear, specific, well-structured prompts to guide a language model toward a desired output (Ch3).

**Prompt injection** — An attack where user-supplied text contains conflicting instructions intended to override a system's original prompt instructions (Ch3, Ch6).

**Prompt template** — A reusable, parameterized prompt structure with placeholders (Instructions, Context, User input, Output indicator) filled at runtime, implemented in LangChain via `PromptTemplate` (Ch3, Ch7, Ch10).

**Quantization** — A technique reducing a model's numeric precision to shrink size/compute requirements, enabling it to run on less powerful hardware at some accuracy cost (Ch10).

**RAG (Retrieval Augmented Generation)** — A technique combining an LLM with an information retrieval system (typically a vector database) so responses are grounded in retrieved external data rather than relying solely on pretraining (Ch5, Ch7).

**Red teamers** — Community contributors who probe open-source models for harms and validate safety mitigations (Ch10).

**RetrievalQA (LangChain)** — A chain type combining a retriever (vector DB) and an LLM into a single question-answering pipeline (Ch7).

**RLHF (Reinforcement Learning via Human Feedback)** — An LLM training technique incorporating end-user/human feedback into model refinement, especially important for open-ended tasks (Ch12).

**ROUGE score** — A metric evaluating text summarization quality; higher scores indicate better capture of the original text's main ideas (Ch12).

**Software 2.0** — Andrej Karpathy's term for neural-network-driven software development, where code is effectively "written" by training rather than hand-coded (Ch4).

**Streamlit** — A Python framework for building data dashboards and interactive web apps without backend/HTML/CSS coding (Ch8, Ch9, Ch10).

**Temperature** — A model parameter controlling response randomness/creativity; lower values produce more deterministic, factual outputs (Ch7, Ch10).

**Together.AI** — A serverless cloud platform offering hosted inference for open-source models like Code Llama, used as an alternative to local model hosting (Ch10).

**Token** — A word-fragment unit the LLM API processes internally; approximately 4 characters or ¾ of a word in English; the basis for cost, latency, and context-limit calculations (Ch4).

**Transformative use** — Modifying/repurposing existing copyrighted content to add new meaning, context, or value, more likely protected under fair use (Ch13).

**Derivative work** — New content substantially based on existing material without transforming its purpose; generally requires the original author's consent (Ch13).

**Transformer** — The encoder-decoder neural network architecture (2018+) underlying modern generative AI/LLMs (Ch1).

**VAE (Variational Autoencoder)** — A generative architecture that compresses input into a lower-dimensional latent representation, then generates new samples from that learned latent space (Ch1).

**Vector database** — A database (Pinecone, Chroma, FAISS, etc.) optimized to store and query embeddings for similarity search (Ch4, Ch5, Ch7, Ch9).

**Vector embedding** — Data (words, images) converted into a numerical array capturing semantic relationships, where similar concepts cluster close together (Ch4, Ch5, Ch7).

**Whisper** — OpenAI's automatic speech recognition system trained on 680,000 hours of multilingual audio data (Ch4).
