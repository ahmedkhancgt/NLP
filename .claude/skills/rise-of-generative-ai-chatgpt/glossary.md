# Glossary

**ALBERT** — A parameter-efficient BERT variant using factorized embedding parameterization and cross-layer parameter sharing; 18x fewer parameters, 1.7x faster training than BERT (Ch 8).

**API pricing (ChatGPT/GPT-3.5)** — Token-based cost model (~$0.0004-$0.002 per 1,000 tokens depending on model tier) for integrating GPT into third-party applications (Ch 10).

**BERT (Bidirectional Encoder Representations from Transformers)** — A bidirectional transformer model using masked-token training; strong on sentiment analysis and chatbot tasks (Ch 7, Ch 8).

**Bias and discrimination (AI)** — The risk that generative models trained on biased data perpetuate or amplify stereotypes/discriminatory outcomes (Ch 4, Ch 6, Ch 20).

**BLOOM** — An open, industrial-scale autoregressive LLM supporting 46 languages and 13 programming languages, competing with GPT (Ch 30).

**ChatGPT Plus** — ChatGPT's $20/month premium subscription tier offering peak-time access, faster responses, and priority feature access (Ch 29).

**ChatSonic / Jasper / Character AI / Perplexity AI** — Named consumer-facing ChatGPT competitor products, each differentiated by niche (general chat, business content, persona impersonation, sourced search) (Ch 30).

**Code summarization** — Using ChatGPT to generate concise summaries of code snippets/functions, aiding onboarding and technical-debt identification (Ch 16).

**Decision science** — An interdisciplinary field (math, statistics, economics, psychology) for structured, bias-mitigated decision-making (Ch 17).

**Deepfake** — AI-generated, convincingly altered images/video/audio used maliciously for impersonation or disinformation (Ch 4, Ch 6, Ch 25).

**Fill-In-the-Middle (FIM) training** — A training-data transformation technique moving a text span from the middle to the end of a document, teaching models to infill content while preserving left-to-right generation (Ch 9).

**Fintech** — Financial technology (software, apps, tools) that streamlines financial services and improves customer experience, disrupting traditional banking (Ch 3, Ch 19).

**GAN (Generative Adversarial Network)** — A deep-learning generative architecture (2010s) capable of producing highly realistic, complex outputs such as photorealistic images (Ch 2).

**GDPR (General Data Protection Regulation)** — EU data-protection regulation cited as a compliance requirement for AI systems handling customer/personal data (Ch 3, Ch 4).

**Generative AI** — A subfield of AI that creates new content/data from inputs using techniques like deep learning and neural networks, as opposed to only classifying/retrieving existing content (Ch 2).

**GPT (Generative Pre-training Transformer)** — The transformer neural network architecture underlying ChatGPT, trained on large-scale text/conversation data to generate human-like responses (Ch 1).

**GPT-1** — The first GPT model (June 2018, 117M parameters, 12 decoder layers), trained via 3-stage pre-training/fine-tuning (Ch 9).

**GPT-2** — The second GPT generation (Feb 2019, 1.5B parameters), introducing task conditioning and zero-shot task transfer (Ch 9).

**GPT-3** — The third GPT generation (May 2020, 175B parameters), evaluated via zero/one/few-shot in-context learning rather than fine-tuning (Ch 9).

**GPT-3.5 / InstructGPT** — GPT-3 fine-tuned via Reinforcement Learning from Human Feedback (RLHF); preferred over raw GPT-3 ~85% of the time by human labelers (Ch 9).

**GPT-4** — The multimodal (image+text input, text output) successor to GPT-3.5, with human-level performance on many professional/academic benchmarks and improved safety alignment (Ch 32).

**Hallucination (confidently-wrong answers)** — ChatGPT producing plausible-sounding but factually/logically incorrect output, rooted in training incentives rather than a simple bug (Ch 10).

**Hidden Markov Model (HMM)** — A 1980s-90s probabilistic model used for sequential decision-making, an early step in generative AI's evolution (Ch 2).

**InstructGPT** — See GPT-3.5.

**LaMDA** — Google's ~137B-parameter conversational model built for free-form dialogue rather than task-based replies (Ch 7).

**Masked Language Modeling (MLM)** — A training objective where some input tokens are hidden and the model must predict them from context, building general-purpose language representations (Ch 1).

**Megatron (MT-NLG)** — NVIDIA's 530B-parameter language model, strong on logical reasoning and mathematical inference benchmarks (Ch 7, Ch 8).

**Moderation API** — OpenAI's mechanism for flagging/blocking harmful content categories in ChatGPT output (Ch 10).

**Natural Language Processing (NLP)** — The AI subdomain focused on computer understanding, interpretation, and generation of human language (Ch 7).

**Natural language programming** — Writing code intent in plain language rather than a specific programming language, making coding accessible to non-technical stakeholders (Ch 16).

**NLM (Neural Language Model)** — The chapter 23 term for ChatGPT's underlying model category, emphasizing its problem-solving (not just lookup) capability (Ch 23).

**OPT (Open Pre-trained Transformers)** — A suite of decoder-only transformers (125M-175B parameters) comparable to GPT-3 at roughly 1/7th the carbon footprint (Ch 30).

**PaLM (Pathways Language Model)** — Google's 540B-parameter multimodal, sparsely-activated model outperforming on 28/29 major NLP benchmarks (Ch 7, Ch 8, Ch 30).

**PCI DSS (Payment Card Industry Data Security Standard)** — Payment-security compliance standard relevant to financial-sector AI systems (Ch 3).

**Prompt sensitivity** — ChatGPT's tendency to give different-quality answers to the same underlying question depending on exact phrasing (Ch 10).

**RLHF (Reinforcement Learning from Human Feedback)** — The alignment technique (SFT → Reward Modeling → PPO) used to fine-tune GPT-3 into InstructGPT/GPT-3.5 (Ch 9).

**RoBERTa** — A BERT variant trained on ~10x more data with longer training and larger batches, outperforming BERT on most benchmarks (Ch 8).

**Robo-advisory** — ChatGPT analyzing financial data/market patterns to generate automated investment recommendations (Ch 19).

**Task conditioning** — GPT-2's mechanism of supplying task description + input together in natural language to enable zero-shot task transfer without separate fine-tuning (Ch 9).

**Transfer learning** — Pre-training a model on a large generic corpus, then fine-tuning on a smaller labeled dataset for a specific task (Ch 7, Ch 9).

**Transformer architecture** — Google's 2017 self-attention-based neural network architecture ("Attention is all you need") underlying GPT, BERT, and most modern LLMs (Ch 1, Ch 7).

**VAE (Variational Autoencoder)** — A deep-learning generative architecture (2010s) enabling generation of complex, diverse outputs alongside GANs (Ch 2).

**Visual Question Answering (VQA)** — GPT-4's capability to perform visual reasoning/logic extraction from images with accuracy comparable to its text performance (Ch 32).

**Whisper** — OpenAI's automatic speech recognition model trained on 680,000 hours of multilingual audio, using mel-spectrogram chunks through an encoder-decoder transformer (Ch 9).

**XLNet** — A model combining TransformerXL's autoregressive approach with BERT's bidirectionality; outperformed BERT on 20 of the tested NLP tasks (Ch 8).

**Zero/one/few-shot learning** — Prompting paradigms where a model is given zero, one, or several examples (plus a task description) in-context, without separate fine-tuning (Ch 9).
