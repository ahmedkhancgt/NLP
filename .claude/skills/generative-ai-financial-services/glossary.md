# Glossary

**ABAC (Attribute-Based Access Control)** — access control mapped to data's use-case and velocity rather than uniform roles, allowing open access to static reporting data while restricting fast-moving transactional data. (Ch6)

**Agent** — an autonomous software application with access to an LLM, memory, and a prompt with instructions/tools, which loops reasoning and action until a final answer is found. (Ch3, Ch11)

**Agentic workflow** — automation built from one or more agents that plan, execute, and memorize steps to complete a business process autonomously. (Ch3, Ch4, Ch11, Ch12)

**Anti-pattern** — a common response to a recurring problem that is ineffective/counterproductive and causes more issues than it solves. (Ch6)

**BERTScore** — evaluation metric using cosine similarity of BERT token embeddings between generated and reference text; reports precision/recall/F1. (Ch6, Ch13)

**BLEU score** — Bilingual Evaluation Understudy; measures n-gram overlap between generated and reference text, originally for translation. (Ch13)

**Catastrophic forgetting** — when a model fine-tuned on a new task forgets how to perform a previously learned task; main risk of full fine-tuning. (Ch3)

**Chain of Thought (CoT)** — prompting technique instructing the model to reason step-by-step before answering. (Ch3, Ch11)

**Chunking** — splitting documents into smaller pieces (fixed-length or semantic/context-aware) for embedding and retrieval in RAG. (Ch3, Ch8, Ch14)

**Compound AI system** — modern AI applications built as chains of multiple LLM calls, RAG systems, and agents rather than one monolithic model. (Ch6)

**Concept drift** — change in the underlying concept a model predicts over time (e.g., evolving fraud patterns). (Ch13)

**Context window** — total input + output tokens a model can process in a single request. (Ch2)

**Data drift** — change in the shape/representation of inference data over time. (Ch13)

**Data governance** — framework for managing data availability, usability, integrity, and security, including policies/standards/procedures. (Ch6)

**DPO (Direct Preference Optimization)** — fine-tuning technique that removes the need for a separate reward model; the fine-tuned model acts as its own reward model, optimizing directly on human-rated response pairs. (Ch3, Ch6, Ch12)

**DoRA** — fine-tuning method decomposing pretrained weights into general-purpose (low-rank, frozen) and task-specific components, improving on LoRA. (Ch3)

**ESG (Environmental, Social, Governance)** — framework measuring an organization's environmental sustainability, diversity/equity, employee experience, privacy/governance, and societal impact. (Ch11)

**EULA (End User License Agreement)** — model provider's usage terms; critical compliance checkpoint for financial institutions before adopting any model. (Ch2, Ch12, Ch14)

**FMOps (Foundation Model Operations)** — technical, business, and cultural practices for productionizing foundation-model-based generative AI, extending MLOps. (Ch1, Ch14)

**Foundation model (FM)** — a model pre-trained on vast curated data that can perform many tasks with minimal task-specific customization. (Ch1, Ch2)

**Full fine-tuning** — updating all weights of a pretrained model; slow, expensive, risks catastrophic forgetting. (Ch3)

**Function calling** — supplying an LLM a single function's signature/inputs/return type so it builds the call arguments directly, vs. an agent's broader autonomous tool orchestration. (Ch11)

**Guardrails** — measures filtering model input/output against organizational policy (bias, toxicity, PII, brand language), implemented via prompt engineering, external tools, or HITL. (Ch7, Ch13, Ch14)

**Hallucination** — a model generating realistic-sounding but factually incorrect information. (Ch5, Ch12, Ch13)

**HITL (Human-in-the-Loop)** — routing low-confidence guardrail/model decisions to human reviewers; not viable for real-time latency-sensitive use-cases. (Ch13, Ch14)

**Indemnity** — legal protection a model provider offers against toxic/leaked model outputs; varies by provider and is excluded from the formal Ch12 scorecard due to measurement difficulty. (Ch12, Ch14)

**Knowledge distillation** — training a smaller model to mimic a larger model's behavior, reducing hosting cost. (Ch7, Ch14)

**LoRA (Low-Rank Adaptation)** — PEFT method injecting small trainable rank-decomposition matrices into transformer layers instead of updating full weight matrices. (Ch3, Ch10)

**METEOR score** — translation/summarization evaluation metric accounting for synonym/paraphrase matches, not just exact word overlap (improves on BLEU). (Ch13)

**MLOps** — technical, business, and cultural practices promoting automation, governance, and scalability for predictive ML products. (Ch1, Ch14)

**Mixture of Experts (MoE)** — architecture replacing transformer FFN layers with a router + sparse group of expert FFNs for efficient, specialized inference. (Ch2)

**Model governance** — organizational process ensuring risks are mitigated and models are used fairly/accountably, spanning the full ML/FM lifecycle. (Ch13)

**ONNX (Open Neural Network Exchange)** — interoperable model format across different ML frameworks. (Ch14)

**Operating model** — an organization's strategy for delivering value via shared, decentralized, or hybrid team/platform structures. (Ch14)

**PEFT (Parameter-Efficient Fine-Tuning)** — fine-tuning approach freezing most base model weights and training only a small added parameter set (LoRA, QLoRA, DoRA). (Ch3, Ch10)

**PII (Personally Identifiable Information)** — data that can identify an individual; in financial services includes name, address, SSN, account numbers, transaction histories, and digital identifiers. (Ch5, Ch13)

**Principle of Least Privilege (PoLP)** — granting actors/resources access to only what they strictly need. (Ch7, Ch14)

**Prompt cache** — stores frequently used prompts and their answers to avoid redundant model calls. (Ch7)

**Prompt engineering** — crafting natural-language instructions to get desired model output without changing model weights. (Ch1, Ch3)

**QLoRA** — PEFT method combining LoRA with 4-bit model quantization to further reduce fine-tuning memory requirements. (Ch3, Ch10)

**RAG (Retrieval Augmented Generation)** — technique augmenting a base model's response with context retrieved from an organization's own data sources, without changing model weights. (Ch3, Ch8)

**ReAct (Reasoning and Acting)** — prompting/agent technique interleaving reasoning traces with tool-use actions. (Ch3, Ch11)

**Red-teaming** — adversarial testing simulating attacks (prompting, data leakage, backdoor, RCE, role-play, data poisoning, exfiltration) to uncover model vulnerabilities. (Ch13)

**Responsible AI** — approach to building/managing/evaluating/deploying AI prioritizing fairness, ethics, trustworthiness, safety, privacy, and accountability. (Ch13)

**Reward model** — a model trained on human-rated response pairs to estimate how a human would rate a given response, central to RLHF. (Ch3)

**RLHF (Reinforcement Learning with Human Feedback)** — fine-tuning approach using a trained reward model (often with PPO) to refine model outputs toward human preferences. (Ch3, Ch6, Ch12)

**ROUGE score** — Recall-Oriented Understudy for Gisting Evaluation; measures n-gram/sequence overlap for summarization quality. (Ch6, Ch13)

**SEC 10-K filing** — annual regulatory disclosure by public companies; used as a rich fine-tuning corpus for financial-domain adaptation. (Ch10)

**Self-consistency** — prompting technique exploring multiple reasoning paths and selecting the most consistent answer, improving on CoT's single greedy path. (Ch3)

**Single/Multi-tenancy** — architectural choice of whether each LoB independently manages its own GenAI stack (single) or accesses shared, centrally governed FM infrastructure (multi). (Ch7)

**TCO (Total Cost of Ownership)** — the full cost of a GenAI solution across compute, storage/database, model, human, and tooling categories, not just model-invocation cost. (Ch6, Ch7, Ch12, Ch14)

**Tenancy** — architectural concept defining how a GenAI application is exposed to its consumers (single vs. multi-tenant). (Ch7)

**Tokenization** — converting raw input data into the smallest units (tokens) a model can process. (Ch2)

**Toxicity score** — probability that model output is toxic (hate, profanity, violence, sexual, insult, racism). (Ch13)

**Tree of Thought (ToT)** — advanced prompting technique generating and evaluating multiple reasoning sub-trees/branches before consolidating an answer. (Ch3)

**Underwriting** — process of minimizing/covering financial risk for an individual or business against a financial instrument (securities, bank/credit, or insurance underwriting). (Ch4)

**Vector database/store** — specialized database storing and retrieving high-dimensional embeddings for similarity search (e.g., ChromaDB, Pinecone, OpenSearch, Weaviate, Faiss). (Ch2, Ch3, Ch8)

**Zero-shot / Few-shot learning** — prompting without examples (zero-shot) or with a small number of examples (few-shot) to guide model behavior. (Ch3)
