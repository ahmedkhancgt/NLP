# Glossary

**Accountability** — Ethical dimension requiring clear responsibility for AI system actions/outputs and the capacity for corrective action (Ch 9).

**Actor layer** — Layer 1 of the operating model: human stakeholders and end-user devices participating in the AI journey (Ch 10).

**Agentic assistant** — An individual agent composed of toolset, data, function, and instruction components, backed by a chosen LLM (Ch 16).

**Agentic RAG** — An evolution of RAG where the system autonomously and proactively retrieves/integrates external data in real time (Ch 12).

**AI agent** — A software entity that performs tasks autonomously by perceiving its environment, deciding, and acting toward goals (Ch 16).

**API economy** — Integration across organizational silos via APIs, essential for real-time AI-driven outcomes (Ch 10).

**Applicability layer** — Layer 7 of the operating model: AI-native/augmented workflows integrated into business/technical processes (Ch 10).

**Architecture (GenAI)** — The four-layer technical framework of algorithms, data management, computational infrastructure, and integration (Ch 1).

**Chunking** — Splitting long documents into length-limited passages so relevant content fits within the LLM's context window (Ch 12).

**CapEx vs. OpEx** — Capital expenditure (upfront investment, e.g., on-premises hardware) vs. operational expenditure (pay-as-you-go, e.g., cloud/MaaS) deployment cost models (Ch 3, Ch 13).

**Co-location** — Physical/logical proximity of LLM, data, and AI applications to minimize latency (Ch 5, Ch 15).

**Confidential AI** — Application of advanced encryption and secure computing environments to protect sensitive data during AI processing phases (Ch 14).

**Consumption layer** — Layer 3 of the operating model: where business requirements and AI solutions are co-created/co-deployed, from PoC to enterprise scale (Ch 10).

**Correlated dimensions** — The six interconnected ethical dimensions: accountability, fairness, transparency, human-orientation, security, lawfulness (Ch 9).

**Data drift / task drift / topic drift / retrieval drift** — Four distinct ways a deployed model's performance or relevance can degrade: input data statistics change, requested task types change, discussed subjects change, or outputs diverge from expected benchmarks (Ch 2).

**Data residency and proximity** — Jurisdictional data storage compliance (residency) vs. minimizing physical/network distance for latency (proximity) (Ch 5).

**Decision quadrants (MaaS)** — Graphical framework for choosing between pay-as-you-go and real-time endpoint MaaS options by environment and token volume (Ch 13).

**Deployment patterns** — The five architectural options for hosting an LLM: cloud service, open on-premises, own on-premises, open cloud IaaS, own cloud IaaS (Ch 4).

**Deployment strategies** — Release-management techniques (Blue/Green, Canary, A/B Testing) for shipping LLM-powered application changes (Ch 4).

**Demarcation** — The boundary and associated responsibilities between cloud and on-premises environments in the GenAI operating system (Ch 5).

**Ethical AI** — The practical application of the six correlated ethical dimensions to a given AI system (Ch 9).

**Execution layer** — Layer 5 of the operating model: practical application domains (app dev, ITOps, digital transformation, industry use cases) (Ch 10).

**Fairness** — Ethical dimension requiring AI systems to prevent bias/discrimination across groups (Ch 9).

**Feedback loop** — Mechanism for continuous improvement, collecting performance/outcome data across operating-model layers to refine algorithms, data, and UX (Ch 10).

**Fine-tuning** — Adjusting a pre-trained model on a smaller, task-specific dataset to improve accuracy/relevance for a particular application (Ch 2, Ch 7).

**Funneling of models** — Taking multiple candidate LLMs through selection criteria to narrow down to one or more final fits (Ch 3).

**GPU TEE** — Graphical processing unit trusted execution environment; secure execution space within a GPU protecting data during AI computation (Ch 14).

**Hallucination** — Incorrect or fabricated model response; mitigated by combining RAG, prompt tuning, and fine-tuning (Ch 12).

**Hardware latency** — Delay from CPU/GPU processing speed, memory loading, storage access, and system bus/cooling/power factors (Ch 15).

**Holistic operating model** — Organization-wide operational framework comprising seven layers plus feedback loop, governance, and cultural shift (Ch 10).

**Human-orientation** — Ethical dimension requiring AI to augment, not replace, human capabilities under human oversight (Ch 9).

**Hybrid retrieval** — Combination of vector search and keyword search for RAG, fused to capture complementary strengths (Ch 12).

**Indemnity** — Legal protection question of whether an organization is protected for a model's output completions (Ch 3).

**Inference task** — The specific operation a model performs (text generation, summarization, translation, classification, Q&A, etc.) (Ch 1, Ch 3).

**Inside periphery** — Cost optimization opportunities intrinsic to training the LLM itself: parameters, data quality, architecture, hardware, skills, algorithms (Ch 11).

**Input network latency** — Delay from the user's device to the server hosting the AI model (Ch 15).

**Integration point latency** — Delay from AI system interactions with APIs, databases, and other backend services (Ch 15).

**Landing zone** — Pre-configured, secure, scalable, compliant architecture (network, identity, logging, monitoring, security) for deploying AI services (Ch 2, Ch 5, Ch 8, Ch 10).

**Lawfulness** — Ethical dimension requiring AI systems to comply with data protection, IP, and geography-specific laws (Ch 9).

**Latency factors** — The five independently-addressable latency sources: input network, hardware, model inference, integration point, output network (Ch 15).

**LLM vs. SLM** — Large language model (general-purpose, broad) vs. small language model (trained for a specific task) (Ch 3).

**LLMOps (Large Language Model Operations)** — Operational management practices tailored to LLMs, distinct from generic MLOps, covering data prep, prompt engineering, fine-tuning, governance, inference/serving, monitoring (Ch 2).

**LoRA (Low-Rank Adaptation)** — Parameter-efficient fine-tuning technique using trainable low-rank matrices to modify weights instead of full retraining (Ch 2, Ch 7).

**Model as a Service (MaaS)** — Deployment of language models through a cloud platform's AI studio, consumed via API without managing underlying infrastructure (Ch 13).

**Model evaluation** — Assessing a model's accuracy measurement in line with desired outcomes, using metrics like fluency, coherence, and accuracy (Ch 2).

**Model governance** — Policies/practices ensuring ethical, transparent, accountable AI use, spanning model, technical-architecture, and organization-wide layers (Ch 2, Ch 9).

**Model inference latency** — Time taken for the AI model itself to process input and generate output (Ch 15).

**Model observability** — Broad insight into model behavior through tracking of data, task, topic, and retrieval drift (Ch 2).

**Model optimization** — Techniques (pruning, quantization, low-rank approximation) to improve a model's performance/cost efficiency (Ch 2, Ch 11, Ch 15).

**Model orchestration** — Automated workflows integrating a model with the broader AI ecosystem (Ch 2, Ch 8).

**Model selection** — Identifying the right model that fits enterprise purpose (Ch 2, Ch 3).

**Model sourcing** — Whether a model is open-source or closed-source proprietary, managed or unmanaged (Ch 2).

**Multi MaaS** — Leveraging Model as a Service across multiple cloud platforms/landing zones simultaneously (Ch 13).

**Multi-agentic** — Multiple autonomous AI agents interacting collaboratively or competitively within a shared environment toward goals (Ch 16).

**Multi-modal** — AI capability to process and integrate information from diverse sources: text, speech, images, video (Ch 16).

**Operating model** — The organizational, seven-layer framework aligning technology, people, process, and culture (distinct from operating system) (Ch 10).

**Operating system (GenAI)** — The technical ecosystem managing AI model/application lifecycle, data, and landing zone (distinct from operating model) (Ch 5).

**Orchestration platform** — System coordinating data, model-related tasks, ecosystem integrations, and continuous enhancements across AI systems (Ch 8).

**Outside periphery** — Cost optimization opportunities external to the LLM itself: model selection, prompt/fine-tuning, token minimization, compute choices (Ch 11).

**Output network latency** — Delay from AI output generation to the output reaching the end-user (Ch 15).

**Overfitting** — The model learns training-data noise rather than generalizable patterns, mitigated via weight decay regularization (Ch 2, Ch 7).

**PAYG (pay-as-you-go)** — MaaS billing based on tokens consumed or API calls made, suited to variable/unpredictable demand (Ch 13).

**Philosophy of generative AI** — The fundamental input-prompt → model → output-completion → business-integration system (Ch 1).

**Platform layer** — Layer 4 of the operating model: the GenAI studio spanning the AI lifecycle (tune, validate, evaluate, test, deploy) (Ch 10).

**Pruning and quantization** — Model-compression techniques removing redundant parameters / reducing numeric precision to cut compute cost and latency (Ch 2, Ch 11, Ch 15).

**Prompt engineering** — Strategic design and implementation of inputs that optimize interactions between AI systems and human users (Ch 6).

**Prompt tuning** — Adjusting the input prompt (not model weights) to elicit more accurate outputs (Ch 11, Ch 12).

**RAG (retrieval-augmented generation)** — Framework combining information retrieval with generative capability so an LLM dynamically accesses and integrates external data (Ch 12).

**Real-time endpoint** — Dedicated, always-on, continuously billed MaaS model access guaranteeing capacity/low latency (Ch 13).

**Requirements layer** — Layer 2 of the operating model: scalability, reliability, maintainability, ethics, governance, IP rights, culture (Ch 10).

**Security (ethical dimension)** — Protecting AI systems from unauthorized access, data theft, tampering, prompt injection (Ch 9).

**Seven layers** — The operating model's layers: actor, requirements, consumption, platform, execution, landing zone, applicability (Ch 10).

**Six ethical dimensions** — Accountability, fairness, transparency, human-orientation, security, lawfulness (Ch 9).

**TEE (Trusted Execution Environment)** — A secure processing area guaranteeing confidentiality and integrity of data loaded inside it (Ch 14).

**Transparency** — Ethical dimension requiring clear communication of how AI systems function and make decisions (Ch 9).

**VM TEE** — Virtual machine trusted execution environment; a secure, isolated area within a VM (Ch 14).

**Weight decay** — Regularization technique penalizing large parameter values to reduce overfitting risk during fine-tuning (Ch 7).
