# Cheatsheet

## Tool Selection: Which Generative AI Stack Fits Which Need?

| Need | Reach for | Why |
|---|---|---|
| General-purpose conversational app, fast iteration | ChatGPT API (GPT-3.5/4) | Mature ecosystem, moderation API, function calling, fine-tuning available |
| Document-grounded Q&A, large context, Google ecosystem | Gemini Pro + LangChain RAG | Multimodal, MoE architecture, up to 1M-token preview context |
| Full data control, no vendor dependency, offline capability | Llama 2 (local, quantized) | Free, self-hostable, fine-tunable without vendor fees |
| Code generation/pair-programming without local hosting | Code Llama via Together.AI (serverless) | No weight management; $25 free credit; specialized for code |
| Quick ML-model demo/proof-of-concept UI | Gradio | Minutes to a shareable interface; Hugging Face Spaces hosting |
| Data dashboard / CSV chat interface | Streamlit + LangChain agent | No HTML/CSS; natural fit for tabular data + charts |
| No-code chatbot prototype, non-technical stakeholders | Flowise AI | Drag-and-drop, zero code, Marketplace templates |
| Large context window for whole-document chat | Claude (Anthropic) | ~100K-200K token context (~75K-150K words) |
| Need an "ethics-first" model framing | Claude (Anthropic) | Built with an explicit ethical constitution |

## Decision Rule: Open-Source vs. Proprietary LLM

| Factor favors... | Proprietary (GPT-4, Claude, Gemini) | Open-source (Llama 2, Mistral, Falcon) |
|---|---|---|
| Time-to-market | Yes — API-ready, minimal infra | No — requires setup/infra investment |
| Data privacy/control | No — third-party data exposure risk | Yes — full control, on-prem/cloud of choice |
| Long-term cost predictability | No — usage-based, can scale expensively | Yes — no licensing fees, predictable |
| Fine-tuning cost | No — can be costly | Yes — often free/low-cost |
| Baseline quality (out-of-box) | Yes — generally higher | Improving rapidly, historically lower |
| Vendor lock-in risk | High | None |
| Staff specialization needed | Minimal | Requires specialized staff |

**Rule of thumb**: time-constrained launch → lean proprietary. Data-sensitive, cost-predictable, long-horizon, or customization-heavy → lean open-source. Prefer Apache 2.0, then MIT license for commercial open-source use.

## Prompt Engineering Technique Selection

| Symptom / Need | Technique | Chapter |
|---|---|---|
| Output ignores part of instructions / injected text hijacks behavior | Delimiters (triple backticks, XML tags) | Ch3, Ch6 |
| Need machine-parseable output | Request structured output (JSON/HTML) | Ch3 |
| Model should only act if a precondition holds | Condition-checking instruction | Ch3 |
| Limited examples of desired style/format | Few-shot prompting | Ch3 |
| Multi-step reasoning task, model jumps to wrong conclusion | Chain-of-Thought / break into smaller steps | Ch3 |
| Model might rubber-stamp a wrong input answer | "Work out your own solution first" | Ch3 |
| Need consistent tone/role | Persona (system message) | Ch3, Ch6 |
| Many input variables / reused prompt logic | Prompt template (LangChain `PromptTemplate`) | Ch3, Ch7, Ch10 |
| Model confidently invents facts | Explicit "say I don't know" instruction + RAG grounding | Ch7, Ch13 |
| Prompting alone insufficient for domain task | Fine-tuning | Ch4, Ch5, Ch6 |
| Need up-to-date/proprietary knowledge without retraining | RAG | Ch5, Ch7 |

## Model Escalation Ladder (cheapest → most expensive fix)
1. **Prompt engineering** — clear/specific instructions, delimiters, few-shot, CoT.
2. **RAG** — add external retrieval if the issue is missing domain knowledge.
3. **Fine-tuning** — if prompting/RAG can't fix behavior/format/tone consistently (needs ~50-100+ examples).
4. **Pretrain a new foundation model** — last resort; can cost millions of dollars. Avoid unless truly justified.

## Token Economics Quick Reference
- 1 token ≈ 4 characters (English) ≈ ¾ of a word.
- 100 tokens ≈ 75 words; 1,500 words ≈ 2,048 tokens.
- Input + output tokens both count toward cost and the model's max context.
- Trailing whitespace in a prompt is tokenized — keep prompts trimmed.
- Capitalization and sentence position change a word's token ID — token count ≠ simple word count.

## Ch12 Best-Practices Checklist (building an application)
- [ ] Chose the right tool/framework for the specific task and goals (not hype-driven).
- [ ] Answered the 4 framing questions: objective, task type, scale, resource constraints.
- [ ] Evaluated model size/capability, knowledge cutoff, fine-tuning support, cost, and ethics.
- [ ] Started small/simple before scaling up.
- [ ] Tested for accuracy, bias, and security vulnerabilities before production.
- [ ] Applied LLMOps workflow: select → prompt-engineer → evaluate → fine-tune → deploy (canary/blue-green) → monitor.
- [ ] Set up monitoring (e.g., LangSmith) for latency, error rate, accuracy, cost.
- [ ] Anonymized training data; enforced access controls; watched for bias; defended against prompt injection; was transparent about model function/data use.
- [ ] Chose Apache 2.0 or MIT licensed open-source models if going that route.

## Ch13 Ethical-Considerations Checklist (deploying responsibly)
- [ ] Embedded ethics at every lifecycle stage (not just start/end): defined the problem, assessed impact, evaluated bias — *before* writing code.
- [ ] Used diverse, representative training data; applied fairness metrics; considered synthetic data for underrepresented groups.
- [ ] Anonymized sensitive attributes with care for correlated proxy features.
- [ ] Built hallucination mitigations across all 3 layers: data/prompt, model/architecture, application design.
- [ ] Established user trust practices: ongoing factual-accuracy evaluation, staying current on reliability research, domain-expert collaboration.
- [ ] Assessed copyright exposure: transformative use vs. derivative work; preferred public-domain/open-access training data; drafted precise licensing agreements.
- [ ] Added deepfake/synthetic-media safeguards: verification mechanisms, detection algorithms, user education, watermarking.
- [ ] Gave users opt-in/opt-out control over their data being used in training.
- [ ] Considered environmental impact: energy-efficient hardware, renewable energy, lower-compute algorithms.
- [ ] Involved diverse stakeholders (ethicists, social scientists, users) in the development process.

## Safety Layer Quick Reference (any user-facing LLM app)
| Risk | Mitigation |
|---|---|
| Harmful user input/output content | OpenAI Moderation API (free, not 100% reliable) |
| Prompt injection | Delimiters + clear system/user message separation |
| Hallucination on domain questions | RAG + explicit "say I don't know" prompt template |
| Hallucination on vague code/task requests | More detailed, context-rich prompts |
| Side-effecting function calls | User confirmation flow before executing model-generated function calls |
| Biased output | Diverse training data, fairness metrics, regular audits |
| Unauthorized data exposure | Anonymization, access controls, encrypted deployment, opt-in/opt-out |
