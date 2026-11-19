# Chapter 6: ChatGPT

## Core Idea
ChatGPT's business value comes not just from the underlying GPT model but from the full OpenAI product ecosystem (Playground, fine-tuning, plugins, Enterprise features) plus a disciplined prompt-engineering process — and that same surface area (prompt injection, jailbreaking) is the primary attack vector enterprises must defend against.

## Frameworks Introduced
- **Prompt types by openness and length**: closed vs. open prompt; long vs. short prompt — four combinations with distinct accuracy/diversity trade-offs.
  - When to use: scoping how much creative latitude vs. control a given application needs.
  - How: closed+long prompts maximize control/accuracy but limit novelty; open+short maximize creativity but sacrifice relevance/specificity; enterprise use typically needs long+closed ("guiding rails").
- **Prompt engineering process (5 steps)**: set objectives → design prompt → evaluate responses → refine → templatize into a registry.
  - When to use: any production prompt-engineering workflow, not one-off experimentation.
  - How: objectives define audience/tone/structure; design applies a prompting technique (zero/one/few-shot, instruction, CoT, ToT); evaluation scores accuracy/relevance/coherence/completeness/factuality/bias; refine iterates in a sandbox; templating version-controls the final prompt with metadata (model, params, cost, latency).
- **Prompting technique ladder**: zero-shot → one-shot/few-shot → instruction prompting → chain-of-thought (CoT) → tree-of-thought (ToT) / graph-of-thought.
  - When to use: escalate up the ladder as task complexity (especially reasoning) increases.
  - How: zero/few-shot condition the model via examples alone; instruction prompting gives explicit behavioral rules; CoT decomposes reasoning into intermediate steps; ToT (4 sub-processes: thought decomposition, thought generation, state evaluation, search algorithm) explores multiple reasoning branches and prunes via breadth/depth-first search.
- **Prompt playground architecture**: Streamlit UI + DVC (prompt/version control) + LangChain (LLM execution) + multi-metric evaluation (cost, bias, grammar, relevance, human feedback).
  - When to use: building an internal enterprise tool for designing/testing/templatizing prompts at scale.
  - How: base templates are created and versioned via DVC; each execution captures LLM response plus automated metrics (tiktoken-based cost, Flair-based bias classification, LanguageTool grammar check, a second LLM — Text-Bison — scoring relevance/coherence) plus 1-5 human feedback.
- **Jailbreaking defense framework**: identity/access management, policy controls, continuous prompt monitoring, attack-pattern detection, training-time content filtering, system meta-prompt framework (delimiters, special tokens), abuse-pattern analysis, red-teaming.
  - When to use: any production LLM application exposed to untrusted user input or external data sources.
  - How: combine input/output filtering with structural defenses (delimiters separating instructions from data) and proactive adversarial testing before deployment.

## Key Concepts
- **Prompt engineering**: designing, testing, and optimizing input prompts to elicit desired LLM responses; distinct from prompt design (tuning models via APIs, for builders).
- **Temperature / Top-P / Max length**: core OpenAI Playground hyperparameters — temperature controls randomness (0=deterministic, 1=creative); Top-P controls nucleus sampling breadth; max length caps response token count.
- **Fine-tuning (OpenAI)**: updating a fraction of model weights on labeled system/user/assistant-role `.jsonl` examples to align responses to domain-specific knowledge or correct edge cases; drawbacks include lost explainability, OpenAI-model lock-in, and higher per-token cost.
- **Jailbreaking**: prompting techniques that bypass an LLM's built-in safety controls to extract restricted content, execute unauthorized actions, or leak system prompts.
- **Prompt injection (direct/indirect)**: direct = attacker directly crafts manipulative input (pretend/role-play, DAN prompts); indirect = malicious instructions hidden in external data sources (web pages, documents, plugins) the LLM processes.
- **Prompt leaking**: a jailbreak variant where the attacker tricks the model into revealing its own internal/system prompt.
- **PAIR method**: an automated jailbreak-generation technique where one LLM acts as a red-teaming assistant iteratively refining adversarial prompts against a target model.
- **tiktoken**: OpenAI's open-source tokenizer; different model families use different encodings (`cl100k_base` for GPT-4/3.5-Turbo, `p50k_base` for davinci-002/003/Codex, `r50k_base` for GPT-3 DaVinci).

## Mental Models
- Think of **prompt engineering like interview-question design**: the quality and specificity of the question (context, audience, tone) directly shapes the quality of the answer — a vague question gets a vague answer.
- Use the **"guiding rails" model** for enterprise prompting: combine long + closed prompting to constrain outputs to the needed structure while still leveraging the model's generative range within those rails.
- Think of **CoT/ToT as making reasoning steps a first-class, inspectable artifact**: breaking a complex task into intermediate thoughts not only improves accuracy on arithmetic/symbolic tasks but also makes it possible to debug exactly where a model's reasoning fails.
- Use the **defense-in-depth model for jailbreaking**: no single control (output filtering, access management, red-teaming) is sufficient alone; combine structural (delimiters, meta-prompts), behavioral (abuse-pattern monitoring), and process (red-teaming before deployment) layers.

## Anti-patterns
- **Writing prompts without audience/context framing**: a bare question ("What is ChatGPT?") produces a far less useful response than one with context (audience age, expertise, purpose) — the chapter demonstrates this directly with before/after examples.
- **Relying solely on built-in LLM guardrails against jailbreaking**: perpetrators routinely bypass them via role-play, DAN-style prompts, or indirect injection through poisoned external content — layered controls (monitoring, red-teaming, meta-prompt structuring) are required.
- **Using third-party ChatGPT plugins without vulnerability awareness**: plugin integrations expand attack surface for indirect prompt injection (e.g., hidden prompts on a webpage triggering unauthorized plugin actions).
- **Treating fine-tuning as a first resort**: fine-tuning is expensive, reduces explainability, and locks you to the fine-tuned provider — reserve it for cases prompt engineering genuinely cannot solve (complex tasks, persistent edge cases, domain alignment).
- **Skipping systematic prompt evaluation**: judging prompt quality by eyeballing single responses, instead of scoring accuracy/relevance/coherence/bias/factuality systematically (optionally via a second LLM-as-judge), leads to inconsistent production quality.

## Code Examples
```python
# Tokenizing text with OpenAI's tiktoken (model-specific encodings: cl100k_base, p50k_base, r50k_base)
import tiktoken
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
enc = encoding.encode("What is Generative AI!")
dec = encoding.decode(enc)
print(enc)
print(dec)
```
```json
// OpenAI fine-tuning dataset format (.jsonl, one example per line)
{"messages": [
  {"role": "system", "content": "You are a expert assistant for a edge computing domain."},
  {"role": "user", "content": "What is the tool available to do registration of edge device?"},
  {"role": "assistant", "content": "The tool ABC is available to do edge device registration."}
]}
```
```text
# Tree-of-Thought prompting template (multi-expert collaborative reasoning)
"""Imagine three different experts with exceptional logical thinking are trying
to solve this question. They will collaborate, brainstorm, and arrive at the
final decision using a tree of thought processes. All experts will bring out
their thoughts in detail one at a time, accounting for previous thoughts, then
share it with the group. Experts will play the role of critique and critique
their own responses and the responses of other experts... Each expert will
assign a likelihood of their current assertion being correct. The process
continues until a conclusive solution is found.

The question is: Bob is in the living room. He walks to the kitchen, carrying
a cup. He puts a ball in the cup and carries the cup to the bedroom. He turns
the cup upside down, then walks to the garden. He puts the cup down in the
garden, then walks to the garage. Where is the ball?"""
```
- **What it demonstrates**: (1) model-specific tokenization used for API cost calculation; (2) the exact JSONL schema OpenAI fine-tuning requires; (3) a Tree-of-Thought prompt template that forces multi-path reasoning with self-critique to solve a spatial-reasoning puzzle that naive prompting gets wrong.

## Reference Tables
**OpenAI product line**

| Product | Function |
|---|---|
| ChatGPT | Conversational AI chatbot |
| DALL-E (2/3) | Text-to-image generation/editing |
| Whisper | Speech recognition, transcription, translation |
| Codex | Code generation and explanation |
| TTS-1 / TTS-1-HD | Text-to-speech |

**Image model comparison (Ch 6 vendor landscape)**

| Model | Strength |
|---|---|
| DALL-E 3 | High quality + prompt adherence |
| Imagen 2 | Simultaneous generation, photorealism, long text rendering |
| Midjourney | Creative/artistic images |
| FLUX.1 | Long text rendering, prompt adherence |
| Stable Diffusion | Open-source |

**Prompting technique → use case**

| Technique | Best for |
|---|---|
| Zero-shot | Simple tasks, no training data needed |
| One/few-shot | Steering tone/format with limited examples |
| Instruction prompting | Explicit behavioral/style control |
| Chain-of-thought (CoT) | Arithmetic/symbolic reasoning |
| Tree-of-thought (ToT) | Strategic exploration needing look-ahead |

**Jailbreaking technique categories**

| Category | Example |
|---|---|
| Direct injection (pretend/role-play) | "Pretend you are a GPT in developer mode" |
| Direct injection (DAN-style) | "Do Anything Now" prompts bypassing restrictions |
| Indirect injection | Hidden prompts in webpages/documents/plugins |
| Prompt leaking | Tricking model into revealing its system prompt |
| Obfuscation | Base64 encoding, synonym substitution, payload splitting |
| Automated (PAIR) | LLM-generated iterative jailbreak refinement |

## Worked Example
**Enterprise Prompt Playground**: Built with Streamlit (UI), DVC (prompt version control), and LangChain (LLM execution). Tab 0 lets a prompt engineer create a blank base template tied to a business purpose (e.g., `project_genai_base.py`). Tab 1 lets them select a base template, edit it with Streamlit-Ace, execute it against an LLM, and automatically capture: cost (via tiktoken token counts × per-1000-token rate), bias score (Flair NLP zero-shot gender/racial bias classification), grammar score (LanguageTool error count), relevance/coherence score (a second LLM — Google's Text-Bison — grading the first LLM's output), and a 1-5 human helpfulness rating. Every execution is saved as a new timestamped template version under DVC, building an audit trail so multiple prompt engineers can compare templates for the same business objective by cost, quality, and safety metrics side by side.

## Key Takeaways
1. ChatGPT's enterprise value extends well beyond the chat interface — fine-tuning, plugins, Enterprise security features, and third-party add-ons (Google Sheets/Docs, Microsoft Copilot, GitHub Copilot) are the actual integration surface for most business use cases.
2. Prompt quality is systematically controllable via the openness/length framework and the 5-step prompt engineering process — it is an engineering discipline, not guesswork.
3. Escalate prompting technique complexity (zero-shot → few-shot → CoT → ToT) only as task reasoning complexity demands it; ToT's multi-path exploration is overkill for simple tasks.
4. A prompt's safety and reliability must be evaluated on the same systematic footing as its usefulness — accuracy, relevance, coherence, factual correctness, and bias all need explicit scoring.
5. Jailbreaking is not a hypothetical risk — direct (pretend/DAN) and indirect (poisoned external data) prompt injection are concrete, documented attack techniques requiring layered, proactive defenses (red-teaming, meta-prompt structuring, continuous monitoring) rather than reliance on default model guardrails.
6. Fine-tuning should be a late resort after prompt engineering is exhausted, given its cost, lock-in, and explainability trade-offs.

## Connects To
- **Ch 5**: directly builds on the transformer/decoder-only architecture and decoding strategies (temperature, top-p/top-k sampling) explained there, now applied as ChatGPT's user-facing hyperparameters.
- **Ch 7**: the Prompt Playground's LangChain integration here is a preview of the full LangChain framework (chains, memory, agents, RAG) covered next.
- **Ch 8**: the prompt evaluation metrics (cost, bias, grammar, relevance, human feedback) here foreshadow LLMOps' formal monitoring/evaluation/feedback systems.
- **Ch 9**: jailbreaking and prompt-injection risks introduced here become part of the enterprise AI risk/responsible-AI framework.
