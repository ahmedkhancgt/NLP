# Chapter 9: Historical flow and development of GPT series

## Core Idea
Each GPT generation (GPT-1 → GPT-2 → GPT-3 → InstructGPT/GPT-3.5) improved primarily by scaling parameters and training data while making targeted architectural tweaks — and the jump from "fine-tune per task" (GPT-1) to "zero/few-shot task transfer" (GPT-2/3) to "human-feedback alignment" (InstructGPT) is the real story, not just raw size.

## Frameworks Introduced
- **GPT-1's 3-stage training framework**: (1) pre-train on unlabeled corpus via likelihood optimization, (2) fine-tune on labeled discriminative tasks, (3) task-specific input transformation (structured token sequences with start/end/delimiter tokens).
  - When to use: understanding the original "pre-train then fine-tune per task" paradigm before zero-shot became viable.
- **Task conditioning / zero-shot task transfer (GPT-2)**: give the model the task description and input together in natural language, with no separate fine-tuning step, so it infers the task from phrasing alone (e.g., show an English sentence + "France" to signal translation).
  - When to use: designing prompts for a model never explicitly fine-tuned for your task.
  - How: supply instructions/examples in natural language within the context window rather than retraining; zero-shot = no examples, one-shot = one example, few-shot = several examples, and the book notes performance rises with example count and model scale.
- **InstructGPT's 3-model training pipeline**: Supervised Fine-Tuning (SFT) → Reward Modeling (RM) → Reinforcement Learning from Human Feedback (RLHF/PPO).
  - When to use: the template for aligning a base LLM to human preferences rather than raw next-token prediction.
  - How: SFT trains on labeler demonstrations (16 epochs, cosine decay); RM trains a ~6B-parameter model to score responses based on human preference rankings; PPO then fine-tunes the policy against the reward model with a per-token KL penalty to avoid over-optimizing away from the SFT baseline.

## Key Concepts
- **GPT-1** (June 2018): 117M parameters, 12 decoder layers, BookCorpus training (7,000+ books); outperformed supervised SOTA on 9/12 tasks.
- **GPT-2** (Feb 2019): 1.5B parameters, trained on WebText (40GB, Reddit-sourced); introduced task conditioning and zero-shot transfer; vocabulary expanded 256→130,000 via Unicode encoding.
- **GPT-3** (May 2020): 175B parameters, 96 layers, context window 2048 tokens, trained on 5 corpora (Common Crawl, WebText2, Books1/2, Wikipedia); evaluated via zero/one/few-shot settings rather than fine-tuning.
- **InstructGPT / GPT-3.5**: fine-tuned via RLHF; preferred over raw GPT-3 by human labelers ~85% of the time.
- **Fill-In-the-Middle (FIM) training**: transforms training data by moving a random text span from the middle to the end of a document, teaching the model to infill text (e.g., docstrings, import statements) while retaining left-to-right generation ability.
- **Whisper**: OpenAI's automatic speech recognition model, trained on 680,000 hours of multilingual/multitask audio; uses mel-spectrogram chunks (30 sec) through an encoder-decoder transformer; ~50% fewer errors than comparable zero-shot models.
- **GPT-3.5 model family (Turbo, Davinci, Curie, Babbage, Ada)**: tiered by capability/cost/speed — Turbo powers ChatGPT; Davinci is most capable/most expensive; Ada is fastest/cheapest for simple tasks.

## Mental Models
- Use the **capability-vs-scale growth curve**: 117M (GPT-1) → 1.5B (GPT-2) → 175B (GPT-3) parameters tracks directly with improved zero-shot performance, but each jump also required a new training-data strategy (BookCorpus → WebText → 5-corpus blend), not scale alone.
- Think of **alignment (InstructGPT) as a separate axis from capability (GPT-3)**: a bigger model isn't automatically a better assistant — RLHF is what made GPT-3.5 preferred over raw GPT-3 even without changing the base capability.

## Worked Example
GPT-2's task-conditioning mechanism: given the English sentence, then the word "France", then a translation prompt, the model infers (without explicit instruction) that the task is English→French translation and produces the French equivalent — demonstrating zero-shot task transfer purely from prompt structure.

## Key Takeaways
1. Each GPT generation's leap in capability came from combined scale increases AND methodology shifts (fine-tuning → zero-shot conditioning → RLHF alignment), not scale alone.
2. Zero/one/few-shot prompting is the practical skill for extracting performance from GPT-2/3-class models without fine-tuning access.
3. InstructGPT's 3-model RLHF pipeline (SFT → RM → PPO) is the direct technical ancestor of ChatGPT's conversational alignment.
4. The GPT-3.5 "Turbo/Davinci/Curie/Babbage/Ada" tiering is a cost/capability trade-off map still relevant for choosing a model for a given task's complexity.
5. FIM training shows the lineage didn't stop at better conversation — code-completion ability (relevant to Ch 16, 28) was trained in via a dedicated data transformation technique.

## Connects To
- **Ch 1, Ch 7**: this chapter supplies the full technical depth behind the GPT/transformer basics introduced there.
- **Ch 10**: continues directly into API pricing for the models detailed here (Turbo, Davinci, Curie, Babbage, Ada).
- **Ch 16, 28**: rely on the FIM/code-training capability established here when discussing ChatGPT's coding abilities.
- **Ch 32**: picks up the GPT lineage at GPT-4, the direct successor to the models covered here.
