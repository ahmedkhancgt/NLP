# Chapter 10: API Pricing model and technical limitations of ChatGPT

## Core Idea
ChatGPT's API made integration affordable (fractions of a cent per 1,000 tokens), but the model carries structural limitations — confident-sounding wrong answers, prompt-phrasing sensitivity, self-referential over-optimization, and weak clarification-seeking behavior — that stem from how it was trained, not simple bugs to patch.

## Frameworks Introduced
- **Token-based API pricing model**: cost scales per 1,000 tokens (~750 words), with tiered pricing across model variants (gpt-3.5-turbo cheapest at $0.002/1K, Davinci positioned as "most powerful" at the same headline rate but typically consuming more tokens per task).
  - When to use: budgeting any ChatGPT/GPT-3.5 integration.
  - How: estimate expected token volume (prompt + completion) per transaction, multiply by the per-1K-token rate for the chosen model tier, and account for the fact that more capable models often need fewer retries/corrections, offsetting raw per-token cost.
- **Root-cause framework for ChatGPT's confident-wrong-answer problem**: (1) no ground-truth source of truth during RL training, (2) over-caution trade-off — making the model refuse more also makes it refuse correct-answerable questions, (3) supervised training mismatch — the "best" answer in training reflects the model's own knowledge gaps, not the demonstrator's.
  - When to use: explaining to stakeholders why hallucination isn't a simple fix.

## Key Concepts
- **API pricing tiers**: gpt-3.5-turbo, Ada (fastest/cheapest), Babbage, Curie, Davinci (most powerful) — priced per 1,000 tokens as of March 2023.
- **Prompt sensitivity**: the same underlying question can get a "don't know" vs. a correct answer purely based on phrasing.
- **Over-optimization bias**: training data favoring longer, more "thorough-sounding" responses causes repetitive stock phrases (e.g., restating it's "a language model by OpenAI").
- **Moderation API**: OpenAI's mechanism to flag/block harmful content categories, used to offset false negatives/positives in content moderation.

## Mental Models
- Treat **hallucination as a training-incentive problem, not a knowledge gap**: the model is rewarded during training for sounding confident/thorough, so confidently-wrong answers are an expected side effect, not a rare glitch.
- Use the **"rephrase before you give up" heuristic** when ChatGPT claims ignorance — prompt sensitivity means a reworded question often succeeds where the first phrasing failed.

## Key Takeaways
1. API cost is low enough (fractions of a cent per 1,000 tokens) that integration cost is rarely the blocker — reliability and governance are the real constraints.
2. Confidently-wrong answers are structurally difficult to eliminate because fixing one failure mode (over-caution) directly worsens another (false refusals).
3. Prompt engineering (rephrasing) is a practical, low-cost mitigation for apparent knowledge gaps.
4. The Moderation API is the first line of defense against harmful-content generation, but the book is explicit that false positives/negatives are expected, not eliminated.

## Connects To
- **Ch 9**: supplies the model family (Turbo/Davinci/Curie/Babbage/Ada) whose pricing this chapter quotes.
- **Ch 29**: continues the ChatGPT product/business narrative with the subsequent Plus subscription and API rollout timeline.
- **Ch 33**: revisits these same technical limitations (lack of common sense, stale training data, bias) as part of the book's closing limitations summary.
