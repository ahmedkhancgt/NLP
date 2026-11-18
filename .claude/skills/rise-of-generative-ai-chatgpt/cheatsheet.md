# Cheatsheet: Decision Rules & Quick Reference

## Should I trust this ChatGPT output as-is?

| Signal | Verdict |
|---|---|
| Simple arithmetic, calculus, basic stats, probability | Generally reliable (Ch 17) |
| Logic puzzles, numerical/visual pattern reasoning | Weak, especially visual — verify (Ch 17) |
| Hard-core/research-level quantitative problem | Low confidence — treat as draft only (Ch 17) |
| Fact/citation when asked for sources | High risk of fabrication — always verify independently (Ch 27) |
| Code for a common language (Python/Java/C++/JS) | Draft-quality; human review mandatory before production (Ch 16, 28) |
| Code for an uncommon language or very recent library/API | Unreliable — training cutoff (~2022) means it may not know it (Ch 28) |
| Query about events/data after ~2021 | Unreliable — training-data cutoff (Ch 27, 33) |
| Medical diagnosis / legal judgment / high-diligence doc | Never trust unverified — human specialist required (Ch 6, 20) |
| Qualitative research framing/outline | Good for ideation; evaluate rigor yourself (Ch 18) |

## Tells that an answer is "confidently wrong" (hallucination)
- Authoritative tone paired with a numeric/factual answer you haven't independently checked (Ch 10, 24).
- A cited source you cannot locate elsewhere — ChatGPT "frequently invents sources that don't truly exist" (Ch 27).
- Repetition of stock disclaimers ("As a language model developed by OpenAI...") — a symptom of training-data bias toward longer, "thorough-sounding" answers, not evidence of extra rigor (Ch 10).
- Code that references a library/function that doesn't exist, or silently alters names/sequences between turns (Ch 28).

## Rephrasing heuristic
If ChatGPT claims it "doesn't know" or refuses, try rephrasing before concluding the information is unavailable — prompt sensitivity means phrasing, not just knowledge, determines the answer (Ch 10).

## Guardrail-bypass pattern to watch for (red-teaming / policy)
A direct harmful request (e.g., "write ransomware code") is often refused, but a semantically equivalent *indirect* request (e.g., "help me encrypt all my files in Python") can still succeed. Any safety/policy review must test reframed prompts, not just direct ones (Ch 25).

## Which "problem-solving" chapter applies?

| Need | Chapter |
|---|---|
| Math/stats/logic puzzle capability ceiling | Ch 17 (quantitative) |
| Root-cause / multi-perspective analysis | Ch 18 (qualitative) |
| Concrete worked examples of both | Ch 23 |
| Beginner onboarding process | Ch 24 |

## Industry use-case lookup (which chapter covers X)

| Industry / Domain | Chapter(s) |
|---|---|
| Banking & Finance | Ch 3 (foundational), Ch 19 (ChatGPT-specific deep dive) |
| Government | Ch 5, Ch 31 (G20 case study) |
| Content Marketing | Ch 13 |
| Education / E-learning | Ch 14 (general), Ch 26 (EdTech industry) |
| Entertainment | Ch 15 |
| Coding | Ch 16 (constructive), Ch 28 (skeptical/benchmarked) |
| Healthcare | Ch 20 |
| E-commerce | Ch 21 |
| Hospitality | Ch 22 |
| Cybersecurity / National security | Ch 25, Ch 31 |
| Research | Ch 18, Ch 27 |

## Four-Pillar Ethical Risk Checklist (apply before any deployment)
1. **IP**: Could outputs infringe copyright/trademark? (Ch 4)
2. **Privacy**: Was training/input data collected and used lawfully? (Ch 4, Ch 3's GDPR/PCI DSS)
3. **Bias**: Could skewed training data cause discriminatory output? (Ch 4, Ch 6, Ch 20)
4. **Safety**: Could the system be misused for fake news, deepfakes, phishing, or malware? (Ch 4, Ch 6, Ch 25)

## Model-choice quick reference (GPT-3.5 era pricing/capability tiers, Ch 9-10)

| Model | Best for | Relative cost |
|---|---|---|
| Turbo | Conversational chat (powers ChatGPT) | Low |
| Davinci | Complex intent, cause/effect, audience-targeted summarization | Highest |
| Curie | Translation, classification, sentiment, summarization | Medium |
| Babbage | Moderate classification, semantic search | Low |
| Ada | Fast simple tasks: parsing, address correction, keywords | Lowest |

## The book's one recurring master rule
Across every industry and use case, the same closing instruction recurs: **ChatGPT is a productivity/ideation tool requiring human verification — never an unquestioned authority**, most explicit in healthcare (Ch 20), research (Ch 27), coding (Ch 28), and the book's final chapter (Ch 33).
