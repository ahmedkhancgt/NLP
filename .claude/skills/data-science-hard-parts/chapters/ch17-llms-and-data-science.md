# Chapter 17: Large Language Models and the Practice of Data Science

## Core Idea
LLMs are already automating many programming-heavy data science tasks, but the "hard parts" — business knowledge, analytical/causal reasoning, and hypothesis generation — remain comparatively unexposed; the profession will likely shift weight away from pure programming/ML execution toward business-savvy analytical judgment.

## Frameworks Introduced
- **Task Exposure Score**: for each task, rate four basic skills (Business `x_B`, ML/stats `x_ML`, Programming `x_P`, Soft skills `x_S`) on a 0/1/3 (low/medium/high) scale, then compute:
  `Exposure = (0.2·x_B + 0.8·x_ML + 1.0·x_P − 0.2·x_S) − (0.2·x_B·x_ML + 0.2·x_B·x_P)`
  - Linear part: weights reflect assumed exposure ordering Soft < Business < ML < Programming.
  - Interaction terms (subtracted): ML/business and Programming/business combinations require analytical/critical-thinking skills that lower overall exposure — i.e., tasks requiring business judgment *combined with* technical skill resist automation more than either skill alone would suggest.
  - When to use: as a rough, directional (not precise) framework for assessing which of your own tasks/skills are more or less exposed to LLM automation — useful for career/skill-investment planning, not a rigorous forecasting tool.
- **Two Long-Run Scenarios for the DS Profession**: (1) nontechnical stakeholders become data-driven themselves, enhanced by AI, and the dedicated data scientist role shrinks; (2) data scientists become business-savvy and use AI to handle technical execution, making a purely-technical stakeholder-intermediary role less needed. Which wins depends on which skill (being data-driven vs. being business-savvy) is cheaper to acquire — unresolved as of writing.

## Key Concepts
- **Exposure** (per Eloundou et al. 2023): whether LLM access would cut the time to complete a task by ≥50%.
- **Emergent abilities**: unexpected capabilities (few-shot learning, chain-of-thought reasoning) that appear once model scale crosses a threshold.
- **O*NET task benchmark**: the author rates 15 standard data-scientist tasks from O*NET's occupational database using the exposure framework — not a perfect or complete list, but a useful proxy.
- **Programming vs. ML/stats vs. business vs. soft skills**: the four skill axes the author uses throughout the book and this chapter to classify both tasks and chapters by AI exposure.
- **"Human in the loop" caveat**: current LLMs still require an expert to prompt/guide them and catch hallucinations/errors — especially critical when outputs must make business sense, not just be technically correct.

## Mental Models
- Programming has the strongest *positive* correlation with AI exposure; science/critical thinking has the strongest *negative* correlation (per Eloundou et al.'s skill-exposure correlations) — this maps directly onto which parts of a data scientist's job are most at risk.
- "The hard part is coming up with good metrics and good hypotheses" — recurring theme: across A/B testing, data cleansing, and ML, the technical execution is increasingly automatable, but defining the right question, understanding business context, and generating causal hypotheses remain human comparative advantages (for now).
- Treat this chapter's conclusions as directional and perishable — the author explicitly warns that fast AI progress will likely outdate specific claims quickly, so the methodology (how to think about exposure) matters more than the specific numbers.

## Anti-patterns
- **Assuming "more ML/stats expertise" is the safest investment**: the chapter argues metaparameter tuning and tool selection are increasingly handled by AutoML/LLM playbooks — specializing purely in technical tuning may be less defensible than understanding *why* a method is appropriate for a business context.
- **Treating LLM output as automatically business-valid**: "programming with data" requires outputs that make business sense, which currently still needs a knowledgeable human reviewer — don't assume technical correctness implies business correctness.
- **Ignoring soft skills' fragility**: the chapter flags that soft skills, while currently safe, could become less relevant if AI eventually replaces one side of human-to-human interactions entirely — don't treat any current "safe" skill as permanently safe.

## Code Examples
```text
Exposure = (0.2·x_B + 0.8·x_ML + 1.0·x_P − 0.2·x_S) − (0.2·x_B·x_ML + 0.2·x_B·x_P)
where x ∈ {0 (low), 1 (medium), 3 (high)} for each of: Business, ML, Programming, Soft skills
```
- **What it demonstrates**: a simple, transparent (if subjective) scoring formula for ranking any task's or chapter's exposure to LLM automation, explicitly built to penalize tasks that combine technical skill with business judgment (since those are harder to automate than either alone).

## Reference Tables
*Table 17-1: Sample of data science tasks and exposure (abridged)*

| Task | Exposure |
|---|---|
| Deliver presentations of modeling/analysis results to management | Low |
| Recommend data-driven solutions to stakeholders | Low |
| Identify business problems addressable through data analysis | Low |
| Clean and manipulate raw data using statistical software | High |
| Test, validate, reformulate models for accurate prediction | High |
| Compare models using statistical performance metrics | High |
| Write new functions/applications in programming languages | High |

*Table 17-2: Book chapters sorted by exposure (abridged)*

| Chapter | Exposure |
|---|---|
| Ch 1 (So What), Ch 2 (Metrics Design), Ch 3 (Growth Decomp.), Ch 4 (2×2), Ch 5 (Business Cases), Ch 6 (Lift), Ch 7 (Narratives) | Low |
| Ch 8 (Datavis), Ch 10 (Linear Regression), Ch 15 (Incrementality), Ch 16 (A/B Tests) | Medium |
| Ch 9 (Simulation), Ch 11 (Data Leakage), Ch 12 (Productionizing), Ch 13 (Storytelling ML), Ch 14 (Prediction to Decisions) | High |

Overall: of 15 O*NET tasks evaluated, 40% rated high exposure, 20% medium, 40% low.

## Worked Example
A/B testing case study: business skill (defining/prioritizing the hypotheses backlog and outcome metric) vs. technical skill (experiment design, randomization, causal-identification assumptions, impact measurement). The author's claim: outside the top ~1% of frontier-knowledge cases, most of the *technical* side can already be automated/mediated by LLMs, but coming up with good metrics and hypotheses remains the genuinely hard, human-advantaged part. Data cleansing case study: executing the cleaning code (SQL/Python/R) is "the easy part"; deciding business-dependent judgment calls (e.g., should nulls become zeros? does this value make business sense?) is the hard part — potentially transferable to a well-trained nontechnical stakeholder with documented playbooks, or to an AI agent trained on those playbooks in the future. ML case study: choosing a technique and tuning metaparameters is increasingly handled by AutoML/LLM-suggested playbooks; the enduring hard part is generating hypotheses about *why* a feature should causally predict an outcome.

## Key Takeaways
1. LLMs are already materially boosting programming productivity for data scientists (cited surveys: >90% of developers using AI tools, some claims of 10x productivity gains).
2. Of 15 benchmark O*NET data-science tasks, ~40% show high AI exposure (mostly programming-heavy), ~40% show low exposure (business/analytical/communication-heavy).
3. The exposure-scoring framework explicitly penalizes tasks that combine technical and business skill, since the combination resists automation more than either skill alone.
4. Two plausible long-run scenarios exist for the DS profession — nontechnical stakeholders becoming data-driven, or data scientists becoming business-savvy — and which wins depends on relative skill-acquisition costs, which remains unresolved.
5. Across A/B testing, data cleansing, and ML workflows, technical execution is increasingly automatable, but defining the right metric, the right hypothesis, and business-sensible judgment calls remain comparatively hard to automate.
6. This book's own chapters split roughly along the same lines: Part I (business, narrative, metrics) chapters are low-exposure; Part II (ML-execution-heavy) chapters are higher-exposure — reinforcing that the book's core "hard parts" investment thesis is in the less-automatable skills.
7. All specific predictions here should be treated as directional and perishable given the pace of AI progress — the methodology matters more than the specific numbers.

## Connects To
- **Ch 1**: The business-knowledge emphasis ("data scientists should be as business-savvy as their stakeholders") is the throughline this chapter uses to argue for durable human comparative advantage.
- **Ch 13**: Hypothesis generation (ex ante storytelling) is explicitly named as the enduring hard part of ML that resists LLM automation.
- **Ch 16**: The A/B testing case study directly revisits the hypotheses-backlog and causal-identification concepts from that chapter to argue where automation can and cannot help.
- **Ch 15**: Causal reasoning/incrementality is flagged as a durable differentiator for humans over LLMs in both prediction improvement and decision-making contexts.
