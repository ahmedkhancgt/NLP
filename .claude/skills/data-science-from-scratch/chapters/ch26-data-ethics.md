# Chapter 26: Data Ethics

## Core Idea
Data ethics is "a framework for thinking about right and wrong behavior involving data"; rather than prescribing rules, the chapter walks through real tensions — bad data products, accuracy/fairness tradeoffs, collaboration with repressive governments, model interpretability, recommendation systems' societal effects, biased training data, and data protection/privacy — to get readers to commit to considering ethical consequences, even without a single "right" answer.

## Frameworks Introduced
- **"Commit to consider ethical consequences" (the author's stance)**: rather than adopting a fixed manifesto or pledge, deliberately think through the downstream effects of your work; you don't have to agree with every "data ethics" advocate's conclusions, but you should understand the issues they raise.
  - When to use: as an ongoing practice throughout a data science career, especially as the scale of impact grows.
- **Accuracy/fairness tradeoff analysis**: when a model's predictions differ by group, multiple competing fairness arguments can be constructed from the *same* confusion-matrix data (equalized prediction rates, calibration/accuracy parity, false-positive-rate parity, false-negative-rate parity) — recognize there is no single, universally "correct" framing.
  - When to use: whenever a model's outcomes vary across sensitive groups; requires choosing which fairness notion matters for the specific context, since the underlying mathematical fairness definitions can be mutually incompatible.
- **Interpretability-vs-performance tradeoff**: weigh an explainable-but-weaker model (decision tree) against an opaque-but-stronger one (neural network), based on whether explanations are actually necessary (legal requirements, life-changing decisions, need to drive systemic — not just one-off — interventions).

## Key Concepts
- **Data ethics**: a framework for reasoning about right/wrong behavior specifically involving data; distinguished from generic "ethics" only by its data-related subject matter.
- **Scale effect of technology decisions**: a flawed algorithm applied nationwide (e.g., a parole-risk model) affects millions uniformly, unlike a flawed human decision-maker whose errors are limited and heterogeneous — this is why tech/data roles carry outsized ethical weight.
- **Building bad data products (unintentional harm)**: harmful outcomes (Microsoft's Tay bot generating offensive tweets; Google Photos mislabeling photos of black people as "gorillas") often arise not from anyone explicitly deciding to ship something unethical, but from failing to anticipate abuse or from biased training data plus model inaccuracy.
- **Biased data / distributional bias in embeddings**: word vectors (Ch21) learn whatever associations exist in their training corpus (e.g., gender associations with "software" if news coverage skews toward male software engineers); models trained on biased real-world outcome data (e.g., arrest records shaped by biased policing) risk *codifying* that bias rather than correcting for it.
- **Data protection tensions**: a single dataset (user behavior, income, connections) can have legitimate-seeming but ethically distinct downstream uses — advertisers, academic researchers, political campaigns, law enforcement — each raising separate concerns even under a technically-valid terms-of-service agreement.

## Mental Models
- **The same confusion-matrix data supports multiple, mutually exclusive "fairness" arguments**: the chapter's worked example shows four different data scientists each making a defensible-sounding fairness claim from the identical numbers — illustrating that "is this model fair?" often has no objective answer independent of which fairness definition and context you choose.
- **Ethics scales with technology's reach**: the ethical stakes of a decision aren't about whether it "involves data" per se, but about how many people and how severely it affects them — a mindset for calibrating how much scrutiny a given data product deserves.
- **Unintentional harm is still harm worth designing against**: most cited failures (Tay, Google Photos) weren't the result of anyone deliberating about ethics and choosing wrong — they resulted from not thinking through misuse/edge cases at all, which is itself the actionable lesson ("think about how the things you build could be abused").
- **Recommendation algorithms optimize for engagement, which can diverge from societal benefit**: YouTube recommending conspiracy-theory videos because people click on them is a direct tension between a platform's business incentive (more watch time/ads) and broader harms, without an agreed-upon "right" resolution.

## Anti-patterns
- **Shipping a product without considering how it could be abused**: Tay's open "repeat what's tweeted at me" design had no defense against being weaponized for offensive content.
- **Training/deploying a model on nonrepresentative data without testing for embarrassing or harmful edge-case outputs**: the Google Photos "gorilla" mislabeling stemmed from inadequate diversity in training/testing, compounded by the real-world offensiveness of that particular error category.
- **Treating "the model doesn't use group membership as a feature" as proof of fairness**: a model can still produce very different outcomes by group (Table 26-2) purely through correlated features — formal non-use of a protected attribute doesn't guarantee fair outcomes.
- **Assuming a terms-of-service agreement settles the ethics question**: having the *legal* right to use/sell user data (per ToS) doesn't resolve whether doing so (to advertisers, political campaigns, law enforcement) is *ethical* — these are separate questions.
- **Training predictive models directly on biased real-world outcome data without scrutiny**: a parole/rearrest model that perfectly predicts biased real-world rearrest patterns risks mechanically perpetuating whatever bias produced those patterns in the first place.

## Code Examples
This chapter is conceptual/discursive rather than code-driven; no from-scratch code or algorithms are introduced. (See Ch21's "Word Vectors" and Ch23's "Recommender Systems" for the technical machinery whose ethical implications this chapter discusses.)

## Reference Tables
| Ethical tension | Example from the chapter |
|---|---|
| Unintentional harm from bad products | Microsoft Tay bot, Google Photos "gorilla" mislabeling |
| Accuracy vs. fairness | 4 competing fairness arguments from one confusion matrix |
| Collaboration under restrictive conditions | Repressive-government content/surveillance request |
| Interpretability vs. performance | Decision tree / neural net / human expert for attrition prediction |
| Recommendation systems' societal effects | YouTube conspiracy-video recommendations, search autocomplete filtering |
| Biased training data | Word-vector gender/occupation associations; biased rearrest data |
| Data protection / privacy | Selling user data to advertisers, academics, political campaigns, law enforcement |

## Worked Example
A hiring/action-prediction model is "pretty good" overall (20% action rate among "unlikely" predictions, 60% among "likely"). Broken down by group A vs. B, four data scientists each make a different fairness claim from the same numbers: (1) the model predicts "likely" for 80% of group B vs. 20% of group A — unequal treatment; (2) predictions mean the same thing regardless of group (20%/60% action rates hold within each group) — the model is "accurate" and well-calibrated; (3) group B is falsely labeled "likely" at 32% vs. group A's 8% — stigmatizes B; (4) group A is falsely labeled "unlikely" at 16% vs. group B's 4% — stigmatizes A. The chapter explicitly declines to declare a winner, noting the "right" framing may depend on context (which groups, what the prediction is used for, whose interests are at stake).

## Key Takeaways
1. Data ethics is not a fixed rulebook — it's a commitment to seriously consider the consequences of your work, even when reasonable people disagree on conclusions.
2. The scale of technology means individual data-science decisions can have outsized, uniform effects on large populations — this raises (but doesn't by itself resolve) the ethical stakes.
3. Many real-world "data ethics failures" stem from not anticipating misuse or edge cases, not from anyone explicitly choosing an unethical path — proactively think about abuse scenarios.
4. Accuracy and fairness can genuinely trade off against each other, and different (valid) fairness definitions can point in opposite directions from the same data — there's often no uniquely "correct" answer, only a context-dependent choice.
5. Training data biases (word embeddings, real-world outcome data shaped by biased processes) can be learned and perpetuated by models — awareness and mitigation techniques exist but rarely eliminate bias completely.
6. Legal permission (terms of service) to use data doesn't settle whether a given use is ethical — these are separate, both-necessary questions.

## Connects To
- **Ch21 (NLP)**: the word-vector bias discussion directly extends the embeddings built there.
- **Ch23 (Recommender Systems)**: the YouTube/search-autocomplete discussion directly extends the recommendation techniques built there.
- **Ch17 (Decision Trees)** / **Ch18-19 (Neural Networks/Deep Learning)**: the interpretability-vs-performance tradeoff example contrasts these model families directly.
- **"Ethics and Data Science" by Loukides, Mason, and Patil**: external book recommended for a more practical treatment of putting data ethics into practice.
