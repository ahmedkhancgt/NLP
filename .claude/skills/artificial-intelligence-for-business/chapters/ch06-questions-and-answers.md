# Chapter 6: Questions and Answers (Annex 3)

## Core Idea
A curated Q&A appendix, organized by case study, that fills in the specific follow-up confusions students actually had — clarifying terminology (Plan vs. Policy), justifying hyperparameter/architecture choices made without full derivation in the main text, and extending the case studies' concepts (Thompson Sampling, distributions) to further real-world applications.

## Frameworks Introduced
This chapter introduces no new algorithms; it consolidates and clarifies concepts from Chapters 1-3 via direct Q&A. Treat it as a cross-reference/clarification layer over the three case studies rather than new material.

## Key Concepts
- **Plan vs. Policy** (Ch1 clarification): the "plan" is the process of *defining* the environment's input states; the "policy" is the function that takes those states and returns actions — in the warehouse case study, defining the warehouse environment is the plan, and the trained Q-table-driven decision process is the policy.
- **Discount factor tuning** (Ch1 clarification): gamma is not derived analytically — it's found by experimentation, starting from 1 (no discount) and decreasing gradually while observing whether results improve, repeating until an optimal value is found.
- **Argmax vs. Softmax choice** (Ch2 clarification): Argmax suits simple/business problems where little exploration is needed; Softmax suits complex problems (games, robot locomotion) that genuinely need exploration vs. exploitation balancing — business problems are explicitly called out as "not too complex," so Argmax suffices.
- **64/32-neuron architecture choice** (Ch2 clarification): not derived from first principles — start from classical architectures seen in published work (ImageNet, ResNet, Inception, MobileNets), try them, and keep what works; the book's own 2-layer (64, 32) network was chosen this way and validated empirically. Overfitting is then addressed separately via Early Stopping and Dropout.
- **Distributions, x/y axes** (Ch3 clarification): repeatedly sampling a strategy-selection experiment (e.g. 100 trials, repeated 500 times) and histogramming the resulting frequencies produces a bell-shaped (Central Limit Theorem) distribution; the x-axis is the possible frequency values, the y-axis is how often each frequency was obtained.
- **Why a right-skewed distribution is "best"** (Ch3 clarification): in a 0 (loss) to 1 (gain) framing, a distribution skewed toward 1 means more of its probability mass represents wins — this is the visual/intuitive read used to explain why one strategy's beta distribution should be preferred over another's.
- **Thompson Sampling and delayed feedback** (Ch3 clarification): Thompson Sampling can keep performing updates (re-estimating distributions, sampling, selecting) using existing data while waiting for a real-world experiment's results to come back — delayed feedback doesn't block the algorithm from continuing to operate.
- **Multi-Armed Bandit applications beyond ad testing** (Ch3 extension): Conversion Rate Optimization (comparing ad CTRs) is the classic example; others given include online algorithm selection (treating each candidate recommendation algorithm as an "arm") and clinical treatment selection (treating each possible treatment as an "arm," with reward defined on a 0-1 scale of treatment effectiveness, optimizing to minimize cumulative regret — i.e., minimize how often a sub-optimal treatment is chosen while searching for the best one).

## Mental Models
- **Hyperparameters and architecture choices are empirical, not derived**: both the discount factor (Ch1) and the neural-network layer sizes (Ch2) are explicitly described as "try, observe, adjust" processes grounded in experimentation and precedent (published architectures), not closed-form calculations — a realistic expectation-setting point for practitioners new to applied AI.
- **"Business problems are not too complex" is a recurring simplifying assumption**: it justifies Argmax over Softmax in Ch2 and implicitly underlies why the book's three case studies use comparatively simple models rather than the Annex 2 state-of-the-art models.
- **Delayed feedback is normal in real-world deployment, and Thompson Sampling tolerates it gracefully**: the algorithm's incremental, per-round update structure means it doesn't need to "pause" while waiting on slow real-world signals (e.g. a customer's eventual subscription decision) — it keeps operating on whatever data has already arrived.

## Anti-patterns
- **Expecting a formula for the "right" discount factor or network architecture**: both are explicitly empirical/trial-based in this book's own account of how they were chosen — searching for a closed-form answer is the wrong mental model.
- **Confusing "plan" and "policy" as interchangeable terms**: they refer to two distinct phases (defining the environment vs. the trained decision function) and mixing them up muddles the MDP framework from Chapter 1.
- **Assuming Multi-Armed Bandit applications are limited to advertising**: the Q&A explicitly extends the same technique to algorithm selection and clinical treatment selection, underscoring that "arm" and "reward" are abstractions that generalize well beyond the book's own online-retail case study.

## Code Examples
This chapter is a Q&A/clarification section with no new code — it references and explains code and concepts already presented in Chapters 1-3 rather than introducing new implementations.

## Reference Tables
| Q&A topic | Clarifies concept from |
|---|---|
| Plan vs. Policy | Ch1 (MDP) |
| Discount factor tuning | Ch1 (Q-Learning) |
| Argmax vs. Softmax | Ch2 (Deep Q-Learning) |
| 64/32-neuron architecture + overfitting (Early Stopping, Dropout) | Ch2 (Deep Q-Learning) |
| Distribution x/y axes, skewed distributions | Ch3 (Thompson Sampling) |
| Delayed feedback tolerance | Ch3 (Thompson Sampling) |
| Further bandit applications (CTR, algorithm selection, clinical trials) | Ch3 (Thompson Sampling) |

## Worked Example
A student asks why, in a comparison of distributions D1 through D5 (0 = loss, 1 = gain), D5 is the best choice over D3: the answer given is that D5 is skewed so that average outcomes cluster near 1 (more wins/gains), while "all casino machines nowadays are carefully programmed to have a distribution like D1 or D3" — a deliberately provocative real-world aside used to cement the intuition that distribution *shape*, not just its mean, determines whether an option is actually favorable. A second question asks about applying Multi-Armed Bandits to find the best clinical treatment: each possible treatment becomes an "arm," reward is defined between 0 (no effect) and 1 (full cure), and the explicit goal is framed as minimizing cumulative regret — i.e., finding the best treatment as fast as possible while minimizing how often patients receive a known-inferior one during the search.

## Key Takeaways
1. Hyperparameters like the discount factor and neural-network architecture sizes are chosen by experimentation and precedent, not derived analytically — expect to iterate.
2. Argmax is the right default for comparatively simple business problems; reserve Softmax-style exploration for genuinely complex domains (games, robotics).
3. Keep "plan" (environment definition) and "policy" (the trained action-selection function) conceptually distinct.
4. Thompson Sampling's incremental update structure makes it naturally tolerant of delayed real-world feedback.
5. The Multi-Armed Bandit abstraction (arms, rewards, regret) generalizes far beyond advertising — algorithm selection and clinical treatment selection are both given as legitimate real-world extensions.

## Connects To
- **Ch1**: directly answers follow-up questions on the Plan/Policy distinction and discount-factor tuning from the warehouse Q-Learning case study.
- **Ch2**: directly answers follow-up questions on Argmax-vs-Softmax and the Brain's 64/32-neuron architecture (plus Early Stopping and Dropout as the overfitting fix) from the Deep Q-Learning case study.
- **Ch3**: directly answers follow-up questions on distribution intuition, delayed feedback, and further applications of Thompson Sampling.
- **External resources**: a Beta-distribution intuition guide (stats.stackexchange.com) and two linked papers/articles on plans vs. policies in the reinforcement-learning literature are pointed to for readers wanting a deeper dive.
