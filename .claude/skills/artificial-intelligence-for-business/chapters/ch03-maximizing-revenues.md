# Chapter 3: Maximizing Revenues (Thompson Sampling)

## Core Idea
An online retail business wants to find which of 9 untested marketing strategies converts the most customers to a premium plan, without emailing millions of people or waiting on slow, expensive A/B tests — solved with Thompson Sampling, an online-learning (multi-armed bandit) algorithm that identifies the best strategy in a few thousand rounds while an outside observer sees no states, actions, or neural networks at all.

## Frameworks Introduced
- **Multi-Armed Bandit / Online Learning**: a fundamentally different branch of AI from Q-Learning/Deep Q-Learning — no states or actions to define, just "rounds" where one of several options is tried and a reward (convert or not) is observed.
  - When to use: picking the best of several discrete options (ad variants, pricing strategies, UI variants) when you can only learn by trying them on real traffic, one interaction at a time, and want to minimize the cost of trying bad options along the way.
- **Thompson Sampling**: maintain a Beta distribution per strategy (parameterized by its accumulated successes/failures); at each round, sample one random value from each strategy's distribution and play whichever strategy sampled highest; update that strategy's success/failure count based on the outcome.
  - When to use: exactly the business problem above — it converges quickly, needs no hyperparameter tuning (no learning rate, no discount factor), and inherently balances exploration (trying less-proven strategies) against exploitation (favoring proven winners).
  - How: Step 1 — draw `theta_i(n) ~ Beta(N1_i(n)+1, N0_i(n)+1)` for every strategy i; Step 2 — play `s(n) = argmax(theta_i(n))`; Step 3 — increment `N1` if reward=1, else increment `N0` for the played strategy. Repeat every round.
- **Random Selection benchmark**: pick a uniformly random strategy every round, with no learning at all — used purely as the baseline to prove Thompson Sampling's value.
  - When to use: always run a trivial baseline alongside a bandit algorithm so its real-world lift can be quantified, not just assumed.
- **Relative Return metric**: `(TotalReward_ThompsonSampling - TotalReward_RandomSelection) / TotalReward_RandomSelection * 100` — the percentage improvement Thompson Sampling delivers over blind random selection on the identical simulated customer stream.
- **Regret Curve**: at each round, plot `(cumulative reward of the single best-known strategy so far) - (cumulative reward of the algorithm being evaluated)`.
  - When to use: visualizing *convergence*, not just final performance — a flattening regret curve proves the algorithm is homing in on the best option, while a random strategy's regret curve never stops climbing.

## Key Concepts
- **Strategy / Arm**: one of the 9 marketing approaches being tested; analogous to a "slot machine arm" in the classic bandit framing.
- **Round (n)**: one customer's visit to the site, during which exactly one strategy is shown and one binary outcome (subscribe or not) is observed.
- **Reward**: 1 if the customer subscribes after seeing the deployed strategy, 0 otherwise.
- **N1_i(n) / N0_i(n)**: running counts of successes/failures for strategy i up to round n — these two numbers are the *entire* memory the algorithm needs per strategy.
- **Beta distribution**: the probability distribution sampled per strategy; shifts rightward (higher expected value) as a strategy accumulates more 1-rewards, and leftward as it accumulates more 0-rewards.
- **Simulated rewards matrix**: a 10,000 (customers) x 9 (strategies) matrix of 0/1 outcomes, used only to *simulate* what a real customer would have done — in production, this matrix would not exist; you only observe the outcome of whichever single strategy was actually shown to each real customer.

## Mental Models
- **No states, no actions, no neural network — and that's the point**: this chapter is explicitly positioned as "a very different kind of Artificial Intelligence" from Ch1/Ch2's Q-Learning/Deep Q-Learning — recognizing which problems are bandit problems (discrete options, binary/simple feedback, no sequential state) vs. MDP problems (states evolve based on actions) is itself a key skill.
- **Two numbers per arm is enough memory**: unlike Q-Learning's full table or DQN's neural network weights, Thompson Sampling's entire "model" is just two integers (successes, failures) per strategy — extreme simplicity is a feature, not a limitation, for this class of problem.
- **Exploration and exploitation are automatic, not hand-tuned**: there's no epsilon parameter to set — the Beta distribution's own shape (wide/uncertain early, narrow/confident later) naturally produces more exploration when a strategy is unproven and more exploitation once it's clearly ahead.
- **The regret curve reveals convergence, final reward alone does not**: two algorithms could end with similar total reward while one is still thrashing and the other locked onto the best arm rounds ago — plot regret whenever you want to claim "the algorithm learned," not just "the algorithm scored well."

## Anti-patterns
- **Emailing or testing on the entire customer base to find the best strategy**: costly and risks spamming/annoying the very customers you want to convert — online learning on a live stream of visitors (with the losing strategies shown progressively less) is far cheaper.
- **Treating the simulated rewards matrix as something that exists in real deployment**: it's a simulation device only, built here purely so the book can later check that the algorithm found the true best strategy (Strategy 7) — in production you only ever observe the one outcome for the one strategy actually shown to each real customer.
- **Judging an online-learning algorithm by final score alone**: always also plot the regret curve to confirm actual convergence, not just a lucky high score from the specific random simulation run.
- **Reaching for Q-Learning/Deep Q-Learning machinery (states, actions, Bellman equation) for a problem with no sequential state**: adds complexity the bandit-style problem doesn't need — recognize the online-learning problem shape and reach for Thompson Sampling directly.

## Code Examples
```python
# Setting the parameters
N = 10000   # customers (rounds)
d = 9       # strategies

# Creating the simulation (ground truth, for evaluation only — never used by the algorithm itself)
conversion_rates = [0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01]
X = np.array(np.zeros([N, d]))
for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i, j] = 1
```
- **What it demonstrates**: building a simulated "ground truth" rewards matrix purely for evaluation — the algorithm itself never sees `conversion_rates` directly.

```python
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
for n in range(0, N):
    strategy_ts = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] += 1
    else:
        numbers_of_rewards_0[strategy_ts] += 1
```
- **What it demonstrates**: the complete Thompson Sampling algorithm — per-strategy Beta sampling, argmax selection, and success/failure count update — in roughly 10 lines.

```python
# Regret of Thompson Sampling at each round
rewards_strategies = [0] * d
regret = []
for n in range(0, N):
    # ... (Thompson Sampling selection as above) ...
    for i in range(0, d):
        rewards_strategies[i] += X[n, i]
    total_reward_bs = max(rewards_strategies)
    regret.append(total_reward_bs - total_reward_ts)
```
- **What it demonstrates**: computing the regret curve by tracking, at every round, the gap between the best strategy knowable so far and the algorithm's actual accumulated reward.

## Reference Tables
| Strategy index | Simulated conversion rate |
|---|---|
| 0 | 5% |
| 1 | 13% |
| 2 | 9% |
| 3 | 16% |
| 4 | 11% |
| 5 | 4% |
| 6 (Strategy 7) | 20% (the true best) |
| 7 | 8% |
| 8 | 1% |

| Algorithm | Needs hyperparameters? | Memory per strategy | Converges? |
|---|---|---|---|
| Random Selection | No | None | No (flat/rising regret forever) |
| Thompson Sampling | No | 2 counters (successes, failures) | Yes (regret flattens) |

## Worked Example
With 100 million customers and a premium plan priced at \$100/year, the book first motivates the whole chapter with a toy example: a 1% vs. an 11% conversion strategy differ by \$1 billion in annual extra revenue — purely from picking the right strategy. The real case study then simulates 9 strategies (true conversion rates 1%-20%, Strategy 7 at index 6 being the true best at 20%) over 10,000 customer rounds. Running Thompson Sampling alongside a Random Selection benchmark on the identical simulated data yields a **91% relative return** — Thompson Sampling almost doubles Random Selection's total conversions. A histogram of selected strategies confirms Strategy 7 was chosen far more often than any other, and the regret curve shows Thompson Sampling's regret flattening (converging) while Random Selection's regret keeps climbing with no sign of convergence. Scaled to the full 100-million-customer base, deploying the discovered 20%-conversion strategy is projected to generate **\$2 billion** in extra annual revenue — while also minimizing the number of customers shown an inferior, money-losing strategy during the learning process itself.

## Key Takeaways
1. Recognize when a business problem is a multi-armed-bandit/online-learning problem (discrete options, immediate binary feedback, no evolving state) rather than an MDP — it calls for a completely different, far simpler algorithm.
2. Thompson Sampling needs no state/action design and no hyperparameters (no learning rate, no discount factor, no exploration schedule) — just two running counts per option.
3. Always benchmark a bandit algorithm against a trivial Random Selection baseline, and report the relative return, not just absolute reward.
4. Plot the regret curve (not just final score) to confirm genuine convergence toward the best option over time.
5. The business value of fast convergence is real money: every round spent on an inferior strategy is foregone revenue and a worse customer experience — minimizing rounds-to-convergence is itself the business objective, not a side benefit.

## Connects To
- **Ch1/Ch2**: explicitly contrasted as "a very different kind of AI" — no states, actions, or neural networks, unlike the MDP-based Q-Learning and Deep Q-Learning case studies.
- **Ch6 (Annex 3)**: the Q&A section directly revisits this chapter's distribution intuition (why one Beta distribution beats another), Thompson Sampling's ability to accept delayed feedback, and further real-world bandit applications (ad CTR optimization, online algorithm selection, clinical trial arm selection).
