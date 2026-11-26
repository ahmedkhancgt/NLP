# Chapter 1: Optimizing Processes (Q-Learning)

## Core Idea
An autonomous warehouse robot must always take the shortest route to the current top-priority storage location (with an optional stop at a second-priority location) — solved from scratch with a Markov Decision Process and tabular Q-Learning, built entirely on a hand-populated reward matrix and the Bellman equation.

## Frameworks Introduced
- **Markov Decision Process (MDP)**: a tuple (S, A, T, R) — states, actions, a transition rule T (here: uniform-random), and a reward function R. Key assumption: the next state depends only on the current state and action (no memory of history).
  - When to use: any sequential decision problem where you can enumerate states and actions and define what reward each transition earns.
  - How: encode states/actions as integers (0 to N-1) so they map directly onto matrix rows/columns; build R as an S x S matrix where cell (s,a) is 0 (can't move there), 1 (can), or a large number (the goal).
- **Q-Learning (tabular)**: learn a Q-value Q(s,a) for every (state, action) pair via repeated random exploration and the Bellman equation, until the Q-matrix encodes the optimal policy.
  - When to use: discrete, finite state/action spaces small enough for a full S x A table (here 12x12).
  - How: initialize Q to all zeros; repeat ~1000 times: pick a random current state, pick a random *playable* action (reward > 0), compute the Temporal Difference, update Q via the Bellman equation with learning rate alpha and discount factor gamma.
- **Argmax vs. Softmax action selection**: Argmax always picks the highest-Q action (deterministic, simple); Softmax samples from a probability distribution over actions (adds exploration).
  - When to use: Argmax for simple/business problems where little exploration is needed; Softmax for complex problems (games, robotics) where exploration vs. exploitation must be balanced.
- **Automating reward attribution + intermediary routing**: instead of manually editing the reward matrix for a new target, copy it (`R_new`) and set the destination cell to the goal reward inside the `route()` function itself; chain two `route()` calls (`best_route()`) to force a path through an intermediary location.
  - When to use: whenever the "goal" changes at runtime and shouldn't require redefining the whole model; chaining two single-leg routes is simpler and more robust than trying to bias the reward matrix with an in-between "high but not highest" reward.

## Key Concepts
- **State**: the robot's current location, encoded as an integer index (A-L -> 0-11).
- **Action**: a possible next-location move, encoded with the same indexes as states.
- **Reward function R(s,a)**: a matrix; 0 = move not possible, 1 = possible routine move, large value (1000) = the goal location.
- **Q-value Q(s,a)**: a learned number representing how good it is to take action a in state s; starts at 0 for every pair.
- **Temporal Difference (TD)**: `R(s,a) + gamma * max(Q(s', .)) - Q(s,a)` — the "surprise" signal driving learning; high TD = good surprise (reinforce), small/negative TD = frustration.
- **Bellman equation**: `Q(s,a) <- Q(s,a) + alpha * TD(s,a)` — the update rule that incrementally moves Q toward its optimal value.
- **Learning rate (alpha)**: controls update step size; higher = faster but less stable learning; range (0,1), e.g. 0.9 in this case study.
- **Discount factor (gamma)**: controls how much future reward matters vs. immediate reward; range [0,1], e.g. 0.75 here.
- **Policy function (pi)**: maps a state to the action to play; the "optimal policy" pi* maximizes accumulated reward over time — this is literally what the final Q-matrix encodes.

## Mental Models
- **Restructure the problem into matrices first, learn second**: almost all the real design work is choosing the state/action encoding and populating the reward matrix by hand; the Q-Learning loop itself is a short, generic block of code reused unchanged.
- **The Q-matrix IS the trained model**: there's no separate "inference" step beyond `argmax(Q[state,])` — training produces a lookup table, and using the AI in production is just reading that table.
- **Use indexes that equal matrix positions**: encoding locations as sequential integers 0..11 is not arbitrary — it's what lets `location_to_state` and matrix row/column indexing become the same operation.
- **Prefer function composition over reward-matrix hacking**: biasing rewards to force a waypoint (give J->K reward 500, or punish J->F with -500) is fragile and doesn't generalize; chaining `route()` twice via `best_route()` is simpler, always correct, and needs only two extra lines of code.

## Anti-patterns
- **Manually re-editing the reward matrix every time the goal changes**: doesn't scale and risks human error — compute the goal-reward update automatically inside the function from its own `ending_location` argument instead.
- **Trying to automatically bias intermediate-route rewards (ideas 1 and 2)**: the book explicitly flags this as "easy to see conceptually, very difficult to implement automatically" since the index of the location that *leads into* the intermediary location depends on the whole route — reach for function composition instead.
- **Choosing Softmax for a simple, low-complexity business problem**: adds unnecessary stochastic exploration where Argmax's deterministic greedy choice is sufficient and easier to reason about.

## Code Examples
```python
# Defining the states
location_to_state = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                      'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# Defining the rewards (0 = no path, 1 = normal path, 1000 = top-priority goal at G)
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              # ... full 12x12 matrix
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              # ...
             ])
```
- **What it demonstrates**: encoding a physical layout as integer states/actions and a reward matrix — the entire "environment definition" step of an MDP.

```python
gamma = 0.75
alpha = 0.9
Q = np.array(np.zeros([12,12]))

for i in range(1000):
    current_state = np.random.randint(0,12)
    playable_actions = [j for j in range(12) if R[current_state, j] > 0]
    next_state = np.random.choice(playable_actions)
    TD = R[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD
```
- **What it demonstrates**: the complete tabular Q-Learning training loop — random exploration, Temporal Difference computation, Bellman-equation update.

```python
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    # ... re-run Q-Learning on R_new ...
    route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]
```
- **What it demonstrates**: making the goal a runtime parameter (automatic reward update) and composing two routes to force a waypoint — the production-ready version of the model.

## Reference Tables
| Element | Encoding | Example |
|---|---|---|
| State | integer 0-11 | Location G -> state 6 |
| Action | same integers as states | Action 10 -> move toward location K |
| Reward | matrix cell (s,a) | 0 = blocked, 1 = normal move, 1000 = top-priority goal |

| Parameter | Role | Value used |
|---|---|---|
| gamma (discount factor) | weight on future Q-value | 0.75 |
| alpha (learning rate) | update step size | 0.9 |
| Training iterations | random (s,a) samples | 1000 |

## Worked Example
A 12-location warehouse (A-L) with an autonomous robot: location G is top priority (reward 1000), K is second priority. After 1000 iterations of tabular Q-Learning (gamma=0.75, alpha=0.9) on the hand-built 12x12 reward matrix, `route('E','G')` returns one of two equally optimal 7-step paths (e.g. `['E','I','J','F','B','C','G']`). Making the reward update automatic (via `R_new`) reproduces the same result without manual matrix edits. Finally, `best_route('E','K','G')` — composing `route('E','K')` and `route('K','G')` — reliably forces the path through the second-priority location K, yielding `['E','I','J','K','L','H','G']` every time.

## Key Takeaways
1. An MDP's real work is defining states, actions, and the reward matrix — the Q-Learning loop itself is short and reusable.
2. Encode states/actions as matrix indexes so environment definition and Q-table lookups are the same operation.
3. The trained Q-matrix doubles as the deployed model: production inference is just `argmax(Q[state,])`.
4. Prefer automatic, parameterized reward updates (and function composition for waypoints) over manually hand-tuning the reward matrix for every new goal.
5. Pick Argmax for simple, low-exploration business problems; reserve Softmax for genuinely complex decision spaces.

## Connects To
- **Ch2**: extends this same Q-Learning foundation into Deep Q-Learning when the state/action space becomes continuous/too large for a table.
- **Ch4 (Annex 1)**: the neural-network fundamentals needed to understand "the Brain" introduced in Chapter 2.
- **Ch6 (Annex 3)**: the Q&A section revisits the Plan/Policy distinction and the discount-factor tuning process from this chapter directly.
