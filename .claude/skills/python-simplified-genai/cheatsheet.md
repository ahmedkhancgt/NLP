# Cheatsheet

## Which sorting algorithm to use?

| Situation | Use |
|---|---|
| Teaching/learning sorting fundamentals | Bubble sort (simplest, maps directly to a flowchart) |
| Tiny list, memory extremely constrained | Bubble sort (in-place, no extra memory) |
| Data nearly sorted already | Insertion sort (fast on near-sorted input) |
| Need guaranteed O(n log n), consistent performance, have memory to spare | Merge sort |
| Need average-case speed on large data, memory-constrained | Quick sort (watch for worst-case O(n²) with bad pivots) |
| Production code in general | Use the language/library's built-in sort (e.g. Pandas `sort_values()`); don't hand-roll |

## Python variable type tells

| Symptom | Likely type/issue |
|---|---|
| Value in quotes | string |
| Whole number, no decimal | int |
| Number with decimal point | float |
| `True`/`False` | bool |
| `[...]` | list (mutable, ordered) |
| `(...)` | tuple (immutable, ordered) |
| `{key: value, ...}` | dict |
| `None` | explicit absence of value |
| "could not convert string to float" error | a natural-language word (e.g. "five") was passed where a number was expected — Python has no NLU |

## Pandas gotchas

| Gotcha | Fix / explanation |
|---|---|
| `axis=0` vs `axis=1` confusion | 0 = rows (vertical), 1 = columns (horizontal) |
| Column looks numeric but operations fail | run `.info()` — it may actually be stored as string (check dtype) |
| Missing values breaking a plot/calculation | `None`/`NaN` present — handle explicitly with `.fillna()`/`.replace()`/drop before analysis |
| Merge drops rows unexpectedly | default `how="inner"` only keeps matching keys — use `outer`/`left`/`right` if you need to retain unmatched rows |
| GitHub CSV import fails / returns HTML | used `www.github.com` view URL instead of `raw.githubusercontent.com` |

## Join-type decision rule

| Want to keep... | Use |
|---|---|
| Only rows present in both tables | `inner` |
| Everything from both, gaps filled with NaN | `outer` |
| Everything from the left table | `left` |
| Everything from the right table | `right` |

## CNN / ML decision rules

| Question | Answer |
|---|---|
| Should I train a CNN from scratch? | No — use transfer learning (pre-trained base model, e.g. ResNet34) unless you have a specific research reason not to. |
| Model accuracy is too low — rewrite the architecture? | No — first check if the dataset is too small/narrow/unbalanced. Model capability is bound by data, not code. |
| How do I pick a learning rate? | Run `lr_find()`; pick a rate in the stable "slide" region, not the most aggressive candidate. |
| Is the problem suited to sequential logic or ML? | If you can write exact if/else rules that cover all cases -> sequential logic. If the rules are too subtle/variable to enumerate (e.g. "what does an ear look like") -> ML. |
| How much of a CNN project is data cleaning? | Typically 50-65% of total effort — budget accordingly. |

## Deployment decision rules

| Question | Answer |
|---|---|
| Where should logic live: client or server? | Server, unless the app is simple with no sensitive data/proprietary logic (e.g., a basic calculator can be client-side JS). |
| Gradio live link expired/temporary — now what? | Deploy to Hugging Face Spaces (or another host) for persistent access. |
| App works locally but fails on another machine | Check `requirements.txt` — pin exact library versions used during development. |
| Secrets/API keys in `app.py`? | Never — use environment variables or a separate config file. |

## GenAI-usage tells (when to push back)

| Signal | Response |
|---|---|
| GPT-4o's solution feels overcomplicated | Ask explicitly: "can you simplify this logic?" |
| GPT-4o infers a dataset-specific structural fact (e.g. file path convention) | Verify yourself against the actual data — GenAI cannot know undocumented dataset conventions. |
| GenAI output seems biased/skewed (image generation, text) | Run the same prompt many times and compare the distribution to a known real-world baseline before concluding; try disambiguating the prompt. |
| You accepted the first answer without understanding why it works | Stop — ask "why does this work?" before moving on; understanding > working code. |

## Bias red flags in a project

| Red flag | Why it matters |
|---|---|
| Training data skewed toward one demographic/group | Model will underrepresent/misclassify underrepresented groups |
| Using historical decision data (e.g. past hiring records) uncritically | May encode and perpetuate past discrimination |
| No human oversight on automated decisions (hiring, lending, healthcare) | Removes the safety net that catches bias algorithms miss |
| No disclosure that content/decisions are AI-generated/assisted | Undermines transparency and user trust |
