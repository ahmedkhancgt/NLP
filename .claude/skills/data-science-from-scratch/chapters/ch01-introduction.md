# Chapter 1: Introduction

## Core Idea
Data science is the practice of extracting insight from messy data by combining programming, statistics, and domain expertise; the book teaches this by building every tool from scratch rather than relying on libraries, using a running fictional company, DataSciencester, as the vehicle.

## Frameworks Introduced
- **Degree centrality**: the number of direct connections (friends) a node has in a graph.
  - When to use: quick, cheap way to rank "central" or "important" nodes in a social network.
  - How: count the length of each user's friend list; sort descending.
  - Limitation: doesn't match intuition when a low-degree node sits in a structurally important position (addressed later with betweenness/eigenvector centrality in Ch22).
- **Friend-of-a-friend (FOAF) recommendation**: suggest connections via a `Counter` over friends' friends, excluding self and existing friends.
  - When to use: "people you may know" style features.
  - How: for each friend, iterate their friends, filter out the user and existing friends, tally with `Counter`.
- **Interest-based matching**: build a bidirectional index (`interest -> user_ids`, `user_id -> interests`) to find shared-interest users in O(1) lookup instead of O(n) scans.
  - When to use: whenever the same lookup will be repeated many times — pay the index-building cost once.

## Key Concepts
- **Data scientist** (author's working definition): someone who extracts insights from messy data.
- **DataSciencester**: the book's running fictional "social network for data scientists," used as a dataset source throughout.
- **Adjacency list**: a dict mapping each node id to a list of neighbor ids; far faster to query than a raw list of edge pairs.
- **Bucketing**: grouping continuous values (e.g., tenure) into discrete bins to make small/sparse data more interpretable.
- **Index (defaultdict)**: a precomputed lookup structure trading one-time build cost for fast repeated queries.

## Mental Models
- **"Don't get hung up on code details yet"**: the book favors showing working code early and deferring full explanation (Python crash course is Ch2) — accept partial understanding and refine later.
- **Eyeballing vs. modeling**: ad hoc cutoffs (e.g., `years_experience < 3.0`) are a legitimate quick first pass but are flagged as provisional, pointing forward to proper regression/classification models (Ch14, Ch16).
- **Raw list vs. index trade-off**: the recurring theme that restructuring data (list of pairs -> dict of lists) is often the real "analysis" work, before any statistics happens.

## Anti-patterns
- **Scanning the full list of pairs on every query**: `data_scientists_who_like` scans all interests every call — fine once, wasteful at scale; build an index instead.
- **Reporting an average over groups with n=1**: averaging salary "by exact tenure" just echoes individual salaries because no two users share a tenure — bucket before averaging.
- **Treating eyeballed thresholds as a model**: `predict_paid_or_unpaid` with hardcoded cutoffs is explicitly called out as not a real model, just illustrative.

## Code Examples
```python
friendships = {user["id"]: [] for user in users}
for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)
```
- **What it demonstrates**: converting an edge-pair list into an adjacency-list dict for O(1) friend lookups.

```python
from collections import Counter

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]
        for foaf_id in friendships[friend_id]
        if foaf_id != user_id
        and foaf_id not in friendships[user_id]
    )
```
- **What it demonstrates**: from-scratch "people you may know" via nested comprehension + `Counter`, excluding self and existing friends.

```python
def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"
```
- **What it demonstrates**: manual bucketing of a continuous variable before aggregation.

## Reference Tables
| Feature request | Technique | Chapter forward-reference |
|---|---|---|
| Key connectors | Degree centrality | Ch22 (Network Analysis) |
| People you may know | FOAF + shared interests | Ch23 (Recommender Systems) |
| Salary vs. tenure | Bucketed averages -> real model | Ch14 (Simple Linear Regression) |
| Paid vs. unpaid prediction | Eyeballed cutoff -> real classifier | Ch16 (Logistic Regression) |
| Topics of interest | Word counting -> real NLP | Ch21 (Natural Language Processing) |

## Worked Example
DataSciencester has 10 users (ids 0-9, e.g., Hero=0, Dunn=1, Chi=3, Thor=4) connected by 12 friendship pairs. Total connections = 24, so average connections = 2.4. Sorting by degree, users 1, 2, 3, 5, 8 tie at 3 friends each; user 9 has only 1. The book flags that Thor (id 4, degree 2) *feels* more central than Dunn (id 1, degree 3) despite lower degree — motivating richer centrality measures later. Salary/tenure data for 10 users, bucketed into "<2 yrs" ($48,000 avg), "2-5 yrs" ($61,500 avg), ">5 yrs" ($79,167 avg), yields the soundbite: data scientists with 5+ years earn ~65% more than those with little/no experience.

## Key Takeaways
1. Restructure raw data (pairs -> adjacency dict, list -> index) before analyzing it; this is often most of the work.
2. Simple metrics (degree centrality, word counts) are cheap wins but have known blind spots that later chapters fix with better theory.
3. `Counter` and `defaultdict` from `collections` are workhorses for from-scratch aggregation.
4. Ad hoc bucketing and eyeballed cutoffs are legitimate first passes but should be replaced by real statistical models once enough data exists.
5. The whole book's running example (DataSciencester) recurs in nearly every later chapter — learn its dataset shapes now.

## Connects To
- **Ch22**: revisits degree centrality with betweenness and eigenvector centrality for the same DataSciencester graph.
- **Ch23**: expands FOAF/shared-interest matching into formal recommender systems (user-based, item-based, matrix factorization).
- **Ch14/Ch16**: replace the eyeballed salary/tenure and paid/unpaid heuristics with simple linear regression and logistic regression.
- **Ch21**: replaces naive word-counting with real NLP (n-grams, topic models).
- **networkx / pandas**: the libraries this book deliberately avoids in favor of from-scratch dicts, lists, and `Counter`.
