# Chapter 22: Network Analysis

## Core Idea
Networks (nodes + edges, undirected or directed) require richer centrality measures than simple degree counting; the chapter builds betweenness and closeness centrality (via from-scratch breadth-first shortest paths), eigenvector centrality (via matrix multiplication and power iteration), and PageRank (for directed endorsement networks) to find a graph's truly "important" nodes.

## Frameworks Introduced
- **Betweenness centrality**: for node `i`, sum over every pair `(j,k)` the fraction of shortest `j-k` paths that pass through `i`.
  - When to use: identifying nodes that act as "bridges" or brokers between other parts of the network, not just nodes with many direct connections.
  - How: `shortest_paths_from` (breadth-first search using a `deque` frontier) computes all shortest paths from every node; betweenness sums `1/num_paths` contributions for each intermediate node on each shortest path.
- **Closeness centrality**: `1 / farness(node)`, where farness is the sum of shortest-path lengths from that node to every other node.
  - When to use: measuring how "close" (in hops) a node is to the rest of the network overall.
- **Eigenvector centrality**: represent the network as an adjacency matrix; a node's centrality is its entry in the matrix's dominant eigenvector, found via power iteration (`find_eigenvector`: repeatedly multiply, rescale to unit magnitude, check convergence).
  - When to use: large networks where betweenness/closeness (requiring full shortest-path computation) are too expensive; captures "connected to people who are themselves well-connected" recursively.
- **PageRank**: for directed (endorsement-style) networks, distribute a fixed total "rank" (1.0) across nodes, where each node passes a damped fraction of its rank evenly along its outgoing edges each iteration, plus a small uniform baseline redistributed to all nodes.
  - When to use: ranking nodes in a directed graph (endorsements, web links) resistant to simple gaming (e.g., mutual endorsement rings) that a raw in-degree count would fall for.

## Key Concepts
- **Undirected vs. directed edges**: undirected (mutual, e.g., Facebook friendship) vs. directed (one-way, e.g., web hyperlinks, endorsements) — determines which centrality measures and algorithms apply.
- **Shortest path**: fewest-edges path between two nodes; multiple shortest paths of equal length can exist and all must be tracked for correct betweenness computation.
- **Frontier (BFS queue)**: a `collections.deque` of `(prev_user, user)` pairs to explore, processed FIFO so nodes are discovered in order of increasing distance from the source.
- **Eigenvector / eigenvalue**: for matrix `A` and nonzero vector `v`, if `Av = scalar * v`, `v` is an eigenvector with eigenvalue `scalar` — eigenvector centrality uses the adjacency matrix's dominant eigenvector.
- **Power iteration**: the repeated "multiply, rescale, check convergence" procedure for finding an eigenvector; not guaranteed to terminate for all matrices (e.g., pure rotation matrices have no real eigenvector; some matrices cycle rather than converge).
- **Damping factor**: PageRank's parameter (typically 0.85) controlling what fraction of a node's rank flows through its outgoing edges each iteration, vs. the uniformly-redistributed remainder.

## Mental Models
- **Centrality numbers are relative, not absolute**: "generally the centrality numbers aren't that meaningful themselves. What we care about is how the numbers for each node compare to the numbers for other nodes" — always interpret centrality comparatively within a given network.
- **Eigenvector centrality is circularly defined, and that's the point**: "your centrality is a constant multiple of the sum of your neighbors' centrality" sounds circular, but the eigenvector equation is precisely the mathematical device that resolves such a circular, mutually-referential definition into concrete numbers.
- **Betweenness/closeness are intuitive but computationally expensive; eigenvector centrality is less intuitive but computationally cheap (with a real matrix library)**: a direct trade-off explaining why large networks favor eigenvector-style (and PageRank-style) measures over shortest-path-based ones.
- **Raw in-degree (endorsement count) is trivially gameable**: a ring of mutual endorsers can inflate their own counts; PageRank defeats this by weighting each endorsement by the endorser's own rank, not just counting endorsements.

## Anti-patterns
- **Using degree centrality alone to find "key" network nodes**: as Ch1 already found, a low-degree node (Thor) can intuitively seem more central than a higher-degree node (Dunn) because degree ignores *where* in the network the connections sit — richer metrics (betweenness, eigenvector) are needed.
- **Trusting raw endorsement/follower counts as a quality signal**: trivially gamed via phony accounts or mutual-endorsement pacts — exactly the scenario users 0, 1, 2 exploit in the chapter's example, which PageRank correctly discounts relative to Thor's fewer-but-higher-quality endorsements.
- **Assuming `find_eigenvector`'s power iteration always converges**: a pure rotation matrix has no real eigenvector (infinite loop); some matrices (e.g., a coordinate-flip matrix) can cycle indefinitely for a non-eigenvector starting guess rather than converge — not every matrix is well-behaved for this method.
- **Computing full shortest-path-based centrality on very large graphs**: betweenness/closeness centrality are explicitly flagged as impractical at scale due to the cost of computing all-pairs shortest paths.

## Code Examples
```python
def shortest_paths_from(from_user_id, friendships):
    shortest_paths_to = {from_user_id: [[]]}
    frontier = deque((from_user_id, friend_id) for friend_id in friendships[from_user_id])
    while frontier:
        prev_user_id, user_id = frontier.popleft()
        new_paths_to_user = [path + [user_id] for path in shortest_paths_to[prev_user_id]]
        old_paths_to_user = shortest_paths_to.get(user_id, [])
        min_path_length = len(old_paths_to_user[0]) if old_paths_to_user else float('inf')
        new_paths_to_user = [p for p in new_paths_to_user
                              if len(p) <= min_path_length and p not in old_paths_to_user]
        shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user
        frontier.extend((user_id, f) for f in friendships[user_id] if f not in shortest_paths_to)
    return shortest_paths_to
```
- **What it demonstrates**: breadth-first search tracking *all* (possibly multiple) shortest paths to every reachable node, the foundation for betweenness/closeness centrality.

```python
def find_eigenvector(m: Matrix, tolerance=0.00001):
    guess = [random.random() for _ in m]
    while True:
        result = matrix_times_vector(m, guess)
        norm = magnitude(result)
        next_guess = [x / norm for x in result]
        if distance(guess, next_guess) < tolerance:
            return next_guess, norm
        guess = next_guess
```
- **What it demonstrates**: power iteration for finding a dominant eigenvector — the core eigenvector-centrality algorithm.

```python
def page_rank(users, endorsements, damping=0.85, num_iters=100):
    outgoing_counts = Counter(target for source, target in endorsements)
    pr = {u.id: 1 / len(users) for u in users}
    base_pr = (1 - damping) / len(users)
    for _ in range(num_iters):
        next_pr = {u.id: base_pr for u in users}
        for source, target in endorsements:
            next_pr[target] += damping * pr[source] / outgoing_counts[source]
        pr = next_pr
    return pr
```
- **What it demonstrates**: PageRank's iterative rank-distribution update — damped flow along outgoing edges plus a uniform baseline.

## Reference Tables
| Centrality measure | Based on | Computational cost | Gameable? |
|---|---|---|---|
| Degree | direct connection count | trivial | yes (easy to inflate) |
| Betweenness | fraction of shortest paths through node | expensive (all-pairs shortest paths) | harder |
| Closeness | inverse of total shortest-path distance | expensive (same) | harder |
| Eigenvector | dominant eigenvector of adjacency matrix | cheap with a real linear algebra library | moderate |
| PageRank | damped iterative rank propagation (directed) | cheap, converges iteratively | resistant (weights by endorser's own rank) |

## Worked Example
On the 10-user DataSciencester friendship network, betweenness centrality finds users 0 and 9 have centrality 0 (never on anyone's shortest path) while users 3, 4, 5 have high centrality (they bridge otherwise-separate parts of the network) — resolving Ch1's "Thor feels more central than Dunn despite lower degree" puzzle. Closeness centrality shows much less variation across nodes. Eigenvector centrality (via power iteration on the adjacency matrix) finds users 1 and 2 most central, since each has three connections to other highly-central people. For the directed "endorsement" network (15 endorsement edges among 10 users), raw endorsement counts favor users 0, 1, 2 (who endorse each other in a tight loop), but PageRank (damping=0.85) correctly identifies user 4 (Thor) as highest-ranked despite having only two endorsements, because both of his endorsers endorse only him, concentrating their full rank onto him undiluted.

## Key Takeaways
1. Different centrality measures answer different questions: degree (direct connections), betweenness (bridging/brokering), closeness (overall proximity), eigenvector/PageRank (connected to the well-connected).
2. Betweenness and closeness require expensive all-pairs shortest-path computation (via BFS), making them impractical for very large networks.
3. Eigenvector centrality and PageRank scale much better (especially with a real linear algebra library) and are the basis of production ranking systems like Google's original PageRank.
4. PageRank's damped, rank-weighted propagation resists naive gaming (mutual endorsement rings) in a way raw degree/endorsement counts cannot.
5. Power iteration for eigenvectors isn't universally guaranteed to converge — some matrices (rotations, certain cycles) break the simple from-scratch approach.

## Connects To
- **Ch1**: directly revisits and resolves the "degree centrality doesn't match intuition" problem (Thor vs. Dunn) first raised there.
- **Ch4**: `matrix_times_vector`, `matrix_times_matrix`, `make_matrix`, `magnitude`, `distance` are the linear algebra primitives eigenvector centrality is built on.
- **Ch23 (Recommender Systems)**: shares the "connections imply relevance" intuition, applied to user-item rather than user-user graphs.
- **NetworkX / Gephi**: external Python library and GUI tool for production-grade network analysis and visualization.
