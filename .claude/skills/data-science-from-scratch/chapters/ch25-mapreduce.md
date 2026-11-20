# Chapter 25: MapReduce

## Core Idea
MapReduce is a programming model for parallel processing across many machines: a mapper turns each input item into zero or more key/value pairs, identical keys are grouped together, and a reducer collapses each group into output values — the chapter builds a generic single-machine `map_reduce` function and applies it to word counting, status-update analytics, and distributed sparse matrix multiplication.

## Frameworks Introduced
- **MapReduce model**: (1) `mapper(item) -> Iterable[(key, value)]`, (2) group all pairs by key, (3) `reducer(key, values) -> (key, output)` per group.
  - When to use: processing datasets too large for one machine, or any aggregation that can be expressed as "transform each item into key/value pairs, then aggregate by key" — scales horizontally by distributing mappers and reducers across machines.
  - How: `map_reduce(inputs, mapper, reducer)` runs the whole pipeline on a single machine using a `defaultdict(list)` collector, as a from-scratch stand-in for a distributed MapReduce system.
- **values_reducer factory**: `values_reducer(values_fn)` wraps any aggregation function (`sum`, `max`, `min`, `len(set(...))`) into a `Reducer`, since most reducers are just "apply an aggregate function to the collected values."
  - When to use: whenever a reducer is a simple aggregation, to avoid writing near-identical reducer boilerplate repeatedly.
- **Combiners**: a mapper-side pre-aggregation step (e.g., combine 500 local `("data", 1)` pairs into one `("data", 500)`) that reduces the volume of data shipped from mapper machines to reducer machines in a real distributed system.
  - When to use: distributed MapReduce specifically, to cut network/transfer costs — requires writing the reducer so it handles pre-combined inputs correctly (e.g., using `sum` rather than `len`).
- **MapReduce matrix multiplication**: represent sparse matrices as `Entry(name, i, j, value)` tuples (only nonzero entries); map each `A[i][j]` to every `C[i][y]` key and each `B[i][j]` to every `C[x][j]` key with the paired index as value; reduce by multiplying matched `(A-value, B-value)` pairs sharing an index and summing.
  - When to use: multiplying very large, sparse matrices across a distributed cluster, where a dense list-of-lists representation would be prohibitively large.

## Key Concepts
- **Key/value pair**: the universal currency MapReduce operates on; a mapper's whole job is choosing the right key to group by for the problem at hand.
- **Mapper**: `Callable[..., Iterable[KV]]` — turns one input item into zero or more key/value pairs.
- **Reducer**: `Callable[[key, Iterable[values]], KV]` — collapses all values sharing a key into one output.
- **Horizontal scaling**: doubling the number of machines should (ignoring fixed overhead) roughly halve runtime, since mapper work and (with enough distinct keys) reducer work both split across more machines — the core value proposition of MapReduce.
- **Sparse matrix representation**: storing only nonzero `(i, j, value)` entries instead of a full dense list-of-lists, essential when matrices are enormous but mostly zero (e.g., a billion x billion matrix).
- **"Moving processing to the data"**: MapReduce's key architectural idea — rather than shipping all data to one processing machine, ship small mapper/reducer *functions* to where data already lives, distributed across many machines.

## Mental Models
- **Choosing the mapper's key is the central design decision**: for "most popular word per user," the chapter walks through three candidate key choices (username; word; username+word) and reasons through which actually supports the needed aggregation (grouping by username, so each user's full word list is visible to one reducer call) — key choice is where the real algorithm design happens.
- **MapReduce generalizes far beyond word count**: the same `map_reduce` function solves word counting, "which day of week mentions data science most," "most popular word per user," "distinct likers per user," and distributed matrix multiplication — only the mapper/reducer pair changes.
- **Combiners trade reducer-code complexity for network efficiency**: pre-aggregating on the mapper side only works correctly if the reducer is written compatibly (e.g., `sum` tolerates pre-summed partial counts; `len` of raw items would not) — a subtle correctness constraint introduced by the optimization.
- **MapReduce as a historical/pedagogical topic, not a current recommendation**: the author explicitly flags that MapReduce "feels a lot less popular now" and nearly cut the chapter, pointing to Spark as the more modern choice for latency-sensitive workloads — treat this chapter as understanding a foundational model, not a tool to reach for by default today.

## Anti-patterns
- **Choosing the wrong grouping key for a multi-step aggregation**: e.g., grouping by word instead of by username when you need each user's full word distribution visible together — breaks the ability to compute a per-user "most popular word."
- **Writing a combiner-incompatible reducer**: if a reducer relies on `len(values)` (counting raw items) rather than `sum(values)` (summing partial counts), introducing mapper-side combining silently produces wrong results.
- **Using a dense list-of-lists representation for very large sparse matrices**: wastes enormous amounts of memory/storage; use a sparse `Entry(name, i, j, value)` representation instead when most entries are zero.
- **Treating MapReduce as the default tool for "big data" problems today**: the chapter's own framing suggests this is now often overkill or outdated versus modern alternatives (Spark) for many workloads, especially latency-sensitive ones.

## Code Examples
```python
def map_reduce(inputs, mapper, reducer) -> List[KV]:
    collector = defaultdict(list)
    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)
    return [output for key, values in collector.items() for output in reducer(key, values)]

def wc_mapper(document): 
    for word in tokenize(document):
        yield (word, 1)

def wc_reducer(word, counts):
    yield (word, sum(counts))
```
- **What it demonstrates**: the fully generic single-machine MapReduce engine, specialized to word counting.

```python
def words_per_user_mapper(status_update):
    user = status_update["username"]
    for word in tokenize(status_update["text"]):
        yield (user, (word, 1))

def most_popular_word_reducer(user, words_and_counts):
    word_counts = Counter()
    for word, count in words_and_counts:
        word_counts[word] += count
    word, count = word_counts.most_common(1)[0]
    yield (user, (word, count))
```
- **What it demonstrates**: a key-choice decision (group by `user`, carry `(word, count)` as the value) that enables finding each user's single most popular word.

```python
def matrix_multiply_mapper(num_rows_a, num_cols_b):
    def mapper(entry: Entry):
        if entry.name == "A":
            for y in range(num_cols_b):
                yield ((entry.i, y), (entry.j, entry.value))
        else:
            for x in range(num_rows_a):
                yield ((x, entry.j), (entry.i, entry.value))
    return mapper
```
- **What it demonstrates**: mapping each sparse matrix entry to every output cell of `C` it contributes to, keyed by that output cell's coordinates.

## Reference Tables
| Problem | Mapper key | Mapper value | Reducer |
|---|---|---|---|
| Word count | word | 1 | sum |
| Data-science mentions by weekday | day_of_week | 1 | sum |
| Most popular word per user | username | (word, 1) | tally, take most common |
| Distinct likers per user | username | liker | count distinct |
| Matrix multiply | (i, j) of output C | (shared index, value) | multiply matched pairs, sum |

## Worked Example
Three documents `["data science", "big data", "science fiction"]` mapped via `wc_mapper` yield pairs grouped into `{"data": [1,1], "science": [1,1], "big": [1], "fiction": [1]}`; `wc_reducer` collapses these into `[("data", 2), ("science", 2), ("big", 1), ("fiction", 1)]`. For matrix multiplication, `A = [[3,2,0],[0,0,0]]` and `B = [[4,-1,0],[10,0,0],[0,0,0]]` are represented as 5 sparse `Entry` tuples; running `map_reduce` with the matrix-multiply mapper/reducer correctly produces the two nonzero entries of the product matrix: `{((0,0), 32), ((0,1), -3)}`, matching the expected dense product `[[32,-3,0],[0,0,0]]`.

## Key Takeaways
1. MapReduce's power comes from moving small processing functions to distributed data rather than moving all data to one machine — enabling (near-)linear horizontal scaling.
2. The mapper's choice of key is the central design decision for any MapReduce algorithm — it determines what information ends up grouped together for the reducer to see.
3. A generic `map_reduce(inputs, mapper, reducer)` function, with reducers often reducible to a simple `values_reducer(aggregate_fn)`, can solve a wide range of aggregation problems by swapping only the mapper/reducer pair.
4. Combiners reduce network transfer in distributed settings but require the reducer to tolerate pre-aggregated partial values.
5. Sparse matrix representations (`Entry(name, i, j, value)`) are essential at scale, and MapReduce can perform distributed matrix multiplication by routing each entry to every output cell it contributes to.
6. MapReduce is now considered somewhat dated for latency-sensitive workloads (Spark is the modern alternative) but remains a useful model for understanding distributed batch aggregation.

## Connects To
- **Ch22's Matrix Multiplication**: directly reused and reframed for distributed, sparse computation via MapReduce.
- **Ch1 / Ch21**: the word-counting and status-update analytics examples echo the simple aggregation patterns (Counter-based word counts) seen throughout the book, now reframed for scale.
- **Hadoop / Amazon Elastic MapReduce / Spark**: external systems the chapter points to for real distributed MapReduce execution, with Spark flagged as the more modern choice for low-latency needs.
