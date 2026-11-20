# Chapter 24: Databases and SQL

## Core Idea
To teach SQL "from scratch," the chapter builds NotQuiteABase, a pure-Python relational-table implementation (`Table` class with `insert`, `update`, `delete`, `select`, `where`, `limit`, `group_by`, `order_by`, `join`), mirroring each SQL operation side-by-side with its NotQuiteABase equivalent, then closes with indexes, query optimization, and a nod to NoSQL databases.

## Frameworks Introduced
- **Table abstraction (schema + rows)**: a `Table(columns, types)` enforces a fixed schema (column names + types) on `insert`, storing each row as a `Dict[str, Any]` for implementation simplicity (real databases wouldn't do this, for space reasons).
  - When to use: as a teaching device mapping 1:1 onto SQL's `CREATE TABLE`/`INSERT INTO`.
- **CRUD operations**: `insert` (type/arity-checked append), `update(updates, predicate)` (mutate matching rows), `delete(predicate)` (filter out matching rows) — direct analogues of SQL `INSERT`, `UPDATE ... WHERE`, `DELETE FROM ... WHERE`.
- **select / where / limit**: `select(keep_columns, additional_columns)` projects/computes new columns (reading Python type annotations off calculation functions to infer result schema); `where(predicate)` filters rows; `limit(n)` truncates — composed to build arbitrary `SELECT ... WHERE ... LIMIT` queries.
- **group_by(columns, aggregates, having)**: buckets rows by a tuple key of group-by column values, applies aggregate functions per bucket, optionally filters buckets via a `having` predicate — mirrors SQL's `GROUP BY ... HAVING`.
- **order_by(order_fn)**: sorts a copy of the table by an arbitrary key function — mirrors `ORDER BY` (with sign-flipping needed to emulate `DESC`).
- **join(other_table, left_join=False)**: joins on whatever columns the two tables share in common, optionally performing a left join (unmatched left rows get `None` for right-table columns) — mirrors SQL `JOIN`/`LEFT JOIN`.
- **Subqueries "for free"**: because every NotQuiteABase operation returns a real `Table`, you can chain/compose results exactly like nesting a SQL subquery inside another query.

## Key Concepts
- **Schema**: fixed column names + types a table enforces on every row — the structural contract a relational table makes, contrasted with schemaless NoSQL documents.
- **NULL / None**: SQL's marker for a missing value, directly analogous to Python's `None`.
- **Normalization**: organizing tables to minimize redundancy (e.g., splitting `user_interests` into its own table rather than storing lists inside `users` rows, since SQL columns can't hold lists) — at the cost of needing joins to reassemble related data.
- **INNER JOIN vs. LEFT JOIN**: inner join returns only rows with matches in both tables; left join also includes every left-table row with no match, filling unmatched right-table columns with `NULL`/`None`.
- **Index**: a data structure (not implemented in NotQuiteABase) that lets a database look up rows by key column(s) without scanning every row, and that can enforce uniqueness constraints — essential for performant real-world joins/lookups.
- **Query optimization (filter-before-join)**: filtering a table down before joining it is "almost certainly more efficient" than joining first and filtering after, since the join then has fewer rows to process — in real SQL this is handled automatically by the query planner, but NotQuiteABase makes you choose the order yourself.
- **NoSQL**: nonrelational databases (document stores like MongoDB, column stores, key/value stores, graph databases, time-series databases, etc.) that trade the relational model's structure for other performance/flexibility characteristics.

## Mental Models
- **Every SQL clause has a direct, literal method-chain equivalent**: `SELECT...FROM...WHERE...GROUP BY...HAVING...ORDER BY...LIMIT` map almost one-to-one onto chained `Table` method calls (`.where().group_by().order_by().limit()`), making the declarative SQL mental model concrete and steppable in Python.
- **Type annotations as real, load-bearing metadata**: `select`'s `additional_columns` reads `calculation.__annotations__['return']` to infer new columns' types — explicitly called out as "the counterexample" to Ch2's claim that type annotations "don't actually do anything," because here code genuinely inspects and uses them at runtime.
- **Declarative SQL vs. imperative query-order choices**: in real SQL you declare *what* you want and the query engine chooses *how* to execute it efficiently (using indexes, choosing join order); in NotQuiteABase, you (the programmer) must manually choose an efficient operation order (e.g., filter before join) since there's no query optimizer.
- **Dict-based rows are a deliberate simplicity trade-off**: real databases would never store rows as dicts (wasteful), but doing so here makes the from-scratch implementation vastly easier to reason about and test.

## Anti-patterns
- **Storing list-valued columns directly in a relational table**: SQL tables can't hold lists; the correct approach is a separate one-to-many table (e.g., `user_interests` with one row per user-interest pair) joined back to the parent table.
- **Joining before filtering when filtering would shrink one side significantly**: produces correct but inefficient results — filter first to reduce the number of rows the (expensive, naive) join has to compare.
- **Scanning every row for key lookups without an index**: NotQuiteABase's `where`/`join` do exactly this, and the chapter explicitly flags it as "extremely inefficient" and liable to "take approximately forever" on large tables — real systems use indexes to avoid full scans.
- **Allowing duplicate values in columns meant to be unique (e.g., `user_id`) without an index/constraint**: without a uniqueness index, nothing prevents inserting conflicting duplicate keys.
- **Confusing `UPDATE`/`DELETE` without a `WHERE`/predicate for the targeted, conditional versions**: an unconditional `DELETE FROM users;` (or the NotQuiteABase `delete()` with no predicate) wipes every row — always double-check whether a filter is intended.

## Code Examples
```python
class Table:
    def insert(self, values: list) -> None:
        if len(values) != len(self.types):
            raise ValueError(f"You need to provide {len(self.types)} values")
        for value, typ3 in zip(values, self.types):
            if not isinstance(value, typ3) and value is not None:
                raise TypeError(f"Expected type {typ3} but got {value}")
        self.rows.append(dict(zip(self.columns, values)))
```
- **What it demonstrates**: schema-enforced row insertion mirroring SQL's `INSERT INTO` type checking.

```python
def group_by(self, group_by_columns, aggregates, having=lambda g: True) -> 'Table':
    grouped_rows = defaultdict(list)
    for row in self.rows:
        key = tuple(row[column] for column in group_by_columns)
        grouped_rows[key].append(row)
    result_table = Table(group_by_columns + list(aggregates.keys()), ...)
    for key, rows in grouped_rows.items():
        if having(rows):
            new_row = list(key) + [agg_fn(rows) for agg_fn in aggregates.values()]
            result_table.insert(new_row)
    return result_table
```
- **What it demonstrates**: `GROUP BY ... HAVING` implemented via a tuple-keyed `defaultdict` bucketing rows, then applying aggregate functions per bucket.

```python
def join(self, other_table, left_join=False) -> 'Table':
    join_on_columns = [c for c in self.columns if c in other_table.columns]
    additional_columns = [c for c in other_table.columns if c not in join_on_columns]
    for row in self.rows:
        other_rows = other_table.where(lambda r: all(r[c] == row[c] for c in join_on_columns)).rows
        for other_row in other_rows:
            join_table.insert([row[c] for c in self.columns] + [other_row[c] for c in additional_columns])
        if left_join and not other_rows:
            join_table.insert([row[c] for c in self.columns] + [None for c in additional_columns])
```
- **What it demonstrates**: a nested-loop join on shared columns, with left-join semantics filling unmatched rows with `None`.

## Reference Tables
| SQL clause | NotQuiteABase method |
|---|---|
| `CREATE TABLE` | `Table(columns, types)` |
| `INSERT INTO ... VALUES` | `table.insert(values)` |
| `UPDATE ... SET ... WHERE` | `table.update(updates, predicate)` |
| `DELETE FROM ... WHERE` | `table.delete(predicate)` |
| `SELECT ... FROM` | `table.select(keep_columns, additional_columns)` |
| `WHERE` | `table.where(predicate)` |
| `LIMIT` | `table.limit(n)` |
| `GROUP BY ... HAVING` | `table.group_by(columns, aggregates, having)` |
| `ORDER BY` | `table.order_by(order_fn)` |
| `JOIN` / `LEFT JOIN` | `table.join(other, left_join=False/True)` |
| subquery | any chained Table result (free, since results are real Tables) |

## Worked Example
A `users` table (`user_id, name, num_friends`) and a `user_interests` table (`user_id, interest`) are built and queried step by step: `users.where(lambda row: row["name"] == "Dunn").select(keep_columns=["user_id"])` replicates `SELECT user_id FROM users WHERE name = 'Dunn'`. Grouping by computed `name_length` and aggregating `min_user_id`/`num_users` replicates a `GROUP BY LENGTH(name)` query. Joining `users` with `user_interests` and filtering for `interest == "SQL"` replicates `SELECT users.name FROM users JOIN user_interests ON ... WHERE interest = 'SQL'`; using `left_join=True` plus a `count_interests` aggregate replicates counting each user's interests (including users with zero interests, via `LEFT JOIN`). The query-optimization section shows filtering `user_interests` for `"SQL"` *before* joining to `users` is more efficient than joining first and filtering after, since filter-then-join operates on far fewer rows.

## Key Takeaways
1. Every core SQL clause (`SELECT/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT/JOIN`) has a direct, composable method-chain equivalent in NotQuiteABase — understanding one deepens understanding of the other.
2. Normalize relational data (split one-to-many relationships into separate tables) since SQL columns can't hold lists; recombine with joins.
3. INNER JOIN keeps only matching rows; LEFT JOIN additionally preserves unmatched left-table rows with NULLs for the right side — choose based on whether you need to count/include "zero match" cases.
4. Filter before joining when possible — it's a basic, generally-applicable query-optimization principle, even though real SQL engines often handle this automatically via their query planner.
5. Indexes (not implemented here) are what make real databases' lookups and joins fast and let them enforce uniqueness constraints — understand their purpose even without implementing one from scratch.
6. NoSQL databases (document, column, key-value, graph, time-series stores) trade the relational model's structure for different performance/flexibility trade-offs — know they exist as alternatives, not that any one is "better."

## Connects To
- **Ch2**: the `select` method's use of function return-type annotations is explicitly called out as the chapter's "counterexample" to Ch2's claim that type annotations don't do anything at runtime.
- **Ch9/Ch10**: the kind of structured, cleaned data this chapter stores/queries is exactly what earlier chapters acquired and cleaned from files/APIs/scraping.
- **SQLite / MySQL / PostgreSQL / MongoDB**: external real-world relational and NoSQL databases the chapter points readers toward for hands-on practice.
