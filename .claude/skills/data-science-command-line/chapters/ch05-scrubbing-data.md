# Chapter 5: Scrubbing Data

## Core Idea
Scrubbing is the (often 80%-of-effort) "cookbook" step of OSEMN where format conversion, line/row filtering, value extraction/replacement, column manipulation, and file combination turn obtained data into a clean, tabular (usually CSV) form that other tools can consume — and it recurs at every point in a pipeline, not just right after obtaining data.

## Frameworks Introduced
- **"Transformations everywhere" principle**: scrubbing isn't confined to step 2 of OSEMN — any time one tool's output doesn't match the next tool's expected input, an intermediate transformation (scrub) is needed.
  - When to use: whenever chaining two tools whose formats don't line up (e.g., `uniq -c` output → CSV for plotting).
  - How: insert a small tool (`awk`, `sed`, `header`) between the two "key" tools to reshape the data; think of pipeline-building as puzzle-solving where key pieces need helper pieces to fit.
- **CSV-awareness workaround via `body`/`header`/`cols`**: classic text tools (`grep`, `tr`, `sort`) have no notion of CSV header/columns, so wrap them.
  - When to use: you need to apply a plain-text tool to a CSV's body only, or to just one column, without breaking the header.
  - How: `body <cmd>` applies `<cmd>` to everything but line 1 (the header), re-prints header automatically; `header` adds/deletes/replaces/applies-to-just the header (`-a`, `-d`, `-r`, `-e`); `cols -c <col> <cmd>` applies `<cmd>` only to named column(s), typically composed with `body`.
- **Format-conversion-first strategy for XML/HTML/JSON**: convert nested/structured formats to CSV as early as possible rather than writing bespoke parsing logic repeatedly.
  - When to use: data arrives as HTML tables, XML, or JSON but downstream tools (visualization, modeling) need tabular CSV.
  - How: `curl` → extract structural subset (`pup` CSS selector or `grep -A`) → `xml2json` (HTML/XML → JSON) → `jq` (reshape/project to flat records, `@csv`) → `header -a` (label columns) → CSV.
- **SQL-as-an-alternative-tool strategy (`csvsql`)**: when ad hoc tool combinations get too complex/inflexible, drop to SQL directly on CSV.
  - When to use: complex WHERE clauses, joins, or robust column-name-based logic that awk/sed one-liners can't match (e.g., values containing delimiters).
  - How: `csvsql --query "SELECT ..." file.csv` (or `FROM stdin` when piped); uses SQLite dialect; multiple CSVs can be named as tables for joins.

## Key Concepts
- **Filtering by location**: `head -n`, `tail -n`/`-n +K`, `sed -n '<range>p'`, `awk 'NR<=N'` — all can select line ranges; `sed`/`awk` can select by modulo (odd/even lines: `sed -n '1~2p'`, `awk 'NR%2'`).
- **Filtering by pattern**: `grep` (`-i` case-insensitive, `-E` extended regex, `-v` invert, `-o` print-only-match).
- **Filtering by randomness**: `sample -r 1%` (probabilistic sampling), with `-d` (delay) and `-s` (duration) for debugging live/streaming pipelines; put `sample`/`head`/`tail` as early as possible in a pipeline to reduce unnecessary computation.
- **tr**: translates/deletes individual (single-byte) characters; `-d` deletes, `-c` complements a character class; fails on non-ASCII/multi-byte characters — use `sed` with `\U&` for those.
- **sed**: stream editor; `s/pattern/replacement/flags` for substitution (`g`=global), supports regex back-references (`\1`, `\2`) for restructuring text, e.g. merging/reordering fields.
- **awk**: pattern-scanning language; `NR` = current record number, `$1`/`$2`/... = fields, `-F` sets field separator, `BEGIN{}` block runs once before processing — powerful for CSV-like column logic without a true CSV parser.
- **cut**: splits lines into fields by delimiter (`-d`, `-f`) or character position (`-c`); does NOT reorder fields — always preserves original order (unlike `csvcut`).
- **csvkit CSV-aware tools**: `csvcut` (-c include / -C complement, reorders), `csvgrep` (-c column, -r regex, -i invert), `csvsql` (SQL on CSV, SQLite dialect), `csvjoin` (SQL-like join on a key column), `csvlook` (pretty table).
- **Horizontal concatenation**: `paste -d, file1.csv file2.csv file3.csv` glues same-row-count CSVs side by side (rows must already align).
- **Joining**: `csvjoin -c <key> file1.csv file2.csv` or `csvsql` with an explicit `JOIN ... ON` — needed when data is normalized across multiple files/tables.
- **pup**: CSS-selector-based HTML extractor (e.g., `pup 'table.wikitable tbody'`).
- **xml2json**: converts XML/HTML to JSON so `jq` can operate on it.

## Mental Models
- Treat CSV handling as needing CSV-aware tools (csvkit, `body`/`header`/`cols`) rather than naive text tools, because plain-text tools don't understand headers/columns/quoting (e.g., commas embedded in quoted fields break naive `sed`/`awk`/`cut` approaches — only `csvsql`/proper CSV parsers handle that correctly).
- When multiple approaches solve the same problem (sed vs awk vs cols+tr vs csvsql), prefer the most robust one when data has edge cases (e.g., embedded commas) — SQL-based `csvsql` tends to win on robustness; simpler text tools win on brevity for clean data.
- "Convert to CSV as quickly as possible" — don't become a specialist in XML/JSON tools; normalize to tabular form fast and spend your learning budget on generic tools instead.

## Anti-patterns
- **Using `tr` on non-ASCII/multi-byte text**: silently mishandles accented/non-Latin characters since `tr` is single-byte — use `sed 's/.../\U&/g'` instead.
- **Applying grep/tr/sort directly to a CSV expecting header-awareness**: these tools don't know about headers and will filter/transform the header row along with the data — wrap with `body`/`header`/`cols`.
- **Assuming `cut -f 5,1,3` reorders columns**: it doesn't; `cut` always outputs fields in their original order — use `csvcut -c` for true reordering.
- **Naive sed/awk field-splitting on CSVs with embedded commas/quotes**: breaks on values like `"Beethoven, van"` — demonstrated explicitly; only `csvsql` (or a real CSV parser) handles quoted-comma values correctly.
- **Exporting from a database to CSV before running SQL on it, when you could query the database directly**: slower, and column types may not be inferred correctly from CSV — query the live database first when possible.

## Code Examples
```bash
$ seq 100 | /data/ch04/fizzbuzz.py | tee fb.seq | grep -E "fizz|buzz" | sort | uniq -c | sort -nr > fb.cnt
$ < fb.cnt awk 'BEGIN { print "value,count" } { print $2","$1 }' > fb.csv
```
- **What it demonstrates**: an intermediate `awk` transformation reshapes `uniq -c` output (count-then-word) into proper CSV (value,count header + reordered fields) so a plotting tool can consume it — the "transformations everywhere" pattern.

```bash
$ seq 5 | header -a count | body wc -l
```
- **What it demonstrates**: `header -a` adds a column name, `body` applies `wc -l` only to the data rows while preserving/reprinting the header untouched.

```bash
$ < tips.csv cols -c day body "tr '[a-z]' '[A-Z]'" | head -n 5 | csvlook
```
- **What it demonstrates**: combining `cols` + `body` + `tr` to uppercase only one column's values without touching the header or other columns.

```bash
$ csvsql --query "SELECT id, first_name || ' ' || last_name AS full_name, born FROM stdin" < names-comma.csv
```
- **What it demonstrates**: `csvsql` correctly handles a last name containing an embedded comma (`"Beethoven, van"`) where sed/awk/cols-based approaches all fail.

```bash
$ < wiki.html pup 'table.wikitable tbody' | xml2json > table.json
$ < table.json jq -r '.tbody.tr[1:][] | [.td[]["$t"]] | @csv' | header -a rank,country,border,surface,ratio > countries.csv
```
- **What it demonstrates**: the full HTML-table-to-CSV pipeline: CSS-select the table body, convert to JSON, reshape/flatten with `jq`, and label columns with `header -a`.

## Reference Tables
| Task | Plain-text tool | CSV-aware tool |
|---|---|---|
| Filter lines by location | `head`/`tail`/`sed`/`awk` | `body` + same, to skip header |
| Filter by pattern | `grep` | `csvgrep -c <col> -r <regex>` |
| Extract/reorder columns | `cut` (no reorder) | `csvcut -c`/`-C` (reorders) |
| Replace/delete chars | `tr`, `sed` | `cols -c <col> tr/sed ...` |
| Row filter on column value | `awk -F,` | `csvgrep`, `csvsql WHERE` |
| Merge columns | `sed`/`awk` back-refs | `csvsql` (`||` concat), `rush unite()` |
| Combine files | `paste -d,` (horizontal), `cat` (vertical) | `csvjoin -c <key>`, `csvsql JOIN` |

| Format conversion | Tool chain |
|---|---|
| HTML/XML table → tabular | `pup` (CSS select) → `xml2json` → `jq` → CSV |
| JSON → CSV | `jq -r '... | @csv'` + `header -a` |

## Worked Example
Full Wikipedia-table-to-CSV pipeline:
```bash
$ curl -sL 'http://en.wikipedia.org/wiki/List_of_countries_and_territories_by_border/area_ratio' > wiki.html
$ grep wikitable -A 21 wiki.html                       # inspect structure first
$ < wiki.html pup 'table.wikitable tbody' | tee table.html > /dev/null   # CSS-select the table body
$ < table.html xml2json > table.json                   # HTML -> JSON
$ < table.json jq -r '.tbody.tr[1:][] | [.td[]["$t"]] | @csv' | \
    header -a rank,country,border,surface,ratio > countries.csv   # flatten + label
$ csvlook --max-column-width 28 countries.csv           # verify result
```
This reconstructs a tabular dataset (rank, country, border length, area, ratio) directly from a live Wikipedia page using only command-line tools — obtain (curl) fused with scrub (pup/xml2json/jq/header) end to end.

## Key Takeaways
1. No single tool cleans all messiness — expect to combine several tools (grep+sed+awk+jq+csvkit) per scrubbing task.
2. Classic text tools (`cut`, `grep`, `tr`, `sort`) have no concept of CSV headers/columns — use `body`/`header`/`cols` to make them CSV-safe, or switch to csvkit tools (`csvcut`, `csvgrep`) outright.
3. `csvsql` is the fallback for complex, robust, or edge-case-safe (embedded delimiters) operations, at the cost of verbosity.
4. Convert nested formats (HTML/XML/JSON) to flat CSV as early as possible via `pup`/`xml2json`/`jq`, rather than writing format-specific logic repeatedly.
5. Put data-reducing tools (`sample`, `head`, `grep`) as early as possible in a pipeline to save computation, then remove them once the pipeline is validated.
6. When multiple tool combinations solve the same problem, pick based on correctness on your actual data (e.g., quoted commas) over brevity.

## Connects To
- **Ch 3 (Obtaining Data)**: this chapter continues directly on data obtained via curl/APIs/spreadsheets.
- **Ch 7 (Exploring Data)**: scrubbed CSV output here (e.g., `fb.csv`) feeds directly into `rush` visualizations.
- **Ch 4**: custom tools like `fizzbuzz.py` and the `body`/`header`/`cols` dsutils are examples of the reusable-tool pattern from Ch 4.
- **csvkit, jq, pup, xml2json**: the core external tool ecosystem for this chapter.
- **Appendix**: many tools here (awk, sed, tr, csvcut, csvsql, csvjoin, jq, pup) are catalogued there.
