# Cheatsheet

## Decision: Which OSEMN step am I in, and how command-line-friendly is it?
- Obtain → highest applicability → `curl`, `in2csv`, `sql2csv`, `jq`
- Scrub → high applicability → `grep`/`sed`/`awk`/`tr` + `body`/`header`/`cols` + csvkit
- Explore → moderate-high → `csvstat`, `rush plot`
- Model → moderate (invoke existing tools, don't implement algorithms) → `tapkee`, `vw`, `skll`
- Interpret → minimal/none → human judgment, no dedicated tool

## Decision tree: scrubbing a data-cleaning task
1. Is the data CSV already? → No: convert first (Excel → `in2csv`; DB → `sql2csv`; HTML/XML → `pup`+`xml2json`+`jq`; JSON → `jq`).
2. Need to filter lines/rows?
   - By position → `head`/`tail`/`sed -n`/`awk 'NR...'` (plain text) or `body` + same (CSV, keeps header)
   - By pattern → `grep` (plain text) or `csvgrep -c <col> -r <regex>` (CSV)
   - By sample/probability → `sample -r <pct>`
3. Need to extract/reorder/merge columns?
   - Extract/reorder → `csvcut -c`/`-C` (reorders) — NOT `cut` (never reorders)
   - Apply a command to one column only → `cols -c <col> <cmd>` (usually with `body`)
   - Merge/concat two columns → `awk`/`sed` back-refs for simple cases; `csvsql` (`||`) when values may contain the delimiter (e.g. embedded commas)
4. Need complex logic / robustness to embedded delimiters? → `csvsql --query "..."` (SQLite dialect)
5. Need to combine files? → same columns, same row order → `paste -d,`; related by a key → `csvjoin -c <key>` or `csvsql JOIN`

## Decision: tr vs. sed for character-level replace
- ASCII-only, single characters → `tr` (fast, simple)
- Non-ASCII / multi-byte / word-level → `sed 's/.../\U&/g'` or full regex

## Decision: loop construct
- Known numeric range → `for i in {start..end..step}; do ...; done`
- Unknown-length lines (file/stdin) → `while read line; do ...; done < file`
- Files (glob) → `for f in /path/*; do ...; done`; recursive/attribute search → `find`
- Repeated many times, want concurrency/control → `parallel` instead of any of the above

## Decision: parallel --jobs value
| Situation | Value |
|---|---|
| Rate-limited API, single-connection resource | `-j1` (serial) |
| Match CPU cores exactly | `-j100%` or omit (default) |
| Leave some cores free | `-j-2` (cores minus 2) |
| Add beyond cores (I/O-bound work) | `-j+4` |
| No idea / testing | never `-j0` in production; use `--dryrun` first |

## Decision: which modeling tool
| Label type | Tool |
|---|---|
| None (unsupervised) | `tapkee` (`--method pca` linear / `--method t-sne` nonlinear) |
| Continuous (regression) | `vw` |
| Categorical (classification) | `skll` / `run_experiment` |

## Thresholds & defaults
- Train/test split: 80% train / 20% test (shuffle with `shuf` first)
- `parallel` default jobs: 100% of CPU cores
- `csvstat --unique` cardinality heuristic: count ≈ row count → identifier/continuous; count ≪ row count → categorical; count == 1 → possibly droppable
- MAE "good enough": context-dependent — always compare against a baseline (e.g., default-hyperparameter run)
- Makefile rule indentation: must be a literal TAB, never spaces

## Tells and smells
- `missing separator (did you mean TAB instead of 8 spaces?)` → soft-tab indentation in a Makefile rule
- `cat: 404.txt: No such file or directory` printed mixed into otherwise-fine output → stderr not redirected (`2> /dev/null` to suppress)
- A file you `>`-redirected into ends up empty → read-and-write-same-file-in-one-pipeline trap; use `sponge` or write-then-`mv`
- `grep '^CHAPTER (.*)\. The'` matches nothing → forgot `-E` for extended regex (pattern treated as literal string)
- `cut -f 5,1,3` doesn't reorder as expected → `cut` always preserves original column order; use `csvcut -c` to truly reorder
- Values with embedded commas scrambled after a sed/awk column-merge → naive text tools don't respect CSV quoting; use `csvsql`
- "`parallel: Warning: Could not figure out number of cpus`" → `parallel` not installed on a remote host; defaults to 1 core there — set `--jobs` explicitly or install remotely
- Suspiciously high classifier accuracy with imbalanced classes → check you balanced the dataset before training, not just split it
- `wc -l file.csv` doesn't match expected row count → `wc -l` counts newlines (includes header, mis-handles embedded newlines in quoted fields) — use `csvstat` row count or `xsv count` instead
- A `rush plot --geom path` chart looks like scribbled connected noise → rows are independent observations; switch to `point`/`smooth`

## Quick reference: redirection & streams
| Syntax | Meaning |
|---|---|
| `cmd > file` | stdout → file (overwrite) |
| `cmd >> file` | stdout → file (append) |
| `cmd 2> file` | stderr → file |
| `cmd 2>/dev/null` | discard stderr |
| `cmd1 \| cmd2` | stdout of cmd1 → stdin of cmd2 |
| `< file cmd` | file → stdin of cmd |
| `cmd \| tee file \| cmd2` | save intermediate output while continuing pipeline |
