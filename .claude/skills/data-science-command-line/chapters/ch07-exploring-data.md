# Chapter 7: Exploring Data

## Core Idea
Exploring data (OSEMN step 3) means inspecting raw data/properties, computing descriptive statistics, and creating visualizations — and the command line, via `csvkit` and `rush` (R one-liners from the shell), lets you do all three without leaving the terminal or writing boilerplate R/Python scripts.

## Frameworks Introduced
- **Three Perspectives on Exploring Data**: inspect → describe → visualize.
  - When to use: as a checklist for systematically getting to know any new dataset.
  - How: (1) Inspect raw data and properties (headers, row/column counts, data types, categorical-vs-identifier distinction); (2) Compute descriptive statistics (`csvstat`) for per-feature summaries; (3) Create visualizations (`rush plot`) to see multi-feature interactions that statistics alone can't convey.
- **"rush" as boilerplate-elimination for R**: wraps R/tidyverse/ggplot2 so one-liners replace full REPL sessions.
  - When to use: you want to do one or two quick R operations (mutate, filter, stat, plot) without leaving the shell or writing a full R script with library-loading/reading/writing boilerplate.
  - How: `rush run -t '<R expression>' file.csv` (or `-` for stdin) executes the expression against a data frame `df` auto-loaded from the CSV; `rush plot --x --y --color --geom ...` is a `qplot`-based shortcut for common chart types; `--dry-run` reveals the generated R script, which can itself be saved as a new reusable tool (shebang included).
- **Textual-vs-graphical output duality**: `rush plot` auto-detects whether output is going to a terminal (textual ASCII/ANSI chart) or being redirected/piped (graphical PNG).
  - When to use: deciding how to preview a plot quickly (textual, in-terminal) vs. produce a shareable final image (graphical, to file).
  - How: same command/flags either way — `rush plot ... file.csv` for a quick textual look, `rush plot ... file.csv > out.png` to render a PNG.

## Key Concepts
- **Header detection**: no fixed way to tell if row 1 is a header — inspect manually (`head`, `nl` for wrapped lines, `trim`) and judge by whether it looks like field names vs. data.
- **less -S**: interactive, memory-efficient pager for inspecting large/raw files without loading them wholesale (unlike `cat`); `-S` disables line wrapping for wide CSVs.
- **csvcut -n**: lists column names with their 1-based index — a quick structural overview, more robust than `head -n1 | tr , '\n'` because it understands CSV quoting.
- **csvsql (no query)**: when invoked without `--query`, prints the inferred `CREATE TABLE` statement — a fast way to see each column's inferred type (DECIMAL, BOOLEAN, VARCHAR, TIMESTAMP, DATE) and whether it's `NOT NULL`.
- **csvstat**: the central descriptive-statistics tool; per column reports type, nulls, unique count, min/max/sum/mean/median/stdev (where applicable), and most-common values; flags: `--max`, `--min`, `--sum`, `--mean`, `--median`, `--stdev`, `--nulls`, `--unique`, `--freq`, `--len`, `-c` (subset columns); final line reports row count (quote/newline-safe, unlike `wc -l` which miscounts by including the header and can be fooled by embedded newlines).
- **xsv count**: alternative, faster way to get an accurate row count.
- **Unique-value cardinality heuristic**: low unique-count relative to row count → likely categorical (candidate for coloring/faceting); unique count == row count → likely identifier or continuous value; count of 1 → possibly droppable (no information).
- **rush run**: executes an arbitrary R(tidyverse) expression against an auto-loaded, auto-cleaned data frame `df`; supports `--library`, `--delimiter`, `--no-header`, `--no-clean-names`, `--dry-run`.
- **rush plot**: `qplot`/ggplot2-backed quick-plotting subcommand; key options `--x`, `--y`, `--color`, `--fill`, `--size`, `--alpha`, `--geom` (bar/histogram/density/boxplot/violin/smooth/path/point/auto), `--facets`, `--title`, `--xlab`/`--ylab`, `--log`, `--output`/redirection.
- **geom (geometry)**: determines chart type in ggplot2/rush — `bar` (categorical counts), `histogram`/`density` (continuous distribution), `boxplot`/`violin` (five-number summary / density by group), `point` (scatter, default for two continuous features), `smooth` (trend line), `path` (connects points in sequence — usually wrong for independent observations).
- **Displaying images from the CLI**: four options — (1) textual ASCII/ANSI rendering inline (works anywhere), (2) iTerm2 Inline Images Protocol via `imgcat`/`display` script (macOS iTerm2 only), (3) external image viewer via a Docker volume mount, (4) a local web server (`servewd`) + browser via a mapped port.

## Mental Models
- Treat statistics and visualization as complementary: statistics are terse and fast (good for single features, scriptable checks); visualizations reveal multi-feature interaction but need a display mechanism.
- "Happy little accidents" — mismatched geometry/column-type combinations (e.g., `geom path` on independent rows) can reveal that you've misunderstood your data's structure (rows aren't sequential/connected) — use them as a signal to pick a more appropriate geometry (e.g., scatter instead of path).
- A dry-run (`rush run -n` / inspecting generated R) turns a quick exploratory one-liner into a reusable, auditable script when needed — exploration and tool-building aren't separate tracks.

## Anti-patterns
- **Using `cat` to inspect a large unfamiliar file**: dumps everything at once with no navigation — use `less -S` instead for memory-efficient, scrollable, non-wrapping inspection.
- **Trusting `wc -l` as a row count for CSV**: it counts newlines, which includes the header and can be thrown off by embedded newlines in quoted fields — use `csvstat`'s row-count line or `xsv count` instead.
- **Assuming a feature's data type tells you its role**: a numeric-looking column (e.g., postal code) might still be a categorical identifier, not something to average — use domain knowledge plus unique-value cardinality, not just inferred SQL type.
- **Drawing a line/path plot (`geom path`) between independent observations**: implies a false sequential/connected relationship between unrelated rows — use a scatter plot (`geom point`) instead.
- **Relying solely on `csvstat`/`csvsql` type inference**: both use heuristics and can get it wrong — always do a manual/visual sanity check on top.

## Code Examples
```bash
$ csvstat venture.csv --unique
$ < venture.csv csvcut -C $(csvstat venture.csv --unique | grep ': 1$' | cut -d. -f 1 | tr -d ' ' | paste -sd,) | trim
```
- **What it demonstrates**: a pipeline that automatically identifies and drops all columns with only one unique value (no information content) using `csvstat`, `grep`, `cut`, `tr`, and `paste`, then feeds the resulting column list into `csvcut -C` via command substitution.

```bash
$ rush run -t 'mutate(df, percent = tip / bill * 100)' tips.csv > percent.csv
```
- **What it demonstrates**: a one-line tidyverse `mutate` replacing a full multi-step R REPL session (library load, read_csv, mutate, write_csv, quit) — the core value proposition of `rush`.

```bash
$ rush plot --x bill --y tip --color time tips.csv > plot-scatter.png
$ rush plot --x 'factor(size)' --y bill --geom violin --title 'Distribution of bill amount per party size' --xlab 'Party size' --ylab 'Bill (USD)' tips.csv > plot-violin.png
```
- **What it demonstrates**: `rush plot`'s declarative flag syntax for a scatter plot and a labeled violin plot, both producible as either textual (default) or graphical (redirected to file) output with the identical command.

```bash
$ rush run --library ggplot2 'ggplot(df, aes(x = bill, y = tip, color = time)) + geom_point() + geom_smooth()' tips.csv > plot-trend-points.png
```
- **What it demonstrates**: dropping down to full ggplot2 syntax via `rush run` when `rush plot`'s shortcut (which can't overlay multiple geometries) isn't flexible enough.

## Reference Tables
| Exploration goal | Tool |
|---|---|
| Row/column count, types | `csvsql` (no query), `csvstat` |
| Column names + index | `csvcut -n` |
| Per-column stats | `csvstat [--max|--min|--sum|--mean|--median|--stdev|--nulls|--unique|--freq|--len] [-c cols]` |
| Accurate row count | `csvstat | tail -1` or `xsv count` (not `wc -l`) |
| Quick R computation | `rush run '<expr>' file.csv` or `-` for stdin |
| Quick plot | `rush plot --x --y --color/--fill --geom <type>` |

| Geom | Use for |
|---|---|
| `bar` | categorical counts |
| `histogram` | continuous distribution (binned counts) |
| `density` | continuous distribution (smoothed) |
| `boxplot` | five-number summary by group |
| `violin` | distribution shape + summary by group |
| `point` (default) | relationship between two continuous features |
| `smooth` | trend line over continuous relationship |
| `path` | connects points in row order (rarely appropriate for independent rows) |

| Image display option | Requirement |
|---|---|
| Textual ASCII/ANSI (in-terminal) | none — works everywhere |
| iTerm2 Inline Images (`imgcat`) | macOS + iTerm2 |
| External image viewer | Docker volume mount (`-v`) |
| Browser via `servewd` | Docker port mount (`-p`) + `servewd` running |

## Worked Example
End-to-end exploration of `tips.csv`:
```bash
$ csvcut -n tips.csv                               # see column names/indices
$ csvsql tips.csv                                  # see inferred types
$ csvstat tips.csv --unique                         # cardinality per column -> day/time/sex/smoker look categorical
$ rush run -t 'mutate(df, percent = tip / bill * 100)' tips.csv > percent.csv   # derive a feature
$ < percent.csv rush run 'mean(df$percent)' -       # quick stat on the new feature
$ rush plot --x bill --y tip --color time percent.csv > scatter.png            # visualize relationship
$ rush plot --x 'factor(size)' --y bill --geom boxplot tips.csv > boxplot.png  # visualize by category
```
This sequence moves from structural inspection, to cardinality-based feature typing, to a derived feature via `rush run`, to two complementary visualizations (scatter for a continuous relationship, boxplot for a categorical breakdown) — mirroring the three-perspectives framework end to end.

## Key Takeaways
1. Always inspect structure first (`csvcut -n`, `csvsql`, `less -S`) before computing statistics or plotting.
2. Use `csvstat --unique` cardinality to distinguish categorical features, identifiers, and continuous values — don't rely on inferred type alone.
3. Trust `csvstat`'s row count or `xsv count` over `wc -l` for accurate CSV row counts.
4. `rush run` eliminates R's REPL boilerplate for one-off computations; `rush plot` eliminates ggplot2 verbosity for common chart types, auto-switching between textual and graphical output based on redirection.
5. Pick geometry based on the actual relationship in the data (don't connect independent rows with `geom path`; use `point`/`smooth` for continuous relationships, `boxplot`/`violin` for distribution-by-category).
6. When `rush plot`'s shortcut syntax can't express what you need (e.g., combining geoms), fall back to full ggplot2 via `rush run --library ggplot2`.
7. Choose your image-display method based on environment: textual for universality, iTerm2/viewer/browser for higher fidelity.

## Connects To
- **Ch 5 (Scrubbing Data)**: the `body`/`header`/`cols`/csvkit toolchain used for inspection builds directly on that chapter's CSV-handling tools.
- **Ch 9 (Modeling Data)**: exploratory statistics and visualizations here inform feature/model choices in the next OSEMN step.
- **Ch 6 (Make)**: the Star Wars boxplot example in Ch 6 reused `rush plot --geom boxplot`, foreshadowing this chapter.
- **ggplot2, plotnine**: external packages recommended for deeper, production-quality visualization beyond `rush plot`'s shortcuts.
- **Appendix**: csvstat, csvcut, csvsql, xsv, less, rush are all catalogued there.
