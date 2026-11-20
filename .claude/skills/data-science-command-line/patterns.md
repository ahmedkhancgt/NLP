# Patterns

## Pipe-and-Filter Chaining
**When to use** Any multi-step transformation where each tool does one thing well (download, filter, count, sort).
**How** Connect tools with `|` so stdout feeds the next tool's stdin: `curl -sL URL | grep PATTERN | sort | uniq -c | sort -nr | head`. Insert intermediate transformation tools (`awk`, `sed`, `header`) whenever two tools' formats don't line up.
**Trade-offs** Extremely composable and inspectable (each stage can be run alone to debug), but long chains can become hard to read/quote — extract into a named tool (Ch 4) once reused.

## CSV-Safe Wrapping of Text Tools
**When to use** You need to apply a classic text tool (`tr`, `sort`, `grep`) to a CSV without corrupting or losing the header.
**How** `body <cmd>` runs `<cmd>` only on data rows, re-printing the header untouched; `cols -c <col> <cmd>` restricts a command to one column; `header -a/-d/-r/-e` manipulates the header line itself.
**Trade-offs** Keeps familiar tools usable on tabular data with minimal new syntax, but still doesn't understand quoted commas/embedded delimiters — fall back to `csvkit`/`csvsql` for those edge cases.

## Promote-a-One-Liner-to-a-Tool
**When to use** A pipeline you've run more than once, or expect to reuse/share.
**How** Six steps: paste into a file → `chmod u+x` → add `#!/usr/bin/env <interpreter>` shebang → remove any fixed data source so the tool reads stdin → parameterize fixed values via positional args (`"${1:-default}"`) → add the tool's directory to `PATH`.
**Trade-offs** Costs a few minutes up front; pays off by eliminating retyping, enabling composition into bigger pipelines/Makefiles, and making behavior auditable/versionable.

## Format-Convert-Early
**When to use** Data arrives as HTML, XML, JSON, or a spreadsheet, but downstream tools expect CSV.
**How** Convert to flat CSV as the very first scrubbing step: HTML/XML → `pup`(CSS-select)/`xml2json` → JSON → `jq -r '...|@csv'` → `header -a` → CSV. Spreadsheets → `in2csv`. Databases → `sql2csv`.
**Trade-offs** Normalizing early lets you use the same generic CSV toolset (csvkit, rush, awk) for everything downstream, rather than learning/maintaining format-specific logic repeatedly.

## Dependency-Graph Workflow (Make)
**When to use** A project with multiple interdependent, re-run-prone steps (fetch → transform → model → visualize).
**How** Declare each step as a `target: prerequisite` pair with a tab-indented rule in a Makefile; use `$@`/`$<` to avoid repeating filenames; mark non-file tasks `.PHONY`; run `make` or `make <target>` — only stale/missing targets rebuild.
**Trade-offs** Upfront structuring effort, but gives free incremental rebuilds, explicit dependencies, and reproducibility for you and collaborators.

## Controlled Fan-Out/Fan-In (GNU Parallel)
**When to use** The same command/pipeline must run many times (API calls, file conversions, model sweeps) and you want bounded concurrency instead of ad hoc `&` backgrounding.
**How** `<input> | parallel [--jobs N] [--keep-order] [--tag] '<command with {} placeholders>'`; use `:::` to supply an argument list directly; `--dryrun` to verify before executing; `--results <dir>` to capture per-job stdout/stderr/seq.
**Trade-offs** Much safer and more traceable than naive background jobs, but concurrency still needs to respect external constraints (API rate limits via `-j1`, shared resources).

## Transfer-Process-Return (Remote Distribution)
**When to use** Local compute/data is insufficient and you have SSH access to other machines.
**How** `--nonall --slf hosts <cmd>` to run everywhere once; `--pipe -N<n> --slf hosts <cmd>` to chunk-distribute piped data; `--basefile <tool> --trc {.}.ext --slf hosts <cmd>` to ship a tool/file, run it, retrieve a named result, and clean up.
**Trade-offs** Scales beyond one machine without installing `parallel` remotely (only SSH is required), but adds network transfer overhead and quoting complexity — extract complex remote commands into an uploadable script when quoting gets unwieldy.

## Inspect-Before-Extract/Transform
**When to use** Before extracting a new/unfamiliar archive, or running an expensive transformation on unfamiliar data.
**How** List contents first (`tar -tzf`, `unzip -l`), peek at structure (`head`, `less -S`, `csvcut -n`, `csvsql` without `--query`) before committing to extraction or a full pipeline run.
**Trade-offs** A small extra step that prevents wasted work (cluttering directories, misinterpreting column meaning) and catches structural surprises early.

## Reduce-Early-in-the-Pipeline
**When to use** A pipeline is slow, expensive (API calls), or operating on more data than needed during development.
**How** Place data-reducing tools (`sample`, `head`, `grep` filters) as early as possible in the chain while iterating; remove them once the pipeline is validated and ready for the full run.
**Trade-offs** Speeds up iteration cycles significantly during development; remember to remove/adjust reducers before the "real" run, or results will be incomplete.

## Textual-Then-Graphical Visualization
**When to use** Exploring a dataset interactively at the terminal, then producing a shareable final chart.
**How** Run `rush plot --x ... --y ... --geom ...` with no redirection to preview an ASCII/ANSI chart in-terminal; redirect to a file (`> out.png`) to render the graphical PNG version with the identical flags.
**Trade-offs** Zero extra syntax to learn for the two modes, but textual charts are coarse — use them only for a quick sanity check, not final analysis/communication.

## Train/Test Split Discipline
**When to use** Before evaluating any supervised model (regression or classification).
**How** Shuffle (`shuf`), split into N roughly-equal parts (`split -d -n r/N`), reserve one part (e.g., 20%) as test, combine the rest (e.g., 80%) as train; for classification, first balance class counts (undersample majority via `body shuf | head -n <count>`).
**Trade-offs** Essential for a meaningful accuracy/error estimate; balancing trades away some majority-class data to avoid a misleadingly high naive accuracy.
