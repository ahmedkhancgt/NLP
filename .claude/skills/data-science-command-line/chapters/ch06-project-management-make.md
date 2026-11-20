# Chapter 6: Project Management with Make

## Core Idea
Ad hoc command-line exploration produces a chaotic, unreproducible workflow; `make` formalizes a data pipeline as targets, rules, and dependencies in a Makefile, so steps run only when needed and the whole project can be rebuilt reliably by you or a collaborator.

## Frameworks Introduced
- **Make's target/rule/dependency model**: the core abstraction for reproducible data workflows.
  - When to use: whenever a project has more than a couple of interdependent steps (download → transform → visualize) that you'll want to rerun or share.
  - How: a **target** (usually a filename, or a `.PHONY` task name) has a **rule** (shell commands, indented with a literal tab) that builds it, and **dependencies** (prerequisites) listed after the colon; `make` only rebuilds a target if its file is missing or older than its dependencies.
- **Reproducibility hierarchy**: shell history → shell script → Makefile, in increasing robustness.
  - When to use: deciding how to record your workflow for future reuse/sharing.
  - How: reject `history`-digging (unreliable) and plain shell scripts (hard to read/maintain, unclear dependencies, reruns everything every time) in favor of make's dependency-aware, selectively-rebuilding model.

## Key Concepts
- **Makefile**: default config file `make` looks for (name it `Makefile`, capitalized, for convention); use `-f <file>` for a different name.
- **Target**: name after which a rule is defined (`target:`); conventionally a file to be created, but can be abstract (phony).
- **Rule**: the recipe (one or more shell commands) indented by a literal TAB character (not spaces — a "soft tab" causes `missing separator` errors).
- **.PHONY**: declares targets that don't correspond to real files, so they always run regardless of whether a same-named file exists (used for tasks like `clean`, `publish`, `all`).
- **"make is all about files"**: it checks only whether the target's name matches an existing file and its timestamp vs. dependencies — it does NOT verify the rule actually produced that file; mismatches between claimed target and actual output file break the dependency tracking silently.
- **Automatic variables**: `$@` = the target's name; `$<` = the name of the first prerequisite. Used to avoid repeating filenames in rules.
- **Dependencies**: listed after the target's colon (`data/starwars.csv: data`); make builds all prerequisites first, in order, before the target's own rule runs.
- **`.ONESHELL`**: makes all lines of a rule run in a single shell invocation (needed for multi-line piped rules to work as one pipeline rather than separate shell calls per line).
- **`SHELL := bash` + `.SHELLFLAGS := -eu -o pipefail -c`**: switches the rule interpreter from default `sh` to `bash` and makes it strict (exit on error, unset-variable error, pipefail) — a recommended best practice so pipeline failures aren't silently swallowed.
- **`$$` escaping**: a single `$` is make's own variable syntax, so a literal shell `$` (e.g., `$(pwd)`) inside a rule must be written as `$$(pwd)`.

## Mental Models
- Think of a Makefile as a dependency graph (DAG): each target points to its prerequisites, and `make <target>` walks that graph, (re)building only stale/missing nodes.
- "make reports 'is up to date'" is the signal that it's saving you time — it is not re-running work unnecessarily, unlike a flat shell script that always reruns everything.
- Treat your custom tools (Ch 4) and Makefile targets as complementary building blocks: tools do the work, Make decides when/in-what-order to call them.

## Anti-patterns
- **Using spaces instead of a tab to indent a rule**: produces `missing separator (did you mean TAB instead of 8 spaces?)` — editors that "soft-tab" break Makefiles.
- **Naming a target something that doesn't match the file the rule actually creates**: make's staleness check is purely by filename, so a mismatch (e.g., writing to `nummers` but naming the target `numbers`) causes the target to rebuild every time, or to be falsely considered up-to-date if some other process creates the file.
- **Putting everything in one giant shell script instead of a Makefile**: hard to read, dependencies between steps are implicit, and every run re-executes every step even when nothing changed.
- **Forgetting `.ONESHELL` with multi-line piped rules**: each line of the rule is sent to the shell separately by default, breaking pipelines that span multiple lines.
- **Not marking task-like targets `.PHONY`**: if a file happens to exist with the same name as a task target (e.g., a file named `clean`), make will skip running it, thinking it's already "built."

## Code Examples
```make
numbers:
	seq 7 > $@
```
- **What it demonstrates**: a minimal target+rule using the automatic variable `$@` to avoid hardcoding the output filename twice.

```make
.PHONY: clean publish docker-run

clean:
	rm book/2e/book.md book/2e/render*.rds

publish:
	(cd www && hugo) && netlify deploy --prod --dir www/public
```
- **What it demonstrates**: using make purely as a task runner (not file-building) via `.PHONY` targets — a common, valid use even outside "real" building.

```make
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

top10: data/starwars.csv
	grep Human $< |
	cut -d, -f 1,2 |
	sort -t, -k2 -nr |
	head
```
- **What it demonstrates**: a strict-bash, multi-line piped rule depending on a data file, using `$<` to reference that dependency by name — fails fast on any pipeline error thanks to `.SHELLFLAGS`.

## Reference Tables
| Concept | Syntax | Purpose |
|---|---|---|
| Target with no rule, pure dependency list | `all: top10 heights.png` | Build multiple targets with one command (`make` or `make all`) |
| Automatic variable `$@` | expands to target name | Avoid repeating the output filename |
| Automatic variable `$<` | expands to first prerequisite | Avoid repeating the input filename |
| `.PHONY` | `.PHONY: target1 target2` | Force always-run regardless of same-named file |
| Custom variable | `URL = "..."` then `$(URL)` | Centralize config values |
| Shell escape | `$$(pwd)` | Literal `$` passed through to the shell |

## Worked Example
Full Star Wars data pipeline Makefile, built step by step:
```make
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

URL = "https://raw.githubusercontent.com/tidyverse/dplyr/master/data-raw/starwars.csv"

.PHONY: all top10

all: top10 heights.png

data:
	mkdir $@

data/starwars.csv: data
	curl -sL $(URL) > $@

top10: data/starwars.csv
	grep Human $< | cut -d, -f 1,2 | sort -t, -k2 -nr | head

heights.png: data/starwars.csv
	< $< rush plot --x height --y species --geom boxplot > $@
```
Running `make` (no target given → builds first target `all`) triggers the full dependency chain: create `data/` → download `data/starwars.csv` into it → run both `top10` (prints top 10 tallest humans) and `heights.png` (boxplot of heights per species) — but only the first time; a second `make` run does nothing because every target is already up to date.

## Key Takeaways
1. Formalize any multi-step data workflow as a Makefile instead of shell history or a flat script — it documents steps, encodes dependencies, and skips already-done work.
2. Use real tab indentation for rules; a soft tab breaks the Makefile with a clear error message.
3. `$@` and `$<` keep rules DRY by referencing the target and first dependency respectively.
4. Mark non-file targets (`clean`, `publish`, `all`, task-like targets) as `.PHONY` so they always execute.
5. Add `SHELL := bash`, `.ONESHELL:`, and `.SHELLFLAGS := -eu -o pipefail -c` as a best-practice header for robust, multi-line, fail-fast rules.
6. make's staleness check is filename-based only — keep target names and actual output filenames in sync.
7. Treat Make as complementary to your custom tools from Ch 4: tools do work, Make orchestrates when they run.

## Connects To
- **Ch 4 (Creating Command-Line Tools)**: custom tools built there become the commands inside Make rules.
- **Ch 8 (Parallel Pipelines)**: Make's dependency graph pairs naturally with parallel execution of independent targets.
- **Ch 3/Ch 5**: the starwars example reuses obtaining (curl) and scrubbing (grep/cut/sort) patterns from those chapters inside Make rules.
- **git/GitHub**: recommended companion for version control, explicitly flagged as out of scope but essential alongside Make.
- **Luigi, Apache Airflow, Nextflow**: alternative workflow managers sharing the same target/rule/dependency concepts.
