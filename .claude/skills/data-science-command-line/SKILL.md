---
name: data-science-command-line
description: "Knowledge base from \"Data Science at the Command Line\" (2nd ed.) by Jeroen Janssens. Use when doing data science with Unix command-line tools — obtaining/scrubbing/exploring data via pipelines, building custom command-line tools, project management with Make, parallel pipelines, modeling data from the shell, or polyglot data science."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Science at the Command Line
**Author**: Jeroen Janssens | **Pages**: ~283 | **Chapters**: 11 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch05`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**OSEMN** (pronounced "awesome") — the book's backbone, devised by Hilary Mason and Chris H. Wiggins: **O**btain, **S**crub, **E**xplore, **M**odel, i**N**terpret data. Treated as iterative/nonlinear, not a waterfall. The command line's usefulness decreases across the gradient: highest for Obtain, lowest (essentially none) for Interpret, which has no dedicated chapter — interpretation is inherently human judgment.

**The Five Advantages of the command line**: Agile (REPL, close to the filesystem), Augmenting (glues tools/languages together, not a replacement for R/Python), Scalable (anything typed can be scripted/parallelized/scheduled), Extensible (language-agnostic, community-built), Ubiquitous (every Unix-like system, including supercomputers and cloud servers).

**Unix philosophy / text-as-universal-interface**: small tools that do one thing well, composed via pipes (`|`); plain text (and CSV/JSON as structured text) is the common currency that lets unrelated tools interoperate.

**stdin/stdout/stderr + piping/redirection**: every tool has three streams; `|` connects one tool's stdout to the next's stdin; `>`/`>>` redirect to files; `2>` isolates errors; `/dev/null` discards; `sponge` safely reads-and-writes the same file.

**The Six Steps to turn a one-liner into a reusable tool** (Ch 4): (1) save to a file, (2) `chmod u+x`, (3) add a portable shebang (`#!/usr/bin/env bash`), (4) remove fixed input so it reads stdin, (5) parameterize fixed values as arguments (`"${1:-default}"`), (6) add its directory to `PATH`.

**CSV-safe wrapping**: classic text tools (`grep`, `tr`, `sort`) don't understand CSV headers/columns — use `body` (apply to data rows only), `header` (manipulate the header line), `cols` (apply to one column), or switch to csvkit (`csvcut`, `csvgrep`, `csvsql`) for full CSV awareness, especially with embedded delimiters.

**Make's target/rule/dependency model** (Ch 6): a Makefile declares `target: prerequisites` with a tab-indented rule; `make` rebuilds only stale/missing targets, using `$@`/`$<` to avoid repeating filenames and `.PHONY` for non-file tasks.

**Serial → naive parallel → GNU Parallel** (Ch 8): shell `for`/`while` loops (serial) → backgrounding with `&` (uncontrolled concurrency) → `parallel` (bounded `--jobs`, ordered `--keep-order`, traceable `--tag`/`--results`, and distributable to remote machines via SSH with `--slf`/`--pipe`/`--basefile`).

**Three Modeling Task Types** (Ch 9): no label → unsupervised dimensionality reduction (`tapkee`: PCA linear, t-SNE nonlinear); continuous label → regression (`vw`/Vowpal Wabbit); categorical label → classification (`skll`/SciKit-Learn Laboratory). Always scale features first, always hold out a test set, balance classes for classification.

**Polyglot Data Science** (Ch 10): use whichever language fits the task; keep the command line "within arm's reach" from Jupyter (`!`), Python (`subprocess.run`), R (`system2()`/`pipe()`/`fifo()`), and Apache Spark (`rdd.pipe("<cmd>")`) rather than treating it as a separate silo.

**Three Pieces of Advice** (Ch 11): Be Patient (tool quirks are decades of independent development — use `--help`/`man`/`tldr`), Be Creative (combine existing tools before reaching for a programming language; generalize useful code into a tool), Be Practical (command line is a companion approach, strongest at Obtain, weakest at Interpret — mix approaches freely).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| 1 | Introduction | OSEMN model; Five Advantages of the command line |
| 2 | Getting Started | Four-layer environment model; Five types of command-line tools; stdin/stdout/stderr |
| 3 | Obtaining Data | Text-as-universal-interface; format-driven tool selection (curl, in2csv, sql2csv, jq) |
| 4 | Creating Command-Line Tools | Six Steps to a reusable tool; Bash vs. Python/R choice; streaming vs. blocking |
| 5 | Scrubbing Data | Transformations-everywhere; CSV-safe wrapping (body/header/cols); format-conversion-first (HTML/XML/JSON → CSV) |
| 6 | Project Management with Make | Target/rule/dependency model; reproducibility hierarchy |
| 7 | Exploring Data | Three Perspectives (inspect/describe/visualize); rush run/plot |
| 8 | Parallel Pipelines | Serial vs. naive parallel vs. GNU Parallel; three distributed-processing flavors |
| 9 | Modeling Data | Three Modeling Task Types; train/test split discipline |
| 10 | Polyglot Data Science | Polyglot principle; per-environment shell-invocation mechanisms |
| 11 | Conclusion | Three Pieces of Advice; OSEMN applicability gradient |

## Topic Index
- **Appendix (tool reference)** → folded into glossary.md
- **awk / sed / tr / grep** → Ch 5
- **body / header / cols (CSV-safe wrapping)** → Ch 5
- **csvkit (csvcut, csvgrep, csvjoin, csvsql, csvstack, csvstat)** → Ch 3, 5, 7, 9
- **curl (downloading, APIs, streaming)** → Ch 3
- **Docker image / environment setup** → Ch 2
- **GNU Parallel** → Ch 8
- **jq (JSON processing)** → Ch 3, 5
- **Make / Makefile** → Ch 6
- **Modeling (tapkee, vw, skll)** → Ch 9
- **OSEMN model** → Ch 1, 11
- **Polyglot (Jupyter, Python, R, Spark)** → Ch 10
- **rush (R one-liners, plotting)** → Ch 5, 7, 9
- **Shebang / chmod / PATH (tool creation)** → Ch 4
- **SSH / distributed processing** → Ch 8
- **Streams (stdin/stdout/stderr), piping, redirection** → Ch 2
- **Train/test split, dimensionality reduction, regression, classification** → Ch 9

## Supporting Files
- [glossary.md](glossary.md) — key terms and command-line tools with chapter cross-references
- [patterns.md](patterns.md) — pipeline techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 11 chapters, synthesized as structure, not verbatim text. The book's own command-line tool appendix is folded into glossary.md rather than kept as a separate chapter.
