# Chapter 1: Introduction

## Core Idea
Data science is practically defined by the five-step OSEMN model, and the Unix command line — despite being 50+ years old — is a uniquely agile, augmenting, scalable, extensible, and ubiquitous environment for doing every step of it except interpretation.

## Frameworks Introduced
- **OSEMN model** (pronounced "awesome"), devised by Hilary Mason and Chris H. Wiggins: Obtain, Scrub, Explore, Model, iNterpret.
  - When to use: as the backbone for structuring any data science project/workflow.
  - How: move through Obtaining → Scrubbing → Exploring → Modeling → Interpreting data, but treat it as **iterative and nonlinear** — you routinely move back and forth (e.g., after modeling, return to scrubbing to adjust features).
- **The "Five Advantages" framework for command-line adoption**: command line is Agile, Augmenting, Scalable, Extensible, Ubiquitous.
  - When to use: to justify/decide when the command line beats a GUI or a single language environment for a given data task.
  - How: Agile (REPL + close-to-filesystem → fast iteration), Augmenting (glues tools together, is callable from R/Python/Spark, lets you wrap code as reusable CLI tools), Scalable (anything typed can be scripted/automated/parallelized/scheduled), Extensible (language-agnostic, community adds new tools constantly), Ubiquitous (runs on supercomputers, servers, laptops — 100% of top 500 supercomputers run Linux).

## Key Concepts
- **OSEMN**: Obtain, Scrub, Explore, Model, iNterpret — the five steps of practical data science.
- **REPL (Read-Eval-Print Loop)**: the command line evaluates each command immediately after Enter, unlike edit-compile-run-debug cycles of large programs/Hadoop jobs.
- **Command-line tool as glue**: tools are chained by piping one tool's output into another's input.
- **Interpreting data**: the one OSEMN step with no dedicated chapter/tool — human judgment, not automatable.
- **Docker image (dsatcl2)**: the book's prepackaged Ubuntu-based environment bundling all CLI tools used.

## Mental Models
- Treat data science as a **cyclical workflow diagram**, not a waterfall — any step can trigger returning to an earlier step.
- Think of the command line as an **augmenting** layer underneath your existing stack (R/Python/Excel), not a replacement for it — use whichever tool fits the task.
- "80% of the work in any data project is cleaning the data" (DJ Patil, quoted) — expect Scrub to dominate effort versus Explore/Model.

## Anti-patterns
- **Treating OSEMN as strictly linear**: leads to brittle workflows; real projects require jumping back to Scrub after Model, etc.
- **Assuming command line replaces R/Python/Spark**: the author explicitly says command line augments, not replaces, your existing workflow.
- **Trying to build new model types from scratch at the command line**: not what it's for — use it to invoke/build models via existing CLI tools or APIs instead.

## Code Examples
```bash
$ echo 'The command line is awesome!' | cowsay -f tux
```
- **What it demonstrates**: piping output of one tool (`echo`) into another (`cowsay`) — the most basic composability pattern that underlies the whole book.

```bash
$ whoami
dst
$ date
Tue Jun 29 02:25:17 PM CEST 2021
```
- **What it demonstrates**: the `$` prompt convention and that each typed command evaluates immediately (REPL).

## Reference Tables
| Advantage | Mechanism |
|---|---|
| Agile | REPL; close to filesystem |
| Augmenting | glue via pipes; callable from R/Python/Spark; code → reusable CLI tool |
| Scalable | automatable via scripts, cron, remote execution, parallelism |
| Extensible | language-agnostic tools; open-source ecosystem |
| Ubiquitous | present on supercomputers, servers, laptops, embedded systems |

## Worked Example
N/A — Chapter 1 is conceptual framing; no multi-step pipeline is built (the cowsay one-liner above is illustrative only).

## Key Takeaways
1. Use OSEMN to structure a data science project, but expect to loop back between steps.
2. The command line's core value is as "glue" connecting specialized small tools and other environments (R, Python, Spark).
3. Interpretation of results is inherently human and has no command-line chapter.
4. ~80% of real-world project effort is in Scrubbing (Ch 5), so budget time accordingly.
5. Command-line skills remain valuable because the technology has been stable for 50+ years and is everywhere you'll encounter a server or supercomputer.

## Connects To
- **Ch 3**: Obtaining Data — first OSEMN step detailed.
- **Ch 5**: Scrubbing Data — where the "80% of the work" claim is operationalized.
- **Ch 7**: Exploring Data — uses tools like `csvstat` and `rush`.
- **Ch 9**: Modeling Data — command line invokes/builds models rather than implementing algorithms from scratch.
- **Ch 4, 6, 8, 10**: the four "intermezzo" chapters (custom tools, Make, parallelism, polyglot) that apply across all OSEMN steps.
- **Docker**: the book's distribution mechanism for a reproducible tool environment.
