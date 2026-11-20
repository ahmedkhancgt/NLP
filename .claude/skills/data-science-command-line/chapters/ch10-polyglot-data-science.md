# Chapter 10: Polyglot Data Science

## Core Idea
A polyglot data scientist uses whichever language/tool best fits each task; rather than treating the command line as a separate silo, this chapter shows how to call out to shell commands from Jupyter, Python, R/RStudio, and Apache Spark so the command line stays "within arm's reach" without breaking your main workflow.

## Frameworks Introduced
- **Polyglot Data Science principle**: use whatever programming language/tool/environment best accomplishes each task, and keep the command line accessible from within your primary environment rather than switching applications.
  - When to use: whenever you're in a Python/R/Spark session and need a quick shell operation (download, inspect, backup, compile) instead of reimplementing it in that language.
  - How: leverage each environment's shell-escape or subprocess-invocation mechanism (Jupyter's `!`, Python's `subprocess.run`, R's `system2()`/`pipe()`, Spark's `.pipe()`) rather than rewriting shell-native tasks in native code.
- **Shell-invocation mechanism per environment**: each language has a canonical, idiomatic way to call out to the shell.
  - When to use: picking the right technique per environment to avoid boilerplate or inefficiency.
  - How: Jupyter Console/Notebook → prefix with `!` (captures output as list of strings; supports `{variable}` interpolation, `{{` for literal braces); Python → `subprocess.run([...], input=..., capture_output=True, text=True)`; R → `system2()` (simple but writes input to a temp file first) or `pipe()`/`fifo()` (true streaming, no disk I/O) or a custom `sh()` helper chained with `%>%`; Spark → RDD `.pipe("<shell command>")` transformation (streams each partition's elements through the command, returns RDD of output strings).

## Key Concepts
- **Jupyter `!` (bang) operator**: runs a shell command inline in a cell/line; its stdout is captured as a Python list of strings when assigned to a variable (e.g., `total_lines = ! < alice.txt wc -l`).
- **Jupyter variable interpolation in shell commands**: `{python_variable}` inside a `!` command is substituted by the Python value; `{{literal_braces}}` escapes to produce literal curly braces in the shell command.
- **Jupyter bash magic**: allows multiline Bash cells, but author explicitly discourages it because passing Python variables into it is awkward — prefer a separate Bash script invoked via `!`.
- **subprocess.run()**: Python's recommended (over legacy `os.system()`) way to run external commands; takes a list (`["grep", "-i", pattern]`, not a single shell string) so there's no shell parsing (no piping/redirection syntax available directly); `input=` feeds stdin, `capture_output=True` + `text=True` captures stdout/stderr as text.
- **system2() (R)**: runs a command, can pass an R vector as `input=` (stdin) and capture `stdout=TRUE`; disadvantage: writes the input vector to a temp file before passing it, which doesn't scale well for large/frequent data.
- **pipe()/fifo() (R)**: lower-level R connections implementing true named-pipe (FIFO) streaming so no data touches disk — more efficient than `system2()` for large data, at the cost of more verbose connection-management boilerplate (open, write, read, close, unlink).
- **Custom `sh()` helper (R)**: wraps the `fifo()`/`pipe()` boilerplate into a reusable function chainable with magrittr's `%>%`, letting a sequence of shell commands read like a tidyverse pipeline (`sh(words, "grep -i alice") %>% sh("wc -l") %>% sh("cowsay")`).
- **Spark RDD.pipe("<command>")**: a Spark transformation that streams the elements of each RDD partition through an external shell command (written to its stdin) and returns the command's stdout as a new RDD of strings — works identically in Scala, PySpark, SparkR, and sparklyr.
- **Distributing custom tools to Spark executors**: a custom tool used inside `.pipe()` must exist on every executor node; use `spark-submit --files <tool>` to ship it to the cluster.
- **IDE terminals (JupyterLab, RStudio) are NOT connected to the code/notebook/console**: they're just an ordinary shell session running alongside your code — convenient for having the command line "within arm's reach," but no variable sharing or interactivity between the terminal and your running code.

## Mental Models
- Treat shell escapes as a lightweight alternative to reimplementing common OS-level operations (download, inspect, count, compress) in your host language — only drop into the shell for tasks the command line is naturally good at, not for core logic.
- Recognize the efficiency/boilerplate tradeoff: writing data to a temp file for the shell (`system2()`) is simple but costly at scale; true pipes (`pipe()`/`fifo()`, Spark's `.pipe()`) avoid disk I/O but need more setup — pick based on data size/frequency.
- The book's own running example (count occurrences of "alice" in `alice.txt` via `grep -i alice | wc -l`) is reimplemented identically across Jupyter, Python, R, and Spark — a useful template for comparing shell-invocation idioms across environments.

## Anti-patterns
- **Using Python's `os.system()` instead of `subprocess.run()`**: deprecated/less safe pattern — `subprocess.run()` is the recommended modern approach.
- **Passing a single shell-syntax string to `subprocess.run()` expecting piping/redirection to work**: without `shell=True`, `run()` takes a list of arguments and does not parse shell syntax — no implicit piping unless you explicitly chain multiple `run()` calls or set `shell=True` (with its own risks).
- **Using R's `system2()` for large or frequent data transfers**: it writes the input to a temporary file before invoking the command, which doesn't scale — use `pipe()`/`fifo()` or the custom `sh()` helper for true streaming.
- **Relying on Jupyter/RStudio's bash-magic/multiline-bash cells with embedded variable interpolation**: awkward and error-prone compared to writing a standalone Bash script invoked with `!`.
- **Forgetting to distribute a custom command-line tool to all Spark executor nodes**: `.pipe()` fails silently or errors if the tool isn't present cluster-wide — use `spark-submit --files`.

## Code Examples
```python
In [5]: total_lines = ! < alice.txt wc -l
In [7]: int(total_lines[0])
Out[7]: 3735
```
- **What it demonstrates**: capturing a shell command's stdout as a Python list in Jupyter, then casting the single captured line to an integer for use in code.

```python
#!/usr/bin/env python
from subprocess import run
grep = run(["grep", "-i", pattern], input=words, capture_output=True, text=True)
print(len(grep.stdout.strip().split("\n")))
```
- **What it demonstrates**: the recommended Python pattern for invoking a shell tool with piped stdin and captured stdout/text output — no shell string parsing involved.

```r
> alice <- system2("grep", c("-i", "alice"), input = words, stdout = TRUE)
> length(alice)
[1] 403
```
- **What it demonstrates**: R's simplest shell-invocation mechanism, passing an R character vector as stdin and capturing stdout as a vector — at the cost of an intermediate temp-file write.

```scala
scala> sc.textFile("alice.txt").flatMap(line => line.split(" ")).pipe("grep -i alice").pipe("wc -l").map(x => x.toInt).reduce(_ + _)
res5: Int = 403
```
- **What it demonstrates**: chaining two shell-command pipes (`grep` then `wc -l`) directly into a Spark RDD transformation pipeline, with per-partition counts finally summed in Scala.

## Reference Tables
| Environment | Shell-invocation mechanism | Notes |
|---|---|---|
| Jupyter Console/Notebook | `! <command>` | captures stdout as list; supports `{var}` interpolation |
| Python | `subprocess.run([...], input=, capture_output=True, text=True)` | list of args, no shell parsing by default |
| R (simple) | `system2(cmd, args, input=, stdout=TRUE)` | writes input to temp file first |
| R (efficient) | `pipe()` + `fifo()`, or custom `sh()` helper | true streaming, more boilerplate |
| Apache Spark (any API) | `rdd.pipe("<command>")` | per-partition streaming transformation |

## Worked Example
The same task — count occurrences of "alice" in `alice.txt` — across all four environments, demonstrating idiom equivalence:
```python
# Jupyter: ! head alice.txt ; total_lines = ! < alice.txt wc -l
# Python: grep = run(["grep","-i","alice"], input=words, capture_output=True, text=True); len(grep.stdout.strip().split("\n"))
# R: alice <- system2("grep", c("-i","alice"), input=words, stdout=TRUE); length(alice)
# Spark: sc.textFile("alice.txt").flatMap(_.split(" ")).pipe("grep -i alice").pipe("wc -l").map(_.toInt).reduce(_+_)
```
All four return **403**, showing that the same shell-tool-based logic (`grep -i alice | wc -l`) can be embedded consistently across interactive notebooks, scripts, statistical environments, and distributed cluster computing.

## Key Takeaways
1. You don't have to choose between the command line and your primary language/environment — most environments offer a clean way to shell out.
2. Prefer `subprocess.run()` over `os.system()` in Python; prefer true pipes (`pipe()`/`fifo()`) over `system2()` in R for anything beyond small, infrequent data.
3. IDE terminals (JupyterLab, RStudio) are just ordinary shells running alongside your session — useful for convenience, not for sharing state with your code.
4. Apache Spark's `.pipe()` transformation lets you embed arbitrary shell/Bash/Perl scripts directly into a distributed data pipeline, available identically across Scala, PySpark, SparkR, and sparklyr.
5. When using a custom tool inside Spark's `.pipe()`, ship it to all executors with `spark-submit --files`.
6. The command line "doesn't exist in a vacuum" — treat it as one tool among many, chosen pragmatically per task.

## Connects To
- **Ch 4 (Creating Command-Line Tools)**: the custom tools built there (in Bash, Python, R) are exactly what gets invoked via these shell-escape mechanisms from other environments.
- **Ch 3/Ch 5**: the curl-download and grep-filter operations reused here as cross-language examples come directly from Obtaining/Scrubbing chapters.
- **Ch 11 (Conclusion)**: this chapter's "command line doesn't exist in a vacuum" theme directly anticipates the book's closing "be practical" advice.
- **reticulate (R package)**: mentioned as an alternative, non-command-line way to integrate R and Python directly.
- **PySpark, SparkR, sparklyr**: language bindings sharing Spark's `.pipe()` mechanism.
