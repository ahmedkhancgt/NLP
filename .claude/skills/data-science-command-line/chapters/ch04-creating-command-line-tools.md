# Chapter 4: Creating Command-Line Tools

## Core Idea
A one-liner that's worth rerunning should be promoted into a reusable command-line tool through a disciplined six-step process; the same tool can be built in Bash, Python, or R, with language chosen by task complexity, not habit.

## Frameworks Introduced
- **The Six Steps to Turn a One-Liner into a Command-Line Tool**:
  - When to use: whenever you catch yourself about to retype or rerun a long pipeline.
  - How: (1) Create a file (e.g., with `fc` to edit the last command, or paste into an editor); (2) Give permission to execute (`chmod u+x`); (3) Define a shebang (`#!/usr/bin/env bash`, never a hardcoded path); (4) Remove the fixed input (delete the data-fetching step so the tool reads from stdin and is reusable on any input); (5) Add arguments/parameters (use positional params like `$1` with defaults, e.g. `NUM_WORDS="${1:-10}"`); (6) Optionally extend `PATH` (`export PATH="${PATH}:/your/tools/dir"`) so the tool runs from anywhere without `./` or full paths.
- **"Command-line tool as a function" analogy**: a tool is like a function — created once, callable with arguments, improves readability when composed into larger pipelines.
  - When to use: deciding whether to generalize a pipeline.
  - How: separate "data" from "operation" (don't bake a fixed input source into the script); expose variable behavior via arguments, not hardcoded flags.
- **Language choice heuristic for tool authoring**: Bash vs. Python/R.
  - When to use: picking implementation language for a new tool.
  - How: use Python/R instead of Bash when (a) you already have code in that language, (b) the Bash version would exceed ~100 lines, or (c) you need more safety/robustness (type checking) than Bash offers.

## Key Concepts
- **Shebang (`#!`)**: first line of a script naming its interpreter; `#!/usr/bin/env bash` (or `env python`, `env Rscript`) is preferred over a hardcoded path like `#!/usr/bin/bash` because `env` finds the interpreter wherever it's installed — more portable.
- **chmod u+x**: grants the file owner execute permission; without it you get "permission denied" even with a correct shebang.
- **File mode bits**: `-rwxr--r--` = file type, owner perms, group perms, other perms (10 characters total).
- **PATH environment variable**: colon-delimited list of directories Bash searches for executables; appending your tools directory lets you invoke tools by name from anywhere.
- **Positional parameters (`$1`, `$2`, ...)**: Bash's built-in mechanism for passing arguments into a script; `${1:-10}` supplies a default when `$1` is unset.
- **Streaming vs. blocking tools**: tools like `sort`/`uniq` must buffer all input before producing output (blocking), while well-written Python/R tools can process stdin line-by-line (streaming) — critical for infinite/live input.
- **stdin as a file object**: in Python, `sys.stdin.read()` (whole-input) vs. `sys.stdin.readline()` loop (streaming); in R, `file("stdin")` opened and read with `readLines`.

## Mental Models
- Think of tool-building as incremental refactoring: keep numbered versions (`top-words-1.sh` → `top-words-5.sh`) so each step's diff is isolated and reviewable.
- A good command-line tool separates three concerns: interpreter (shebang), fixed logic (the pipeline body), and variable inputs (stdin + arguments) — never conflate data with code.
- "Whichever approach works best for the task" — the command line, Bash scripting, and Python/R are complementary, not competing; mix them freely across a pipeline of tools.

## Anti-patterns
- **Leaving a data-fetching command (e.g., `curl`) inside the reusable tool**: locks the tool to one dataset; remove it and let data arrive via stdin instead.
- **Hardcoded shebang path (`#!/usr/bin/bash`)**: breaks if the interpreter lives elsewhere — use `#!/usr/bin/env bash`.
- **Letting a script write output to a hardcoded filename**: removes the user's ability to redirect output where they want; let output redirection (`>`) be the caller's job.
- **Using `$1` directly scattered through a long script instead of a named variable**: hurts readability once more than one or two arguments are involved.
- **Reading all of stdin before processing when input is an infinite/live stream**: blocking reads (like `sort` requires) make a tool useless on nonstop streams — use line-by-line reads for streaming data.

## Code Examples
```bash
$ curl -sL "https://www.gutenberg.org/files/11/11-0.txt" |
  tr '[:upper:]' '[:lower:]' |
  grep -oE "[a-z\']{2,}" |
  sort |
  grep -Fvwf stopwords |
  uniq -c |
  sort -nr |
  head -n 10
```
- **What it demonstrates**: the original top-10-words one-liner — download, lowercase, tokenize, dedupe/count, filter stopwords, sort by frequency, truncate — that becomes the basis for the tool.

```bash
#!/usr/bin/env bash
NUM_WORDS="${1:-10}"
tr '[:upper:]' '[:lower:]' |
grep -oE "[a-z\']{2,}" |
sort |
grep -Fvwf stopwords |
uniq -c |
sort -nr |
head -n "${NUM_WORDS}"
```
- **What it demonstrates**: the final `top-words-5.sh` — shebang, parameterized argument with default, and data read from stdin (not fixed input) — the fully generalized reusable tool.

```bash
$ export PATH="${PATH}:/data/ch04"
$ curl "https://www.gutenberg.org/files/11/11-0.txt" | top-words 10
```
- **What it demonstrates**: once on PATH and renamed without the `.sh`/version suffix, the tool is invoked like any built-in command, no `./` needed.

```python
#!/usr/bin/env python
import sys
CYCLE_OF_15 = ["fizzbuzz", None, None, "fizz", None, "buzz", "fizz", None, None,
               "fizz", "buzz", None, "fizz", None, None]
def fizz_buzz(n: int) -> str:
    return CYCLE_OF_15[n % 15] or str(n)
if __name__ == "__main__":
    try:
        while (n := sys.stdin.readline()):
            print(fizz_buzz(int(n)))
    except:
        pass
```
- **What it demonstrates**: a Python tool that processes stdin line-by-line (streaming), unlike `sys.stdin.read()`, so it works on infinite input.

## Reference Tables
| Step | Action | Command |
|---|---|---|
| 1 | Create file | `fc` or paste into editor |
| 2 | Make executable | `chmod u+x file.sh` |
| 3 | Add shebang | `#!/usr/bin/env bash` |
| 4 | Remove fixed input | delete data-source command; read stdin |
| 5 | Add arguments | `VAR="${1:-default}"` |
| 6 | Extend PATH | `export PATH="${PATH}:/dir"` |

| Language | Choose when |
|---|---|
| Bash | Short pipeline (<~100 lines), no complex logic |
| Python/R | Existing code to reuse, >100 lines of Bash needed, or need type safety/robustness |

## Worked Example
Build-up across the six steps, verified against two other-language ports:
```bash
$ cp top-words-{1,2}.sh && chmod u+x top-words-2.sh         # steps 1-2
$ echo '#!/usr/bin/env bash' | cat - top-words-2.sh > top-words-3.sh  # step 3
$ sed -i '2d' top-words-4.sh                                 # step 4: drop curl
$ bat top-words-5.sh                                          # step 5: NUM_WORDS="${1:-10}"
$ export PATH="${PATH}:/data/ch04" && cp top-words-5.sh top-words  # step 6
$ time < alice.txt top-words 5      # Bash
$ time < alice.txt top-words.py 5   # Python port
$ time < alice.txt top-words.R 5    # R port
```
All three produce identical top-5 word counts, demonstrating language-agnostic equivalence of the same tool concept; Bash is fastest/shortest here, Python/R need more lines but offer richer libraries (NLTK, pandas) for heavier text/numeric work.

## Key Takeaways
1. Promote any one-liner you'll reuse into a tool via the six-step process: file → executable → shebang → remove fixed input → add arguments → extend PATH.
2. Always use `#!/usr/bin/env <interpreter>` for portability, never a hardcoded interpreter path.
3. Decouple data from logic: read from stdin, write to stdout, let the caller redirect — this is what makes tools composable.
4. Use named variables (`NUM_WORDS="${1:-10}"`) instead of bare `$1`/`$2` once scripts grow.
5. Choose Bash for short/simple tools; switch to Python/R for existing code, complexity, or robustness needs.
6. For infinite/live input, process stdin line-by-line (streaming) rather than reading it all at once.
7. Keep your personal tools directory on PATH so custom tools behave like built-in Ubuntu commands.

## Connects To
- **Ch 2**: builds directly on the "five types of command-line tools" (interpreted scripts, shell functions) introduced there.
- **Ch 5**: the `tr`/`grep`/`sort`/`uniq` tools used in the one-liner are covered in depth as scrubbing tools.
- **Ch 6 (Make)**: Make formalizes running/chaining these custom tools as part of a reproducible workflow.
- **Ch 8 (Parallel Pipelines)**: reusable tools built here become the units parallelized with GNU Parallel.
- **docopt, ShellCheck**: external tools recommended for adding help text and linting Bash scripts respectively.
