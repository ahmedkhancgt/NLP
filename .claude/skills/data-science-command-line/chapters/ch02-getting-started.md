# Chapter 2: Getting Started

## Core Idea
Before doing data science at the command line you need the book's datasets, a consistent tool environment (via Docker), and a working mental model of the four environment layers (tools, terminal, shell, OS) plus the core Unix mechanics: streams, piping, redirection, and getting help.

## Frameworks Introduced
- **Four-layer environment model**: Command-line tools → Terminal → Shell → Operating system.
  - When to use: to reason about where a behavior comes from (e.g., is `cd` a tool, a shell feature, or an OS call?).
  - How: Terminal is the app you type into; it sends text to the Shell (e.g., zsh), which interprets and executes Command-line tools, which are run by the Linux kernel (Operating system).
- **Five types of command-line tools**: binary executable, shell builtin, interpreted script, shell function, alias.
  - When to use: deciding how to build/extend your own toolbox (Ch 4 builds on this).
  - How: binary executables and shell builtins are fixed/preinstalled; interpreted scripts, shell functions, and aliases are the three types you can write/edit yourself to become more productive.
- **stdin/stdout/stderr model**: every tool has three standard streams.
  - When to use: whenever chaining or debugging tool output.
  - How: stdout and stderr both default to the terminal; redirect/pipe each independently (`>`, `2>`, `|`).

## Key Concepts
- **Docker image/container**: a bundle of an app + dependencies (image) run in an isolated environment (container); `docker run --rm -it -v "$(pwd)":/data datasciencetoolbox/dsatcl2e` mounts local data at `/data`.
- **Prompt ($)**: convention marking a typed command; never typed by the user.
- **Pipe operator (`|`)**: connects stdout of one tool to stdin of the next — "automated copy and paste."
- **Output redirection (`>`, `>>`)**: write/append stdout to a file, overwriting or appending.
- **`/dev/null`**: special file used to discard unwanted output (commonly stderr via `2>`).
- **sponge**: soaks up all stdin before writing to a file, enabling safe "read from and write to the same file" pipelines that would otherwise truncate the file to empty.
- **type**: shell builtin that reveals which of the five tool types a given command is.
- **man / --help / tldr**: three escalating ways to get help — full manual, quick built-in help, and community-curated practical examples respectively.

## Mental Models
- Think of a pipeline as water flowing through pipes: stdout of tool A becomes stdin of tool B; stderr is a separate side-channel that still goes to the terminal unless redirected.
- "Read-then-write-same-file" is a trap: the writing tool truncates the file before the reading tool finishes, so either rename-after-write or use `sponge`.
- Help-seeking is tiered: try `--help` for quick syntax, `man` for full detail, `tldr` for example-driven quick reference, and the internet last.

## Anti-patterns
- **`< file.txt tool > file.txt`**: empties the file because the output redirection truncates it immediately, before `tool` reads anything — use `sponge` or write-then-`mv` instead.
- **Treating cat as wasteful ("useless use of cat")**: author explicitly dismisses this purism — clarity over micro-optimization.
- **Manually editing book-style trimmed output as if it were real tool behavior**: `trim`/`csvlook` wrappers are presentation aids, not functional changes to underlying tools.

## Code Examples
```bash
$ docker run --rm -it -v "$(pwd)":/data datasciencetoolbox/dsatcl2e
```
- **What it demonstrates**: mounting a local directory as `/data` inside the container for getting data in/out.

```bash
$ curl -s "https://www.gutenberg.org/files/11/11-0.txt" | grep " CHAPTER" | wc -l
12
```
- **What it demonstrates**: a three-tool pipeline (download → filter → count) — curl's stdout piped into grep's stdin piped into wc.

```bash
$ dseq 5 > dates.txt
$ < dates.txt nl | sponge dates.txt
```
- **What it demonstrates**: safely numbering lines of a file and overwriting that same file without emptying it, using `sponge`.

```bash
$ cat movies.txt 404.txt 2> /dev/null
```
- **What it demonstrates**: suppressing stderr (a "file not found" error) while still printing stdout from the files that do exist.

## Reference Tables
| Tool type | Editable? | Examples |
|---|---|---|
| Binary executable | No | `ls`, `cat` |
| Shell builtin | No | `cd`, `pwd` |
| Interpreted script | Yes | Python/R/Bash scripts, `fac.py` |
| Shell function | Yes | `fac() { ... }` in `.zshrc` |
| Alias | Yes (no params) | `alias l='ls --color -lhF --group-directories-first'` |

| Help method | Depth | Best for |
|---|---|---|
| `--help`/`-h` | Shallow | Quick syntax reminder |
| `man <tool>` | Deep | Full option reference |
| `tldr <tool>` | Example-driven | Practical common-case usage |

## Worked Example
Goal: get the chapter count of an online book and save the chapter list.
```bash
$ curl -s "https://www.gutenberg.org/files/11/11-0.txt" | grep " CHAPTER" > chapters.txt
$ cat chapters.txt
$ wc -l < chapters.txt
```
This chains download (curl) → filter matching lines (grep) → persist to file (`>`) → re-read and count (`wc -l`), illustrating obtain→filter→save→inspect as a minimal pipeline pattern reused throughout the book.

## Key Takeaways
1. Use the book's Docker image (`datasciencetoolbox/dsatcl2e`) with a volume mount (`-v "$(pwd)":/data`) for a reproducible, pre-provisioned toolset.
2. Know your four environment layers (tool/terminal/shell/OS) to reason about tool behavior.
3. Master piping (`|`) and redirection (`>`, `>>`, `2>`, `/dev/null`) as the foundation of all later chapters.
4. Never read and write the same file in one step without `sponge` or a rename work-around.
5. Prefer building tools as interpreted scripts, shell functions, or aliases — the three editable/extensible tool types.
6. Escalate help-seeking: `--help` → `man` → `tldr` → internet.

## Connects To
- **Ch 4**: Creating Command-Line Tools builds directly on the "five types of tools" distinction, especially interpreted scripts and shell functions.
- **Ch 3**: Obtaining Data continues the curl/grep pattern shown here into fuller data-acquisition workflows.
- **Docker**: core infrastructure dependency for the whole book's examples.
- **csvlook, bat, trim, tee**: output-management helper tools introduced here and reused throughout.
