# Chapter 8: Parallel Pipelines

## Core Idea
Repetitive command-line tasks can be looped serially (for/while over numbers, lines, files) using native shell constructs, but GNU `parallel` replaces naive background-job parallelism with controlled, ordered, loggable parallel/distributed execution — across local cores or remote machines via SSH — without modifying the underlying tools.

## Frameworks Introduced
- **Three loopable item types (serial processing)**: numbers, lines, files.
  - When to use: before reaching for `parallel`, know the native shell looping idiom appropriate to what you're iterating over.
  - How: numbers → `for i in {0..100..2}; do ...; done` (Z shell brace expansion); lines → `while read line; do ...; done < file` (needed because shell doesn't know line count ahead of time); files → `for f in /data/*; do ...; done` (globbing) or `find` for recursive/attribute-based search (handles special characters, size/time/permission filters).
- **Serial vs. naive parallel vs. GNU Parallel (three processing models)**: a conceptual progression.
  - When to use: deciding how to execute a repeated command at scale.
  - How: Serial = one job at a time (slow, simple); Naive parallel = backgrounding with `&` (uncontrolled concurrency, output interleaving, no way to limit job count); GNU Parallel = controlled concurrency (`--jobs`), ordered output (`--keep-order`), tagged output (`--tag`), and works with unmodified existing tools.
- **Three flavors of distributed processing with `parallel` + SSH**: running commands remotely, distributing data remotely, processing files remotely.
  - When to use: local cores aren't enough; you have access to remote machines (e.g., AWS EC2) reachable via SSH.
  - How: (1) `--nonall --sshloginfile hostnames <cmd>` runs the same command on every remote machine; (2) `--pipe -N<chunk> --slf hostnames <cmd>` splits piped input into chunks distributed to remote machines; (3) `--basefile <tool> --trc {.}.ext --slf hostnames <cmd>` transfers a file/tool, runs it remotely, and returns results, cleaning up afterward.

## Key Concepts
- **`&` (background operator)**: sends a command to the background so the shell continues immediately; enables naive parallelism but with no concurrency control or output/input correlation.
- **GNU Parallel core syntax**: `<input-source> | parallel [options] <command with {} placeholders>` or `parallel [options] <command> ::: <arg-list>`.
- **Placeholders**: `{}` = whole input item (appended automatically if omitted); `{1}`,`{2}`,... = fields split by `--colsep`/`-C`; `{/}` = basename; `{.}` = filename without last extension; `{#}` = job sequence number; `--header :` + column-name placeholders when input is CSV with a header.
- **`--jobs`/`-j`**: controls concurrency — a bare number N (exactly N jobs), `+N` (N plus CPU cores), `-N` (cores minus N), a percentage of cores (default 100%), `-j1` (serial, useful for rate-limited APIs), `-j0` (unlimited, discouraged).
- **`--dryrun`**: prints the commands parallel would run without executing them — essential for verifying placeholder/quoting correctness before a real run.
- **`--results <dir>`**: stores each job's `seq` (job number), `stdout`, and `stderr` in separate files/subdirectories instead of interleaving everything to the terminal.
- **`--keep-order`/`-k`**: preserves input order in output despite concurrent/out-of-order completion.
- **`--tag`**: prepends each output line with its corresponding input item, for traceability.
- **`--sshloginfile`/`--slf <file>`**: file listing remote hostnames (one per line) for distributed execution; `--sshlogin :` targets the local machine (useful for following along without real remote machines).
- **`--nonall`**: run the same command on every remote machine once, without distributing any argument list.
- **`--pipe` + `-N<n>`**: splits piped standard input into chunks of `n` items and distributes chunks across (remote) jobs — the mechanism for distributing large data rather than just commands.
- **`--basefile <file>` / `--trc {.}<ext>`**: transfers a local file (binary, script, or data) to each remote machine before running the job; `--trc` (`--transfer --return <pattern> --cleanup`) transfers input, returns named output, and removes both from the remote afterward.
- **pbc**: a custom tool (`parallel -C, -k -j100% "echo '$1' | bc -l"`) demonstrating how to wrap `parallel` + `bc` into a new, genuinely parallel reusable command-line tool (per Ch 4's six-step tool-creation pattern).

## Mental Models
- Picture GNU Parallel as a dispatcher that takes an input stream (numbers/lines/files/CSV rows) and fans it out across a bounded pool of worker processes (local cores or remote machines via SSH), then funnels stdout/stderr back — controlled fan-out/fan-in, unlike raw `&` backgrounding which has no fan-in discipline.
- "Not everything can be parallelized" — rate-limited APIs or single-instance resources may require `-j1` (effectively serial) even when using `parallel`'s interface, because the tool's job-control features (tagging, dry-run, results) still add value regardless of concurrency level.
- Treat quoting/escaping complexity as a signal to extract the pipeline into its own command-line tool (as with `add` and `pbc`) rather than fighting nested shell quoting inline.

## Anti-patterns
- **Backgrounding jobs naively with `&` in a loop**: no control over concurrent job count (resource contention risk) and no reliable way to match output to input — use `parallel` instead.
- **Letting `parallel` guess where to insert input (omitting `{}`)**: works but is implicit; explicitly placing `{}` (or numbered/named placeholders) avoids ambiguity, especially with multiple inputs.
- **Running `-j0` (max unlimited concurrency) by default**: author explicitly advises against this — prefer a bounded `--jobs` value matched to actual resource constraints.
- **Redirecting parallel output to per-item files or one big file manually**: works but duplicates parallel's own `--results` mechanism, which already organizes stdout/stderr/job-number per job.
- **Assuming all remote machines have `parallel` installed**: triggers a "could not figure out number of cpus" warning and silent fallback to 1 core per machine — either install it remotely, specify `--jobs` explicitly, or prefix hostnames with `N/` for per-host core counts.
- **Using the search-path version of a tool on a remote machine when you explicitly uploaded your own copy**: must reference the uploaded binary as `./toolname`, not bare `toolname`, or you may run a different/absent version.

## Code Examples
```bash
$ seq 0 2 100 | parallel "echo {}^2 | bc" | trim
```
- **What it demonstrates**: the simplest form of `parallel` — items piped via stdin, explicit `{}` placeholder, running one job per number concurrently.

```bash
$ parallel --jobs 2 ./slow.sh ::: {A..C}
```
- **What it demonstrates**: `:::` syntax for supplying an argument list directly (instead of piping), combined with `--jobs 2` to bound concurrency to 2 simultaneous jobs.

```bash
$ seq 1000 | parallel -N100 --pipe --slf hostnames "paste -sd+ | bc" | paste -sd+ | bc
500500
```
- **What it demonstrates**: distributing a large numeric stream in chunks of 100 across remote machines (`--pipe -N100 --slf`), each chunk summed remotely, then the partial sums combined locally — correct distributed computation of the same result as a local-only `seq 1000 | paste -sd+ | bc`.

```bash
$ ls *.json.gz | parallel -v --basefile jq --trc {.}.csv --slf hostnames \
  "zcat {} | ./jq -r '.borough' | tr '[A-Z] ' '[a-z]_' | sort | uniq -c | awk '{print \$2\",\"\$1}' > {.}.csv"
```
- **What it demonstrates**: full remote file-processing flavor — transfer the `jq` binary once (`--basefile`), transfer each `.json.gz` input and return its `.csv` output while cleaning up both afterward (`--trc {.}.csv`), with escaped `awk`/`jq` quoting inside the remote command string.

## Reference Tables
| Loop target | Native shell construct |
|---|---|
| Numbers | `for i in {start..end..step}; do ...; done` |
| Lines (file/stdin) | `while read line; do ...; done < file` |
| Files | `for f in /path/*; do ...; done` (glob) or `find ... -exec`/piped to `parallel` |

| Distributed flavor | Key option |
|---|---|
| Run same command everywhere | `--nonall --slf hostnames <cmd>` |
| Distribute data in chunks | `--pipe -N<n> --slf hostnames <cmd>` |
| Transfer file, run, retrieve result | `--basefile <file> --trc {.}<ext> --slf hostnames <cmd>` |

| `--jobs` value | Meaning |
|---|---|
| `N` | exactly N concurrent jobs |
| `+N` | cores + N |
| `-N` | cores − N |
| `N%` | percentage of cores (default 100%) |
| `1` | serial (rate-limited APIs) |
| `0` | unlimited (discouraged) |

## Worked Example
NYC 311 service-call counts by borough, computed via distributed file processing:
```bash
$ seq 0 100 900 | parallel "curl -sL '...erm2-nwe9.json?\$limit=100&\$offset={}' | jq -c '.[]' | gzip > nyc-{#}.json.gz"  # fetch 10 chunks in parallel
$ ls *.json.gz | parallel -v --basefile jq --trc {.}.csv --slf hostnames \
    "zcat {} | ./jq -r '.borough' | tr '[A-Z] ' '[a-z]_' | sort | uniq -c | awk '{print \$2\",\"\$1}' > {.}.csv"
$ cat nyc*csv | header -a borough,count | \
    csvsql --query 'SELECT borough, SUM(count) AS count FROM stdin GROUP BY borough ORDER BY count DESC' | csvlook
```
This fetches paginated API data in parallel, distributes per-file counting work to remote machines (transferring the `jq` binary once and retrieving per-file CSV results), then aggregates the partial results locally with `csvsql` — demonstrating all three distributed-processing flavors in one realistic pipeline.

## Key Takeaways
1. Use native shell loops (for/while + globbing/`find`) for simple serial repetition; reach for `parallel` once speed, concurrency control, or distribution matters.
2. GNU Parallel works with any existing, unmodified command-line tool — no special API needed.
3. Always use explicit placeholders (`{}`, `{1}`, `{/}`, `{.}`, `{#}`) rather than relying on implicit argument appending, for clarity and control.
4. Bound concurrency deliberately with `--jobs`; avoid `-j0` in production use.
5. Use `--dryrun` to verify command construction before a real (possibly expensive/destructive) run.
6. `--keep-order` and `--tag` restore traceability that naive backgrounding loses.
7. Distributing to remote machines needs only SSH access — `parallel` itself doesn't need to be installed remotely, though it helps for core-count detection; otherwise set `--jobs` or per-host core counts explicitly.
8. When a remote pipeline's quoting gets too complex, extract it into its own uploadable script/tool (as with `add`/`pbc`).

## Connects To
- **Ch 4 (Creating Command-Line Tools)**: `pbc` is a direct application of Ch 4's six-step tool-creation process, wrapping `parallel`+`bc` into a reusable parallel tool.
- **Ch 6 (Make)**: Make's dependency graph and `parallel`'s job distribution are complementary — independent Make targets are natural parallelization candidates.
- **Ch 3 (Obtaining Data)**: the NYC 311 API example reuses curl/jq-based API fetching from Ch 3, scaled up via `parallel`.
- **Ch 5 (Scrubbing Data)**: the aggregation step reuses `csvsql`/`header`/`rush run` from Ch 5 and Ch 7.
- **AWS CLI (`aws`), SSH**: external infrastructure tools underpinning the distributed-processing section.
