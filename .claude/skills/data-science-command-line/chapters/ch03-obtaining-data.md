# Chapter 3: Obtaining Data

## Core Idea
Text is the universal interface of the command line, so obtaining data — from local files, the internet, compressed archives, spreadsheets, databases, or web APIs — is really about converting whatever format the data arrives in into plain text/CSV/JSON that downstream tools can consume.

## Frameworks Introduced
- **"Text as universal interface" (Unix philosophy)**: almost every CLI tool reads/writes text, which is why tools compose.
  - When to use: whenever deciding whether a new data source will play well with the rest of your pipeline.
  - How: convert binary/proprietary formats (xlsx, databases) to CSV/JSON as early as possible (`in2csv`, `sql2csv`).
- **Format-driven tool selection for obtaining data**: pick the tool by the shape of the source, not by habit.
  - When to use: deciding how to get data onto disk in usable form.
  - How: local file → `cp`/drag-drop into `/data`; internet resource → `curl`; compressed archive → `tar`/`unzip`/`unrar`/`unpack`; spreadsheet → `in2csv`; relational DB → `sql2csv`; web API (JSON/XML) → `curl` + `jq`; streaming API → `curl` piped through `sample`/`sed`/`jq` with a manual or timed cutoff.

## Key Concepts
- **curl**: Swiss-army-knife downloader supporting 20+ protocols (HTTP, FTP, DICT, etc.); prints raw, uninterpreted data to stdout.
- **-s (silent)**: suppresses curl's progress meter (sent to stderr) so it doesn't pollute output in a pipeline.
- **-O / -o**: save curl output to a file (auto-named from URL, or explicitly named).
- **-L (--location)**: follow HTTP redirects (needed for shortened URLs); without it you get the redirect page's HTML instead of the target.
- **-I (--head)**: fetch only HTTP headers — useful for inspecting status codes (303, 404, 403) when debugging.
- **CSV's 3-point informal spec** (Shafranovich/RFC 4180): one record per line; last record's trailing newline is optional; optional header row with same field count as data rows.
- **in2csv**: csvkit tool converting Excel/other tabular formats to CSV, format auto-detected from extension; `--sheet`/`--names` select/list worksheets.
- **sql2csv**: csvkit tool executing a SQL query against a database URL (`dialect+driver://user:pass@host:port/db`) and emitting CSV.
- **jq**: JSON processor used here to pretty-print and later filter API responses.
- **unpack**: convenience script that auto-detects archive type by extension and calls the right decompression tool.
- **Command substitution `$(...)`**: used to inject secrets (e.g., API keys) from a file into a command without exposing them in plaintext/history.

## Mental Models
- Obtaining data is itself a mini-pipeline: fetch (curl) → inspect/convert (in2csv, jq, tar -t) → persist (redirection) → verify (wc, cat, csvlook).
- Always inspect an unfamiliar archive (`tar -tzf`) before extracting it (`tar -xzf`), and extract into a fresh subdirectory to avoid clutter.
- Streaming APIs are just pipelines with no natural end — use a tool like `sample -s <secs>` or Ctrl-C to bound them.

## Anti-patterns
- **Manually exporting Excel to CSV via GUI each time**: doesn't scale to many files, can't be automated, and may not be available on a headless server — use `in2csv` instead.
- **Hardcoding API keys/passwords directly in commands**: exposes secrets in shell history/scripts — use command substitution reading from a protected secret file instead.
- **Forgetting `-L` on shortened/redirecting URLs**: silently returns an HTML stub or nothing instead of the real resource.
- **Extracting unknown archives directly into the current directory**: clutters the workspace — `mkdir` + `tar -C` into a dedicated directory first.

## Code Examples
```bash
$ curl -s "https://en.wikipedia.org/wiki/List_of_windmills_in_Friesland" | pup -n 'table.wikitable tr'
234
```
- **What it demonstrates**: silent download piped directly into an HTML-scraping tool (`pup`) to count table rows — obtain+scrub fused in one line.

```bash
$ tar -tzf logs.tar.gz | trim
$ mkdir logs
$ tar -xzf logs.tar.gz -C logs
```
- **What it demonstrates**: inspect-before-extract pattern, and extracting into an isolated subdirectory with `-C`.

```bash
$ sql2csv --db 'sqlite:///r-datasets.db' \
  --query 'SELECT row_names AS car, mpg FROM mtcars ORDER BY mpg' | csvlook
```
- **What it demonstrates**: querying a relational (SQLite) database directly from the shell and rendering the CSV result as a table.

```bash
$ curl -s "http://newsapi.org/v2/everything?q=linux&apiKey=${NEWSAPI_KEY}" | jq '.' | trim 30
```
- **What it demonstrates**: authenticated API call using a shell variable (populated from a locally protected secrets file via command substitution, never typed inline or committed to shell history) to supply the API key.

## Reference Tables
| Source type | Primary tool(s) |
|---|---|
| Local file | `cp`, drag-and-drop into `/data` |
| Internet file/page | `curl` (`-s`, `-O`/`-o`, `-L`, `-I`, `-u`) |
| Compressed archive | `tar`, `unzip`, `unrar`, or `unpack` |
| Excel spreadsheet | `in2csv` (`--sheet`, `--names`) |
| Relational database | `sql2csv --db ... --query ...` |
| Web API (JSON/XML) | `curl` + `jq` |
| Streaming API | `curl` + `sample -s N` (or Ctrl-C) + `sed`/`jq` |

## Worked Example
Sampling and filtering a live streaming API end-to-end:
```bash
$ curl -s "https://stream.wikimedia.org/v2/stream/recentchange" | sample -s 10 > wikimedia-stream-sample
$ < wikimedia-stream-sample sed -n 's/^data: //p' | \
    jq 'select(.type == "edit" and .server_name == "en.wikipedia.org") | .title'
```
This captures 10 seconds of a live Server-Sent-Events stream to a file, then (as a separate, repeatable step) strips the `data: ` SSE prefix with `sed` and filters+projects JSON records with `jq` — obtain once, scrub/explore repeatedly from the saved snapshot.

## Key Takeaways
1. curl is the default entry point for almost any non-local data source, with `-s`, `-L`, `-I`, `-u` as its most-used flags.
2. Convert proprietary formats (Excel, SQL databases) to CSV/JSON immediately with `in2csv`/`sql2csv` rather than relying on GUI exports.
3. Always inspect archive contents (`tar -t...`) before extracting, and extract into a dedicated directory.
4. Save streaming or API data to a file first (so scrubbing/exploring can be repeated without re-fetching).
5. Keep secrets out of commands/history by reading them from files via command substitution.
6. jq is the standard tool for both pretty-printing and later filtering JSON API responses.

## Connects To
- **Ch 5 (Scrubbing Data)**: continues directly on the log files and streamed JSON obtained here (sed/jq filtering).
- **Ch 7 (Exploring Data)**: uses the same obtained datasets (e.g., log files) for statistics/visualization.
- **csvkit**: the suite providing `in2csv`, `sql2csv`, `csvlook`, `csvgrep` used throughout.
- **jq**: central JSON tool reused in Ch 5 for deeper JSON scrubbing.
