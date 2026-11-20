# Chapter 9: Getting Data

## Core Idea
Getting data into Python takes many forms — stdin/stdout pipelines, reading text/delimited files, scraping HTML, and consuming JSON/XML APIs (unauthenticated or OAuth-authenticated, like Twitter) — and each has characteristic gotchas the chapter walks through with working code.

## Frameworks Introduced
- **Unix-style pipeline scripts**: scripts that read `sys.stdin` line by line and write to `sys.stdout`, composed with the shell `|` operator into data-processing pipelines.
  - When to use: quick command-line text processing (grep-like filtering, counting, word-frequency) without a full script/notebook.
- **csv module for delimited files**: `csv.reader` (list-of-fields per row) or `csv.DictReader` (dict-of-fields per row, using header names as keys) for comma/tab/colon-delimited files; `csv.writer` for output.
  - When to use: any delimited file — never hand-roll a parser/writer, since edge cases (commas/newlines inside fields) will break a naive `split(",")`/`join(",")` approach.
- **BeautifulSoup + Requests for HTML scraping**: fetch a page with `requests.get(url).text`, parse with `BeautifulSoup(html, 'html5lib')`, then query with `.find`, `.find_all`/`soup(tag)`, attribute access via dict-style `tag['attr']`, and class-based filters.
  - When to use: data not available through a clean API; requires careful, page-specific selection logic and regex validation of extracted URLs.
- **JSON API consumption**: `json.loads(response_text)` deserializes a JSON API response into native Python dicts/lists; the chapter's GitHub example shows the common unauthenticated-API pattern.
  - When to use: any REST API returning JSON — the default serialization format for modern web APIs.
- **OAuth-authenticated streaming API (Twitter/Twython)**: subclass `TwythonStreamer`, override `on_success`/`on_error`, then call `stream.statuses.filter(track=...)` to consume a live firehose sample.
  - When to use: large-volume, real-time data collection from an API requiring full (OAuth 1) authentication, as opposed to simple keyword search.

## Key Concepts
- **stdin/stdout piping**: `sys.stdin` for line-by-line input, `sys.stdout.write` for output; composed via shell `|` into pipelines (`cat file | python egrep.py "[0-9]" | python line_count.py`).
- **Delimited file**: a text file where each line has multiple fields separated by a consistent character (comma, tab, colon); requires a dedicated parser due to escaping edge cases.
- **HTML Tag / soup**: BeautifulSoup's tree representation of parsed HTML; `Tag.text` gets contents, `Tag['attr']`/`Tag.get('attr')` gets attributes.
- **robots.txt**: a site's statement of how/whether it should be scraped and how frequently — polite scrapers respect it (the book jokes that Congress doesn't need such courtesy).
- **API authentication (OAuth 1 vs. OAuth 2)**: OAuth 2 suffices for simple read-only search; OAuth 1 (more complex) is required for actions like tweeting or streaming.
- **API credentials hygiene**: never commit API keys/secrets to a public repo; store in an untracked `credentials.json` or environment variables (`os.environ`).

## Mental Models
- **Never hand-roll a delimited-file parser**: commas/tabs/newlines inside field values will break any naive split/join logic — always use `csv`, `pandas`, or an equivalent library.
- **Scraping is the "last refuge"**: prefer an official API when one exists; scraping is fragile (depends on page structure that can change) and ethically/legally murkier (robots.txt).
- **Regex-validate scraped URLs before trusting them**: the House.gov example shows raw `<a href>` extraction returning 965 URLs, most irrelevant — a carefully tested regex (`^https?://.*\.house\.gov/?$`) plus `set()` deduplication narrows it to the real 431.
- **In-memory collection is a toy-project shortcut**: appending streamed tweets to a global Python list is explicitly flagged as "pretty poor form" — real projects persist to a file or database.

## Anti-patterns
- **Hand-parsing CSV/delimited files with `split(",")`/`join(",")`**: breaks on any field containing the delimiter or embedded newlines — explicitly called out ("Never parse a comma-separated file yourself. You will screw up the edge cases!").
- **Trusting raw scraped link lists without filtering/deduplication**: produces massive, noisy, duplicate-laden URL lists; always validate format (regex) and dedupe (`set`).
- **Committing API keys to source control or publishing them**: a basic but critical credential-hygiene failure; use environment variables or an untracked credentials file.
- **Assuming one page of "press releases" is the complete history**: paginated pages (5-10 items per page) mean naive scraping only captures the most recent items — a thorough scraper must follow pagination.
- **Relying on an in-memory list for streamed data in production**: loses everything on crash/restart — persist to file/DB instead.

## Code Examples
```python
import csv
with open('tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        date, symbol, closing_price = row[0], row[1], float(row[2])
```
- **What it demonstrates**: safe delimited-file parsing via the `csv` module instead of manual string splitting.

```python
from bs4 import BeautifulSoup
import requests
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')
all_urls = [a['href'] for a in soup('a') if a.has_attr('href')]
regex = r"^https?://.*\.house\.gov/?$"
good_urls = list(set(url for url in all_urls if re.match(regex, url)))
```
- **What it demonstrates**: fetching, parsing, regex-filtering, and deduplicating scraped links (House.gov representative sites example).

```python
class MyStreamer(TwythonStreamer):
    def on_success(self, data):
        if data.get('lang') == 'en':
            tweets.append(data)
        if len(tweets) >= 100:
            self.disconnect()
    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()
```
- **What it demonstrates**: consuming Twitter's streaming API via a subclassed handler with success/error callbacks.

## Reference Tables
| Data source | Tool | Notes |
|---|---|---|
| Command-line pipeline | `sys.stdin`/`sys.stdout` | compose with shell `\|` |
| Delimited file | `csv.reader`/`csv.DictReader`/`csv.writer` | never hand-parse |
| HTML page | `requests` + `BeautifulSoup` | regex-validate + dedupe extracted data |
| JSON API | `json.loads` | native dict/list output |
| XML API | BeautifulSoup (XML mode) | less common now |
| OAuth-gated stream (Twitter) | `Twython`/`TwythonStreamer` | OAuth 1 needed for streaming/actions |

## Worked Example
Scraping all 435 US House representatives' websites from house.gov: raw `<a href>` extraction yields 965 URLs; filtering with regex `^https?://.*\.house\.gov/?$` narrows to 862; deduplicating with `set()` narrows to 431 (close to the true 435, accounting for vacant seats). The script then visits each site, finds "press releases" links (often relative, e.g. `/media/press-releases`), and checks each press release's `<p>` tags for the keyword "data" via `paragraph_mentions(text, "data")`, surfacing ~20 representatives whose press releases mention data — illustrating that pagination limits mean only the most recent releases are checked.

## Key Takeaways
1. Never hand-roll delimited-file parsing — use `csv`/`pandas`; the edge cases (embedded delimiters, newlines) will bite you.
2. Prefer an official API (JSON via `json.loads`) over scraping when one exists; scraping is fragile and should be regex-validated and deduplicated.
3. Respect site politeness conventions (robots.txt) when scraping, except in cases like public government data where the book takes a more relaxed stance.
4. Keep API credentials out of source control — environment variables or an untracked credentials file.
5. For high-volume/real-time data (e.g., Twitter firehose), use the provider's streaming API with proper OAuth, and persist results to durable storage rather than an in-memory list.

## Connects To
- **Ch21 (NLP)**: the word-counting pipeline script (`most_common_words.py`) and scraped/streamed text here feed directly into later text-processing techniques.
- **Ch10 (Working with Data)**: the raw CSV/JSON/scraped data acquired here is exactly what gets cleaned and explored in the next chapter.
- **pandas / Scrapy / Kaggle**: external tools/resources the book points to for more robust data import, full-featured scraping, and ready-made datasets respectively.
