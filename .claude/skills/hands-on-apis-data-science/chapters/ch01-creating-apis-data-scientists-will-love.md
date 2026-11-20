# Chapter 1: Creating APIs That Data Scientists Will Love

## Core Idea
Data scientists are a distinct API consumer segment with specific needs (JSON, SDKs, standard IDs, strict data types, bulk downloads, delta queries); successful APIs are built from real user research and user stories, not built blindly and hoped into use.

## Frameworks Introduced
- **Design Thinking Triad (for picking API products)**: evaluate every candidate API/feature against three criteria simultaneously.
  - When to use: deciding which API products to build first out of many user requests.
  - How: score each candidate for (1) User desirability — do users actually want it; (2) Technical feasibility — can your team/tech build it; (3) Economic viability — is it worth the investment. Only pursue items that clear all three.
- **User Story Template**: structured format to capture requirements from the end user's perspective.
  - When to use: translating user research into concrete, buildable requirements before any API design starts.
  - How: "As a (user type), I want to (goal/intent), so that (motivation/benefit)."

## Key Concepts
- **Web API**: a software program called over HTTP to retrieve data or execute commands; has a producer (hosts it) and consumers (call it).
- **API communication**: the request/response cycle — consumer sends a request, API returns a response.
- **Data pipeline**: a scheduled sequence of software tasks pulling from multiple sources, cleaning/reformatting data for downstream use.
- **Delta query (last-changed-date query)**: an API parameter allowing consumers to fetch only records changed since a given date, enabling incremental pipeline updates after an initial full load.
- **Bulk download**: a dedicated mechanism (e.g., CSV/Parquet export) for retrieving an entire dataset at once, avoiding timeouts/memory issues from paginated API calls.
- **Pain point**: a specific problem a user type faces attempting a task — the diagnostic unit of user research.
- **SDK (Software Development Kit)**: a published library (installable via pip/conda) that wraps raw API calls, enforcing good usage patterns for consumers.
- **OpenAPI Specification (OAS)**: a formal contract defining an API's data types/structure, which data returned must conform to.

## Mental Models
- **"Field of Dreams" anti-pattern vs. demand-driven APIs**: building an API speculatively ("if you build it, they will come") rarely succeeds; successful APIs are reverse-engineered from proven, researched demand (support tickets, scraping traffic, competitor analysis, surveys).
- **Three data-scientist activity buckets** (from the Anaconda State of Data Science Report): cleaning/prep (38%), reporting/visualization (29%), model training/deployment (27%) — API design should explicitly serve all three.
- **Producer vs. consumer duality**: data scientists are both API consumers (pulling data/model inference) and API producers (exposing their own trained models as APIs).

## Anti-patterns
- **Building without user research**: creates APIs that "languish unused and unremembered."
- **Returning XML or non-standard formats**: data scientists' tooling (Python dict/list structures) expects JSON; XML adds friction.
- **No bulk export option**: forces data scientists to page through an API for full-dataset analysis, risking timeouts and memory overflows on both sides.
- **No external/industry-standard identifiers**: prevents data scientists from joining the API's data with other sources.
- **Loose data typing**: an invalid string might render fine on a webpage but silently breaks a numeric field used in calculations/models.

## Code Examples
No code in this chapter (conceptual/product chapter). The chapter sets up the running example used in all later chapters' code.

## Reference Tables
**Table 1-1. Potential consumers for your APIs (reconstructed)**

| User type | Task | Pain point |
|---|---|---|
| Current SWC team managers | Viewing teams/leagues in fantasy advice sites | SWC league host not supported by advice sites; no mobile apps; no API access |
| Advice website provider | Giving advice / analytics across many league hosts | SWC data not accessible through APIs or any other method |
| Data science users | Creating analytics products | Can't access SWC data reliably |
| Mobile app developers | Creating third-party apps using SWC data | SWC not available for use by mobile apps |
| Generative AI developers | Creating chat agents / LLM apps using SWC data | No easy access to current SWC data |

**Six design tips for APIs data scientists will love:**
1. Always return JSON (not XML).
2. Provide an SDK to consume the API.
3. Provide standard external identifiers for joining data.
4. Conform strictly to data type definitions (e.g., via OAS).
5. Provide a bulk download method (CSV, Parquet).
6. Support querying by last-changed date (delta queries).

**Signs an organization needs an API:**
- Wants to extend reach of a core product to a broader audience.
- Has an existing system that partners need access to.
- Seeing web scraping / reverse-engineered API usage — a demand signal.
- Has valuable data/analytics/metrics to share.
- Has statistical/ML models to share.
- Has generative AI models to share with app builders.

## Worked Example
SportsWorldCentral (SWC), a fictional fantasy football/soccer host, notices rising web-scraper traffic and a backlog of support tickets from (a) advice sites unable to import SWC leagues, (b) data scientists wanting direct data access, and (c) mobile developers wanting to build apps. The product owner runs surveys, interviews top advice sites (illustrated via an interview with Kyle Borgognoni of The Fantasy Footballers), and does competitive analysis of other fantasy hosts. This produces four user stories (team manager, advice provider, data science user, generative AI developer), filtered through the desirability/feasibility/viability triad down to a buildable initial scope: an API serving data science users and advice websites, which becomes the book's running portfolio project built out in Chapters 2–7.

## Key Takeaways
1. Treat data scientists as a first-class API persona with needs distinct from typical web/app consumers.
2. Six concrete design rules make an API "data-scientist-friendly": JSON, SDK, standard IDs, strict types, bulk export, delta queries.
3. Never build an API speculatively — validate demand through tickets, scraping patterns, surveys, and competitive research first.
4. Use the desirability/feasibility/viability triad to prioritize which API to build first.
5. Capture requirements as user stories before architecture or code begins.

## Connects To
- **Ch 2**: Takes these user stories and turns them into concrete API architecture decisions (REST, tech stack).
- **Ch 9**: Demonstrates APIs used in exploratory data analysis (one of the three data-scientist activity buckets).
- **Ch 10**: Demonstrates APIs in data pipelines, using the delta/last-changed-date design tip directly.
- **Ch 13**: Demonstrates deploying an ML model as an API — the "data scientist as API producer" side.
- **External**: Anaconda State of Data Science Report; design thinking (IDEO Field Guide to Human-Centered Design); *APIs: A Strategy Guide* (Jacobson/Brail/Woods); *Continuous API Management* (Medjaoui et al.).
