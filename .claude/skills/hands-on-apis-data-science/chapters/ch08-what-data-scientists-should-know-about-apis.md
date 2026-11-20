# Chapter 8: What Data Scientists Should Know About APIs

## Core Idea
Data scientists need a building-block skill set for API consumption — handling multiple architectural styles, HTTP mechanics, responsible usage, SoC via SDKs/clients, basic API-building/testing/deployment literacy, and version control — and this chapter pivots the book from API *producer* (Part I) to API *consumer* (Part II), launching the second portfolio project.

## Frameworks Introduced
- **Building-Block Skill Acquisition Model**: don't try to master every API specialty at once.
  - When to use: structuring your own learning as a data scientist working with APIs.
  - How: master one or two API-related skills at a time (e.g., REST basics, then responsible consumption, then SoC/SDK usage), stack additional skills as you grow, and reinforce via hands-on projects shared publicly for fast feedback.
- **Responsible API Consumption Checklist**: a practical discipline for calling others' APIs.
  - When to use: before integrating any external API into a real pipeline/application.
  - How: read and follow the terms of service (rate limits, allowed uses); implement gentle retries (backoff, Ch. 7); handle credentials via secrets managers/environment variables (never in code or repos); validate both inputs sent and outputs received (guard against SQL injection and type mismatches); log and diagnose errors systematically, especially in recurring pipelines.
- **Separation of Concerns (SoC) for API Code**: isolate API-calling logic from business/analysis logic.
  - When to use: any project (even a single Jupyter Notebook) that calls an API, so maintenance and testing stay manageable.
  - How: if the provider publishes an SDK, use it; if not, build your own standalone API client module providing the same kind of validation/retry/logging functionality, and reuse that client across projects (demonstrated later across Ch. 9 and Ch. 11).
- **Four Major API Testing Types (per Postman)**: a taxonomy for what "testing an API" can mean.
  - When to use: scoping a test strategy as either an API producer or consumer.
  - How: contract testing (format/behavior of each endpoint) → unit testing (individual endpoint behavior) → end-to-end testing (multi-endpoint workflows) → load testing (concurrency/performance under peak conditions); complement with the Agile Testing Quadrants (technology-facing vs. business-facing tests) for a fuller picture, including documentation/SDK testing.

## Key Concepts
- **REST recap for consumers**: multiple endpoints, each an HTTP verb + URL; GET for reads; POST/PUT/DELETE with the HTTP body carrying JSON/XML for create/update actions; response shape is fixed by the producer.
- **GraphQL recap for consumers**: single endpoint, queried via HTTP POST with a query body; consumer specifies exactly which fields to return; cannot be called directly from a browser address bar (which defaults to GET).
- **gRPC recap for consumers**: cross-language RPC using protocol buffers over HTTP/2, supporting two-way streaming; data scientists most likely to encounter it calling ML/LLM inference services.
- **HTTP verbs/methods**: GET (read), POST (create), PUT (update), DELETE (remove) — the "purpose" of a request per the HTTP standard.
- **HTTP status code families**: 2XX success, 3XX redirection, 4XX client error, 5XX server error — each with specific common codes consumers must handle distinctly (e.g., retry on 503, don't retry on 400/404).
- **HTTPS-only rule**: only call APIs over HTTPS so traffic is encrypted in transit — the chapter's first HTTP tip.
- **Service Level Agreement (SLA)**: a formal producer-consumer agreement on uptime/performance/other service aspects, relevant to both testing and trust.
- **Containerization (recap)**: packaging code into a portable, runnable unit (Docker), enabling consistent deployment across hosts — tied back to Chapter 6.
- **Version control**: foundational skill (not API-specific) for tracking codebase changes and enabling collaboration; GitHub/Codespaces continue as the book's chosen toolchain into Part II.
- **Inference endpoint**: an API exposing a trained statistical/ML model so others can submit data and receive predictions — previewed here, built in Chapter 13.

## Mental Models
- **Producer vs. consumer perspective shift**: Part I built APIs (producer mindset); Part II consumes APIs (consumer mindset) — many of the same concepts (REST, OAS, SDKs) now viewed from "the other side of the desk."
- **"The field of APIs is nearly as wide as data science itself"**: no one masters everything; competence is built incrementally and practically, not exhaustively.
- **SoC as a discipline against Notebook sprawl**: data scientists' habit of writing everything in one Jupyter Notebook works against maintainability once API calls (with their validation/retry/logging needs) are involved — pulling that logic into a separate client module pays off operationally.

## Anti-patterns
- **Ignoring an API's terms of service**: risks violating rate limits, scraping restrictions, or prohibited uses (e.g., "can't be used to cheat in your fantasy league or harvest user data" per MyFantasyLeague's ToS).
- **Aggressive/immediate retries without backoff**: risks overwhelming the API and having your access disabled (echoes Chapter 7's backoff discussion).
- **Storing credentials in code or committed files**: API keys/tokens must live in secrets managers or environment variables, never in source control; if exposed, deactivate immediately.
- **Skipping input/output validation as a consumer**: leaves you exposed to malformed data and injection-style risks even when you're the one calling (not hosting) the API.
- **Mixing API-calling logic directly into analysis/notebook code with no separation**: makes testing, reuse, and error handling much harder than pulling it into a client/SDK layer.

## Code Examples
```python
# REST GET via httpx
import httpx
rest_url = "https://api.sportsworldcentral.com/v0/players/1491"
api_response = httpx.get(rest_url)
print(api_response.json())
```
- **What it demonstrates**: calling a REST endpoint with a plain GET request using httpx, mirroring what a browser does implicitly.

```python
# GraphQL POST via httpx
import httpx
graphql_url = "https://countries.trevorblades.com"
json_query = {'query': '''
{
  country(code: "US") {
    name
    native
    currency
    languages { code name }
  }
}
'''}
api_response = httpx.post(graphql_url, json=json_query)
print(api_response.json())
```
- **What it demonstrates**: GraphQL always uses POST (not GET), with the consumer specifying exactly the fields wanted; response: `{'data': {'country': {'name': 'United States', ...}}}`.

```python
# Using a published SDK (separation of concerns)
from pyswc import SWCClient
from pyswc import SWCConfig

config = SWCConfig()
client = SWCClient(config)
player_response = client.get_player_by_id(352)
```
- **What it demonstrates**: how little code is needed to call an API responsibly when an SDK handles backoff, validation, error handling, and logging internally (the Chapter 7 SDK applied from the consumer's side).

## Reference Tables
**Table 8-1. HTTP verbs and REST API usage**

| Verb | Use | Example |
|---|---|---|
| GET | Read a resource or list | `GET api.sportsworldcentral.com/players/` |
| POST | Create a new resource | `POST api.sportsworldcentral.com/team/` |
| PUT | Update an existing resource | `PUT api.sportsworldcentral.com/players/1234` |
| DELETE | Remove an existing resource | `DELETE api.sportsworldcentral.com/players/1234` |

**Table 8-2. HTTP status codes**

| Code | Meaning |
|---|---|
| 200 OK | Request successful |
| 201 Created | POST successfully created a resource |
| 301/308 Moved Permanently | Address changed permanently — update your call |
| 302 Found | Temporary redirect — keep using the original address |
| 400 Bad Request | Error or invalid request |
| 401 Unauthorized | Invalid credentials |
| 404 Not Found | Resource/address doesn't exist |
| 500 Internal Server Error | Unexpected server failure |
| 503 Service Unavailable | Temporary issue; retry may be appropriate |

**REST vs. GraphQL vs. gRPC (consumer-facing)**

| Aspect | REST | GraphQL | gRPC |
|---|---|---|---|
| Endpoints | Many | One | Service methods |
| Verb to read | GET | POST (always) | N/A (RPC call) |
| Response shape | Fixed by producer | Chosen by consumer | Defined by .proto |
| Browser-callable (GET) | Yes | No | No |
| Typical DS use case | General API consumption | Flexible field selection | ML/LLM inference |

**Table 8-3. Updated endpoints for SWC API v0.2 (used in Part II)**

| Endpoint | Verb | URL | Update |
|---|---|---|---|
| Read counts | GET | /v0/counts/ | Added week count |
| Read week list | GET | /v0/weeks/ | New endpoint, max potential scoring |
| Read league list | GET | /v0/leagues/ | Added league size |
| Read individual league | GET | /v0/leagues/{league_id} | Added league size |
| Read team list | GET | /v0/teams/ | Added weekly_scores per team |

## Worked Example
The chapter shows the same conceptual query ("get info about player 1491 / country US") across REST (simple `httpx.get`) and GraphQL (`httpx.post` with a query body returning only requested fields), contrasting response shape control. It then walks through responsible-consumption practices using MyFantasyLeague's real terms of service as an example (free API, no cheating/harvesting, caching slow-changing data to reduce load). It introduces the SoC principle with a worked comparison: calling the Chapter 7 `swcpy` SDK (three lines of code, backoff/validation/logging all handled) versus hand-rolling an API client (which Chapter 9 will build and reuse in Chapter 11). Finally, it sets up the Part II portfolio project: clone the `api-book-part-two` repo as `analytics-project`, launch a Codespace, and run SWC API v0.2 (`cd api && pip3 install -r requirements.txt && fastapi run main.py`), confirming the health check and making the Codespace port public for use across the Jupyter (Ch. 9), Airflow (Ch. 10), and Streamlit (Ch. 11) projects that follow.

## Key Takeaways
1. Treat API expertise as an incrementally built skill set, not an all-or-nothing mastery goal — stack building blocks over time via hands-on projects.
2. Be fluent in calling REST (GET for reads, verb+URL per action), GraphQL (always POST, consumer-chosen fields), and recognize gRPC's role in ML/LLM inference contexts.
3. Know your HTTP verbs and status code families (2XX/3XX/4XX/5XX) well enough to branch program logic on them correctly (e.g., retry on 5XX, don't retry on 4XX).
4. Consume APIs responsibly: respect terms of service and rate limits, retry gently with backoff, secure credentials outside of code, validate data both ways, and log systematically.
5. Apply separation of concerns — use a provided SDK when available, or build your own reusable API client — rather than embedding raw API calls throughout analysis code.
6. Even as primarily a consumer, basic literacy in building, testing (contract/unit/end-to-end/load), and deploying (containerized, cloud-hosted) APIs deepens your effectiveness as a user of others' APIs.
7. Version control (Git/GitHub) remains a foundational, non-API-specific skill underlying all of this work.

## Connects To
- **Ch 2**: Recaps and reapplies the REST/GraphQL/gRPC architecture discussion from a consumer's perspective.
- **Ch 6**: Containerization/deployment concepts reapplied here as consumer-relevant literacy.
- **Ch 7**: The backoff/retry and SDK patterns built there are directly invoked as responsible-consumption and SoC techniques.
- **Ch 9**: Builds the hand-rolled API client this chapter previews, used in Jupyter Notebook analytics.
- **Ch 11**: Reuses that same Chapter 9 client in a Streamlit data app, demonstrating the SoC payoff.
- **Ch 13**: Builds the "inference endpoint" concept introduced here into a deployed ML model API.
- **External**: Postman's API testing types guide; Janet Gregory & Lisa Crispin's Agile Testing Quadrants ("Quick Tools for Agile Testing"); Google's "Best practices for securely using API keys"; *Software Engineering for Data Scientists* (Catherine Nelson); Locust (load-testing tool).
