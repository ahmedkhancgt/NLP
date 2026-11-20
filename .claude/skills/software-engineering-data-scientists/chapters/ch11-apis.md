# Chapter 11: APIs

## Core Idea
APIs let two systems communicate over a standard client/server interface — as a data scientist you'll both consume existing APIs (e.g., to fetch public data reproducibly) and build your own (e.g., to expose a model's predictions), and FastAPI + Requests make both sides straightforward in Python.

## Frameworks Introduced
- **Client/server request-response model**: the universal shape of API interaction.
  - When to use: understanding or designing any web API interaction.
  - How: the client sends an HTTP request (method + URL + optional headers/body) to a server; the server processes it and returns a response (status code + headers + body, usually JSON).
- **RESTful endpoint design (GET/POST/PUT/DELETE)**: map operations to standard HTTP methods.
  - When to use: designing your own API's endpoints.
  - How: GET = retrieve data (optionally parameterized via path/query params), POST = send data to create something/trigger a computation and get a result, PUT = update an existing resource, DELETE = remove a resource.
- **Separate business logic from API framework code**: keep the functions that do the real work independent of the web framework wiring.
  - When to use: whenever building an API around existing functionality.
  - How: put core functions (e.g., `fit_trendline`, `process_sdg_data`) in their own module; have the API endpoint functions import and call them — this keeps logic testable, framework-swappable (FastAPI ↔ Flask), and aligned with modular code design (Ch 8).
- **Validate API input with Pydantic**: use typed schemas at the API boundary.
  - When to use: any POST endpoint accepting structured JSON input.
  - How: define a `pydantic.BaseModel` subclass describing expected fields/types; use it as the endpoint function's parameter type — FastAPI automatically parses and validates incoming JSON against it, rejecting malformed input before your logic runs.

## Key Concepts
- **API (Application Programming Interface)**: a standardized way for two systems to exchange data/trigger actions, hiding implementation details behind a consistent interface (abstraction).
- **Client / server**: the server waits for requests and responds; the client initiates the request. The server "shares," the client "asks."
- **HTTP methods (endpoints)**: GET (retrieve), POST (send data to create/compute and get a response), PUT (update existing), DELETE (remove existing) — Table 11-1's four common operations.
- **HTTP status codes**: 2xx = success (e.g., 200 OK); 4xx = client error (you did something wrong, e.g., 404 Not Found for a bad path); 5xx = server error (a bug in the API itself).
- **Headers**: request/response metadata — e.g., authorization credentials (request) or content-type (response, telling you the data format returned).
- **URL/path parameters**: values embedded in the URL path (e.g., `/Goal/{goalCode}/Target/List`) or as `?key=value` query parameters (e.g., `?includechildren=true`) that customize what an endpoint returns.
- **JSON (JavaScript Object Notation)**: the standard lightweight, human-readable, key-value data interchange format most REST APIs use for request/response bodies — structurally similar to Python dicts.
- **REST (REpresentational State Transfer)**: Roy Fielding's 2000 architectural style defining how resources should be represented/manipulated via HTTP methods; the most common API style encountered in practice (vs. older SOAP or newer GraphQL).
- **Requests library**: the standard Python library for *calling* APIs (`requests.get(...)`, `requests.post(url=..., json=...)`); exposes `.status_code`, `.headers`, and `.json()` on the response object.
- **Swagger UI / OpenAPI specification**: the common standard for documenting API endpoints, parameters, and responses — many public APIs expose this, and FastAPI auto-generates it for your own API.
- **FastAPI**: a modern Python web framework (Sebastián Ramírez, 2018) for *building* APIs; easy to use, integrates with modern Python tooling, conforms to OpenAPI, and auto-generates interactive documentation.
- **Uvicorn**: the ASGI web server needed to actually serve/run a FastAPI app (`uvicorn module_name:app`), making endpoints reachable locally or when deployed.
- **`@app.get` / `@app.post` decorators**: FastAPI's way of registering a Python function as an endpoint at a given route; FastAPI auto-converts returned dicts to JSON.
- **Path parameter in FastAPI**: a variable segment of the route (e.g., `/say_hello/{name}`) passed as a function argument; adding type annotations (`name: str`) both documents and validates the input and improves the auto-generated docs.
- **Flask / Django**: alternative frameworks — Flask is older/similar but more verbose than FastAPI; Django is a full-featured web framework considered "overkill" for typical data science API needs.

## Mental Models
- **APIs as reproducibility tools, not just integration tools**: manually downloading CSVs from a web page is tedious and hard to document precisely; calling the same data via its API is automatable, scriptable, and exactly reproducible months later — a direct extension of the book's reproducibility theme (Ch 1, Ch 7).
- **An API as a "standing" version of a function**: rather than manually rerunning a script every time a prediction/calculation is needed, an API-wrapped function sits running, waiting to accept input and return output on demand, at scale (e.g., many times per second) — this is the production-serving model for ML.
- **Framework as a wrapper, not the logic itself**: FastAPI/Flask are "a wrapper around your code" — the actual computation (business logic) should live independently so it can be tested on its own and the wrapping framework can be swapped later with minimal disruption.
- **Type annotations as dual-purpose at the API boundary**: the same annotation (`name: str`, or a Pydantic `BaseModel`) both documents the expected input/output in auto-generated docs *and* gets enforced/validated at runtime — directly connecting Ch 6 (type checking) and Ch 7 (Pydantic validation) to API design.

## Anti-patterns
- **Manually downloading data repeatedly instead of calling the API**: tedious, error-prone, and hard to document precisely enough to reproduce months later.
- **Mixing API-framework code with core logic**: makes it harder to switch frameworks, test logic independently, or follow the "divide code into discrete pieces" principle from Ch 8.
- **Skipping input validation on POST endpoints**: accepting arbitrary unvalidated JSON risks the function failing deep inside your logic rather than failing cleanly and informatively at the boundary — use Pydantic models instead.
- **No error handling in API endpoint code**: the chapter explicitly flags this simple example as missing error handling (Ch 5) that a real API would need.
- **Only testing an API by eyeballing it in a browser**: fine for a quick GET sanity check, but not useful for programmatic reuse or for POST requests — use the Requests library to actually call and consume your own API as other code would.

## Code Examples
```python
# Calling a public GET API with Requests
import requests

response = requests.get("https://unstats.un.org/SDGAPI/v1/sdg/Goal/List")
response.status_code                      # 200
response.headers["content-type"]          # 'application/json; charset=utf-8'
response.json()                           # [{'code': '1', 'title': 'End poverty...'}]
```
- **What it demonstrates**: the standard pattern — make the request, check status, inspect headers, parse JSON — for consuming any REST API.

```python
# GET request with a path parameter and a query/URL parameter
response = requests.get(
    "https://unstats.un.org/sdgapi/v1/sdg/Goal/5/Target/List?includechildren=true"
)
response.json()
# [{..., 'targets': [{'code': '5.1', 'title': 'End all forms of discrimination...'}]}]
```
- **What it demonstrates**: slotting a value into a URL path segment (`{goalCode}` → `5`) and appending an optional query parameter (`?includechildren=true`).

```python
# Minimal FastAPI app with a simple GET endpoint
from fastapi import FastAPI

app = FastAPI()

@app.get("/say_hi/")
def say_hi():
    return {"Hi": "There"}
```
```bash
$ uvicorn chapter_11_api:app
# Uvicorn running on http://127.0.0.1:8000
```
- **What it demonstrates**: declaring an endpoint with `@app.get` and serving it with Uvicorn; FastAPI auto-converts the returned dict to JSON.

```python
# GET endpoint with a typed path parameter
@app.get("/say_hello/{name}")
def say_hello(name: str) -> str:
    return {"Hello": name}
```
- **What it demonstrates**: type annotations on a path parameter both document and (with a type checker/FastAPI's own validation) help ensure correct input, and surface in the auto-generated docs at `/docs#/`.

```python
# Separating business logic (chapter_11_functions.py) from the API
import pandas as pd
from scipy.stats import linregress

def fit_trendline(year_timestamps, data):
    result = linregress(year_timestamps, data)
    slope = round(result.slope, 3)
    r_squared = round(result.rvalue**2, 3)
    return slope, r_squared

def process_sdg_data(input_excel_file, columns_to_drop):
    df = pd.read_excel(input_excel_file)
    df = df.drop(columns_to_drop, axis=1)
    df = df.set_index("GeoAreaName").transpose()
    return df

def country_trendline(country_name):
    df = process_sdg_data("../data/SG_GEN_PARL.xlsx",
        ["Goal", "Target", "Indicator", "SeriesCode", "SeriesDescription",
         "GeoAreaCode", "Reporting Type", "Sex", "Units"])
    timestamps = [int(i) for i in df.index.tolist()]
    country_data = df[country_name].tolist()
    slope, r_squared = fit_trendline(timestamps, country_data)
    return slope, r_squared
```
```python
# chapter_11_api.py: a GET endpoint wired to that logic
@app.get("/country_trendline/{country}")
def calculate_country_trendline(country: str):
    slope, r_squared = country_trendline(country)
    return {"slope": slope, "r_squared": r_squared}
```
- **What it demonstrates**: keeping computation logic in its own module, imported and called by a thin endpoint function.

```python
# POST endpoint with Pydantic input validation
from pydantic import BaseModel
from typing import List

class TrendlineInput(BaseModel):
    timestamps: List[int]
    data: List[float]

@app.post("/fit_trendline/",
         summary="Fit a trendline to any data",
         description="Provide a list of integer timestamps and a list of floats")
def calculate_trendline(trendline_input: TrendlineInput):
    slope, r_squared = fit_trendline(trendline_input.timestamps,
                                     trendline_input.data)
    return {"slope": slope, "r_squared": r_squared}
```
```python
# Calling it with Requests
url = "http://127.0.0.1:8000/fit_trendline/"
json_data = {"timestamps": [2000, 2001, 2002], "data": [0.5, 0.6, 0.7]}
response = requests.post(url=url, json=json_data)
response.json()   # {'slope': 0.292, 'r_squared': 0.869}
```
- **What it demonstrates**: a `BaseModel` enforces that `timestamps` is a list of ints and `data` a list of floats before `fit_trendline` ever runs; `summary`/`description` on the decorator enrich the auto-generated docs; the POST request sends a JSON body rather than URL parameters.

## Reference Tables
| HTTP method | Function |
|---|---|
| GET | Retrieves data |
| POST | Sends data to create something / get a computed response |
| PUT | Updates an existing resource |
| DELETE | Deletes an existing resource |

| Status code range | Meaning |
|---|---|
| 2xx | Success (e.g., 200 OK) |
| 4xx | Client error (e.g., 404 Not Found — bad request from you) |
| 5xx | Server error (bug in the API itself) |

| Tool | Role |
|---|---|
| Requests | Calling/consuming APIs from Python |
| FastAPI | Building APIs in Python (routes, validation, docs) |
| Uvicorn | Serving a FastAPI app (the actual running web server) |
| Pydantic | Validating structured input/output at the API boundary |
| Swagger UI / OpenAPI | Standard for documenting and exploring API endpoints |

## Worked Example
The chapter builds a small API around the `fit_trendline`/`process_sdg_data` functions used throughout the book. A GET endpoint `/country_trendline/{country}` takes a country name as a path parameter, internally calls `country_trendline(country)` (which loads and processes the UN SDG Excel data, then fits a trendline), and returns `{"slope": ..., "r_squared": ...}` as JSON — e.g., visiting `/country_trendline/India` returns `{'slope': 0.292, 'r_squared': 0.869}`. A parallel POST endpoint `/fit_trendline/` accepts arbitrary `timestamps`/`data` lists (validated via a `TrendlineInput` Pydantic model) and returns the same kind of trendline result for any caller-supplied dataset — showing the GET-endpoint pattern (fixed internal data, parameterized by path) versus the POST-endpoint pattern (arbitrary caller-supplied data, validated at the boundary).

## Key Takeaways
1. APIs implement a client/server request-response model, almost always RESTful (GET/POST/PUT/DELETE over HTTP), returning data as JSON with a status code indicating success (2xx), client error (4xx), or server error (5xx).
2. Use the Requests library to call existing APIs reproducibly (vs. manual downloads) — check `.status_code`, inspect `.headers`, and parse with `.json()`.
3. Build your own APIs with FastAPI (served via Uvicorn) to expose your code's functionality to other systems/users on demand, including for serving ML model predictions at scale.
4. Keep core business logic separate from API-framework code — it stays testable, reusable, and framework-independent (Ch 8's modularity principle applied to APIs).
5. Use type annotations on path parameters and Pydantic `BaseModel`s for POST request bodies to validate input and automatically improve generated documentation.
6. FastAPI auto-generates interactive Swagger UI documentation (`/docs#/`), including any `summary`/`description` you add to endpoint decorators.
7. A production-ready API needs error handling (Ch 5) beyond what this introductory example shows.

## Connects To
- **Ch 5**: Error handling, explicitly flagged as missing from this chapter's simple example and needed in real APIs.
- **Ch 6**: Type annotations/type checking, used on path parameters and reinforced by Pydantic.
- **Ch 7**: Pydantic data validation, applied here at the API input boundary.
- **Ch 8**: Modularity/coupling principles, applied to separating business logic from API-framework wiring.
- **Ch 12**: Deployment — the next chapter covers deploying this API to a cloud hosting environment.
- **External**: FastAPI documentation (authorization, form handling, database connections); Flask and Django as alternative frameworks; the OpenAPI specification and Swagger UI.
