# Chapter 2: Selecting Your API Architecture

## Core Idea
Choosing an API architectural style (REST vs. GraphQL vs. gRPC) is a consumer-centric decision; REST wins for this project due to industry dominance and broad tooling support, and the chapter also establishes the terminology (API/version/endpoint) and technology architecture (separate API app + read replica) used for the rest of Part I.

## Frameworks Introduced
- **API Architectural Style Selection**: pick a style based on consumer familiarity/tooling support, not technical novelty.
  - When to use: at project kickoff, before any endpoint design.
  - How: survey what your target consumers already use/understand (Postman State of the API Report: REST 86%, Webhooks 36%, GraphQL 29%, SOAP 26%, WebSockets 25%, gRPC 11%); default to REST unless a specific need (flexible queries → GraphQL; ultra-low-latency streaming between internal microservices → gRPC) justifies otherwise.
- **API Hierarchy Model**: `api → version → endpoint`.
  - When to use: structuring and naming any REST API.
  - How: group related endpoints under one API (e.g., "SWC Fantasy Football API"); group stable endpoint sets under a version (e.g., `v1`) so consumers can rely on them; define each endpoint as an HTTP verb + URL path pair.
- **Read-Replica API Architecture**: decouple the API from the production web app's database.
  - When to use: building a read-only API on top of an existing production application without degrading its performance.
  - How: stand up a separate API application that reads from a read-only replica database (continuously synced from the primary) rather than the live production database directly.

## Key Concepts
- **REST (Representational State Transfer)**: client/server, stateless architecture where producers expose addressable resources (e.g., `/products`) and consumers act on them via standard HTTP verbs; response structure is producer-defined; formalized by Roy Fielding's dissertation; "Pragmatic REST"/"RESTful" describes real-world looser implementations.
- **Statelessness**: each request/response stands alone; any needed context from a prior response must be explicitly resent (e.g., passing a player ID retrieved earlier).
- **GraphQL**: a query language + runtime (Facebook, open-sourced 2015); single endpoint (`/graphql`) via POST; consumer defines response shape/contents; reduces number of round trips; versioning not recommended since consumers already control response shape.
- **gRPC**: Google's RPC framework (open-sourced 2015) for fast microservice-to-microservice communication; uses protocol buffers (compact binary serialization) instead of JSON, a `.proto` file instead of OpenAPI spec, and supports continuous streaming, not just stateless request/response; relevant to data science via LLM-serving infrastructure.
- **API endpoint / operation**: an HTTP verb + URL path combination; each combination can only be used for one action.
- **API version**: a set of endpoints kept stable over time so consumers can depend on them.
- **GitHub Codespaces**: cloud-hosted VS Code dev container used as the book's development environment.
- **Fork vs. clone**: forking keeps a link back to the source repo (for contributing back); cloning makes a fully independent copy (preferred here, for a standalone portfolio).
- **.gitignore**: file listing patterns/paths excluded from version control (generated/config/sensitive files).

## Mental Models
- **Consumer-centric architecture choice**: don't pick the most interesting technology — pick what your actual consumers (data scientists, advice sites) already know how to use.
- **Architecture is not permanent**: current/future state diagrams (Figures 2-1, 2-2) show architecture evolving incrementally; today's decisions can be revisited later.
- **Separation of concerns via replica**: isolating API read traffic from the OLTP database protects the core web app's performance while keeping data near-real-time.

## Anti-patterns
- **Building the API directly against the live production database**: keeps data perfectly fresh but risks degrading the production web app under API load.
- **Over-engineering with gRPC/GraphQL prematurely**: adds complexity (protocol buffers, flexible query engines) the current user stories don't need; "you are safe to wait until your users request it" (re: GraphQL).
- **Committing everything in the dev container**: without a `.gitignore`, local/generated/sensitive config files get pushed to the repo.
- **Infrequent commits**: risks losing work and reduces the credibility signal of a consistent GitHub activity history.

## Code Examples
```bash
$ python3 --version
Python 3.10.13
```
- **What it demonstrates**: verifying the Codespace's preloaded Python version meets the book's minimum requirement (3.10+).

```markdown
# API Portfolio Project
This project demonstrates API coding best practices using Python and FastAPI.

This project was built from examples from the book
[Hands-On APIs For API and Data Science](https://hands-on-api-book.com).
```
- **What it demonstrates**: editing the repository's `README.md` (Markdown) as the first tracked change, used to walk through the stage/commit/sync Git workflow in Codespaces.

## Reference Tables
**Table 2-1. Example endpoints in Acme API version 1** (api.acme.com/v1)

| Action | Verb | URL | Body |
|---|---|---|---|
| Read product list | GET | /products/ | Empty |
| Read individual product | GET | /products/{product_id}/ | Empty |
| Create new product | POST | /products/ | Contains new product |
| Update existing product | PUT | /products/{product_id}/ | Contains updated product |
| Delete existing product | DELETE | /products/{product_id}/ | Varies |

**REST vs. GraphQL vs. gRPC comparison**

| Aspect | REST | GraphQL | gRPC |
|---|---|---|---|
| Model | Client/server, resource-based | Client/server, query-based | Remote procedure calls |
| State | Stateless | Stateless | Can stream continuously |
| Response format | JSON (or XML) | JSON | Protocol buffers |
| Response shape defined by | Producer | Consumer | Service contract (.proto) |
| Spec format | OpenAPI Specification | GraphQL schema | .proto file |
| Endpoint count | Many (per resource) | One (e.g., /graphql) | Service methods |
| Versioning | Recommended | Not recommended | N/A |
| Typical use here | Chosen for this project | Candidate if users need flexible queries later | Used for LLM-serving infra, not this project |

## Worked Example
SWC's current architecture is a monolithic web app with its own relational database serving fantasy managers directly. The future-state architecture adds: a read replica database (continuously synced, read-only copy of the website DB) and a new, separate API application hosted at `api.sportsworldcentral.com` that reads only from the replica. New consumers (fantasy advice websites, data science users) call this API instead of touching the production database, so API traffic can scale independently without risking the main site's performance. The chapter then walks through the concrete dev setup: create a GitHub account, clone (not fork) the `api-book-part-one` repo into a new `portfolio-project` repo, launch a GitHub Codespace (cloud VS Code), verify Python 3.10+, edit `README.md`, and perform the add-message → stage → commit → sync Git workflow for the first commit.

## Key Takeaways
1. Pick an API architectural style based on what your consumers already understand and support — REST's ~86% adoption makes it the safe default.
2. Reserve GraphQL for cases needing consumer-defined flexible queries and fewer round trips; reserve gRPC for high-performance internal microservice/streaming use cases (e.g., LLM infrastructure), not typical data-science-facing APIs.
3. Structure your API as `api → version → endpoint`, where each endpoint is a unique HTTP verb + URL combination.
4. Decouple a read-only API from your production database using a read replica to protect the main application's performance.
5. Treat architecture as evolvable, not permanent — start simple and adjust as real demand (e.g., for GraphQL) appears.
6. Use GitHub Codespaces for a consistent, shareable dev environment, and commit frequently with clear messages to build a credible portfolio history.

## Connects To
- **Ch 1**: Directly implements the user stories and consumer list identified there into concrete architecture choices.
- **Ch 3**: Next step — building the actual database (SQLite + SQLAlemy) that this architecture's read replica concept will sit on top of.
- **Ch 4**: The FastAPI application built in this architecture's "separate API app" slot.
- **External**: Roy Fielding's dissertation ("Architectural Styles and the Design of Network-Based Software Architectures"); *Mastering API Architecture* (Gough, Bryant, Auburn); Postman 2023 State of the API Report.
