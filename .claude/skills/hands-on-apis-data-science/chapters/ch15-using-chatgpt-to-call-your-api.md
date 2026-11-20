# Chapter 15: Using ChatGPT to Call Your API

## Core Idea
Achieve the same "AI calling your API" outcome as Chapter 14's LangGraph agent, but with a low-code approach: build a custom GPT in ChatGPT and attach a custom action imported directly from your API's OAS file — no Python agent code required, only the API itself needs to run.

## Frameworks Introduced
- **Custom GPT + Custom Action Pattern**: OpenAI's low-code equivalent of a LangChain toolkit.
  - When to use: giving a conversational AI application API access without writing agent/tool-wrapping code yourself.
  - How: create a custom GPT (name, description, instructions, icon) in ChatGPT's GPT Builder; add a custom action via "Create new action"; import the API's live OAS file by URL (`Import URL`), which auto-populates the action's schema with every path/operation in the spec; optionally prune unwanted endpoints from the imported schema; save, and the model will decide — from the OAS's own `summary`/`description`/`operationId` fields — when and how to call each endpoint.
- **OAS `servers` Field as the Connectivity Contract**: tell GPT (or any OAS-consuming tool) exactly where to send requests.
  - When to use: before importing an OAS file into any external tool that needs to call the live API, not just read its documentation.
  - How: add a `servers` list to the `FastAPI()` constructor (`servers=[{"url": "<public base URL>", "description": "Main server"}]`); this populates both the Swagger UI "Servers" selector and the OAS file's `servers` section, which GPT's action import step reads to know the real endpoint host.
- **Public-Port-Required Architecture**: a custom GPT runs in OpenAI's cloud, not your dev environment, so your API must be reachable from the outside.
  - When to use: any time an external SaaS AI tool (not code running in your own Codespace/notebook) needs to call your API.
  - How: launch the API in Codespaces and make its port public (not just locally forwarded), since ChatGPT's servers — unlike a LangGraph notebook running in the same container — call the API over the public internet.

## Key Concepts
- **GPT Builder / "My GPTs"**: ChatGPT's UI for creating custom, instructable chat applications, available on Plus/Team/Enterprise plans.
- **Custom action**: a GPT's mechanism for calling an external API, defined by importing (or pasting) an OpenAPI Specification; equivalent in role to a LangChain toolkit, but configured declaratively through the OAS rather than through Python classes.
- **Available actions list**: once an OAS is imported, ChatGPT auto-lists every endpoint it found as a usable action; unwanted endpoints can be manually removed from the imported schema to restrict what the GPT can call.
- **Authentication = None**: appropriate here because the SWC API is public and read-only; a real-world action connecting to an authenticated API would configure credentials in this same section.
- **GPT-4o System Card**: OpenAI's model-risk documentation (misinformation, violent speech, etc.) — the same model-card concept introduced in Ch. 12/13, now applied to the model underlying ChatGPT itself.
- **"Always Allow"**: the permission prompt ChatGPT shows before a custom GPT calls an action for the first time in a conversation — a user-facing safeguard against uncontrolled API calls.
- **Knowledge gap (recap)**: demonstrated identically to Chapter 14 — the custom GPT initially cannot answer SWC-specific questions (hallucination risk noted), until the custom action gives it live API access.

## Mental Models
- **Same outcome, two very different implementation costs**: this chapter explicitly contrasts itself with Chapter 14 — LangGraph required writing a Python toolkit (Pydantic input schemas + `BaseTool` subclasses) where the *model* decides tool use based on your hand-written descriptions; ChatGPT needs only the OAS file, where the *custom GPT UI* auto-derives available actions from your API's own existing path descriptions. The "AI-readiness" documentation work from Chapter 5/12 (summaries, descriptions, operationIds) is what makes this low-code path work well.
- **"Low-code" means the OAS does double duty**: the same OAS file written for human developer documentation (Ch. 5) and for an OpenAPI-tools ecosystem (Ch. 12) now directly becomes the AI integration layer with zero extra authoring — rewarding earlier investment in well-described endpoints.
- **The custom GPT is a separate application from ChatGPT itself**: once saved, it appears as its own entry in the sidebar, with its own scoped knowledge/actions — not a general ChatGPT session with a plugin bolted on.

## Anti-patterns
- **Leaving the OAS file without a `servers` entry**: without it, an imported action has no way to know the real base URL to call — Swagger UI's local testing doesn't need this, but an external SaaS tool like ChatGPT does.
- **Forgetting to make the Codespace port public**: a custom GPT calling from OpenAI's infrastructure cannot reach a port that is only locally forwarded within your Codespace session.
- **Assuming GPT's answers are reliable without verification**: the chapter repeats OpenAI's own "ChatGPT can make mistakes. Check important info" warning, and notes developers have found GPTs "struggle to process statistics accurately for APIs that return large amounts of data" — echoing Chapter 12's data-size design guidance.
- **Granting unrestricted endpoint access by default**: all endpoints in the OAS import automatically become available actions; if any shouldn't be AI-callable, they must be manually removed from the imported schema (this API happens to be all public read-only, so none were removed here).

## Code Examples
```python
# main.py — adding servers to the OAS via the FastAPI constructor
app = FastAPI(
    description=api_description,
    title="Sports World Central (SWC) Fantasy Football API",
    version="0.2",
    servers=[
        {"url": "[your base URL from previous step]", "description": "Main server"}
    ]
)
```
- **What it demonstrates**: the one code change this chapter requires — declaring the API's public base URL in the OAS so external tools like a ChatGPT custom action know where to send requests; this also surfaces a "Servers" selector in Swagger UI.

```text
# Observed API calls in the Codespace terminal after chatting with the custom GPT
INFO: 10.240.2.131:0 - "GET / HTTP/1.1" 200 OK
INFO: 10.240.4.161:0 - "GET /v0/leagues/?skip=0&limit=100 HTTP/1.1" 200 OK
```
- **What it demonstrates**: the custom GPT autonomously calling the health-check endpoint first (confirming availability), then the leagues list endpoint (with default pagination) to answer the user's question — mirroring the agent behavior from Chapter 14, achieved here with zero Python tool code.

## Reference Tables
**ChatGPT custom GPT vs. LangGraph agent (Ch. 14 contrast)**

| Aspect | LangGraph (Ch. 14) | Custom GPT (Ch. 15) |
|---|---|---|
| Tool definition | Hand-written Python (`BaseTool` + Pydantic schemas) | Auto-imported from the OAS file |
| What the model reads to decide tool use | Tool `description` fields you write in code | Existing OAS `summary`/`description`/`operationId` fields |
| Restricting available endpoints | Only include wanted tools in the toolkit | Manually delete unwanted paths from imported schema |
| Code required | Significant (toolkit, graph, notebook) | Minimal (one `servers` field added to FastAPI app) |
| Where the AI app runs | Your own Codespace/notebook | OpenAI's infrastructure (requires your API's port to be public) |
| Model used | Anthropic Claude 3.5 Sonnet | OpenAI GPT-4o |

**Custom GPT setup steps**

| Step | Action |
|---|---|
| 1 | Sign up for ChatGPT (Plus/Team/Enterprise for Create GPTs) |
| 2 | Create a GPT (name, instructions, icon) |
| 3 | Launch & publicly expose the SWC API in Codespaces |
| 4 | Add `servers` to the FastAPI OAS |
| 5 | Create a custom action, import the OAS URL |
| 6 | Review/prune available actions, test (e.g., health check) |
| 7 | Save, set sharing to "Only me," chat with the GPT |

## Worked Example
After signing up for ChatGPT and creating a custom GPT ("SWC Fantasy Football Advisor," with an icon and basic instructions, no special capabilities enabled), the chapter first demonstrates the knowledge gap: asking the un-augmented GPT for SWC's league names returns a non-answer, since the platform's data isn't public/training data. The API is then launched in Codespaces with its port made public, and `main.py` is updated to add a `servers` entry pointing at that public URL — visible afterward both in Swagger UI's new "Servers" dropdown and in the raw `openapi.json`. Back in the GPT Builder, a custom action is created with Authentication set to None, and the OAS URL is imported directly, auto-populating the action's schema with every SWC endpoint (health check, leagues, players, teams, etc.) as available actions; the health-check action is tested successfully in-place. After saving (shared "Only me"), chatting with the GPT and re-asking the leagues question triggers a permission prompt ("Always Allow"), after which the GPT calls `GET /` (health check) and `GET /v0/leagues/?skip=0&limit=100`, correctly answering with the five actual SWC league names — confirmed by the Codespace terminal's request logs. The chapter closes Part III (and the book's main portfolio arc) by branching a `learning-branch` backup and reorganizing the repo into `api/`, `langchain/`, and `model-training/` folders with an updated README.

## Key Takeaways
1. A well-documented OAS file (titles, summaries, descriptions, operationIds — built up across Chapters 4, 5, and 12) can become a zero-code AI integration layer when imported directly into a ChatGPT custom action.
2. Adding a `servers` entry to your FastAPI app is the one essential code change needed to make your OAS usable by an external tool calling your API over the public internet.
3. Because a custom GPT runs on OpenAI's infrastructure (not your own dev container), the API it calls must be genuinely publicly reachable, not just locally port-forwarded.
4. Custom GPT actions import *all* OAS endpoints by default; prune the imported schema if some endpoints shouldn't be AI-accessible.
5. The low-code (custom GPT) and code-first (LangGraph) approaches solve the identical problem — an LLM deciding when to call which API endpoint — via different mechanisms with very different implementation costs; choose based on how much control vs. speed you need.
6. Treat GPT outputs with the same skepticism urged throughout Part III: verify important information, and expect difficulty with large/complex datasets.

## Connects To
- **Ch 4/Ch 5**: The FastAPI/OAS documentation work from these chapters is the direct prerequisite enabling this chapter's zero-code integration — reinforcing why good endpoint descriptions matter beyond human readers.
- **Ch 6**: Reuses the "make the Codespace port public" deployment technique first introduced there.
- **Ch 12**: Applies the AI-readiness design tips (clear descriptions, operationIds, limiting result size) directly, now visible in how well (or poorly) the custom GPT uses the API.
- **Ch 14**: Explicitly contrasted throughout as the code-first alternative to this chapter's low-code approach, solving the same "agentic API calling" problem.
- **External**: OpenAI's GPT-4o System Card; OpenAI ChatGPT pricing/plans page; OpenAI privacy portal (opt-out of chat-based model training).
