# Chapter 11: Building Multimodal GenAI Systems with Voice as Input

## Core Idea
Voice becomes a first-class input/output modality in RAG by inserting STT (speech-to-text) at the entry point and TTS (text-to-speech) at the exit point around an otherwise standard LangGraph-orchestrated RAG pipeline, with LLM-based routing deciding between local vector-store retrieval and live web-search fallback.

## Frameworks Introduced
- **RAG beyond image/text**: Multimodal RAG extended to audio (speech), video, sensor data, tabular inputs, and knowledge graphs — queries can originate from speech (via STT), gesture (pose estimation), spatial data (LiDAR/IoT), or AR/VR interactions, requiring modality-aware retrievers or cross-modal alignment over heterogeneous index structures (embedding DBs, graph DBs, structured warehouses).
- **Voice-enabled RAG pipeline architecture**: User input (keyboard or voice) → STT normalizes voice to text → unified query routing → RAG chatbot checks vector DB → if context found, Mistral LLM generates grounded response; if not found, falls back to live web search (Tavily) for grounding → optional TTS converts final answer back to speech.
  - How: `route_question_and_get_source()` uses an LLM router prompt to classify the query as "vectorstore" (pdf) or "websearch" (web) *before* retrieval begins.
- **LangGraph StateGraph for hybrid retrieval orchestration**: Nodes — `retrieve` (vector search), `web_search` (Tavily fallback), `generate` (LLM answer synthesis, with web-content summarization step), `route_question` (LLM-based source decision), `grade_documents` (placeholder for future scoring), `decide_to_generate` (routes to generate vs. websearch), `grade_generation_v_documents_and_question` (placeholder quality gate with useful/not useful/not supported/max retries outcomes).
- **STT/TTS as bidirectional bridge**: STT (ASR models — attention-based encoder-decoder or conformer architectures) transcribes speech to text at the input; TTS (neural vocoders like WaveNet/HiFi-GAN, or end-to-end models like FastSpeech) synthesizes speech from generated text at the output — together forming a closed voice interaction loop.

## Key Concepts
- **ASR (Automatic Speech Recognition)**: Deep neural STT models handling temporal dependencies, accents, speech rates, acoustic environments — the input gateway to downstream RAG components.
- **Query routing via constrained LLM output**: Router prompt forces binary JSON output (`{"datasource": "websearch"}` or `{"datasource": "vectorstore"}`), classifying by domain cues (e.g., "LangChain, prompt engineering" → vectorstore; "recent events, weather, people" → websearch).
- **safe_json_parse()**: Regex-based fault-tolerant JSON extraction (`re.search(r'{.*?}', text, re.DOTALL)`) from LLM output, defending against malformed/verbose LLM responses before downstream parsing.
- **Grounded generation prompt with explicit refusal instruction**: `rag_prompt` instructs the LLM to use *only* the given context and to respond "I don't know based on the context" if unsure — direct hallucination-prevention mechanism (echoes the Ch3 "Missing content" failure-point fix).
- **SKLearnVectorStore**: Lightweight in-memory vector store (scikit-learn nearest-neighbor backend) suitable for prototyping/development, not production-scale persistent/distributed indexing (contrast with Qdrant/Faiss used elsewhere in the book).
- **Tavily web search fallback**: `search_tavily()` POSTs to Tavily's API, wraps results as LangChain `Document` objects, and is invoked specifically when local vector retrieval is insufficient — provides live knowledge augmentation for time-sensitive queries.
- **Nomic Embeddings (local inference)**: `NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")` — offline, dependency-free embedding generation.

## Mental Models
- Think of STT/TTS as a "modality adapter pair" bracketing an unchanged text-based RAG core — the RAG pipeline itself doesn't need to know whether its input came from a keyboard or a microphone, as long as STT normalizes to text first.
- Treat the router (vectorstore vs. websearch) as a lightweight, LLM-driven dispatcher replacing hard-coded rules — the same "modality/source routing" pattern seen in Chapter 10's modality-based routing, now applied to retrieval *source* rather than retrieval *modality*.
- The grounded-generation prompt's "say I don't know" instruction is the single most load-bearing anti-hallucination lever in a RAG system — pair every generation prompt with an explicit refusal path.
- A placeholder node (`grade_documents`, `grade_generation_v_documents_and_question` hard-coded to "useful") signals where the architecture anticipates future grading sophistication (per Chapter 6) without yet implementing it — a common, honest way to scaffold extensibility.

## Anti-patterns
- **Treating voice input as a separate pipeline from text input**: the right design normalizes voice to text immediately via STT, then reuses the exact same downstream RAG logic — building parallel voice-specific retrieval/generation logic duplicates effort and risks inconsistency.
- **Skipping fault-tolerant JSON parsing for LLM-based routing decisions**: LLMs asked for structured JSON can still emit extra prose or malformed output; without a `safe_json_parse`-style defensive layer, routing logic will crash unpredictably in production.
- **Using an in-memory vector store (SKLearnVectorStore) in production**: fine for prototyping, but lacks persistence/scalability — swap for Qdrant/Faiss/Pinecone before production deployment.
- **Omitting an explicit "I don't know" fallback in the generation prompt**: without it, the LLM may fabricate an answer when retrieved context (local or web) is insufficient.
- **Leaving quality-gating nodes as unconditional pass-throughs in production**: `grade_documents` and `grade_generation_v_documents_and_question` being hard-coded placeholders is acceptable for a teaching example, but must be replaced with real grading logic (Ch6/Ch8 graders) before production use.

## Code Examples
```python
router_instructions = """
You are a router deciding whether a question should be answered using the user's private PDFs or from a web search.
If the question is about LangChain, prompt engineering, or other topics covered in the provided documents -> use 'vectorstore'.
If the question is about recent events, weather, people, locations, or real-world data -> use 'websearch'.
Return ONLY a JSON like:
{ "datasource": "websearch" }
or
{ "datasource": "vectorstore" }
"""

def route_question_and_get_source(question: str) -> str:
    messages = [SystemMessage(content=router_instructions), HumanMessage(content=question)]
    response = llm_json_mode.invoke(messages)
    try:
        result = safe_json_parse(response.content)
        datasource = result.get("datasource", "vectorstore").lower()
        return "web" if datasource == "websearch" else "pdf"
    except Exception as e:
        print(f"[ROUTER] JSON parsing failed: {e}")
        return "pdf"
```
- **What it demonstrates**: LLM-driven, fault-tolerant query routing between local document retrieval and web-search fallback.

```python
rag_prompt = """
You are a helpful assistant. Use ONLY the information in the CONTEXT below to answer the QUESTION.
If the CONTEXT does not contain the answer, respond exactly with:
"I don't know based on the context."
---
CONTEXT:
{context}
---
QUESTION:
{question}
---
INSTRUCTIONS:
- Do NOT use prior knowledge.
- Do NOT make up any answers.
- ONLY use information in the context above.
- Answer in 2-3 concise sentences.
"""
```
- **What it demonstrates**: Hallucination-resistant grounded-generation prompt with explicit refusal instruction.

```python
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("websearch", web_search)
workflow.add_node("grade_documents", grade_documents)
workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
    "useful": END,
    "not useful": "websearch",
    "not supported": "generate",
    "max retries": END
})
graph = workflow.compile()
```
- **What it demonstrates**: LangGraph StateGraph wiring retrieval, generation, web fallback, and quality-gated conditional routing into a single executable workflow.

```python
query = listen_from_microphone()   # STT via SpeechRecognition
...
final_answer = response.content.strip()
speak_text(final_answer)            # TTS via pyttsx3
```
- **What it demonstrates**: The voice-input/voice-output bracket around the standard RAG query/response cycle.

## Reference Tables
| Component | Role |
|---|---|
| LangChain | RAG backbone — prompt templating, document loading, LLM orchestration |
| LangGraph | Graph-based execution model for conditional flows (fallback, retry) |
| Ollama | Hosts local LLMs (Mistral/Llama) for offline inference |
| Streamlit | Web frontend supporting real-time voice and text input |
| Tavily API | Live web search fallback when local context is insufficient |
| Nomic Embeddings | Local document vectorization |
| pyttsx3 | Offline TTS (text-to-speech) |
| SpeechRecognition | Local microphone STT (speech-to-text) |

## Worked Example
A Streamlit app ("Multimodal RAG Chatbot (PDF + Web + Voice)") offers a radio toggle between "Keyboard" and "Voice" input modes. In Keyboard mode, `st.text_input()` captures the query; in Voice mode, `listen_from_microphone()` captures and transcribes speech via SpeechRecognition. Either way, the query flows into `graph.invoke({...})` — the compiled LangGraph workflow that routes to `retrieve` (local SKLearnVectorStore, built from PDFs loaded via `load_pdfs()` + Nomic embeddings) or `websearch` (Tavily API), formats the result into `rag_prompt`, and invokes the local Mistral LLM via `ChatOllama`. The final answer is rendered as text (`st.markdown`) and spoken aloud via `speak_text()` (pyttsx3 TTS), closing the voice-in/voice-out loop. A one-time `run_once.py` script pre-builds the vectorstore from `data/documents/` before the app launches.

## Key Takeaways
1. Voice input/output is best implemented as a thin STT/TTS adapter layer around an unchanged text-based RAG core, not as a parallel pipeline.
2. LLM-based query routing (vectorstore vs. web search) replaces brittle hard-coded rules but requires fault-tolerant JSON parsing to be production-safe.
3. An explicit "I don't know based on the context" instruction in the generation prompt is a simple, high-leverage anti-hallucination technique.
4. LangGraph's StateGraph with conditional edges cleanly expresses retrieval→generation→quality-gate control flow, including retries and fallback to web search.
5. In-memory vector stores (SKLearnVectorStore) are appropriate for prototyping only; production systems need persistent/distributed stores (Qdrant, Faiss, Pinecone).
6. Multimodal RAG extends naturally beyond text/image to speech, gesture, spatial/sensor data, and AR/VR interactions, each needing modality-aware retrieval or cross-modal alignment.

## Connects To
- **Ch 3-4**: Reuses the PDF-loading, chunking, and local-vs-API LLM patterns established in the unimodal local/API-based GenAI chapters, now with voice bracketing.
- **Ch 6**: The LangGraph StateGraph routing/grading pattern here directly extends the grading-mechanism and multi-stage RAG routing framework from Chapter 6.
- **Ch 5**: Builds on LangGraph orchestration patterns and retry logic first introduced in the HITL agentic chapter.
- **Ch 12-13**: Sets up the transition to advanced multimodal GenAI systems, which further extend input/output modality handling.
