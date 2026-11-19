# Chapter 16: GenAI for Extracting Text from Images

## Core Idea
There are three distinct architectural approaches to GenAI-based OCR — wrapping a standalone OCR engine (EasyOCR) within a GenAI pipeline (modular, interpretable), using an OCR foundation model (Mistral OCR — end-to-end image/PDF-to-structured-text), or fine-tuning a multimodal LLM for joint visual+text reasoning (LLaVA, Llama 3.2 Vision) — and each trades modularity against integration/accuracy differently.

## Frameworks Introduced
- **Three GenAI-OCR approaches** (the chapter's central framework):
  1. **Wrapping a traditional OCR tool in a GenAI pipeline** (this chapter's primary focus): EasyOCR extracts raw text → separate LLM (via Ollama) interprets/transforms it. Modular, interpretable, independently tunable/replaceable components; ideal for local deployment and language control.
  2. **OCR foundation model** (e.g., Mistral OCR, Google TrOCR): Transformer trained end-to-end on visual+textual data, ingesting pixels directly and outputting recognized/structured text without an external OCR engine. Higher integration, better on noisy/unstructured inputs (handwriting, scans), but less modular.
  3. **Fine-tuned multimodal LLM for OCR** (e.g., LLaVA, Llama 3.2 Vision): Joint visual-text reasoning model specifically tuned on OCR datasets — preserves contextual reasoning about surrounding visual elements (diagrams, tables, UI) while improving extraction accuracy in a single inference step.
  - When to use: Approach 1 for local/offline/modular/interpretable needs; Approach 2 for challenging/noisy documents needing enterprise-grade accuracy (Mistral OCR claims 97-99% on tables/equations/multilingual); Approach 3 for unified single-step extraction + contextual reasoning.
- **OCR-to-shopping-assistant pipeline**: Image (product photo) → EasyOCR text extraction → LLM converts raw text to a natural search query → construct e-commerce search URLs (Amazon/Flipkart/eBay) → scrape snippets → LLM summarizes cross-platform results → present comparison to user.
- **Mistral OCR's three-layer document intelligence stack**: Basic OCR (layout-preserving text+structure extraction to Markdown, with bounding boxes/metadata) → Annotations (`bbox_annotation` for region-targeted captions, `document_annotation` for schema-driven JSON extraction) → Document QA (LLM answers natural-language questions grounded in extracted structure).

## Key Concepts
- **EasyOCR**: Lightweight, open-source, deep-learning-based OCR library; `easyocr.Reader(['en'], gpu=False)` + `reader.readtext(image_path, detail=0)` returns plain text strings from an image.
- **Search query generation via LLM**: Converting raw, noisy OCR text (e.g., "boAt 3.5mm Wired Headphones ₹799") into a natural, user-style search phrase (e.g., "wired headphones with mic under 800") — bridges unstructured visual content to search-ready intent.
- **Redirect-only URL construction**: Building direct e-commerce search URLs (`urllib.parse.quote_plus` + platform-specific templates) instead of calling paid APIs — API-independent, robust to platform changes, fast to deploy.
- **bbox_annotation vs. document_annotation**: Region-specific captioning (bbox) vs. whole-document schema-driven structured extraction (document) — two distinct Mistral OCR annotation modes for different granularities of structured output.
- **Document QA**: LLM-answered natural language questions grounded in OCR-extracted text+layout (e.g., "what is the total amount due?") — moves beyond raw extraction into document understanding/analysis.
- **Multimodal receipt/invoice extraction**: Using a vision-enabled LLM (Llama 3.2 Vision via Ollama) to directly extract all data from a receipt image, then chaining with a text LLM (LangChain + ChatOllama) to extract structured JSON fields, parse with regex, and convert to a pandas DataFrame for analysis.

## Mental Models
- Think of the three OCR approaches as points on a modularity-vs-integration spectrum: wrapped OCR (most modular, most interpretable) ↔ OCR foundation model (balanced, best for noisy inputs) ↔ fine-tuned multimodal LLM (most integrated, single inference step, least modular).
- Treat the shopping-assistant pipeline as "OCR → intent → search → summarize" — a four-stage template reusable for any "visual input → actionable insight" problem, not just shopping.
- Document QA over OCR output is "RAG applied to a single document's extracted structure" — the same retrieval-then-generate pattern from Chapter 1, now scoped to one document's content rather than a corpus.
- Redirect-URL construction (rather than API calls) is a pragmatic "simulate the user's own browsing" design choice — trading real-time structured data for zero-dependency robustness.

## Anti-patterns
- **Using a fine-tuned multimodal LLM when a simple wrapped-OCR pipeline would suffice**: adds unnecessary training/deployment complexity for straightforward text extraction tasks where modularity and interpretability matter more than single-step integration.
- **Relying on conventional OCR for highly variable inputs (handwriting, noisy screenshots)**: conventional OCR tools degrade on such inputs — an OCR foundation model or fine-tuned multimodal LLM is better suited.
- **Treating raw OCR text as a usable search query directly**: raw extracted text (currency symbols, inconsistent formatting) needs LLM-based transformation into a natural search phrase and sanitization (currency symbol replacement, etc.) before being usable.
- **Skipping error handling for missing images, empty OCR, or failed LLM output**: the chapter's `main.py` explicitly catches `FileNotFoundError`, `ValueError`, and generic exceptions — without this, OCR pipelines fail silently or crash unpredictably.
- **Combining OCR and structure-extraction into a single LLM call without separating concerns**: the receipt-extraction example explicitly chains a vision model (extract raw text) with a separate text-only LLM (structure into JSON) — conflating both into one step reduces debuggability.

## Code Examples
```python
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(image_path, detail=0)
```
- **What it demonstrates**: Basic EasyOCR text extraction from a product image.

```python
prompt = f"The following product text was extracted from an image:\n\n{ocr_text}..."
response = ollama.chat(model="llama3.2:3b-instruct-fp16", messages=[{"role": "user", "content": prompt}])
# -> "wired headphones with mic under 800"
```
- **What it demonstrates**: LLM-based transformation of raw OCR text into a natural, user-style search query.

```python
import ollama
response = ollama.chat(
    model="llama3.2-vision",
    messages=[{"role": "user", "content": "get all the data from the image", "images": [image_path]}],
)
cleaned_text = response['message']['content'].strip()
```
- **What it demonstrates**: Direct multimodal extraction from a receipt/invoice image using a vision-enabled Llama 3.2 model via Ollama — no separate OCR engine needed.

```python
from mistralai.client import Mistral
uploaded_file = client.files.upload(file={"file_name": "doc.pdf", "content": f}, purpose="ocr")
signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
messages = [ChatMessage(role="user", content=[
    {"type": "text", "text": "Summarize post-school education growth over the years."},
    {"type": "document_url", "document_url": signed_url.url}
])]
response = client.chat.complete(model="mistral-small-latest", messages=messages)
```
- **What it demonstrates**: Mistral OCR Document QA — uploading a PDF and asking a grounded natural-language question about its content in one API call.

```python
chain = (prompt | llm | StrOutputParser())
result = chain.invoke({"response": response})
json_match = re.search(r"```\n(.*?)\n```", result, re.DOTALL)
parsed_data = json.loads(receipt_data)
items_df = pd.DataFrame(receipt_dict['Items'])
```
- **What it demonstrates**: Chaining vision-model output through a text LLM + JSON extraction + pandas conversion for structured receipt analytics.

## Reference Tables
| Approach | Mechanism | Modularity | Best for |
|---|---|---|---|
| Wrapped OCR (EasyOCR + LLM) | Separate OCR engine + separate LLM interpretation | High | Local/offline, interpretable, tunable components |
| OCR foundation model (Mistral OCR) | End-to-end transformer trained on visual+text, pixels-to-structured-text | Medium | Noisy/unstructured inputs, enterprise accuracy (97-99%), batch (2000 pages/min) |
| Fine-tuned multimodal LLM (LLaVA, Llama 3.2 Vision) | Joint visual-text reasoning fine-tuned for OCR | Low (unified) | Single-step extraction + contextual reasoning (diagrams, tables, UI) |

| Mistral OCR capability | Function |
|---|---|
| Basic OCR | Layout-preserving extraction to Markdown with bounding boxes/metadata |
| bbox_annotation | Region-targeted captions/descriptions |
| document_annotation | Whole-document schema-driven JSON extraction |
| Document QA | Natural-language Q&A grounded in extracted document structure |

## Worked Example
Shopping-assistant pipeline: a screenshot of a headphone advertisement is placed in `assets/`. `main.py` locates it, calls `extract_text_from_image()` (EasyOCR) to get raw text ("boAt 3.5mm Wired Headphones ₹799"), then `generate_search_query()` (Ollama LLM) converts it to "wired headphones with mic under 800" with sanitization (₹→rs, removing "alternative"). `get_product_listings()` builds redirect URLs for Amazon/Flipkart/eBay via `urllib.parse.quote_plus`. `summarize_product_pages()` fetches each URL's HTML with `requests`+`BeautifulSoup`, extracts visible text from `<a>`/`<div>`/`<span>` tags, and summarizes via a second LLM call — e.g., noting that Flipkart's page emphasized promotional urgency cues rather than substantive product info. The user receives both raw redirect links and LLM-generated cross-platform summaries, entirely from a single product photo.

## Key Takeaways
1. Choose among wrapped-OCR, OCR-foundation-model, or fine-tuned-multimodal-LLM based on your modularity/integration/accuracy trade-off needs — there is no universally best approach.
2. Wrapped OCR (EasyOCR + separate LLM) remains the most practical choice for local, interpretable, independently-tunable pipelines.
3. LLM-based query generation (raw OCR text → natural search phrase) is a small but critical bridge step that unlocks downstream search/retrieval usability.
4. Mistral OCR's three-layer stack (Basic OCR, Annotations, Document QA) turns OCR from passive transcription into an active, structured, queryable document-intelligence system.
5. Vision-enabled LLMs (Llama 3.2 Vision) can extract receipt/invoice data directly from images in one step, then be chained with text LLMs for structured JSON extraction and downstream analytics (pandas DataFrames).
6. Robust error handling (missing files, empty OCR, LLM failures) is essential for production OCR pipelines, which have many more failure points than pure-text pipelines.

## Connects To
- **Ch 2**: Directly implements the "image-to-text" output classification introduced in the multimodal-systems chapter.
- **Ch 7**: The OCR-to-search pipeline parallels the image-to-text bidirectional retrieval pattern, now applied to real-world product images rather than a curated image corpus.
- **Ch 1 & 8**: Document QA (Mistral OCR) is a specialized, single-document instance of the RAG pattern established in Chapter 1 and implemented fully in Chapter 8.
- **Ch 17**: Explicitly previewed as the next chapter — wrapping traditional ML models (e.g., recommendation engines) with GenAI.
