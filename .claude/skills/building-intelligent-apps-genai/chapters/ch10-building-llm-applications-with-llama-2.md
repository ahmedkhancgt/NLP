# Chapter 10: Building LLM Applications with Llama 2

## Core Idea
Open-source LLMs like Meta's Llama 2 (and Code Llama) let developers build fully local or low-cost cloud-hosted LLM applications — a Streamlit blog generator, a pair-programming assistant via Together.AI's serverless Code Llama, and LangSmith for tracing/evaluating the resulting LangChain pipelines.

## Frameworks Introduced
- **Open-source LLM adoption drivers**: security/data privacy (no need to share proprietary data with a third party), accessibility/flexibility (self-host on your own hardware, including Small Language Models for lighter hardware), cost-effective fine-tuning (no expensive vendor fine-tuning fees; as few as 50-100 training examples), and community collaboration (multi-provider support, red-teaming).
  - When to use: choosing open-source over proprietary APIs when data sovereignty, cost control, or customization matter more than bleeding-edge raw capability.
- **Quantization**: reducing a model's internal numeric precision to shrink size/compute requirements, trading some accuracy for the ability to run on less powerful hardware (phones, edge devices, laptops).
  - When to use: running Llama 2 locally without a GPU/powerful hardware, as the book does with `llama-2-7b-chat.Q2_K.gguf`.
- **Local Llama 2 app pattern (via `CTransformers` + LangChain `PromptTemplate` + Streamlit)**: load a local quantized `.gguf` model file with `CTransformers`, fill a `PromptTemplate` with user-supplied variables, call the LLM, display result in a Streamlit UI.
  - When to use: fully offline/local LLM apps (no internet required after model download) — the book's blog-generation example.
- **Serverless Code Llama pattern (via Together.AI + `langchain-together`)**: instead of downloading model weights, call a hosted serverless Code Llama instance through an API key, configured via `.env` (`TOGETHER_API_KEY`, `TOGETHER_BASE_URL`), then build `ChatPromptTemplate` → chain → `StrOutputParser` pipelines.
  - When to use: pair-programming/code-assistant use cases where local hosting of a large Code Llama variant is impractical; Together.AI offers $25 free credit.
- **LangSmith observability pattern**: set `LANGCHAIN_PROJECT`, `LANGCHAIN_TRACING_V2=true`, and `LANGCHAIN_API_KEY` in `.env` to automatically trace every LangChain call (prompt, parameters, response, tokens, latency, cost) into a LangSmith project dashboard.
  - When to use: any LangChain-based application before moving from prototype to production, to debug/monitor prompt behavior and catch issues pre-deployment.

## Key Concepts
- **Llama 2 parameter sizes**: 7B, 13B, 70B; pre-trained on ~2 trillion tokens with a 4,096-token context length; training data up-sampled from the most factual sources to reduce hallucination, excluding Meta's own product data and sites with heavy personal information.
- **Code Llama variants**: 7B, 13B, 34B, 70B — specialized for programming languages.
- **Vicuna-13B**: an open-source chatbot trained on ShareGPT conversation data atop Llama 1, reportedly reaching >90% of ChatGPT/Bard quality for ~$300 training cost — cited as evidence of open-source competitiveness.
- **Hugging Face**: the distribution point for Llama 2 model weights (Meta does not host downloads directly); offers both full-precision and quantized model files.
- **CTransformers**: the Python library used to load and run a local quantized `.gguf` Llama 2 model file.
- **LangSmith**: LangChain's companion product (opened to all developers March 2024) for visual tracing/evaluation of LangChain app calls — prompt, response, tokens, latency, cost.
- **Red teamers**: community contributors who probe open-source models for harms and help validate safety mitigations.

## Mental Models
- Think of quantization like image compression — you lose some fidelity but gain major storage/processing savings, enabling the model to run on modest hardware.
- Use the "escalation ladder" of detail-in-prompt vs. hallucination: the book explicitly warns that vague instructions to Code Llama will "most likely hallucinate a response" — more detail in the prompt directly reduces hallucination risk.
- Treat LangSmith as "x-ray vision" into a LangChain app: without it, prompts hidden behind templates and multi-step chains are opaque; with it, every input/output/cost/latency at each step becomes visible for debugging before production.

## Anti-patterns
- **Vague prompts to a code-generation model**: the chapter explicitly notes that insufficiently detailed instructions to Code Llama will likely cause hallucinated, incorrect output — be as detailed as possible.
- **Hard-coding the Together.AI or LangChain API keys in source**: reuses the Chapter 6 lesson — store them in `.env` (`TOGETHER_API_KEY`, `TOGETHER_BASE_URL`, `LANGCHAIN_API_KEY`, etc.).
- **Downloading all available Hugging Face model files for a given Llama 2 variant**: unnecessary — only one file (matching your hardware/quantization choice) is needed to run the model.
- **Deploying a LangChain app to production without tracing/evaluation**: skipping LangSmith integration leaves developers blind to prompt issues, cost, and latency problems until users encounter them.

## Code Examples
```bash
conda create –n blog-gen-env python=3.9 --y
conda activate blog-gen-env
pip install sentence-transformers uvicorn ctransformers langchain python-box streamlit
```

```python
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers

## Prompt template for generating blog
prompt_template = """
You are a {blog_style} who writes blogs about {input_text}.
Your blog should be {no_words} words long.
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["input_text", "no_words", "blog_style"])

## LLM model to generate blog
llm = CTransformers(model="models/llama-2-7b-chat.Q2_K.gguf",
                    model_type='llama',
                    config={'max_new_tokens':256,
                            'temperature':0.01})

## Generate blog
blog = llm(prompt.format(input_text=input_text, no_words=no_words, blog_style=blog_style))
print(blog)
return blog
```

```python
st.set_page_config(
    page_title="Llama 2 Blog Generator",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="collapsed")
st.header("LLama 2 Blog Generator")

input_text = st.text_area("Enter your blog topic here:")

col1,col2=st.columns([5,5])
with col1:
    no_words=st.text_input("Enter number of words in the blog:")
with col2:
    blog_style=st.selectbox("Select blog style",
                            ["Researcher",
                             "Data Scientist",
                             "Journalist",
                             "Student"],
                             index=0)

submit=st.button("Generate Blog")

## Generate final output
if submit:
    st.write(generate_blog(input_text, no_words, blog_style))
```

```bash
%pip install --upgrade --quiet langchain-together --no-warn-script-location
```

```text
# .env
TOGETHER_API_KEY=YOUR_API_KEY
TOGETHER_BASE_URL=https://api.together.xyz

LANGCHAIN_PROJECT=LLama2-Project
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<YOUR LANGCHAIN API KEY>
```
- **What it demonstrates**: a fully local Llama 2 pipeline (quantized `.gguf` file + CTransformers + PromptTemplate + Streamlit), and the environment-variable configuration needed to connect to both Together.AI's serverless Code Llama and LangSmith's tracing service.

## Reference Tables
| Model | Sizes |
|---|---|
| Llama 2 | 7B, 13B, 70B |
| Code Llama | 7B, 13B, 34B, 70B |

| Open-source LLM benefit | Why it matters |
|---|---|
| Security/data privacy | No need to share proprietary data with a third-party provider |
| Accessibility/flexibility | Self-host on private servers or lighter hardware (SLMs) |
| Cost-effective fine-tuning | As few as 50-100 examples; avoids proprietary fine-tuning fees |
| Community collaboration | Multi-provider support, red-teaming, diverse contributions |

| LangSmith capability | Benefit |
|---|---|
| View full sent prompt | Debug otherwise-hidden templated/complex prompts |
| View raw LLM output pre-processing | Understand model behavior before transformation |
| Track tokens/cost/latency | Manage API costs and performance |
| Monitor user interactions | Gather usage metrics |

## Worked Example
**Blog generator** (fully local): create `blog-gen-env` Conda environment (Python 3.9); install `sentence-transformers uvicorn ctransformers langchain python-box streamlit`; download a quantized Llama 2 7B chat model (`llama-2-7b-chat.Q2_K.gguf`) from Hugging Face into a local `models/` folder; build `generate_blog(input_text, no_words, blog_style)` using a `PromptTemplate` ("You are a {blog_style} who writes blogs about {input_text}. Your blog should be {no_words} words long.") fed into a local `CTransformers` Llama 2 instance (`temperature=0.01`); build a Streamlit UI with a topic text area, word-count input, and a blog-style selectbox (Researcher/Data Scientist/Journalist/Student); run with `streamlit run app.py`.

**Pair programming assistant** (serverless, Together.AI): sign up at together.ai for an API key (+$25 free credit); store `TOGETHER_API_KEY`/`TOGETHER_BASE_URL` in `.env`; install `langchain-together`; build two use cases — (1) a technical-documentation writer with system prompt "You are a world class technical writer..." answering "Give me a short and concise explanation of what is a Javascript linked list"; (2) a JavaScript coding assistant with system prompt "You are an expert JavaScript developer..." asked to "Write the complete JavaScript code to calculate the minimum of the list {temp_min}. Provide two alternative JavaScript methods..." — Code Llama correctly returns both a spread-operator (`Math.min(...list)`) and a `reduce()`-based solution with explanations; a third use case feeds Code Llama a buggy JavaScript sort algorithm and it correctly identifies and fixes the inner-loop bug.

**LangSmith tracing**: create a LangSmith account (smith.langchain.com), generate an API key, create a project named "Llama2-Project," add the three `LANGCHAIN_*` variables to `.env`, then re-run the pair-programming exercises — all prompts, responses, token counts, latency, and cost become visible per-call in the LangSmith dashboard, including the exact debugging exchange from the Code Llama bug-fix example.

## Key Takeaways
1. Open-source models let you trade some raw capability for data privacy, cost control, and self-hosting flexibility — evaluate per use case, not as a blanket rule.
2. Quantized model files (e.g., `.gguf`) are the practical way to run Llama 2 locally without powerful/GPU hardware.
3. Serverless hosted inference (Together.AI, DeepInfra, Anyscale) is a middle ground between full local hosting and a fully proprietary API, useful when you want Code Llama's coding specialization without managing weights yourself.
4. Detailed, context-rich prompts are essential for code generation/debugging tasks — vague prompts materially increase hallucination risk.
5. Integrate LangSmith (or similar tracing) before production deployment to catch prompt, cost, and latency issues early.

## Connects To
- **Ch3**: Reuses prompt templates and the "give detail to avoid hallucination" principle from the prompt-engineering chapter.
- **Ch5/Ch7**: Llama 2 was named in both earlier chapters as a RAG-compatible open-source alternative to GPT-4/Gemini; this chapter is its full hands-on treatment.
- **Ch9**: Both chapters build a LangChain + Streamlit application, contrasting a data-viz agent (Ch9) with a local blog-generation LLM app (Ch10).
- **Ch11**: The next chapter's Flowise AI document chatbot offers a no-code alternative to the hand-coded LangChain pipelines built here.
