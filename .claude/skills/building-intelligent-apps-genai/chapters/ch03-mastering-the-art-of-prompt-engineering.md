# Chapter 3: Mastering the Art of Prompt Engineering

## Core Idea
Effective prompt engineering rests on two guidelines — write clear and specific instructions, and give the model time to think — implemented through concrete, repeatable techniques (delimiters, structured output, few-shot examples, step-by-step reasoning, personas, templates) refined through a 7-step iterative process.

## Frameworks Introduced
- **Two core prompting guidelines**: (1) write clear and specific instructions, (2) give the model time to think.
  - When to use: as the first checklist applied to any new prompt before trying advanced techniques.
- **Delimiters**: wrap distinct parts of a prompt in triple quotes `"""`, triple backticks ` ``` `, triple dashes `---`, angle brackets `<>`, or XML tags `<tag></tag>`.
  - When to use: separating instructions from user-supplied content; the primary defense against prompt injection.
  - How: `prompt = f"""Summarize the text delimited by triple backticks into a single sentence. ```{text}``` """`.
- **Structured output requests**: explicitly ask the model to return JSON/HTML so output is machine-parseable.
  - When to use: whenever the response needs to feed downstream code.
- **Condition-checking**: instruct the model to check whether a precondition holds (e.g., "if the text contains a sequence of steps... otherwise output 'No steps provided'") before proceeding.
  - When to use: guarding against the model inventing structure that isn't actually present in the input.
- **Few-shot prompting**: give the model a handful of worked examples demonstrating the desired input→output pattern before the real request.
  - When to use: scarce/costly training data, or when you need a consistent voice/tone/format.
- **Chain-of-Thought (CoT) / "give the model time to think"**: break a complex task into smaller sequential steps (summarize → translate → list names → output JSON) rather than asking for the end result directly.
  - When to use: multi-step reasoning tasks prone to premature/incorrect conclusions.
- **"Work out your own solution first"**: ask the model to independently solve a problem before evaluating a given (e.g., student's) answer, preventing anchoring bias.
  - When to use: grading/verification tasks (math problems, code review) where the model might otherwise rubber-stamp a wrong answer.
- **Persona prompting**: assign a `system` role message (e.g., "You are a helpful personal assistant...") to set tone/context before the `user` message.
  - When to use: aligning tone/behavior with a defined role or brand voice.
- **Prompt templates (LangChain)**: a reusable structure with placeholders — Instructions, Context, User input, Output indicator — filled at runtime; supports partial templates when some variables arrive later than others.
  - When to use: production LLM apps with many input variables or business logic embedded in prompts.
  - How: `PromptTemplate(template=prompt_template, input_variables=[...])`; `prompt.partial(brand="...")` to pre-fill known variables.
- **7-step iterative prompt-design process**: Define the goal → Write the initial prompt → Test the prompt → Analyze the output → Refine the prompt → Iterate → Implementation.
  - When to use: as the standard lifecycle for developing any production prompt, not a one-shot activity.

## Key Concepts
- **Prompt**: a short piece of text provided to a language model as input to control its output.
- **Prompt injection**: an attack where user-supplied text contains conflicting instructions (e.g., "forget the previous instructions...") intended to hijack the model's behavior; mitigated by delimiters.
- **Hallucination**: the model fabricates plausible-sounding but false information (book's example: a detailed, convincing but entirely fictional product review for a nonexistent "AeroGlide UltraSlim Smart Toothbrush").
- **Prompt engineer**: an emerging, highly paid role (cited: up to $335K, per TIME) requiring no formal CS degree, focused on crafting prompts that yield accurate domain-relevant outputs.
- **Zero-/few-shot learning**: the model performs a task with zero or very few labeled examples, relying on patterns learned during pretraining.

## Mental Models
- Think of giving a model "time to think" the way you'd give a person solving a complex math problem time to work it out — rushing either produces mistakes.
- Use delimiters as punctuation marks that separate "code" (instructions) from "data" (user input) — this is the core defense against prompt injection.
- Treat prompt design as iterative, not a single draft: expect to go through testing → analyzing → refining multiple times before a prompt is production-ready.

## Anti-patterns
- **Short, vague prompts**: the book explicitly notes that while shorter prompts seem efficient, longer prompts with ample detail and context produce better outcomes.
- **Skipping delimiters around user-supplied text**: leaves the application open to prompt injection, where an attacker's embedded instructions override the intended task.
- **Trusting hallucinated output without verification**: convincingly-written but false responses (e.g., fictional product reviews) can mislead users if deployed without cross-referencing credible sources.
- **Asking a model to evaluate a solution before solving it independently**: anchors the model toward agreeing with a possibly-wrong input instead of reasoning from scratch.

## Code Examples
```python
!pip install openai
```

```python
prompt = f"""
Summarize the text delimited by triple backticks \
into a single sentence.
```{text}```
"""
```

```python
{"role": "system",
 "content": "You are a helpful personal assistant..."
},
{"role": "user",
 "content": "Tell me about the weather in Las Vegas for today"
},
```

```python
## Prompt template for generating blog
prompt_template = """
You are a {blog_style} who writes blogs about {input_text}.
Your blog should be {no_words} words long.
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["input_text", "no_words", "blog_style"])
```

```python
from langchain.prompts import PromptTemplate
## Define desired product details
product_name = "AeroGlide UltraSlim Smart Toothbrush"
prompt = PromptTemplate(
    template="Write a concise and informative product description for the following item: {product_name} by the brand: {brand}",
    input_variables=["product_name", "brand"],
)
partial_prompt = prompt.partial(brand="Some Funny Brand")
print(partial_prompt.format(product_name="AeroGlide UltraSlim Smart Toothbrush"))
```
- **What it demonstrates**: environment setup (OpenAI library install), delimiter-based prompt structuring to prevent injection, persona-based role prompting, and LangChain `PromptTemplate`/partial-template patterns for reusable, parameterized prompts.

## Reference Tables
No formal tables in this chapter; content is organized as a sequence of named prompting strategies with before/after prompt examples (figures in the original book).

## Worked Example
The chapter's running example is iteratively designing a prompt for blog-post titles about surfing, following the 7-step process: (1) Goal: get the LLM to generate blog topic titles; (2) Initial prompt: `Suggest five blog post titles about surfing.`; (3) Test: model returns generic titles like "Riding the Waves: A Beginner's Guide to Surfing"; (4) Analyze: titles are too general/non-specific; (5) Refine: `Suggest five blog titles related to finding the best surfing locations around the world.`; (6) Iterate: re-test and compare, e.g., "Surfing Paradise Unveiled: Top 5 Must-Visit Destinations Around the World"; (7) Implementation: once satisfied, the final prompt is wired into a chatbot/content-generator/research-tool in production.

## Key Takeaways
1. Apply the two core guidelines first — clear/specific instructions and giving the model time to think — before reaching for advanced tricks.
2. Always delimit user-supplied text (triple backticks, XML tags, etc.) to prevent prompt injection.
3. Request structured output (JSON/HTML) whenever a response must be parsed programmatically.
4. Use few-shot examples when training data is scarce, and chain-of-thought step breakdowns when tasks require multi-stage reasoning.
5. Treat prompt design as a 7-step iterative loop (define → write → test → analyze → refine → iterate → implement), not a single draft.
6. Watch for hallucinations — the model will produce confident, well-formatted, but potentially fabricated content; verify critical facts externally.
7. LangChain's `PromptTemplate` (with partial-filling support) is the recommended pattern for managing complex, parameterized production prompts.

## Connects To
- **Ch4**: The OpenAI API setup (`.env` keys, `get_completion` helper) established here is reused when integrating models into applications.
- **Ch5**: Fine-tuning is introduced as the next escalation step when prompt engineering alone cannot fix model behavior.
- **Ch6**: The ChatGPT API build chapter directly applies these prompting techniques (delimiters, personas, structured output) in a working chat application.
- **Ch9**: LangChain prompt templates introduced here are reused in the LangChain+Streamlit data-visualization chapter.
