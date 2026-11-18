# Chapter 2: Jupyter Notebook

## Core Idea
Jupyter Notebook's cell-based interactive model (code cells + Markdown/LaTeX text cells) lets you combine executable Python, narrative documentation, and GPT-4o-generated explanations into a single shareable artifact, and Google Colab removes all setup friction by providing this environment free in the browser with GPU/TPU access.

## Frameworks Introduced
- **Code/Text cell separation**: keep executable logic in code cells and narrative/explanation (including copy-pasted GPT-4o explanations) in Markdown text cells.
  - When to use: every notebook, to keep documentation evolving alongside code instead of in a separate file.
  - How: `+ Code` / `+ Text` buttons, or Alt+A (Windows) / Option+A (Mac) for a new code cell, Ctrl+M / Option+M for a new text cell.
- **Copy-GenAI-code-then-explain workflow**: ask GPT-4o to write a Python snippet, paste the code into a code cell, then paste GPT-4o's prose explanation into an adjoining text cell.
  - When to use: learning any new construct (variables, lists, if/else, loops).
  - How: prompt GPT-4o precisely ("write a Python example of a list with inline documentation"), copy only the code into the code cell, copy the explanation into Markdown so the notebook is self-contained without later needing GenAI access.

## Key Concepts
- **Jupyter Notebook**: open-source interactive document format mixing code execution, output, and narrative text in one file; kernel-based (supports Python, R, Julia, Scala, JavaScript).
- **Google Colab**: Google's free cloud-hosted Jupyter environment with GPU/TPU access, Google Drive integration, and real-time multi-user collaboration; each new notebook spins up a fresh server instance with no retained state from prior sessions.
- **Code cell**: a cell that executes Python and shows output immediately below; cells can be run in any order, not strictly top-to-bottom.
- **Text (Markdown) cell**: a cell for narrative, using Markdown syntax (`#`/`##`/`###` headers, `**bold**`, `*italic*`, `-` bullets) and LaTeX (for math notation), rendered in a live split-pane preview.
- **drive.mount()**: the `google.colab` library function that mounts a user's Google Drive at `/content/drive` inside a Colab session after an OAuth permission flow.
- **GPU/TPU tiers**: Colab's free "Standard" tier (12 GB RAM, NVIDIA T4, 8 GB GPU RAM) versus paid Colab Pro tiers (L4, A100 GPUs; v2-8/v5e-1 TPUs) for progressively heavier ML workloads.
- **In-line comments**: single-line (`#`) and multi-line (triple-quoted `"""..."""`) annotations inside code cells, distinct from Markdown text cells but serving a similar explanatory purpose.

## Mental Models
- Think of Jupyter cells like a detective examining clues one at a time: each code cell is independently re-runnable, so a bug in one chunk doesn't force you to re-run the whole script (unlike traditional top-to-bottom script execution in other IDEs).
- Treat a fresh Colab notebook as a brand-new computer each time: no previously installed libraries or variables persist across new notebook instances, so re-run setup/import cells every session.
- Use the free Standard Colab tier as the default; only reach for Pro/Pro+ tiers (L4/A100 GPU, TPU) when a task concretely requires training large models (LLMs, Stable Diffusion-scale image generation).

## Anti-patterns
- **Expecting notebook state to persist across new Colab instances**: each "New Notebook" creates a fresh server — installed packages and variables from earlier sessions are gone.
- **Skipping the text-cell explanation step**: pasting only GPT-4o's code without its explanation into the notebook means future readers (including yourself) need live GenAI access to understand it later.
- **Upgrading to Colab Pro by default**: unnecessary for most small/medium projects; the free tier's GPU/TPU already covers typical learning and experimentation workloads.

## Code Examples
```python
# variables (Figure 2.4)
name = "Ada Lovelace"   # string variable
age = 28                 # integer variable
height = 1.7              # float variable
print(f"{name} is {age} years old and {height} meters tall")
# -> Ada Lovelace is 28 years old and 1.7 meters tall

# lists (Figure 2.5)
fruits = ["apple", "banana", "cherry"]
print(fruits[0])    # apple
print(fruits[-1])   # cherry
fruits.append("orange")
print(fruits)        # ['apple', 'banana', 'cherry', 'orange']

# if/else (Figure 2.6)
if age >= 18:
    print(f"{name} is an adult.")
else:
    print(f"{name} is a minor.")

# looping (Figure 2.7)
my_list = [10, 20, 30, 40, 50]
for value in my_list:
    print(value)

# mounting Google Drive (Figure 2.12)
from google.colab import drive
drive.mount('/content/drive')
```
- **What it demonstrates**: the four foundational Python building blocks (variables, lists, conditionals, loops) exactly as GPT-4o generated them for the authors, plus the standard Colab/Drive integration snippet.

## Reference Tables

**Colab resource tiers (Section: Free access to GPUs and TPUs)**

| Tier | CPU RAM | GPU/TPU | Best for |
|---|---|---|---|
| Standard (Free) | 12 GB | NVIDIA T4 (8 GB) / v2 TPU (8 cores, 64 GB) | Small/medium projects, light ML |
| L4 GPU (Pro) | 43 GB | NVIDIA L4 (22 GB) | Moderate deep learning, larger datasets |
| A100 GPU (Pro) | 86 GB | NVIDIA A100 (40 GB) | LLM training, image generation (Stable Diffusion) |
| v2-8 TPU (Pro+) | 64 GB | 128 GB TPU RAM / 8 cores | Large-scale TensorFlow/PyTorch NLP, image classification |
| v5e-1 TPU (Pro+) | 128 GB | 256 GB TPU RAM | GPT-scale LLMs, RL, high-res generation |

**Markdown quick reference (from GPT-4o, Figure 2.8)**: `#`/`##`/`###` = heading levels; `---` = horizontal rule; `*text*` = italic; `**text**` = bold; `- item` = bullet list; `1. item` = numbered list.

**Jupyter-hosting platform comparison**

| Platform | Strength | Weakness |
|---|---|---|
| Google Colab | Free GPU/TPU, zero setup, Drive integration | Slower at peak times, limited hardware customization |
| Microsoft Azure Notebooks | Enterprise Azure integration, scalable | Steep learning curve, cost scales up |
| IBM Watson Studio | AI/AutoML tools, enterprise security | Complex UI, costly advanced features |
| Databricks Community Edition | Spark/big-data integration, free clusters | Limited vs. full version, less intuitive for non-Spark users |
| Kaggle Kernels | Preloaded datasets/libraries, free, community | Not suited for private/commercial projects |
| Local install (pip) | Full control, offline, private | Manual setup/maintenance, no free GPU |

## Worked Example
Local installation via pip, step by step: (1) verify Python 3.10+ with `python3 --version`; (2) `python3 -m pip install --upgrade pip`; (3) `python3 -m pip install notebook`; (4) launch with `jupyter notebook`, which opens the notebook interface in the default browser. Contrasted against the Colab route (browser to colab.research.google.com -> sign in with Google -> "New Notebook" -> fresh server instance with Example/Recent/GitHub/Upload tabs for sourcing notebooks) and the Google Drive mount flow (run `drive.mount('/content/drive')` -> four-step OAuth popup sequence: connect to Drive -> choose Google account -> confirm sign-in -> grant the listed permissions -> Drive is then browsable under `/content/drive/My Drive/Colab Notebooks/` with standard Python file I/O, e.g. `import os; os.listdir(...)`).

## Key Takeaways
1. Use Google Colab as the default, zero-setup Jupyter environment; fall back to local pip installation only when offline work or full environment control is required.
2. Structure every notebook with code cells for logic and text cells for GPT-4o-sourced explanations, so the notebook is self-documenting and shareable without needing live GenAI access later.
3. Remember a new Colab notebook starts with a clean server — re-import libraries and re-run setup cells each session.
4. Mount Google Drive (`drive.mount('/content/drive')`) when a project's datasets are too large to re-upload every session.
5. Reserve paid Colab Pro/Pro+ tiers for genuinely heavy workloads (LLM training, large image generation); the free tier covers most learning projects.

## Connects To
- **Ch 3**: the code/text cell workflow and GPT-4o prompting habits established here become the method for building and documenting the calculator app.
- **Ch 4**: the `%%time` magic command and Matplotlib visualization approach introduced for notebooks here are applied to benchmark sorting algorithms.
