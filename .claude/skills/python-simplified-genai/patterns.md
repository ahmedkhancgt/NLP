# Patterns

## Play-First / Project-Based Learning
**When to use**: learning any new Python concept, library, or paradigm (variables through CNNs).
**How**: build or request a small, concrete, working project immediately (calculator, sorting function, data analysis, CNN classifier) rather than memorizing syntax rules first; learn the syntax through hands-on exploration of that project.
**Trade-offs**: faster intuition and retention, more engaging, but can leave gaps in systematic/theoretical coverage — supplement with GenAI explanations when a concept needs formal grounding.

## GenAI-as-Coding-Companion
**When to use**: throughout any coding task — generating code, explaining unfamiliar lines, debugging, or brainstorming design options.
**How**: prompt GPT-4o conversationally ("write X with inline documentation"), copy the generated code into a Jupyter code cell, then ask follow-up questions ("explain this line," "explain it more simply," "simplify this logic") until the explanation is clear; copy the explanation into a Jupyter text cell for permanence.
**Trade-offs**: dramatically accelerates learning and code production, but risks passive over-reliance; must be paired with independent critical verification (see "Question GenAI's Output" below).

## Prompt Engineering Loop
**When to use**: whenever GenAI's first response is too vague, generic, or mismatched to your actual need.
**How**: iteratively add precision/context to the prompt (narrow from "best cake recipe" to "best birthday cake that's not chocolate and has an interesting flavor"), or front-load all constraints into one well-specified prompt; explicitly instruct desired format/detail/structure.
**Trade-offs**: takes a little upfront thought, but produces dramatically better results than accepting a vague first answer or stacking many follow-up questions.

## Question GenAI's Output Critically
**When to use**: any time a GPT-4o-generated solution feels more complex than the problem warrants, or you're unsure it's correct.
**How**: explicitly ask "can you simplify this logic?" or "is there a simpler way?"; verify dataset-specific or domain-specific claims yourself rather than trusting GenAI's inference (e.g., the snake dataset's file-path convention had to be reverse-engineered by the authors, not GPT-4o).
**Trade-offs**: takes extra time and requires enough base understanding to judge quality, but is essential — GenAI can confidently produce plausible-but-overcomplicated or factually wrong output.

## Debug-Log-Driven Understanding
**When to use**: understanding or troubleshooting any loop-heavy or recursive algorithm.
**How**: insert `print(f"Debug log: ...")` statements inside loop bodies to expose intermediate state (pass number, current list/values) at each iteration; remove or comment out before deploying production code.
**Trade-offs**: simple and immediate, but verbose output can clutter notebooks on large inputs — use targeted/conditional logging or remove once understanding is achieved.

## Flowchart-First Algorithm Design
**When to use**: planning any multi-step, nested-loop, or recursive algorithm before or while coding it.
**How**: ask GPT-4o which library can draw flowcharts (Graphviz), then ask it to generate Graphviz code visualizing the specific algorithm; use the diagram to map logic before/while writing Python.
**Trade-offs**: adds a planning step, but catches logic errors early and builds durable mental models, especially valuable for teaching/sharing.

## Code-and-Explanation Pairing in Jupyter
**When to use**: every notebook meant to be revisited or shared later.
**How**: keep code in code cells; copy GPT-4o's prose explanation of that code into an adjoining Markdown text cell (using LaTeX for any math); this makes the notebook self-contained without needing live GenAI access later.
**Trade-offs**: extra copy-paste step per concept, but pays off enormously for future readability, sharing, and review.

## Toy-Dataset-Then-Real-Dataset Progression
**When to use**: learning any data-manipulation library or workflow (Pandas, CNN data pipelines).
**How**: master every operation on a small, hand-built dataset first (e.g., 8 students), then apply the identical function calls unchanged to a large real-world dataset (e.g., 11,222-row climate data, 23,816-row snake metadata).
**Trade-offs**: the toy stage can feel slow/trivial, but it isolates the mechanics of each function from the complexity of messy real data, making debugging far easier later.

## info()-First Data Triage
**When to use**: immediately upon importing any new or unfamiliar dataset.
**How**: run `df.info()` before any other operation to check row/column counts, dtypes, and null counts; investigate any anomaly (wrong dtype, unexpected nulls) before proceeding to analysis.
**Trade-offs**: minimal cost, high payoff — catches data-quality issues (e.g., a numeric column stored as a prefixed string) before they silently corrupt downstream analysis.

## GitHub-as-Data-Bridge
**When to use**: importing an external dataset (e.g., downloaded from Kaggle) into a cloud notebook (Colab) repeatedly without manual re-upload.
**How**: download the dataset, push it to a personal GitHub repo, then use the `raw.githubusercontent.com` URL (not the `www.github.com` viewer URL) in `pd.read_csv()`.
**Trade-offs**: requires a GitHub account and one-time setup, but eliminates repeated manual uploads and works seamlessly with GPT-4o-generated import code.

## Transfer Learning by Default
**When to use**: virtually any CNN/image-classification project.
**How**: select a pre-trained base architecture (e.g. ResNet34) rather than training from scratch; use Fast.ai's `vision_learner()` and `fine_tune()` to adapt it to your specific dataset.
**Trade-offs**: dramatically cheaper/faster than training from scratch (minutes vs. months and millions of dollars), at the cost of being bound to the base model's existing learned representations (generally not a meaningful limitation for most applied tasks).

## Learning-Rate-Search Before Training
**When to use**: before fine-tuning any CNN (or other deep learning model).
**How**: run Fast.ai's `lr_find()` to get candidate learning rates, choose a rate in the stable "slide" region rather than the most aggressive option, then call `fine_tune()` with that rate.
**Trade-offs**: adds one extra step/cell, but avoids wasted training cycles from a poorly chosen default learning rate.

## Function-First, GUI-Second Deployment
**When to use**: turning any working Python function into a shareable web application.
**How**: fully build and test the core logic function first (calculator, sort, predict), then wrap it separately with Gradio components bound to that function — keep logic and UI code decoupled.
**Trade-offs**: slightly more setup than writing UI-coupled logic directly, but keeps the function independently testable/reusable (e.g., via the API) and easier to debug.

## Benchmark-Before-You-Choose
**When to use**: deciding which algorithm/approach to use in practice, not just by theoretical complexity.
**How**: use Jupyter's `%%time` magic on each candidate implementation against the same realistic-sized input, then compare actual wall-clock results.
**Trade-offs**: results are specific to your test data/hardware and may not generalize, but give concrete, actionable comparison beyond Big-O alone.

## Prompt-Sensitivity Bias Testing
**When to use**: auditing a text-to-image or text-generation model for representational bias before trusting or deploying its output.
**How**: run an ambiguous prompt many times, tally the demographic/stereotype distribution of outputs, compare against a trusted real-world baseline statistic; then re-run with explicit disambiguating context to see if the bias persists or resolves.
**Trade-offs**: informal/manual (not a rigorous statistical audit), but accessible to any user and effective at surfacing gross representational skew quickly.
