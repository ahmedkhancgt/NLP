# Chapter 27: Go Forth and Do Data Science

## Core Idea
The closing chapter is a guided "what next" tour: master IPython, deepen the underlying mathematics (linear algebra, statistics, probability), and — critically — stop building everything "from scratch" and adopt the real libraries (NumPy, pandas, scikit-learn, visualization tools, R, deep learning frameworks), then find data and go build personally-motivated projects.

## Frameworks Introduced
- **"From scratch" was pedagogical, not practical**: the book's own closing argument is that implementing things from scratch is great for understanding mechanics but "generally not great for performance... ease of use, rapid prototyping, or error handling" — the explicit next step is to switch to well-designed libraries that solidly implement the fundamentals you now understand.
  - When to use: immediately after finishing this book — treat every from-scratch implementation in Chapters 1-26 as understanding scaffolding to be replaced by NumPy/pandas/scikit-learn/etc. in real work.
- **Project-driven learning ("do data science")**: pick projects that "tickle some sort of itch" (personal curiosity) rather than only working through generic data catalogs — the author's own examples (Hacker News classifier, Fire Truck network analysis, T-shirt image classifier, Tweets-on-a-globe visualization) model this approach.

## Key Concepts
- **IPython**: an enhanced Python shell with "magic functions" (e.g., easy paste handling despite whitespace-sensitive code) — the author considers mastering it a high-leverage, low-effort investment. (Notably, the author expresses skepticism toward Jupyter/IPython *Notebooks* specifically, considering them confusing for beginners and encouraging of bad coding practices — a dissenting opinion flagged as such.)
- **NumPy**: provides high-performance arrays/matrices ("real" scientific computing) that outperform the book's list-based `Vector`/`Matrix`; described as "a building block for many other libraries."
- **pandas**: provides the `DataFrame` abstraction, conceptually similar to but far more capable/performant than the book's own `NotQuiteABase` `Table` (Ch24) — the primary tool for munging/slicing/grouping real datasets in Python.
- **scikit-learn**: the most popular Python ML library, containing production implementations of every model family built from scratch in this book (and many more) — "on a real problem, you'd never build a decision tree from scratch."
- **Visualization options beyond basic matplotlib**: seaborn (prettier matplotlib), D3.js (interactive web visualizations, JavaScript), Bokeh (D3-style functionality in Python).
- **Deep learning frameworks**: TensorFlow (Google, older/more widely used) vs. PyTorch (Facebook, the author's personal preference for ease of use and beginner-friendliness).
- **R**: worth knowing even if you primarily use Python, both to read others' R-based work and to be an informed participant in "R versus Python" debates.

## Mental Models
- **Understanding mechanics (from scratch) vs. using tools (in practice) are sequential, not competing, goals**: the book deliberately built everything from scratch first so that the subsequent, inevitable switch to libraries (NumPy, pandas, scikit-learn) would be an informed one — you now know what those libraries are actually doing internally.
- **Curiosity-driven projects as the real engine of skill growth**: the author's personal project examples (a Hacker News classifier using Naive Bayes just like Ch13's spam filter; a "TruckRank" fire-truck centrality metric paralleling Ch22's PageRank; an "eigenshirts" PCA-based t-shirt classifier paralleling Ch10's PCA) show that the book's techniques transfer directly to whatever genuinely interests you, not just canonical textbook problems.
- **Mathematics depth is an ongoing investment, not a finished prerequisite**: linear algebra, statistics, and probability were only "dabbled in" during the book; real proficiency requires further dedicated study beyond what any single introductory book can cover.

## Anti-patterns
- **Continuing to hand-roll algorithms in real production work**: the book's own closing stance is that doing so is a pedagogical choice, not a professional practice — "you'd never write an optimization algorithm by hand; you'd count on scikit-learn to already be using a really good one."
- **Treating Jupyter/IPython Notebooks as a default tool without considering the author's specific objections**: flagged as potentially confusing beginners and encouraging bad coding habits — worth evaluating deliberately rather than adopting by default.
- **Only working from generic public data catalogs without pursuing a personally motivating question**: the chapter explicitly contrasts "looking through data catalogs is fine" with projects that "tickle some sort of itch," favoring the latter for sustained engagement and learning.

## Code Examples
This chapter contains no new from-scratch code; it is a guide to external tools and further learning. The personal-project anecdotes reference techniques from earlier chapters applied to new domains (Naive Bayes for Hacker News story classification; a PageRank-style centrality metric for fire trucks; PCA/logistic regression for t-shirt image classification) rather than introducing new code.

## Reference Tables
| Area | Recommended tool(s) |
|---|---|
| Interactive shell | IPython |
| Numeric arrays/matrices | NumPy |
| Data manipulation | pandas (`DataFrame`) |
| Machine learning | scikit-learn |
| Static visualization | matplotlib, seaborn |
| Interactive/web visualization | D3.js, Bokeh |
| Deep learning | PyTorch (author's preference) or TensorFlow |
| Alternative language | R |
| Finding data | Data.gov, Reddit (r/datasets, r/data), Amazon public datasets, Kaggle, Google Dataset Search |

## Worked Example
The author's own side projects illustrate direct extensions of the book's techniques: a Hacker News story classifier (hand-labeled training data, title words and link domains as features, a Naive Bayes classifier "not unlike our spam filter" from Ch13); a Seattle Fire Department social-network analysis using scraped real-time 911 data, requiring inventing a fire-truck-specific centrality measure ("TruckRank," echoing Ch22's PageRank); a toddler t-shirt gender-marketing classifier built from pixel-color vectors and logistic regression (Ch16), with a second approach using the first 10 principal components ("eigenshirts," echoing Ch10's PCA) of shirt image vectors; and a real-time "spinning globe" web visualization of geotagged tweets during the 2016 election (a JavaScript-based project, prompting the suggestion to "learn some JavaScript").

## Key Takeaways
1. The book's "from scratch" approach was explicitly a teaching device — real work should use NumPy, pandas, and scikit-learn rather than hand-rolled equivalents.
2. Deepen your mathematics (linear algebra, statistics, probability) beyond this book's necessarily shallow treatment via dedicated textbooks/courses.
3. IPython (the shell, not necessarily notebooks) is a high-value tool worth mastering early.
4. Choose a visualization stack (static: matplotlib/seaborn; interactive: D3.js/Bokeh) and a deep learning framework (PyTorch or TensorFlow) based on your needs, with the author favoring PyTorch for beginners.
5. The best way to keep growing as a data scientist is to find a personally compelling question and go pursue it with real data, not just work through generic datasets.

## Connects To
- **Ch4 (Linear Algebra)**: NumPy explicitly replaces the book's from-scratch Vector/Matrix machinery.
- **Ch13 (Naive Bayes)**: the Hacker News classifier anecdote directly reuses the spam-filter approach.
- **Ch22 (Network Analysis)**: the Fire Truck "TruckRank" anecdote directly echoes PageRank/eigenvector centrality.
- **Ch10 (Working with Data / PCA)** and **Ch16 (Logistic Regression)**: the t-shirt classifier anecdote directly reuses both techniques.
- **Ch24 (Databases and SQL)**: pandas's `DataFrame` is explicitly compared to the book's own `NotQuiteABase` `Table`.
- **Python Data Science Handbook (Jake VanderPlas)**: external book recommended as the natural "now learn the libraries" follow-up to this book.
