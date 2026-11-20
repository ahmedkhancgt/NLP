# Chapter 15: Next Steps

## Core Idea
The book closes by consolidating its five pillars of good code (simplicity, modularity, readability, efficiency, robustness), cautioning that AI coding assistants are useful starting points but require solid software-engineering judgment to use well, and encouraging pragmatism, ethical awareness, and community engagement as a data scientist's career in code continues.

## Frameworks Introduced
- **Five-pillar recap as a pre-coding checklist**: the same framework from Ch 1, now framed as what to recall "next time you sit down to write some code."
  - When to use: as a final habitual check before/while writing any new code.
  - How: Simplicity (easier for others to use), Modularity (sensible chunks, OOP where it fits, good overall structure), Readability (documentation + consistent formatting — code is read far more than written), Efficiency (as efficient as it needs to be, no more), Robustness (handle and log errors, test thoroughly).
- **AI-assisted coding verification workflow**: treat AI-generated code as a draft requiring the same scrutiny as any other code.
  - When to use: whenever using tools like GitHub Copilot or ChatGPT to generate functions/tests.
  - How: run the generated code and its generated tests yourself; verify the expected values are actually correct (don't trust docstring examples or test assertions at face value); check for missing concerns the tool won't add unprompted (logging, documentation, error handling, efficiency trade-offs) unless explicitly asked for; apply your own software-engineering judgment to catch what the tool got wrong.
- **Pragmatism principle**: match code quality investment to the actual stakes of the project.
  - When to use: deciding how much engineering rigor a given piece of code deserves.
  - How: for short-term/throwaway/deadline-driven work, "write whatever code gets the job done" — the value of this book is knowing *when* you're writing that kind of code, not refusing to ever write it.

## Key Concepts
- **The five pillars, final framing**: Simplicity, Modularity, Readability, Efficiency (renamed slightly from "Performance" in Ch 1), Robustness — presented as the throughline of the entire book.
- **Standardization**: a cross-cutting theme (naming, formatting, project structure, documentation templates) that increases coding speed and eases collaboration.
- **Abstraction**: putting implementation details behind a sensible interface so code is easy to reuse elsewhere — another cross-cutting theme spanning OOP interfaces, API design, and modular code.
- **Automation**: repeatedly emphasized (formatting, testing, deployment) as a way to save time and accelerate work over the long term.
- **AI coding assistants' current limitation**: they produce plausible-looking code (with docstrings and tests) that can nonetheless contain outright factual/arithmetic errors — demonstrated concretely with ChatGPT's `weighted_mean` example, where the generated test's `expected_result` was simply wrong (30.0 instead of the correct 32.0), and the error was baked into both the docstring example and the test assertion.
- **"You need to know what to ask for"**: AI tools won't spontaneously add logging, documentation, or performance considerations unless you request them — effective use of AI coding assistants presupposes the same software-engineering knowledge this book teaches.
- **Code's real-world consequences**: data scientists' code can scale to millions of users, so personal ethics should inform what you choose to build.
- **Leading by example / team culture**: senior practitioners shape their team's coding culture through code reviews and setting a good standard, not just individual output.
- **Community engagement as a closing theme**: building connections, giving talks, joining open source, and being open to feedback from software engineers are framed as enriching, not optional extras.

## Mental Models
- **The book as a translation layer, not a replacement for deeper study**: Nelson frames her own book as giving data scientists enough vocabulary and context to then "dive into" denser software-engineering material (Pragmatic Programmer, A Philosophy of Software Design, etc.) and know which parts are relevant to their work.
- **AI assistants as fast-but-unreliable junior collaborators**: useful for a "starting point," but their output demands the same verification rigor as any other code — maybe more, since confident-looking docstrings/tests can mask wrong answers.
- **Software engineering as a permanently moving target**: "software engineering won't stay the same in the future" — specific tools/languages rise and fall, but the underlying principles (the five pillars) are the durable asset to carry forward.
- **Code as a lever with ethical weight**: because data science code can underlie large-scale products, the closing message reframes "writing good code" as partly a matter of professional responsibility, not just craftsmanship.

## Anti-patterns
- **Trusting AI-generated code/tests without verification**: the chapter's own worked example shows a generated test with an incorrect expected value (and the same error repeated in the docstring) that would have shipped a false "all tests pass" signal if run naively — you must actually re-derive or independently check expected values.
- **Writing an unnecessarily tight tolerance check when an exact comparison would do**: ChatGPT's generated test used `assert abs(result - expected_result) < epsilon` (an epsilon-based float comparison) where a plain `assert result == expected_result` would have sufficed for this calculation — an example of AI-generated code being technically functional but not minimal/clean.
- **Pursuing "great code" uniformly regardless of context**: treating every script as deserving full engineering rigor is itself a mistake — pragmatism means recognizing when "horrible code that works" is the right call.
- **Relying on AI tools to supply non-functional requirements unprompted**: expecting logging, documentation, or performance-aware design to appear without being explicitly requested.

## Code Examples
```python
# ChatGPT-generated weighted_mean function (with a latent bug in its own docstring example)
from typing import List

def weighted_mean(numbers: List[float], weights: List[float]) -> float:
    """
    Calculate the weighted mean of a list of numbers.
    ...
    Example:
    >>> numbers = [10, 20, 30, 40, 50]
    >>> weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    >>> weighted_mean(numbers, weights)
    30.0
    """
    if len(numbers) != len(weights):
        raise ValueError("The number of numbers and weights must be the same.")
    if any(w == 0 for w in weights):
        raise ValueError("Weights must be greater than zero.")
    weighted_sum = sum(x * w for x, w in zip(numbers, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight
```
```python
# ChatGPT-generated test — contains the SAME wrong expected value (30.0 instead of 32.0)
import pytest
from weighted_mean_function import weighted_mean

def test_weighted_mean():
    numbers = [10, 20, 30, 40, 50]
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    expected_result = 30.0   # WRONG — correct value is 32.0

    result = weighted_mean(numbers, weights)

    epsilon = 1e-6
    assert abs(result - expected_result) < epsilon   # fails when corrected

def test_weighted_mean_with_zero_weight():
    numbers = [10, 20, 30, 40, 50]
    weights = [0.1, 0.2, 0.3, 0.0, 0.2]
    with pytest.raises(ValueError):
        weighted_mean(numbers, weights)

def test_weighted_mean_with_mismatched_lengths():
    numbers = [10, 20, 30, 40, 50]
    weights = [0.1, 0.2, 0.3, 0.2]
    with pytest.raises(ValueError):
        weighted_mean(numbers, weights)
```
```python
# Corrected, simplified test (exact comparison, correct expected value)
def test_weighted_mean():
    numbers = [10, 20, 30, 40, 50]
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    expected_result = 32.0

    result = weighted_mean(numbers, weights)

    assert result == expected_result
```
- **What it demonstrates**: an AI assistant produced well-structured, well-documented, plausible code and tests — yet the actual numeric answer baked into both the docstring and the test was wrong, and would only be caught by someone who actually ran the code and independently verified the arithmetic; the fix also simplifies an unnecessarily approximate float comparison into an exact one, since this calculation has no floating-point precision concern here.

## Reference Tables
| Pillar (final recap) | Chapter(s) covering it |
|---|---|
| Simplicity | Ch 1 |
| Modularity | Ch 1, Ch 4, Ch 8 |
| Readability | Ch 1, Ch 6, Ch 9 |
| Efficiency | Ch 1, Ch 2, Ch 3 |
| Robustness | Ch 1, Ch 5, Ch 7 |

| Recommended next book | Focus |
|---|---|
| *The Pragmatic Programmer* (Thomas & Hunt) | General software craftsmanship |
| *A Philosophy of Software Design* (Ousterhout) | Managing complexity/design |
| *High Performance Python* (Gorelick & Osvald) | Deep performance optimization |
| *Robust Python* (Viafore) | Type safety, robustness patterns |

## Worked Example
The chapter's single worked example is the ChatGPT-generated `weighted_mean` function and its accompanying tests, described in detail above: a plausible-looking, well-documented, tested function that is actually wrong (expected weighted mean of `[10,20,30,40,50]` with weights `[0.1,0.2,0.3,0.2,0.2]` is 32.0, not the 30.0 baked into both the docstring's example and the test's `expected_result`). This single example carries the chapter's central warning about AI coding assistants: fluent, confident-looking output is not the same as correct output, and the software-engineering skills taught throughout the book (independently verifying results, writing precise tests, knowing what non-functional requirements to ask for) are exactly what's needed to catch this kind of failure.

## Key Takeaways
1. Before writing code, recall the five pillars: Simplicity, Modularity, Readability, Efficiency, Robustness — the organizing structure of the entire book.
2. Standardization, abstraction, and automation are cross-cutting themes that compound in value as projects and teams grow.
3. Be pragmatic: not every piece of code needs full rigor — short-term/throwaway code can reasonably be "whatever gets the job done," as long as you recognize that's what you're doing.
4. AI coding assistants (GitHub Copilot, ChatGPT) are useful starting points but can produce confidently wrong code and tests — verify generated code and its expected values yourself, and know what non-functional concerns (logging, documentation, performance, error handling) to explicitly request.
5. Software engineering practices and tools will keep changing, but the underlying principles in this book are meant to be durable and transferable to deeper, more specialized material.
6. Code can have real-world consequences at scale — bring personal ethics into what you choose to build, and help shape good team culture through code reviews and leading by example.
7. Engage with the wider community (open source, conferences, feedback from software engineers) as an ongoing, enriching part of a career in code — not a one-time checklist item.

## Connects To
- **Ch 1**: Direct recap of the five pillars that opened the book.
- **Ch 4 / Ch 8**: Modularity pillar ties back to OOP and project/code design.
- **Ch 6 / Ch 9**: Readability pillar ties back to formatting and documentation.
- **Ch 2 / Ch 3**: Efficiency pillar ties back to performance analysis and data structures.
- **Ch 5 / Ch 7**: Robustness pillar ties back to error handling/logging and testing.
- **Ch 14**: Community engagement, team culture, and career themes continue directly from the previous chapter.
- **External**: *The Pragmatic Programmer* (Thomas & Hunt); *A Philosophy of Software Design* (Ousterhout); *High Performance Python* (Gorelick & Osvald); *Robust Python* (Viafore); PyCon US talks on YouTube as a free ongoing learning resource.
