# Chapter 5: Errors, Logging, and Debugging

## Core Idea
Robust code requires three complementary practices: deliberately handling or raising errors so failures are predictable, logging what the code did so others (and future you) can reason about it, and having a toolkit of debugging strategies/tools to track down bugs efficiently.

## Frameworks Introduced
- **Error-handling decision**: for each failure point, decide explicitly whether to let it crash, handle it (return default/alternate behavior), or raise a custom error — never silently swallow it.
  - When to use: anywhere code could receive unexpected input or an external call could fail.
  - How: use `try`/`except`/`else`/`finally`; in `except`, either recover with a clearly-typed default return (same shape as the success case) or re-raise/raise a new, informative (optionally custom) exception; never use a bare `except: return`.
- **Logging-levels framework**: five severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) let you record everything during development and filter to only what matters in production.
  - When to use: throughout a script/service's lifetime — task start/end, errors, function inputs/outputs, file paths, call chains, model metrics.
  - How: `logging.basicConfig(level=..., filename=..., format=...)` once at script start; call `logging.debug/info/warning/error/exception(...)` at appropriate points; prefer logging over print() for anything beyond a quick one-off script.
- **Debugging strategy ladder**: run small experiments → use tests → save failing inputs/logs → search the error message → talk it through (rubber duck) → step away → use a debugger/breakpoints for fine-grained inspection.
  - When to use: when a bug can't be spotted by reading code alone.
  - How: change one thing at a time and observe effect; write/run a test that currently fails; persist the exact inputs that caused the failure (logging helps); Google the exact error text; explain the code aloud to a person or rubber duck; take a break; as a last resort, set breakpoints and step through variable state with an IDE debugger or `pdb`.

## Key Concepts
- **Syntax error**: code that isn't valid Python at all (e.g., unclosed parens, missing `def`) — caught at parse time, before execution.
- **Exception**: a runtime error from otherwise-valid code (e.g., `NameError`, `KeyError`, `TypeError`, `ValueError`) — the main category handled with `try`/`except`.
- **Traceback**: Python's error report showing the chain of calls leading to the failure; best read bottom-up — last line gives error type + hint, lines above point to the failing code line (though the *true* cause may be in an earlier line, e.g. a missing function argument).
- **`try` / `except` / `else` / `finally`**: `try` wraps risky code; `except <ErrorType>` defines recovery behavior for that error type (can list multiple types in a tuple, or have multiple except blocks); `else` runs only if no exception occurred; `finally` always runs (e.g., releasing a resource such as closing a file) and must come last.
- **`raise`**: deliberately triggers an exception (built-in or custom) with a descriptive message, e.g. `raise ValueError("Timestamps and data cannot be empty lists")`, so dependent code also stops and the cause is clear.
- **Custom exception classes**: user-defined errors created via `class NewException(Exception): pass` (an application of inheritance, Ch 4) — give large projects named, specific error types so callers can catch/handle precisely and debugging is more efficient.
- **Logging vs. printing**: logging persists structured, timestamped, severity-tagged messages to a separate file (or log aggregation system) that can be searched later, independent of whether the program is still running — essential for production/long-running processes where you can't just watch stdout.
- **`logging.basicConfig()`**: sets global logging configuration (minimum `level`, `filename`, `filemode`, `format`) once at the start of a script; default level is WARNING (so INFO/DEBUG calls are silently dropped unless you lower the threshold) and default destination is the console (not a file).
- **`logging.exception()`**: logs a message at ERROR level *and* records the full traceback — most useful call for capturing debugging context inside an `except` block.
- **Debugger / breakpoint**: a tool that pauses code execution at a chosen line and lets you inspect/step through variable state one line at a time — more scalable than scattering print statements when there are many variables or large structures to inspect.
- **pdb**: Python's built-in command-line debugger; insert `breakpoint()` in code, run the script normally, and use commands like `p <expr>` (print), `n` (next line), `c` (continue), `l` (list), `r` (return), `q` (quit).

## Mental Models
- **"Read tracebacks from the bottom up"**: the last line names the error type and often a fix hint; working upward from there is more efficient than reading top-down, even though the *root cause* may sit one line earlier than the highlighted line.
- **"Logging is evidence, not just diagnostics"**: a log isn't only for catching failures — it's a durable record proving a task completed and what result it produced (e.g., logging a trained model's accuracy), useful for both debugging and auditing.
- **"Err on the side of logging more"**: because logs are filterable (by severity level) and searchable (especially with structured/keyword-rich, f-string messages), it's cheap to over-log and expensive to under-log when you later need the missing context.
- **Debugging as a graduated toolbox**: print statements → logging → interactive debugger (IDE or `pdb`) form an escalating path matched to problem complexity — simple scripts might only need prints, but many-variable/production code needs debuggers or persistent logs.
- **"You are not the first person with this bug"**: treating an error message as a searchable string (literally pasting it into a search engine) is a legitimate, high-leverage debugging step, not a shortcut to be ashamed of.

## Anti-patterns
- **Bare `except: return` (silencing all errors)**: hides failures entirely, making it impossible to know a function didn't work — explicitly called out as "how you should not handle errors."
- **Uninformative error messages**: `"An error occurred"` tells the reader nothing; messages should state what went wrong and, ideally, what to do about it.
- **Cluttering code with permanent print statements**: makes output hard to parse, doesn't work well in production (no console access), and must be manually removed — logging is the production-safe substitute.
- **Leaving `breakpoint()` calls in committed code**: pdb will trigger on every run until they're removed; Chapter 12 shows how to automate their removal (e.g., via pre-commit hooks/linters).
- **Letting an except block return a different shape than the success path**: breaks the function's contract for downstream callers — keep return types consistent across success and handled-failure paths.
- **Logging at INFO/DEBUG without ever setting the level**: because the default logging level is WARNING, these calls silently do nothing — a common "why isn't anything showing up" trap.

## Code Examples
```python
# try / except / else for error handling with a default return
def fit_trendline(year_timestamps, data):
    try:
        result = linregress(year_timestamps, data)
    except TypeError:
        print("Both lists must contain floats or integers.")
        return 0.0, 0.0
    else:
        slope = round(result.slope, 3)
        r_squared = round(result.rvalue**2, 3)
        return slope, r_squared
```
- **What it demonstrates**: handling a `TypeError` with a same-shaped default return so callers get a consistent interface.

```python
# Raising a custom, informative error
from scipy.stats import linregress

def fit_trendline(year_timestamps, data):
    if not year_timestamps or data:
        raise ValueError("Timestamps and data cannot be empty lists")
    result = linregress(year_timestamps, data)
    slope = round(result.slope, 3)
    r_squared = round(result.rvalue**2, 3)
    return slope, r_squared
```
- **What it demonstrates**: proactively validating input and raising a descriptive `ValueError` rather than letting a cryptic downstream error occur.

```python
# Custom exception class via inheritance
class NewException(Exception):
    pass
```
- **What it demonstrates**: defining project-specific, named error types for finer-grained handling in large codebases.

```python
# Logging configuration and usage, including exceptions
import logging

logging.basicConfig(filename="chapter_5_logs.log",
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

def fit_trendline(year_timestamps, data):
    logging.info("Running fit_trendline function")
    try:
        result = linregress(year_timestamps, data)
    except TypeError as e:
        logging.error("Both lists must contain floats or integers.")
        logging.exception(e)
    else:
        slope = round(result.slope, 3)
        r_squared = round(result.rvalue**2, 3)
        logging.info(f"Completed analysis. Slope of the trendline is {slope}.")
        return slope, r_squared
```
- **What it demonstrates**: timestamped file-based logging with INFO-level progress messages and ERROR-level + full-traceback capture on failure.

```python
# Buggy function, debugged with print, then logging, then pdb
def weighted_mean(num_list, weights):
    running_total = 0
    for i in range(len(num_list)):
        running_total += (num_list[i] * weights[0])   # BUG: should be weights[i]
        breakpoint()                                    # pdb pauses here
    return (running_total / len(num_list))

weighted_mean([1, 6, 8], [1, 3, 2])
```
```
# pdb session
(Pdb) p running_total
1
(Pdb) p weights
[1, 3, 2]
(Pdb) n
...
(Pdb) q
```
- **What it demonstrates**: a real off-by-reference bug (`weights[0]` instead of `weights[i]`), diagnosed first via print output, then via file logging, then via interactive stepping with `pdb` commands `p`, `n`, `q`. Fix: `running_total += (num_list[i] * weights[i])`.

## Reference Tables
| Logging level | When used |
|---|---|
| DEBUG | Detailed info, useful only when diagnosing problems |
| INFO | Confirmation things are working as expected |
| WARNING (default threshold) | Unexpected event or near-future problem; software still works |
| ERROR | A more serious problem — some function could not be performed |
| CRITICAL | Serious error; program itself may be unable to continue |

| pdb command | Function |
|---|---|
| `p expr` | Print the value of an expression |
| `l` | Print the lines of code around the current line |
| `n` | Step to the next line |
| `c` | Continue to the next breakpoint |
| `r` | Continue until the current function returns |
| `q` | Quit debugging |

| try/except construct | Runs when |
|---|---|
| `try` | Always attempted first |
| `except <Type>` | The specified exception type was raised |
| `else` | No exception was raised in `try` |
| `finally` | Always, regardless of exception (e.g., closing a file) |

## Worked Example
The chapter's running bug-hunt: `weighted_mean([1, 6, 8], [1, 3, 2])` should compute `(1*1 + 6*3 + 8*2)/3`, but the buggy code always multiplies by `weights[0]` instead of `weights[i]`. The author demonstrates three escalating diagnostic techniques on the exact same bug: (1) print statements showing `running_total` is 1, 7, 15 instead of the expected 1, 19, 35; (2) the identical values written to a log file via `logging.debug(f"...")` instead of printed, so the record persists even in a production context without console access; (3) an IDE debugger (or pdb with `breakpoint()`) that pauses execution each loop iteration and lets you inspect `running_total`, `i`, and `weights` directly, confirming the exact line (`weights[0]` vs `weights[i]`) responsible.

## Key Takeaways
1. Read Python tracebacks from the bottom up: last line gives error type and hint, but the true cause may be a line earlier in your own code.
2. Use `try`/`except`/`else`/`finally` to make error behavior an explicit choice — never use a bare `except: return` that silently hides failures.
3. Raise informative, ideally custom, exceptions (`raise ValueError("...")`, or `class NewException(Exception)`) so failures are easy to diagnose and callers can catch specific error types.
4. Logging (via the `logging` module) persists a searchable, timestamped, severity-filtered record of what your code did — essential for long-running or production code where print statements aren't viable; default level is WARNING and default destination is console, so configure both explicitly.
5. `logging.exception()` captures the full traceback alongside your error message — very useful for later debugging.
6. Debugging escalates from print statements → logging → interactive debuggers (IDE breakpoints or `pdb`) as complexity grows; remove `breakpoint()` calls before shipping.
7. Non-technical debugging strategies — experimenting with small changes, writing a test, searching the exact error text, explaining the problem to someone (or a rubber duck), and taking a break — are legitimate and often highly effective.

## Connects To
- **Ch 1**: Robustness pillar — errors/logging are two of its three pillars (with testing).
- **Ch 4**: Custom exception classes use inheritance (`class NewException(Exception)`).
- **Ch 7**: Testing — tests are described as a debugging tool (confirming whether a change produces the expected result) and are explored fully in the next chapter.
- **Ch 12**: Automation — mentions a way to automatically remove leftover `breakpoint()` calls (pre-commit hooks/linting).
- **External**: Python standard library `logging` docs/tutorials; VS Code and PyCharm debugger documentation; Nina Zakharenko's article on pdb commands; Julia Evans's debugging philosophy quote.
