# Chapter 4: Object-Oriented Programming and Functional Programming

## Core Idea
OOP and FP are two complementary ways to structure code — OOP organizes code around objects (things with data and behavior), FP organizes code around pure, non-mutating functions — and Python data scientists benefit from recognizing both, since common libraries (pandas, scikit-learn, Matplotlib) use them extensively, even if you don't write strictly in either style yourself.

## Frameworks Introduced
- **Class/Method/Attribute model (OOP)**: a class is a blueprint for objects; an attribute is data a class instance stores; a method is behavior (a function) bound to instances of that class.
  - When to use: when your problem space contains many instances of similar "things" that need consistent behavior but hold different data.
  - How: define `class Name:` with `__init__(self, ...)` to set attributes, and other methods (`def method(self, ...)`) for behavior; access via `instance.attribute` (no parens) and `instance.method()` (with parens).
- **Four OOP pillars**: inheritance, encapsulation, abstraction, polymorphism.
  - When to use: inheritance when a new class is a specialized variant of an existing one; encapsulation/abstraction when designing a stable public interface over changeable internals; polymorphism when multiple classes should expose the same method name for interchangeable use.
  - How: inheritance via `class Child(Parent): ... super().__init__(...)`; encapsulation/abstraction by only exposing intended methods/attributes as the "interface" and keeping internals free to change; polymorphism by giving unrelated classes a method of the same name/signature (e.g., `.fit()`).
- **Paradigm-selection heuristic**: choose OOP vs. FP vs. "just write modular scripts" based on the shape of the problem, not ideology.
  - When to use: at the start of a larger project, or when deciding whether to formalize code into classes/pure functions.
  - How: reach for OOP when you have many instances of similar things needing shared behavior with varying data (and don't bother defining a class for a single instance); reach for FP when you have fixed data you want to repeatedly transform or want to parallelize; for small scripts, neither paradigm is necessary.

## Key Concepts
- **Object**: a "thing" with data (attributes) and associated behavior (methods) that can interact with other objects; e.g., a pandas DataFrame, NumPy array, Matplotlib figure, scikit-learn estimator.
- **Class**: the blueprint/definition for a type of object; instances are individual objects created from it. Convention: CamelCase names (`MyClass`).
- **`__init__`**: Python's special method run on object creation, used to set initial attributes; its first parameter is conventionally `self`, referring to the new instance.
- **classmethod / staticmethod**: special methods that don't take `self`; classmethods apply to the whole class, staticmethods can be called without any instance.
- **Inheritance**: a child class (`class Child(Parent):`) reuses and extends a parent class's attributes/methods, calling `super().__init__(...)` to initialize inherited attributes, and can add new methods or override existing ones. Reduces duplication when a new class is a close variant of an existing one.
- **Encapsulation**: hiding a class's internal implementation behind a public interface (methods/attributes) so callers interact only with that interface, not the internals — e.g., pandas DataFrames hide their NumPy-based internals.
- **Abstraction**: dealing with a class at the appropriate level of detail; closely tied to encapsulation — deciding what should be exposed through the interface vs. kept as internal detail. Interfaces should be designed to be stable since other code depends on them (expanded on in Ch 8).
- **Polymorphism**: different, often unrelated, classes expose a method of the same name/behaviorally similar contract (e.g., every scikit-learn estimator has `.fit()`), letting calling code stay uniform regardless of which concrete class is used.
- **Functional programming (FP)**: a paradigm built on pure, immutable functions with no side effects — functions don't mutate external state/global variables, and always return a new value based solely on their inputs (e.g., add-to-list returns a new list rather than mutating the original).
- **Lambda function**: a small, unnamed, single-expression function (`lambda arguments: expression`), commonly passed to `map()`, `filter()`, or pandas' `.apply()`.
- **`map()` / `filter()` / list comprehensions / generators**: common FP-style constructs in Python; `itertools`/`more-itertools` are FP-oriented libraries worth knowing.

## Mental Models
- **Object = custom data structure + behavior**: rather than thinking of a class purely abstractly, think of it as "I'm designing a structure to hold specific data, plus the operations I'll want to perform on it" (e.g., pandas' designers built a tabular structure, then gave it row/column access methods).
- **Real-world class analogy**: the book *Software Engineering for Data Scientists* is an instance of class `Book`; attributes = page count, author name; method = "read()"; many `Book` instances share the same possible methods but differ in attribute values.
- **"Many instances" test for OOP**: only define a class when you expect multiple instances needing the same behavior — a single-use class just adds unnecessary complexity.
- **FP = "fixed data, many transformations"**: FP suits situations where the underlying data doesn't change but you want to apply many different operations to it, or where you want safe parallelization because nothing is mutated.
- **Most real Python code is a blend**: strict adherence to either paradigm is rare; production data science code typically mixes OOP (custom classes) and FP (pure helper functions, lambdas, `map`/`apply`) pragmatically.

## Anti-patterns
- **Defining a class for a single instance**: adds complexity with no benefit — OOP is justified only when you have multiple instances of a similar "thing."
- **Mutating state inside what's meant to be a "pure" function**: breaks FP's core guarantee (same input → same output, no side effects), undermining testability and parallelizability.
- **Overusing `.apply()`/`.iterrows()`-style custom functions on DataFrames when a vectorized built-in exists**: `apply()` iterates row-by-row internally and is slower than vectorized operations (per Ch 3); use built-ins for simple numeric/string ops and reserve `apply()`/lambdas for logic with no vectorized equivalent.
- **Overusing lambda functions for anything reusable or complex**: a lambda can't be named, tested independently, or reused elsewhere — prefer a named function once logic is non-trivial or might be reused/debugged.
- **Changing a class's public interface casually**: other code depends on existing attributes/methods; changing internals is fine, but changing the interface can break dependents (ties to abstraction/encapsulation and to Ch 8's discussion of interfaces).

## Code Examples
```python
# Defining a simple class with __init__ and a method
class RepeatText():
    def __init__(self, n_repeats):
        self.n_repeats = n_repeats

    def multiply_text(self, some_text):
        print((some_text + " ") * self.n_repeats)

repeat_twice = RepeatText(n_repeats=2)
print(repeat_twice.n_repeats)         # 2
repeat_twice.multiply_text("hello")   # 'hello hello'
```
- **What it demonstrates**: basic class definition, attribute storage via `__init__`, and a bound method.

```python
# A data-holding class with a summary method
class Goal5Data():
    def __init__(self, name, population, women_in_parliament):
        self.name = name
        self.population = population
        self.women_in_parliament = women_in_parliament

    def print_summary(self):
        null_women_in_parliament = (len(self.women_in_parliament)
            - np.count_nonzero(self.women_in_parliament))
        print(f"There are {len(self.women_in_parliament)} data points for "
              f"Indicator 5.5.1, 'Proportion of seats held by women in "
              f"national parliaments'.")
        print(f"{null_women_in_parliament} are nulls.")

usa = Goal5Data(name="USA", population=336262544,
                women_in_parliament=[13.33, 14.02, 14.02])
usa.print_summary()
```
- **What it demonstrates**: packaging related data and behavior together for reuse and clarity.

```python
# Inheritance: extending a class with super()
from scipy.stats import linregress

class Goal5TimeSeries(Goal5Data):
    def __init__(self, name, population, women_in_parliament, timestamps):
        super().__init__(name, population, women_in_parliament)
        self.timestamps = timestamps

    def fit_trendline(self):
        result = linregress(self.timestamps, self.women_in_parliament)
        slope = round(result.slope, 3)
        r_squared = round(result.rvalue**2, 3)
        return slope, r_squared

india = Goal5TimeSeries(name="India", population=1417242151,
                        women_in_parliament=[9.02, 9.01, 8.84],
                        timestamps=[2000, 2001, 2002])
india.print_summary()      # inherited method still works
india.fit_trendline()      # (0.292, 0.869) — new child-only method
```
- **What it demonstrates**: child class reuses parent's `__init__`/methods via `super()` and adds new functionality without duplicating code.

```python
# Polymorphism: shared .fit() interface across unrelated classes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
```
- **What it demonstrates**: different classes exposing the same method name/contract, so calling code can swap implementations with minimal changes.

```python
# FP: lambda + map (does not mutate original list)
usa_govt_percentages = [13.33, 14.02, 14.02, 14.25]
usa_govt_proportions = list(map(lambda x: x / 100, usa_govt_percentages))
print(usa_govt_proportions)   # [0.1333, 0.1402, 0.1402, 0.1425]
```
- **What it demonstrates**: a pure transformation producing a new list, leaving the original untouched — core FP behavior.

```python
# Lambda vs. named function with pandas .apply()
df["USA_processed"] = df["United States of America"].apply(
    lambda x: "Mostly male" if x < 50 else "Mostly female")

# Preferred when reusable/testable:
def binary_labels(women_in_govt):
    if women_in_govt < 50:
        return "Mostly male"
    else:
        return "Mostly female"

df["USA_processed"] = df["United States of America"].apply(binary_labels)
```
- **What it demonstrates**: a named function is reusable, independently testable, and can hold more complex logic than a one-line lambda.

## Reference Tables
| OOP Term | Meaning | Data science example |
|---|---|---|
| Class | Blueprint for objects | `DataFrame`, `LogisticRegression` |
| Instance/Object | A specific thing made from a class | `df`, `clf` |
| Attribute | Data property (no parens to access) | `df.columns`, `df.shape` |
| Method | Behavior/function bound to instance (called with parens) | `df.to_csv()`, `clf.fit()` |
| Inheritance | Child class extends parent | `Goal5TimeSeries(Goal5Data)` |
| Encapsulation | Hides internals behind interface | DataFrame hides NumPy internals |
| Polymorphism | Same method name, different classes | `.fit()` on any scikit-learn estimator |

| Paradigm | Best fit | Python examples |
|---|---|---|
| OOP | Many instances of similar things, shared behavior, varying data | Custom classes, pandas, scikit-learn, Matplotlib (OO interface) |
| FP | Fixed data, many transformations, or parallelization | lambda, `map`, `filter`, list comprehensions, generators, `itertools` |
| Neither (plain scripts) | Small, one-off projects | Simple modular scripts |

## Worked Example
The `Goal5Data` → `Goal5TimeSeries` inheritance chain is the chapter's running worked example: `Goal5Data` stores a country's name, population, and `women_in_parliament` time series with a `print_summary()` method; `Goal5TimeSeries` inherits all of that via `super().__init__(...)`, adds a `timestamps` attribute, and layers on a new `fit_trendline()` method that uses `scipy.stats.linregress` to compute a slope and R² for the trend. Calling `india.print_summary()` shows the inherited behavior still works unchanged, while `india.fit_trendline()` shows new child-specific behavior — illustrating inheritance's core benefit: extend without duplicating.

## Key Takeaways
1. OOP organizes code around objects (data + behavior); use it when you have many instances of similar things needing consistent behavior but different data — not for a single instance.
2. Classes, methods, and attributes are the core vocabulary; `__init__` sets up attributes on creation, `self` refers to the instance.
3. Inheritance (`class Child(Parent): super().__init__(...)`) avoids duplicating closely related classes' logic; polymorphism (shared method names across unrelated classes, e.g., `.fit()`) keeps calling code uniform; encapsulation/abstraction protect a stable public interface over changeable internals.
4. FP favors pure, non-mutating functions — easy to test, easy to parallelize, enforces modularity; common Python FP tools are lambda functions, `map()`, `filter()`, comprehensions, generators.
5. Prefer named functions over lambdas once logic needs reuse, testing, or complexity beyond a single expression.
6. Vectorized/built-in operations beat `.apply()` with a custom function for simple numeric/string work; reserve `.apply()` for logic with no vectorized equivalent.
7. Choose the paradigm based on the shape of the problem (many similar instances → OOP; fixed data with many transforms/parallelization → FP) or team convention — neither is mandatory for small scripts.

## Connects To
- **Ch 1**: Modularity pillar — OOP and FP are both frameworks for decomposing code into smaller pieces.
- **Ch 3**: `.apply()` performance trade-offs vs. vectorized operations on DataFrames.
- **Ch 8 (Data Validation with Pydantic / interfaces)**: inheritance used for data validation classes; stable interfaces discussed further.
- **Ch 11 (APIs)**: "Adding Functionality to Your API" — inheritance applied to API classes.
- **External**: *Introducing Python* by Bill Lubanovic (O'Reilly) for deeper OOP coverage; "Why You Hate Matplotlib" article on Matplotlib's dual OO/MATLAB-style interfaces.
