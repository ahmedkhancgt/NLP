# Chapter 10: Sharing Your Code: Version Control, Dependencies, and Packaging

## Core Idea
Sharing code effectively with others — teammates or the open-source community — requires three complementary systems: version control (Git) to track and coordinate changes, dependency management (pip/Poetry + virtual environments) to ensure others can reproduce your environment, and packaging (pyproject.toml + a build tool + PyPI) to make your code easily installable.

## Frameworks Introduced
- **Git workflow (add → commit → push/pull)**: the core local/remote collaboration loop.
  - When to use: continuously while developing, especially before sharing work or at logical checkpoints.
  - How: `git init` once per project; `git add <file>` to stage; `git commit -m "..."` to snapshot locally; `git remote add origin <url>` + `git push` to sync to a central server; `git pull` to fetch others' changes; `git clone <url>` to start from an existing remote repo.
- **Branch → pull request → review → merge**: the standard collaborative-change workflow.
  - When to use: whenever multiple people work on the same codebase, or you want isolated, reviewable changes.
  - How: `git checkout -b <branch>` to create/switch; commit changes on the branch; `git push origin <branch>`; open a pull request describing *what* and *why*; get it reviewed; `git checkout main` + `git merge <branch>`; delete the branch after merging.
- **Virtual-environment-per-project discipline**: isolate each project's dependencies instead of installing to the system Python.
  - When to use: at the start of every new project, before installing any third-party library.
  - How: create an environment (`venv`, `virtualenv`, `pyenv`, `conda`, or via Poetry/PDM/Hatch), activate it, install only what that project needs, deactivate when done; never install libraries system-wide/root.
- **Packaging pipeline**: configuration file → build tool → distribution upload.
  - When to use: when you want others (teammates or the public) to `pip install` and `import` your code like any other library.
  - How: (1) ensure code is functional, formatted, tested, documented (Ch 6/7/9) and organized with an `src/<package_name>/` layout; (2) write/generate a `pyproject.toml` with build-system + project metadata; (3) run a build tool (`build`/setuptools, or `poetry build`) to produce sdist + wheel artifacts; (4) upload via Twine (`twine upload`, testing first on TestPyPI) or `poetry publish`.

## Key Concepts
- **Version control**: a system tracking changes to a codebase over time, enabling multi-person collaboration, conflict prevention, change attribution, and easy rollback. Should be distributed (local + central copies) so code is always backed up.
- **Git**: the dominant distributed version control system (Linus Torvalds, 2005; ~96% of professional developers per the 2022 Stack Overflow survey). GitHub/GitLab/Bitbucket are hosting platforms for Git repos (GitHub ≠ Git).
- **Repository (repo)**: a directory Git tracks via a hidden `.git` folder containing a snapshot database of all changes.
- **Staging area**: the holding area for changes you intend to include in the next commit (`git add`); changes not staged aren't saved in a snapshot.
- **Commit**: a saved snapshot of staged changes, with a unique hash ID, author, timestamp, and message; best practice is one logical change per commit (small, focused, easy to trace if it later causes a problem), ideally after running tests.
- **Local vs. remote repository**: local = on your machine; remote = central server copy (e.g., GitHub); `push` sends local commits to remote, `pull` fetches others' remote commits into local, `clone` downloads an entire remote repo as a new local one.
- **Branch**: an isolated line of change from a point in the project's history, letting you develop features without affecting the main line; default branch conventionally named `main` (not `master`, due to slavery connotations). Merging incorporates a branch's commits back into another branch (e.g., `main`).
- **Merge conflict**: arises when the same line was changed differently on two branches being merged; must be manually resolved. Notebooks are especially prone to this because their JSON format tracks cell outputs too — mitigated by clearing outputs, using `nbdime` (compares only code, not outputs), or converting to `.py`/`.md` via Jupytext.
- **Pull request (PR)**: a request for another developer to review changes before merging; should explain *what* changed and *why*, to focus the reviewer's attention.
- **Code review**: a structured practice of reviewing code (often via PRs) before it ships, to catch bugs/security risks, share knowledge, and record decisions; best practices include covering readability/simplicity/modularity, giving everyone a voice, highlighting strengths not just problems, and keeping reviews time-bounded.
- **Dependency**: a third-party library your code requires (e.g., pandas, NumPy); your code may need a specific version or range.
- **Versioning / SemVer**: version numbers (e.g., `1.24.3`) following Semantic Versioning — major.minor.patch, where major = breaking changes, minor = new features without breaking changes, patch = small fixes (e.g., bug fixes only); Calendar Versioning is an alternative scheme.
- **Virtual environment**: an isolated space for a project's installed libraries, separate from the system-wide Python install, so different projects can use different (even conflicting) library versions without interference. Tools: `venv` (stdlib), `virtualenv` (older, more features), `pyenv` (manages multiple Python versions, + `pyenv-virtualenv` plugin), `Conda` (manages envs + installs packages from its own repository, not PyPI), Poetry/PDM/Hatch (dependency + env + packaging management combined).
- **requirements.txt**: pip's simple dependency-list format, generated via `pip freeze > requirements.txt` inside an active virtual environment, and installed elsewhere via `pip install -r requirements.txt`. Downsides: no automatic sub-dependency pruning when removing a library, no Python-version pinning.
- **pyproject.toml**: the standardized project configuration/build-metadata file (TOML format); used by Poetry (and other modern tools) in place of `requirements.txt`, and also doubles as the packaging configuration file.
- **poetry.lock**: stores exact resolved versions (with hashes) of all dependencies for reproducibility across operating systems; regenerated on each `poetry add`/install; should be committed to version control.
- **PyPI (Python Package Index)**: the standard public repository for installable Python packages (473,661+ projects at writing time); packages can also be distributed privately within a company.
- **Python module vs. package (two senses)**: a module = one `.py` file; a package = either (a) a directory with `__init__.py` enabling imports from it, or (b) a *distribution package* — the installable, PyPI-uploadable unit (the sense used throughout packaging discussion).
- **Build artifacts**: `sdist` (zipped source code archive) and `wheel` (zipped, pip-installable binary-ish archive) — the two files a build tool typically produces.
- **Twine**: the tool for uploading build artifacts to PyPI (or TestPyPI first, to verify before the real upload).

## Mental Models
- **Version control as a backup + collaboration substrate**: beyond "undo," Git's distributed nature means every clone is itself a backup, and the add/commit/push/pull cycle is what makes simultaneous multi-person work tractable without constant conflict.
- **Branches as parallel, non-destructive timelines**: a branch doesn't copy files — it's a tracked divergence point — so you and a teammate can both change `main`-derived code simultaneously without either of you blocking the other, merging only when ready.
- **Keep data out of code version control**: Git is built for tracking text/code changes, not large binary data files — conflating the two risks bloat, duplication, and security/privacy exposure; use a dedicated data-versioning system instead.
- **Isolation as the cure for "works on my machine"**: virtual environments exist because two projects needing different versions of the same library is a *normal*, expected situation, not an edge case — the fix is isolation per project, never upgrading/downgrading the system-wide install.
- **Packaging turns "code I wrote" into "a tool others can depend on"**: the shift from a script to a package is the shift from "run this file" to "`pip install` this and `import` it anywhere," which comes with an implicit ongoing maintenance responsibility to those who adopt it.
- **pyproject.toml as the single source of project truth**: one file capturing both *what this project needs to run* (dependencies) and *how to build it into something installable* (build-system), replacing the older, narrower `requirements.txt` approach.

## Anti-patterns
- **Installing libraries in the system-wide (root) Python environment**: causes version conflicts across projects and can break the operating system's own Python-dependent tooling if packages are added/removed outside the system package manager.
- **Storing data files in a code version control system**: Git isn't designed for large files and this risks unnecessary duplication and security/privacy exposure — use a dedicated data-versioning system.
- **Using `master` as the default branch name**: discouraged due to its slavery connotations; `main` is now the convention.
- **Making huge, multi-purpose commits**: harder to trace which change caused a later bug; prefer small, single-purpose commits.
- **Committing without running tests first**: risks saving broken code as a checkpoint (Ch 12 covers automating this check).
- **Letting notebook cell outputs cause merge-conflict noise**: forgetting to clear notebook outputs before committing creates spurious diffs/conflicts; address with `nbdime`, Jupytext, or an automated clearing step (Ch 12).
- **Manually tracking sub-dependencies with pip/requirements.txt**: removing a library doesn't auto-remove its now-unneeded dependents from `requirements.txt` — use a smarter tool (Poetry/PDM/Hatch) if this matters to you.
- **Skipping TestPyPI before a real PyPI upload**: risks publishing a broken or misconfigured package to the permanent public index; PyPI's own docs recommend testing uploads first.

## Code Examples
```bash
# Basic Git local workflow
$ git init
$ git add README.md
$ git status
$ git commit -m "Initial commit"

$ git add api_functions.py
$ git commit -m "Created helper functions for API"
```
- **What it demonstrates**: initializing tracking, staging specific files, and committing focused, well-described snapshots.

```bash
# Connecting to and syncing with a remote repository
$ git remote add origin https://github.com/drcat101/SEforDS.git
$ git push -u origin main

# Starting from an existing remote repo instead:
$ git clone https://github.com/drcat101/SEforDS.git
```
- **What it demonstrates**: linking a local repo to a remote one and pushing commits, or cloning an existing project to start contributing.

```bash
# Branching workflow
$ git checkout -b new_branch      # create + switch in one step
# ... make changes, commit ...
$ git push origin new_branch

$ git checkout main
$ git merge new_branch            # incorporate the branch's changes
```
- **What it demonstrates**: isolating feature work on a branch, then merging it back into `main` once ready.

```bash
# venv: create, activate, deactivate a virtual environment
$ python -m venv SEforDS
$ source SEforDS/bin/activate
(SEforDS)$ pip install pandas
(SEforDS)$ deactivate
```
- **What it demonstrates**: per-project isolation of installed libraries using the standard-library `venv` tool.

```bash
# pip-based dependency recording and restoration
$ python -m pip freeze > requirements.txt
$ pip install -r requirements.txt
```
```text
numpy==1.25.2
pandas==2.0.3
pytz==2023.3
```
- **What it demonstrates**: pip's simple snapshot-and-restore approach to recording exact installed versions.

```bash
# Poetry: project + dependency management
$ poetry new SE_for_DS          # new project scaffold
$ poetry init                   # or initialize in an existing folder
$ poetry add pandas             # adds + installs a dependency
$ poetry shell                  # activate the managed virtual environment
$ poetry install                # install dependencies from poetry.lock
```
```toml
[tool.poetry.dependencies]
python = "^3.10"
pandas = "^2.1.0"
```
- **What it demonstrates**: Poetry manages both the `pyproject.toml` dependency spec and a `poetry.lock` file of exact, hashed versions, more robust than plain pip/requirements.txt.

```toml
# pyproject.toml for packaging (setuptools-based)
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "se_for_ds"
version = "0.0.1"
authors = [{ name="Catherine Nelson", email="email_address" }]
description = "An example package for Software Engineering for Data Scientists"
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
```
- **What it demonstrates**: the metadata + build-backend declaration a packaging tool reads to turn source code into an installable artifact.

```bash
# Building and publishing a package
$ pip install build
$ python3 -m build
# Successfully built se_for_ds-0.1.0.tar.gz and se_for_ds-0.1.0-py3-none-any.whl

$ pip install twine
$ twine upload -r testpypi dist/*   # test first
$ twine upload dist/*               # then the real PyPI

# Or, if using Poetry as the build system:
$ poetry build
$ poetry publish
```
- **What it demonstrates**: the standard build → test-upload → real-upload pipeline, with Poetry offering a streamlined one-tool alternative to setuptools + build + twine.

## Reference Tables
| Git command | Purpose |
|---|---|
| `git init` | Start tracking a directory |
| `git add <file>` | Stage changes |
| `git commit -m "..."` | Save a snapshot locally |
| `git push` / `git pull` | Sync local ↔ remote |
| `git clone <url>` | Download a remote repo as new local |
| `git checkout -b <branch>` | Create + switch to a new branch |
| `git merge <branch>` | Incorporate a branch into the current one |

| SemVer position | Meaning |
|---|---|
| Major (X.0.0) | Breaking changes |
| Minor (0.X.0) | New features, no breaking changes |
| Patch (0.0.X) | Small fixes |

| Virtual env / dependency tool | Scope |
|---|---|
| `venv` | Stdlib; environment isolation only |
| `virtualenv` | Older, superset of venv's features |
| `pyenv` (+ pyenv-virtualenv) | Multiple Python *versions* + envs |
| Conda | Envs + packages, from Anaconda's own repo (not PyPI) |
| Poetry / PDM / Hatch | Envs + dependencies + packaging, all-in-one |

| Packaging term | Meaning |
|---|---|
| sdist | Zipped source-code archive |
| wheel | Zipped, pip-installable build artifact |
| PyPI / TestPyPI | Public package index / its test counterpart |

## Worked Example
Setting up the book's own example project (`SE_for_DS`) end to end: `poetry init` (or `poetry new SE_for_DS`) creates `pyproject.toml` + `poetry.lock`; `poetry add pandas` records `pandas = "^2.1.0"` in `pyproject.toml` and pins the exact resolved version/hash in `poetry.lock`; the source code lives at `src/SE_for_DS/` alongside `tests/` and `docs/` folders and a top-level `README.md`/`LICENSE`; once the code is formatted (Ch 6), tested (Ch 7), and documented (Ch 9), `poetry build` produces `se_for_ds-0.1.0.tar.gz` and a `.whl` file, and `poetry publish` uploads them to PyPI — letting anyone `pip install se-for-ds` and `import` it exactly like NumPy or pandas.

## Key Takeaways
1. Git (add → commit → push/pull) is the near-universal standard for tracking code changes and enabling safe, backed-up, multi-person collaboration.
2. Use branches to isolate work and pull requests + code review to get changes reviewed before merging into `main`; keep commits small and focused.
3. Never store data files in code version control, and never store notebook cell outputs if you can avoid merge-conflict noise (clear outputs, use nbdime, or convert via Jupytext).
4. Always use a virtual environment per project — never install third-party libraries system-wide — to avoid cross-project version conflicts.
5. Record dependencies so others can reproduce your environment: pip's `requirements.txt` (simple but manual sub-dependency management) or a full dependency manager like Poetry (`pyproject.toml` + `poetry.lock`, with exact hashed versions for cross-platform reproducibility).
6. Semantic Versioning (major.minor.patch) communicates the nature of a library update — major = breaking, minor = new features, patch = small fixes.
7. Packaging (pyproject.toml → build tool → PyPI/Twine, or Poetry's `build`/`publish`) turns code into something others can `pip install` and import — but code should be functional, formatted, tested, and documented first, and publishing implies an ongoing maintenance responsibility.

## Connects To
- **Ch 6**: Formatted code is a packaging prerequisite.
- **Ch 7**: Tested code is a packaging prerequisite; tests should run before committing.
- **Ch 8**: Refactoring workflow explicitly ends each safe change with a version-control commit.
- **Ch 9**: Documentation (README.md) is a packaging prerequisite and lives alongside code in the repo.
- **Ch 12**: Automation — running tests before commits, clearing notebook outputs automatically, and CI/CD pipelines that build on Git workflows.
- **External**: Git/GitHub documentation; the Dangit, Git website; Tim Hopper's blog post on data science code reviews; *Hypermodern Python Tooling* by Claudio Jolowicz; PyPI/TestPyPI/Conda/conda-forge documentation.
