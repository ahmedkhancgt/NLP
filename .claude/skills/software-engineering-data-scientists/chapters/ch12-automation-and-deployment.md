# Chapter 12: Automation and Deployment

## Core Idea
Automating repetitive workflow steps (formatting, linting, testing) via pre-commit hooks and CI/CD (e.g., GitHub Actions) makes code quality standardized and reproducible with minimal ongoing effort, and packaging code into Docker containers makes deployment to a cloud provider (and the resulting running API) reproducible as well.

## Frameworks Introduced
- **CI/CD pipeline (Continuous Integration / Continuous Delivery / Continuous Deployment)**: the staged automation of build→test→(review)→deploy.
  - When to use: for codebases deployed frequently, with a team committed to tests and version control.
  - How: Continuous Integration = every commit triggers an automated build + test run with alerts on failure; Continuous Delivery = after CI passes, code is deploy-ready but a human manually triggers deployment; Continuous Deployment = passing code is deployed to production automatically, no human step.
- **Pre-commit hooks**: lightweight, local Git-triggered automation for small, frequent tasks.
  - When to use: formatting, linting, stripping debug statements/notebook outputs — anything you want enforced on every single commit, fast enough not to slow you down.
  - How: `pip install pre-commit`; write `.pre-commit-config.yaml` listing hook repos/ids (or a `local` hook with a custom `entry` command); `pre-commit install`; hooks then run automatically on `git commit`, and can also be run on demand (`pre-commit run --all-files`).
- **GitHub Actions CI workflow**: cloud-run automation triggered by repository events (push, PR).
  - When to use: automatically running your full test suite (or longer-running checks) on every push, especially when tests take too long for a pre-commit hook or need to run in a clean environment.
  - How: `.github/workflows/<name>.yaml` defining `name`, `on` (trigger, e.g. push to `main`), and `jobs` (runner OS, then `steps`: checkout code, set up Python, install dependencies, run tests) — runs on GitHub's servers, reports pass/fail in the repo's Actions tab and via email.
- **Containerized cloud deployment**: package code + dependencies into a reproducible unit, then run it on a cloud provider.
  - When to use: deploying an API (or any service) so it runs continuously, restarts after failure, and behaves identically regardless of host machine.
  - How: write a `Dockerfile` (base image → copy requirements → install deps → copy code → run command); `docker build` an image; test with `docker run` locally; push the image to a cloud container registry (Google Artifact Registry / Azure Container Registry / AWS Elastic Container Registry); deploy from the registry to a container-running service (Google Cloud Run / Azure App Service / AWS Elastic Container Service), which exposes a public URL.

## Key Concepts
- **Deploying code**: making a "done" (satisfied-with-for-now) version of code operational for end users — contrasted with development, where ideas are still being tried out. Typically involves a staging/testing environment that mirrors production before the real production deploy.
- **CI (Continuous Integration) server**: automatically builds the project and runs tests on every commit, alerting developers to failures.
- **Continuous Delivery vs. Continuous Deployment**: Delivery keeps a manual human approval gate before deployment; Deployment removes it — passing code ships automatically.
- **Git hook**: a custom script Git runs before/after a specific action (e.g., a commit); pre-commit hooks run *before* the commit completes, letting you catch problems first.
- **`.pre-commit-config.yaml`**: YAML configuration listing which hooks (by repo + revision + hook id) run on each commit; pre-commit hooks run in their own isolated virtual environment (slow on first run, fast afterward).
- **YAML**: a human-readable markup language commonly used for configuration files (used by both pre-commit and GitHub Actions).
- **Prebuilt pre-commit hooks**: e.g., the `debug-statements` hook (removes leftover `pdb` breakpoints, connecting to Ch 5) or the Black formatting hook; custom/local hooks can run arbitrary shell/Python commands, e.g., stripping Jupyter notebook outputs via `jupyter nbconvert --clear-output --inplace` (connecting to Ch 10's notebook merge-conflict problem).
- **GitHub Actions**: a CI/CD platform tightly integrated with GitHub, triggered by repository events; workflows run on GitHub-hosted (or self-hosted) runners, not your own machine — useful for long tests and for catching missing-dependency issues since the environment starts clean. Has a free tier with usage-based costs beyond it.
- **GitHub Actions workflow structure**: `name` (workflow name) → `on` (trigger event/branch) → `jobs` (named jobs, each with `runs-on` an OS image and a `steps` list — prebuilt actions like `actions/checkout@v4`/`actions/setup-python@v4` use `uses:`, custom shell commands use `run:`).
- **Container**: an isolated, reproducible environment (like a small Linux system on a larger host) bundling your code with its exact runtime/library dependencies.
- **Docker**: the system for building/managing containers; a container is instantiated from an **image**, which is built from a **Dockerfile** (a text blueprint of setup instructions) — often starting `FROM` a prebuilt base image (official Python, FastAPI, TensorFlow, PyTorch images, etc.).
- **Dockerfile instructions**: `FROM` (base image), `COPY` (copy files into the image — copying `requirements.txt` first and installing before copying the rest speeds up rebuilds via layer caching), `RUN` (execute a build-time command, e.g., `pip install`), `CMD` (the command executed when a container starts, e.g., launching Uvicorn).
- **Image vs. container**: an image is the static template/blueprint; a container is a running (or stopped) instance of that image — you can launch multiple independent container instances from one image, each with its own runtime state.
- **Container registry**: a cloud service that stores Docker images (Google Artifact Registry, Azure Container Registry, AWS Elastic Container Registry) without running them.
- **Container-running service**: the cloud service that actually deploys/runs a container from the registry and exposes it to the internet (Google Cloud Run, Azure App Service, AWS Elastic Container Service), assigning a public URL.

## Mental Models
- **"Automation is mostly front-loaded complexity"**: the setup step for pre-commit/CI/Docker can feel intimidating, but once configured, these systems "run without any action needed from you" — the investment is one-time, the payoff recurring.
- **Automation as a culture-shaping force, not just a time-saver**: if checks run automatically, "everyone is more likely to run them," reducing team-wide mistakes and establishing a shared standard — automation is a social/process tool as much as a technical one.
- **Dockerfile as a blueprint, container as the building**: the same blueprint (image) can be used to construct many identical, independent buildings (container instances) — reproducibility comes from everyone building from the same blueprint, not from manually replicating a specific instance.
- **Pre-commit (local, fast, per-commit) vs. CI/CD (remote, thorough, per-push)**: a deliberate division of labor — fast, cheap checks (formatting, linting, stripping debug code) run on every commit locally; slower, more complete checks (full test suite, clean-environment dependency verification) run remotely on push, where waiting doesn't block your local work.
- **Separate environments as a safety buffer**: testing/staging environments let code be exercised against "the rest of the software product" without risking the live, user-facing production system — a risk-reduction pattern independent of any specific tool.

## Anti-patterns
- **Manually remembering to format/lint/strip debug code before every commit**: error-prone and easy to forget — exactly what pre-commit hooks exist to eliminate ("this is much easier than remembering to manually remove data from your notebooks").
- **Running a full, slow test suite as a pre-commit hook on a large project**: pre-commit hooks should stay lightweight; for large test suites, use a CI/CD system (GitHub Actions) instead so you're not blocked locally.
- **Deploying straight to a Docker container without testing locally first**: building and running containers is time-consuming, so test your code directly before going through the containerization step — otherwise you waste cycles rebuilding/redeploying to catch a simple bug.
- **No logging setup inside a deployed container**: without it, you lose easy access to error messages once code is running remotely; set up logging (Ch 5) before relying on a containerized deployment.
- **Leaving cloud resources running unmonitored**: incurs ongoing monetary cost; set up billing alerts and remember to stop/delete resources when not in use.
- **Skipping company security review before deploying with customer data**: the chapter explicitly flags that its simple personal-exploration deployment process is not necessarily compliant with a company's security policies for real customer data (foreshadowing Ch 13).

## Code Examples
```yaml
# .pre-commit-config.yaml: run Black on every commit
repos:
  - repo: https://github.com/psf/black-pre-commit-mirror
    rev: 23.10.1
    hooks:
      - id: black
        language_version: python3.10
```
```bash
$ pip install pre-commit
$ pre-commit install
$ git commit -m "..."
# black.......................................Failed
# - files were modified by this hook
# reformatted se_for_ds/chapter_11_functions.py
```
- **What it demonstrates**: installing a hook that auto-formats code at commit time; a "Failed" status here means files were *changed* by the hook, requiring you to re-add and re-commit to review/accept the formatting.

```yaml
# Adding a prebuilt hook (removes pdb debug statements) and a custom local hook
repos:
  - repo: https://github.com/psf/black-pre-commit-mirror
    rev: 23.10.1
    hooks:
      - id: black
        language_version: python3.10
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.3.0
    hooks:
      - id: debug-statements
  - repo: local
    hooks:
      - id: remove-notebook-output
        name: Remove notebook output
        description: Strips out any output or data from Jupyter Notebook cells
        language: system
        files: \.ipynb$
        entry: jupyter nbconvert --clear-output --inplace
```
- **What it demonstrates**: combining a prebuilt hook (catches leftover `breakpoint()` calls from Ch 5) with a custom local hook (strips notebook outputs, addressing Ch 10's merge-conflict problem) in one config.

```yaml
# .github/workflows/run-tests.yaml: CI workflow to run Pytest on every push to main
name: run-tests

on:
  push:
    branches: [ "main" ]

jobs:
  test:
    name: Run all tests for trendline code
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install requirements
        run: python -m pip install -r requirements.txt

      - name: Run tests
        run: python -m pytest
```
- **What it demonstrates**: a complete, minimal GitHub Actions workflow — checkout code, set up Python, install deps from `requirements.txt`, run the test suite — reported as a green check or red cross in the repo's Actions tab.

```dockerfile
# Dockerfile for the Chapter 11 trendline API
FROM python:3.10

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
$ docker build -t trendline_image .
$ docker run -d --name trendline_image -p 8000:8000 trendline_container
# API now reachable at http://127.0.0.1:8000/country_trendline/India
$ docker stop trendline_container
```
- **What it demonstrates**: building a reproducible image (base Python image, install deps before copying all code to leverage layer caching, run Uvicorn as the container's main process) and running/stopping a local container instance from it.

## Reference Tables
| CI/CD stage | Human step? | Outcome |
|---|---|---|
| Continuous Integration | No (alerts on failure) | Build + test run on every commit |
| Continuous Delivery | Yes (manual deploy trigger) | Code is deploy-ready, human approves |
| Continuous Deployment | No | Passing code deployed automatically |

| Automation layer | Runs where | Best for |
|---|---|---|
| Pre-commit hooks | Locally, on `git commit` | Fast, frequent checks: formatting, linting, stripping debug code/notebook output |
| GitHub Actions (CI/CD) | GitHub's servers, on push/PR | Full test suites, clean-environment dependency checks, longer-running jobs |

| Cloud provider | Container registry | Container-running service |
|---|---|---|
| Google Cloud | Artifact Registry | Cloud Run |
| Microsoft Azure | Azure Container Registry | Azure App Service |
| AWS | Elastic Container Registry | Elastic Container Service |

| Dockerfile instruction | Purpose |
|---|---|
| `FROM` | Base image to build on |
| `COPY` | Copy files into the image |
| `RUN` | Execute a command during image build (e.g., install deps) |
| `CMD` | Command executed when the container starts |

## Worked Example
Deploying the Chapter 11 trendline API end to end: the API code (`main.py`, `chapter_11_functions.py`) plus a `requirements.txt` and `Dockerfile` live in one folder; `docker build -t trendline_image .` builds the image (installing dependencies from `requirements.txt` before copying the rest of the code, so dependency-layer caching speeds up future rebuilds); `docker run -d --name trendline_image -p 8000:8000 trendline_container` runs it locally, reachable at `http://127.0.0.1:8000/country_trendline/India` exactly as in Chapter 11, but now inside a reproducible container. The same image is then pushed to Google Artifact Registry and deployed via Google Cloud Run, which assigns a public URL; Cloud Run's logs show a `200 OK` confirming the deployed API is reachable from anywhere on the internet, completing the path from local script to publicly callable cloud service.

## Key Takeaways
1. Automation (pre-commit hooks, CI/CD) trades one-time setup complexity for ongoing standardization, reproducibility, and time savings — especially valuable with frequent deployments or team-wide adoption.
2. Pre-commit hooks run lightweight, frequent checks (formatting via Black, linting, stripping `pdb` debug statements, clearing notebook outputs) automatically on every `git commit`, configured via `.pre-commit-config.yaml`.
3. GitHub Actions (or similar CI/CD tools — Jenkins, Travis CI, CircleCI) runs heavier checks like full test suites on GitHub's own servers whenever you push, catching issues (including missing dependencies) that a clean environment reveals and a local pre-commit hook couldn't practically catch.
4. CI/CD distinguishes Continuous Integration (automated build+test), Continuous Delivery (tested code awaiting manual deploy approval), and Continuous Deployment (fully automatic deploy on passing tests) — and is increasingly applied to ML pipelines, triggered by data or model-performance changes as well as code changes.
5. Docker containers package code with its exact dependencies into a reproducible, isolated unit, built from a Dockerfile (a blueprint) into an image, from which independent container instances can be run.
6. Cloud deployment follows a common three-step pattern across providers: build and push a container image to a registry, then deploy it via a container-running service that exposes a public URL.
7. Test code locally before containerizing, set up logging before deploying (you'll lose easy access to error output otherwise), and monitor/limit cloud costs (billing alerts, stopping unused resources).

## Connects To
- **Ch 5**: `breakpoint()`/debug statements removed automatically via a pre-commit hook; logging recommended before deployment.
- **Ch 6**: Black formatting automated via pre-commit, as introduced manually in Ch 6.
- **Ch 7**: Pytest test suite automated here via GitHub Actions.
- **Ch 10**: Version control (Git) is a prerequisite for both pre-commit hooks and GitHub Actions; `requirements.txt` dependency management reused in both the Actions workflow and the Dockerfile; notebook-output merge-conflict problem solved via a custom pre-commit hook.
- **Ch 11**: The trendline API built in this chapter is the exact artifact containerized and deployed here.
- **Ch 13**: Security practices flagged as essential before exposing a deployed API/container publicly, especially with customer data.
- **External**: Mariatta Wijaya's talk on CI; Hamel Husain's course on CI/CD for machine learning; GitHub Actions documentation; Docker documentation; Google Cloud / AWS / Azure documentation for container registries and run services.
