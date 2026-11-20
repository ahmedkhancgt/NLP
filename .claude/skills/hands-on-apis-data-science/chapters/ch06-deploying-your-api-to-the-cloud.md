# Chapter 6: Deploying Your API to the Cloud

## Core Idea
Deploying an API publicly makes a project real and enables feedback, portfolio value, and downstream AI/analytics consumption, but it carries real cost and security responsibilities; the chapter walks through three deployment paths of increasing complexity: GitHub Codespaces port forwarding, Render (source-based PaaS), and Docker + AWS Lightsail (containerized).

## Frameworks Introduced
- **Deployment-target selection by learning goal**: choose a cloud host based on what you're optimizing for, not a single "best" platform.
  - When to use: picking a cloud provider for a prototype/portfolio project.
  - How: optimize for simplicity (Render) if the goal is "just ship it"; optimize for learning a specific technology (AWS, Docker, CI) if the goal is skill-building, accepting more setup time/cost.
- **Cost & Credential Control Checklist**: a standing discipline for any cloud deployment.
  - When to use: before and during any paid cloud service usage.
  - How: review pricing/use cost calculators before enabling services; prefer free tiers/trials; set budgets with email alerts; shut down or delete unused resources; use short-lived, minimally-scoped credentials; never commit credentials to source control; deactivate/rotate immediately if credentials leak.
- **Docker Containerization Workflow**: package an application with its runtime so it runs identically anywhere.
  - When to use: any deployment target that accepts container images (AWS Lightsail, most modern PaaS/IaaS).
  - How: write a `Dockerfile` (base image → working dir → copy deps manifest → install deps → copy app code/data → set `CMD` to launch the server) → `docker build -t <image> .` → test locally with `docker run --publish <host>:<container> --name <container> <image>` → push the built image to a cloud container registry/service → deploy a container from that image with the needed open ports.

## Key Concepts
- **The cloud**: the collective public-internet infrastructure of servers rented on demand; its core benefit over owning hardware is elastic, disposable capacity (spin up, use, delete).
- **Render**: a simple PaaS that deploys directly from a Git repository (build command + start command), with auto-deploy on repo changes; no containerization required.
- **Amazon Lightsail (Containers)**: a simplified AWS service for running Docker containers in the cloud, used here as the AWS deployment target.
- **Dockerfile**: a text file of build instructions (base image, working directory, file copies, dependency installation, startup command) used by `docker build` to produce an image.
- **Container image vs. container**: an image is the static, built artifact (filesystem + run parameters); a container is a running instance of that image, created by a container runtime (Docker).
- **Container registry/repository**: a stored collection of container images (local Docker repository, or a cloud-hosted one like Lightsail's image store) that a deployment pulls from.
- **.dockerignore**: analogous to `.gitignore`; excludes files (e.g., `README.md`, `__pycache__`) from being copied into the build context/image.
- **Build context**: the set of files available to the `docker build` command (here, the `chapter6` directory), from which `COPY` statements pull files into the image.
- **AWS CLI**: command-line tool for interacting with AWS services (used here to push the local Docker image to Lightsail and deploy it).
- **Multifactor authentication (MFA)** and root vs. administrative AWS accounts: security practices flagged for any real AWS usage, even for learning projects.
- **Time to Hello World** (carried over from Ch. 5): deployment ease factors into how quickly a working, shareable API exists.

## Mental Models
- **"If your work isn't online, it doesn't exist"** (Austin Kleon epigraph): deployment is itself a milestone — it enables sharing, feedback, and AI/tool consumption that a local dev environment cannot.
- **Three deployment environments, one increasing-complexity ladder**: Codespaces port-forwarding (zero setup, ephemeral) → Render (minimal setup, source-based, persistent) → Docker + AWS Lightsail (most setup, portable, closest to real production practice) — each teaches a different piece of real-world deployment skill.
- **Containers as "a brand-new virtual server you fully control"**: the Docker image captures exact library versions and OS environment, eliminating "works on my machine" drift between dev and prod.

## Anti-patterns
- **Treating a free tier as risk-free**: cloud bills can surprise even careful users; always attach monitoring/budgets even on "free" services, since most require a card on file.
- **Committing credentials to source control**: explicitly called out multiple times as the main security risk of cloud deployment — leaked keys can be used for crypto mining or botnets at the account owner's expense.
- **Using `fastapi run`/`fastapi dev` CLI wrappers in production deployment**: both Render's start command and the Dockerfile's `CMD` call Uvicorn directly (`uvicorn main:app --host 0.0.0.0 --port ...`) instead of the `fastapi-cli` convenience wrapper used in local dev.
- **Leaving unused cloud resources running**: forgetting to delete Lightsail containers, Render services, or AWS resources after use continues to incur cost.
- **Using the AWS root account for ongoing work**: acceptable for this one-off learning project, but the chapter flags it as not best practice long-term — create a separate administrative account instead.

## Code Examples
```dockerfile
# Dockerfile for Chapter 6
FROM python:3.10-slim
WORKDIR /code
COPY requirements.txt /code/
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
COPY *.py /code/
COPY *.db /code/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```
- **What it demonstrates**: a minimal, layered Dockerfile — base image, working directory, dependency install (cached as its own layer), app code/data copy, and the container's default startup command.

```bash
# Build, run, and verify locally
docker build -t apicontainerimage .
docker images
docker run --publish 80:80 --name apicontainer1 apicontainerimage
```
- **What it demonstrates**: building a named image from the Dockerfile in the current directory, listing local images, and running a container that maps container port 80 to host port 80.

```bash
# Push to AWS Lightsail and verify AWS CLI identity
aws sts get-caller-identity
aws lightsail push-container-image --region us-west-2 \
  --service-name aws-api-container --label aws-api --image apicontainerimage:latest
```
- **What it demonstrates**: verifying AWS CLI credentials, then pushing a local Docker image into a Lightsail container service's image store for deployment.

```bash
# Render deployment settings (as form fields, not a script)
# Build command: pip install -r requirements.txt
# Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```
- **What it demonstrates**: Render's source-based deployment model — it builds and starts your app directly from the repo without a Dockerfile, using `$PORT` injected by the platform.

## Reference Tables
**Table 6-1. New tools/services used in this chapter**

| Software/service | Version | Purpose |
|---|---|---|
| Amazon Lightsail | N/A | AWS virtual cloud server / container service |
| AWS CLI | 2.15 | Command-line interface for AWS |
| Docker | 24.0 | Package/run the app in a container |
| Render | N/A | Cloud hosting provider (PaaS) |

**Table 6-2. Docker commands used**

| Command | Purpose |
|---|---|
| `docker --version` | Verify installed version |
| `docker build -t <name> .` | Build an image from a Dockerfile |
| `docker images` | List local images |
| `docker run` | Run a container from a local image |

**Deployment path comparison**

| Target | Setup effort | Source | Persistence | Teaches |
|---|---|---|---|---|
| GitHub Codespaces (public port) | Minimal | Running dev container | Only while Codespace runs | Quick sharing/demoing |
| Render | Low | Git repo (build/start commands) | Persistent, auto-deploys on push | Simple PaaS deployment |
| AWS Lightsail + Docker | Higher | Pushed container image | Persistent | Containerization, AWS CLI, portable deployment |

## Worked Example
Starting from Chapter 5's completed API, the chapter first exposes the running Codespace's port 8000 publicly for a quick demo. It then deploys to Render by connecting a public GitHub repo, setting root directory `chapter6`, build command `pip install -r requirements.txt`, and start command `uvicorn main:app --host 0.0.0.0 --port $PORT`; Render builds, deploys, and gives a public `onrender.com` URL, and auto-redeploys when the health-check message is edited and pushed. Finally, the chapter builds a Docker image from a hand-written `Dockerfile` (Python 3.10-slim base, copy `requirements.txt`, `pip install`, copy `*.py`/`*.db`, `CMD` to launch Uvicorn on port 80), runs it locally to confirm it works, then uses the AWS CLI and Lightsail Container Services plug-in to push the image to a new `aws-api-container` Lightsail service, opens port 80, and deploys — yielding a public AWS domain serving the same health-check and `/docs` endpoints. The chapter closes by updating `README.md`'s "Public API" section with the live URL, `/docs`, and `/openapi.json` links.

## Key Takeaways
1. Public deployment is a meaningful milestone, not just a technical step — it enables feedback, portfolio sharing, and downstream tool/AI consumption (set up in Parts II and III).
2. Treat cost and credential security as first-class concerns from day one: budgets/alerts, free tiers, minimal-scope short-lived keys, and never committing secrets to git.
3. Match the deployment platform to your current goal: Render for fast, low-friction shipping; Docker + AWS Lightsail for learning containerization and a major cloud provider's tooling.
4. A Dockerfile captures the exact runtime environment (base image, dependencies, code, startup command), making deployments reproducible across local, Render, and AWS targets.
5. Both deployment paths explicitly run Uvicorn directly (not the `fastapi` CLI wrapper) because production hosting needs explicit host/port/bind control.
6. Update your documentation (README) with the live API URL, `/docs`, and `/openapi.json` once deployed — docs aren't complete until consumers have a real address to hit.

## Connects To
- **Ch 5**: The README's "Public API" placeholder from Chapter 5 gets filled in here once a real URL exists.
- **Ch 7**: The deployed API becomes the target the Python SDK will call.
- **Part II (Ch 9–11)**: The publicly deployed API is the one subsequently consumed in analytics notebooks, data pipelines, and Streamlit apps.
- **Part III (Ch 14–15)**: Public deployment is a prerequisite for LLM/ChatGPT-based consumption of the API via its OAS file.
- **External**: Docker's official documentation and cheat sheet; Snyk.io's "Best practices for containerizing Python applications with Docker"; AWS Identity and Access Management (IAM) User Guide; Render's "Deploy a FastAPI App" guide.
