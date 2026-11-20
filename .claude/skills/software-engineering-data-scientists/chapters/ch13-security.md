# Chapter 13: Security

## Core Idea
Data scientists handle sensitive data and deploy systems that can be attacked, so understanding core security terminology, common risk vectors (credentials, third-party packages, pickle, version control leaks, API exposure), and both general and ML-specific mitigation practices is essential — even without being a security expert.

## Frameworks Introduced
- **The Security Triad (C-I-A)**: Confidentiality, Integrity, Availability — a lens for evaluating any system's security posture.
  - When to use: assessing what "secure" means for a given system or reasoning about what a specific attack threatens.
  - How: ask — can only authorized people see this data (Confidentiality)? can an attacker modify this data undetected (Integrity)? can an attacker stop the system from working (Availability)? Map any threat/vulnerability onto one or more of these three.
- **Threat / Vulnerability / Risk / Mitigation vocabulary**: a precise causal chain for reasoning about security incidents.
  - When to use: structuring any security discussion or review.
  - How: a *vulnerability* (weakness/bug) can be exploited by a *threat* (potential harmful event) by an *attacker*; *risk* = likelihood × severity of that exploitation; *mitigation* (control) = action taken to reduce the risk (e.g., MFA).
- **Least privilege principle**: limit data/system access to only what's essential for a role.
  - When to use: designing access control policies for any system handling sensitive data.
  - How: grant each person/service only the minimum access needed to do their job — fewer people with access to any given dataset means fewer opportunities for leaks.
- **Static analysis / code scanning**: check code (and its dependencies) against known-vulnerability rules without executing it.
  - When to use: continuously, ideally automated in pre-commit hooks or CI/CD, so insecure code/dependencies are caught before deployment.
  - How: run a code scanner (Bandit) against your own code and a dependency auditor (pip-audit, or GitHub's Dependabot) against your `requirements.txt`; review flagged issues, decide which are relevant to your context, and fix/upgrade accordingly.

## Key Concepts
- **Security vs. privacy**: security protects against theft/damage/disruption/unwanted access to a system; privacy concerns what data an individual shares and controls over its collection/use. A security breach can also be a privacy breach if personal data is exposed without consent (GDPR, California's CCPA are relevant privacy laws; data privacy itself is out of scope for this book — see *Practical Data Privacy* by Katharine Jarmul).
- **Attacker**: someone seeking to steal data or disrupt a system, often using automated tools.
- **PII (Personally Identifiable Information)**: personal data that can identify an individual — a key category of sensitive data data scientists must protect.
- **OWASP (Open Worldwide Application Security Project)**: publishes an annual "Top Ten" list of web security risks (and a separate one for ML security risks) — a standard reference for common vulnerability categories.
- **Social engineering**: attacks that manipulate people (e.g., phishing emails mimicking real organizations) rather than exploiting code directly, often the actual entry point behind major breaches.
- **Third-party package vulnerabilities**: libraries your code depends on can have known security flaws (tracked in the MITRE vulnerability database) or, worse, be outright malicious packages that install malware; outdated dependencies with known vulnerabilities caused the 2017 Equifax breach (140M+ people's PII, ~$1.4B cost).
- **Pickle module insecurity**: Python's `pickle` can execute arbitrary attacker-planted code when unpickled — unsafe for saving/loading data or ML models from untrusted sources; safer alternatives include JSON (strictly format-validated) or, for ML models specifically, Skops or ONNX (recommended over pickle in scikit-learn's own docs).
- **Version control secret leakage**: committing API keys/credentials to a repo (especially a public one) exposes them — and because Git retains full history, removing a leaked secret after the fact is difficult; store secrets as environment variables or in files excluded from version control (over 100,000 GitHub repos were found in 2019 to contain leaked secrets).
- **API security basics**: an undefended API (like the one built in Ch 11) is fully open to anyone with the URL; real protection requires HTTPS encryption, user authentication, input validation (to prevent SQL injection on database-backed APIs), and awareness of risks like cross-site request forgery/scripting — generally a job for a security expert, not a DIY effort by a data scientist.
- **Security review / threat modeling**: a (formal or informal) process, often run by a company's security team, examining potential threats to a system, their risk level, and needed mitigations before software ships.
- **Bandit**: an open-source static-analysis command-line tool (`bandit -r .`) that scans Python source code for known vulnerability patterns (e.g., flags use of `assert`, which is stripped in optimized bytecode and thus unsafe for enforcing real constraints).
- **pip-audit**: scans a `requirements.txt` file (not your code) against known-vulnerability databases, reporting which installed library versions have CVEs and which fixed versions to upgrade to.
- **Dependabot**: GitHub's automated tool for alerting you when a dependency in your project has a known vulnerability.

## Key ML-Specific Concepts
- **Input manipulation attack**: an attacker perturbs an input (knowing the model's structure) to cause misclassification at inference time — e.g., a crafted sticker causing an image classifier to label a banana as a toaster (Brown et al., 2018).
- **Training data extraction**: an attacker crafts prompts/inputs to get a deployed model to reveal memorized training data — Google DeepMind researchers extracted real names/addresses from ChatGPT simply by asking it to repeat a word forever (Nasr et al., 2023), illustrating how unintuitive ML security failures can be.
- **Model theft**: an attacker queries a deployed model heavily enough, and records enough input/output pairs, to reconstruct (steal) the model itself — an intellectual-property risk.
- **Data poisoning / adversarial training data attack**: an attacker contaminates training data (public datasets like ImageNet, or a company's own data) with perturbations designed to plant a "backdoor" into the resulting model's behavior.
- **Transfer learning attack**: since many models are fine-tuned from open-source base models rather than trained from scratch, a maliciously manipulated base model can compromise every model fine-tuned from it — mitigated by sourcing base models (and ML libraries) only from reputable sources.

## Mental Models
- **Security as a moving target, not a solved state**: because dependent code, libraries, and attacker techniques constantly change, "it's impossible to completely eliminate threats" — security practices are ongoing maintenance, not a one-time checklist.
- **The chain from vulnerability to risk is probabilistic, not binary**: a vulnerability existing doesn't automatically mean harm occurs — risk formalizes *both* how likely exploitation is *and* how bad the consequences would be, which is why not every flagged issue (e.g., Bandit's `assert` warning) demands immediate action in every context.
- **Security breaches often start outside the code**: credential theft via phishing, physical theft, or even unrelated "innocuous-seeming software" (e.g., the LastPass breach traced to media software with a keylogger) shows that code-level security is necessary but not sufficient — human/physical/social vectors matter just as much.
- **"This is a job for a specialist, know when to hand it off"**: the chapter's repeated advice — don't implement your own API security, consult your security team before exposing sensitive data via an API — frames the data scientist's job as recognizing risk and escalating, not necessarily becoming a security engineer.
- **ML introduces attack surfaces code review can't see**: unlike a classic code vulnerability, an ML model's "vulnerability" can be a property of its training data or learned weights (e.g., memorized training data, a poisoned backdoor) — invisible by inspecting source code alone, requiring data validation and model monitoring as parallel defenses.

## Anti-patterns
- **Using `pickle` to save/load models or data from any source you don't fully trust**: an attacker-crafted pickle file can execute arbitrary code on load — use JSON, Skops, or ONNX instead.
- **Committing API keys/credentials or data directly into a Git repository**: secrets become part of permanent (hard-to-scrub) history; store them as environment variables or in a file excluded from version control.
- **Leaving an API with no authentication/encryption when it serves sensitive or proprietary data**: the Chapter 11 example API, as built, has "no security at all" — fine for a local learning exercise, never for real data.
- **Letting third-party dependencies go unpatched**: known-vulnerable library versions (the direct cause of the 2017 Equifax breach) persist if you don't regularly update and audit dependencies.
- **Treating ML models as secure just because the surrounding code is reviewed**: input manipulation, training data extraction, model theft, data poisoning, and transfer-learning attacks exploit the model/data itself, not the application code — these require their own mitigations (data validation, adversarial testing, model monitoring).
- **Implementing API security yourself as a non-expert**: the chapter explicitly recommends getting expert help rather than DIY-ing authentication/encryption for anything beyond trivial, non-sensitive use.

## Code Examples
```bash
# Bandit: static security scan of your own code
$ pip install bandit
$ bandit -r .
```
```text
>> Issue: [B101:assert_used] Use of assert detected.
   The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low Confidence: High
   Location: ./test_trendline_functions.py:24:4
23 assert slope == 0.836
24 assert r_squared == 0.868
```
- **What it demonstrates**: Bandit flags `assert` statements as a vulnerability pattern (because `assert` is stripped when Python is compiled to optimized bytecode) — a flag that's low-severity and context-dependent (irrelevant if your code is never compiled with `-O`), illustrating that scanner output needs human judgment, not blind action.

```bash
# pip-audit: scan your dependency list for known vulnerabilities
$ pip install pip-audit
$ pip-audit --requirement requirements.txt
```
```text
Found 11 known vulnerabilities in 7 packages
Name      Version    ID               Fix Versions
--------  ---------  ---------------  -------------
pillow    10.0.0     PYSEC-2023-175   10.0.1
requests  2.29.0     PYSEC-2023-74    2.31.0
```
- **What it demonstrates**: pip-audit cross-references exact installed versions against a vulnerability database and names the fixed version to upgrade to — unlike Bandit, it inspects dependencies, not your own source code.

## Reference Tables
| Term | Definition |
|---|---|
| Attacker | Someone seeking to steal data or disrupt a system |
| Threat | A potentially harmful event (data exposure, disruption) |
| Vulnerability | A weakness (bug/mistake) that could be exploited |
| Risk | Likelihood × severity of a threat exploiting a vulnerability |
| Mitigation (control) | Action taken to reduce a risk (e.g., MFA) |

| C-I-A Triad element | Meaning |
|---|---|
| Confidentiality | Data seen only by those authorized to see it |
| Integrity | Data cannot be modified by an attacker |
| Availability | System keeps working as expected |

| ML-specific attack | Target | Mechanism |
|---|---|---|
| Input manipulation | Deployed model, inference time | Crafted input causes misclassification |
| Training data extraction | Deployed model | Crafted prompts reveal memorized training data |
| Model theft | Deployed model | Heavy querying reconstructs the model |
| Data poisoning | Training data | Contaminated data creates a backdoor |
| Transfer learning attack | Base model | Malicious pretrained model compromises fine-tuned models |

| Tool | Scans |
|---|---|
| Bandit | Your own Python source code |
| pip-audit | Your `requirements.txt` dependencies |
| Dependabot | GitHub-integrated dependency vulnerability alerts |

## Worked Example
Running Bandit on a test file (`test_trendline_functions.py`) from earlier in the book flags four low-severity, high-confidence issues, all `B101:assert_used`, pointing at lines like `assert r_squared == 0.868` — the exact kind of test assertion written in Chapter 7. Bandit's documentation link explains the actual concern: `assert` statements are stripped out when Python code is compiled to optimized bytecode (`python -O`), so code that relies on `assert` for real enforcement (not just tests) silently loses that protection in optimized builds. Because this particular code is test code, never compiled with `-O`, the flagged "vulnerability" is not actually a risk in this context — demonstrating the chapter's point that scanner output must be triaged with judgment (or, in a company, by the security team) rather than acted on uniformly.

## Key Takeaways
1. Security protects systems from theft/damage/disruption/unwanted access; it's distinct from (but can overlap with) privacy, which concerns individuals' control over their own data.
2. Learn the core vocabulary — attacker, threat, vulnerability, risk, mitigation, and the C-I-A triad — to reason precisely about security issues.
3. Common risks for data scientists: stolen credentials (phishing/social engineering), vulnerable or malicious third-party packages, the insecure `pickle` module, secrets/data committed to version control, and unprotected APIs exposing sensitive data.
4. Use JSON/Skops/ONNX instead of `pickle` for saving data or ML models from/to untrusted contexts; store API keys/secrets outside version control (env vars or excluded files).
5. Don't build your own API authentication/encryption as a non-expert — consult your security team; be aware of risks like SQL injection if your API touches a database.
6. Automate security checks where possible: Bandit (code scanning) and pip-audit (dependency scanning) can run standalone or via pre-commit hooks/CI (GitHub Actions), similar to Dependabot's automated dependency alerts.
7. ML systems face unique attack vectors beyond standard code vulnerabilities — input manipulation, training data extraction, model theft, data poisoning, and transfer-learning attacks — mitigated by data validation, adversarial testing, model monitoring, input validation at inference, and the ability to rapidly redeploy a fixed model.
8. Security is a continuously evolving field; stay current via resources like the NIST AI Knowledge Base, AI Village, LLM Security, and the Adversarial Robustness Toolbox.

## Connects To
- **Ch 7**: Data validation (testing data properties) reused as an ML security practice against poisoned/unexpected training data.
- **Ch 10**: Version control risks (leaking secrets/data) directly extend Ch 10's Git practices.
- **Ch 11**: The undefended trendline API built in Ch 11 is explicitly cited as an example lacking any security.
- **Ch 12**: Pre-commit hooks and GitHub Actions (CI/CD) are the automation mechanisms recommended for running Bandit/pip-audit regularly.
- **External**: OWASP Top Ten (web and ML security risks); MITRE vulnerability database; *Practical Data Privacy* by Katharine Jarmul; Equifax, Okta, LastPass, US Dept. of Labor phishing breach case studies; Brown et al. 2018 (adversarial stickers); Nasr et al. 2023 (ChatGPT training data extraction); NIST AI Knowledge Base, AI Village, LLM Security website, Adversarial Robustness Toolbox.
