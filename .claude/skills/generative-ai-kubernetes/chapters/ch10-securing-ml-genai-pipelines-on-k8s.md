# Chapter 10: Securing ML/GenAI Pipelines on K8s

## Core Idea
Securing GenAI in production requires understanding GenAI-specific attack vectors (prompt injection, data/model poisoning, denial-of-service, model theft — per NIST's taxonomy and the OWASP Top 10 for LLMs), layering Kubernetes' built-in security primitives (RBAC, Secrets, encryption-at-rest, NetworkPolicies, Pod Security Standards, auditing) on top, and establishing organizational governance (data classification, continuous monitoring, cost management, third-party risk) through a GenAI program office.

## Frameworks Introduced
- **NIST Adversarial ML taxonomy (CIA Triad + Abuse)**: four broad attack categories — Availability, Integrity, Confidentiality (the classic CIA triad), plus Abuse (manipulating the model beyond intended use).
  - When to use: as a structured way to classify and reason about any GenAI threat you encounter.
  - How: for each new threat, ask which of Availability/Integrity/Confidentiality/Abuse it targets, then apply category-appropriate mitigations (below).
- **OWASP Top 10 for LLM Applications**: a community-maintained, example-and-mitigation-rich list including LLM01 Prompt Injection, LLM02 Insecure Output Handling, LLM03 Training Data Poisoning, LLM04 Model Denial of Service, LLM05 Supply-Chain Vulnerabilities, LLM06 Sensitive Information Disclosure, LLM07 Insecure Plugin Design, LLM08 Excessive Agency, LLM09 Overreliance, LLM10 Model Theft.
  - When to use: as an application-design-focused complement to NIST's taxonomy, with concrete scenarios and references.
  - How: review each of the 10 items against your architecture (model inputs, outputs, plugins/tool-calls, supply chain, infra) during design and security review.
- **Kubernetes RBAC fine-grained role pattern**: `Role` (namespace-scoped permissions) + `RoleBinding` (binds a Role to a user/certificate) — vs. `ClusterRole` for cluster-wide scope.
  - When to use: granting least-privilege access to application administrators or service accounts, rather than broad/cluster-admin access.
  - How: create a user certificate via a `CertificateSigningRequest` (CSR), approve it, define a `Role` restricting resources/verbs, bind with a `RoleBinding`, then switch kubectl context to test enforcement.
- **Encryption-at-rest via EncryptionConfiguration**: layered provider priority (e.g. cloud KMS → local KMS → plaintext fallback) applied selectively per resource type (e.g. stronger encryption for Secrets, lighter/none for ConfigMaps/Events).
  - When to use: any cluster storing sensitive configuration/credentials in etcd.
  - How: define an `EncryptionConfiguration` resource with ordered `providers` (kms, identity) per `resources` group; pass as an API server startup argument.
- **GenAI program office governance model**: AI-use policy → inventory of existing AI/ML usage (including shadow AI, tracked via AIBOM-style bills of material) → data classification/lineage → continuous monitoring (NIST TEVV) → cost management (model optimization, prompt optimization/caching, LLM cascading) → third-party risk management.
  - When to use: any organization scaling GenAI beyond isolated pilot projects.
  - How: start with a clear AI-use policy and inventory, then build out data governance, monitoring, and cost/third-party processes incrementally.

## Key Concepts
- **Prompt injection (direct & indirect)**: direct — malicious instructions in the user prompt itself; indirect — malicious content injected via an external data source fed into the context (e.g. a poisoned RAG document).
- **Model/data poisoning**: corrupting training or augmentation data (or a published model, e.g. via a compromised HuggingFace upload) so the model produces attacker-desired wrong output.
- **Model Denial of Service**: forcing computationally expensive/looping inference to exhaust compute, memory, or API token budgets.
- **Model theft**: extracting proprietary model weights/parameters via prompt-based extraction or traditional infrastructure breaches.
- **Pod Security Standards**: Baseline, Privileged, Restricted levels controlling container access to host processes, namespaces, volumes, and ports.
- **Service Account**: a non-human Kubernetes identity scoped to a namespace, used by applications instead of reusing human user credentials to access Secrets/ConfigMaps.
- **x509 certificate authentication**: Kubernetes' default `kubectl` authentication mechanism (vs. username/password), with users tied to certificates and assigned Roles.
- **AIBOM**: an AI-specific extension of Software Bill of Materials (SBOM) tracking models, dependencies, training data, and metadata for governance/audit purposes.

## Mental Models
- Think of **prompt injection as "SQL injection for natural language"** — just as SQL injection exploits unsanitized string concatenation, prompt injection exploits the model's inability to reliably distinguish "instructions" from "data" in its context window.
- Use **"separate control plane from data plane, and separate privilege by role"** as the unifying Kubernetes security principle — this single idea underlies RBAC, namespace isolation (Ch2/Ch6), network policies, and control/data-plane host separation.
- Treat **alignment/filtering as probabilistic defense-in-depth, not a guarantee** — because LLMs are non-deterministic, no single mitigation (fine-tuning, input sanitization, output monitoring) is sufficient alone; layer multiple defenses.
- Think of **governance as "shadow AI" risk reduction** — the program office's first job is discovering what AI usage already exists in the organization before it can be governed; you cannot secure what you don't know about.

## Anti-patterns
- **Granting cluster-admin or broad Secret access ("describe"/"update"/"delete") to application administrators**: the chapter's RBAC example deliberately grants only `list` on Secrets, not full CRUD — minimize blast radius for any compromised credential.
- **Storing models/weights in pickle format**: allows arbitrary code execution on deserialization — use Safetensors (reiterated from Ch3) as a secure alternative.
- **Downloading and deploying community models without supply-chain scrutiny**: the chapter cites a study finding 100+ compromised models on a public model hub — treat model provenance like any other software dependency.
- **Filtering only user input for injection, ignoring indirect/RAG-sourced injection**: if a filter targets only the user-facing prompt, malicious content injected via a poisoned retrieval data source can bypass it entirely.
- **Running production control-plane and data-plane components on the same host without redundancy**: reduces availability and security isolation; separate hosts and consider etcd cluster separation for the highest resilience tier.
- **Treating AI governance as an afterthought during rapid deployment**: the chapter notes companies have repeatedly shipped models without full testing/hardening amid competitive pressure since 2023 — an explicit organizational anti-pattern.

## Code Examples
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: appx-admin-role
rules:
- apiGroups: [""]
  resources: ["pods","deployments","services","configmaps"]
  verbs: ["*"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["list"]
```
- **What it demonstrates**: least-privilege RBAC — full control over application resources (Pods/Deployments/Services/ConfigMaps) but only `list` (not describe/update/delete) on Secrets, forcing actual secret consumption through Service Accounts instead.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: appx-admin-rolebinding
  namespace: default
subjects:
- kind: User
  name: appxadmin
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: appx-admin-role
  apiGroup: rbac.authorization.k8s.io
```
- **What it demonstrates**: binding a certificate-authenticated user to a namespace-scoped Role, completing the RBAC least-privilege pattern.

```yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
- resources:
  - secrets
  providers:
  - kms:
      apiVersion: v2
      name: cloudKMSProviderPlugin
      endpoint: unix:///tmp/kms/socketfile.sock
      timeout: 3s
  - kms:
      name: localVault
      endpoint: unix:///tmp/localkms/socketfile.sock
      cachesize: 100
  - identity: {}
- resources:
  - events
  - configmaps
  providers:
  - identity: {}
- resources:
  - '*.*'
  providers:
  - kms:
      name: localVault
      endpoint: unix:///tmp/localkms/socketfile.sock
      cachesize: 100
  - identity: {}
```
- **What it demonstrates**: tiered encryption-at-rest configuration — strongest (cloud KMS, falling back to local KMS, falling back to plaintext) for Secrets specifically; explicit plaintext for low-sensitivity ConfigMaps/Events; a wildcard default (K8s 1.27+) for everything else.

## Reference Tables
| NIST Category | OWASP Top 10 Mapping | Example |
|---|---|---|
| Availability | LLM04 Model Denial of Service | Forcing expensive/looping inference to exhaust compute/tokens |
| Integrity | LLM01 (indirect) / LLM03 Training Data Poisoning | Poisoned RAG data source; corrupted training data (e.g. Tay chatbot, 2016) |
| Confidentiality | LLM06 Sensitive Information Disclosure | Prompt injection to leak training data, system prompts, or proprietary info |
| Abuse | LLM01 Prompt Injection / Jailbreaking | Circumventing alignment to generate harmful/illegal content |
| (Supply chain) | LLM05 Supply-Chain Vulnerabilities / LLM10 Model Theft | Compromised models on public hubs; weight/parameter extraction |

| Kubernetes Cluster Size (control nodes) | Tolerable Failures | Notes |
|---|---|---|
| 3 | 1 | Needs 2 of 3 for consensus (Raft) |
| 5 | 2 | Needs 3 of 5 for consensus |
| 7 (general max recommended) | 3 | Beyond 7, etcd write scaling becomes challenging |

## Worked Example
**Creating a least-privilege application admin user:** Generate a 2048-bit key and CSR with OpenSSL (`openssl genrsa` + `openssl req -new ... -subj "/CN=appxadmin/O=ProductX"`) → base64-encode and submit as a Kubernetes `CertificateSigningRequest` → `kubectl certificate approve appxadmin` → extract the signed cert → `kubectl config set-credentials appxadmin --client-certificate=... --client-key=...` → create a new context (`appxadmin-context`) bound to this user → attempting `kubectl get pods` fails (no permissions yet) → switch back to the privileged context, apply the `appx-admin-role` Role and `appx-admin-rolebinding` RoleBinding → switch back to `appxadmin-context` → `kubectl get pods` now succeeds in the `default` namespace but fails against `kube-system` (no permission) → `kubectl get secrets` succeeds (list-only) but `kubectl describe secret <name>` fails (no describe permission) — concretely demonstrating RBAC's namespace- and verb-level granularity.

## Key Takeaways
1. Classify GenAI threats using NIST's CIA-triad-plus-Abuse taxonomy and cross-reference the OWASP Top 10 for LLMs for concrete scenarios and mitigations.
2. Apply Kubernetes RBAC with least privilege — grant only the verbs/resources actually needed (e.g. `list` on Secrets, not full CRUD) and use Service Accounts for application-level secret consumption.
3. Encrypt sensitive etcd-stored data (especially Secrets) with a tiered KMS provider strategy; keep low-sensitivity data unencrypted only deliberately, not by default omission.
4. Prompt injection defenses require multiple layers (input formatting/sanitization, alignment/fine-tuning, continuous output monitoring) because no single technique is reliably sufficient against a non-deterministic model.
5. Indirect prompt injection via RAG/external data sources is a distinct risk from direct user-prompt injection — filtering only user input misses this vector entirely.
6. A GenAI program office should start with an AI-use policy and usage inventory (surfacing shadow AI) before building out data governance, continuous monitoring, and cost/third-party-risk processes.
7. Model/data supply-chain risk is real and measurable (100+ compromised models found on a public hub in one study) — treat model provenance with the same scrutiny as any software dependency.

## Connects To
- **Ch2**: directly extends the namespace/RBAC/Secrets/ResourceQuota primitives introduced there into a full production security model.
- **Ch3**: revisits the pickle-vs-Safetensors security distinction raised during model loading.
- **Ch7-9**: data classification/governance (Ch7), RAG indirect-injection risk (Ch8), and continuous monitoring (Ch9) are all woven into this chapter's program-office governance framework as the capstone of the book.
