# Chapter 10: Security and Privacy

## Core Idea
Security must be the first thing a data engineer thinks about in every stage of the lifecycle, not an afterthought — it is addressed in order of people, processes, and technology, because people are consistently the weakest link, and genuine security habits beat compliance theater.

## Frameworks Introduced
- **People → Processes → Technology ordering**: security is addressed in that priority order, not technology-first.
  - When to use: structuring any security program or review.
  - How: first harden human behavior and judgment, then institutionalize habitual processes, then apply technical controls (patching, encryption, monitoring, network access) on top.
- **Active security / negative thinking** (inspired by Atul Gawande's op-ed): deliberately imagine disaster/attack scenarios rather than assuming things will go fine.
  - When to use: designing any new pipeline, storage system, or access policy.
  - How: ask "what's the worst that could happen with this data/access, and how would an attacker exploit it?" before deployment; avoid collecting sensitive data unless there's an actual downstream need.
- **Security habit vs. security theater**: genuine, simple, frequently practiced security habits vs. compliance checkbox exercises (hundred-page policies nobody reads, annual reviews immediately forgotten).
  - When to use: evaluating whether an organization's security program is real.
  - How: keep policies short and practiced regularly (e.g., monthly training) rather than exhaustive and ignored.
- **Shared Responsibility Model (revisited)**: cloud vendor secures the physical data center/hardware; the customer secures applications/configuration built on top — most cloud breaches stem from customer-side misconfiguration, not vendor failure.
  - When to use: any cloud deployment.
  - How: never assume the cloud provider's security covers your application-level configuration; audit your own IAM, bucket permissions, and network rules explicitly.
- **Broken-glass process**: sensitive data normally inaccessible is unlocked only via an emergency approval process, with access revoked immediately after.
  - When to use: data needed rarely but which must remain available for critical incidents.
  - How: require explicit approval, time-box access, and revoke immediately once the task is done.

## Key Concepts
- **Principle of least privilege (revisited, applied operationally)**: give people/systems only the IAM roles/data access needed, only for as long as needed, then remove it — applies equally to humans and service accounts.
- **Encryption at rest vs. in transit**: at rest = full-disk encryption on devices, server-side encryption for stored data/backups; in transit = HTTPS/TLS by default — encryption is "not a magic bullet" and does nothing against a human-level credential breach or an openly public bucket.
- **Monitoring surface areas**: access (who/what/when/where, unusual patterns), resources (CPU/disk/memory/I/O anomalies), billing (unexpected spend spikes signaling misuse), and excess/unused permissions (stale access that should be revoked automatically or on review).
- **Network access hygiene**: whitelist specific inbound IPs/ports rather than opening to 0.0.0.0/0; avoid publicly exposed S3 buckets, open SSH, or unauthenticated database endpoints — a recurring real-world failure mode the authors say they've personally witnessed repeatedly.
- **Hardened perimeter vs. zero-trust (revisited)**: cloud environments default closer to zero-trust (every action authenticated); on-prem/air-gapped systems rely on hardened perimeters — both have trade-offs, and even air-gapped systems remain vulnerable to human failure.
- **Data backup as security-adjacent**: regular, tested backups are essential given ransomware risk (some insurers now reduce payouts for ransomware incidents) — backup belongs under disaster-prevention but is closely tied to security practice.
- **Internal/active security research**: every technical contributor, not just dedicated security staff, should actively look for and report vulnerabilities in the systems they know best — domain experts are best positioned to spot domain-specific holes.

## Mental Models
- **"The weakest link in security and privacy is you"**: assume you are personally a target at all times; conduct yourself defensively online and offline.
- **Security as a habit, not a project**: like flossing, it works only if practiced consistently and simply, not as an annual box-checking exercise.
- **Security and privacy are enablers, not blockers**: well-designed access controls increase the amount of valuable analytics/ML that can safely happen, rather than restricting it.
- **No link in the chain can be taken for granted**: even obscure logging libraries, CPU microcode, and memory caches are potential vulnerability surfaces for engineers working at a low level.

## Anti-patterns
- **Granting administrative/carte-blanche access by default**: a frequent, serious violation of least privilege seen "in the cloud" repeatedly by the authors.
- **Compliance-only security theater**: satisfying SOC-2/ISO 27001 on paper while leaving real gaps unaddressed.
- **Embedding credentials in code/version control**: a persistent, high-risk mistake; use secrets managers and never commit secrets.
- **Publicly open S3 buckets, unrestricted SSH (0.0.0.0/0), or open database ports**: concrete, repeatedly observed real-world security failures.
- **Treating encryption as a complete solution**: encrypting a field while leaving broad access control loose doesn't solve the underlying access problem — the real question is often "who can access this," not "is it encrypted."
- **Using FTP or other legacy unencrypted protocols on public networks**: vulnerable to man-in-the-middle attacks even for ostensibly "public" data.
- **Oversharing screens/desktops on video calls**: risks exposing sensitive information unintentionally; share only the specific window/document needed.
- **No regular backup testing**: backups that have never been restored in practice may not actually work when needed.

## Code Examples
None — Chapter 10 is practice-focused; no code snippets in source (it does include a condensed example organizational security policy covering credentials, devices, and software updates, reproduced in the Reference Tables below).

## Reference Tables

| Area | Key practices |
|---|---|
| Credentials | SSO + MFA by default; never share passwords; disable/delete old credentials; never commit secrets to code; use a secrets manager; least privilege always |
| Devices | Device management (remote wipe capability); MFA; sign in with company credentials; treat devices as an extension of yourself; careful screen sharing; "do not disturb" on calls |
| Software updates | Restart browser on update prompts; apply minor OS updates; wait on major OS releases; avoid beta OS versions |
| Technology | Patch/update systems; encrypt at rest and in transit; log/monitor/alert (access, resources, billing, excess permissions); whitelist network access; avoid legacy insecure protocols (FTP) |

| Privacy law (examples) | Era | Domain |
|---|---|---|
| FERPA | 1970s (US) | Education records |
| HIPAA | 1990s (US) | Health information |
| GDPR | mid-2010s (EU) | General personal data |

## Worked Example
A mid-size data team adopts the chapter's layered approach: (1) People — monthly security training reinforces "always be paranoid" norms, and engineers are told to verify any credential request via a second channel before acting; (2) Processes — all IAM roles are granted just-in-time and reviewed quarterly, with a broken-glass process defined for emergency production-data access requiring two-person approval and automatic revocation; (3) Technology — all data is encrypted at rest and in transit by default, S3 buckets are explicitly audited for public access monthly, database inbound access is IP-whitelisted rather than open, and a monitoring dashboard tracks access patterns, resource anomalies, billing spikes, and stale/unused permissions, feeding an incident response plan that is rehearsed periodically rather than left untested.

## Key Takeaways
1. Security should be the first consideration in every lifecycle stage, addressed via people, then processes, then technology, in that order.
2. Practice "active"/negative thinking: proactively imagine attack and leak scenarios rather than assuming safety.
3. Favor genuine, simple, habitual security practices over compliance-driven security theater.
4. Apply least privilege consistently to both people and systems, with time-boxed, revocable access.
5. Remember the cloud's shared responsibility model: most breaches come from customer-side misconfiguration, not the vendor.
6. Encryption, while essential, is not sufficient on its own — access control is usually the real issue.
7. Monitor access, resources, billing, and unused permissions continuously, and rehearse incident response.
8. Every technical contributor should take an active role in spotting and reporting security issues in their own domain of expertise, not leave it solely to dedicated security staff.

## Connects To
- **Ch 2**: Security introduced there as the first undercurrent; this chapter is its full treatment.
- **Ch 3**: Zero-trust and shared-responsibility models reused directly here.
- **Ch 7, 9**: Credential handling and least-privilege access controls applied concretely to ingestion and serving.
- **External concept**: Atul Gawande's "The Power of Negative Thinking," OWASP, Building Secure and Reliable Systems (Adkins et al.), Practical Cloud Security (Dotson).
