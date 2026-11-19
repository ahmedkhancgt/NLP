# Chapter 7: Data Security Design Patterns

## Core Idea
Valuable, flowing data (Ch 5–6) is also a liability — data security patterns address four concerns in order: complying with personal data removal (privacy law), restricting who can access what at fine granularity, protecting data itself (encryption/anonymization) even if access controls fail, and securing connections to data stores without leaking credentials.

## Frameworks Introduced
- **Vertical Partitioner**: split rows into mutable and immutable/PII column groups stored separately, joined by a shared key.
  - When to use: need cheap, low-volume personal-data removal and some columns repeat identically across many rows.
  - How: identify split columns and a merge key (e.g., `user_id`); write a job that projects two column sets to two destinations; removal then only touches the small, deduplicated PII store, not every repeated occurrence.
- **In-Place Overwriter**: delete/filter data directly in its existing (often legacy, non-partitioned) storage.
  - When to use: no time/resources to restructure storage, or the store lacks a vertical-split-friendly layout.
  - How: native DELETE + VACUUM/compaction for stores supporting it (Delta Lake, Iceberg); for raw formats (JSON/CSV), rewrite the filtered dataset to a staging area, then atomically promote (rename/move) it over the public location to avoid partial-failure data loss; for Kafka, emit a tombstone (null-value) message per key under a compaction cleanup policy.
- **Fine-Grained Accessor for Tables**: restrict access below table level (columns, rows).
  - When to use: users are authorized on a table but should see only a subset of columns/rows.
  - How: column-level via `GRANT SELECT(col_a, col_b)` (Redshift/PostgreSQL), policy tags (BigQuery Data Catalog), or masking functions (Databricks/Snowflake); row-level via native row-security objects (Databricks ROW FILTER, Redshift Row-Level Security, BigQuery/Snowflake row access policies) or a view with a `WHERE` guard condition (`current_user`-based) when native support is absent.
- **Fine-Grained Accessor for Resources**: apply least-privilege access to cloud resources (buckets, streams) rather than database tables.
  - When to use: enforcing least privilege on non-database cloud services (object stores, streaming services).
  - How: resource-based policies (attach an IAM policy directly to the resource, e.g., GCS bucket IAM policy) or identity-based policies (attach permissions to a role/identity, e.g., AWS IAM role assumed by a job); optionally scope further with tag-based conditions (e.g., S3 tag-key conditions).
- **Encryptor**: encrypt data at rest and in transit.
  - When to use: need to guarantee stolen/intercepted data remains unusable without keys, as a defense layer beneath access control.
  - How: server-side encryption via a cloud KMS/Key Vault (data store transparently requests the decryption key on each read, subject to the client's key-store authorization) or client-side encryption (producer manages keys itself); for data in transit, enforce a minimum TLS version at the client SDK and service configuration.
- **Anonymizer**: irreversibly remove or transform sensitive attributes so individuals can never be re-identified.
  - When to use: sharing data externally/broadly where PII must never be recoverable, even combined with other datasets.
  - How: data removal (drop the column), data perturbation (add noise, e.g., alter IP octets), or synthetic data replacement (ML-generated realistic-but-fake substitute values).
  - **Pseudo-Anonymizer** (companion/escalation pattern): replace sensitive values with a form that retains more business utility but is theoretically reversible or re-identifiable when combined with other data — via data masking (partial character replacement), tokenization (reversible via a secured token vault), hashing (one-way, irreversible per value but structurally revealing), or encryption (reversible with the key). When to use: anonymization destroys too much analytical value but true anonymization isn't legally required. How: apply per-column transformation functions; always warn stakeholders this is weaker than true anonymization — re-identification risk grows when datasets are combined.
- **Secrets Pointer**: reference credentials by name via a secrets manager instead of embedding them in code/config.
  - When to use: need centralized credential management/rotation and reduced leak risk (e.g., avoiding accidental credential sharing/git commits).
  - How: store logins/passwords/API keys in a secrets manager (AWS Secrets Manager, GCP Secret Manager); consumers fetch by name at runtime (optionally with a local cache); protect access to the secrets manager itself with a Fine-Grained Accessor pattern.
- **Secretless Connector**: eliminate static credentials entirely via identity-based cloud authentication.
  - When to use: want to avoid managing rotation and leak risk of any credential at all.
  - How: IAM-based access (cloud IAM validates the calling identity/role per request — e.g., AWS assume-role + STS temporary credentials, GCP Service Account bound to a job) or certificate-based authentication (a certificate authority validates client certificates, e.g., PostgreSQL `sslmode=verify-full`).

## Key Concepts
- **Horizontal vs. vertical partitioning**: horizontal groups related rows (e.g., by date); vertical splits a row's columns across separate storage locations (the basis of Vertical Partitioner).
- **Deletion vector**: a table-file-format technique that records removed rows in a small side file rather than rewriting the main data files immediately; consumers filter them out at read time.
- **Staging/promotion pattern**: write a transformed dataset to a private staging location first, then atomically promote (rename/move) it over the public location — avoids partial-failure corruption in non-transactional writers.
- **Tombstone message**: a Kafka record with a real key and null value, used with log compaction to physically remove a key's prior values.
- **Resource-based vs. identity-based access control**: permissions attached directly to a resource vs. attached to a user/role/identity.
- **Least privilege**: granting only the minimum permissions a component needs, balanced against the maintenance cost of many narrow policies.
- **Anonymization vs. pseudo-anonymization**: anonymized data stays unidentifiable even when combined with other datasets; pseudo-anonymized data can become re-identifiable when combined with other sources (the "false sense of security" risk).

## Mental Models
- Choose Vertical Partitioner when you control the pipeline design and repeated immutable PII dominates row count; fall back to In-Place Overwriter for legacy/uncontrolled storage, accepting its higher I/O cost.
- Treat Fine-Grained Accessor as "push access control as close to the data as the store allows" — table/column/row level beats document-level/application-level checks.
- Encryption answers "what if access control fails?" — it's a second independent layer, not a replacement for Fine-Grained Accessor.
- Anonymizer vs. Pseudo-Anonymizer is a deliberate security/utility trade-off: full anonymization for external/broad sharing where re-identification must be impossible; pseudo-anonymization only when stronger internal controls already limit combination risk.
- Secrets Pointer removes credentials from code; Secretless Connector removes credentials altogether — prefer the latter whenever the cloud platform supports identity-based auth.

## Anti-patterns
- **Running the Vertical Partitioner on only the latest transformation step while keeping raw unsplit data indefinitely**: raw storage still holds full PII and needs its own retention/removal strategy.
- **Overwriting a dataset in place without a staging/promotion step**: a failed or retried job can corrupt or lose data; always stage first, then atomically promote.
- **Relying only on wildcard-based resource access policies (e.g., `visits*`) for convenience**: can silently violate least privilege for future resources matching the prefix.
- **Mixing resource-based and identity-based access models in the same project without a clear default**: increases operational complexity unnecessarily.
- **Treating pseudo-anonymized datasets as safe to combine freely**: re-identification becomes possible once enough pseudo-anonymized attributes are joined (the San Marino/software-engineer example in the chapter).
- **Caching secrets without a refresh/invalidation strategy in long-running streaming jobs**: stale credentials cause silent connection failures; combine with Ch 4 idempotency patterns to stay correct across forced restarts.
- **Logging secret values retrieved via Secrets Pointer**: defeats the whole purpose of centralizing credentials.

## Code Examples
```python
# Vertical Partitioner: splitting PII user context from the event stream
visits_without_user_context = (visits_to_save
  .filter('user_id IS NOT NULL AND context.user.login IS NOT NULL')
  .withColumn('context', F.col('context').dropFields('user'))
  .select(F.col('visit_id').alias('key'), F.to_json(F.struct('*')).alias('value')))
user_context_to_save = (visits_to_save.selectExpr('context.user.*', 'user_id')
  .select(F.col('user_id').alias('key'), F.to_json(F.struct('*')).alias('value')))
```
- **What it demonstrates**: separating per-event data from per-user PII into two topics so personal data removal only touches the small, deduplicated user-context store.

```sql
-- Fine-Grained Accessor for Tables: row-level security policy in PostgreSQL
ALTER TABLE dedp.users ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_row_access ON dedp.users USING (login = current_user);
```
- **What it demonstrates**: a native row-level policy that dynamically restricts which rows each user can see.

```python
# Anonymizer vs. Pseudo-Anonymizer: removal + synthetic replacement vs. masking + generalization
users.drop('birthday').withColumn('email', replace_email(users.email))  # Anonymizer

# Pseudo-Anonymizer: partial masking (reversible structure) + range generalization
rows['ssn'] = rows['ssn'].apply(lambda ssn: f'{ssn[0]}***-{ssn[5]}***-{ssn[10]}***')
```
- **What it demonstrates**: the contrast between irreversible anonymization (drop/replace) and value-preserving but weaker pseudo-anonymization (partial masking).

```python
# Secrets Pointer: credentials fetched by reference, never embedded
secretsmanager_client = boto3.client('secretsmanager')
db_user = secretsmanager_client.get_secret_value(SecretId='user')['SecretString']
db_password = secretsmanager_client.get_secret_value(SecretId='pwd')['SecretString']
```
- **What it demonstrates**: resolving credential names to values at runtime instead of hardcoding them.

## Reference Tables
| Data removal pattern | Cost | Applicability |
|---|---|---|
| Vertical Partitioner | Low (touches deduplicated PII store only) | New/controllable pipelines |
| In-Place Overwriter | High (reads/rewrites full dataset) | Legacy/uncontrollable storage |

| Anonymizer method | Reversible? | Re-identification risk when combined |
|---|---|---|
| Data removal | No | None |
| Data perturbation | No | Very low |
| Synthetic replacement | No | None |
| Data masking (pseudo) | Partially | Moderate–high |
| Tokenization (pseudo) | Yes, via vault | Depends on vault security |
| Hashing (pseudo) | No (per value) | Moderate (structure-revealing) |
| Encryption (pseudo) | Yes, via key | Depends on key security |

## Worked Example
Blog analytics platform: incoming visit events are split by a Spark `foreachBatch` job into a PII-free `visits_without_user_context` Kafka topic and a `user_context` topic keyed by `user_id` (Vertical Partitioner), with the latter deduplicated into a Delta Lake `users` table via MERGE. A GDPR deletion request triggers `DeltaTable.delete('user_id = ...')` against the small `users` table (cheap) plus a Kafka tombstone message for the same key under topic compaction — far cheaper than rewriting every historical visit event. Meanwhile, a legacy, non-partitioned JSON dataset without this split structure requires the In-Place Overwriter: a Spark job filters out the removed user's rows, writes the result to a staging directory, then promotes it over the public location with `aws s3 mv`. Access to the `users` table is restricted with PostgreSQL row-level security (`login = current_user`) and column grants limiting most analysts to non-sensitive columns (Fine-Grained Accessor for Tables), while the underlying S3 buckets are locked down with least-privilege IAM roles scoped to specific resource ARNs (Fine-Grained Accessor for Resources) and server-side KMS encryption (Encryptor). When sharing a dataset with an external analytics vendor, the team fully anonymizes direct identifiers (drop birthday, replace email with Faker-generated addresses) while pseudo-anonymizing country (generalized to region) and SSN (partially masked) for internal analytics where some utility must be preserved. Finally, the geolocation-enrichment API credentials are moved out of code into AWS Secrets Manager (Secrets Pointer), and the Spark-to-PostgreSQL connection is later upgraded to certificate-based `sslmode=verify-full` authentication, removing the password entirely (Secretless Connector).

## Key Takeaways
1. Prefer Vertical Partitioner over In-Place Overwriter whenever you control pipeline design — it makes personal-data removal orders of magnitude cheaper.
2. Push access control as granular as the data store allows (column/row level) before resorting to application-level filtering.
3. Encryption is a second, independent layer beneath access control, not a substitute for it — plan for key-loss recovery (soft-deleted keys) as a real operational risk.
4. Full Anonymizer guarantees re-identification is impossible even combined with other data; Pseudo-Anonymizer techniques only reduce risk and can create a false sense of security when datasets are joined.
5. Move credentials out of code with Secrets Pointer at minimum; prefer Secretless Connector (IAM/certificate-based identity) whenever the platform supports it to eliminate credential management entirely.
6. Always stage-then-promote when overwriting data in place to avoid partial-failure corruption from retried or failed jobs.

## Connects To
- **Ch 4 (Idempotency)**: Secrets Pointer's credential-refresh-on-restart scenario and In-Place Overwriter's staging-then-promotion both need idempotency patterns to stay correct under retries.
- **Ch 6 (Data Flow)**: Isolated Sequencer's access concerns and least-privilege job design connect directly to Fine-Grained Accessor for Resources.
- **Ch 8 (Data Storage)**: horizontal partitioning (contrasted with Vertical Partitioner here) is detailed further; the Proxy pattern (Ch 4) mitigates In-Place Overwriter's "impossible rollback" consequence.
- **Ch 10 (Data Observability)**: Dataset Tracker (lineage) helps consumers understand Vertical Partitioner's split storage locations.
- **AWS KMS/Secrets Manager/IAM, GCP KMS/IAM/Data Catalog, Azure Key Vault, Delta Lake, Apache Iceberg, Apache Kafka, PostgreSQL, Databricks Unity Catalog, Snowflake**: primary technologies implementing these patterns.
