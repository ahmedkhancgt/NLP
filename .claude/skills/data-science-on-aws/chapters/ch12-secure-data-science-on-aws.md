# Chapter 12: Secure Data Science on AWS

## Core Idea
Security is "job zero" at AWS: the shared responsibility model requires customers to secure everything "in" the cloud via least-privilege IAM, compute/network isolation (VPCs, VPC Endpoints), encryption at rest and in transit (KMS), governance (Organizations, SCPs, Control Tower), and auditability (CloudTrail, CloudWatch, resource tagging) — applied consistently across the entire data science workflow from notebooks to S3 to model endpoints.

## Frameworks Introduced
- **AWS IAM (users, policies, user roles, service roles, condition keys)**: controls authentication (who) and authorization (what) for every AWS resource.
  - When to use: every AWS resource, always, with least-privilege as the governing principle.
  - How: separate IAM *user* roles per job function (`DataScientistRole`, `MLEngineerRole`) and separate IAM *service* roles per SageMaker task (`SageMakerNotebookExecutionRole`, `SageMakerProcessingRole`, `SageMakerTrainingRole`, `SageMakerModelRole`); use condition keys (`aws:SourceVpc`, `sagemaker:InstanceTypes`, `sagemaker:RootAccess`, etc.) to add guardrails.
- **VPC + VPC Endpoints (PrivateLink)**: network/compute isolation and private connectivity to AWS services without traversing the public internet.
  - When to use: any sensitive data science workload — isolate dev/staging/prod via separate accounts/VPCs; use VPC Endpoints for S3, SageMaker, Redshift, Athena, CloudWatch.
  - How: `AppNetworkAccessType='VpcOnly'` for SageMaker Studio; `SubnetId`/`SecurityGroupIds` for notebook instances; VPC Endpoint Policies to restrict which API actions are allowed through the endpoint.
- **S3 Access Points**: simplified, per-team access paths into a shared bucket, each with its own hostname and IAM policy.
  - When to use: large shared buckets/data lakes with multiple teams needing different prefix-level permissions, replacing complex single-bucket-policy sprawl.
- **AWS KMS (Customer Master Keys)**: encrypts data at rest (S3, EBS, EFS) and manages key-exchange for TLS in transit, including post-quantum options.
  - When to use: always — encrypt all data artifacts (notebooks, features, models, predictions) and enforce via deny policies on unencrypted uploads.
  - How: `volume_kms_key`/`output_kms_key` on a SageMaker estimator; S3 bucket policy denying `s3:PutObject` without `x-amz-server-side-encryption`.
- **AWS Organizations + Service Control Policies (SCPs) + AWS Control Tower**: centralized multi-account governance.
  - When to use: enforcing organization-wide guardrails (e.g., mandatory KMS encryption) that even member-account admins cannot override; setting up isolated data-science/staging/production accounts.
- **AWS Lake Formation**: fine-grained row/column-level access control plus real-time data-access logging/alerting, layered on top of S3-based data lakes.
- **AWS Secrets Manager**: encrypted, rotatable storage for database credentials/API keys — never hard-code secrets in notebooks or scripts.
- **CloudTrail + CloudWatch + resource tagging**: auditability — logging API calls/events, tracking user activity, and enforcing tag-based governance via condition keys.

## Key Concepts
- **Shared responsibility model**: AWS secures "of" the cloud (physical infra, host OS, virtualization); the customer secures "in" the cloud (IAM, data classification, encryption, network config) — repeated from Ch. 1, now operationalized.
- **Preventive vs. detective vs. responsive/corrective controls**: preventive controls stop events (deny policies); detective controls quickly surface events (AWS Config, CloudTrail); responsive/corrective controls remediate violations after detection.
- **IAM condition keys**: SageMaker-specific (`sagemaker:` prefix) and global (`aws:` prefix) keys that let a policy allow/deny based on context — e.g., `sagemaker:NetworkIsolation`, `sagemaker:VolumeKmsKey`, `sagemaker:InstanceTypes`, `aws:SourceVpc`.
- **CMK types**: customer-managed (full control, optional annual rotation), AWS-managed, and AWS-owned — differ in who can view/manage metadata and rotation cadence.
- **SSE-S3 vs. SSE-C vs. SSE-KMS**: three server-side S3 encryption options differing in who manages the data key vs. the master key.
- **"Right to be forgotten" data deletion**: regulatory requirement (e.g., GDPR) implemented via copy-modify-rewrite patterns (e.g., Redshift Spectrum: copy external table → delete records → rewrite to S3, overwriting the key).
- **S3 Object Lock (WORM)**: write-once-read-many enforcement on versioned buckets, required for financial/healthcare compliance.

## Mental Models
- **Defense in depth across the full stack**: apply least privilege and encryption at every layer simultaneously — IAM policies, VPC network isolation, S3 bucket/access-point policies, and KMS encryption are complementary, not substitutes for one another.
- **Separate roles by function, not by convenience**: one IAM role per job function (data scientist, ML engineer, DevOps) and one service role per SageMaker task (notebook, processing, training, model hosting) keeps blast radius small if any single role is compromised.
- **SCPs as an organization-wide ceiling, IAM as the floor inside it**: the effective permission for any principal is the *intersection* of what the SCP allows and what the IAM policy grants — SCPs cap what even account admins can authorize.
- **Multi-account separation mirrors the ML pipeline's staging/production split (Ch. 10)**: separate AWS accounts for data science, staging, and production let SageMaker Projects (Ch. 10) enforce real organizational boundaries, not just logical ones.

## Anti-patterns
- **Hard-coding credentials in notebooks or scripts**: use AWS Secrets Manager instead — hard-coded secrets are a direct path to data breaches.
- **Using one IAM role for every job function/service**: defeats least privilege and makes auditing/revocation imprecise; always separate by function and task.
- **Leaving SageMaker Studio/notebooks with direct internet access and root access enabled**: both should be disabled by default (`DirectInternetAccess='Disabled'`, `RootAccess='Disabled'`) unless there's a specific, reviewed need.
- **Treating encryption as optional or "added later"**: encrypt all data at rest (S3, EBS, EFS, ECR images, scratch volumes) and in transit from the start; enforce it with deny policies rather than relying on convention.
- **Allowing GPU instance types for long-lived real-time endpoints when not needed**: wastes cost; reserve GPUs for short-lived, compute-intensive training/batch jobs and restrict endpoint instance types via IAM policy.
- **Granting third-party AWS Marketplace/Partner integrations access without checking for VPC Endpoint support**: the connection may be secure but not private — verify before connecting.

## Code Examples
```json
{
  "Sid": "DenyUnencryptedObjectUploads",
  "Effect": "Deny",
  "Principal": "*",
  "Action": "s3:PutObject",
  "Resource": "arn:aws:s3:::<bucket_name>/*",
  "Condition": {
    "StringNotEquals": { "S3:x-amz-server-side-encryption": "aws:kms" }
  }
}
```
- **What it demonstrates**: an S3 bucket policy that denies any upload lacking KMS-based server-side encryption, enforcing encryption at rest at the bucket level.

```python
estimator = TensorFlow(entry_point='tf_bert_reviews.py', role=role,
    instance_count=2, instance_type='ml.p4d.24xlarge',
    framework_version='<TENSORFLOW_VERSION>',
    volume_kms_key=key_id, output_kms_key=key_id,
    encrypt_inter_container_traffic=True)
```
- **What it demonstrates**: encrypting a distributed SageMaker Training Job's EBS volumes, S3 output model artifacts, and inter-container network traffic, all via KMS.

```json
{
  "Sid": "EncryptSageMakerTraining",
  "Effect": "Deny",
  "Action": "sagemaker:CreateTrainingJob",
  "Resource": ["*"],
  "Condition": {
    "ArnNotEquals": { "sagemaker:OutputKmsKey": ["arn:aws:kms:<REGION>:<ACCOUNT_ID>:key/<KMS_KEY_ID>"] }
  }
}
```
- **What it demonstrates**: an SCP enforced at the AWS Organizations level that denies `CreateTrainingJob` across every member account unless the specified KMS key is used — overriding even account-admin permissions.

```python
sagemaker.create_domain(DomainName='default', AuthMode='IAM',
    DefaultUserSettings={'ExecutionRole': <ROLE_ARN>, 'SecurityGroups': <SECURITY_GROUP_IDS>},
    SubnetIds='<SUBNET_IDS>', VpcId='<VPC_ID>',
    AppNetworkAccessType='VpcOnly', KmsKeyId='<EFS_KMS_KEY_ID>')
```
- **What it demonstrates**: creating a SageMaker Studio domain locked to a private VPC (no public internet path) with its EFS volume KMS-encrypted.

## Reference Tables
| SageMaker service role | Purpose |
|---|---|
| SageMakerNotebookExecutionRole | Notebook/Studio app access to training/hosting services |
| SageMakerProcessingRole | Processing Job access to S3 input/output |
| SageMakerTrainingRole | Training/Tuning Job permissions |
| SageMakerModelRole | Model-hosting inference container permissions |

| S3 server-side encryption option | Who manages the data key | Who manages the master key |
|---|---|---|
| SSE-S3 | Amazon S3 | Amazon S3 |
| SSE-C | Customer | Customer |
| SSE-KMS | AWS | Customer (via KMS CMK) |

| Governance layer | Purpose |
|---|---|
| AWS Control Tower | Set up new secure multi-account environments with best-practice blueprints |
| AWS Organizations + OUs | Group/manage multiple accounts centrally |
| Service Control Policies (SCPs) | Cap maximum permissions org-wide, even over account admins |
| AWS Config | Detective control: evaluate resource configs against policy |

## Worked Example
A data science organization sets up a 3-account structure under AWS Organizations (`DATA_SCIENCE` OU → data science account; `STAGING` OU → staging account; `PRODUCTION` OU → production account), each with its own VPC. An SCP enforced at the organization root denies `sagemaker:CreateTrainingJob` anywhere unless a specific KMS key is passed as `OutputKmsKey` — a training job without it fails with `AccessDeniedException` even though the IAM role would otherwise permit it; adding `output_kms_key`, `subnets`, and `security_group_ids` to the estimator call succeeds. SageMaker Studio is deployed with `AppNetworkAccessType='VpcOnly'` so all traffic routes through private VPC Endpoints rather than the public internet, and notebook instances are created with `DirectInternetAccess='Disabled'` and `RootAccess='Disabled'`. S3 Access Points (`ap1-ds`, `ap2-bi`) replace a sprawling single bucket policy, giving the data science team read/write on `feature-store/*` and the BI team read-only on `data-warehouse/*` within the same shared bucket. All API calls are logged to CloudTrail, and resource tags (e.g., `Project=development`) are enforced via IAM condition keys for auditability.

## Key Takeaways
1. Apply least privilege everywhere: separate IAM roles per job function and per SageMaker service task, never a single shared role.
2. Isolate compute/network with VPCs and VPC Endpoints — require `VpcOnly` for SageMaker Studio and disable direct internet/root access on notebooks by default.
3. Encrypt everything at rest (S3, EBS, EFS, ECR) and in transit (default TLS, inter-container encryption for distributed training), enforced via deny policies, not convention.
4. Use S3 Access Points instead of sprawling single-bucket policies when multiple teams need different prefix-level permissions to a shared bucket.
5. Use SCPs and AWS Organizations to enforce organization-wide security guardrails that even account administrators cannot bypass.
6. Separate data science, staging, and production into distinct AWS accounts to align technical isolation with the model-promotion pipeline from Ch. 10.
7. Log everything (CloudTrail, CloudWatch) and tag resources for auditability — this is a compliance requirement, not just an operational nicety.
8. Never hard-code credentials — use AWS Secrets Manager with scheduled rotation.

## Connects To
- **Ch 1**: Directly implements the shared-responsibility model and security best practices previewed there.
- **Ch 4**: S3 data lake access patterns (Athena, Redshift Spectrum) now secured with VPC Endpoints, bucket policies, and Access Points.
- **Ch 7/8**: Network isolation and encryption requirements for SageMaker Training/Tuning Jobs introduced there are detailed here.
- **Ch 9**: SageMaker Endpoint instance-type cost controls (CPU vs. GPU) build on the Ch. 9 deployment cost discussion.
- **Ch 10**: Multi-account data-science/staging/production separation directly supports the SageMaker Projects GitOps pipeline built there.
