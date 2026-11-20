# Chapter 8: Data Governance and Data Security

## Core Idea
Data governance (what data represents and may be used for) and data security (enforcing authorized access accordingly) are two sides of the same coin; both must be built pragmatically and incrementally — federated roles, a living governance framework, data contracts, and a metadata-driven, trust-boundary-based security model (RBAC/ABAC with PEP/PDP/PIP/PAP) — rather than as heavyweight, big-bang bureaucratic programs.

## Frameworks Introduced
- **Data Governance Framework** (roles + framework artifact + governance body + processes): A federated model with domain-level roles (data owner, application owner, data steward, data creator, data user, business user) and a central governance body/chief data officer defining what "good governance" means.
  - When to use: Building a governance operating model at any enterprise scale.
  - How: Document roles/responsibilities in a living framework (doc, wiki, or maturity model); separate data ownership from application ownership; staff a governance body for oversight, training, and standards (not execution); use a RACI matrix.
- **Data Contract Framework**: A formal, often machine-readable agreement between data provider and consumer covering purpose, scope, usage restrictions, SLAs, and schema compatibility.
  - When to use: Whenever data crosses a trust boundary (domain-to-domain, consumer-to-provider).
  - How: Start manual (forms), progress to automated contract-to-test-validation pipelines (e.g., Great Expectations, dbt test), then to workflow-driven approvals, then to self-service with ML-assisted approvals.
- **ABAC Architecture (PEP/PDP/PIP/PAP)**: Attribute-Based Access Control combining data attributes (classification), user/system context, and action type for fine-grained authorization decisions.
  - When to use: When RBAC/ACL granularity or static role explosion becomes unmanageable, or contextual factors (location, time, device) must influence access.
  - How: PEP intercepts requests → PDP evaluates policy → PIP supplies external attributes (identity, data classification) → PAP stores/administers all policies centrally.
- **Trust Boundary Model**: Domain boundaries act as trust boundaries; security requirements tighten specifically at the edges (interfaces to data products/APIs/events) where data crosses into contexts where risk understanding is lower.
  - When to use: Deciding where to apply strict vs. relaxed security controls.
  - How: Implement classification/purpose scheme, data contracts at every trust-boundary crossing, lineage-backed monitoring, and automated enforcement (encryption, masking, identity-based access) at every edge.

## Key Concepts
- **Data owner vs. application owner**: Deliberately separate roles — data mesh theory's single composite "owner" role breaks down because business-context expertise (data owner) and technical engineering expertise (application owner) are different skillsets, and shared applications may need multiple data owners.
- **Data steward**: Subject-matter expert ensuring data policies/standards are adhered to.
- **RACI matrix**: Responsibility assignment matrix mapping tasks/roles/responsibilities to avoid overlap or gaps.
- **Data Contract Application (DCA)**: Central application registering all data contracts; plays the PAP/PIP role in the security architecture.
- **Data classification / data usage classification**: Two complementary taxonomies — classification labels *what the data is* (e.g., PII, Credit Card Number → Highly Confidential); usage classification labels *what it may be used for* (e.g., HR_ONLY, DATA_SCIENCE).
- **Data observability**: An organization's ability to understand the freshness, integrity, volume, lineage, and quality of its data estate and the trends therein.
- **Compute-based vs. data-centric access model**: Compute-based delegates trust down a chain of admins at the consuming layer; data-centric enforces authorization directly where data lives (e.g., in the data lake).
- **RBAC / ACL / ABAC**: Role-based (role→permissions, risk of "role explosion"), access-control-list (per-object user permissions), and attribute-based (policy combining data+context+action attributes) access control models, often layered together.
- **Identity Provider (IDP)**: External authentication service (Active Directory, OpenLDAP) using SAML/OpenID Connect/SSO protocols to supply identity attributes for RBAC/ABAC decisions.

## Mental Models
- **Governance as enabling, not blocking**: Effective governance "makes tasks easier" (self-service discovery, risk mitigation) rather than gatekeeping everything until it's "brought under control" — ineffective governance is bureaucratic and generates no enthusiasm.
- **Security at trust-boundary edges, not everywhere uniformly**: Within a domain, high trust and context knowledge allow looser internal controls; crossing boundaries requires classification, contracts, and automated enforcement because the receiving party lacks the originating context.
- **Governance as a journey, not a destination**: "Data governance is never finished" — automate progressively (computational governance), starting manual-and-slow, ending fully self-service and ML-assisted.

## Anti-patterns
- **Treating data governance as a one-time control project**: Fails because requirements, regulations, and data keep evolving — governance must be continuously operated and automated, not "finished."
- **Combining data owner and application owner into one role**: Loses the fine-grained ownership needed for shared capabilities/applications and conflates business and technical expertise.
- **Over-classifying or over-listing data/usage classifications**: Long classification lists become unusable — keep them centrally maintained and short.
- **Pure RBAC at enterprise scale**: Leads to "role explosion" (thousands of roles) and can't incorporate contextual factors like location/time/device.
- **Tight coupling to a single third-party security vendor without a DCA abstraction layer**: Many security tools work only in silos; without decoupling via a vendor-agnostic data contract layer, you lose a holistic security view.
- **Big-bang governance tooling rollouts**: Complex tooling that's hard to configure/integrate kills enthusiasm; start with simple forms and manual processes, automate incrementally.

## Reference Tables
| Access Control Model | Strength | Weakness |
|---|---|---|
| ACL | Fine-grained, per-object, individual-user level | Doesn't scale to org-wide policy; no behavior nuance |
| RBAC | Org-wide, administrator-friendly | "Role explosion"; static, no contextual awareness |
| ABAC | Combines data/context/action attributes; most flexible | Requires more infrastructure (PEP/PDP/PIP/PAP) |

| ABAC Component | Role |
|---|---|
| PEP (Policy Enforcement Point) | Intercepts data request, generates authorization request |
| PDP (Policy Decision Point) | Evaluates request against policies, returns approve/reject |
| PIP (Policy Information Point) | Supplies external attributes (identity, classification context) |
| PAP (Policy Administration Point) | Central repository managing/administering all policies |

| Data Contract Maturity Stage | Characteristic |
|---|---|
| 1. Ownership + stable pipelines | Prerequisite before any contract process |
| 2. Manual forms | Simple templates (e.g., Microsoft Forms), limited metadata |
| 3. Web app / DB / queue | Replace manual forms; coarse-grained access (containers/folders) |
| 4. Automated validation | Contracts become unit tests / schema validations in pipelines |
| 5. Workflow-driven approval | Data owners/stewards lead; fine-grained filtering (dynamic masking) |
| 6. Full self-service | Automated enforcement, ML-assisted approval predictions |

## Code Examples
```json
{
  "@AgreementId": "1342",
  "contactPoint": { "ConsumerId": "894", "fn": "Marketing Department" },
  "ApplicationId": "201",
  "DataRequestType": "dcat:FlatFile",
  "scope": ["DEVELOPMENT"],
  "purpose": ["HR_ONLY"],
  "dataElements": ["fd11d32f-4944-4bd9-a563-36d0f891e649", "..."],
  "ApprovalFlag": true
}
```
- **What it demonstrates**: A data contract metadata record using GUIDs (stable even as physical storage locations/paths change) to bind a consumer, scope, purpose, and specific data elements into one auditable agreement.

## Worked Example
A composite security architecture (Figures 8-6/8-7 in the book): the Data Contract Administration Application (DCA) acts as PDP/PAP; the metadata-driven ingestion framework acts as PIP (what's ready for consumption); the data product observability store acts as PIP (health/state, SLA input); the data product "sidecar" acts as PEP (enforcement, receiving decisions from PDP/PAP); the data catalog acts as PIP in security and as scanner/cataloger in the data product architecture. Process flow: onboarding (register in catalog, classify/label, often ML-assisted auto-classification) → data contract agreement (owner + consumer agree on purpose/scope) → attribute collection (user/app/environment context via PIPs) → submission to PDP → policy enforcement (dynamic views, row/column-level security, masking, or pre-processed de-identified duplicate datasets).

## Key Takeaways
1. Separate data ownership from application ownership — they require different expertise and must scale independently for shared capabilities.
2. Build the governance framework as a living artifact (roles, responsibilities, maturity model) owned by a governance body that coaches and sets standards rather than executing everything centrally.
3. Start data contracts and security enforcement simple (manual forms, coarse-grained access) and automate progressively — never attempt full self-service enforcement on day one.
4. Apply security proportional to trust boundaries: relax controls within a domain, tighten them (classification, contracts, lineage, automated enforcement) at every cross-domain interface.
5. Use ABAC (PEP/PDP/PIP/PAP) once RBAC/ACL granularity or contextual nuance becomes insufficient, but don't skip straight to it — most orgs should start with RBAC+ACL.
6. For shared-capability data ownership, choose between physically splitting data products per owner (coarse-grained) or encapsulating ownership metadata for dynamic filtering (fine-grained) — and consider an ownership delegation model to avoid requiring every owner's sign-off on every request.
7. Secure C-suite buy-in early, pair governance initiatives with existing regulatory/compliance programs, and measure/communicate business value to sustain momentum.

## Connects To
- **Ch 1**: Directly addresses the privacy/security/regulatory pressures (GDPR, CCPA) raised there.
- **Ch 2**: Builds on domain/bounded-context boundaries as the basis for trust boundaries.
- **Ch 4**: Data contracts extend data product design principles with formal usage governance.
- **Ch 5/6**: Extends security enforcement specifically to API-based and event-driven architectures (OAuth, token-based auth, policy-based message filtering).
- **Ch 9**: Metadata management (catalog, classification infrastructure) is elaborated next as the technical foundation governance/security depend on.
- **Ch 10**: Master data management is explicitly tied to compliant handling of personal data (identity matching, consent).
