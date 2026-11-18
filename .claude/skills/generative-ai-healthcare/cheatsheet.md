# Cheatsheet

## Stakeholder-Lens Lookup

| Stakeholder | Primary book chapter | Core incentive | Signature GenAI use cases |
|---|---|---|---|
| Payer (insurer/health plan) | Ch3 | Cost control, risk management, fraud reduction | Synthetic claims data, fraud anomaly detection, personalized plan recs |
| Provider (hospital/clinic/clinician) | Ch4 | Clinical outcomes, workflow efficiency | Imaging enhancement, anomaly detection/segmentation, decision support, EHR documentation |
| Health tech company | Ch5 | Product adoption across payers AND providers | Drug discovery, explainable diagnostics, virtual patients, wellness apps |
| Patient/family | Ch1 (Smith family), Ch8 | Access, affordability, trust | Personalized education, remote monitoring, engagement tools |
| Regulator/policymaker | Ch7 | Safety, fairness, legal compliance | AI ethics committees, HITRUST, audits, regulatory sandboxes |
| Cross-cutting (data) | Ch6 | Data quality, privacy, interoperability | De-identification, synthetic data, federated learning |

## Architecture Selection Decision Rule

| Data shape | Use | Example |
|---|---|---|
| Independent/static features | FNN | Disease risk from demographics |
| Sequential/time-series | RNN (+BPTT) | ECG/vitals interpretation |
| Grid-like/image | CNN | Tumor detection, segmentation |
| Need realistic/novel synthetic output | GAN | Synthetic images, drug candidates |
| Need interpretable latent space / anomaly detection | VAE | Anomaly detection, pattern analysis |
| Small or scarce labeled data | Transfer learning / fine-tuning | Any domain with privacy-limited data |

## Ethical/Regulatory Checklist (from Ch7)

1. **Equity/non-discrimination** — Is training data representative across demographic groups? Has a bias audit run pre- and post-deployment?
2. **Transparency/integrity** — Does the model have a model card? Are decisions explainable (LIME/SHAP, saliency maps, counterfactuals)?
3. **Accountability** — Is there a defined owner (provider vs. developer vs. institution) if the AI errs? Is there an AI ethics committee with real (not just advisory) authority?
4. **Privacy/security** — Is data de-identified (Safe Harbor/k-anonymity) or protected (encryption, differential privacy, federated learning)? Is HIPAA/GDPR/HITRUST compliance documented?
5. **Human oversight** — Is there a Human-in-the-Loop checkpoint before clinical action is taken on an AI recommendation?
6. **Continuous monitoring** — Is there a post-deployment audit cadence (not just a pre-launch check) to catch model drift and emergent bias?

## Tells/Smells of Data-Governance or Ethical Risk

| Smell | Likely problem | Fix |
|---|---|---|
| Model trained on one demographic, deployed broadly | Underperformance/bias for other groups | Diverse/representative retraining; stratified sampling |
| No model card or documentation | Non-auditable, non-compliant | Create documentation; maintain audit trails |
| AI output acted on without clinician review | Accountability gap, overreliance risk | Mandate Human-in-the-Loop |
| Single anonymization technique only | Re-identification risk | Layer de-identification + encryption + differential privacy |
| No external/third-party audit ever done | Unverified compliance/fairness claims | Pursue HITRUST / ISO 27001 certification; engage outside auditors |
| Ethics committee is purely advisory | Governance is cosmetic | Grant stop/redirect authority |
| Synthetic data never validated against real data | Misleading model performance | Periodic validation against real-world benchmarks |
| Payer and provider pursuing AI independently | Stalled adoption from incentive mismatch | Establish shared objectives/metrics across stakeholders |

## HIPAA vs. GDPR Quick Reference

| | HIPAA | GDPR |
|---|---|---|
| Scope | US healthcare sector only | Any org processing EU residents' data |
| Consent | Often implied | Explicit required |
| Breach notice window | 60 days | 72 hours |
| Max penalty | ~$1.5M/category/year | €20M or 4% global turnover |
| Right to be forgotten | No (limited rights) | Yes, plus data portability |

## Decision Rule: When Is Synthetic Data the Right Tool?

- Use synthetic data when: real data is privacy-restricted, the condition/population is rare or underrepresented, or you need to stress-test a model against scenarios not yet observed (e.g., novel fraud patterns, rare disease presentations).
- Do NOT treat synthetic data as a full substitute when: clinical validity of the final model must be proven on real populations before deployment — always validate synthetic-trained models against real-world holdout data.

## Decision Rule: Build vs. Buy vs. Partner (Health Tech Lens)

- If the use case requires deep clinical workflow integration → partner with providers early (co-design, pilot in real settings).
- If the use case requires reimbursement/coverage → engage payers on cost-effectiveness data before scale-up.
- If the use case is diagnostic/clinical-decision-facing → XAI is non-negotiable from day one, not a later add-on.

## ROI Measurement Toolkit (Payer Lens, Ch3)

| Metric | Answers |
|---|---|
| Cost-Benefit Analysis (CBA) | Do total benefits exceed total costs? |
| Net Present Value (NPV) | What's the value of future cash flows today? |
| Internal Rate of Return (IRR) | What's the effective return rate vs. cost of capital? |
| Payback Period | How fast does the investment pay for itself? |

Always pair with baseline measurement (pre-AI metrics) — without it, attributing improvement to AI is unreliable.
