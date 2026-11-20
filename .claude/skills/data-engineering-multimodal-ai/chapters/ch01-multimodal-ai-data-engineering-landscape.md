# Chapter 1: The Multimodal AI Data Engineering Landscape

## Core Idea
Multimodal AI systems fail not because models are weak but because data foundations are fragmented across modalities; the fix is a **semantic layer** that aligns entities, preserves relationships, and enforces ontology across text, images, audio, video, and sensor data so AI agents and humans can reason over the same coherent context.

## Frameworks Introduced
- **The Semantic Layer (expanded definition)**: goes beyond analytics semantic layers (dbt/Cube/AtScale metrics consistency) by adding three capabilities:
  - When to use: any system fusing 2+ modalities that must support cross-modal reasoning, not just lookup.
  - How: (1) Ontological reasoning — define entity types, relationship rules, inference logic; (2) Entity resolution — link the same real-world entity across systems with different IDs; (3) Cross-modal integration — fuse text/image/audio/sensor into coherent events.
- **Data Contract**: a formal guarantee about shape, content, and quality of every record entering the system.
  - When to use: at ingestion, for every modality, before data reaches the feature layer.
  - How: define required fields (entity_id, encounter_id, modality, timestamp, provenance, quality_score), validation rules (e.g., entity_id resolves to known entity, timestamp drift < 24h, quality_score < 0.3 quarantines), and violation handling (reject, log reason, route to review queue — never silent failure).
- **The Multimodal AI Data Lifecycle (nine-stage loop)**: Ingestion → Staging/Organization → Core Processing → Feature Engineering → Model Serving → Continuous Learning → Deployment → Monitoring/Analysis → feedback loop back to Ingestion, with the semantic layer overlaying every stage.
  - When to use: as the master mental model for any production multimodal pipeline.
  - How: treat it as a closed loop, not a linear pipeline — corrections/drift signals from monitoring always feed back into ingestion.
- **Four Cloud-Native Tenets for Multimodal AI**: microservices by modality, elastic scaling, serverless preprocessing, event-driven streams — the operational fabric beneath the semantic layer.
  - When to use: when designing the infrastructure layer before the semantic layer can be built on top.
  - How: split pipelines per modality (GPU for images, CPU for text tokenizers); scale each independently; use serverless for bursty jobs (embedding refresh); use event streams with back-pressure handling to synchronize asynchronous arrivals.

## Key Concepts
- **Entity Resolution**: linking disparate identifiers (e.g., EHR ID, insurer ID, PACS ID) for the same real-world entity, starting with deterministic rules and extending to ML-based probabilistic matching plus human adjudication for edge cases.
- **Ontology Enforcement**: a schema of entities (patient, diagnosis, procedure) and relationships (caused_by, confirmed_by, authorized_by) standardizing interpretation.
- **Cross-Modal Correlation**: aligning disparate data by shared ID, timestamps, and inferred events.
- **Context Preservation**: encoding relationships like "lab result confirms imaging finding."
- **Late-Arriving Data Problem**: notes/labs can appear days after the event they describe, endemic to healthcare-style domains.
- **Event Window**: a time span (e.g., 48 hours) across which asynchronous signals must be stitched into one coherent story.
- **Multimodal Fabric**: the result of wiring cloud-native infra without a semantic layer — scalable and resilient but has no understanding of cross-modal relationships.
- **Agent-to-Tool Interface Pattern (MCP)**: structured handshake where an agent declares needs, the semantic API describes capabilities, and responses return typed/validated.
- **Agent-to-Agent Coordination Pattern (A2A)**: protocol for specialized agents (e.g., triage, imaging, claims) to delegate and reconcile results through a shared semantic layer.

## Mental Models
- Think of the **expanded semantic layer** as the analytics semantic layer's "more ambitious sibling" — it doesn't just define what "revenue" means, it defines what a "patient encounter" means across five systems speaking different data languages.
- Use the **nine-stage loop, not pipeline** framing: if you're drawing a straight line from ingestion to serving with no feedback arrow, you're missing the architecture.
- "**Data contracts are not optional**" — treat contract enforcement as the mechanism that makes the semantic layer possible at all; without it, entity resolution and cross-modal joins "produce garbage."
- Each modality pipeline should **fail independently**: image service down ≠ text/vitals pipelines down; the semantic layer should degrade gracefully on partial data rather than produce nothing.

## Anti-patterns
- **Building modality pipelines in isolation with no shared entity model**: each pipeline may be technically flawless but the system can't reason across them (hospital claims-without-evidence-link example; telecom billing-dispute example).
- **Treating vector/graph/feature stores as the end goal**: they are supporting layers; without ontology + entity resolution + cross-modal joins on top, you only get a "multimodal fabric" with no understanding.
- **Silent data failures**: allowing malformed or low-quality records into the feature layer instead of rejecting/logging/quarantining them at the contract boundary.
- **One-shot processing**: processing data once and forgetting it, instead of closing the feedback loop (corrections → retraining → ontology updates → drift detection).

## Code Examples
```bash
# Verify ingestion setup
ls data/raw/images/*.png | wc -l   # Should show: 50
python -c "import fastapi, duckdb, pandas; print('Ready!')"

# Run ingestion pipeline (enforces data contracts on 3 modalities)
python main.py --ingest
```
- **What it demonstrates**: contract-enforced ingestion of heterogeneous modalities (images, notes, vitals) into a unified manifest, with violation counting.

```bash
# Start the semantic API server (FastAPI + Uvicorn + DuckDB)
python main.py --server
# Query a fused cross-modal event
curl http://localhost:8000/event/enc_00001
```
- **What it demonstrates**: exposing fused, evidence-linked events via a narrow semantic API rather than disconnected raw tables.

```bash
# Submit a human correction, closing the feedback loop
curl -X POST http://localhost:8000/corrections \
  -H "Content-Type: application/json" \
  -d '{"encounter_id": "enc_00001", "diagnosis": "pneumonia", \
       "confidence": 0.9, "corrections": {"reason": "Clinical assessment"}}'
```
- **What it demonstrates**: provenance-tracked correction capture that would drive retraining, ontology updates, and drift detection in production.

## Reference Tables
**Minimal multimodal data contract (Figure 1-2)**
| Section | Content |
|---|---|
| Required fields | entity_id, encounter_id, modality, timestamp, provenance, quality_score |
| Validation rules | entity_id must resolve to known entity; timestamp drift < 24h from ingest; quality_score < 0.3 → quarantine |
| On violation | Reject record, log violation reason, route to review queue (no silent failures) |

**Granite's layered architecture (Figure 1-4), top to bottom**
| Layer | Function |
|---|---|
| Ingestion | Streaming (ICU/notes), object storage (images), batch (EHR/claims) |
| Per-modality processing | NLP, CV, ASR, time-series normalization (microservices, independently scaled) |
| Unified feature layer | Quality gate — every record needs entity_id, timestamp, provenance, embedding ref |
| Semantic integration layer | Hybrid index (vector + graph) + entity resolution engine |
| Serving layer | Fused context exposed to triage/claims/optimization agents |
| Feedback loop | Corrections → ontology updates, retraining, drift monitoring, governance |

## Worked Example
**Granite-Lite pipeline (hands-on playbook)**: Ingest three modalities for a single clinical encounter — ChestMNIST X-ray images (50 PNGs, 28×28 grayscale, filename-encoded 14-condition binary label vector), synthetic clinical notes (template-based, ~30% chance of condition-specific terms), and vital-sign time series (120 points at 1Hz: HR 60–120bpm, SpO2 95–100%, RR 12–20/min). Steps: (1) set up and verify data; (2) run `--ingest` to validate all records against `RecordContract` (entity_id, encounter_id, modality, timestamp, provenance, quality) and build a `curated_manifest` joining all three modality tables in DuckDB; (3) start the FastAPI/Uvicorn semantic API server exposing `/event/{encounter_id}` which returns a fused Semantic Event with evidence (X-ray opacity region, note mentions, abnormal vitals); (4) use the Reviewer UI to submit a correction (e.g., "pneumonia" → "bronchitis"), which is logged with full provenance (who, when, original prediction, evidence) to close the feedback loop. Extension ideas: tighten/loosen contract validation to see rejection behavior; swap in domain-specific embeddings; try fusion strategies (majority voting, weighted confidence, attention-based).

## Key Takeaways
1. Multimodal AI's hardest problem is making modalities "mean something together," not processing each one well in isolation — that is data engineering's core discipline now.
2. Data contracts are the non-negotiable gate that makes entity resolution and the semantic layer possible; without them cross-modal joins produce garbage.
3. The multimodal data lifecycle is a closed nine-stage loop, not a linear pipeline — the feedback arrow from monitoring back to ingestion is what makes the system adaptive.
4. Cloud-native tenets (microservices-by-modality, elasticity, serverless, event-driven streams) are necessary infrastructure but insufficient alone — they produce a "multimodal fabric" with no cross-modal understanding until a semantic layer is added.
5. Entity resolution must combine deterministic rules, ML-based probabilistic matching, and human adjudication for the hardest edge cases.
6. Standardized agent interface patterns (MCP for agent-to-tool, A2A for agent-to-agent) are becoming part of the data engineer's contract surface, not just an AI concern.
7. Each modality pipeline should fail independently so the system degrades gracefully rather than losing all output on a single component failure.

## Connects To
- **Ch 2**: Deepens the data architecture components (vector DBs, event streaming, feature stores, metadata services) introduced here as supporting layers beneath the semantic layer.
- **Ch 11**: Governance, lineage, auditability, and compliance (HIPAA/GDPR/EU AI Act) requirements raised here are expanded into full governance practices.
- **MCP / A2A protocols**: the agent-to-tool and agent-to-agent interface patterns connect this chapter's semantic APIs directly to external agent-orchestration standards.
