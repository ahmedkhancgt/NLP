# Chapter 1: Big Data

## Core Idea
Big data is not just about volume — it's all data, inside and outside a company, of any size/speed/type, that could be useful; a company's ability to exploit it depends on its data maturity stage, and the end goal is self-service BI rather than IT-bottlenecked "traditional BI."

## Frameworks Introduced
- **The Six Vs of Big Data**: a framework for describing data beyond mere size.
  - When to use: scoping what "big data" means for a given organization or data source.
  - How: assess each source/dataset against Volume (amount), Variety (structured/semi-structured/unstructured/binary), Velocity (batch vs. real-time), Veracity (accuracy/reliability), Variability (consistency of format/quality/meaning), and Value (business usefulness — "the most important V").
- **Enterprise Data Maturity Stages**: a 4-stage model for assessing an organization's data capability.
  - When to use: diagnosing where a company sits before recommending an architecture.
  - How: Stage 1 Reactive (spreadmarts, no governance) → Stage 2 Informative (centralized, batch, historical "rearview mirror" reporting) → Stage 3 Predictive (cloud-based, higher volume/frequency, real-time ML-driven decisions) → Stage 4 Transformative (handles any data size/speed/type, short onboarding lead time, nontechnical self-service). Stages 1–2 are historical/reactive; stages 3–4 (this book's focus) add predictive and transformative capability.
- **Self-Service BI**: a design goal, not just a feature.
  - When to use: always, as the north star for any data architecture.
  - How: do up-front work with end users to learn their data needs, then build an architecture where end users (who know the data best) can query, model, and report without IT as a bottleneck — contrasted with "traditional BI," where IT fields requests via a queue.

## Key Concepts
- **Spreadmart**: an informal, decentralized collection of spreadsheets/desktop databases used to store and analyze data outside any central governance (Stage 1 symptom).
- **Digital transformation**: embedding technology across a business to fundamentally change how value is derived from data and delivered to customers.
- **Predictive analysis**: using statistical algorithms/ML on historical data to forecast future events, enabling proactive (not just reactive) business decisions.
- **On-prem**: hosting IT infrastructure (servers, storage, networking) within an organization's own facilities, as opposed to cloud providers (Azure, AWS, GCP).
- **Traditional BI**: the IT-request-queue model where end users submit requirements and wait for IT to extract, model, and build reports.
- **Business intelligence (BI)**: the process of collecting, analyzing, and using data to help businesses make more informed decisions.
- **"Data is the new oil"**: analogy — data is a raw material that must be extracted/refined to be useful, is highly valuable, usable in many ways, and transformative/powerful.

## Mental Models
- Data maturity as a journey, not a binary state — "rearview mirror" (reactive to the past) vs. forward-looking (predictive/transformative) stages.
- The self-service BI question as a design test: "How easy will it be for people to build their own reports?" — ask this at every design decision.

## Anti-patterns
- **Treating "big" as only about volume**: ignoring variety, velocity, veracity, variability, and value leads to architectures that handle size but fail on usability or trust.
- **Staying in traditional BI mode**: funneling every reporting need through an IT request queue creates a bottleneck, long lead times (days to months), and frustrated end users.
- **Skipping veracity checks**: ingesting data from unreliable/incomplete IoT or external sources without validation corrupts downstream analysis (e.g., a security camera falsely triggering on weather).

## Reference Tables
| Stage | Name | Characteristics |
|---|---|---|
| 1 | Reactive | Spreadmarts, scattered data, no governance, inconsistent/duplicated effort |
| 2 | Informative | Centralized data, limited scalability, infrequent (nightly) ingestion, "rearview mirror" |
| 3 | Predictive | Cloud-based, larger/varied/more-frequent data, real-time ML-driven decisions |
| 4 | Transformative | Handles any size/speed/type, short onboarding lead time, nontechnical self-service |

## Worked Example
An online bookstore at Stage 3 recommends additional books on the checkout page in real time based on a user's prior purchases — illustrating the shift from nightly batch reporting (Stage 2) to in-the-moment, ML-driven decisioning (Stage 3).

## Key Takeaways
1. Define big data by all six Vs, not just volume, when scoping a solution.
2. Diagnose a company's data maturity stage before proposing an architecture — most companies sit at Stage 2.
3. Design every architecture decision around the question "how easy is this for a nontechnical end user?"
4. Validate data veracity at ingestion; don't assume incoming data (especially IoT/streaming) is accurate.
5. Favor self-service BI over the traditional IT-request-queue model to eliminate bottlenecks.

## Connects To
- **Ch 2**: Data maturity and six Vs motivate why multiple architecture types (RDW, data lake, MDW, fabric, lakehouse, mesh) exist.
- **Ch 16**: On-prem vs. cloud tradeoff introduced here is expanded with cloud provider benefits.
- **External concept**: Gartner's historical "3 Vs" of big data (volume, variety, velocity), which this book extends to six.
