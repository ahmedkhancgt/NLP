# Chapter 16: Technologies

## Core Idea
The technology layer of a data architecture rests on three interlocking decisions — open source vs. cloud provider software, on-prem vs. cloud hosting, and single-cloud vs. multi-cloud — followed by choosing software frameworks (Hadoop, Databricks, Snowflake); this chapter takes a deliberately vendor-neutral, historical view since specific products change faster than architectural concepts.

## Frameworks Introduced
- **Open Source vs. Cloud Provider Software (OSS vs. CSP) Decision**: the foundational platform choice.
  - When to use: for every architectural component (storage, compute, orchestration, etc.) — most real architectures blend both rather than choosing exclusively.
  - How: weigh OSS's cost savings, flexibility/customization, transparency, rapid community innovation, and vendor independence against its limited support, steeper learning curve, fragmentation/compatibility risk, security/liability burden, uncertain long-term maintenance, and IP/licensing complexity.
- **On-Prem vs. Cloud Decision**: whether to keep (or move) infrastructure in your own datacenter.
  - When to use: evaluate per-database/per-workload, not as one blanket company-wide decision.
  - How: favor on-prem only for specific exceptions — no/poor internet connectivity (submarine, mine, oil rig), millisecond-performance-critical workloads, third-party contract constraints, locked-in datacenter leases or recent large hardware purchases, very large locally-born data with insufficient pipeline bandwidth to the cloud, near-term application retirement, or extremely sensitive data (though top-secret clouds now cover most such cases). Otherwise, default to cloud given on-prem's limited scalability, high up-front capital cost, personnel cost, and software licensing cost.
- **Cloud Service Models (IaaS / PaaS / SaaS)**: increasing levels of abstraction/ease vs. decreasing control/customization.
  - When to use: choosing how much infrastructure management to hand off to the CSP.
  - How: **IaaS** — CSP manages hardware/virtualized infra (VMs); customer manages OS/apps/data (most control, most responsibility). **PaaS** — CSP manages runtime/middleware/OS, no VM exposure; customer focuses purely on applications/data (the author's preferred model for data warehouses — no VM management, automatic patching, built-in backups, simple DR). **SaaS** — CSP manages everything; customer just uses the software over the internet (least control, least effort, e.g. Salesforce).
- **Multi-Cloud Decision**: whether to use more than one CSP.
  - When to use: primarily for data sovereignty/regional compliance needs where one CSP lacks a required region; rarely justified for leverage/negotiation (largely a myth) or redundancy (CSPs already engineer cross-region failover within one provider).
  - How: weigh benefits (leveraging each CSP's regional/feature strengths, satisfying data residency requirements) against drawbacks (slow/clunky cross-cloud data movement, interoperability gaps, egress fees, duplicated personnel/security/billing/monitoring overhead, forced use of lower-value IaaS to maintain portability, and the fact that it does NOT meaningfully increase negotiating leverage due to consumption-based volume discounts).

## Key Concepts
- **TCO (Total Cost of Ownership) calculator**: tools offered by all major CSPs (Azure, AWS, GCP) to estimate cloud migration savings from an on-prem environment description.
- **Egress fees**: charges CSPs apply when moving data out of their cloud to a competitor's — a key multi-cloud cost friction.
- **Enterprise license agreements (ELAs) / commitment-based discounts**: CSP pricing that rewards consumption commitment to a single provider (various names: reserved instance pricing, pre-commit discounts, volume-based discounts) — the main reason multi-cloud rarely improves pricing leverage despite the common belief that it does.
- **HDFS (Hadoop Distributed File System)**: Hadoop's storage component, splitting data into blocks distributed across cluster nodes for fault tolerance and high-throughput access.
- **MapReduce**: Hadoop's parallel processing programming model, composed of a Map() procedure (filter/sort) and a Reduce() procedure (summarize), automatically handling parallelization, fault tolerance, and data distribution.
- **Data locality (Hadoop)**: moving computation to where data resides rather than transferring data over the network, reducing overhead.
- **Delta Lake / Project Tungsten / MLlib / GraphX / Spark Streaming**: Databricks' core contributions extending Apache Spark — transactional storage layer, CPU-optimization project, and ML/graph/streaming libraries respectively.
- **Snowflake's storage-compute separation**: Snowflake's founding architectural innovation (2012) allowing independent, automatic scaling of storage and compute.

## Mental Models
- "The cloud is a massive, invisible storage room" — an intuitive framing for cloud elasticity: capacity is "whatever you need, whenever you need it," removing the physical constraints (datacenter space, hardware procurement lead time) that bound on-prem scaling.
- "If you have your own datacenter, you're in the air-conditioning business" — a vivid reminder that on-prem infrastructure ownership diverts organizational effort away from actual data analysis toward facilities/hardware management.
- Multi-cloud as "not putting all your eggs in one basket" vs. the author's counter-framing: it's really about trading concentrated risk for distributed, duplicated operational complexity and cost — a trade-off, not a free lunch.
- The three software frameworks (Hadoop, Databricks, Snowflake) as a rough historical/capability progression: Hadoop (on-prem, batch-oriented, OSS cluster computing) → Databricks (cloud-native Spark commercialization, adds transactional lakehouse capability) → Snowflake (cloud-native, storage/compute separation, can itself act as a lake).

## Anti-patterns
- **Choosing OSS or CSP dogmatically rather than per-component**: most real architectures blend both; treating it as an exclusive either/or choice ignores how the two commonly coexist in practice.
- **Assuming multi-cloud improves negotiating leverage**: the author calls this "mostly a myth" — CSP volume/commitment discounts typically reward consolidating spend with one provider, not threatening to leave.
- **Using multi-cloud as a disaster-recovery strategy**: CSPs already engineer cross-region failover within their own network; the right DR approach is failing over to another region within the same CSP, not switching providers.
- **Underestimating Hadoop's batch-processing latency for real-time needs**: MapReduce's batch nature makes Hadoop unsuitable for real-time/interactive processing — needs are better served by Kafka, Flink, or Spark-based alternatives.
- **Expecting easy portability while using high-value PaaS/SaaS features**: true cloud portability requires sticking to IaaS, which sacrifices the higher-level managed services that make PaaS/SaaS valuable — a real opportunity cost of pursuing multi-cloud flexibility.

## Reference Tables
| Factor | OSS | CSP Software |
|---|---|---|
| Cost | Free to use | Subscription-based, pay-for-use |
| Customization | High (modify source) | Limited |
| Support | Community-based, variable; can purchase vendor support contract | Dedicated support, SLAs |
| Learning curve | Steeper, requires skilled staff | Easier, large talent pool |
| Standardization | Fragmented across projects/versions | Standardized within provider ecosystem |
| Security | Community-reviewed but patch management is your responsibility | CSP invests heavily, handles patching |

| Service Model | CSP Manages | Customer Manages | Control Level |
|---|---|---|---|
| IaaS | Hardware, virtualization | OS, applications, data | High customer control |
| PaaS | Hardware, OS, runtime, middleware | Applications, data (author's preferred model) | Balanced |
| SaaS | Everything | Just usage | Lowest customer control, easiest |

| Framework | Origin | Core Strength | Best Fit |
|---|---|---|---|
| Hadoop | Apache, 2005 | Distributed storage (HDFS) + batch processing (MapReduce) | Large-scale batch analytics, cost-effective commodity-hardware scaling |
| Databricks | 2013, commercializing Spark (UC Berkeley AMPLab, 2009) | Unified Spark platform, Delta Lake, notebooks | Lakehouse architecture, ML/data science collaboration, data mesh domain infrastructure |
| Snowflake | Founded 2012, launched 2014 | Storage/compute separation, can act as data lake + warehouse | Unified warehousing with lake-like flexibility, data mesh via per-domain virtual warehouses |

## Worked Example
TechTreasures Inc., a fictional global ecommerce company based in Florida, adopts a deliberate multi-cloud strategy driven by regional strengths and data sovereignty: Azure for North America (low-latency regional network), GCP for Europe (advanced data analytics, strong EU presence), and AWS for Asia (cloud regions including Taiwan, needed for data residency compliance) — illustrating the one scenario (genuine regional/compliance need) where the book considers multi-cloud clearly justified, as opposed to using it for perceived negotiating leverage or redundancy.

## Key Takeaways
1. Decide OSS vs. CSP per component based on cost, support, customization, and skill availability — not as a single all-or-nothing choice.
2. Default to cloud hosting unless a specific, concrete on-prem constraint applies (connectivity, latency, contracts, lease lock-in, bandwidth, imminent retirement, extreme sensitivity).
3. Prefer PaaS for data warehouse workloads when possible — it eliminates VM management, patching burden, and simplifies disaster recovery compared to IaaS.
4. Pursue multi-cloud only for genuine regional/data-sovereignty requirements; don't expect it to deliver pricing leverage or meaningful redundancy benefits, and budget for its real costs (personnel, security duplication, egress fees, portability trade-offs).
5. Match software framework to need: Hadoop for legacy/on-prem-style batch-heavy distributed processing, Databricks for a Spark-based lakehouse with strong ML/collaboration tooling, Snowflake for a unified cloud-native warehouse that can also act as a lake — all three can support data mesh domains, each in a different way.

## Connects To
- **Ch 5**: Hadoop's HDFS is the historical origin of the data lake concept introduced there.
- **Ch 12**: Delta Lake (Databricks) is the technology underlying the data lakehouse architecture detailed in Ch 12.
- **Ch 13, 14**: Both Databricks and Snowflake are explicitly discussed as viable per-domain infrastructure choices for a data mesh.
- **Ch 2, 10**: On-prem vs. cloud and the broader cloud benefits discussion foreshadowed in Ch 1 and Ch 2 are fully developed here.
- **External concepts**: Apache Hadoop, Apache Spark, Apache Kafka, Apache Flink, Microsoft Azure, AWS, Google Cloud Platform, Databricks, Snowflake, the Free Software Movement (Richard Stallman), Open Source Initiative.
