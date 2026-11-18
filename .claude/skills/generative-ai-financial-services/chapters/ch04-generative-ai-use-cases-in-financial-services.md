# Chapter 4: Generative AI Use-cases in Financial Services

## Core Idea
Generative AI augments (not replaces) financial-services workflows across customer experience, investment research, underwriting, and sales/marketing — the pattern is always: automate the manual search/retrieval/summarization of information and leave the expert's final judgment and decision intact.

## Frameworks Introduced
- **Augmented contact-center workflow**: Documents + ticket history → vector store (setup) → agent/customer query → generative AI orchestrator (e.g., LangChain agents or a microservice) parses the question, invokes vector store/customer DB → FM produces a summarized answer for the agent.
  - When to use: reducing average 6-minute-to-1-hour query resolution times in call centers.
- **Augmented investment research workflow**: at each manual research step (reading reports, identifying trends/risks/opportunities), the investment agent uses a generative AI chatbot as an assistant; iterative, not unidirectional — agent asks follow-ups and directs the chatbot to specific sources.
  - When to use: speeding up analysis of annual/quarterly reports, risk factors, income statements/balance sheets without removing human judgment.
- **Augmented underwriting workflow**: generative AI chatbot replaces static rule engines for information search/retrieval and runs "what-if" scenarios to incrementally build a quote; underwriter still makes the final decline/approve decision.
  - When to use: securities, bank (credit), or insurance underwriting — all manual, rule-driven, iterative processes.
- **Hyper-personalized marketing**: a fixed system prompt (brand voice + product details) combined with a per-customer user prompt (demographics/financial circumstances) generates a unique, tailored marketing email per customer instead of one generic templated email with only the name swapped.
  - When to use: sales/marketing campaigns where generic bulk email underperforms due to lack of personalization.

## Key Concepts
- **Financial institution categories**: Insurance (health/general/reinsurance), Capital markets (stocks, bonds, currencies, derivatives), Banking (retail, commercial, central, investment, credit unions, S&Ls, brokerage), Payments (PoS/gateways, card issuers, remittances, wallets/BNPL, credit underwriting).
- **Investment research**: manual process of analyzing a financial instrument's reports/risk factors/financials plus market sentiment to produce an investment opportunity report.
- **Underwriting**: process of minimizing/covering risk for an individual/business against a financial instrument; three types — securities, bank, insurance.
- **Reinsurance**: an entity that insures the insurance company itself against part of its risk exposure.
- **Line of Business (LoB)**: organizational unit within a financial institution generative AI use-cases are typically scoped to.

## Mental Models
- Use "internal process first, customer-facing later" as the default rollout sequencing — organizations build confidence with generative AI internally before exposing it to customers.
- Frame every use-case as "generative AI replaces the manual search/retrieval/summarization sub-step, the human keeps the decision" — this is the recurring pattern across contact centers, investment research, and underwriting.
- Treat domain terminology as a forcing function toward fine-tuning: generic FMs struggle with financial jargon, which is itself a reason some use-cases need fine-tuning rather than just prompting/RAG.

## Anti-patterns
- **Generic, templated bulk marketing content**: sending the same email body to all customers (only the name is personalized) underperforms vs. generative AI's ability to tailor message to each customer's actual financial circumstances.
- **Assuming generative AI replaces the human expert**: the book explicitly warns this is not the goal — it boosts productivity by surfacing information faster; final decisions remain with the human agent/underwriter/analyst.

## Code Examples
```text
# System prompt (constant across customers)
You are a helpful AI powered email generator. You work for a prestigious bank MyBank,
serving customers for past 150 years. You are polite, friendly and helpful to your customers.
You bank has created a new investment product. The details of this product are described below:
An investment account which aims at providing at stable 8% interest contribution of at least
$500 a month. Customers can withdraw money from this investment account 4 times a year,
with up to 10% of the total amount present in the investment account.

# User prompt (varies per customer)
Below is the information of a customer:
John is one of the customers of bank for past 10 years. He is 35 years old, married with two
kids of age of 7 and 10. His wife works as a receptionist. His gross salary before taxes is
$200,000. He has monthly liability of approx. $3000 and he invests $500 is various investment
instruments.
Based on the information provided, create a marketing email which is personalized to john
based on his personal and financial circumstances...
```
- **What it demonstrates**: A fixed system prompt (brand/product facts) + a swappable per-customer user prompt is enough to produce genuinely different, personalized marketing copy per customer — no fine-tuning or RAG required, just prompt engineering with structured customer data.

## Reference Tables
**Table 4.1: Generative AI Use-cases by Financial Institution**

| Banking | Capital markets | Insurance | Payments |
|---|---|---|---|
| Credit risk assessment with explainability/reproducibility | Sentiment analysis for investment portfolios | Policy FAQ | Fraud detection |
| Fraud detection and prevention | Trading recommendations | Claims processing | Payment integration |
| Employee onboarding | Financial planning | Policy comparison and summarization | Personalized payment gateway integrations |
| Generating loan and policy documentation | Portfolio rebalancing | Actuarial assistant | Currency conversion optimizations |

**Cross-industry generic use-cases**: coding companions, IT support/ticket automation, documentation search, API documentation generation, customer onboarding, vendor onboarding/contract analysis, multi-channel customer service automation.

## Worked Example
**Hyper-personalized email generation**: Using the same system prompt (MyBank's new 8%-interest investment product details) but two different customer profiles — (1) John, 35, married with two kids, $200K salary, $3K monthly liabilities, already investing $500/month; (2) Jane, 24, single, $100K salary, $300/month student loan, just started investing $400/month — the model produces two distinctly different emails. John's email emphasizes stability for a family with existing liabilities; Jane's emphasizes starting her financial journey and growing wealth steadily as a young professional. This demonstrates prompt-engineering-only personalization at scale, with no fine-tuning or RAG needed.

## Key Takeaways
1. Map any financial-services process to the "automate search/retrieval/summarization, keep human decision" pattern before reaching for a heavier implementation approach.
2. Contact center, investment research, and underwriting workflows all follow the same architectural shape: vector store + orchestrator + FM summarization layer.
3. Hyper-personalization at scale is achievable with prompt engineering alone (fixed system prompt + variable customer data) — no need for fine-tuning for this use-case.
4. Roll out generative AI internally first (employee-facing) to build trust before customer-facing deployment.
5. Domain-specific jargon is a real adoption barrier and a legitimate driver toward fine-tuning (Ch3, Ch10).

## Connects To
- **Ch3**: The orchestrator/vector-store/agent components used here are exactly the RAG and agentic-workflow patterns defined in Ch3.
- **Ch8**: Deep-dives the customer-experience use-case introduced here into a full implementation.
- **Ch9**: Deep-dives contact-center insights, extending the contact-center workflow here.
- **Ch10**: Fine-tuning use-case addresses the domain-terminology gap noted in this chapter.
- **Ch11**: ESG analysis agentic use-case extends the investment-research pattern introduced here.
