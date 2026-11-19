# Chapter 9: Use-case: Contact Center Insights with Generative AI

## Core Idea
Post-call contact-center analysis (summarization, case notes, email drafting, QA/compliance checks, call Q&A) can all be automated with pure prompt engineering over a call transcript — no RAG or fine-tuning needed — by reusing one shared system prompt (the transcript as context) with five task-specific prompts.

## Frameworks Introduced
- **Post-call GenAI analysis pipeline**: Contact center software (audio) → Transcription (speech-to-text, e.g., Amazon Transcribe) → GenAI analysis (prompt-engineering tasks) → Outputs (call summary, email correspondence, sentiment, QA check).
  - When to use: automating the manual, error-prone, time-consuming post-call workflow (summaries, notes, emails, compliance review) that otherwise burdens agents and auditors.
- **Shared system prompt + task-specific prompt pattern**: one system prompt embeds the full transcript as context (`<transcript>$transcript</transcript>`); each of the five requirements below swaps in a different task-specific user prompt against that same system prompt — avoiding redundant context-setting per task.
- **Five requirements implemented via prompt engineering**:
  1. **Call summarization**: dot-point summary from the agent's perspective.
  2. **Case notes**: structured Case/Notes/Actions format for CRM entry.
  3. **Email correspondence**: full prose email to the customer summarizing the call and follow-ups.
  4. **Post-call quality (QA) checks**: binary yes/no compliance questions scored into a percentage.
  5. **Question answering on a specific call**: free-form Q&A against the transcript for escalation investigation.

## Key Concepts
- **Amazon Bedrock Converse API**: a unified API across multiple model providers/models (used here with Llama 3 70B) — simplifies switching models without rewriting integration code.
- **QA scoring formula**: `(number of questions answered [Y]) / (total number of questions)` — a simple, auditable compliance scoring mechanism built entirely from LLM yes/no answers over a checklist.
- **System prompt reuse**: the transcript-context system prompt is defined once and reused across all five task prompts, reducing prompt-engineering duplication and keeping cost/latency predictable.

## Mental Models
- Treat the call transcript as a single reusable "context document" and treat each business need (summary, notes, email, QA, Q&A) as a separate, swappable task prompt against that same context — a clean separation of context vs. task in prompt design.
- Use explicit "Respond immediately without preamble" and fixed output-format instructions to make LLM outputs machine-parseable/consistent enough to feed into downstream systems (CRM fields, QA scorecards).
- For compliance/QA checks, convert subjective "did they do X" judgments into binary Y/N answers against an explicit checklist — this produces an auditable, quantifiable score rather than vague qualitative review.

## Anti-patterns
- **Manually reviewing every call for compliance and summary creation**: time-consuming, error-prone, and does not scale with call volume — exactly the anti-pattern this use-case automates away.
- **Re-deriving transcript context for every task prompt**: wasteful; reuse one system prompt with the transcript, and vary only the task instruction.
- **Letting free-form LLM output go directly into structured CRM/QA systems without a fixed output format**: the chapter's prompts always specify exact output format/structure to keep downstream parsing reliable.

## Code Examples
```python
import boto3
from string import Template

def generate_conversation(bedrock_client, model_id, system_text, input_text):
    message = {"role": "user", "content": [{"text": input_text}]}
    messages = [message]
    system_prompts = [{"text": system_text}]
    inference_config = {"temperature": 0.1}
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config
    )
    return response

def ask_llm(system_text, input_text):
    model_id = "meta.llama3-70b-instruct-v1:0"
    aws_region = "us-east-1"
    bedrock = boto3.client(service_name='bedrock-runtime', region_name=aws_region)
    response = generate_conversation(bedrock, model_id, system_text, input_text)
    return response['output']['message']['content']

# Shared system prompt (reused across all 5 requirements)
system_prompt_template = """
  Inside the <transcript></transcript> XML tags is a transcript from a phone call between a
  Banking Customer Advisor (also known as the Agent) and a Customer regarding an online
  website related issue.
  <transcript>$transcript</transcript>
  Use only this transcript for context.
"""
system_prompt = Template(system_prompt_template).safe_substitute(transcript=transcript)

# Requirement 1: Call summarization
summary_prompt = """
  Your task is to write a detailed summary of the call in dot point form, from the perspective
  of the Agent. Include a line break between each dot point.
  Respond immediately without preamble...
"""
output_lines = ask_llm(system_prompt, summary_prompt)

# Requirement 4: Post-call QA compliance check
qa_prompt = """
  Your task is to analyse the transcript and answer each of questions inside the <qa></qa> XML tags.
  <qa>
  Did the Agent identify themself?
  Did the Agent explicitly state they were conducting a 'privacy check'?
  Did the Agent mention that the call is being recorded?
  Confirmed the nature of the customer problem?
  Confirmed the customer preferred method of contact?
  Communicated follow-up actions?
  </qa>
  Answer with [Y] if the question can be answered (yes), or [N] if it cannot (no).
  Based on your answer, calculate a score percentage. The formula for calculating the score
  percentage is: (number of questions answered with [Y]) / (total number of questions)
  Do not output in XML.
"""
output_lines = ask_llm(system_prompt, qa_prompt)
# => Score percentage: (4/6) * 100% = 66.67%

# Requirement 5: Free-form Q&A over the call
insights_prompt_template = "Answer the following question: $question"
question = "what was the customer problem?"
insights_prompt = Template(insights_prompt_template).safe_substitute(question=question)
output_lines = ask_llm(system_prompt, insights_prompt)
```
- **What it demonstrates**: a single reusable system prompt (transcript as context) driving five distinct downstream tasks via the Bedrock Converse API — including a self-scoring QA compliance checklist that converts qualitative review into a quantifiable percentage.

## Reference Tables
**Five contact-center requirements and their prompt pattern**

| Requirement | Output format | Key prompt instruction |
|---|---|---|
| Call summarization | Dot points, agent's perspective | "write a detailed summary... in dot point form" |
| Case notes | Case / Notes / Actions sections | "Summarise nature of injury, admin steps, follow-up actions, 1st person" |
| Email correspondence | Full prose email | "Thank customer, highlight issue+actions, sign off, personable but professional tone" |
| Post-call QA check | Y/N checklist + score % | "Answer [Y]/[N] for each question... calculate score percentage" |
| Call Q&A | Free-form answer | "Answer the following question: $question" |

## Worked Example
**Account lockout call walkthrough**: Transcript of "Lisa" (National Bank agent) helping a customer locked out of online banking with a blocked debit card due to unauthorized Eastern European transactions. Running the same transcript through all 5 task prompts produces: (1) a bullet summary of the troubleshooting steps and resolution; (2) structured case notes noting unauthorized transactions, card blocking, and credential reset actions; (3) a polished customer email confirming the new card/PIN mailing timeline and password reset email; (4) a QA compliance score of 66.67% (4 of 6 checklist items met — notably missing the explicit "privacy check" statement and confirming preferred contact method); (5) a direct answer to "what was the customer problem?" extracted from the transcript. This demonstrates one transcript, one system prompt, five distinct business outputs.

## Key Takeaways
1. Post-call analysis (summarization, notes, email, QA, Q&A) is a prompt-engineering-only use-case — no RAG or fine-tuning is required when the full transcript fits in context.
2. Reuse one system prompt (context) across multiple task-specific prompts to reduce duplication and keep costs predictable.
3. Converting compliance review into an explicit Y/N checklist with a scoring formula makes QA auditable and quantifiable, not just qualitative.
4. Specify exact output formats in the prompt to keep LLM output machine-parseable for CRM/downstream system integration.
5. The same pattern extends naturally to additional outputs (e.g., churn-risk flagging) by adding new task prompts against the same transcript context.

## Connects To
- **Ch3**: Pure prompt-engineering implementation of the approach spectrum (no RAG/fine-tuning needed here).
- **Ch4**: Extends the contact-center use-case introduced there into full post-call automation.
- **Ch8**: Companion chapter — Ch8 handles the live-call RAG assistant, Ch9 handles the post-call analysis.
- **Ch13**: The QA compliance scoring pattern here is a precursor to the formal trustworthiness/evaluation frameworks in Ch13.
