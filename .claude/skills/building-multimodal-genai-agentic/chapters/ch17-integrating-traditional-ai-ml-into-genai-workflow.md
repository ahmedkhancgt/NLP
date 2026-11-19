# Chapter 17: Integrating Traditional AI/ML into GenAI Workflow

## Core Idea
Enterprises with existing traditional ML models (classifiers, regressors, CNNs, forecasters) should wrap them as callable tools (via FastAPI microservices + LangChain tool objects) rather than discarding or retraining — the LLM becomes the reasoning/orchestration/explanation layer while the legacy model keeps doing high-accuracy prediction, preserving past investment while adding natural-language accessibility.

## Frameworks Introduced
- **Hybrid GenAI/ML integration pattern**: Traditional model → wrapped as REST microservice (FastAPI, e.g., `/predict_churn`, `/get_recommendations`, `/classify_product`) → exposed as a LangChain `Tool` object with name/func/description → LLM agent (CoT/ReAct-driven) parses user intent, extracts structured features, invokes the tool, interprets the numeric/categorical output, and generates a natural-language explanation.
  - When to use: whenever an organization already has validated ML models (recommendation, churn, fraud, forecasting, CNN, segmentation, OCR) and wants conversational/agentic access without retraining or rebuilding them as LLMs.
- **LLM's four roles in a hybrid system**: (1) Natural language interface — parse plain-English questions into structured feature inputs; (2) Tool invocation/model wrapping — decide when/how to call the wrapped ML model; (3) Result interpretation/explanation — translate raw scores (e.g., "fraud probability = 0.92") into human-readable narrative, optionally using SHAP for feature-importance explanation; (4) Chained decision-making/workflow triggering — decide next action (flag for review, auto-reject, request more evidence) based on model output, akin to CoT reasoning augmented by model outputs.
- **Model-type-specific integration recipes** (classification, regression, forecasting, ANN, CNN, segmentation, OCR): each has its own API exposure pattern (`/predict_class`, `/predict_price`, `/forecast_sales`, `/predict_churn`, `/classify_image`, `/segment_image`, `/extract_text`), tool-calling format (feature vectors vs. images vs. time-series arrays), and LLM reasoning style (explain probability vs. explain non-linear relationship vs. contextualize visual label).
- **Interactive vs. batch hybrid system initialization**: Interactive — user issues an NL query, LLM parses intent and calls the ML tool on demand. Batch — agent autonomously iterates over a task queue (e.g., nightly invoice folder) and invokes ML tools without user prompting, orchestrated via LangChain-like pipelines monitoring triggers/workflows.

## Key Concepts
- **Stacked ensemble learning for imbalanced classification**: Base learners (XGBoost, LightGBM, Gradient Boosting) output probabilities fed into a meta-classifier (logistic regression) that learns to optimally combine them — improves recall/precision trade-off over any single model on highly imbalanced data (e.g., 6:94 fraud ratio).
- **Threshold tuning**: Rather than defaulting to 0.5, compute precision/recall/F1 across multiple decision thresholds and select the cutoff maximizing F1 — critical for imbalanced classification tasks like fraud detection.
- **Multi-metric evaluation for imbalanced classifiers**: Confusion matrix, PR curve, ROC curve, Matthews Correlation Coefficient (MCC), Cohen's Kappa — together give a comprehensive reliability picture that accuracy alone would mask.
- **Flag-based encoding for domain-meaningful missing values**: Features like `IS_MISSING_MOBILE` carry semantic value when missing (not random) — encode as explicit flags rather than imputing with statistical means, preserving interpretability.
- **Recursive Feature Elimination (RFE)**: Selects the top-N most predictive features (e.g., top 10) before training, reducing dimensionality while preserving model performance.
- **LangChain Tool object**: `Tool(name=..., func=call_xgboost_api, description=...)` — the standard wrapper pattern letting an LLM agent treat a traditional ML model's API call as one of its available actions.
- **`handle_parsing_errors=True`**: LangChain agent initialization flag allowing recovery from ambiguous/malformed LLM output during the ReAct reasoning loop — a robustness necessity in production agent deployments.

## Mental Models
- Think of the traditional ML model as a "specialist employee" and the LLM as a "translator/manager" — the specialist still does the precise technical work (prediction), the manager just makes that work accessible via natural conversation and decides when to call on the specialist.
- Treat this chapter's architecture as the direct generalization of Chapter 15's agentic text-to-SQL design principle ("one agent, many plain tools") — here the "tools" are legacy ML models instead of SQL generation/grading functions.
- The four LLM roles (interface, orchestration, explanation, decision-triggering) map cleanly onto the ReAct Thought/Action/Observation loop from Chapter 12: parsing intent = Thought, tool invocation = Action, interpreting output = Observation-processing, triggering next action = next Thought.
- Multi-metric evaluation (MCC, Kappa, PR-AUC alongside accuracy) is the classification analogue of Chapter 14's "use multiple text-to-SQL metrics in combination" principle — no single number tells the whole story for imbalanced problems.

## Anti-patterns
- **Discarding or expensively fine-tuning LLMs to replicate existing ML model capability**: when a validated XGBoost/CNN/ANN model already solves the prediction task well, wrapping it as a tool is far cheaper and preserves proven accuracy — don't re-solve a solved problem with a more expensive, less precise LLM.
- **Using default 0.5 classification threshold on imbalanced data**: on a 6:94 fraud ratio, the optimal F1-maximizing threshold (e.g., 0.70 in the case study) differs substantially from 0.5 — always tune.
- **Imputing domain-meaningful missing values with statistical means**: for features like `IS_MISSING_MOBILE`, the fact of being missing *is* the signal — mean-imputation destroys that signal; use explicit flag encoding instead.
- **Skipping `handle_parsing_errors=True` in production agents**: LLM output is not always perfectly structured; without graceful error recovery, the agent's ReAct loop can crash on malformed intermediate output.
- **Relying on accuracy alone for imbalanced classifiers**: a model predicting "not fraud" every time would show high accuracy on a 6:94 dataset while being useless — always pair with MCC, Kappa, precision/recall, and PR-AUC.

## Code Examples
```python
Tool(
  name="FraudScoringTool",
  func=call_xgboost_api,
  description="Predicts fraud probability for a telecom claim."
)
```
- **What it demonstrates**: Minimal LangChain tool wrapper exposing a trained XGBoost model's API as an LLM-callable action.

```python
from langchain_core.tools import Tool
from tools.fraud_tool import call_fraud_model
fraud_detection_tool = Tool(
    name="FraudDetectionTool",
    func=call_fraud_model,
    description="Use this to check if a telecom claim is likely fraudulent. Provide structured features like IS_MISSING_MOBILE, HOUR_TO_RAISE_CLAIM, and TOTAL_VERIFICATIONS."
)
```
- **What it demonstrates**: A fully specified LangChain tool with descriptive metadata guiding the LLM on when and how to invoke the wrapped fraud model.

```python
agent = initialize_agent(
    tools=[fraud_detection_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
```
- **What it demonstrates**: ReAct-style LangChain agent initialization with graceful parsing-error recovery, wired to the fraud-detection tool and a local Mistral LLM (via Ollama).

## Reference Tables
| Model type | Use case | Serving strategy | LLM integration |
|---|---|---|---|
| ANNs | Churn prediction, risk scoring (tabular) | API taking feature vectors | LLM sends vector, receives prediction, explains in NL |
| CNNs | Image classification, defect detection | REST API accepting image files | LLM uploads image + metadata, invokes CNN, interprets label |
| Segmentation models | Pixel-wise classification (medical imaging) | GPU-backed serving (TorchServe/TF Serving) | LLM sends image+context, receives mask, explains region |
| OCR | Text extraction from documents/receipts | Tool/API returning raw extracted text | LLM combines OCR output with semantic reasoning (Q&A) |

| Evaluation metric | What it captures |
|---|---|
| Confusion matrix | Raw TP/FP/TN/FN counts |
| PR curve / PR-AUC | Precision-recall trade-off across thresholds |
| ROC curve / ROC-AUC | True/false positive rate trade-off |
| Matthews Correlation Coefficient (MCC) | Balanced correlation metric robust to class imbalance |
| Cohen's Kappa | Agreement beyond chance |

## Worked Example
RastrAI/Company X Analytics case study: telecom fraud detection on a 6:94 imbalanced dataset. Pipeline: `train_xgb_model.py` loads/encodes/scales data, applies RFE (top 10 features), trains a class-weighted XGBoost model, tunes the decision threshold (optimal F1 at 0.70, achieving 89% accuracy, 0.25 precision, 0.44 recall for the fraud class, MCC=0.270, Kappa=0.257), and saves artifacts (`xgb_model_final.pkl`, `scaler.pkl`, `label_encoders.pkl`, `selected_features.pkl`) via joblib. `fraud_model_api.py` serves this via FastAPI's `/predict_fraud` endpoint with pydantic request validation. `fraud_tool.py` wraps the API call and adds natural-language explanation logic; `langchain_fraud_tool.py` exposes it as a `FraudDetectionTool`. `run_agent.py` initializes a ZERO_SHOT_REACT_DESCRIPTION agent with a local Mistral LLM (via Ollama) and `handle_parsing_errors=True`. A user query like "Why is customer retention dropping for Segment B?" or "Can you check if this claim is fraudulent? Mobile number missing, raised at night" triggers the agent to extract structured features (`IS_MISSING_MOBILE=1`, `HOUR_TO_RAISE_CLAIM=2`), call the tool, and return a human-readable fraud assessment with contributing factors.

## Key Takeaways
1. Wrap existing, validated ML models as callable tools instead of discarding them or retraining LLMs to replicate their function — preserves accuracy and investment.
2. The LLM's job in a hybrid system is reasoning/orchestration/explanation, not prediction — let the specialist model predict, let the LLM translate.
3. Threshold tuning and multi-metric evaluation (MCC, Kappa, PR-AUC) are essential for imbalanced classification tasks like fraud detection — accuracy alone is misleading.
4. Flag-based encoding of domain-meaningful missing values preserves interpretability that mean-imputation would destroy.
5. `handle_parsing_errors=True` is a necessary robustness setting for production ReAct agents calling external tools.
6. Different traditional model types (ANN, CNN, segmentation, OCR) each need their own API exposure and tool-calling conventions, but all follow the same wrap-expose-invoke-explain pattern.

## Connects To
- **Ch 5 & 15**: Directly reuses the tool-wrapping and single-agent-multi-tool orchestration patterns from the HITL agentic chapter and the agentic text-to-SQL chapter, now applied to legacy ML models instead of retrieval/SQL tasks.
- **Ch 16**: OCR integration here explicitly references and extends the OCR-as-callable-tool pattern established in the previous chapter.
- **Ch 6**: Explanation-generation (translating model scores into narrative) parallels the grading-mechanism explanation style (justification alongside binary_score).
- **Ch 18**: Explicitly previewed as the next chapter — LLMOps and GenAI evaluation techniques, which will formalize the evaluation rigor (metrics, monitoring) touched on here.
