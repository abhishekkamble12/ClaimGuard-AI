# ProofPilot — AI Dispute Risk Manager

## Razorpay AI Buildathon Positioning (Track 02: AI Risk Manager)

**ProofPilot** is a **Hybrid ML + AI Risk Manager for chargeback evidence readiness** designed specifically for Razorpay merchants and payment risk operations.

It evaluates the strength of merchant evidence **before** a chargeback response is submitted. It extracts evidence elements, assesses completeness and TF-IDF semantic relevance, predicts calibrated dispute win probabilities $P(\text{Win})$ via supervised ML (`scikit-learn`), calculates financial expected ROI (₹), provides counterfactual risk improvement insights, and decides whether to auto-draft a response, request additional evidence, or route to human review.

### Why ProofPilot vs. Generic AI Writers

| Standard AI Writer | ProofPilot Hybrid ML + AI Risk Manager |
| --- | --- |
| Blindly drafts response letters for all cases | Quality/Risk gate: evaluates evidence *before* drafting |
| Increases risk of losing chargeback fees on weak cases | Confidence-gated: auto-drafts only high-win cases |
| Ignores evidence gaps | Explains exact evidence gaps & calculates counterfactual gain |
| No risk metrics or financial ROI | Predicts ML Win Probability $P(\text{Win})$, Expected ROI (₹), & ROC-AUC |

---

## Architecture: Hybrid ML + LLM Reasoning

```text
Razorpay Dispute Webhook / API Event
                │
                ▼
Unstructured Merchant Documents & Transaction Metadata
                │
                ▼
      ┌───────────────────────────────┐
      │ ML/NLP TF-IDF Semantic Matcher│ (Computes text vector similarity)
      └───────────────┬───────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│ Supervised ML Win Predictor (scikit-learn)│ (Predicts P(Win) % & Expected Financial ROI ₹)
└─────────────────────┬────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│ Counterfactual Risk Engine               │ (Projects score improvements for missing items)
└─────────────────────┬────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│ Confidence Routing Gate                  │ ──── LOW RISK ────► LLM Response Drafter (Gemini)
└─────────────────────┬────────────────────┘
                      │
                MEDIUM / HIGH RISK
                      │
                      ▼
┌──────────────────────────────────────────┐
│ LLM Gap Explanation Engine               │ ───► Actionable Evidence Recommendations
└──────────────────────────────────────────┘
```

---

## Machine Learning & Evaluation Metrics

ProofPilot reports dual evaluation metrics on held-out test splits:

* **ML Model ROC-AUC Score**
* **Supervised Feature Importance Weights**
* **Financial Expected ROI (₹)**
* **Route Accuracy & Abstention Rate**
* **False-Positive Auto-Draft Rate**

---

## Repository Structure

```text
ClaimGuard-AI / ProofPilot
├── app.py                         # Streamlit Risk Dashboard
├── data_generator.py              # Razorpay synthetic dispute dataset generator
├── requirements.txt               # Dependencies
├── config/
│   └── reason_codes/
│       └── amex.json              # Reason-code evidence weighting config
├── eval/
│   ├── evaluate.py                # Benchmark evaluation harness
│   └── test_scorer.py             # Unit test suite
├── extraction/
│   └── extractor.py               # ML/NLP & Gemini LLM Evidence status extractor
├── generation/
│   └── llm_reasoning.py           # LLM evidence gap explanation & drafter
├── ml/
│   ├── win_predictor.py           # Supervised ML Win Probability model & EV calculator
│   └── semantic_matcher.py        # TF-IDF Cosine Similarity vector matcher
├── razorpay_integration/
│   └── webhook_simulator.py       # Razorpay dispute webhook event simulator
└── scoring/
    └── scorer.py                  # Completeness scorer & counterfactual risk engine
```
