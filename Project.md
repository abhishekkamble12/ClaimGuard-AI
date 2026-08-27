# ProofPilot — AI Dispute Risk Manager

## Razorpay AI Buildathon Positioning (Track 02: AI Risk Manager)

**ProofPilot** is a **Hybrid ML + AI Risk Manager for chargeback evidence readiness** designed specifically for Razorpay merchants and payment risk operations.

It evaluates the strength of merchant evidence **before** a chargeback response is submitted. It extracts evidence elements, assesses completeness and TF-IDF semantic relevance, predicts calibrated dispute win probabilities $P(\text{Win})$ via supervised ML (`scikit-learn`), calculates financial expected ROI (₹), provides counterfactual risk improvement insights, provides SHAP ML explainability, enforces cryptographic Razorpay webhook security, and decides whether to auto-draft a response, request additional evidence, or route to human review.

### Why ProofPilot vs. Generic AI Writers

| Standard AI Writer | ProofPilot Hybrid ML + AI Risk Manager |
| --- | --- |
| Blindly drafts response letters for all cases | Quality/Risk gate: evaluates evidence *before* drafting |
| Increases risk of losing chargeback fees on weak cases | Confidence-gated: auto-drafts only high-win cases |
| Ignores evidence gaps | Explains exact evidence gaps & calculates counterfactual gain |
| No risk metrics or financial ROI | Predicts ML Win Probability $P(\text{Win})$, Expected ROI (₹), & ROC-AUC |
| Black-box predictions | SHAP Linear Explainer feature attribution visualizations |
| Card-only coverage | Covers both Card (Amex) and NPCI UPI disputes (U001, U005, U008) |
| No portfolio monitoring | Real-time VAMP/VCMP chargeback ratio threshold monitor |

---

## Architecture: Hybrid ML + LLM Reasoning

```text
Razorpay Dispute Webhook (HMAC-SHA256 Signed)
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
│ Supervised ML Win Predictor (scikit-learn)│ (Predicts P(Win) % & SHAP Explainability)
└─────────────────────┬────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│ Economic ROI Engine (₹)                  │ (Recommends CONTEST vs ACCEPT_LOSS)
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

ProofPilot reports rigorous evaluation metrics on held-out test splits:

* **ML Model ROC-AUC Score**: ~0.88
* **SHAP Feature Importance Attribution**
* **Contest vs Accept Monetary Expected ROI (₹)**
* **False-Positive Financial Cost in Rupees (₹0.00 on test split)**
* **Route Accuracy & Evidence Precision/Recall**
* **Abstention Rate (Human Review routing)**

---

## Repository Structure

```text
ClaimGuard-AI / ProofPilot
├── app.py                         # Streamlit Risk Dashboard
├── data_generator.py              # Razorpay & UPI synthetic dispute generator
├── requirements.txt               # Dependencies
├── Dockerfile                     # Production container definition
├── .env.example                   # Environment configuration template
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── config/
│   └── reason_codes/
│       ├── amex.json              # Amex chargeback reason-code weights & thresholds
│       └── upi.json               # NPCI UPI dispute reason-code config
├── eval/
│   ├── evaluate.py                # Benchmark evaluation harness with ₹ FP cost
│   ├── test_edge_cases.py         # Adversarial edge case tests
│   └── test_scorer.py             # Scorer, ML, security & economics unit tests
├── extraction/
│   └── extractor.py               # ML/NLP & Gemini LLM Evidence status extractor
├── generation/
│   └── llm_reasoning.py           # LLM evidence gap explanation & drafter
├── ml/
│   ├── win_predictor.py           # Supervised ML Win Probability & Economic ROI engine
│   ├── semantic_matcher.py        # TF-IDF Cosine Similarity vector matcher
│   └── explainability.py          # SHAP linear explainer feature attribution
├── razorpay_integration/
│   └── webhook_simulator.py       # Razorpay dispute webhook simulator & HMAC validator
└── scoring/
    ├── scorer.py                  # Completeness scorer & counterfactual risk engine
    └── audit_log.py               # Audit logger for decision traces
```
