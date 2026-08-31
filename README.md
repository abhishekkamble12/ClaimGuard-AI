# ProofPilot (ClaimGuard-AI) — AI Dispute Risk Manager

[![Razorpay Buildathon 2026](https://img.shields.io/badge/Razorpay_Buildathon_2026-Track_02:_AI_Risk_Manager-0C2340?style=for-the-badge&logo=shield)](https://razorpay.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=for-the-badge)](https://github.com/shap/shap)

> **Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**
> ProofPilot is an **Enterprise Pre-Submission AI Risk Firewall** that scores merchant chargeback evidence readiness **before** response submission, predicts calibrated dispute win probabilities via a **Calibrated Stacked ML Ensemble** ($P(\text{Win})$, ROC-AUC 93.4%, Brier Score 0.089), calculates financial Expected Value in Rupees (₹), enforces cryptographic Razorpay webhook security, and monitors merchant portfolio risk against VAMP/VCMP network ceilings.

---

## 🎯 The Problem & Our Solution

- **The Problem**: Indian merchants lose crores annually in winnable chargebacks because they submit incomplete evidence. Generic LLM tools make it worse by blindly drafting responses for doomed disputes — triggering network losses and non-refundable ₹500 dispute penalty fees.
- **The Solution**: ProofPilot is **NOT** a blind letter generator. It is a pre-submission AI Risk Firewall that evaluates evidence readiness, runs counterfactual simulations, predicts calibrated win probability with SHAP explainability, calculates economic ROI, and refuses to draft responses when safety thresholds are breached.

---

## 🏗️ Architecture: Production ML & Microservice Pipeline

```text
               Razorpay Webhook Event (dispute.created / dispute.action_required)
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Razorpay HMAC-SHA256 Validator│ (Cryptographic Gateway Security)
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Corpus-Aware Semantic Matcher │ (TF-IDF vector similarity to RC spec)
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 22-Dim Feature Engineering    │ (Entropy, Log Amount, Base Win Prior)
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Calibrated Stacked ML Ensemble│ (GBT 50% + LR 30% + RF 20% | AUC 93.4%)
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Contest vs Accept ROI Engine  │ (Monetary EV ₹ & Fee Savings)
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Counterfactual Risk Engine    │ (Simulates projected evidence gains)
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Confidence Risk Gate          │
                       └───────┬───────────────┬───────┘
                               │               │
                     PASSED (LOW RISK)   BLOCKED (MED / HIGH RISK)
                               │               │
                               ▼               ▼
                ┌─────────────────────┐ ┌─────────────────────────┐
                │ Response Drafter    │ │ Actionable Gap Guidance │
                │ (Gemini 2.5 Flash)  │ │ & Evidence Retrieval    │
                └─────────────────────┘ └─────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- pip package manager

### 2. Installation & Setup
```bash
# Clone the repository
git clone https://github.com/abhishekkamble12/ClaimGuard-AI.git
cd ClaimGuard-AI

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy environment template for Gemini API key
cp .env.example .env
```

### 3. Launch the Platform

#### Interactive Streamlit Risk Dashboard
```bash
python -m streamlit run app.py
# or: streamlit run app.py
```
Access at `http://localhost:8501`.

#### Production FastAPI Microservice
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# or: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation at `http://localhost:8000/docs`.

---

## 🧪 Testing & Evaluation Benchmark

### Run All Unit & Integration Test Suites
```bash
python scratch/test_runner.py
# or run individually:
python -m unittest discover -s evaluation -p "test_*.py" -v
python -m unittest discover -s api/tests -t . -p "test_*.py" -v
```

### Run Benchmark Evaluation Harness
```bash
python evaluation/evaluate.py
```

### Audit Log Maintenance & Pruning CLI
```bash
python -m scoring.audit_log --prune 90
```

---

## 💎 Key Innovations & Senior ML Engineering Highlights

| Feature | Senior AI/ML Architecture Description | Track 02 Rubric Impact |
| --- | --- | --- |
| **Calibrated Stacked ML Ensemble** | Multi-model ensemble combining Gradient Boosted Trees (50%), Logistic Regression (30%), and Random Forest (20%) wrapped in `CalibratedClassifierCV` (ROC-AUC ~93.4%, Brier Score 0.089). | Core ML Rigor |
| **22-Dimensional Feature Engineering** | Incorporates Shannon evidence entropy, log-transformed amounts, critical evidence flags, semantic relevance distributions, and category-level empirical win priors. | Advanced Feature Engineering |
| **4-Phase XAI Explainability** | Local SHAP waterfall plot, Global SHAP summary, Evidence Contribution Decomposition, and 6-step Decision Pathway Trace. | Transparency & Interpretability |
| **Economic Decision Engine (₹ EV)** | Computes Expected Value $\text{EV} = P(\text{Win}) \cdot \text{Amount} - (1 - P(\text{Win})) \cdot 500$ in ₹ to recommend `CONTEST` vs `ACCEPT_LOSS`. | Financial ROI & Honest Metrics |
| **Merchant Portfolio Intelligence** | Aggregates portfolio analytics, monitors VAMP/VCMP chargeback ratio ceilings (0.65% early warning, 0.90% excessive), and forecasts monthly dispute trajectories. | Enterprise Risk Operations |
| **FastAPI Microservice Layer** | Production RESTful API with automated `X-Razorpay-Signature` HMAC-SHA256 authentication middleware and token-bucket rate limiting. | System Design & Integration |
| **Model Observability & Drift Detection** | Computes Population Stability Index (PSI) to detect live feature distribution drift and trigger retraining alerts. | Production ML MLOps |
| **NPCI UPI & Card Reason Codes** | Native support for Indian UPI dispute specifications (`U001`, `U005`, `U008`) alongside Visa, Mastercard, and Amex. | Indian FinTech Domain Depth |

---

## 🎬 5-Minute Demo Walkthrough Script

1. **[0:00–0:45] The Problem**: Explain that Indian merchants lose crores in winnable chargebacks because they submit incomplete evidence. Generic AI tools blindly draft letters for doomed cases, triggering ₹500 penalty fees.
2. **[0:45–1:30] The Solution**: Introduce ProofPilot as an Enterprise AI Risk Firewall with a Calibrated Stacked ML Ensemble and Monetary EV Engine in ₹.
3. **[1:30–3:00] Live Case Interaction (Tab 1)**:
   - Select a dispute case (e.g. `disp_1000000000009`): Initial 55% Readiness → Medium Risk → Auto-drafting is **BLOCKED**.
   - Show: Economic Engine recommends `ACCEPT_LOSS` to save the ₹500 penalty fee.
   - Click: Select missing evidence item (e.g. `Delivery Signature` or `Signed POD`) and click **"Apply Evidence & Re-Score Case"**.
   - Watch: Score jumps to 88% → Win Probability jumps to 91% → EV turns positive → Auto-Draft **UNLOCKS**.
   - Show: Generated formal dispute response packet in authentic Razorpay format.
4. **[3:00–4:00] Portfolio Intelligence & VAMP Monitoring (Tab 2)**:
   - Switch to **Portfolio Intelligence**: Review the 20 Indian Merchant Cohort Profiles, VAMP chargeback ratio health status, and top systemic evidence bottlenecks across India.
5. **[4:00–5:00] ML Diagnostics & API Integration (Tab 3 & FastAPI)**:
   - Review the Model Comparison table (LR vs GBT vs RF vs Ensemble), Brier score calibration metric, and demonstrate the FastAPI `POST /webhook/dispute` endpoint.

---

## 📂 Repository Structure

```text
ProofPilot (ClaimGuard-AI)
├── app.py                         # Elevated 3-Tab Streamlit Risk Dashboard
├── data_generator.py              # Scaled 500-case Card & UPI dispute generator
├── requirements.txt               # Dependencies (FastAPI, scikit-learn, shap, streamlit)
├── Dockerfile                     # Production multi-port container definition
├── .env.example                   # Environment configuration template
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── api/                           # Production FastAPI Microservice
│   ├── main.py                    # Application entrypoint & middleware
│   ├── dependencies.py            # Shared dependency injectors
│   ├── middleware/                # HMAC auth & Token Bucket Rate Limiting
│   ├── routes/                    # Webhook, single/batch dispute scoring & health
│   └── schemas/                   # Pydantic serialization models
├── config/
│   └── reason_codes/
│       ├── amex.json              # Amex chargeback reason-code weights & thresholds
│       ├── card.json              # Visa / Mastercard / RuPay reason codes
│       └── upi.json               # NPCI UPI dispute reason-code config (U001, U005, U008)
├── evaluation/
│   ├── evaluate.py                # Benchmark evaluation harness with ₹ FP cost
│   ├── test_edge_cases.py         # Adversarial edge case tests
│   ├── test_ensemble_and_api.py   # Ensemble, intelligence & API unit tests
│   └── test_scorer.py             # Scorer, ML, security & economics unit tests
├── extraction/
│   └── extractor.py               # Rule-based NLP + Gemini LLM evidence extractor
├── generation/
│   └── llm_reasoning.py           # Actionable gap explanation & gated response drafter
├── intelligence/                  # Merchant & Portfolio Risk Intelligence
│   ├── merchant_risk_profiler.py  # Merchant cohort risk profiling & VAMP tiers
│   ├── portfolio_analytics.py     # Macro financial benefit & systemic gap tracking
│   └── dispute_velocity.py        # Time-series velocity & surge detection
├── ml/
│   ├── feature_engineering.py     # 22+ feature extraction & entropy calculations
│   ├── win_predictor.py           # Calibrated Stacked ML Ensemble (GBT + LR + RF)
│   ├── model_registry.py          # Lightweight ML model versioning & metadata
│   ├── semantic_matcher.py        # Corpus-aware TF-IDF vector matcher
│   └── explainability.py          # SHAP linear explainer feature attribution
├── monitoring/                    # Production Observability & Alerting
│   ├── model_monitor.py           # Population Stability Index (PSI) drift tracker
│   └── alerting.py                # Threshold alert rules for risk ops
├── outputs/
│   ├── synthetic_chargeback_dataset.json # 500 synthetic dispute cases
│   ├── model_comparison.json      # Cross-model validation benchmark
│   ├── model_registry.json        # Persisted model metadata
│   ├── metrics.json               # Held-out benchmark metrics
│   └── audit_logs/                # Immutable JSON decision traces
├── razorpay_integration/
│   └── webhook_simulator.py       # Razorpay dispute webhook simulator & HMAC validator
└── scoring/
    ├── scorer.py                  # Completeness scorer & counterfactual risk engine
    └── audit_log.py               # Audit logger for decision traces
```

---

## 🔒 Security & LLM Data Flow Note

- **Data Privacy**: All evidence scoring, TF-IDF vector calculations, ensemble predictions, decision routing, and audit trail logging are computed **locally**.
- **Gemini API Integration**: LLM features (gap explanations and dispute response drafting) use `GEMINI_API_KEY` when configured. If omitted, ProofPilot seamlessly switches to deterministic local templates without network calls.
- **Webhook HMAC Security**: All simulated Razorpay Webhook payloads are signed and verified with SHA256 HMAC digest validation.

---

## 📜 License
MIT License. Built with pride for the Razorpay AI Buildathon 2026.
