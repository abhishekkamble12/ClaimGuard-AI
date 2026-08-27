# ProofPilot (ClaimGuard-AI) — AI Dispute Risk Manager

[![Razorpay Buildathon 2026](https://img.shields.io/badge/Razorpay_Buildathon_2026-Track_02:_AI_Risk_Manager-0C2340?style=for-the-badge&logo=shield)](https://razorpay.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=for-the-badge)](https://github.com/shap/shap)

> **Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**
> ProofPilot is a **Hybrid Supervised ML + LLM Risk Firewall** that scores merchant chargeback evidence readiness **before** response submission, predicts win probability $P(\text{Win})$, calculates financial Expected Value in Rupees (₹), enforces cryptographic webhook security, and blocks doomed cases to eliminate non-refundable ₹500 dispute penalty fees.

---

## 🎯 The Problem & Our Solution

- **The Problem**: Indian merchants lose crores annually in winnable chargebacks because they submit incomplete evidence. Generic LLM tools make it worse by blindly drafting responses for doomed disputes — triggering network losses and ₹500 penalty fees.
- **The Solution**: ProofPilot is **NOT** a blind letter generator. It is a pre-submission AI Risk Firewall that evaluates evidence readiness, runs counterfactual simulations, predicts calibrated win probability with SHAP explainability, calculates economic ROI, and refuses to draft responses when safety thresholds are breached.

---

## 🏗️ Architecture: Hybrid ML + LLM Reasoning

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
                       │ TF-IDF Semantic Matcher (NLP) │ (Text vector similarity to RC spec)
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Supervised ML Win Predictor   │ (Calibrated P(Win) % & ROC-AUC 88%)
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

### 3. Launch the ProofPilot Dashboard
```bash
streamlit run app.py
```

---

## 🧪 Testing & Evaluation Benchmark

### Run Unit & Edge Case Test Suite
```bash
python -m unittest discover -s eval -p "test_*.py" -v
```

### Run Benchmark Evaluation Harness
```bash
python eval/evaluate.py
```

### Benchmark Metrics Snapshot (Held-Out Test Split)
```text
=== ProofPilot Evaluation Benchmark ===
Dataset                   : outputs/synthetic_chargeback_dataset.json
Split                     : test
Total Cases Evaluated     : 20
Route Accuracy            : 100%
Evidence Precision        : 94%
Evidence Recall           : 88%
Abstention Rate (Human)   : 30%
False-Positive Auto-Drafts: 0 (0%)
FP Cost (₹/case)          : ₹3,850.00
Total FP Financial Loss   : ₹0.00
Predicted Route Dist.     : {'request_more_evidence': 8, 'auto_draft_response': 6, 'human_review': 6}
```

---

## 🐳 Docker Deployment

You can build and deploy ProofPilot with Docker in one command:

```bash
# Build image
docker build -t proofpilot .

# Run container
docker run -p 8501:8501 proofpilot
```
Access the dashboard at `http://localhost:8501`.

---

## 💎 Key Innovations & Rubric Highlights (Track 02)

| Feature | Description | Track 02 Rubric Impact |
| --- | --- | --- |
| **Supervised ML Win Predictor** | Calibrated Logistic Regression model predicting $P(\text{Win})$ trained on dispute features with ROC-AUC ~0.88. | Core ML Rigor |
| **SHAP Explainability** | Integrated `shap.LinearExplainer` waterfall plot visualizing exact feature attributions for risk decisions. | Transparency & Interpretability |
| **Economic Decision Engine** | Computes Expected Value $\text{EV} = P(\text{Win}) \cdot \text{Amount} - (1 - P(\text{Win})) \cdot 500$ in ₹ to recommend `CONTEST` vs `ACCEPT_LOSS`. | Financial ROI & Honest Metrics |
| **Zero False-Positive Loss** | Rigorous gating prevents submitting weak disputes, proving ₹0 false-positive financial loss on benchmark tests. | Honest False-Positive Cost (₹) |
| **Razorpay HMAC-SHA256** | Cryptographic webhook signature generation and verification adhering to Razorpay gateway specs. | Gateway Security Compliance |
| **NPCI UPI Reason Codes** | Full support for Indian payment network dispute rules (`U001`, `U005`, `U008`) alongside card networks. | Indian FinTech Domain Depth |
| **VAMP / VCMP Health Monitor** | Real-time merchant portfolio tracker monitoring the 0.90% chargeback penalty threshold. | Domain Depth & Operations |
| **Decision Audit Trace** | Immutable JSON audit logs exported to `outputs/audit_logs/` for regulatory compliance. | Defensive & Auditable AI |

---

## 🎬 5-Minute Demo Walkthrough Script

1. **[0:00–0:45] The Problem**: Explain that Indian merchants lose crores in winnable chargebacks because they submit incomplete evidence. Generic AI tools blindly draft letters for doomed cases, costing the ₹500 fee.
2. **[0:45–1:30] The Solution**: Introduce ProofPilot as an AI Risk Firewall that scores evidence readiness and calculates monetary EV in ₹.
3. **[1:30–3:00] Live Interaction (Dispute `disp_1000000000009`)**:
   - Show: Initial 55% Readiness → Medium Risk → Auto-drafting is **BLOCKED**.
   - Show: Economic Engine recommends `ACCEPT_LOSS` to save the ₹500 fee.
   - Click: Select missing evidence item (e.g. `Authentication Signal` or `Signed POD`) and click **"Apply Evidence & Re-Score"**.
   - Watch: Score jumps to 80% → Win Probability 88% → EV turns positive → Auto-Draft **UNLOCKS**.
   - Show: Generated formal dispute response packet in Razorpay format.
4. **[3:00–4:00] ML Diagnostics & Explainability**:
   - Review the SHAP feature importance plot showing how completeness score and missing count drive win probability.
   - Highlight the benchmark metrics: 94% precision, 88% recall, 0% FP rate, ₹0 financial loss.
5. **[4:00–5:00] Business Impact**:
   - Demonstrate the VAMP/VCMP Chargeback Ratio Monitor and explain how ProofPilot protects Razorpay merchants from network penalties.

---

## 📂 Repository Structure

```text
ProofPilot (ClaimGuard-AI)
├── app.py                         # Interactive Streamlit Risk Dashboard
├── data_generator.py              # Razorpay & UPI synthetic dispute generator
├── requirements.txt               # Dependencies (scikit-learn, shap, matplotlib, streamlit)
├── Dockerfile                     # Production container definition
├── .env.example                   # Environment configuration template
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── config/
│   └── reason_codes/
│       ├── amex.json              # Amex chargeback reason-code weights & thresholds
│       └── upi.json               # NPCI UPI dispute reason-code config (U001, U005, U008)
├── eval/
│   ├── evaluate.py                # Benchmark evaluation harness with ₹ FP cost
│   ├── test_edge_cases.py         # Adversarial edge case tests
│   └── test_scorer.py             # Scorer, ML, security & economics unit tests
├── extraction/
│   └── extractor.py               # Rule-based NLP + Gemini LLM evidence extractor
├── generation/
│   └── llm_reasoning.py           # Actionable gap explanation & gated response drafter
├── ml/
│   ├── win_predictor.py           # Supervised ML win probability & economic ROI engine
│   ├── semantic_matcher.py        # TF-IDF cosine similarity vector matcher
│   └── explainability.py          # SHAP linear explainer feature attribution
├── outputs/
│   ├── synthetic_chargeback_dataset.json # 60 synthetic dispute cases
│   ├── metrics.json               # Held-out benchmark metrics
│   └── audit_logs/                # Immutable JSON decision traces
├── razorpay_integration/
│   └── webhook_simulator.py       # Razorpay dispute webhook simulator & HMAC validator
└── scoring/
    ├── scorer.py                  # Completeness scorer & counterfactual risk engine
    └── audit_log.py               # Audit logger for decision traces
```

---

## 📜 License
MIT License. Built with pride for the Razorpay AI Buildathon 2026.
