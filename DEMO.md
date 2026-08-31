# ProofPilot Demo & Architecture Pitch Guide

## 1. Start the Platform

### Option A: Launch Interactive Dashboard
```bash
python -m streamlit run app.py
# or: streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Launch Production FastAPI Microservice
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# or: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs at `http://localhost:8000/docs`.

---

## 2. 5-Minute Senior AI/ML Engineer Pitch Script

### Segment 1: The Problem & Architecture (0:00 – 1:00)
- "Indian merchants lose crores in winnable chargebacks because they submit incomplete evidence. Generic LLM tools naively generate dispute response letters for doomed disputes, costing merchants lost revenue and mandatory non-refundable ₹500 dispute penalty fees."
- "ProofPilot is an **AI Risk Firewall** that sits between Razorpay dispute webhooks and dispute response submission. It evaluates evidence readiness, calculates monetary Expected Value in ₹, and gates automation."

### Segment 2: Live Case Risk Analysis & Counterfactual Simulator (1:00 – 2:30)
- Select a case in **Tab 1** (e.g. `disp_1000000000009`).
- **Show Initial State**: Evidence Readiness 55% → Medium Risk → Auto-drafting is **BLOCKED**.
- **Show Economic Engine**: Recommends `ACCEPT_LOSS` because EV is negative (saving the ₹500 fee).
- **Hero Moment**: Under **Counterfactual Risk Simulator**, select `Delivery Signature` or `Signed POD` and click **"Apply Evidence & Re-Score Case"**.
- **Watch Live Elevation**: Readiness jumps to 88% → Calibrated Ensemble $P(\text{Win})$ jumps to 91% → EV turns strongly positive → Auto-Draft **UNLOCKS** and generates the formal response letter.

### Segment 3: Explainable AI & Decision Pathway Trace (2:30 – 3:30)
- Expand **ProofPilot XAI — Explainability Summary Card**: Explain how the 4-Phase XAI decomposes decisions into human-readable plain English.
- Show the **Local SHAP Waterfall**: Walk through the feature attributions pushing $P(\text{Win})$ up or down.
- Show the **Decision Pathway Trace**: Show the 6-step gate verification with explicit thresholds.

### Segment 4: Merchant Portfolio Intelligence & VAMP Ceilings (3:30 – 4:15)
- Switch to **Tab 2 (Portfolio Intelligence)**:
  - Walk through the **Merchant Cohort Profiles** (20 Indian merchants categorized into Healthy, Warning, and VAMP Penalty zones).
  - Highlight the **Top Systemic Evidence Gaps** causing dispute losses across India.
  - Highlight the **Projected Annual Savings** KPI counter.

### Segment 5: Production ML Rigor & Model Registry (4:15 – 5:00)
- Switch to **Tab 3 (ML Diagnostics)**:
  - Show the **Model Comparison Benchmark** table (Stacked Ensemble with 93.4% ROC-AUC vs base GBT, LR, and RF models).
  - Explain why **`CalibratedClassifierCV`** is essential: uncalibrated model scores cause distorted financial Expected Value calculations.
  - Conclude by demonstrating the **FastAPI OpenAPI docs** at `http://localhost:8000/docs`.

---

## 3. Key Defensive & Responsible AI Safeguards

1. **Intelligent Abstention**: ProofPilot refuses to generate formal responses for high/medium-risk disputes, preventing automatic fee losses.
2. **Cryptographic HMAC Security**: All webhook payloads verify SHA256 HMAC signatures with constant-time equality matching.
3. **Immutable Decision Trace**: Every dispute evaluation automatically outputs a timestamped JSON defense log in `outputs/audit_logs/`.
4. **Drift Observability**: Population Stability Index (PSI) tracks incoming evidence distribution shifts against training baselines.