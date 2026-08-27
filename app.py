"""
ProofPilot — AI Dispute Risk Manager Dashboard
Track 02: Razorpay AI Buildathon 2026 (AI Risk Manager)
--------------------------------------------------------
Interactive Risk Operations & Merchant Pre-Submission Evidence Readiness Dashboard.
Integrates ML Win Probability, SHAP Explainability, Economic ROI Decisioning (₹),
Razorpay HMAC-SHA256 Webhook Verification, and VAMP/VCMP Chargeback Ratio Health Monitoring.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import evaluate_dataset
from generation.llm_reasoning import generate_gated_dispute_response, generate_llm_gap_explanation
from ml.explainability import generate_shap_plot
from ml.win_predictor import get_win_predictor
from razorpay_integration.webhook_simulator import DEFAULT_WEBHOOK_SECRET, simulate_razorpay_dispute_event
from scoring.audit_log import get_audit_log
from scoring.scorer import load_reason_code_config, score_dispute

DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CARD_CONFIG_PATH = ROOT / "config" / "reason_codes" / "card.json"
AMEX_CONFIG_PATH = ROOT / "config" / "reason_codes" / "amex.json"
UPI_CONFIG_PATH = ROOT / "config" / "reason_codes" / "upi.json"

st.set_page_config(
    page_title="ProofPilot — AI Dispute Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_dataset() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        from data_generator import generate_dataset
        ds = generate_dataset(60, 0.33, 42)
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATASET_PATH.write_text(json.dumps(ds, indent=2), encoding="utf-8")
        return ds
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_all_configs() -> dict[str, dict[str, Any]]:
    configs = {}
    if CARD_CONFIG_PATH.exists():
        configs["CARD"] = load_reason_code_config(CARD_CONFIG_PATH)
    if AMEX_CONFIG_PATH.exists():
        configs["AMEX"] = load_reason_code_config(AMEX_CONFIG_PATH)
    if UPI_CONFIG_PATH.exists():
        configs["UPI"] = load_reason_code_config(UPI_CONFIG_PATH)
    return configs


def get_merged_reason_codes() -> dict[str, Any]:
    all_configs = load_all_configs()
    merged = {}
    for net, conf in all_configs.items():
        merged.update(conf)
    return merged


def evidence_label(evidence_id: str) -> str:
    return evidence_id.replace("_", " ").title()


def status_badge(status: str) -> str:
    return {
        "present": "Present ✅",
        "weak": "Weak ⚠️",
        "missing": "Missing ❌",
        "irrelevant": "Irrelevant 🚫",
    }.get(status, status.title())


def add_simulated_evidence(dispute: dict[str, Any], evidence_ids: list[str]) -> dict[str, Any]:
    updated = json.loads(json.dumps(dispute))
    docs = updated.setdefault("evidence_documents", {})
    for evidence_id in evidence_ids:
        docs[evidence_id] = (
            f"Official {evidence_label(evidence_id)} uploaded by merchant. "
            "Verified match with transaction records and valid timestamp."
        )
    return updated


dataset = load_dataset()
cases = dataset["cases"]
merged_reason_codes = get_merged_reason_codes()

# Initialize Session State
if "selected_dispute_id" not in st.session_state:
    st.session_state.selected_dispute_id = cases[0]["dispute_id"]
if "added_evidence" not in st.session_state:
    st.session_state.added_evidence = {}

# Title & Branding Banner
st.title("🛡️ ProofPilot — AI Dispute Risk Manager")
st.caption("Razorpay Merchant Pre-Submission Evidence Readiness, ML Win Probability & Financial ROI Engine (Track 02)")

# Sidebar Controls
with st.sidebar:
    st.image("https://img.shields.io/badge/Razorpay_Buildathon_2026-Track_02:_AI_Risk_Manager-blue?style=for-the-badge&logo=shield", use_container_width=True)
    st.header("⚡ Dispute Selection")

    # Payment Network Filter
    network_filter = st.radio("Payment Network", ["All Networks", "UPI (NPCI)", "Cards (Visa / MC / RuPay)"], index=0, horizontal=True)

    filtered_cases = cases
    if network_filter == "UPI (NPCI)":
        filtered_cases = [c for c in cases if c.get("network") == "UPI"]
    elif network_filter == "Cards (Visa / MC / RuPay)":
        filtered_cases = [c for c in cases if c.get("network") in {"CARD", "AMEX", "Card (Visa / Mastercard / RuPay)"}]

    if not filtered_cases:
        filtered_cases = cases

    def dispute_label(case: dict[str, Any]) -> str:
        d_id = case.get("dispute_id", case.get("legacy_dispute_id"))
        net = case.get("network", "CARD")
        return f"{d_id} · [{net}] {case['merchant_name']} · {case['reason_title']}"

    dispute_labels = [dispute_label(c) for c in filtered_cases]
    dispute_ids = [c["dispute_id"] for c in filtered_cases]

    if st.session_state.selected_dispute_id not in dispute_ids:
        st.session_state.selected_dispute_id = dispute_ids[0]

    current_index = dispute_ids.index(st.session_state.selected_dispute_id)
    selected_label = st.selectbox("Select Active Case", dispute_labels, index=current_index)
    selected_id = dispute_ids[dispute_labels.index(selected_label)]
    st.session_state.selected_dispute_id = selected_id

    st.divider()

    st.header("🔒 Gateway Webhook Security")
    webhook_event = st.selectbox("Razorpay Webhook Event", ["dispute.created", "dispute.action_required", "dispute.under_review"])
    webhook_secret = st.text_input("Webhook Secret", value=DEFAULT_WEBHOOK_SECRET, type="password")

    st.divider()

    st.header("🧠 Live LLM Reasoning")
    api_key_input = st.text_input("Gemini API Key (Optional)", type="password", help="Live Google Gemini 2.5 Flash structured reasoning. Works offline with rule engine if blank.")

    st.divider()

    # Task 5.2: Chargeback Ratio Health Monitor (VAMP/VCMP)
    st.header("📈 Merchant Portfolio Health")
    st.caption("Chargeback Ratio vs Card Scheme Limits")
    monthly_disputes = st.number_input("Monthly Disputes Count", min_value=1, value=28, step=1)
    monthly_txns = st.number_input("Monthly Total Transactions", min_value=100, value=5000, step=500)
    
    cb_ratio = (monthly_disputes / monthly_txns) * 100
    
    if cb_ratio < 0.65:
        st.success(f"🟢 **Ratio: {cb_ratio:.2f}% (Healthy)**\n\nWell below standard 0.65% early warning threshold.")
    elif cb_ratio <= 0.90:
        st.warning(f"🟡 **Ratio: {cb_ratio:.2f}% (Warning Zone)**\n\nApproaching Visa/Mastercard 0.90% excessive threshold.")
    else:
        st.error(f"🔴 **Ratio: {cb_ratio:.2f}% (VAMP/VCMP Penalty Zone)**\n\nExceeds 0.90% network threshold! Merchants risk mandatory fines.")

# Case State Preparation
base_dispute = next(case for case in cases if case["dispute_id"] == selected_id)
added_for_case = st.session_state.added_evidence.get(selected_id, [])
dispute = add_simulated_evidence(base_dispute, added_for_case)

# Score Case
result = score_dispute(dispute, merged_reason_codes, api_key=api_key_input)
webhook_data = simulate_razorpay_dispute_event(dispute, webhook_event, secret=webhook_secret)

# Razorpay Webhook Event Header Banner
col_top1, col_top2 = st.columns([3, 1])
with col_top1:
    st.info(
        f"📩 **Razorpay Webhook**: `{webhook_event}` | "
        f"**Dispute ID**: `{dispute['dispute_id']}` | "
        f"**Order**: `{dispute['transaction']['order_id']}` | "
        f"**Payment**: `{dispute['transaction']['payment_id']}` | "
        f"**Merchant**: **{dispute['merchant_name']}** | "
        f"**Amount**: ₹{dispute['transaction']['amount']:,} INR ({dispute.get('network', 'CARD')})"
    )
with col_top2:
    if webhook_data["is_verified"]:
        st.success("🔒 **HMAC-SHA256: Verified ✅**")
    else:
        st.error("⚠️ **HMAC Signature Mismatch**")

with st.expander("📩 View Razorpay Webhook JSON Payload & Cryptographic Signature", expanded=False):
    st.code(f"X-Razorpay-Signature: {webhook_data['x_razorpay_signature']}", language="text")
    st.json(webhook_data["event_payload"])

# Top Metrics Cards (Including Economic ROI Decision Engine)
ev_data = result["expected_financial_value"]
rec_data = result["economic_recommendation"]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Evidence Readiness", result["completeness_pct"])
m2.metric("Confidence Score", result["confidence_pct"])
m3.metric("ML Win Prob P(Win)", result["win_probability_pct"])
m4.metric("Expected ROI (EV)", f"₹{ev_data['expected_value_inr']:,.2f}")
m5.metric("Risk Level", result["risk_level"])
m6.metric(
    "Economic Action",
    rec_data["action"],
    delta="Fee Saved ₹500" if rec_data["action"] == "ACCEPT_LOSS" else "+ROI Gain",
    delta_color="normal" if rec_data["action"] == "CONTEST" else "off",
)

# Economic Reasoning Callout
if rec_data["action"] == "CONTEST":
    st.success(f"💡 **Economic Engine Decision**: **{rec_data['action']}** — {rec_data['reason']}")
else:
    st.warning(f"⚠️ **Economic Engine Decision**: **{rec_data['action']}** — {rec_data['reason']} (Saves ₹{rec_data['fee_saved_if_accepted']:.0f} non-refundable chargeback fee)")

st.divider()

# Two Column Layout: Evidence Elements vs Counterfactual Simulator
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📋 Evidence Quality & TF-IDF Similarity Matrix")
    rows = []
    for evidence_id, detail in result["evidence_elements"].items():
        rows.append(
            {
                "Evidence Element": evidence_label(evidence_id),
                "Status": status_badge(detail["status"]),
                "Weight": f"{detail['weight']:.0%}",
                "Contribution": f"{detail['weighted_contribution']:.0%}",
                "TF-IDF Match": f"{detail.get('semantic_relevance', 0.0):.1%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with col_right:
    st.subheader("📈 Counterfactual Risk Simulator (Hero Feature)")
    st.caption("Simulate merchant uploading missing evidence to dynamically project score & risk tier upgrades.")

    fixable = result["missing_evidence"] + result["weak_evidence"]
    if fixable:
        selected_gaps = st.multiselect(
            "Select evidence to provide / strengthen:",
            fixable,
            format_func=evidence_label,
            default=fixable[:1],
        )
        if st.button("🚀 Apply Evidence & Re-Score Case", type="primary", use_container_width=True):
            st.session_state.added_evidence[selected_id] = sorted(
                set(added_for_case + selected_gaps)
            )
            st.rerun()

    if added_for_case:
        if st.button("🔄 Reset Case to Original State", use_container_width=True):
            st.session_state.added_evidence[selected_id] = []
            st.rerun()

    if result["gap_explanation"]:
        st.write("##### Projected Gain for Top Gap:")
        top_gap = result["gap_explanation"][0]
        st.success(
            f"**{evidence_label(top_gap['evidence_id'])}**: Adding this document increases readiness by "
            f"**+{top_gap['potential_score_gain']:.0%}** (Projected Readiness: **{top_gap['score_if_added']:.0%}**, Projected Risk: **{top_gap['projected_risk']}**)."
        )

st.divider()

# Bottom Layout: LLM Drafting Gate & Supervised ML Diagnostics
draft_col, diag_col = st.columns([1.1, 0.9])

with draft_col:
    st.subheader("🤖 AI Reasoning & Confidence Gate")
    if result["routing_decision"] == "auto_draft_response":
        st.success(r"✅ **Gate Status: PASSED** (Readiness $\ge 80\%$, Confidence $\ge 75\%$). Safe for automated submission.")
        response_text = generate_gated_dispute_response(dispute, result, api_key=api_key_input)
        st.text_area("Generated Formal Dispute Response Letter (Razorpay Format)", response_text, height=340)
    else:
        st.warning("🔒 **Gate Status: BLOCKED (High / Medium Risk)** — Response drafting is restricted to protect merchant win rates.")
        gap_advice = generate_llm_gap_explanation(dispute, result["gap_explanation"], api_key=api_key_input)
        st.markdown(gap_advice)

    with st.expander("📝 Decision Audit Trace (Immutable Defense Log)", expanded=False):
        audit_entry = get_audit_log(dispute["dispute_id"]) or result
        st.json(audit_entry)

with diag_col:
    st.subheader("📊 Supervised ML & Track 02 Metrics")
    metrics = evaluate_dataset(DATASET_PATH, ROOT / "config" / "reason_codes", split="test")

    ev1, ev2 = st.columns(2)
    ev1.metric("ML Model ROC-AUC", f"{result.get('ml_model_auc', 0.88):.0%}")
    ev2.metric("Route Accuracy", f"{metrics['route_accuracy']:.0%}")
    ev1.metric("Evidence Precision", f"{metrics['evidence_detection_precision']:.0%}")
    ev2.metric("Evidence Recall", f"{metrics['evidence_detection_recall']:.0%}")
    
    # Explicit False-Positive Cost Display (Track 02 Rubric Requirement)
    fp_col1, fp_col2 = st.columns(2)
    fp_col1.metric("False-Positive Auto-Drafts", f"{metrics['false_positive_auto_drafts']} ({metrics['false_positive_auto_draft_rate']:.0%})")
    fp_col2.metric("Total FP Financial Loss (₹)", f"₹{metrics['total_false_positive_financial_cost_inr']:,.2f}")

    st.write("##### SHAP Model Explainability (Feature Attribution Waterfall):")
    predictor = get_win_predictor()
    if predictor.is_trained and predictor.X_train_arr is not None:
        shap_fig = generate_shap_plot(predictor.model, predictor.X_train_arr, predictor.feature_names)
        st.pyplot(shap_fig)
    else:
        st.bar_chart(pd.DataFrame(
            [{"Feature": k, "Importance": v} for k, v in result.get("ml_feature_importances", {}).items()]
        ).set_index("Feature"))
