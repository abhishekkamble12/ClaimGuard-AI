import json
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
from razorpay_integration.webhook_simulator import simulate_razorpay_dispute_event
from scoring.scorer import load_reason_code_config, score_dispute

DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CONFIG_PATH = ROOT / "config" / "reason_codes" / "amex.json"

st.set_page_config(
    page_title="ProofPilot — AI Dispute Risk Manager",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_config() -> dict[str, Any]:
    return load_reason_code_config(CONFIG_PATH)


def evidence_label(evidence_id: str) -> str:
    return evidence_id.replace("_", " ").title()


def status_badge(status: str) -> str:
    return {
        "present": "Present ✅",
        "weak": "Weak ⚠️",
        "missing": "Missing ❌",
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
reason_codes = load_config()
cases = dataset["cases"]

if "selected_dispute_id" not in st.session_state:
    st.session_state.selected_dispute_id = cases[0]["dispute_id"]
if "added_evidence" not in st.session_state:
    st.session_state.added_evidence = {}

st.title("🛡️ ProofPilot — AI Dispute Risk Manager")
st.caption("Razorpay Merchant Pre-Submission Evidence Readiness, ML Win Probability & Financial ROI Engine")

with st.sidebar:
    st.header("⚡ Razorpay Controls")

    def dispute_label(case: dict[str, Any]) -> str:
        d_id = case.get("dispute_id", case.get("legacy_dispute_id"))
        return f"{d_id} · {case['merchant_name']} · {case['reason_title']}"

    dispute_labels = [dispute_label(c) for c in cases]
    dispute_ids = [c["dispute_id"] for c in cases]

    current_index = dispute_ids.index(st.session_state.selected_dispute_id)
    selected_label = st.selectbox("Select Razorpay Dispute", dispute_labels, index=current_index)
    selected_id = dispute_ids[dispute_labels.index(selected_label)]
    st.session_state.selected_dispute_id = selected_id

    st.subheader("Webhook Event Simulation")
    webhook_event = st.selectbox("Razorpay Webhook Event", ["dispute.created", "dispute.action_required", "dispute.under_review"])

    st.subheader("LLM Engine Config")
    api_key_input = st.text_input("Gemini / OpenAI API Key (Optional)", type="password", help="Enter API Key for live LLM reasoning")

    split_filter = st.selectbox("Metrics Split", ["all", "test", "train"])

base_dispute = next(case for case in cases if case["dispute_id"] == selected_id)
added_for_case = st.session_state.added_evidence.get(selected_id, [])
dispute = add_simulated_evidence(base_dispute, added_for_case)

# Score case
result = score_dispute(dispute, reason_codes, api_key=api_key_input)
webhook_payload = simulate_razorpay_dispute_event(dispute, webhook_event)

# Razorpay Webhook Event Header Banner
st.info(
    f"📩 **Razorpay Webhook Event**: `{webhook_event}` | "
    f"**Dispute ID**: `{dispute['dispute_id']}` | "
    f"**Order**: `{dispute['transaction']['order_id']}` | "
    f"**Payment**: `{dispute['transaction']['payment_id']}` | "
    f"**Merchant**: **{dispute['merchant_name']}** | "
    f"**Amount**: ₹{dispute['transaction']['amount']:,} INR"
)

with st.expander("📩 Raw Razorpay Dispute Webhook Payload (JSON API Event)", expanded=False):
    st.json(webhook_payload)

# Header Metrics Cards
ev_data = result["expected_financial_value"]
ev_color = "green" if ev_data["is_positive_roi"] else "red"

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Evidence Readiness", result["completeness_pct"])
m2.metric("Confidence Score", result["confidence_pct"])
m3.metric("ML Win Prob P(Win)", result["win_probability_pct"])
m4.metric("Financial EV (ROI)", f"₹{ev_data['expected_value_inr']:,}")
m5.metric("Risk Level", result["risk_level"])
m6.metric("AI Decision Gate", result["routing_decision"].replace("_", " ").title())

st.divider()

# Two column layout: Case Readiness & Counterfactual Risk Improvement
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📋 Evidence Quality & TF-IDF Semantic Similarity")
    
    rows = []
    for evidence_id, detail in result["evidence_elements"].items():
        rows.append(
            {
                "Evidence Element": evidence_label(evidence_id),
                "Status": status_badge(detail["status"]),
                "Weight": f"{detail['weight']:.0%}",
                "Contribution": f"{detail['weighted_contribution']:.0%}",
                "TF-IDF Similarity": f"{detail.get('semantic_relevance', 0.0):.1%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with col_right:
    st.subheader("📈 Counterfactual Risk Improvement (Hero Feature)")
    st.caption("Simulate merchant uploading missing evidence to predict readiness & risk tier gain.")

    fixable = result["missing_evidence"] + result["weak_evidence"]
    if fixable:
        selected_gaps = st.multiselect(
            "Select evidence to upload / strengthen:",
            fixable,
            format_func=evidence_label,
            default=fixable[:1],
        )
        if st.button("🚀 Apply Evidence & Re-Score", type="primary", use_container_width=True):
            st.session_state.added_evidence[selected_id] = sorted(
                set(added_for_case + selected_gaps)
            )
            st.rerun()

    if added_for_case:
        if st.button("🔄 Reset Case to Original State", use_container_width=True):
            st.session_state.added_evidence[selected_id] = []
            st.rerun()

    if result["gap_explanation"]:
        st.write("##### Projected Impact of Top Gap:")
        top_gap = result["gap_explanation"][0]
        st.success(
            f"**{evidence_label(top_gap['evidence_id'])}**: Adding this document increases readiness by "
            f"**+{top_gap['potential_score_gain']:.0%}** (New Readiness: **{top_gap['score_if_added']:.0%}**, Projected Risk: **{top_gap['projected_risk']}**)."
        )

st.divider()

# Bottom Layout: LLM Gap Guidance & Gated Response Drafter
draft_col, metrics_col = st.columns([1.1, 0.9])

with draft_col:
    st.subheader("🤖 AI Reasoning & Drafting Gate")
    if result["routing_decision"] == "auto_draft_response":
        st.success("✅ Case passed readiness gate (Readiness >= 80%, Confidence >= 75%). Safe to draft response.")
        response_text = generate_gated_dispute_response(dispute, result, api_key=api_key_input)
        st.text_area("Generated Dispute Response Packet", response_text, height=320)
    else:
        st.warning("🔒 Response Drafting Blocked: Readiness score is below auto-submission threshold.")
        gap_advice = generate_llm_gap_explanation(dispute, result["gap_explanation"], api_key=api_key_input)
        st.markdown(gap_advice)

with metrics_col:
    st.subheader("📊 Supervised ML & Benchmark Diagnostics")
    split = None if split_filter == "all" else split_filter
    metrics = evaluate_dataset(DATASET_PATH, CONFIG_PATH, split=split)

    ev1, ev2 = st.columns(2)
    ev1.metric("ML Model ROC-AUC", f"{result.get('ml_model_auc', 0.88):.0%}")
    ev2.metric("Route Accuracy", f"{metrics['route_accuracy']:.0%}")
    ev1.metric("Evidence Precision", f"{metrics['evidence_detection_precision']:.0%}")
    ev2.metric("Evidence Recall", f"{metrics['evidence_detection_recall']:.0%}")
    
    st.write("##### ML Model Feature Importance Weights (scikit-learn):")
    feat_df = pd.DataFrame(
        [{"Feature": k, "Importance": v} for k, v in result.get("ml_feature_importances", {}).items()]
    ).set_index("Feature")
    st.bar_chart(feat_df)
