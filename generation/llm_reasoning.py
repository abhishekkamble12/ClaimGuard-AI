"""
ProofPilot — LLM Reasoning & Gap Explanation Engine
------------------------------------------------------
Provides AI reasoning for:
1. Explaining evidence gaps & recommended actions when cases are incomplete.
2. Generating a formal, high-win-rate dispute response letter when auto-drafting is safe.
Uses Google Gemini LLM when API key is available, with structured local fallbacks.
"""

import os
from typing import Any


def generate_llm_gap_explanation(
    dispute: dict[str, Any],
    gap_analysis: list[dict[str, Any]],
    api_key: str | None = None,
) -> str:
    """Generate structured human-readable gap guidance for the merchant."""
    if not gap_analysis:
        return "All required compelling evidence is verified and present. The dispute packet is ready for submission."

    effective_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if effective_api_key:
        try:
            from google import genai
            client = genai.Client(api_key=effective_api_key)
            prompt = f"""You are ProofPilot AI, an expert dispute risk advisor for Razorpay merchants.
Explain the evidence gaps for this chargeback dispute in a clear, highly actionable markdown format.

Dispute Case:
Merchant: {dispute.get('merchant_name')}
Reason: {dispute.get('reason_title')} ({dispute.get('reason_code')})
Amount: INR {dispute.get('transaction', {}).get('amount')}

Evidence Gap Analysis:
{gap_analysis}

Give 2-3 concise bullet points explaining what evidence is missing or weak, and state which single evidence item will provide the highest readiness gain. Keep it professional and direct.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception:
            pass  # Fallback to local template below

    # Deterministic local fallback
    missing_items = [gap["evidence_id"].replace("_", " ").title() for gap in gap_analysis if gap["current_status"] == "missing"]
    weak_items = [gap["evidence_id"].replace("_", " ").title() for gap in gap_analysis if gap["current_status"] == "weak"]

    advice_lines = [
        f"**Dispute Case Analysis ({dispute.get('reason_title', 'Chargeback')})**",
        "",
        "The current evidence packet is insufficient to win this chargeback. Submitting now risks dispute loss fees.",
    ]

    if missing_items:
        advice_lines.append(f"• **Missing Proof**: Retrieve and upload {', '.join(missing_items)}.")
    if weak_items:
        advice_lines.append(f"• **Weak Proof**: Strengthen details in {', '.join(weak_items)} to ensure explicit transaction matching.")

    top_gap = gap_analysis[0]
    top_name = top_gap["evidence_id"].replace("_", " ").title()
    advice_lines.append(
        f"\n💡 **Top Priority**: Adding **{top_name}** will boost overall evidence readiness score by **+{top_gap['potential_score_gain']:.0%}**."
    )

    return "\n".join(advice_lines)


def generate_gated_dispute_response(
    dispute: dict[str, Any],
    scoring_result: dict[str, Any],
    api_key: str | None = None,
) -> str:
    """Generate structured response letter for gated auto-drafting."""
    transaction = dispute.get("transaction", {})
    merchant_name = dispute.get("merchant_name", "Merchant")
    dispute_id = dispute.get("dispute_id", "disp_0000")
    order_id = transaction.get("order_id", "ORD-0000")
    payment_id = transaction.get("payment_id", "pay_0000")
    amount = transaction.get("amount", 0)
    currency = transaction.get("currency", "INR")

    effective_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if effective_api_key:
        try:
            from google import genai
            client = genai.Client(api_key=effective_api_key)
            prompt = f"""You are ProofPilot AI, drafting a formal chargeback dispute response letter for a Razorpay merchant.

Dispute ID: {dispute_id} | Order ID: {order_id} | Payment ID: {payment_id}
Merchant: {merchant_name}
Reason: {dispute.get('reason_title')} ({dispute.get('reason_code')})
Amount: {currency} {amount}
Verified Evidence Elements: {scoring_result.get('evidence_elements')}
Readiness Score: {scoring_result.get('completeness_pct')} | Confidence: {scoring_result.get('confidence_pct')}

Draft a persuasive, professional chargeback response letter asserting why this claim should be dismissed.
Include transaction details, verified evidence summary, and a formal conclusion.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception:
            pass  # Fallback to local template below

    # Deterministic local fallback template
    evidence_summary = []
    for evidence_id, detail in scoring_result.get("evidence_elements", {}).items():
        if detail["status"] == "present":
            evidence_summary.append(f"- Verified {evidence_id.replace('_', ' ').title()}")

    evidence_block = "\n".join(evidence_summary) if evidence_summary else "- Verified Transaction Log"

    return f"""DISPUTE RESPONSE LETTER — PROOFPILOT AI RISK MANAGER
------------------------------------------------------------
Dispute ID   : {dispute_id}
Order ID     : {order_id}
Payment ID   : {payment_id}
Merchant     : {merchant_name}
Dispute Type : {dispute.get('reason_title', 'Chargeback')}
Disputed Amt : {currency} {amount}
Readiness    : {scoring_result.get('completeness_pct', '100%')} (Confidence: {scoring_result.get('confidence_pct', '100%')})

REASON FOR CONTESTING:
We are contesting this chargeback claim on behalf of {merchant_name}. The cardholder's claim is refuted by verified, compelling evidence attached to this response packet.

EVIDENCE PACKET SUMMARY:
{evidence_block}

TRANSACTION VERIFICATION:
The transaction was processed via Razorpay gateway on {transaction.get('payment_date', 'N/A')}. All customer authentication and delivery tracking signals match the cardholder's order specifications.

CONCLUSION:
Based on the submitted evidence readiness score of {scoring_result.get('completeness_pct')}, we respectfully request the payment network to dismiss this chargeback and credit the disputed funds back to {merchant_name}.

Sincerely,
Risk Operations Team | {merchant_name}
(Powered by ProofPilot AI Risk Manager)
"""
