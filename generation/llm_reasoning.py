"""
ProofPilot — LLM Reasoning & Gap Explanation Engine
------------------------------------------------------
Provides AI reasoning for:
1. Explaining evidence gaps & recommended actions when cases are incomplete.
2. Generating a formal, high-win-rate dispute response letter when auto-drafting is safe.
3. Generating XAI decision narrative summaries across all 4 explainability pillars.
4. Streaming real-time response generation with Google Gemini 2.5 Flash & local fallbacks.

Hardened with PromptGuard defense layer, ResilientCallExecutor (retries/timeouts/circuit-breaker),
and InferenceTracer telemetry spans.
"""

import logging
import os
from collections.abc import Iterator
from typing import Any

from generation.prompt_guard import PromptGuard
from monitoring.trace import get_global_tracer, trace_span
from utils.resilience import ResilientCallExecutor, retry_with_backoff

logger = logging.getLogger("proofpilot.generation")

# ── LLM Provider Configuration ────────────────────────────────────────────────
# Set LLM_PROVIDER=groq (default) | gemini | local in your .env
# GROQ_API_KEY  — get free key at https://console.groq.com
# GEMINI_API_KEY — get free key at https://aistudio.google.com
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _get_active_provider() -> str:
    """Resolve the active LLM provider based on available API keys and LLM_PROVIDER setting."""
    provider = LLM_PROVIDER
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        # Auto-downgrade to gemini if Groq key missing
        provider = "gemini"
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        provider = "local"
    return provider


def _call_llm(prompt: str, api_key: str | None = None) -> tuple[str, str]:
    """
    Unified LLM call router: Groq → Gemini → raises RuntimeError.
    Returns (response_text, provider_used).
    Priority:
      1. Groq Cloud (llama-3.3-70b-versatile, ~700 tok/sec) — if GROQ_API_KEY set
      2. Google Gemini 2.5 Flash — if GEMINI_API_KEY set
      Raises RuntimeError if neither key is available (triggers local fallback).
    """
    groq_key = api_key if LLM_PROVIDER == "groq" else None
    groq_key = groq_key or os.getenv("GROQ_API_KEY")
    gemini_key = api_key if LLM_PROVIDER == "gemini" else None
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")

    # ── 1. Groq (primary) ──────────────────────────────────────────────────
    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
                stream=False,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip(), f"groq/{GROQ_MODEL}"
        except Exception as exc:
            logger.warning("Groq LLM call failed (%s). Trying Gemini fallback.", exc)

    # ── 2. Gemini (secondary fallback) ─────────────────────────────────────
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = resp.text or ""
            if text.strip():
                return text.strip(), f"gemini/{GEMINI_MODEL}"
        except Exception as exc:
            logger.warning("Gemini LLM call failed (%s). Using local fallback.", exc)

    raise RuntimeError("No LLM provider available — using local deterministic fallback.")


def _stream_llm(prompt: str, api_key: str | None = None):
    """
    Unified streaming LLM generator: Groq → Gemini → local fallback.
    Yields text chunks. Returns (provider_used,) via StopIteration value.
    """
    groq_key = api_key if LLM_PROVIDER == "groq" else None
    groq_key = groq_key or os.getenv("GROQ_API_KEY")
    gemini_key = api_key if LLM_PROVIDER == "gemini" else None
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")

    # ── 1. Groq streaming ──────────────────────────────────────────────────
    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
            with client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
                stream=True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
            return  # Successfully streamed from Groq
        except Exception as exc:
            logger.warning("Groq streaming failed (%s). Trying Gemini stream.", exc)

    # ── 2. Gemini streaming ────────────────────────────────────────────────
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            for chunk in client.models.generate_content_stream(model=GEMINI_MODEL, contents=prompt):
                if chunk.text:
                    yield chunk.text
            return
        except Exception as exc:
            logger.warning("Gemini streaming failed (%s). Falling back to local stream.", exc)

    # ── 3. Local fallback ──────────────────────────────────────────────────
    return  # Caller handles local fallback when generator is empty


def _get_guard() -> PromptGuard:
    return PromptGuard()


def _get_executor(timeout_seconds: float = 15.0, max_retries: int = 3) -> ResilientCallExecutor:
    return ResilientCallExecutor(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        initial_delay=0.5,
        backoff_factor=2.0,
    )


def _build_local_gap_explanation(dispute: dict[str, Any], gap_analysis: list[dict[str, Any]]) -> str:
    """Deterministic local fallback for gap explanations."""
    if not gap_analysis:
        return "All required compelling evidence is verified and present. The dispute packet is ready for submission."

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


def _build_local_dispute_response(dispute: dict[str, Any], scoring_result: dict[str, Any]) -> str:
    """Deterministic local fallback template for dispute response letters."""
    transaction = dispute.get("transaction", {})
    merchant_name = dispute.get("merchant_name", "Merchant")
    dispute_id = dispute.get("dispute_id", "disp_0000")
    order_id = transaction.get("order_id", "ORD-0000")
    payment_id = transaction.get("payment_id", "pay_0000")
    amount = transaction.get("amount", 0)
    currency = transaction.get("currency", "INR")

    evidence_summary = []
    for evidence_id, detail in scoring_result.get("evidence_elements", {}).items():
        if detail.get("status") == "present":
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


def _build_local_xai_summary(dispute: dict[str, Any], result: dict[str, Any]) -> str:
    """Deterministic local fallback template for XAI narrative summaries."""
    dispute_id = result.get("dispute_id", "—")
    reason_title = result.get("reason_title", "Chargeback")
    reason_code = result.get("reason_code", "—")
    score_pct = result.get("completeness_pct", "—")
    conf_pct = result.get("confidence_pct", "—")
    pwin_pct = result.get("win_probability_pct", "—")
    ev = result.get("expected_financial_value", {}).get("expected_value_inr", 0)
    routing = result.get("routing_decision", "—").replace("_", " ").title()
    risk = result.get("risk_level", "—")
    gaps = result.get("gap_explanation", [])
    missing_items = [g["evidence_id"].replace("_", " ").title() for g in gaps if g.get("current_status") == "missing"]
    weak_items = [g["evidence_id"].replace("_", " ").title() for g in gaps if g.get("current_status") == "weak"]
    top_gap = gaps[0] if gaps else None

    lines = [f"📋 **XAI Summary for `{dispute_id}`** — {reason_title} ({reason_code})", ""]

    if missing_items:
        lines.append(
            f"• **Readiness ({score_pct})**: Penalized by {len(missing_items)} missing item(s) — "
            f"{', '.join(missing_items[:3])}. These contribute zero weight to the score."
        )
    elif weak_items:
        lines.append(
            f"• **Readiness ({score_pct})**: Partial — {len(weak_items)} weak item(s) ({', '.join(weak_items[:3])}) "
            f"contribute only 50% of their potential weight."
        )
    else:
        lines.append(f"• **Readiness ({score_pct})**: All required evidence is present and strong. ✅")

    if weak_items:
        lines.append(
            f"• **Confidence ({conf_pct})**: Reduced by −{len(weak_items) * 10}% penalty from "
            f"{len(weak_items)} weak item(s). Strengthening these removes the penalty."
        )
    else:
        lines.append(f"• **Confidence ({conf_pct})**: High — no weak evidence items detected.")

    lines.append(
        f"• **ML Win Probability P(Win) = {pwin_pct}**: Driven by completeness ({score_pct}), "
        f"confidence ({conf_pct}), and {len(missing_items)} missing / {len(weak_items)} weak item(s). "
        + ("Model predicts favorable outcome." if result.get("win_probability", 0) >= 0.50
           else "Model predicts unfavorable outcome — evidence gaps reduce win probability significantly.")
    )

    ev_sign = "positive" if ev > 0 else "negative"
    lines.append(
        f"• **Expected ROI (EV = ₹{ev:,.2f})**: {ev_sign.title()} — "
        + ("contesting is financially worthwhile." if ev > 0
           else "contesting costs more than the potential recovery. Accepting loss saves ₹500 dispute fee.")
    )

    lines.append(
        f"• **Routing = {routing} | Risk = {risk}**: "
        + {
            "auto_draft_response": "Both readiness and confidence gates passed — safe to auto-draft and submit.",
            "request_more_evidence": f"Readiness gate passed ({score_pct} ≥ 50%) but auto-draft gate failed — gather missing evidence before submission.",
            "human_review": f"Both gates failed ({score_pct} readiness, {conf_pct} confidence) — case escalated to human review.",
        }.get(result.get("routing_decision", ""), "")
    )

    if top_gap:
        lines.append(
            f"• **Top Action**: Upload **{top_gap['evidence_id'].replace('_', ' ').title()}** → "
            f"readiness jumps from {score_pct} to **{top_gap['score_if_added']:.0%}** "
            f"(+{top_gap['potential_score_gain']:.0%} gain, projected risk: **{top_gap['projected_risk']}**)."
        )
    else:
        lines.append("• **Top Action**: Evidence packet is complete — proceed to submission.")

    return "\n".join(lines)


def generate_llm_gap_explanation(
    dispute: dict[str, Any],
    gap_analysis: list[dict[str, Any]],
    api_key: str | None = None,
) -> str:
    """
    Generate structured human-readable gap guidance for the merchant.
    Uses Google Gemini LLM with PromptGuard defense, TraceSpan telemetry,
    and ResilientCallExecutor retries.
    """
    if not gap_analysis:
        return "All required compelling evidence is verified and present. The dispute packet is ready for submission."

    guard = _get_guard()
    dispute_id = dispute.get("dispute_id", "disp_unknown")

    # Inspect inputs for prompt injection
    check_text = f"{dispute.get('merchant_name', '')} {dispute.get('reason_title', '')} {gap_analysis}"
    val_res = guard.inspect(check_text)
    if not val_res.is_safe:
        logger.warning(
            "Prompt injection patterns detected in gap explanation inputs for %s: %s (risk score: %.2f)",
            dispute_id,
            val_res.flagged_patterns,
            val_res.risk_score,
        )
        return _build_local_gap_explanation(dispute, gap_analysis)

    preamble = PromptGuard.get_hardened_system_preamble()
    framed_merchant = guard.frame_untrusted_input(str(dispute.get("merchant_name", "")), "merchant_name")
    framed_reason = guard.frame_untrusted_input(f"{dispute.get('reason_title')} ({dispute.get('reason_code')})", "dispute_reason")
    framed_gaps = guard.frame_untrusted_input(str(gap_analysis), "gap_analysis")

    prompt = f"""{preamble}

You are ProofPilot AI, an expert dispute risk advisor for Razorpay merchants.
Explain the evidence gaps for this chargeback dispute in a clear, highly actionable markdown format.

Dispute Case:
Merchant: {framed_merchant}
Reason: {framed_reason}
Amount: INR {dispute.get('transaction', {}).get('amount')}

Evidence Gap Analysis:
{framed_gaps}

Give 2-3 concise bullet points explaining what evidence is missing or weak, and state which single evidence item will provide the highest readiness gain. Keep it professional and direct.
"""

    active_provider = _get_active_provider()
    if active_provider != "local":
        try:
            executor = _get_executor(timeout_seconds=15.0, max_retries=3)

            def _call_routed_llm() -> str:
                text, _ = _call_llm(prompt, api_key=api_key)
                return text

            with trace_span(operation_name="llm_gap_explanation", dispute_id=dispute_id, model=active_provider) as span:
                result_text = executor.execute(
                    primary_fn=_call_routed_llm,
                    fallback_fn=lambda: _build_local_gap_explanation(dispute, gap_analysis),
                )
                span.finish(status="SUCCESS", input_tokens=len(prompt.split()), output_tokens=len(result_text.split()))
                return result_text

        except Exception as exc:
            logger.warning("LLM gap explanation call failed for %s (%s). Using local fallback.", dispute_id, exc)

    return _build_local_gap_explanation(dispute, gap_analysis)


def generate_gated_dispute_response(
    dispute: dict[str, Any],
    scoring_result: dict[str, Any],
    api_key: str | None = None,
) -> str:
    """
    Generate structured response letter for gated auto-drafting.
    Protected by PromptGuard framing, TraceSpan telemetry, and ResilientCallExecutor.
    """
    guard = _get_guard()
    dispute_id = dispute.get("dispute_id", "disp_0000")
    transaction = dispute.get("transaction", {})
    order_id = transaction.get("order_id", "ORD-0000")
    payment_id = transaction.get("payment_id", "pay_0000")
    amount = transaction.get("amount", 0)
    currency = transaction.get("currency", "INR")

    # Inspect inputs for prompt injection
    raw_check = f"{dispute.get('merchant_name', '')} {dispute.get('reason_title', '')} {scoring_result.get('evidence_elements', {})}"
    val_res = guard.inspect(raw_check)
    if not val_res.is_safe:
        logger.warning(
            "Prompt injection detected in dispute response inputs for %s: %s",
            dispute_id,
            val_res.flagged_patterns,
        )
        return _build_local_dispute_response(dispute, scoring_result)

    preamble = PromptGuard.get_hardened_system_preamble()
    framed_merchant = guard.frame_untrusted_input(str(dispute.get("merchant_name", "Merchant")), "merchant_name")
    framed_reason = guard.frame_untrusted_input(f"{dispute.get('reason_title')} ({dispute.get('reason_code')})", "dispute_reason")
    framed_evidence = guard.frame_untrusted_input(str(scoring_result.get("evidence_elements", {})), "verified_evidence")

    prompt = f"""{preamble}

You are ProofPilot AI, drafting a formal chargeback dispute response letter for a Razorpay merchant.

Dispute ID: {dispute_id} | Order ID: {order_id} | Payment ID: {payment_id}
Merchant: {framed_merchant}
Reason: {framed_reason}
Amount: {currency} {amount}
Verified Evidence Elements: {framed_evidence}
Readiness Score: {scoring_result.get('completeness_pct')} | Confidence: {scoring_result.get('confidence_pct')}

Draft a persuasive, professional chargeback response letter asserting why this claim should be dismissed.
Include transaction details, verified evidence summary, and a formal conclusion.
"""

    active_provider = _get_active_provider()
    if active_provider != "local":
        try:
            executor = _get_executor(timeout_seconds=15.0, max_retries=3)

            def _call_routed_llm() -> str:
                text, _ = _call_llm(prompt, api_key=api_key)
                return text

            with trace_span(operation_name="llm_gated_dispute_response", dispute_id=dispute_id, model=active_provider) as span:
                result_text = executor.execute(
                    primary_fn=_call_routed_llm,
                    fallback_fn=lambda: _build_local_dispute_response(dispute, scoring_result),
                )
                span.finish(status="SUCCESS", input_tokens=len(prompt.split()), output_tokens=len(result_text.split()))
                return result_text

        except Exception as exc:
            logger.warning("LLM dispute response generation failed for %s (%s). Using local fallback.", dispute_id, exc)

    return _build_local_dispute_response(dispute, scoring_result)


def generate_xai_summary(
    dispute: dict[str, Any],
    result: dict[str, Any],
    api_key: str | None = None,
) -> str:
    """
    Phase 4 XAI — Generate a concise human-readable narrative explanation card
    summarising ALL ProofPilot scoring decisions for the current dispute.
    Protected by PromptGuard, TraceSpan, and ResilientCallExecutor.
    """
    guard = _get_guard()
    dispute_id = result.get("dispute_id", "—")
    reason_title = result.get("reason_title", "Chargeback")
    reason_code = result.get("reason_code", "—")
    score_pct = result.get("completeness_pct", "—")
    conf_pct = result.get("confidence_pct", "—")
    pwin_pct = result.get("win_probability_pct", "—")
    ev = result.get("expected_financial_value", {}).get("expected_value_inr", 0)
    routing = result.get("routing_decision", "—").replace("_", " ").title()
    risk = result.get("risk_level", "—")
    action = result.get("economic_recommendation", {}).get("action", "—")
    gaps = result.get("gap_explanation", [])
    missing_items = [g["evidence_id"].replace("_", " ").title() for g in gaps if g.get("current_status") == "missing"]
    weak_items = [g["evidence_id"].replace("_", " ").title() for g in gaps if g.get("current_status") == "weak"]
    top_gap = gaps[0] if gaps else None

    # Inspect inputs
    val_res = guard.inspect(f"{dispute.get('merchant_name', '')} {reason_title}")
    if not val_res.is_safe:
        logger.warning("Prompt injection flagged in XAI summary for %s: %s", dispute_id, val_res.flagged_patterns)
        return _build_local_xai_summary(dispute, result)

    preamble = PromptGuard.get_hardened_system_preamble()
    framed_merchant = guard.frame_untrusted_input(str(dispute.get("merchant_name", "Merchant")), "merchant_name")
    framed_reason = guard.frame_untrusted_input(f"{reason_title} ({reason_code})", "dispute_reason")
    framed_gaps = guard.frame_untrusted_input(f"Missing: {missing_items}, Weak: {weak_items}", "evidence_gaps")

    prompt = f"""{preamble}

You are ProofPilot XAI, an explainability engine for an AI chargeback dispute risk manager.

Generate a concise 5-6 bullet point XAI narrative summary explaining all AI decisions for this dispute.
Be direct, numerical, and merchant-friendly. Each bullet should start with a bold label.

Dispute: {dispute_id} | Merchant: {framed_merchant}
Reason: {framed_reason}
Amount: ₹{dispute.get("transaction", {}).get("amount", 0):,}
Readiness Score: {score_pct} | Confidence: {conf_pct}
ML Win Probability P(Win): {pwin_pct} | Expected ROI (EV): ₹{ev:,.2f}
Routing Decision: {routing} | Risk Tier: {risk} | Economic Action: {action}
Evidence State: {framed_gaps}
Top Recommended Action: {top_gap["explanation"] if top_gap else "N/A"}

Format as 5-6 bullet points (no headings). Each bullet: **Label**: explanation.
"""

    active_provider = _get_active_provider()
    if active_provider != "local":
        try:
            executor = _get_executor(timeout_seconds=15.0, max_retries=3)

            def _call_routed_llm() -> str:
                text, _ = _call_llm(prompt, api_key=api_key)
                return text

            with trace_span(operation_name="llm_xai_summary", dispute_id=dispute_id, model=active_provider) as span:
                result_text = executor.execute(
                    primary_fn=_call_routed_llm,
                    fallback_fn=lambda: _build_local_xai_summary(dispute, result),
                )
                span.finish(status="SUCCESS", input_tokens=len(prompt.split()), output_tokens=len(result_text.split()))
                return result_text

        except Exception as exc:
            logger.warning("LLM XAI summary failed for %s (%s). Using local fallback.", dispute_id, exc)

    return _build_local_xai_summary(dispute, result)


def stream_gated_dispute_response(
    dispute: dict[str, Any],
    scoring_result: dict[str, Any],
    api_key: str | None = None,
) -> Iterator[str]:
    """
    Stream dispute response chunks in real-time.
    Routes to Groq (primary, ~700 tok/sec) → Gemini (secondary) → local line-by-line fallback.
    """
    guard = _get_guard()
    dispute_id = dispute.get("dispute_id", "disp_0000")
    transaction = dispute.get("transaction", {})
    order_id = transaction.get("order_id", "ORD-0000")
    payment_id = transaction.get("payment_id", "pay_0000")
    amount = transaction.get("amount", 0)
    currency = transaction.get("currency", "INR")

    preamble = PromptGuard.get_hardened_system_preamble()
    framed_merchant = guard.frame_untrusted_input(str(dispute.get("merchant_name", "Merchant")), "merchant_name")
    framed_reason = guard.frame_untrusted_input(f"{dispute.get('reason_title')} ({dispute.get('reason_code')})", "dispute_reason")
    framed_evidence = guard.frame_untrusted_input(str(scoring_result.get("evidence_elements", {})), "verified_evidence")

    prompt = f"""{preamble}

You are ProofPilot AI, drafting a formal chargeback dispute response letter for a Razorpay merchant.

Dispute ID: {dispute_id} | Order ID: {order_id} | Payment ID: {payment_id}
Merchant: {framed_merchant}
Reason: {framed_reason}
Amount: {currency} {amount}
Verified Evidence Elements: {framed_evidence}
Readiness Score: {scoring_result.get('completeness_pct')} | Confidence: {scoring_result.get('confidence_pct')}

Draft a persuasive, professional chargeback response letter asserting why this claim should be dismissed.
Include transaction details, verified evidence summary, and a formal conclusion.
"""

    active_provider = _get_active_provider()
    if active_provider != "local":
        with trace_span(operation_name="llm_stream_dispute_response", dispute_id=dispute_id, model=active_provider) as span:
            total_chars = 0
            yielded_any = False
            try:
                for chunk in _stream_llm(prompt, api_key=api_key):
                    yielded_any = True
                    total_chars += len(chunk)
                    yield chunk

                if yielded_any:
                    span.finish(status="SUCCESS", input_tokens=len(prompt.split()), output_tokens=max(1, total_chars // 4))
                    return
            except Exception as exc:
                logger.warning("Streaming LLM response failed for %s (%s). Falling back to local stream.", dispute_id, exc)

    # Local fallback: stream line-by-line
    fallback_letter = _build_local_dispute_response(dispute, scoring_result)
    for line in fallback_letter.splitlines(keepends=True):
        yield line


def batch_generate_llm_responses(
    prompts: list[str],
    api_key: str | None = None,
) -> list[str]:
    """
    Bulk LLM calls via unified router (Groq → Gemini → local fallback) with per-prompt error recovery.
    """
    if not prompts:
        return []

    active_provider = _get_active_provider()
    results = []

    if active_provider != "local":
        for prompt in prompts:
            try:
                text, provider = _call_llm(prompt, api_key=api_key)
                results.append(text)
            except Exception as ex:
                results.append(f"[ProofPilot Local Fallback]: Batch payload processed ({prompt[:100]}...)")
        return results

    # Full local fallback
    for p in prompts:
        results.append(f"[ProofPilot Local Fallback]: Batch payload processed ({p[:100]}...)")
    return results
