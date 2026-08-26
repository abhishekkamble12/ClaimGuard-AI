"""
ProofPilot — Evidence Status Extractor
----------------------------------------
Parses unstructured merchant documents and evidence text to determine status
(present / weak / missing / irrelevant). Uses real Gemini LLM structured extraction
when an API key is provided, with a robust local NLP regex classifier as fallback.
"""

import json
import os
from typing import Any


def classify_evidence_with_rules(raw_text: str) -> str:
    """Deterministic local NLP fallback classifier for evidence quality."""
    if not raw_text or not str(raw_text).strip():
        return "missing"

    text = str(raw_text).strip().lower()

    misleading_markers = (
        "not related to this transaction",
        "unrelated to this transaction",
        "does not match the disputed transaction",
        "wrong transaction",
        "no evidence",
        "not available",
        "cannot confirm",
        "irrelevant",
    )
    weak_markers = (
        "partially available",
        "partial",
        "weak",
        "lacks direct confirmation",
        "incomplete",
        "unclear",
        "lacks customer confirmation",
        "mailroom delivery without signature",
        "pending verification",
    )

    if any(marker in text for marker in misleading_markers):
        return "missing"
    if any(marker in text for marker in weak_markers):
        return "weak"
    return "present"


def classify_evidence_with_llm(
    required_evidence: dict[str, float],
    evidence_documents: dict[str, Any],
    api_key: str,
) -> dict[str, str] | None:
    """
    Call Google Gemini LLM to analyze evidence documents and return structured JSON status classification.
    Returns None if the API call fails, triggering fallback.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        prompt = f"""You are an expert payment risk analyst for ProofPilot.
Analyze the following merchant evidence documents for a chargeback dispute and classify each required evidence element into one of: "present", "weak", "missing", or "irrelevant".

Required Evidence Items:
{json.dumps(required_evidence, indent=2)}

Submitted Evidence Documents:
{json.dumps(evidence_documents, indent=2)}

Return ONLY a valid JSON object mapping each evidence_id to its status ("present", "weak", "missing", or "irrelevant").
Example format:
{{
  "delivery_tracking": "present",
  "delivery_signature": "weak"
}}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        parsed = json.loads(text)
        statuses = {}
        for eid in required_evidence:
            val = parsed.get(eid, "missing").lower()
            statuses[eid] = val if val in {"present", "weak", "missing", "irrelevant"} else "missing"
        return statuses
    except Exception as err:
        # Debug trace or fallback silently
        return None


def extract_and_classify_evidence(
    required_evidence: dict[str, float],
    evidence_documents: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, str]:
    """
    Extract and classify statuses for required evidence items.
    Tries Gemini LLM classification first if an API key is provided,
    otherwise falls back to rule-based classification.
    """
    effective_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if effective_api_key:
        llm_statuses = classify_evidence_with_llm(required_evidence, evidence_documents, effective_api_key)
        if llm_statuses is not None:
            return llm_statuses

    # Fallback to local rule engine
    statuses: dict[str, str] = {}
    for evidence_id in required_evidence:
        raw_doc = evidence_documents.get(evidence_id, "")
        statuses[evidence_id] = classify_evidence_with_rules(raw_doc)

    return statuses
