"""
ProofPilot — Decision Trace & Audit Log Engine
------------------------------------------------
Provides immutable, defensive-only decision logging for chargeback risk evaluations.
Every dispute routing, ML score, and evidence element is saved to outputs/audit_logs/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "audit_logs"


def log_decision(dispute_id: str, scoring_result: dict[str, Any]) -> Path:
    """
    Log a complete decision audit trail for a dispute evaluation.
    Guarantees transparent, reproducible, and verifiable risk assessments.
    """
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dispute_id": dispute_id,
        "reason_code": scoring_result.get("reason_code"),
        "reason_category": scoring_result.get("reason_category"),
        "completeness_score": scoring_result.get("completeness_score"),
        "confidence": scoring_result.get("confidence"),
        "win_probability": scoring_result.get("win_probability"),
        "routing_decision": scoring_result.get("routing_decision"),
        "risk_level": scoring_result.get("risk_level"),
        "economic_recommendation": scoring_result.get("economic_recommendation"),
        "expected_financial_value": scoring_result.get("expected_financial_value"),
        "missing_evidence": scoring_result.get("missing_evidence", []),
        "weak_evidence": scoring_result.get("weak_evidence", []),
        "evidence_elements": scoring_result.get("evidence_elements", {}),
    }
    path = AUDIT_DIR / f"{dispute_id}.json"
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return path


def get_audit_log(dispute_id: str) -> dict[str, Any] | None:
    """Retrieve existing decision audit log for a dispute."""
    path = AUDIT_DIR / f"{dispute_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None
