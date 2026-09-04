"""
ProofPilot — Decision Trace & Audit Log Engine
------------------------------------------------
Provides immutable, defensive-only decision logging for chargeback risk evaluations.
Every dispute routing, ML score, and evidence element is saved to outputs/audit_logs/.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from utils.logging_config import get_logger

logger = get_logger(__name__)

AUDIT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "audit_logs"


def log_decision(dispute_id: str, scoring_result: dict[str, Any], dispute: dict[str, Any] | None = None) -> Path:
    """
    Log a complete decision audit trail for a dispute evaluation.
    Guarantees transparent, reproducible, and verifiable risk assessments.
    """
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dispute_id": dispute_id,
        "dispute": dispute,
        "reason_code": scoring_result.get("reason_code"),
        "reason_category": scoring_result.get("reason_category"),
        "completeness_score": scoring_result.get("completeness_score"),
        "confidence": scoring_result.get("confidence"),
        "win_probability": scoring_result.get("win_probability"),
        "risk_score_100": scoring_result.get("risk_score_100"),
        "risk_tier": scoring_result.get("risk_tier"),
        "model_confidence_pct": scoring_result.get("model_confidence_pct"),
        "executive_summary": scoring_result.get("executive_summary"),
        "routing_decision": scoring_result.get("routing_decision"),
        "risk_level": scoring_result.get("risk_level"),
        "economic_recommendation": scoring_result.get("economic_recommendation"),
        "expected_financial_value": scoring_result.get("expected_financial_value"),
        "missing_evidence": scoring_result.get("missing_evidence", []),
        "weak_evidence": scoring_result.get("weak_evidence", []),
        "evidence_elements": scoring_result.get("evidence_elements", {}),
        "gap_explanation": scoring_result.get("gap_explanation", []),
        "local_shap_explanation": scoring_result.get("local_shap_explanation"),
        "decision_trace": scoring_result.get("decision_trace"),
    }
    # Validate dispute_id to prevent path traversal attacks.
    if not re.fullmatch(r'[A-Za-z0-9_-]+', dispute_id):
        logger.error(f'Invalid dispute_id supplied: {dispute_id}')
        raise ValueError('Malformed dispute_id')
    path = AUDIT_DIR / f"{dispute_id}.json"
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    logger.info(f"Audit trace recorded for dispute {dispute_id} -> {path}")
    return path


def get_audit_log(dispute_id: str) -> dict[str, Any] | None:
    """Retrieve existing decision audit log for a dispute."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', dispute_id):
        logger.error(f'Invalid dispute_id supplied: {dispute_id}')
        raise ValueError('Malformed dispute_id')
    path = AUDIT_DIR / f"{dispute_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read audit log {path}: {exc}")
            return None
    return None


def prune_audit_logs(days: int = 90) -> int:
    """
    Delete audit log JSON files in AUDIT_DIR older than `days` days.
    Returns the total number of deleted log files.
    """
    if not AUDIT_DIR.exists():
        return 0

    cutoff_seconds = days * 86400
    now = time.time()
    deleted_count = 0

    for log_file in AUDIT_DIR.glob("*.json"):
        try:
            file_age = now - log_file.stat().st_mtime
            if file_age > cutoff_seconds:
                log_file.unlink()
                deleted_count += 1
        except Exception as exc:
            logger.warning(f"Error deleting audit log {log_file}: {exc}")

    logger.info(f"Pruned {deleted_count} audit logs older than {days} days.")
    return deleted_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ProofPilot Audit Log Rotator & Maintenance Tool")
    parser.add_argument("--prune", type=int, default=90, help="Prune audit logs older than N days (default: 90)")
    args = parser.parse_args()

    pruned = prune_audit_logs(args.prune)
    print(f"Audit Log Maintenance Complete: Pruned {pruned} log file(s) older than {args.prune} days.")
