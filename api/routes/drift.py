"""
ProofPilot — ML Observability & Concept Drift Endpoint
"""

from typing import Any
from fastapi import APIRouter
from monitoring.model_monitor import get_drift_monitor

router = APIRouter(prefix="/drift", tags=["ML Observability"])


@router.get("/evaluate")
def evaluate_model_feature_drift() -> dict[str, Any]:
    """
    Evaluate feature distribution drift (PSI) between active inference
    traffic and baseline training data distribution.
    """
    monitor = get_drift_monitor()
    reports = monitor.evaluate_drift()

    return {
        "status": "evaluated",
        "total_inferences_logged": len(monitor.recent_inferences),
        "features_monitored": len(monitor.feature_names),
        "drift_reports": [r.to_dict() for r in reports],
    }
