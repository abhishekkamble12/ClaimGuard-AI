"""
ProofPilot — API v1 drift endpoints (wrapper for existing drift implementation)
Provides /api/v1/drift and /api/v1/drift/alert endpoints as required.
"""

from fastapi import APIRouter
from api.routes.drift import (
    get_drift_status as _get_drift_status,
    get_drift_alert as _get_drift_alert,
    evaluate_model_feature_drift as _evaluate_model_feature_drift,
    simulate_traffic_drift as _simulate_traffic_drift,
    reset_drift_monitor as _reset_drift_monitor,
)

router = APIRouter(prefix="/api/v1/drift", tags=["ML Observability v1"])

@router.get("")
@router.get("/status")
def get_status():
    """Alias to existing drift status endpoint."""
    return _get_drift_status()

@router.get("/alert")
def get_alert():
    """Alias to existing drift alert endpoint."""
    return _get_drift_alert()

@router.get("/evaluate")
def evaluate():
    """Alias to existing drift evaluation endpoint."""
    return _evaluate_model_feature_drift()

@router.post("/simulate_drift")
def simulate_drift(n_cases: int = 30):
    """Alias to existing simulate drift endpoint."""
    return _simulate_traffic_drift(n_cases=n_cases)

@router.post("/reset")
def reset():
    """Alias to existing drift reset endpoint."""
    return _reset_drift_monitor()
