"""
ProofPilot — ML Observability & Concept Drift Endpoint
-------------------------------------------------------
Provides live demo-ready Population Stability Index (PSI) tracking,
drift alerting, and synthetic drift simulation for presentation/production.
"""

from typing import Any
import numpy as np
from fastapi import APIRouter
from monitoring.model_monitor import get_drift_monitor

router = APIRouter(prefix="/drift", tags=["ML Observability"])


@router.get("")
@router.get("/status")
def get_drift_status() -> dict[str, Any]:
    """
    Get live system drift status, PSI alerts, and retrain recommendations.
    """
    monitor = get_drift_monitor()
    reports = monitor.evaluate_drift()
    should_retrain, drifted = monitor.should_retrain(psi_threshold=0.20)

    # Sort reports by PSI descending
    sorted_reports = sorted(reports, key=lambda r: r.psi_score, reverse=True)
    max_psi = sorted_reports[0].psi_score if sorted_reports else 0.0

    if max_psi >= 0.25 or should_retrain:
        health_status = "CRITICAL_DRIFT"
        recommendation = "Automated retraining required: significant distribution shift detected."
    elif max_psi >= 0.10:
        health_status = "MODERATE_DRIFT"
        recommendation = "Distribution shift observed: monitor incoming dispute characteristics."
    else:
        health_status = "HEALTHY"
        recommendation = "Feature distributions match training baseline within operating bounds."

    return {
        "status": health_status,
        "should_retrain": should_retrain,
        "max_psi": max_psi,
        "drifted_features": drifted,
        "drifted_feature_count": len(drifted),
        "total_inferences_monitored": len(monitor.recent_inferences),
        "features_tracked_count": len(monitor.feature_names),
        "recommendation": recommendation,
        "top_drift_features": [r.to_dict() for r in sorted_reports[:5]],
    }


@router.get("/evaluate")
def evaluate_model_feature_drift() -> dict[str, Any]:
    """
    Evaluate feature distribution drift (PSI) between active inference
    traffic and baseline training data distribution.
    """
    monitor = get_drift_monitor()
    reports = monitor.evaluate_drift()
    should_retrain, drifted = monitor.should_retrain(psi_threshold=0.20)

    return {
        "status": "evaluated",
        "should_retrain": should_retrain,
        "drifted_features": drifted,
        "total_inferences_logged": len(monitor.recent_inferences),
        "features_monitored": len(monitor.feature_names),
        "drift_reports": [r.to_dict() for r in reports],
    }


@router.get("/alert")
def get_drift_alert() -> dict[str, Any]:
    """
    Lightweight health-check endpoint for automated alerting and retrain webhooks.
    """
    monitor = get_drift_monitor()
    should_retrain, drifted = monitor.should_retrain(psi_threshold=0.20)
    reports = monitor.evaluate_drift()
    max_psi = max([r.psi_score for r in reports], default=0.0)

    return {
        "alert": should_retrain,
        "level": "CRITICAL" if max_psi >= 0.25 else ("WARNING" if should_retrain else "INFO"),
        "max_psi": max_psi,
        "drifted_features": drifted,
        "inferences_logged": len(monitor.recent_inferences),
    }


@router.post("/simulate_drift")
def simulate_traffic_drift(n_cases: int = 30) -> dict[str, Any]:
    """
    LIVE DEMO ENDPOINT: Injects synthetic drifted inferences (e.g. sudden influx of high-amount disputes
    with missing evidence) to showcase live PSI drift alerting and retrain trigger in real time.
    """
    monitor = get_drift_monitor()
    # Ensure baseline is set
    monitor.evaluate_drift()

    if monitor.baseline_features is None:
        return {"error": "Baseline features not yet initialized"}

    # Generate drifted feature vectors: extreme normalized amounts, low completeness, high missing count
    n_features = len(monitor.feature_names)
    injected_count = 0

    for _ in range(n_cases):
        # Create synthetic shifted feature vector (simulates aggressive fraud wave)
        drifted_feat = [0.0] * n_features
        # Feature 0: Completeness Score -> shifted to near 0.1
        drifted_feat[0] = float(np.random.uniform(0.05, 0.20))
        # Feature 1: Confidence Score -> shifted to near 0.1
        drifted_feat[1] = float(np.random.uniform(0.05, 0.20))
        # Feature 2: Missing Count -> shifted to 4-5
        drifted_feat[2] = float(np.random.choice([4.0, 5.0]))
        # Feature 3: Weak Count -> 1.0
        drifted_feat[3] = 1.0
        # Feature 4: Normalized Amount -> extreme amount (>Rs 50k -> norm > 5.0)
        drifted_feat[4] = float(np.random.uniform(5.0, 10.0))
        # Feature 5: Log Amount -> high
        drifted_feat[5] = float(np.random.uniform(1.0, 1.2))
        # Any critical missing -> 1.0
        if "Any Critical Missing" in monitor.feature_names:
            idx = monitor.feature_names.index("Any Critical Missing")
            drifted_feat[idx] = 1.0

        # Log inference
        monitor.log_inference(drifted_feat, predicted_prob=0.08)
        injected_count += 1

    # Re-evaluate drift
    reports = monitor.evaluate_drift()
    should_retrain, drifted = monitor.should_retrain(psi_threshold=0.20)
    sorted_reports = sorted(reports, key=lambda r: r.psi_score, reverse=True)

    return {
        "status": "drift_injected",
        "injected_cases": injected_count,
        "total_inferences_now": len(monitor.recent_inferences),
        "should_retrain": should_retrain,
        "drifted_features": drifted,
        "top_drifted_features": [r.to_dict() for r in sorted_reports[:5]],
        "demo_note": "Simulated live fraud wave. PSI threshold crossed -> automated retrain alert active.",
    }


@router.post("/reset")
def reset_drift_monitor() -> dict[str, Any]:
    """
    Reset recent inferences buffer to restore monitor to clean state.
    """
    monitor = get_drift_monitor()
    monitor.recent_inferences.clear()
    monitor.recent_predictions.clear()
    return {"status": "reset", "inferences_logged": 0}
