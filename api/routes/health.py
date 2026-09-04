import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from ml.model_registry import get_model_registry
from ml.win_predictor import get_win_predictor
from monitoring.model_monitor import get_drift_monitor
from monitoring.trace import get_global_tracer

router = APIRouter(tags=["Health & Diagnostics"])
ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_PATH = ROOT / "outputs" / "metrics.json"
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"


def _get_dataset_hash() -> str:
    if DATASET_PATH.exists():
        try:
            return hashlib.md5(DATASET_PATH.read_bytes()).hexdigest()[:12]
        except Exception:
            pass
    return "unknown"


@router.get("/health")
def health_check() -> dict:
    predictor = get_win_predictor()
    registry = get_model_registry()
    drift_mon = get_drift_monitor()

    drift_reports = drift_mon.evaluate_drift()
    last_psi = round(max((r.psi_score for r in drift_reports), default=0.0), 4)
    should_retrain, drifted_features = drift_mon.should_retrain()

    return {
        "status": "healthy",
        "service": "ProofPilot AI Dispute Risk Manager",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "2.0.0",
        "active_model": registry.data.get("active_model", "calibrated_stacked_ensemble"),
        "training_timestamp": registry.data.get("last_updated"),
        "dataset_hash": _get_dataset_hash(),
        "active_threshold": predictor.ev_optimal_threshold,
        "model_trained": predictor.is_trained,
        "ensemble_roc_auc": predictor.auc_score,
        "ensemble_brier_score": predictor.brier_score,
        "last_psi_score": last_psi,
        "drift_status": "CRITICAL_DRIFT" if should_retrain else ("MODERATE_DRIFT" if last_psi >= 0.10 else "STABLE"),
        "should_retrain": should_retrain,
        "drifted_features": drifted_features,
    }


@router.get("/metrics")
def operational_metrics() -> dict:
    metrics_data = {}
    if METRICS_PATH.exists():
        try:
            metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics_data = {}

    tracer = get_global_tracer()
    tracer_summary = tracer.get_summary_metrics()

    metrics_data["tracer_metrics"] = tracer_summary
    metrics_data["llm_telemetry"] = {
        "p50_latency_ms": tracer_summary["latency_ms"]["p50"],
        "p95_latency_ms": tracer_summary["latency_ms"]["p95"],
        "p99_latency_ms": tracer_summary["latency_ms"]["p99"],
        "total_tokens": tracer_summary["tokens"]["total_tokens"],
        "total_cost_usd": tracer_summary["financials"]["total_cost_usd"],
    }
    if "status" not in metrics_data:
        metrics_data["status"] = "healthy"

    return metrics_data


@router.get("/version")
def service_version() -> dict:
    registry = get_model_registry()
    return {
        "version": "2.0.0",
        "track": "Razorpay AI Buildathon 2026 — Track 02 (AI Risk Manager)",
        "active_model": registry.data.get("active_model", "stacked_ensemble"),
    }
