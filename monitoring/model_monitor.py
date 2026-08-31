"""
ProofPilot — ML Model Drift & Observability Monitor
-----------------------------------------------------
Monitors production model inference distributions, calculates Population
Stability Index (PSI) against training baselines, and flags data drift.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)


class PopulationStabilityIndex:
    """
    Computes PSI between baseline distribution (training set)
    and target distribution (production inference batch).
    """

    @staticmethod
    def calculate_psi(
        expected: np.ndarray | list[float],
        actual: np.ndarray | list[float],
        num_buckets: int = 10,
    ) -> float:
        """
        Calculates Population Stability Index (PSI).
        PSI < 0.10: No shift (Green)
        0.10 <= PSI < 0.25: Moderate shift (Yellow)
        PSI >= 0.25: Significant distribution shift (Red / Retrain)
        """
        exp_arr = np.array(expected)
        act_arr = np.array(actual)

        if len(exp_arr) < 5 or len(act_arr) < 5:
            return 0.0

        # Create quantiles from expected distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(exp_arr, percentiles)
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        # Extend outer boundaries to cover all potential ranges
        breakpoints = breakpoints.astype(float)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        exp_counts, _ = np.histogram(exp_arr, bins=breakpoints)
        act_counts, _ = np.histogram(act_arr, bins=breakpoints)

        # Smooth zero counts with Laplace epsilon
        eps = 1e-4
        exp_pct = (exp_counts + eps) / (len(exp_arr) + eps * len(exp_counts))
        act_pct = (act_counts + eps) / (len(act_arr) + eps * len(act_counts))

        psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return round(float(psi_val), 4)


@dataclass
class DriftReport:
    feature_name: str
    psi_score: float
    status: str  # STABLE, MODERATE_DRIFT, CRITICAL_DRIFT
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelDriftMonitor:
    """
    Monitors dispute features and output probability distributions for concept drift.
    """

    def __init__(self, baseline_features: np.ndarray | None = None, feature_names: list[str] | None = None):
        self.baseline_features = baseline_features
        self.feature_names = feature_names or []
        self.recent_inferences: list[list[float]] = []
        self.recent_predictions: list[float] = []

    def set_baseline(self, baseline_features: np.ndarray, feature_names: list[str]) -> None:
        self.baseline_features = baseline_features
        self.feature_names = feature_names

    def log_inference(self, feature_vector: list[float], predicted_prob: float) -> None:
        self.recent_inferences.append(feature_vector)
        self.recent_predictions.append(predicted_prob)
        # Keep rolling buffer of last 500 inferences
        if len(self.recent_inferences) > 500:
            self.recent_inferences.pop(0)
            self.recent_predictions.pop(0)

    def evaluate_drift(self) -> list[DriftReport]:
        if self.baseline_features is None or len(self.recent_inferences) < 5:
            return []

        recent_arr = np.array(self.recent_inferences)
        reports = []

        num_features = min(self.baseline_features.shape[1], recent_arr.shape[1], len(self.feature_names))
        for i in range(num_features):
            fname = self.feature_names[i]
            exp_feat = self.baseline_features[:, i]
            act_feat = recent_arr[:, i]

            psi = PopulationStabilityIndex.calculate_psi(exp_feat, act_feat)

            if psi < 0.10:
                status = "STABLE"
                rec = "Feature distribution is within normal operating variance."
            elif psi < 0.25:
                status = "MODERATE_DRIFT"
                rec = "Mild distribution shift observed. Monitor incoming dispute patterns."
            else:
                status = "CRITICAL_DRIFT"
                rec = "Significant feature drift detected! Retraining recommended."

            reports.append(DriftReport(
                feature_name=fname,
                psi_score=psi,
                status=status,
                recommendation=rec,
            ))

        return reports


_drift_monitor_instance = None


def get_drift_monitor() -> ModelDriftMonitor:
    global _drift_monitor_instance
    if _drift_monitor_instance is None:
        _drift_monitor_instance = ModelDriftMonitor()
    return _drift_monitor_instance
