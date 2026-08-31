"""
ProofPilot — Lightweight Machine Learning Model Registry
---------------------------------------------------------
Tracks trained model artifacts, hyperparameter configurations, cross-validation
metrics (ROC-AUC, Brier score, Precision, Recall), and active model selection.
Persists model metadata to outputs/model_registry.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "outputs" / "model_registry.json"


class ModelRegistry:
    """
    Manages versioning and benchmark comparisons of dispute win prediction models.
    """

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"models": [], "active_model": "stacked_ensemble", "last_updated": None}

    def _save(self) -> None:
        try:
            self.registry_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def register_model(
        self,
        model_name: str,
        metrics: dict[str, float],
        feature_names: list[str],
        hyperparameters: dict[str, Any] | None = None,
        dataset_size: int = 0,
    ) -> None:
        """Register a newly trained model candidate or update an existing version."""
        entry = {
            "model_name": model_name,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "hyperparameters": hyperparameters or {},
            "dataset_size": dataset_size,
        }

        # Replace existing or append
        existing_idx = next((i for i, m in enumerate(self.data["models"]) if m["model_name"] == model_name), None)
        if existing_idx is not None:
            self.data["models"][existing_idx] = entry
        else:
            self.data["models"].append(entry)

        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def set_active_model(self, model_name: str) -> None:
        """Set the active production model identifier."""
        self.data["active_model"] = model_name
        self._save()

    def get_comparison_table(self) -> pd.DataFrame:
        """Return a formatted DataFrame comparing all registered models."""
        rows = []
        for m in self.data.get("models", []):
            met = m.get("metrics", {})
            rows.append({
                "Model": m.get("model_name"),
                "ROC-AUC": f"{met.get('roc_auc', 0.0):.1%}",
                "Brier Score": f"{met.get('brier_score', 0.0):.4f}",
                "Precision": f"{met.get('precision', 0.0):.1%}",
                "Recall": f"{met.get('recall', 0.0):.1%}",
                "Features": m.get("feature_count", 0),
                "Active": "⭐ Active" if m.get("model_name") == self.data.get("active_model") else "",
            })
        return pd.DataFrame(rows)


_registry_instance = None


def get_model_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
