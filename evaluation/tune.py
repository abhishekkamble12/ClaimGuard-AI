"""
ProofPilot — Hyperparameter Tuning Module
------------------------------------------
Searches hyperparameter space for the ensemble base models to maximize Expected Value (EV)
and precision on the validation split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"


def run_tuning(dataset_path: Path | str = DATASET_PATH) -> dict[str, Any]:
    """Run hyperparameter search on GradientBoostingClassifier to find EV-maximizing configuration."""
    print("=== ProofPilot Hyperparameter Search (Tuning Mode) ===")
    from ml.win_predictor import get_win_predictor

    predictor = get_win_predictor(force_retrain=True)
    if predictor.X_train_arr is None or predictor.y_train_arr is None:
        print("Model training arrays not available. Cannot tune.")
        return {}

    X_train = predictor.X_train_arr
    y_train = predictor.y_train_arr
    X_val = predictor.X_val_arr
    y_val = predictor.y_val_arr

    print(f"Dataset train size: {len(X_train)}, val size: {len(X_val) if X_val is not None else 0}")
    print("Searching GradientBoosting parameter distribution...")

    param_dist = {
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08, 0.10],
        "subsample": [0.8, 0.9, 1.0],
        "min_samples_leaf": [3, 5, 10],
    }

    gbt = GradientBoostingClassifier(random_state=42)
    search = RandomizedSearchCV(
        gbt,
        param_distributions=param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"Best GBT Params (CV ROC-AUC={search.best_score_:.4f}):")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # Evaluate on val set
    best_gbt: Any = search.best_estimator_
    if X_val is not None and y_val is not None:
        from sklearn.metrics import precision_score, recall_score, roc_auc_score

        val_probs = best_gbt.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.35).astype(int)
        val_auc = roc_auc_score(y_val, val_probs)
        val_prec = precision_score(y_val, val_preds, zero_division=0)
        val_rec = recall_score(y_val, val_preds, zero_division=0)
        print(f"Validation Performance: AUC={val_auc:.4f}, Prec@0.35={val_prec:.1%}, Rec@0.35={val_rec:.1%}")

    return search.best_params_


if __name__ == "__main__":
    run_tuning()
