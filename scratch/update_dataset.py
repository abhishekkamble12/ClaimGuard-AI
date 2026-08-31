import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_generator import generate_dataset
from ml.win_predictor import get_win_predictor
from evaluation.evaluate import evaluate_dataset

ds = generate_dataset(500, 0.25, 42)
out_path = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(ds, indent=2), encoding="utf-8")
print(f"Dataset generated: {ds['total_cases']} cases -> {out_path}")

# Re-train model ensemble
predictor = get_win_predictor()
predictor._train_ensemble()
print(f"Model re-trained: Ensemble AUC = {predictor.auc_score:.1%}, Brier = {predictor.brier_score:.4f}")

# Re-run evaluation metrics
metrics = evaluate_dataset(out_path, ROOT / "config" / "reason_codes", split="test")
print(f"Evaluation completed: Route Accuracy = {metrics['route_accuracy']:.1%}, Evidence Precision = {metrics['evidence_detection_precision']:.1%}")
