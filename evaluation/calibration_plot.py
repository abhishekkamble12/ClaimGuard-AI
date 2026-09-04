"""
ProofPilot — Calibration Reliability Diagram
----------------------------------------------
Plots a reliability diagram (calibration curve) for the stacked ensemble
and each base model (LR, GBT, RF) on the held-out test split.

A well-calibrated model has points close to the diagonal: when it predicts
a 70% win probability, the merchant actually wins ~70% of the time.
This is critical for the ₹ Expected Value calculation — miscalibrated
probabilities produce systematically biased CONTEST/ACCEPT recommendations.

Output
------
  outputs/calibration_curve.png   — reliability diagram (saved automatically)

Usage
-----
  python evaluation/calibration_plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── lazy imports so the script fails fast with a clear message ─────────────────
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend — safe in all envs
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

OUTPUT_PATH = ROOT / "outputs" / "calibration_curve.png"
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures weighted average difference between predicted confidence and observed accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1]) if i < n_bins - 1 else (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 4)


def generate_calibration_plot(save_path: Path = OUTPUT_PATH) -> Path:
    """
    Build and save the reliability diagram.

    Derives X_test / y_test directly from the dataset file using the same
    feature extractor and train/test split used during training — identical
    to what evaluate.py measures, so the diagram is directly comparable to
    the reported precision/recall numbers.
    """
    from ml.win_predictor import get_win_predictor, DisputeFeatureExtractor

    predictor = get_win_predictor()

    if not predictor.is_trained or not predictor.models:
        print("ERROR: Predictor not trained. Run: python data_generator.py && python evaluation/evaluate.py", file=sys.stderr)
        sys.exit(1)

    # ── rebuild test features from dataset (same split used by evaluate.py) ──
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}. Run: python data_generator.py", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    extractor = DisputeFeatureExtractor()

    X_test_list, y_test_list = [], []
    for c in raw["cases"]:
        if c.get("split") != "test":
            continue
        X_test_list.append(extractor.extract_features(c))
        y_test_list.append(1 if c["expected_outcome"] == "won" else 0)

    if len(X_test_list) < 10:
        print(f"ERROR: Only {len(X_test_list)} test cases found. Regenerate with: python data_generator.py", file=sys.stderr)
        sys.exit(1)

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    print(f"Test set: {len(y_test)} cases  |  win rate: {y_test.mean():.1%}")

    n_bins = 8   # fewer bins → more samples per bucket → more stable estimates

    # ── collect (fraction_of_positives, mean_predicted) per model ────────────
    model_curves: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []
    # (label, frac_pos, mean_pred, colour, linestyle)

    model_specs = [
        ("lr",  "Logistic Regression",  "#2196F3", "--"),
        ("gbt", "Gradient Boosting",    "#FF9800", "-."),
        ("rf",  "Random Forest",        "#4CAF50", ":"),
    ]

    for key, label, colour, ls in model_specs:
        model = predictor.models.get(key)
        if model is None:
            continue
        probs = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=n_bins, strategy="uniform")
        model_curves.append((label, frac_pos, mean_pred, colour, ls))

    # Stacked ensemble
    probs_gbt = predictor.models["gbt"].predict_proba(X_test)[:, 1]
    probs_lr  = predictor.models["lr"].predict_proba(X_test)[:, 1]
    probs_rf  = predictor.models["rf"].predict_proba(X_test)[:, 1]
    ens_probs = (
        predictor.ensemble_weights["gbt"] * probs_gbt
        + predictor.ensemble_weights["lr"] * probs_lr
        + predictor.ensemble_weights["rf"] * probs_rf
    )
    ens_frac, ens_mean = calibration_curve(y_test, ens_probs, n_bins=n_bins, strategy="uniform")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "ProofPilot — Calibration Reliability Diagram\n"
        "Stacked Ensemble vs Base Models  |  Test Split  |  Class = 'Won'",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── left: reliability diagram ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration", alpha=0.6)

    for label, frac_pos, mean_pred, colour, ls in model_curves:
        ax.plot(mean_pred, frac_pos, ls, color=colour, lw=1.5, marker="o",
                markersize=4, label=label, alpha=0.75)

    ax.plot(ens_mean, ens_frac, "-", color="#9C27B0", lw=2.5, marker="D",
            markersize=6, label="Stacked Ensemble (active)", zorder=5)

    ax.set_xlabel("Mean predicted win probability", fontsize=10)
    ax.set_ylabel("Fraction of actual wins", fontsize=10)
    ax.set_title("Reliability Diagram", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Brier score & ECE annotation
    from sklearn.metrics import brier_score_loss
    ens_brier = brier_score_loss(y_test, ens_probs)
    ens_ece = expected_calibration_error(y_test, ens_probs, n_bins=n_bins)
    ax.text(0.97, 0.05,
            f"Ensemble Brier = {ens_brier:.4f}\nECE = {ens_ece:.4f}\n(0 = perfect calibration)",
            ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="lavender", alpha=0.7),
            transform=ax.transAxes)

    # ── right: probability histogram ──────────────────────────────────────────
    ax2 = axes[1]
    won_probs  = ens_probs[y_test == 1]
    lost_probs = ens_probs[y_test == 0]

    bins = np.linspace(0, 1, 21)
    ax2.hist(lost_probs, bins=bins, alpha=0.6, color="#F44336", label="Actual: lost",  density=True)
    ax2.hist(won_probs,  bins=bins, alpha=0.6, color="#4CAF50", label="Actual: won",   density=True)
    ax2.axvline(0.50, color="black", lw=1.2, linestyle="--", label="Decision threshold (0.50)")

    ax2.set_xlabel("Predicted win probability (ensemble)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Probability Distribution by True Outcome", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── caption ───────────────────────────────────────────────────────────────
    caption = (
        "Left: Points near the diagonal indicate well-calibrated probabilities. "
        "Right: Separation between 'won' and 'lost' distributions reflects discriminative power.\n"
        f"Test split: {len(y_test)} cases  |  Win rate: {y_test.mean():.1%}  |  "
        f"Ensemble Brier: {ens_brier:.4f}  |  ECE: {ens_ece:.4f}  |  "
        "EV math depends on calibration — this chart is the proof, not just the claim."
    )
    fig.text(0.5, -0.04, caption, ha="center", fontsize=7.5,
             style="italic", color="#555555", wrap=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration plot saved -> {save_path}")
    return save_path


if __name__ == "__main__":
    generate_calibration_plot()
