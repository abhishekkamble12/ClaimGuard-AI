"""
ProofPilot — Comprehensive Evaluation & Statistical Significance Benchmark
-------------------------------------------------------------------------
Evaluates ProofPilot against 4 distinct baselines on the held-out test split:
  1. Baseline 1: Rule-Only (contest if completeness > 0.60)
  2. Baseline 2: Always Contest (naive aggressive)
  3. Baseline 3: Never Contest (naive conservative / total inaction)
  4. Baseline 4: Single Gradient Boosting Trees (GBT)
  5. Proposed: Calibrated Stacked Ensemble (LR + RF + GBT + XGB) with EV-optimal threshold

Computes:
  - PR-AUC, ROC-AUC, Brier Score, Accuracy, Precision, Recall, F1
  - Two-sided financial metrics: Gross Recovered, Fees Lost, Net Expected Value (Rs)
  - McNemar's paired test for statistical significance (Proposed vs Single GBT, Rule-Only, Always Contest)
  - Bootstrap Confidence Intervals (95% CI, 1,000 resamples) for PR-AUC and Net EV
  - Saves full structured benchmark to outputs/evaluation_benchmark.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.middleware.rule_override import apply_safety_net
from ml.win_predictor import get_win_predictor
from scoring.scorer import (
    compute_confidence,
    extract_and_classify_evidence,
    load_reason_code_config,
    score_evidence,
)

DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CONFIG_PATH = ROOT / "config" / "reason_codes"
OUTPUT_BENCHMARK_PATH = ROOT / "outputs" / "evaluation_benchmark.json"
DISPUTE_FEE_INR = 500.0


def compute_mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """
    McNemar's test with continuity correction comparing two classifiers on the same test set.
    b = A correct, B incorrect
    c = A incorrect, B correct
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {
            "b_a_better": b,
            "c_b_better": c,
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "is_significant": False,
        }

    # Edwards continuity correction
    chi2_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_val = float(stats.chi2.sf(chi2_stat, 1))

    return {
        "b_proposed_better": b,
        "c_baseline_better": c,
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_val, 6),
        "is_significant": p_val < 0.05,
    }


def compute_bootstrap_ci(
    metric_fn: Any,
    y_true: np.ndarray,
    y_score_or_pred: np.ndarray,
    extra_arg: np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    """
    Empirical bootstrap percentile confidence interval.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    scores = []

    for _ in range(n_resamples):
        indices = rng.randint(0, n, n)
        y_b = y_true[indices]
        s_b = y_score_or_pred[indices]
        if extra_arg is not None:
            e_b = extra_arg[indices]
            score = metric_fn(y_b, s_b, e_b)
        else:
            score = metric_fn(y_b, s_b)
        if not math.isnan(score):
            scores.append(score)

    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    scores = sorted(scores)
    alpha = 1.0 - confidence_level
    low_idx = int(alpha / 2.0 * len(scores))
    high_idx = int((1.0 - alpha / 2.0) * len(scores))
    high_idx = min(high_idx, len(scores) - 1)

    return {
        "mean": round(float(np.mean(scores)), 4),
        "ci_lower": round(float(scores[low_idx]), 4),
        "ci_upper": round(float(scores[high_idx]), 4),
    }


def evaluate_benchmark(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    print("=" * 70)
    print("ProofPilot: Running Full Benchmark & Statistical Significance Suite")
    print("=" * 70)

    # 1. Load dataset
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    test_cases = [c for c in data.get("cases", []) if c.get("split") == "test"]
    n_test = len(test_cases)
    print(f"Loaded {n_test} held-out test cases.")

    # 2. Extract features and load predictor
    predictor = get_win_predictor()
    rc_configs = load_reason_code_config(CONFIG_PATH)

    y_test_list = []
    amounts_list = []
    completeness_list = []
    X_test_list = []

    for c in test_cases:
        y_test_list.append(1 if c.get("expected_outcome") == "won" else 0)
        amt = float(c.get("transaction", {}).get("amount", 0.0))
        amounts_list.append(amt)

        category = c.get("reason_category", "goods_not_received")
        rc = rc_configs.get(category, {})
        required = rc.get("required_evidence", {})
        docs = c.get("evidence_documents", {})
        statuses = extract_and_classify_evidence(docs, required)
        sc_res = score_evidence(required, statuses, docs, rc)
        comp = sc_res["completeness_score"]
        conf = compute_confidence(comp, statuses)
        completeness_list.append(comp)

        partial_res = {
            "completeness_score": comp,
            "confidence": conf,
            "missing_evidence": [k for k, v in sc_res["elements"].items() if v["status"] == "missing"],
            "weak_evidence": [k for k, v in sc_res["elements"].items() if v["status"] == "weak"],
            "evidence_elements": sc_res["elements"],
        }
        feats = predictor._extract_features(c, partial_res)
        X_test_list.append(feats)

    y_test = np.array(y_test_list)
    amounts = np.array(amounts_list)
    completeness = np.array(completeness_list)
    X_test = np.array(X_test_list)

    # 3. Predict probabilities from models
    ensemble_probs = np.zeros(n_test)
    for m_name, model in predictor.models.items():
        weight = predictor.ensemble_weights.get(m_name, 0.25)
        ensemble_probs += weight * model.predict_proba(X_test)[:, 1]

    gbt_model = predictor.models.get("gbt")
    if gbt_model:
        gbt_probs = gbt_model.predict_proba(X_test)[:, 1]
    else:
        gbt_probs = ensemble_probs

    threshold = predictor.ev_optimal_threshold if predictor.ev_optimal_threshold is not None else 0.4497

    # 4. Generate Predictions for all candidates
    # Baseline 1: Rule-Only (contest if completeness > 0.60)
    pred_rule_only = (completeness > 0.60).astype(int)
    prob_rule_only = completeness

    # Baseline 2: Always Contest
    pred_always_contest = np.ones(n_test, dtype=int)
    prob_always_contest = np.ones(n_test)

    # Baseline 3: Never Contest
    pred_never_contest = np.zeros(n_test, dtype=int)
    prob_never_contest = np.zeros(n_test)

    # Baseline 4: Single GBT (standard threshold 0.50)
    pred_gbt = (gbt_probs >= 0.50).astype(int)

    # Proposed: Stacked Ensemble + Rule Safety Net + EV Threshold
    pred_proposed = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        cat = test_cases[i].get("reason_category", "")
        action = apply_safety_net(
            dispute=test_cases[i],
            category=cat,
            amount_inr=amounts[i],
            model_score=float(ensemble_probs[i]),
            model_threshold=threshold,
            completeness_score=completeness[i],
        )
        pred_proposed[i] = 1 if action == "CONTEST" else 0

    # Helper to calculate financial EV
    def calc_ev(y: np.ndarray, pred: np.ndarray, amt: np.ndarray) -> dict[str, float]:
        tp_mask = (pred == 1) & (y == 1)
        fp_mask = (pred == 1) & (y == 0)
        fn_mask = (pred == 0) & (y == 1)
        tn_mask = (pred == 0) & (y == 0)

        recovered = float(amt[tp_mask].sum())
        fp_fees = float(fp_mask.sum() * DISPUTE_FEE_INR)
        fn_forfeit = float(amt[fn_mask].sum())
        tn_saved = float(tn_mask.sum() * DISPUTE_FEE_INR)
        net_ev = round(recovered - fp_fees, 2)
        total_error_cost = round(fp_fees + fn_forfeit, 2)

        return {
            "recovered_inr": round(recovered, 2),
            "fp_fees_wasted_inr": round(fp_fees, 2),
            "fn_forfeit_inr": round(fn_forfeit, 2),
            "tn_saved_fees_inr": round(tn_saved, 2),
            "net_ev_inr": net_ev,
            "total_error_cost_inr": total_error_cost,
            "tp": int(tp_mask.sum()),
            "fp": int(fp_mask.sum()),
            "fn": int(fn_mask.sum()),
            "tn": int(tn_mask.sum()),
        }

    # Compute metrics dict for a method
    def compute_all_metrics(name: str, y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None) -> dict[str, Any]:
        fin = calc_ev(y, pred, amounts)
        acc = float(accuracy_score(y, pred))
        prec = float(precision_score(y, pred, zero_division=0))
        rec = float(recall_score(y, pred, zero_division=0))
        f1 = float(f1_score(y, pred, zero_division=0))

        roc_auc = None
        pr_auc = None
        brier = None
        if probs is not None and len(np.unique(y)) > 1:
            try:
                roc_auc = round(float(roc_auc_score(y, probs)), 4)
                pr_auc = round(float(average_precision_score(y, probs)), 4)
                brier = round(float(brier_score_loss(y, probs)), 4)
            except Exception:
                pass

        return {
            "name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "financial": fin,
        }

    models_data = {
        "baseline_1_rule_only": compute_all_metrics("Rule-Only (Completeness > 0.60)", y_test, pred_rule_only, prob_rule_only),
        "baseline_2_always_contest": compute_all_metrics("Always Contest (Blind)", y_test, pred_always_contest, prob_always_contest),
        "baseline_3_never_contest": compute_all_metrics("Never Contest (Inaction)", y_test, pred_never_contest, prob_never_contest),
        "baseline_4_single_gbt": compute_all_metrics("Single GBT (Standard 0.50 Threshold)", y_test, pred_gbt, gbt_probs),
        "proposed_calibrated_ensemble": compute_all_metrics("Proposed: Calibrated Stacked Ensemble + EV Cutoff", y_test, pred_proposed, ensemble_probs),
    }

    # 5. McNemar Significance Tests
    mcnemar_results = {
        "proposed_vs_single_gbt": compute_mcnemar_test(y_test, pred_proposed, pred_gbt),
        "proposed_vs_rule_only": compute_mcnemar_test(y_test, pred_proposed, pred_rule_only),
        "proposed_vs_always_contest": compute_mcnemar_test(y_test, pred_proposed, pred_always_contest),
    }

    # 6. Bootstrap Confidence Intervals (1,000 resamples)
    def bootstrap_ev(y_b: np.ndarray, pred_b: np.ndarray, amt_b: np.ndarray) -> float:
        tp_b = (pred_b == 1) & (y_b == 1)
        fp_b = (pred_b == 1) & (y_b == 0)
        return float(amt_b[tp_b].sum() - fp_b.sum() * DISPUTE_FEE_INR)

    def bootstrap_prauc(y_b: np.ndarray, prob_b: np.ndarray) -> float:
        if len(np.unique(y_b)) < 2:
            return float("nan")
        return float(average_precision_score(y_b, prob_b))

    def bootstrap_precision(y_b: np.ndarray, pred_b: np.ndarray) -> float:
        return float(precision_score(y_b, pred_b, zero_division=0))

    print("\nCalculating 1,000 bootstrap resamples for 95% Confidence Intervals...")
    ci_ev = compute_bootstrap_ci(bootstrap_ev, y_test, pred_proposed, extra_arg=amounts, n_resamples=1000)
    ci_prauc = compute_bootstrap_ci(bootstrap_prauc, y_test, ensemble_probs, n_resamples=1000)
    ci_prec = compute_bootstrap_ci(bootstrap_precision, y_test, pred_proposed, n_resamples=1000)

    confidence_intervals = {
        "net_expected_value_inr": ci_ev,
        "pr_auc": ci_prauc,
        "decision_precision": ci_prec,
    }

    benchmark_output = {
        "dataset": str(dataset_path),
        "split": "test",
        "sample_size": n_test,
        "base_win_rate": round(float(y_test.mean()), 4),
        "optimal_threshold": threshold,
        "models": models_data,
        "statistical_tests": {
            "mcnemar_tests": mcnemar_results,
        },
        "bootstrap_95_ci": confidence_intervals,
    }

    OUTPUT_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_BENCHMARK_PATH.write_text(json.dumps(benchmark_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Benchmark artifact saved to: {OUTPUT_BENCHMARK_PATH}")

    # 7. Print Formatted Benchmark Table
    print("\n" + "=" * 88)
    print("           BENCHMARK COMPARISON TABLE (HELD-OUT TEST SPLIT, N=400)")
    print("=" * 88)
    print(f"{'Method / Model':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'PR-AUC':>7} {'Brier':>7} {'Net EV (Rs)':>14}")
    print("-" * 88)

    for k, v in models_data.items():
        pr_str = f"{v['pr_auc']:.4f}" if v['pr_auc'] is not None else "  -  "
        brier_str = f"{v['brier_score']:.4f}" if v['brier_score'] is not None else "  -  "
        print(
            f"{v['name']:<35} "
            f"{v['accuracy']:>5.1%} "
            f"{v['precision']:>5.1%} "
            f"{v['recall']:>5.1%} "
            f"{v['f1']:>5.1%} "
            f"{pr_str:>7} "
            f"{brier_str:>7} "
            f"Rs{v['financial']['net_ev_inr']:>12,.2f}"
        )

    print("-" * 88)
    print("\n" + "=" * 88)
    print("           STATISTICAL SIGNIFICANCE (McNEMAR'S PAIRED TEST)")
    print("=" * 88)
    for pair_name, m in mcnemar_results.items():
        clean_pair = pair_name.replace("_", " ").title()
        sig = "STATISTICALLY SIGNIFICANT (p < 0.05)" if m["is_significant"] else "NOT SIGNIFICANT"
        print(f"  * {clean_pair:<38}: Chi2 = {m['chi2_stat']:>7.2f}, p-value = {m['p_value']:>7.5f} -> {sig}")

    print("\n" + "=" * 88)
    print("           BOOTSTRAP 95% CONFIDENCE INTERVALS (1,000 RESAMPLES)")
    print("=" * 88)
    print(f"  * Net Expected Value (INR) : Rs {ci_ev['mean']:,.2f}  [95% CI: Rs {ci_ev['ci_lower']:,.2f} to Rs {ci_ev['ci_upper']:,.2f}]")
    print(f"  * PR-AUC                   : {ci_prauc['mean']:.4f}         [95% CI: {ci_prauc['ci_lower']:.4f} to {ci_prauc['ci_upper']:.4f}]")
    print(f"  * Decision Precision       : {ci_prec['mean']:.1%}          [95% CI: {ci_prec['ci_lower']:.1%} to {ci_prec['ci_upper']:.1%}]")
    print("=" * 88 + "\n")

    return benchmark_output


if __name__ == "__main__":
    evaluate_benchmark()
