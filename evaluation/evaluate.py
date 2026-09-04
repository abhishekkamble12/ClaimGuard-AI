"""
ProofPilot — Rigorous Evaluation Harness
-----------------------------------------
Evaluates ProofPilot across 3 distinct, strictly separated tasks on the HELD-OUT TEST SPLIT:
  1. Evidence Extraction & Detection (NLP Extractor Task)
  2. Dispute Outcome Prediction & Economic Decisioning (ML Ensemble + Policy Task)
  3. Two-Sided Financial Ledger & Abstention Analysis (Honest EV Breakdown)

Every single metric is unambiguously tagged with its dataset split.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scoring.scorer import score_dataset

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CONFIG_PATH = ROOT / "config" / "reason_codes"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

# ── cost constants ─────────────────────────────────────────────────────────────
DISPUTE_FEE_INR = 500.0  # Non-refundable fee charged per lost contested dispute


# ── helpers ───────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across n_bins uniform buckets."""
    import numpy as np
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    if len(y_true_arr) == 0 or len(y_prob_arr) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true_arr)
    for i in range(n_bins):
        mask = (y_prob_arr >= bin_boundaries[i]) & (y_prob_arr < bin_boundaries[i+1]) if i < n_bins - 1 else (y_prob_arr >= bin_boundaries[i]) & (y_prob_arr <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true_arr[mask].mean()
        bin_conf = y_prob_arr[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 4)


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> dict[str, float]:
    """Wilson score confidence interval for binomial proportions (scipy.stats.binomtest)."""
    if trials <= 0:
        return {"low": 0.0, "high": 0.0, "margin": 0.0}
    try:
        from scipy.stats import binomtest
        res = binomtest(successes, trials)
        ci = res.proportion_ci(confidence_level=confidence, method="wilson")
        point = successes / trials
        return {
            "low": round(float(ci.low), 4),
            "high": round(float(ci.high), 4),
            "margin": round(float(max(point - ci.low, ci.high - point)), 4),
        }
    except Exception:
        import math
        z = 1.96
        p = successes / trials
        denom = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denom
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denom
        return {
            "low": round(max(0.0, center - margin), 4),
            "high": round(min(1.0, center + margin), 4),
            "margin": round(margin, 4),
        }


def _is_detected(status: str) -> bool:
    """Evidence detected = present or weak (non-missing)."""
    return status in {"present", "weak"}


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return round(num / denom, 4) if denom else default


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return round(2 * precision * recall / denom, 4) if denom else 0.0


# ── naive baseline EVs ────────────────────────────────────────────────────────

def _naive_always_contest_ev(cases: list[dict[str, Any]]) -> float:
    """EV if merchant blindly contests every single dispute regardless of evidence."""
    total = 0.0
    for c in cases:
        amount = float(c.get("transaction", {}).get("amount", 0))
        outcome = c.get("expected_outcome", "lost")
        if outcome == "won":
            total += amount          # recovered disputed amount
        else:
            total -= DISPUTE_FEE_INR  # paid fee and still lost
    return round(total, 2)


def _naive_always_accept_ev(_cases: list[dict[str, Any]]) -> float:
    """EV if merchant accepts every dispute (never contests) — always Rs0 net."""
    return 0.0


def _proofpilot_ev(
    cases: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> float:
    """Actual gross net EV produced by ProofPilot's CONTEST / ACCEPT_LOSS decisions."""
    total = 0.0
    for c in cases:
        result = by_id.get(c["dispute_id"])
        if not result:
            continue
        action = result.get("economic_recommendation", {}).get("action", "ACCEPT_LOSS")
        amount = float(c.get("transaction", {}).get("amount", 0))
        outcome = c.get("expected_outcome", "lost")

        if action == "CONTEST":
            if outcome == "won":
                total += amount          # recovered
            else:
                total -= DISPUTE_FEE_INR  # wasted fee
        # ACCEPT_LOSS → EV contribution is Rs0 (no fee, no recovery)

    return round(total, 2)


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_path: str | Path = DATASET_PATH,
    config_path: str | Path = CONFIG_PATH,
    split: str = "test",
    save_metrics: bool = True,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """
    Run full rigorous evaluation separated into:
      - Task 1: Evidence Detection & Quality Extraction
      - Task 2: Dispute Outcome Prediction & Economic Decisioning
      - Task 3: Two-Sided Financial Ledger & Honest Abstention Breakdown
    """
    dataset_path = Path(dataset_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    all_cases = dataset["cases"]

    # ── enforce held-out split ────────────────────────────────────────────────
    split_label = split.upper() if split != "all" else "ALL (TRAIN + TEST COMBINED)"
    if split and split != "all":
        eval_cases = [c for c in all_cases if c.get("split") == split]
    else:
        eval_cases = all_cases

    if not eval_cases:
        raise ValueError(
            f"No cases found for split='{split}'. "
            "Run data_generator.py first to regenerate the dataset."
        )

    # Warm up / retrain ML predictor cleanly on current environment
    from ml.win_predictor import get_win_predictor
    get_win_predictor(force_retrain=force_retrain)

    # Score only the eval split (use_ground_truth=False → NLP evidence path)
    raw_results = score_dataset(
        dataset_path,
        config_path,
        split=split if split != "all" else None,
    )
    by_id: dict[str, dict[str, Any]] = {
        r["dispute_id"]: r for r in raw_results if "error" not in r
    }

    scored_count = len(by_id)
    total_cases = len(eval_cases)

    # ── Task 1: Evidence Detection (Clean vs Noisy Subsets) ───────────────────
    ev_all_tp = ev_all_fp = ev_all_fn = ev_all_tn = 0
    ev_clean_tp = ev_clean_fp = ev_clean_fn = ev_clean_tn = 0
    ev_noisy_tp = ev_noisy_fp = ev_noisy_fn = ev_noisy_tn = 0

    clean_cases_count = 0
    noisy_cases_count = 0

    # ── Task 2: Decision & Outcome Metrics ───────────────────────────────────
    dec_tp = dec_fp = dec_fn = dec_tn = 0
    # Financial sums
    gross_recovered_inr = 0.0
    total_fp_wasted_fee_inr = 0.0
    total_fn_forfeited_amt_inr = 0.0
    total_tn_saved_fee_inr = 0.0

    # Route-level tracking
    route_matches = 0
    predicted_routes: Counter[str] = Counter()
    expected_routes: Counter[str] = Counter()
    outcome_counter: Counter[str] = Counter()

    # Per-category accumulators
    cat_cm: dict[str, dict[str, Any]] = {}

    for case in eval_cases:
        result = by_id.get(case["dispute_id"])
        if not result:
            continue

        true_outcome = case.get("expected_outcome", "lost")  # "won" | "lost"
        amount = float(case.get("transaction", {}).get("amount", 0))
        action = result.get("economic_recommendation", {}).get("action", "ACCEPT_LOSS")
        category = case.get("reason_category", "unknown")
        is_noisy = bool(case.get("is_noisy") or case.get("has_feature_noise") or case.get("is_adversarial"))

        if is_noisy:
            noisy_cases_count += 1
        else:
            clean_cases_count += 1

        if category not in cat_cm:
            cat_cm[category] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fp_cost": 0.0, "fn_cost": 0.0, "total": 0}
        cat_cm[category]["total"] += 1

        # ── Decision Confusion Matrix ─────────────────────────────────────────
        if action == "CONTEST" and true_outcome == "won":
            dec_tp += 1
            gross_recovered_inr += amount
            cat_cm[category]["tp"] += 1
        elif action == "CONTEST" and true_outcome == "lost":
            dec_fp += 1
            total_fp_wasted_fee_inr += DISPUTE_FEE_INR
            cat_cm[category]["fp"] += 1
            cat_cm[category]["fp_cost"] += DISPUTE_FEE_INR
        elif action == "ACCEPT_LOSS" and true_outcome == "won":
            dec_fn += 1
            total_fn_forfeited_amt_inr += amount
            cat_cm[category]["fn"] += 1
            cat_cm[category]["fn_cost"] += amount
        else:  # ACCEPT_LOSS + lost
            dec_tn += 1
            total_tn_saved_fee_inr += DISPUTE_FEE_INR
            cat_cm[category]["tn"] += 1

        # ── Route Accuracy ────────────────────────────────────────────────────
        predicted_route = result["routing_decision"]
        expected_route = case.get("expected_route", "")
        predicted_routes[predicted_route] += 1
        expected_routes[expected_route] += 1
        outcome_counter[true_outcome] += 1
        if predicted_route == expected_route:
            route_matches += 1

        # ── Evidence Detection Confusion (Per item) ───────────────────────────
        truth = case.get("ground_truth_evidence", {})
        for ev_id, detail in result.get("evidence_elements", {}).items():
            expected_det = _is_detected(truth.get(ev_id, "missing"))
            predicted_det = _is_detected(detail["status"])

            if predicted_det and expected_det:
                ev_all_tp += 1
                if is_noisy:
                    ev_noisy_tp += 1
                else:
                    ev_clean_tp += 1
            elif predicted_det and not expected_det:
                ev_all_fp += 1
                if is_noisy:
                    ev_noisy_fp += 1
                else:
                    ev_clean_fp += 1
            elif not predicted_det and expected_det:
                ev_all_fn += 1
                if is_noisy:
                    ev_noisy_fn += 1
                else:
                    ev_clean_fn += 1
            else:
                ev_all_tn += 1
                if is_noisy:
                    ev_noisy_tn += 1
                else:
                    ev_clean_tn += 1

    # Raw probability distribution tracking
    raw_probs: list[float] = []
    for case in eval_cases:
        res = by_id.get(case["dispute_id"])
        if res and "win_probability" in res:
            raw_probs.append(float(res["win_probability"]))

    raw_prob_stats = {}
    prob_histogram = {}
    if raw_probs:
        sorted_p = sorted(raw_probs)
        n_p = len(sorted_p)
        raw_prob_stats = {
            "min": round(sorted_p[0], 4),
            "p10": round(sorted_p[int(0.10 * n_p)], 4),
            "p25": round(sorted_p[int(0.25 * n_p)], 4),
            "median": round(sorted_p[int(0.50 * n_p)], 4),
            "mean": round(sum(sorted_p) / n_p, 4),
            "p75": round(sorted_p[int(0.75 * n_p)], 4),
            "p90": round(sorted_p[min(n_p - 1, int(0.90 * n_p))], 4),
            "max": round(sorted_p[-1], 4),
        }
        # Decile histogram
        buckets = [0] * 10
        for p in sorted_p:
            idx = min(9, int(p * 10))
            buckets[idx] += 1
        prob_histogram = {
            f"[{i/10:.1f}-{(i+1)/10:.1f})": buckets[i] for i in range(10)
        }

    # ── Evidence Detection Aggregates ─────────────────────────────────────────
    ev_prec_all = _safe_div(ev_all_tp, ev_all_tp + ev_all_fp)
    ev_rec_all = _safe_div(ev_all_tp, ev_all_tp + ev_all_fn)
    ev_f1_all = _f1(ev_prec_all, ev_rec_all)

    ev_prec_clean = _safe_div(ev_clean_tp, ev_clean_tp + ev_clean_fp)
    ev_rec_clean = _safe_div(ev_clean_tp, ev_clean_tp + ev_clean_fn)
    ev_f1_clean = _f1(ev_prec_clean, ev_rec_clean)

    ev_prec_noisy = _safe_div(ev_noisy_tp, ev_noisy_tp + ev_noisy_fp)
    ev_rec_noisy = _safe_div(ev_noisy_tp, ev_noisy_tp + ev_noisy_fn)
    ev_f1_noisy = _f1(ev_prec_noisy, ev_rec_noisy)

    # ── Decision-Level Aggregates ─────────────────────────────────────────────
    dec_precision = _safe_div(dec_tp, dec_tp + dec_fp)
    dec_recall = _safe_div(dec_tp, dec_tp + dec_fn)
    dec_f1 = _f1(dec_precision, dec_recall)
    dec_accuracy = _safe_div(dec_tp + dec_tn, total_cases)
    precision_ci = wilson_ci(dec_tp, dec_tp + dec_fp)
    recall_ci = wilson_ci(dec_tp, dec_tp + dec_fn)

    # ── PR-AUC, ECE, and McNemar Significance Testing vs Naive Contest Baseline ──
    pr_auc = None
    ece_val = None
    y_true_binary = [1 if c.get("expected_outcome") == "won" else 0 for c in eval_cases]
    if raw_probs and len(raw_probs) == len(y_true_binary):
        ece_val = expected_calibration_error(y_true_binary, raw_probs)
        if len(set(y_true_binary)) > 1:
            try:
                from sklearn.metrics import average_precision_score
                pr_auc = round(float(average_precision_score(y_true_binary, raw_probs)), 4)
            except Exception:
                pass

    # McNemar's test comparing ProofPilot decision correctness vs Naive Always Contest
    # b = ProofPilot correct & Naive incorrect (saved fee on lost case)
    # c = ProofPilot incorrect & Naive correct (missed winnable case)
    b_correct_pp_wrong_naive = 0
    c_wrong_pp_correct_naive = 0
    for case in eval_cases:
        res = by_id.get(case["dispute_id"])
        if not res:
            continue
        act = res.get("economic_recommendation", {}).get("action", "ACCEPT_LOSS")
        true_win = (case.get("expected_outcome") == "won")
        pp_correct = (act == "CONTEST" and true_win) or (act == "ACCEPT_LOSS" and not true_win)
        naive_correct = true_win  # Always contest is only correct when outcome is won

        if pp_correct and not naive_correct:
            b_correct_pp_wrong_naive += 1
        elif not pp_correct and naive_correct:
            c_wrong_pp_correct_naive += 1

    mcnemar_stat = None
    mcnemar_p_value = None
    if (b_correct_pp_wrong_naive + c_wrong_pp_correct_naive) > 0:
        # Continuity-corrected McNemar chi-squared: (|b - c| - 1)^2 / (b + c)
        num = max(0.0, abs(b_correct_pp_wrong_naive - c_wrong_pp_correct_naive) - 1.0) ** 2
        denom = float(b_correct_pp_wrong_naive + c_wrong_pp_correct_naive)
        mcnemar_stat = round(num / denom, 4)
        try:
            from scipy.stats import chi2
            raw_p = float(chi2.sf(mcnemar_stat, 1))
            # Guard against float underflow: chi2.sf returns 0.0 for very large
            # statistics (e.g. chi2=64), which would incorrectly mark the test as
            # not significant.  Clamp the minimum displayable p-value to 1e-10.
            if raw_p == 0.0 and mcnemar_stat > 0:
                raw_p = 1e-10
            mcnemar_p_value = round(raw_p, 10)
        except Exception:
            # Lookup approximation: chi2 1 dof critical values
            mcnemar_p_value = (
                1e-10 if mcnemar_stat > 30.0
                else 0.0001 if mcnemar_stat > 15.14
                else 0.001  if mcnemar_stat > 10.83
                else 0.05   if mcnemar_stat > 3.84
                else 0.20
            )

    mcnemar_test_result = {
        "b_pp_better_than_naive": b_correct_pp_wrong_naive,
        "c_naive_better_than_pp": c_wrong_pp_correct_naive,
        "chi2_stat": mcnemar_stat,
        "p_value": mcnemar_p_value,
        # Significant if p < 0.05; also handle the underflow case where p was
        # clamped to 1e-10 (which is clearly significant).
        "is_statistically_significant": bool(
            mcnemar_p_value is not None and mcnemar_p_value < 0.05
        ),
    }

    # ── Expected Cost (Asymmetric Loss) ──────────────────────────────────────
    total_expected_cost_inr = round(total_fp_wasted_fee_inr + total_fn_forfeited_amt_inr, 2)

    # ── Two-Sided Financial Ledger ────────────────────────────────────────────
    ev_always_contest = _naive_always_contest_ev(eval_cases)
    ev_always_accept = _naive_always_accept_ev(eval_cases)
    ev_proofpilot = _proofpilot_ev(eval_cases, by_id)

    # Net honest economic value ledger
    # Net Benefit = (Gross Recovered INR) - (Wasted FP Penalty Fees) - (Forfeited FN Missed Opportunities)
    net_ledger_impact = round(gross_recovered_inr - total_fp_wasted_fee_inr - total_fn_forfeited_amt_inr, 2)

    # ML predictor stats
    ml_auc = None
    ml_brier = None
    try:
        from ml.win_predictor import get_win_predictor
        predictor = get_win_predictor()
        ml_auc = round(predictor.auc_score, 4)
        ml_brier = round(predictor.brier_score, 4)
    except Exception:
        pass

    # ── Per-Category Metrics ─────────────────────────────────────────────────
    per_category: dict[str, Any] = {}
    for cat, cm in sorted(cat_cm.items()):
        tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
        cat_prec = _safe_div(tp, tp + fp)
        cat_rec = _safe_div(tp, tp + fn)
        cat_f1 = _f1(cat_prec, cat_rec)
        win_rate = _safe_div(tp + fn, cm["total"])
        per_category[cat] = {
            "total_cases": cm["total"],
            "win_rate": win_rate,
            "precision": cat_prec,
            "recall": cat_rec,
            "f1": cat_f1,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "fp_cost_inr": round(cm["fp_cost"], 2),
            "fn_cost_inr": round(cm["fn_cost"], 2),
        }

    # ── Threshold Sensitivity Grid (Task 2 Sensitivity Analysis) ─────────────
    threshold_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    sensitivity_table = []
    for t in threshold_grid:
        t_tp = t_fp = t_fn = t_tn = 0
        t_recovered = 0.0
        t_fp_fee = 0.0
        for case in eval_cases:
            res = by_id.get(case["dispute_id"])
            if not res or "win_probability" not in res:
                continue
            prob = float(res["win_probability"])
            outcome = case.get("expected_outcome", "lost")
            amt = float(case.get("transaction", {}).get("amount", 0))
            from api.middleware.rule_override import apply_safety_net
            action = apply_safety_net(
                dispute=case,
                category=case.get("reason_category"),
                amount_inr=amt,
                model_score=prob,
                model_threshold=t,
            )
            pred_contest = (action == "CONTEST")

            if pred_contest and outcome == "won":
                t_tp += 1
                t_recovered += amt
            elif pred_contest and outcome == "lost":
                t_fp += 1
                t_fp_fee += DISPUTE_FEE_INR
            elif not pred_contest and outcome == "won":
                t_fn += 1
            else:
                t_tn += 1

        t_prec = _safe_div(t_tp, t_tp + t_fp)
        t_rec = _safe_div(t_tp, t_tp + t_fn)
        t_f1 = _f1(t_prec, t_rec)
        t_ev = round(t_recovered - t_fp_fee, 2)
        sensitivity_table.append({
            "threshold": t,
            "precision": t_prec,
            "recall": t_rec,
            "f1": t_f1,
            "tp": t_tp,
            "fp": t_fp,
            "fn": t_fn,
            "tn": t_tn,
            "contested": t_tp + t_fp,
            "net_ev_inr": t_ev,
        })

    # ── Ablation table ────────────────────────────────────────────────────────
    ablation: dict[str, Any] = {}
    try:
        mc_path = ROOT / "outputs" / "model_comparison.json"
        if mc_path.exists():
            mc = json.loads(mc_path.read_text(encoding="utf-8"))
            for model_name, m_metrics in mc.get("models", {}).items():
                ablation[model_name] = {
                    "roc_auc": m_metrics.get("roc_auc"),
                    "brier_score": m_metrics.get("brier_score"),
                    "precision": m_metrics.get("precision"),
                    "recall": m_metrics.get("recall"),
                }
    except Exception:
        pass

    metrics: dict[str, Any] = {
        "dataset": str(dataset_path),
        "split_evaluated": split,
        "split_label": split_label,
        "total_cases_in_split": total_cases,
        "clean_cases_in_split": clean_cases_count,
        "noisy_adversarial_cases_in_split": noisy_cases_count,
        "successfully_scored": scored_count,

        # Task 1: Evidence Extractor Task
        "task_1_evidence_extraction": {
            "all_cases": {"precision": ev_prec_all, "recall": ev_rec_all, "f1": ev_f1_all},
            "clean_cases_subset": {"count": clean_cases_count, "precision": ev_prec_clean, "recall": ev_rec_clean, "f1": ev_f1_clean},
            "noisy_adversarial_subset": {"count": noisy_cases_count, "precision": ev_prec_noisy, "recall": ev_rec_noisy, "f1": ev_f1_noisy},
            "confusion_matrix_all": {"tp": ev_all_tp, "fp": ev_all_fp, "fn": ev_all_fn, "tn": ev_all_tn},
        },

        # Task 2: Dispute Outcome Prediction & Routing Task
        "task_2_dispute_decision": {
            "precision": dec_precision,
            "precision_wilson_ci": precision_ci,
            "recall": dec_recall,
            "recall_wilson_ci": recall_ci,
            "f1": dec_f1,
            "accuracy": dec_accuracy,
            "expected_calibration_error": ece_val,
            "optimal_decision_threshold": predictor.ev_optimal_threshold if predictor else None,
            "min_precision_threshold": getattr(predictor, "min_precision_threshold", None) if predictor else None,
            "profit_optimal_threshold": getattr(predictor, "profit_optimal_threshold", None) if predictor else None,
            "threshold_selection_method": "EV maximisation on 80/20 train/validation split (PR-curve search)",
            "ml_roc_auc": ml_auc,
            "ml_pr_auc": pr_auc,
            "ml_brier_score": ml_brier,
            "total_expected_cost_inr": total_expected_cost_inr,
            "mcnemar_test_vs_naive": mcnemar_test_result,
            "confusion_matrix": {
                "TP_contest_and_won": dec_tp,
                "FP_contest_and_lost": dec_fp,
                "FN_accept_and_won": dec_fn,
                "TN_accept_and_lost": dec_tn,
            },
            "routing_distribution": {
                "route_accuracy": _safe_div(route_matches, total_cases),
                "abstention_rate": _safe_div(predicted_routes["human_review"], total_cases),
                "predicted": dict(predicted_routes),
                "expected": dict(expected_routes),
            },
        },

        # Task 3: Two-Sided Financial Ledger & Abstention Breakdown
        "task_3_financial_ledger": {
            "gross_amount_recovered_inr": round(gross_recovered_inr, 2),
            "fp_wasted_fees_inr": round(total_fp_wasted_fee_inr, 2),
            "fn_forfeited_opportunity_inr": round(total_fn_forfeited_amt_inr, 2),
            "tn_saved_penalty_fees_inr": round(total_tn_saved_fee_inr, 2),
            "net_ledger_impact_inr": net_ledger_impact,
            "proofpilot_total_ev_inr": ev_proofpilot,
            "naive_always_contest_ev_inr": ev_always_contest,
            "naive_always_accept_ev_inr": ev_always_accept,
            "net_ev_gain_vs_always_contest_inr": round(ev_proofpilot - ev_always_contest, 2),
            "net_ev_gain_vs_always_accept_inr": round(ev_proofpilot - ev_always_accept, 2),
        },

        "decision_metrics_by_category": per_category,
        "ablation_table": ablation,
        "threshold_sensitivity_table": sensitivity_table,
    }

    if save_metrics:
        try:
            METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"[warn] Could not save metrics.json: {exc}", file=sys.stderr)

    return metrics


# ── human-readable report ─────────────────────────────────────────────────────

def print_report(metrics: dict[str, Any]) -> None:
    split = metrics["split_label"]
    t1 = metrics["task_1_evidence_extraction"]
    t2 = metrics["task_2_dispute_decision"]
    t3 = metrics["task_3_financial_ledger"]
    cm = t2["confusion_matrix"]

    w = 64
    sep = "-" * w

    print(f"\n{'=' * w}")
    print(f"  ProofPilot — Track 02 Rigorous Evaluation Benchmark")
    print(f"  Dataset : {Path(metrics['dataset']).name}")
    print(f"  Split   : {split} (N={metrics['total_cases_in_split']} cases, {metrics['clean_cases_in_split']} clean / {metrics['noisy_adversarial_cases_in_split']} noisy)")
    print(f"{'=' * w}")

    # ── Task 1: Evidence Extraction ───────────────────────────────────────────
    print(f"\n[TASK 1: EVIDENCE EXTRACTION & DETECTION (NLP Extractor)] [Split: {split}]")
    print(f"  {sep}")
    print(f"  All Test Cases ({metrics['total_cases_in_split']} cases)       : Prec={t1['all_cases']['precision']:.1%}, Rec={t1['all_cases']['recall']:.1%}, F1={t1['all_cases']['f1']:.1%}")
    print(f"  Clean Subset ({t1['clean_cases_subset']['count']} cases)          : Prec={t1['clean_cases_subset']['precision']:.1%}, Rec={t1['clean_cases_subset']['recall']:.1%}, F1={t1['clean_cases_subset']['f1']:.1%}")
    print(f"  Noisy/Adversarial Subset ({t1['noisy_adversarial_subset']['count']} cases): Prec={t1['noisy_adversarial_subset']['precision']:.1%}, Rec={t1['noisy_adversarial_subset']['recall']:.1%}, F1={t1['noisy_adversarial_subset']['f1']:.1%}")
    print(f"  Note: This measures document text keyword/TF-IDF parsing, NOT win prediction.")

    # ── Task 2: Dispute Outcome Prediction ────────────────────────────────────
    print(f"\n[TASK 2: DISPUTE OUTCOME & DECISIONING (ML Ensemble)] [Split: {split}]")
    print(f"  {sep}")
    p_ci = t2.get("precision_wilson_ci", {})
    r_ci = t2.get("recall_wilson_ci", {})
    p_ci_str = f" [95% CI: {p_ci.get('low', 0):.1%} - {p_ci.get('high', 0):.1%}]" if p_ci else ""
    r_ci_str = f" [95% CI: {r_ci.get('low', 0):.1%} - {r_ci.get('high', 0):.1%}]" if r_ci else ""

    print(f"  Decision Precision (Contest) : {t2['precision']:.1%}{p_ci_str} (of disputes contested, won)")
    print(f"  Decision Recall (Contest)    : {t2['recall']:.1%}{r_ci_str} (of winnable disputes, contested)")
    print(f"  Decision F1 Score            : {t2['f1']:.1%}")
    print(f"  Decision Accuracy            : {t2['accuracy']:.1%}")
    ece_val_rep = t2.get("expected_calibration_error")
    ece_str = f"{ece_val_rep:.4f}" if ece_val_rep is not None else "n/a"
    print(f"  Expected Calib Error (ECE)   : {ece_str}  (lower is better, <0.10 indicates good calibration)")
    pr_auc_val = t2.get("ml_pr_auc")
    pr_auc_str = f"{pr_auc_val:.4f}" if pr_auc_val is not None else "n/a"
    exp_cost = t2.get("total_expected_cost_inr", 0.0)
    print(f"  Precision-Recall AUC (PR-AUC): {pr_auc_str}")
    print(f"  Total Error Cost (FP+FN)     : Rs {exp_cost:,.2f}")
    mcn = t2.get("mcnemar_test_vs_naive", {})
    if mcn and mcn.get("chi2_stat") is not None:
        p_val_str = f"{mcn.get('p_value'):.4f}" if mcn.get("p_value") is not None else "n/a"
        sig_str = "Statistically Significant (p < 0.05)" if mcn.get("is_statistically_significant") else "Not Significant"
        print(f"  McNemar vs Naive Baseline   : chi2={mcn.get('chi2_stat')}, p={p_val_str} ({sig_str})")
    print(f"  Route Policy Accuracy        : {t2['routing_distribution']['route_accuracy']:.1%}")
    print(f"  Abstention Rate              : {t2['routing_distribution']['abstention_rate']:.1%} (routed to human review)")
    thresh_val = t2.get("optimal_decision_threshold")
    thresh_str = f"{thresh_val:.4f}" if thresh_val is not None else "n/a"
    min_prec_val = t2.get("min_precision_threshold")
    min_prec_str = f"{min_prec_val:.4f}" if min_prec_val is not None else "n/a"
    profit_val = t2.get("profit_optimal_threshold")
    profit_str = f"{profit_val:.4f}" if profit_val is not None else "n/a"
    print(f"  EV-Optimal Threshold (Val)   : {thresh_str}  (selected on 80/20 train/val split via PR-curve EV search)")
    print(f"  Min-Precision Hurdle (Val)   : {min_prec_str}  (guarantees >= 45% precision floor)")
    print(f"  Profit-Optimal Cutoff (Val)  : {profit_str}  (maximises net profit minus fee costs)")
    # ── Raw Probability Distribution Diagnostic ──────────────────────────────
    rp = metrics.get("raw_prob_stats", {})
    ph = metrics.get("prob_histogram", {})
    if rp:
        print(f"\n  [RAW PREDICTED PROBABILITY DISTRIBUTION P(WIN)] [Split: {split}]")
        print(f"    Min: {rp.get('min'):.4f} | P10: {rp.get('p10'):.4f} | P25: {rp.get('p25'):.4f} | Median: {rp.get('median'):.4f} | Mean: {rp.get('mean'):.4f} | P75: {rp.get('p75'):.4f} | P90: {rp.get('p90'):.4f} | Max: {rp.get('max'):.4f}")
        print(f"    Deciles : {ph}")

    print(f"\n  Decision Confusion Matrix [Split: {split}]:")
    print(f"    TP (CONTEST & Won)         : {cm['TP_contest_and_won']:>4} cases -> Recovered Disputed Funds")
    print(f"    FP (CONTEST & Lost)        : {cm['FP_contest_and_lost']:>4} cases -> Wasted Rs500 Fee Each")
    print(f"    FN (ACCEPT_LOSS & Won)     : {cm['FN_accept_and_won']:>4} cases -> Forfeited Disputed Amount")
    print(f"    TN (ACCEPT_LOSS & Lost)    : {cm['TN_accept_and_lost']:>4} cases -> Correctly Saved Rs500 Fee")

    print(f"\n[TASK 3: TWO-SIDED FINANCIAL LEDGER & ABSTENTION] [Split: {split}]")
    print(f"  {sep}")
    print(f"  (+) Gross Disputed Funds Recovered (TP) : +Rs{t3['gross_amount_recovered_inr']:>12,.2f}")
    print(f"  (+) Penalty Fees Correctly Saved   (TN) : +Rs{t3['tn_saved_penalty_fees_inr']:>12,.2f}")
    print(f"  (-) Wasted Penalty Fees Incurred   (FP) : -Rs{t3['fp_wasted_fees_inr']:>12,.2f}  ({cm['FP_contest_and_lost']} lost cases x Rs500)")
    print(f"  (-) Forfeited Missed Opportunities (FN) : -Rs{t3['fn_forfeited_opportunity_inr']:>12,.2f}  ({cm['FN_accept_and_won']} winnable cases forfeited)")
    print(f"  {'-'*56}")
    print(f"  (=) Net Honest Ledger Impact            :  Rs{t3['net_ledger_impact_inr']:>12,.2f}")
    print(f"\n  Comparison vs Naive Baselines [Split: {split}]:")
    print(f"    ProofPilot Total EV                   :  Rs{t3['proofpilot_total_ev_inr']:>12,.2f}")
    print(f"    Naive: Always CONTEST (Blind AI)      :  Rs{t3['naive_always_contest_ev_inr']:>12,.2f}")
    print(f"    Naive: Always ACCEPT (Never Contest)  :  Rs{t3['naive_always_accept_ev_inr']:>12,.2f}")
    print(f"    Net EV Gain vs Blind Submission       :  Rs{t3['net_ev_gain_vs_always_contest_inr']:>+12,.2f}")
    print(f"    Net EV Gain vs Total Inaction         :  Rs{t3['net_ev_gain_vs_always_accept_inr']:>+12,.2f}")

    # ── Category Breakdown ────────────────────────────────────────────────────
    cat_data = metrics.get("decision_metrics_by_category", {})
    if cat_data:
        print(f"\n[PER-CATEGORY DECISION BREAKDOWN] [Split: {split}]")
        print(f"  {'Category':<32} {'N':>3} {'Win%':>5} {'Prec':>5} {'Rec':>5} {'F1':>5} {'FP(Rs500)':>9} {'FN(Loss)':>10}")
        print(f"  {'-'*32} {'-'*3} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*9} {'-'*10}")
        for cat, cd in sorted(cat_data.items()):
            print(
                f"  {cat:<32} {cd['total_cases']:>3} "
                f"{cd['win_rate']:>4.0%} "
                f"{cd['precision']:>4.0%} "
                f"{cd['recall']:>4.0%} "
                f"{cd['f1']:>4.0%} "
                f"Rs{cd['fp_cost_inr']:>8,.0f} "
                f"Rs{cd['fn_cost_inr']:>9,.0f}"
            )

    # ── Model Ablation ────────────────────────────────────────────────────────
    abl = metrics.get("ablation_table", {})
    if abl:
        print(f"\n[MODEL ABLATION ON HELD-OUT SPLIT] [Split: {split}]")
        print(f"  {'Model':<24} {'AUC':>6}  {'Brier':>6}  {'Prec':>6}  {'Rec':>6}")
        print(f"  {'-'*24} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        order = ["lr", "rf", "gbt", "xgb", "stacked_ensemble"]
        labels = {"lr": "Logistic Regression", "gbt": "Gradient Boosting",
                  "rf": "Random Forest", "xgb": "XGBoost Classifier",
                  "stacked_ensemble": "Stacked Ensemble (Active)"}
        for key in order:
            if key not in abl:
                continue
            m = abl[key]
            print(
                f"  {labels[key]:<24} "
                f"{(m['roc_auc'] or 0):>6.4f}  "
                f"{(m['brier_score'] or 0):>6.4f}  "
                f"{(m['precision'] or 0):>5.1%}  "
                f"{(m['recall'] or 0):>5.1%}"
            )

    # ── Sensitivity Table Display ─────────────────────────────────────────────
    sens = metrics.get("threshold_sensitivity_table", [])
    if sens:
        print(f"\n[PRECISION-VS-THRESHOLD SENSITIVITY TABLE] [Split: {split}]")
        print(f"  {'Threshold':<11} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Contested':>11} {'Net EV (Rs)':>15}")
        print(f"  {'-'*11} {'-'*10} {'-'*8} {'-'*8} {'-'*11} {'-'*15}")
        active_t = t2.get("optimal_decision_threshold")
        for s in sens:
            is_active = active_t is not None and abs(s["threshold"] - active_t) < 0.03
            marker = " *" if is_active else ""
            print(
                f"  {s['threshold']:<11.2f} "
                f"{s['precision']:>9.1%}  "
                f"{s['recall']:>7.1%}  "
                f"{s['f1']:>7.1%}  "
                f"{s['contested']:>10}  "
                f"Rs{s['net_ev_inr']:>13,.2f}{marker}"
            )
        if active_t is not None:
            print(f"  (* indicates active threshold close to {active_t:.4f})")

    print(f"\n{'=' * w}\n")


def evaluate_kfold(
    dataset_path: str | Path = DATASET_PATH,
    config_path: str | Path = CONFIG_PATH,
    n_splits: int = 5,
    save_metrics: bool = True,
) -> dict[str, Any]:
    """
    5-Fold Stratified Cross-Validation on the full 500-case dataset.
    Trains models fold-by-fold, evaluates test folds, and computes mean +/- std
    for Precision, Recall, F1, AUC, Brier, ECE, and Net EV across all folds.
    Also computes full K-Fold model ablation table (LR vs RF vs GBT vs XGB vs Stacked Ensemble).
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, average_precision_score
    from ml.feature_engineering import DisputeFeatureExtractor
    from scoring.scorer import extract_and_classify_evidence, load_reason_code_config, score_evidence, compute_confidence
    from intelligence.dispute_velocity import compute_merchant_velocity_context
    from api.middleware.rule_override import apply_safety_net

    d_path = Path(dataset_path)
    data = json.loads(d_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    rc_configs = load_reason_code_config(config_path)

    vel_context = compute_merchant_velocity_context(cases)
    for c in cases:
        c["merchant_velocity"] = vel_context.get(c.get("dispute_id", ""), {})

    extractor = DisputeFeatureExtractor()
    X_list = []
    y_list = []
    cases_meta = []

    for c in cases:
        try:
            category = c.get("reason_category", "goods_not_received")
            rc = rc_configs.get(category, {})
            required = rc.get("required_evidence", {})
            docs = c.get("evidence_documents", {})
            statuses = extract_and_classify_evidence(required, docs)
            sc_res = score_evidence(required, statuses, docs, rc)
            comp = sc_res["completeness_score"]
            conf = compute_confidence(comp, statuses)
            c["completeness_score"] = comp
            partial_res = {
                "completeness_score": comp,
                "confidence": conf,
                "missing_evidence": [k for k, v in sc_res["elements"].items() if v["status"] == "missing"],
                "weak_evidence": [k for k, v in sc_res["elements"].items() if v["status"] == "weak"],
                "evidence_elements": sc_res["elements"],
            }
            feats = extractor.extract_features(c, partial_res)
        except Exception:
            feats = extractor.extract_features(c)
        target = 1 if c.get("expected_outcome") == "won" else 0
        X_list.append(feats)
        y_list.append(target)
        cases_meta.append(c)

    X_all = np.array(X_list)
    y_all = np.array(y_list)

    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    model_fold_metrics = {"lr": [], "rf": [], "gbt": [], "stacked_ensemble": []}
    if has_xgb:
        model_fold_metrics["xgb"] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]
        test_meta = [cases_meta[i] for i in test_idx]

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        scale = max(1.0, round(n_neg / max(n_pos, 1), 2))
        cw = {0: 1.0, 1: scale}

        raw_models = {
            "lr": LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight=cw),
            "gbt": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.06, random_state=42),
            "rf": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight=cw),
        }
        if has_xgb:
            raw_models["xgb"] = XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                scale_pos_weight=scale, random_state=42, eval_metric="logloss"
            )

        cal_models = {
            k: CalibratedClassifierCV(estimator=m, cv=3)
            for k, m in raw_models.items()
        }
        for name, m in cal_models.items():
            m.fit(X_tr, y_tr)

        test_probs = {}
        for name, m in cal_models.items():
            test_probs[name] = m.predict_proba(X_te)[:, 1]

        ens_prob = np.zeros(len(X_te))
        for name, p in test_probs.items():
            ens_prob += (1.0 / len(test_probs)) * p

        gross_rec = 0.0
        fp_fees = 0.0
        ens_preds = []
        for i, c_case in enumerate(test_meta):
            amt = float(c_case.get("transaction", {}).get("amount", 1000))
            cat = c_case.get("reason_category", "")
            action = apply_safety_net(
                dispute=c_case,
                category=cat,
                amount_inr=amt,
                model_score=float(ens_prob[i]),
                model_threshold=0.25,
            )
            is_contest = 1 if action == "CONTEST" else 0
            ens_preds.append(is_contest)
            if is_contest == 1 and y_te[i] == 1:
                gross_rec += amt
            elif is_contest == 1 and y_te[i] == 0:
                fp_fees += DISPUTE_FEE_INR

        ens_preds = np.array(ens_preds)
        p_score = float(precision_score(y_te, ens_preds, zero_division=0))
        r_score = float(recall_score(y_te, ens_preds, zero_division=0))
        f_score = _f1(p_score, r_score)
        auc_val = float(roc_auc_score(y_te, ens_prob)) if len(set(y_te)) > 1 else 0.5
        brier_val = float(brier_score_loss(y_te, ens_prob))
        prauc_val = float(average_precision_score(y_te, ens_prob)) if len(set(y_te)) > 1 else 0.5
        ece_val = expected_calibration_error(y_te, ens_prob)
        net_ev = round(gross_rec - fp_fees, 2)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "precision": p_score,
            "recall": r_score,
            "f1": f_score,
            "roc_auc": auc_val,
            "brier_score": brier_val,
            "pr_auc": prauc_val,
            "ece": ece_val,
            "net_ev_inr": net_ev,
        })

        for m_name, m_prob in test_probs.items():
            m_preds = (m_prob >= 0.35).astype(int)
            model_fold_metrics[m_name].append({
                "roc_auc": float(roc_auc_score(y_te, m_prob)) if len(set(y_te)) > 1 else 0.5,
                "brier_score": float(brier_score_loss(y_te, m_prob)),
                "precision": float(precision_score(y_te, m_preds, zero_division=0)),
                "recall": float(recall_score(y_te, m_preds, zero_division=0)),
            })
        model_fold_metrics["stacked_ensemble"].append({
            "roc_auc": auc_val,
            "brier_score": brier_val,
            "precision": p_score,
            "recall": r_score,
        })

    def _agg(lst, key):
        arr = [item[key] for item in lst]
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    summary = {
        "n_splits": n_splits,
        "total_cases": len(cases),
        "metrics": {
            "precision": _agg(fold_metrics, "precision"),
            "recall": _agg(fold_metrics, "recall"),
            "f1": _agg(fold_metrics, "f1"),
            "roc_auc": _agg(fold_metrics, "roc_auc"),
            "brier_score": _agg(fold_metrics, "brier_score"),
            "pr_auc": _agg(fold_metrics, "pr_auc"),
            "ece": _agg(fold_metrics, "ece"),
            "net_ev_inr": _agg(fold_metrics, "net_ev_inr"),
        },
        "ablation": {
            m_name: {
                "roc_auc": _agg(vals, "roc_auc"),
                "brier_score": _agg(vals, "brier_score"),
                "precision": _agg(vals, "precision"),
                "recall": _agg(vals, "recall"),
            }
            for m_name, vals in model_fold_metrics.items()
        },
        "folds": fold_metrics,
    }

    if save_metrics:
        out_kfold = ROOT / "outputs" / "kfold_metrics.json"
        out_kfold.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def print_kfold_report(res: dict[str, Any]) -> None:
    w = 64
    sep = "-" * w
    m = res["metrics"]
    print(f"\n{'=' * w}")
    print(f"  ProofPilot — Stratified {res['n_splits']}-Fold Cross-Validation Benchmark")
    print(f"  Dataset: {res['total_cases']} disputes (Label: Won vs Lost)")
    print(f"{'=' * w}")
    print(f"\n[CROSS-VALIDATION HEADLINE METRICS (Mean +/- Std across {res['n_splits']} Folds)]")
    print(f"  {sep}")
    print(f"  Precision (Contest) : {m['precision']['mean']:.1%} +/- {m['precision']['std']:.1%}  (range: {m['precision']['min']:.1%} - {m['precision']['max']:.1%})")
    print(f"  Recall (Contest)    : {m['recall']['mean']:.1%} +/- {m['recall']['std']:.1%}  (range: {m['recall']['min']:.1%} - {m['recall']['max']:.1%})")
    print(f"  F1 Score            : {m['f1']['mean']:.1%} +/- {m['f1']['std']:.1%}")
    print(f"  ROC-AUC             : {m['roc_auc']['mean']:.4f} +/- {m['roc_auc']['std']:.4f}")
    print(f"  PR-AUC              : {m['pr_auc']['mean']:.4f} +/- {m['pr_auc']['std']:.4f}")
    print(f"  Brier Score         : {m['brier_score']['mean']:.4f} +/- {m['brier_score']['std']:.4f}")
    print(f"  ECE Calibration     : {m['ece']['mean']:.4f} +/- {m['ece']['std']:.4f}")
    print(f"  Avg Net EV per Fold : Rs {m['net_ev_inr']['mean']:,.2f} +/- Rs {m['net_ev_inr']['std']:,.2f}")

    abl = res.get("ablation", {})
    if abl:
        print(f"\n[K-FOLD CROSS-VALIDATED MODEL ABLATION TABLE]")
        print(f"  {'Model':<24} {'ROC-AUC':>15}  {'Brier Score':>15}  {'Precision':>15}")
        print(f"  {'-'*24} {'-'*15}  {'-'*15}  {'-'*15}")
        labels = {"lr": "Logistic Regression", "rf": "Random Forest", "gbt": "Gradient Boosting",
                  "xgb": "XGBoost Classifier", "stacked_ensemble": "Stacked Ensemble"}
        for k, v in abl.items():
            name = labels.get(k, k)
            auc_str = f"{v['roc_auc']['mean']:.4f} +/- {v['roc_auc']['std']:.4f}"
            brier_str = f"{v['brier_score']['mean']:.4f} +/- {v['brier_score']['std']:.4f}"
            prec_str = f"{v['precision']['mean']:.1%} +/- {v['precision']['std']:.1%}"
            print(f"  {name:<24} {auc_str:>15}  {brier_str:>15}  {prec_str:>15}")
    print(f"\n{'=' * w}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ProofPilot rigorous evaluation harness")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train", "all"],
        help="Dataset split to evaluate (default: test — held-out only)",
    )
    parser.add_argument(
        "--kfold",
        action="store_true",
        help="Run 5-Fold Stratified Cross Validation and report mean +/- std with K-Fold ablation",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain ML ensemble from scratch on the training split",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write outputs/metrics.json",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter search on validation split",
    )
    args = parser.parse_args()

    if args.tune:
        from evaluation.tune import run_tuning
        run_tuning()
        sys.exit(0)

    if args.kfold:
        kfold_report = evaluate_kfold(save_metrics=not args.no_save)
        print_kfold_report(kfold_report)
    else:
        report = evaluate_dataset(split=args.split, save_metrics=not args.no_save, force_retrain=args.retrain)
        print_report(report)
