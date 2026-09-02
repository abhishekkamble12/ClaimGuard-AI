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

from scoring.scorer import score_dataset

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CONFIG_PATH = ROOT / "config" / "reason_codes"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

# ── cost constants ─────────────────────────────────────────────────────────────
DISPUTE_FEE_INR = 500.0  # Non-refundable fee charged per lost contested dispute


# ── helpers ───────────────────────────────────────────────────────────────────

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
            "mean": round(float(sum(sorted_p) / n_p), 4),
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
            "recall": dec_recall,
            "f1": dec_f1,
            "accuracy": dec_accuracy,
            "optimal_decision_threshold": predictor.ev_optimal_threshold if predictor else None,
            "threshold_selection_method": "EV maximisation on 80/20 train/validation split (PR-curve search)",
            "ml_roc_auc": ml_auc,
            "ml_brier_score": ml_brier,
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
    print(f"  Decision Precision (Contest) : {t2['precision']:.1%}  (of disputes contested, won)")
    print(f"  Decision Recall (Contest)    : {t2['recall']:.1%}  (of winnable disputes, contested)")
    print(f"  Decision F1 Score            : {t2['f1']:.1%}")
    print(f"  Decision Accuracy            : {t2['accuracy']:.1%}")
    print(f"  Route Policy Accuracy        : {t2['routing_distribution']['route_accuracy']:.1%}")
    print(f"  Abstention Rate              : {t2['routing_distribution']['abstention_rate']:.1%} (routed to human review)")
    thresh_val = t2.get("optimal_decision_threshold")
    thresh_str = f"{thresh_val:.4f}" if thresh_val is not None else "n/a"
    print(f"  EV-Optimal Threshold (Val)   : {thresh_str}  (selected on 80/20 train/val split via PR-curve EV search)")
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

    # ── Task 3: Two-Sided Financial Ledger ────────────────────────────────────
    print(f"\n[TASK 3: TWO-SIDED FINANCIAL LEDGER & ABSTENTION] [Split: {split}]")
    print(f"  {sep}")
    print(f"  (+) Gross Disputed Funds Recovered (TP) : +Rs{t3['gross_amount_recovered_inr']:>12,.2f}")
    print(f"  (+) Penalty Fees Correctly Saved   (TN) : +Rs{t3['tn_saved_penalty_fees_inr']:>12,.2f}")
    print(f"  (-) Wasted Penalty Fees Incurred   (FP) : -Rs{t3['fp_wasted_fees_inr']:>12,.2f}  ({cm['FP_contest_and_lost']} lost cases × Rs500)")
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
        order = ["lr", "rf", "gbt", "stacked_ensemble"]
        labels = {"lr": "Logistic Regression", "gbt": "Gradient Boosting",
                  "rf": "Random Forest", "stacked_ensemble": "Stacked Ensemble (Active)"}
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

    report = evaluate_dataset(split=args.split, save_metrics=not args.no_save, force_retrain=args.retrain)
    print_report(report)
