"""
ProofPilot — Evaluation Harness
---------------------------------
Evaluates ProofPilot's risk routing, evidence detection precision/recall,
abstention rate, false-positive auto-draft rate, and false-positive financial cost (₹)
on held-out benchmark splits. Exports results to outputs/metrics.json.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring.scorer import score_dataset

DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CONFIG_PATH = ROOT / "config" / "reason_codes"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

DISPUTE_FEE_INR = 500.0  # Non-refundable fee per lost chargeback in India


def _is_detected(status: str) -> bool:
    return status in {"present", "weak"}


def evaluate_dataset(
    dataset_path: str | Path = DATASET_PATH,
    config_path: str | Path = CONFIG_PATH,
    split: str | None = None,
    save_metrics: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if split and split != "all":
        cases = [case for case in cases if case.get("split") == split]

    results = score_dataset(dataset_path, config_path, split=split if split != "all" else None)
    by_id = {result["dispute_id"]: result for result in results if "error" not in result}

    route_matches = 0
    evidence_tp = evidence_fp = evidence_fn = evidence_tn = 0
    false_positive_auto_drafts = 0
    predicted_routes: Counter[str] = Counter()
    expected_routes: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()

    for case in cases:
        result = by_id.get(case["dispute_id"])
        if not result:
            continue
        predicted_route = result["routing_decision"]
        expected_route = case["expected_route"]

        predicted_routes[predicted_route] += 1
        expected_routes[expected_route] += 1
        outcomes[case["expected_outcome"]] += 1

        if predicted_route == expected_route:
            route_matches += 1

        if predicted_route == "auto_draft_response" and case["expected_outcome"] == "lost":
            false_positive_auto_drafts += 1

        truth = case["ground_truth_evidence"]
        for evidence_id, detail in result["evidence_elements"].items():
            expected_detected = _is_detected(truth.get(evidence_id, "missing"))
            predicted_detected = _is_detected(detail["status"])

            if predicted_detected and expected_detected:
                evidence_tp += 1
            elif predicted_detected and not expected_detected:
                evidence_fp += 1
            elif not predicted_detected and expected_detected:
                evidence_fn += 1
            else:
                evidence_tn += 1

    total_cases = len(cases)
    auto_drafts = predicted_routes["auto_draft_response"]
    precision = evidence_tp / (evidence_tp + evidence_fp) if evidence_tp + evidence_fp else 0
    recall = evidence_tp / (evidence_tp + evidence_fn) if evidence_tp + evidence_fn else 0

    # Financial False-Positive Cost Calculation (Track 02 Rubric Requirement)
    avg_disputed_amount = (
        sum(float(c.get("transaction", {}).get("amount", 0)) for c in cases) / max(total_cases, 1)
    )
    fp_cost_per_case = avg_disputed_amount + DISPUTE_FEE_INR
    total_fp_financial_cost = false_positive_auto_drafts * fp_cost_per_case

    metrics = {
        "dataset": str(dataset_path),
        "split": split or "all",
        "total_cases": total_cases,
        "route_accuracy": round(route_matches / total_cases, 4) if total_cases else 0,
        "evidence_detection_precision": round(precision, 4),
        "evidence_detection_recall": round(recall, 4),
        "false_positive_auto_drafts": false_positive_auto_drafts,
        "false_positive_auto_draft_rate": round(false_positive_auto_drafts / auto_drafts, 4) if auto_drafts else 0,
        "false_positive_cost_per_case_inr": round(fp_cost_per_case, 2),
        "total_false_positive_financial_cost_inr": round(total_fp_financial_cost, 2),
        "abstention_rate": round(predicted_routes["human_review"] / total_cases, 4) if total_cases else 0,
        "predicted_routes": dict(predicted_routes),
        "expected_routes": dict(expected_routes),
        "expected_outcomes": dict(outcomes),
        "evidence_confusion": {
            "true_positive": evidence_tp,
            "false_positive": evidence_fp,
            "false_negative": evidence_fn,
            "true_negative": evidence_tn,
        },
    }

    if save_metrics:
        try:
            METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        except Exception:
            pass

    return metrics


def print_report(metrics: dict[str, Any]) -> None:
    print("=== ProofPilot Evaluation Benchmark ===")
    print(f"Dataset                 : {metrics['dataset']}")
    print(f"Split                   : {metrics['split']}")
    print(f"Total Cases Evaluated   : {metrics['total_cases']}")
    print(f"Route Accuracy          : {metrics['route_accuracy']:.0%}")
    print(f"Evidence Precision      : {metrics['evidence_detection_precision']:.0%}")
    print(f"Evidence Recall         : {metrics['evidence_detection_recall']:.0%}")
    print(f"Abstention Rate (Human) : {metrics['abstention_rate']:.0%}")
    print(
        f"False-Positive Auto-Drafts: {metrics['false_positive_auto_drafts']} "
        f"({metrics['false_positive_auto_draft_rate']:.0%})"
    )
    print(f"FP Cost (₹/case)        : ₹{metrics['false_positive_cost_per_case_inr']:,.2f}")
    print(f"Total FP Financial Loss : ₹{metrics['total_false_positive_financial_cost_inr']:,.2f}")
    print(f"Predicted Route Dist.   : {metrics['predicted_routes']}")


if __name__ == "__main__":
    report = evaluate_dataset()
    print_report(report)
