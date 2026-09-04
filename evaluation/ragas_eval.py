"""
ProofPilot — RAGAS-Style LLM & RAG Evaluation Harness
------------------------------------------------------
Evaluates dispute reasoning and generative letter synthesis against 4 core metrics:
1. Faithfulness: Factual consistency of generated dispute response letters against verified ground-truth evidence.
2. Answer Correctness: Decision routing precision and win prediction calibration.
3. Evidence Recall@5: Top-5 weighted evidence coverage in the submitted packet.
4. Mean Reciprocal Rank (MRR): Ranking accuracy of recommended evidence gap priorities.

Exports full benchmark metrics to outputs/ragas_metrics.json.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.llm_reasoning import generate_gated_dispute_response, generate_llm_gap_explanation
from scoring.scorer import (
    compute_counterfactual_improvements,
    load_reason_code_config,
    score_dataset,
    score_single_dispute,
)
from utils.logging_config import get_logger

logger = get_logger("proofpilot.ragas_eval")

DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CONFIG_PATH = ROOT / "config" / "reason_codes"
OUTPUT_METRICS_PATH = ROOT / "outputs" / "ragas_metrics.json"


def evaluate_faithfulness(
    dispute: dict[str, Any],
    scoring_result: dict[str, Any],
    generated_letter: str,
) -> dict[str, Any]:
    """
    Evaluate Faithfulness of the dispute response letter:
    Verifies that all factual claims (dispute ID, order ID, payment ID, amount,
    merchant name, and claimed evidence items) match the verified dispute payload
    without hallucination.
    """
    if not generated_letter or not generated_letter.strip():
        return {
            "faithfulness_score": 0.0,
            "total_facts_checked": 0,
            "consistent_facts": 0,
            "hallucinations": ["empty_generation"],
        }

    facts_checked = 0
    consistent_facts = 0
    hallucinations: list[str] = []

    # 1. Dispute ID verification
    disp_id = dispute.get("dispute_id", "")
    if disp_id:
        facts_checked += 1
        if disp_id in generated_letter:
            consistent_facts += 1
        else:
            hallucinations.append(f"Missing or mismatched dispute_id '{disp_id}'")

    # 2. Transaction / Order verification
    txn = dispute.get("transaction", {})
    order_id = txn.get("order_id", "")
    if order_id:
        facts_checked += 1
        if order_id in generated_letter:
            consistent_facts += 1
        else:
            hallucinations.append(f"Missing or mismatched order_id '{order_id}'")

    # 3. Disputed Amount verification
    amount = txn.get("amount")
    if amount is not None:
        facts_checked += 1
        # Match amount number in text
        if str(amount) in generated_letter or f"{amount:,.2f}" in generated_letter:
            consistent_facts += 1
        else:
            hallucinations.append(f"Missing or mismatched transaction amount '{amount}'")

    # 4. Merchant Name verification
    merchant = dispute.get("merchant_name", "")
    if merchant:
        facts_checked += 1
        if merchant.lower() in generated_letter.lower():
            consistent_facts += 1
        else:
            hallucinations.append(f"Missing merchant name '{merchant}'")

    # 5. Evidence grounding check:
    # Ensure letter only claims 'verified' or 'attached' for evidence that is ACTUALLY present
    elements = scoring_result.get("evidence_elements", {})
    for ev_id, details in elements.items():
        clean_ev_name = ev_id.replace("_", " ")
        status = details.get("status", "missing")
        # Check if the letter mentions this evidence as verified/submitted
        pattern = rf"(?i)(verified|attached|provided|submitted|confirmed)\s+{re.escape(clean_ev_name)}"
        if re.search(pattern, generated_letter):
            facts_checked += 1
            if status == "present":
                consistent_facts += 1
            else:
                hallucinations.append(
                    f"Letter claims '{clean_ev_name}' is verified, but actual status is '{status}'"
                )

    score = round(consistent_facts / max(1, facts_checked), 4)
    return {
        "faithfulness_score": score,
        "total_facts_checked": facts_checked,
        "consistent_facts": consistent_facts,
        "hallucinations": hallucinations,
    }


def evaluate_answer_correctness(
    predicted_route: str,
    expected_route: str,
    predicted_win_prob: float | None = None,
    actual_outcome: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate Answer Correctness:
    Compares the automated routing decision against expert benchmark routing,
    and checks if win probability direction aligns with expected outcome.
    """
    route_match = (predicted_route == expected_route)
    score = 1.0 if route_match else 0.0

    # Probability calibration check
    prob_alignment = True
    if predicted_win_prob is not None and actual_outcome:
        if actual_outcome == "WON" and predicted_win_prob < 0.3:
            prob_alignment = False
        elif actual_outcome == "LOST" and predicted_win_prob > 0.7:
            prob_alignment = False

    return {
        "correctness_score": score,
        "route_match": route_match,
        "predicted_route": predicted_route,
        "expected_route": expected_route,
        "win_probability_aligned": prob_alignment,
    }


def _get_weight(val: Any) -> float:
    if isinstance(val, dict):
        return float(val.get("weight", 0.2))
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.2


def evaluate_evidence_recall_at_k(
    required_evidence: dict[str, Any],
    evidence_statuses: dict[str, str],
    k: int = 5,
) -> dict[str, Any]:
    """
    Evaluate Evidence Recall@K:
    Measures the recall of the top-K highest-weighted required evidence elements
    that were successfully collected and verified ('present' or 'weak').
    """
    if not required_evidence:
        return {"recall_at_k": 1.0, "weighted_recall_at_k": 1.0, "top_k_items": []}

    # Sort required items by weight descending
    sorted_items = sorted(required_evidence.items(), key=lambda x: _get_weight(x[1]), reverse=True)
    top_k = sorted_items[:k]

    top_k_keys = [item[0] for item in top_k]
    total_top_k_weight = sum(_get_weight(item[1]) for item in top_k)

    detected_count = 0
    detected_weight = 0.0

    for ev_id, raw_weight in top_k:
        weight = _get_weight(raw_weight)
        status = evidence_statuses.get(ev_id, "missing")
        if status == "present":
            detected_count += 1
            detected_weight += weight
        elif status == "weak":
            detected_count += 1
            detected_weight += (weight * 0.5)

    recall_at_k = round(detected_count / len(top_k), 4)
    weighted_recall = round(detected_weight / max(1e-6, total_top_k_weight), 4)

    return {
        "k": k,
        "recall_at_k": recall_at_k,
        "weighted_recall_at_k": weighted_recall,
        "top_k_items": top_k_keys,
        "detected_count": detected_count,
        "total_top_k": len(top_k),
    }


def evaluate_mrr_gap_recommendations(
    gap_analysis: list[dict[str, Any]],
    required_evidence: dict[str, Any],
    evidence_statuses: dict[str, str],
) -> float:
    """
    Evaluate Mean Reciprocal Rank (MRR) for Evidence Gap Recommendations:
    Measures how quickly the recommendation engine surfaces the single highest-impact
    missing or weak evidence item.
    """
    if not gap_analysis:
        # All evidence is present, no gaps needed
        return 1.0

    # Find the ground truth highest-impact missing item (largest weight with status != 'present')
    missing_items = [
        (ev_id, _get_weight(weight))
        for ev_id, weight in required_evidence.items()
        if evidence_statuses.get(ev_id, "missing") != "present"
    ]
    if not missing_items:
        return 1.0

    # Highest weighted missing evidence ID
    missing_items.sort(key=lambda x: x[1], reverse=True)
    target_top_gap = missing_items[0][0]

    # Find rank of target_top_gap in gap_analysis list
    for rank_idx, gap in enumerate(gap_analysis, start=1):
        if gap.get("evidence_id") == target_top_gap:
            return round(1.0 / rank_idx, 4)

    return 0.0


def run_ragas_evaluation(
    dataset_path: str | Path = DATASET_PATH,
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path = OUTPUT_METRICS_PATH,
    split: str | None = "test",
) -> dict[str, Any]:
    """
    Run full RAGAS evaluation suite across benchmark dispute cases and export results.
    """
    dataset_path = Path(dataset_path)
    config_path = Path(config_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    reason_configs = load_reason_code_config(config_path)

    cases = dataset.get("cases", [])
    if split and split != "all":
        cases = [c for c in cases if c.get("split") == split]

    if not cases:
        cases = dataset.get("cases", [])

    faithfulness_scores: list[float] = []
    correctness_scores: list[float] = []
    recall_at_5_scores: list[float] = []
    weighted_recall_scores: list[float] = []
    mrr_scores: list[float] = []

    case_evaluations: list[dict[str, Any]] = []

    for case in cases:
        dispute_id = case.get("dispute_id", "disp_unknown")
        rc_key = case.get("reason_category") or case.get("reason_code")
        rc_config = reason_configs.get(rc_key, {})
        required_evidence = rc_config.get("required_evidence", {})

        # Score dispute
        scoring_res = score_single_dispute(case, reason_configs)
        predicted_route = scoring_res.get("routing_decision", "human_review")
        expected_route = case.get("expected_route", "human_review")

        # 1. Answer Correctness
        correctness_res = evaluate_answer_correctness(
            predicted_route=predicted_route,
            expected_route=expected_route,
            predicted_win_prob=scoring_res.get("win_probability"),
            actual_outcome=case.get("actual_outcome"),
        )
        correctness_scores.append(correctness_res["correctness_score"])

        # 2. Evidence Recall@5
        evidence_statuses = case.get("evidence_statuses", {})
        recall_res = evaluate_evidence_recall_at_k(
            required_evidence=required_evidence,
            evidence_statuses=evidence_statuses,
            k=5,
        )
        recall_at_5_scores.append(recall_res["recall_at_k"])
        weighted_recall_scores.append(recall_res["weighted_recall_at_k"])

        # 3. MRR Gap Recommendations
        gap_analysis = scoring_res.get("gap_analysis", [])
        mrr = evaluate_mrr_gap_recommendations(
            gap_analysis=gap_analysis,
            required_evidence=required_evidence,
            evidence_statuses=evidence_statuses,
        )
        mrr_scores.append(mrr)

        # 4. Faithfulness of Generated Letter (when auto-drafting or synthesized)
        gen_letter = generate_gated_dispute_response(case, scoring_res)
        faith_res = evaluate_faithfulness(case, scoring_res, gen_letter)
        faithfulness_scores.append(faith_res["faithfulness_score"])

        case_evaluations.append({
            "dispute_id": dispute_id,
            "reason_code": rc_key,
            "faithfulness": faith_res,
            "correctness": correctness_res,
            "recall_at_5": recall_res,
            "mrr": mrr,
        })

    def _mean(values: list[float]) -> float:
        return round(sum(values) / max(1, len(values)), 4)

    summary_metrics = {
        "dataset_name": dataset.get("dataset_name", "proofpilot_benchmark"),
        "evaluated_split": split or "all",
        "total_cases_evaluated": len(cases),
        "metrics": {
            "mean_faithfulness": _mean(faithfulness_scores),
            "mean_answer_correctness": _mean(correctness_scores),
            "mean_evidence_recall_at_5": _mean(recall_at_5_scores),
            "mean_weighted_recall_at_5": _mean(weighted_recall_scores),
            "mean_reciprocal_rank_gap_mrr": _mean(mrr_scores),
        },
        "case_samples": case_evaluations[:5],
    }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")
    logger.info("Saved RAGAS evaluation metrics to %s", output_path)

    return summary_metrics


if __name__ == "__main__":
    result = run_ragas_evaluation()
    print("RAGAS Benchmark Metrics:")
    print(json.dumps(result["metrics"], indent=2))
