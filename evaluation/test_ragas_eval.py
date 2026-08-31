"""
ProofPilot — RAGAS Evaluation Harness Unit Tests
-------------------------------------------------
Run:
    python -m unittest evaluation.test_ragas_eval
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.ragas_eval import (
    evaluate_answer_correctness,
    evaluate_evidence_recall_at_k,
    evaluate_faithfulness,
    evaluate_mrr_gap_recommendations,
    run_ragas_evaluation,
)


class TestRagasEvaluation(unittest.TestCase):

    def setUp(self):
        self.dispute = {
            "dispute_id": "disp_test_101",
            "merchant_name": "Acme Retailers",
            "reason_code": "goods_not_received",
            "transaction": {
                "order_id": "ORD-9988",
                "payment_id": "pay_5544",
                "amount": 2499.00,
                "currency": "INR",
            },
        }
        self.scoring_result = {
            "completeness_score": 0.85,
            "confidence_score": 0.90,
            "routing_decision": "auto_draft_response",
            "evidence_elements": {
                "order_confirmation": {"status": "present"},
                "delivery_tracking": {"status": "present"},
                "delivery_signature": {"status": "missing"},
                "customer_communication": {"status": "present"},
                "transaction_metadata": {"status": "present"},
            },
        }

    def test_faithful_dispute_letter(self):
        valid_letter = """
        Dispute ID: disp_test_101
        Order ID: ORD-9988
        Merchant: Acme Retailers
        Amount: 2499.00
        We provide attached verified order confirmation and verified delivery tracking.
        """
        eval_result = evaluate_faithfulness(self.dispute, self.scoring_result, valid_letter)
        self.assertEqual(eval_result["faithfulness_score"], 1.0)
        self.assertEqual(len(eval_result["hallucinations"]), 0)

    def test_hallucinated_dispute_letter(self):
        # Letter claims verified delivery signature (which is actually missing) and wrong order ID
        hallucinated_letter = """
        Dispute ID: disp_test_101
        Order ID: ORD-FAKE-0000
        Merchant: Acme Retailers
        Amount: 2499.00
        We provide verified delivery signature.
        """
        eval_result = evaluate_faithfulness(self.dispute, self.scoring_result, hallucinated_letter)
        self.assertLess(eval_result["faithfulness_score"], 1.0)
        self.assertTrue(any("delivery signature" in h for h in eval_result["hallucinations"]))

    def test_answer_correctness(self):
        match = evaluate_answer_correctness(
            predicted_route="auto_draft_response",
            expected_route="auto_draft_response",
            predicted_win_prob=0.88,
            actual_outcome="WON",
        )
        self.assertEqual(match["correctness_score"], 1.0)
        self.assertTrue(match["route_match"])
        self.assertTrue(match["win_probability_aligned"])

        mismatch = evaluate_answer_correctness(
            predicted_route="auto_draft_response",
            expected_route="human_review",
        )
        self.assertEqual(mismatch["correctness_score"], 0.0)
        self.assertFalse(mismatch["route_match"])

    def test_evidence_recall_at_5(self):
        required_evidence = {
            "order_confirmation": 0.15,
            "delivery_tracking": 0.25,
            "delivery_signature": 0.30,
            "customer_communication": 0.15,
            "transaction_metadata": 0.15,
        }
        statuses = {
            "order_confirmation": "present",
            "delivery_tracking": "present",
            "delivery_signature": "missing",
            "customer_communication": "present",
            "transaction_metadata": "weak",
        }

        recall_res = evaluate_evidence_recall_at_k(required_evidence, statuses, k=5)
        # 4 out of 5 detected (3 present + 1 weak)
        self.assertEqual(recall_res["detected_count"], 4)
        self.assertEqual(recall_res["recall_at_k"], 0.8)
        self.assertGreater(recall_res["weighted_recall_at_k"], 0.6)

    def test_mrr_gap_recommendations(self):
        required_evidence = {
            "order_confirmation": 0.15,
            "delivery_tracking": 0.25,
            "delivery_signature": 0.30,  # Highest missing
            "customer_communication": 0.15,
        }
        statuses = {
            "order_confirmation": "present",
            "delivery_tracking": "present",
            "delivery_signature": "missing",
            "customer_communication": "missing",
        }

        # Case 1: Highest missing (delivery_signature) ranked #1
        gap_rank_1 = [
            {"evidence_id": "delivery_signature", "potential_score_gain": 0.30},
            {"evidence_id": "customer_communication", "potential_score_gain": 0.15},
        ]
        self.assertEqual(evaluate_mrr_gap_recommendations(gap_rank_1, required_evidence, statuses), 1.0)

        # Case 2: Highest missing ranked #2
        gap_rank_2 = [
            {"evidence_id": "customer_communication", "potential_score_gain": 0.15},
            {"evidence_id": "delivery_signature", "potential_score_gain": 0.30},
        ]
        self.assertEqual(evaluate_mrr_gap_recommendations(gap_rank_2, required_evidence, statuses), 0.5)

    def test_benchmark_dataset_eval_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_out = Path(tmpdir) / "metrics.json"
            res = run_ragas_evaluation(output_path=temp_out, split="test")
            self.assertIn("metrics", res)
            self.assertIn("mean_faithfulness", res["metrics"])
            self.assertIn("mean_answer_correctness", res["metrics"])
            self.assertIn("mean_evidence_recall_at_5", res["metrics"])
            self.assertIn("mean_reciprocal_rank_gap_mrr", res["metrics"])
            self.assertTrue(temp_out.exists())


if __name__ == "__main__":
    unittest.main()
