"""
ProofPilot — Scorer, ML & Extraction Unit Tests
------------------------------------------------
Run:
    python -m unittest discover -s eval -p "test_*.py"
"""

import unittest

from extraction.extractor import classify_evidence_with_rules
from ml.semantic_matcher import compute_semantic_relevance
from ml.win_predictor import get_win_predictor
from scoring.scorer import compute_confidence, compute_routing_and_risk, score_evidence


class TestProofPilotScorer(unittest.TestCase):

    def setUp(self):
        self.required_evidence = {
            "order_confirmation": 0.15,
            "delivery_tracking": 0.25,
            "delivery_signature": 0.30,
            "customer_communication": 0.15,
            "transaction_metadata": 0.15,
        }

    def test_full_evidence_completeness(self):
        statuses = {k: "present" for k in self.required_evidence}
        scoring = score_evidence(self.required_evidence, statuses)
        self.assertEqual(scoring["completeness_score"], 1.0)

        confidence = compute_confidence(1.0, statuses)
        self.assertEqual(confidence, 1.0)

        route, risk = compute_routing_and_risk(1.0, confidence)
        self.assertEqual(route, "auto_draft_response")
        self.assertEqual(risk, "LOW")

    def test_missing_evidence(self):
        statuses = {k: "missing" for k in self.required_evidence}
        scoring = score_evidence(self.required_evidence, statuses)
        self.assertEqual(scoring["completeness_score"], 0.0)

        confidence = compute_confidence(0.0, statuses)
        self.assertEqual(confidence, 0.0)

        route, risk = compute_routing_and_risk(0.0, confidence)
        self.assertEqual(route, "human_review")
        self.assertEqual(risk, "HIGH")

    def test_ml_semantic_relevance(self):
        doc = "Official courier tracking ID confirmed delivery to customer address on Aug 12."
        desc = "Tracking record showing the shipment was dispatched and in transit."
        sim = compute_semantic_relevance(doc, desc)
        self.assertGreater(sim, 0.0)

    def test_ml_win_predictor(self):
        predictor = get_win_predictor()
        dummy_dispute = {"transaction": {"amount": 4999, "payment_method": "card"}, "reason_category": "goods_not_received"}
        dummy_result = {"completeness_score": 0.85, "confidence": 0.80, "missing_evidence": [], "weak_evidence": []}
        prob = predictor.predict_win_probability(dummy_dispute, dummy_result)
        self.assertGreater(prob, 0.5)

        ev = predictor.calculate_expected_financial_value(4999, prob)
        self.assertTrue("expected_value_inr" in ev)
        self.assertGreater(ev["expected_value_inr"], 0)


if __name__ == "__main__":
    unittest.main()
