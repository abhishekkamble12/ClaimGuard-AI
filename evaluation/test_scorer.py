"""
ProofPilot — Scorer, ML, Economics & Security Unit Tests
---------------------------------------------------------
Run:
    python -m unittest discover -s evaluation -p "test_*.py" -v
"""

import unittest
from pathlib import Path

from extraction.extractor import classify_evidence_with_rules
from ml.semantic_matcher import compute_semantic_relevance
from ml.win_predictor import get_win_predictor
from razorpay_integration.webhook_simulator import sign_webhook_payload, verify_webhook_signature
from scoring.audit_log import get_audit_log, log_decision
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

    def test_ml_win_predictor_and_normalization(self):
        predictor = get_win_predictor()
        dummy_dispute = {"transaction": {"amount": 4999, "payment_method": "card"}, "reason_category": "goods_not_received"}
        dummy_result = {"completeness_score": 0.85, "confidence": 0.80, "missing_evidence": [], "weak_evidence": []}
        prob = predictor.predict_win_probability(dummy_dispute, dummy_result)
        self.assertGreater(prob, 0.5)

        ev = predictor.calculate_expected_financial_value(4999, prob)
        self.assertTrue("expected_value_inr" in ev)
        self.assertGreater(ev["expected_value_inr"], 0)

        # Check normalization sum of top 5 displayed features
        if predictor.feature_importances_:
            total_norm = sum(predictor.feature_importances_.values())
            self.assertAlmostEqual(total_norm, 1.0, places=2)

    def test_economic_decision_engine(self):
        predictor = get_win_predictor()
        
        # High win probability case -> Should contest
        contest_rec = predictor.recommend_action(amount_inr=5000.0, win_probability=0.85, dispute_fee=500.0)
        self.assertEqual(contest_rec["action"], "CONTEST")
        self.assertGreater(contest_rec["expected_gain_inr"], 0)

        # Low win probability case -> Should accept loss to save fee
        loss_rec = predictor.recommend_action(amount_inr=500.0, win_probability=0.20, dispute_fee=500.0)
        self.assertEqual(loss_rec["action"], "ACCEPT_LOSS")
        self.assertEqual(loss_rec["fee_saved_if_accepted"], 500.0)

    def test_webhook_hmac_signature_verification(self):
        payload = {"event": "dispute.created", "amount": 250000, "dispute_id": "disp_test_999"}
        secret = "super_secret_buildathon_key"
        
        sig = sign_webhook_payload(payload, secret)
        self.assertTrue(isinstance(sig, str))
        self.assertEqual(len(sig), 64)  # SHA256 hex length

        # Valid verification
        is_valid = verify_webhook_signature(payload, sig, secret)
        self.assertTrue(is_valid)

        # Invalid secret or altered payload
        self.assertFalse(verify_webhook_signature(payload, sig, "wrong_secret"))
        self.assertFalse(verify_webhook_signature({"altered": True}, sig, secret))

    def test_audit_log_trace(self):
        test_id = "disp_audit_unit_test_001"
        dummy_result = {
            "dispute_id": test_id,
            "completeness_score": 0.88,
            "confidence": 0.85,
            "win_probability": 0.91,
            "routing_decision": "auto_draft_response",
            "risk_level": "LOW",
            "missing_evidence": [],
            "weak_evidence": [],
            "evidence_elements": {},
        }
        path = log_decision(test_id, dummy_result)
        self.assertTrue(path.exists())

        loaded = get_audit_log(test_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["dispute_id"], test_id)
        self.assertEqual(loaded["routing_decision"], "auto_draft_response")


if __name__ == "__main__":
    unittest.main()
