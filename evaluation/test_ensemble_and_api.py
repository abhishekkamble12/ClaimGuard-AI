"""
ProofPilot — Stacked Ensemble, Intelligence & API Unit Tests
--------------------------------------------------------------
Run:
    python -m unittest discover -s evaluation -p "test_*.py" -v
"""

import unittest
from pathlib import Path

from intelligence.merchant_risk_profiler import MerchantRiskProfiler
from intelligence.portfolio_analytics import PortfolioAnalytics
from ml.feature_engineering import DisputeFeatureExtractor, compute_evidence_entropy
from ml.model_registry import get_model_registry
from ml.semantic_matcher import CorpusAwareSemanticMatcher
from ml.win_predictor import get_win_predictor
from monitoring.alerting import AlertManager
from monitoring.model_monitor import PopulationStabilityIndex


class TestProofPilotElevation(unittest.TestCase):

    def setUp(self):
        self.sample_dispute = {
            "dispute_id": "disp_test_1001",
            "merchant_id": "acc_100001",
            "merchant_name": "UrbanKart India",
            "network": "CARD",
            "reason_code": "4553",
            "reason_category": "goods_not_received",
            "transaction": {
                "amount": 4999.0,
                "amount_paise": 499900,
                "currency": "INR",
                "payment_method": "card",
            },
            "evidence_documents": {
                "delivery_signature": "Signed POD verified with customer signature.",
                "delivery_tracking": "Courier tracking active.",
            },
            "ground_truth_evidence": {
                "order_confirmation": "present",
                "delivery_tracking": "present",
                "delivery_signature": "present",
                "customer_communication": "weak",
                "transaction_metadata": "present",
            },
            "missing_evidence": [],
            "weak_evidence": ["customer_communication"],
            "expected_completeness_score": 0.90,
            "expected_confidence": 0.88,
            "expected_outcome": "won",
        }

    def test_feature_engineering_22_dimensions(self):
        extractor = DisputeFeatureExtractor()
        names = extractor.get_feature_names()
        self.assertGreaterEqual(len(names), 22)

        feats = extractor.extract_features(self.sample_dispute)
        self.assertEqual(len(feats), len(names))
        # Completeness score should match
        self.assertEqual(feats[0], 0.90)

    def test_evidence_entropy_calculation(self):
        # All same status -> 0 entropy
        entropy_zero = compute_evidence_entropy({"a": {"status": "present"}, "b": {"status": "present"}})
        self.assertEqual(entropy_zero, 0.0)

        # Mixed statuses -> Positive entropy
        entropy_mixed = compute_evidence_entropy({
            "a": {"status": "present"},
            "b": {"status": "weak"},
            "c": {"status": "missing"},
        })
        self.assertGreater(entropy_mixed, 0.0)

    def test_stacked_ensemble_win_predictor(self):
        predictor = get_win_predictor()
        dummy_result = {
            "completeness_score": 0.90,
            "confidence": 0.88,
            "missing_evidence": [],
            "weak_evidence": ["customer_communication"],
        }
        prob = predictor.predict_win_probability(self.sample_dispute, dummy_result)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertGreater(prob, 0.5)

        # Ensure ROC-AUC and Brier Score are tracked
        self.assertGreater(predictor.auc_score, 0.70)
        self.assertTrue(0.0 <= predictor.brier_score <= 0.35)

    def test_corpus_aware_semantic_matcher(self):
        configs = {
            "goods_not_received": {
                "required_evidence": {
                    "delivery_signature": {"description": "Signed proof of delivery POD with customer signature"}
                }
            }
        }
        matcher = CorpusAwareSemanticMatcher(configs)
        sim = matcher.compute_relevance(
            "Courier confirmed package delivered with digital signature.",
            "Signed proof of delivery POD with customer signature",
        )
        self.assertGreater(sim, 0.0)

    def test_merchant_risk_profiler_and_vamp(self):
        profiler = MerchantRiskProfiler(assumed_monthly_txns=1000)
        profile = profiler.build_profile("UrbanKart India", [self.sample_dispute])

        self.assertEqual(profile.merchant_name, "UrbanKart India")
        self.assertEqual(profile.total_disputes, 1)
        self.assertEqual(profile.vamp_risk_tier, "HEALTHY")
        self.assertGreater(profile.merchant_health_score, 50.0)

    def test_portfolio_analytics(self):
        analytics = PortfolioAnalytics()
        report = analytics.generate_portfolio_report([self.sample_dispute])

        self.assertEqual(report.total_disputes, 1)
        self.assertEqual(report.total_won_disputes, 1)
        self.assertGreater(report.total_amount_at_risk_inr, 0.0)
        self.assertGreater(report.amount_recovered_by_contesting_inr, 0.0)

    def test_population_stability_index_psi(self):
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        similar_dist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        psi_low = PopulationStabilityIndex.calculate_psi(baseline, similar_dist)
        self.assertLess(psi_low, 0.10)  # No shift

        shifted_dist = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
        psi_high = PopulationStabilityIndex.calculate_psi(baseline, shifted_dist)
        self.assertGreater(psi_high, 0.25)  # Critical shift

    def test_alert_manager(self):
        manager = AlertManager()
        # High chargeback ratio > 0.90% should trigger critical alert
        alerts = manager.evaluate_merchant(chargeback_ratio_pct=1.20, win_rate=0.30)
        rule_ids = [a.rule_id for a in alerts]
        self.assertIn("RULE_VAMP_LIMIT", rule_ids)
        self.assertIn("RULE_WIN_RATE_DROP", rule_ids)


if __name__ == "__main__":
    unittest.main()
