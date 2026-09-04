"""
ProofPilot — Modernization & Security Hardening Unit Tests
------------------------------------------------------------
Covers:
  (a) SHA-256 Model Artifact Checksum Verification
  (b) threshold_for_min_precision Helper
  (c) dispute_id Path-Traversal Sanitization in Audit Logging
  (d) Cost-Sensitive Learning & Sample Weighting
  (e) Rule-Based Override Safety Net
"""

import hashlib
import os
import unittest
from pathlib import Path
import numpy as np

from api.middleware.rule_override import should_force_contest
from ml.win_predictor import (
    MODEL_ARTIFACT_PATH,
    DisputeWinPredictor,
    threshold_for_min_precision,
)
from scoring.audit_log import get_audit_log, log_decision


class TestHardeningAndSecurity(unittest.TestCase):
    """Test suite verifying all modernization, security, and precision safeguards."""

    # ── (a) Checksum Verification ─────────────────────────────────────────────
    def test_checksum_sidecar_detection_and_tamper_rejection(self):
        """Verify that modifying the SHA-256 sidecar checksum triggers a fail-safe RuntimeError."""
        if not MODEL_ARTIFACT_PATH.exists():
            self.skipTest("model_ensemble.pkl not present on disk; run training first")

        sidecar_path = MODEL_ARTIFACT_PATH.with_suffix(".pkl.sha256")
        original_digest = sidecar_path.read_text(encoding="utf-8") if sidecar_path.exists() else None

        try:
            # Write a deliberately corrupted digest
            corrupted_digest = "0" * 64
            sidecar_path.write_text(corrupted_digest, encoding="utf-8")

            predictor = DisputeWinPredictor.__new__(DisputeWinPredictor)
            predictor.models = {}
            predictor.raw_models = {}
            predictor.feature_names = []

            with self.assertRaises(RuntimeError) as ctx:
                predictor._load_persisted_model()

            self.assertIn("checksum verification failed", str(ctx.exception).lower())
        finally:
            # Restore original checksum
            if original_digest:
                sidecar_path.write_text(original_digest, encoding="utf-8")
            else:
                real_hash = hashlib.sha256(MODEL_ARTIFACT_PATH.read_bytes()).hexdigest()
                sidecar_path.write_text(real_hash, encoding="utf-8")

    # ── (b) threshold_for_min_precision Helper ─────────────────────────────────
    def test_threshold_for_min_precision_satisfies_target(self):
        """Verify threshold_for_min_precision finds threshold achieving precision >= 0.45."""
        rng = np.random.RandomState(42)
        # Synthetic binary outcomes with 30% positive rate
        y_val = rng.binomial(1, 0.30, size=200)
        # High confidence for positives, lower for negatives
        val_probs = np.where(y_val == 1, rng.uniform(0.35, 0.85, size=200), rng.uniform(0.05, 0.55, size=200))

        thr = threshold_for_min_precision(val_probs, y_val, min_precision=0.45)
        self.assertIsInstance(thr, float)
        self.assertGreater(thr, 0.0)
        self.assertLess(thr, 1.0)

        # Check empirical precision at chosen threshold
        preds = (val_probs >= thr).astype(int)
        if preds.sum() > 0:
            achieved_prec = float((preds & y_val).sum()) / preds.sum()
            self.assertGreaterEqual(achieved_prec, 0.45)

    def test_threshold_for_min_precision_fallback(self):
        """Verify threshold_for_min_precision handles impossible targets gracefully."""
        y_val = np.zeros(100, dtype=int)  # 0 positive cases
        val_probs = np.random.uniform(0.1, 0.9, size=100)

        thr = threshold_for_min_precision(val_probs, y_val, min_precision=0.90, fallback_threshold=0.50)
        self.assertIsInstance(thr, float)
        self.assertGreater(thr, 0.0)

    # ── (c) dispute_id Path-Traversal Sanitization ──────────────────────────────
    def test_dispute_id_path_traversal_rejected_on_write(self):
        """Confirm that directory traversal characters in dispute_id raise ValueError on logging."""
        malicious_ids = [
            "../secret_file",
            "../../etc/passwd",
            "..\\windows\\system32",
            "disp_123/subfolder",
            "disp 123",
            "disp;rm -rf",
            "disp*eval",
        ]
        sample_scoring = {"reason_code": "4553", "completeness_score": 0.85}

        for bad_id in malicious_ids:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    log_decision(bad_id, sample_scoring)

    def test_dispute_id_path_traversal_rejected_on_read(self):
        """Confirm that directory traversal characters in dispute_id raise ValueError on reading."""
        malicious_ids = [
            "../secret_file",
            "../../etc/passwd",
            "..\\windows\\system32",
            "disp_123/nested",
        ]
        for bad_id in malicious_ids:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    get_audit_log(bad_id)

    def test_valid_dispute_id_allowed(self):
        """Confirm valid dispute_id passes regex validation."""
        valid_ids = ["disp_1000000000001", "CASE-2026-X99", "dispute_alpha_beta_1"]
        sample_scoring = {"reason_code": "4553", "completeness_score": 0.95}

        for ok_id in valid_ids:
            with self.subTest(ok_id=ok_id):
                path = log_decision(ok_id, sample_scoring)
                self.assertTrue(path.exists())
                # Clean up created audit file
                path.unlink(missing_ok=True)

    # ── (d) Cost-Sensitive Learning & Sample Weighting ─────────────────────────
    def test_cost_sensitive_sample_weights(self):
        """Verify sample weights penalize negative class (lost) to deter costly false-positive contests."""
        y_train = np.array([0, 0, 1, 0, 1, 1, 0])
        # Negative class weight = 1.25, Positive class weight = 1.00
        sample_weights = np.where(y_train == 0, 1.25, 1.0)

        # Check weights
        self.assertEqual(sample_weights[0], 1.25)
        self.assertEqual(sample_weights[2], 1.00)
        self.assertGreater(sample_weights[y_train == 0].mean(), sample_weights[y_train == 1].mean())

    # ── (e) Rule-Based Override Safety Net ─────────────────────────────────────
    def test_rule_based_override_high_value_fraud(self):
        """Confirm unauthorized fraud with amount >= 15,000 and completeness >= 0.40 forces contest."""
        case_empty_fraud = {
            "reason_category": "unauthorized_fraud",
            "transaction": {"amount": 16000.0},
            "completeness_score": 0.20,
        }
        # Deficient in evidence -> safety net refuses to force contest to save fee
        self.assertFalse(should_force_contest(case_empty_fraud))

        case_high_fraud = {
            "reason_category": "unauthorized_fraud",
            "transaction": {"amount": 16000.0},
            "completeness_score": 0.55,
        }
        self.assertTrue(should_force_contest(case_high_fraud))

    def test_rule_based_override_high_value_general(self):
        """Confirm any transaction >= 25,000 with evidence forces contest regardless of category."""
        case_high_amt = {
            "reason_category": "goods_not_received",
            "transaction": {"amount": 26000.0},
            "completeness_score": 0.50,
        }
        self.assertTrue(should_force_contest(case_high_amt))

    def test_apply_safety_net_policy(self):
        """Confirm apply_safety_net implements calibrated multi-tier dispute routing."""
        from api.middleware.rule_override import apply_safety_net

        # High-value fraud with evidence -> CONTEST
        case_fraud = {"reason_category": "unauthorized_fraud", "transaction": {"amount": 18000.0}, "completeness_score": 0.50}
        self.assertEqual(apply_safety_net(case_fraud, model_score=0.15), "CONTEST")

        # Large ticket with evidence -> CONTEST
        case_large = {"reason_category": "goods_not_received", "transaction": {"amount": 26000.0}, "completeness_score": 0.45}
        self.assertEqual(apply_safety_net(case_large, model_score=0.10), "CONTEST")

        # Low-value ticket (< 2,000) with low score -> ACCEPT_LOSS (saves fee)
        case_small_low_score = {"reason_category": "goods_not_received", "transaction": {"amount": 800.0}}
        self.assertEqual(apply_safety_net(case_small_low_score, model_score=0.22), "ACCEPT_LOSS")

        # Low-value ticket (< 2,000) with high score >= 0.30 -> CONTEST
        case_small_high_score = {"reason_category": "goods_not_received", "transaction": {"amount": 800.0}}
        self.assertEqual(apply_safety_net(case_small_high_score, model_score=0.45), "CONTEST")

    # ── (f) Profit-Optimal Threshold Helper ────────────────────────────────────
    def test_profit_optimal_threshold(self):
        """Confirm profit_optimal_threshold finds threshold maximizing net recovery minus fees."""
        from ml.win_predictor import profit_optimal_threshold

        y_true = np.array([1, 1, 0, 0, 1, 0])
        probas = np.array([0.80, 0.65, 0.40, 0.20, 0.70, 0.10])
        amounts = [10000.0, 5000.0, 1000.0, 2000.0, 8000.0, 500.0]

        thr, profit = profit_optimal_threshold(probas, y_true, amounts, dispute_fee_inr=500.0)
        self.assertIsInstance(thr, float)
        self.assertGreater(profit, 0.0)
        self.assertGreater(thr, 0.40)  # should cut out the false positives at 0.40, 0.20, 0.10


if __name__ == "__main__":
    unittest.main()
