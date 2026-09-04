"""
ProofPilot — FastAPI Microservice Integration Test Suite
----------------------------------------------------------
Tests endpoint availability, HMAC-SHA256 signature verification, API key authentication,
security headers, request ID tracing, and model drift telemetry.
"""

import json
import os
import unittest
from fastapi.testclient import TestClient

from api.main import app
from razorpay_integration.webhook_simulator import DEFAULT_WEBHOOK_SECRET, sign_webhook_payload


class TestProofPilotAPIIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_dispute_case = {
            "dispute_id": "disp_integration_test_001",
            "merchant_id": "acc_test_9999",
            "merchant_name": "UrbanKart India",
            "network": "CARD",
            "reason_code": "4553",
            "reason_category": "goods_not_received",
            "transaction": {
                "amount": 4999.0,
                "currency": "INR",
                "payment_method": "card",
            },
            "evidence_documents": {
                "delivery_signature": "Signed proof of delivery POD verified.",
                "delivery_tracking": "FedEx tracking #992819 confirmed delivered.",
            },
        }

    def test_health_endpoint_and_headers(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue("ensemble_roc_auc" in data)

        # Check tracing and security headers
        self.assertIn("X-Request-ID", response.headers)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")

    def test_version_endpoint(self):
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["version"], "2.0.0")
        self.assertEqual(data["active_model"], "stacked_ensemble")

    def test_webhook_rejection_on_missing_or_invalid_signature(self):
        payload = {
            "entity": "event",
            "account_id": "acc_100001",
            "event": "dispute.created",
            "contains": ["dispute"],
            "payload": {
                "dispute": {
                    "entity": {
                        "id": "disp_webhook_test_101",
                        "payment_id": "pay_test_202",
                        "amount": 499900,
                        "currency": "INR",
                        "reason_code": "4553",
                        "status": "action_required",
                        "phase": "chargeback",
                    }
                }
            },
        }

        # Missing signature
        res_missing = self.client.post("/webhook/dispute", json=payload)
        self.assertEqual(res_missing.status_code, 401)

        # Invalid signature
        res_invalid = self.client.post(
            "/webhook/dispute",
            json=payload,
            headers={"X-Razorpay-Signature": "invalid_hex_signature_hash_0000000000"},
        )
        self.assertEqual(res_invalid.status_code, 401)

    def test_webhook_success_on_valid_signature(self):
        payload = {
            "entity": "event",
            "account_id": "acc_100001",
            "event": "dispute.created",
            "contains": ["dispute"],
            "payload": {
                "dispute": {
                    "entity": {
                        "id": "disp_webhook_test_102",
                        "payment_id": "pay_test_203",
                        "amount": 499900,
                        "currency": "INR",
                        "reason_code": "4553",
                        "status": "action_required",
                        "phase": "chargeback",
                    }
                }
            },
        }
        secret = os.getenv("WEBHOOK_SECRET", DEFAULT_WEBHOOK_SECRET)
        valid_sig = sign_webhook_payload(payload, secret)

        response = self.client.post(
            "/webhook/dispute",
            json=payload,
            headers={"X-Razorpay-Signature": valid_sig},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["dispute_id"], "disp_webhook_test_102")
        self.assertIn("routing_decision", data)
        self.assertIn("win_probability", data)

    def test_dispute_score_endpoint(self):
        req_body = {
            "dispute": self.valid_dispute_case,
            "use_ground_truth": False,
        }
        api_key = os.getenv("PROOFPILOT_API_KEY", "proofpilot_sec_key_demo_2026")
        response = self.client.post(
            "/disputes/score",
            json=req_body,
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["dispute_id"], "disp_integration_test_001")
        self.assertGreater(data["win_probability"], 0.0)
        self.assertIn("expected_financial_value", data)
        self.assertIn("economic_recommendation", data)

    def test_drift_evaluate_endpoint(self):
        api_key = os.getenv("PROOFPILOT_API_KEY", "proofpilot_sec_key_demo_2026")
        response = self.client.get(
            "/api/v1/drift/evaluate",
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 404:
            response = self.client.get("/drift/evaluate", headers={"X-API-Key": api_key})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "evaluated")
        self.assertIn("drift_reports", data)


if __name__ == "__main__":
    unittest.main()
