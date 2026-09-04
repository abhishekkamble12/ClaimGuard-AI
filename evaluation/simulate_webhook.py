#!/usr/bin/env python3
"""
ProofPilot — Live Razorpay Webhook Simulation Script
------------------------------------------------------
Fires a POST /webhook/dispute request to the running FastAPI service,
demonstrating real-time HMAC-SHA256 signature verification and end-to-end
dispute scoring through the full pipeline.

Usage (run with FastAPI service on port 8000):
    python evaluation/simulate_webhook.py
    python evaluation/simulate_webhook.py --dispute-id disp_1000000000001
    python evaluation/simulate_webhook.py --event dispute.action_required
    python evaluation/simulate_webhook.py --all  # fire all three demo scenarios

Demo scenarios from DEMO.md:
    Scenario A (Blocked → Unlocked): disp_1000000000001  goods_not_received  Rs 2,499
    Scenario B (Doomed Case):        disp_1000000000036  unauthorized_fraud   Rs 1,200
    Scenario C (Safety Net):         disp_1000000000004  unauthorized_fraud   Rs42,000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_case(dispute_id: str) -> dict:
    """Load a specific case from the synthetic dataset."""
    dataset_path = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at {dataset_path}")
        print("        Run: python data_generator.py --cases 500 --seed 42")
        sys.exit(1)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])

    if dispute_id == "first":
        return cases[0]

    for case in cases:
        if case.get("dispute_id") == dispute_id or case.get("legacy_dispute_id") == dispute_id:
            return case

    # Fallback: return first case and warn
    print(f"[WARN] Dispute ID '{dispute_id}' not found — using first case instead.")
    return cases[0]


def simulate_and_fire(
    dispute_id: str = "disp_1000000000001",
    event: str = "dispute.created",
    api_url: str = "http://localhost:8000",
    webhook_secret: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Build a Razorpay-style webhook payload, sign it with HMAC-SHA256,
    and POST it to the FastAPI /webhook/dispute endpoint.
    """
    try:
        import requests
    except ImportError:
        print("[ERROR] 'requests' package not installed. Run: pip install requests")
        sys.exit(1)

    from razorpay_integration.webhook_simulator import simulate_razorpay_dispute_event

    case = _load_case(dispute_id)

    # Secret resolution order: CLI --secret → WEBHOOK_SECRET env var → .env file value
    secret = webhook_secret or os.getenv("WEBHOOK_SECRET")
    if not secret:
        # Try loading from .env directly as a last resort
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("WEBHOOK_SECRET="):
                    secret = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not secret:
        secret = "test_webhook_secret_proofpilot"
        if verbose:
            print(f"  [WARN] WEBHOOK_SECRET not set — using demo secret. "
                  f"Set WEBHOOK_SECRET in .env to match the server.")

    webhook_data = simulate_razorpay_dispute_event(case, event, secret=secret)

    payload = webhook_data["event_payload"]
    signature = webhook_data["x_razorpay_signature"]
    # CRITICAL: must use same canonical encoding as webhook_simulator.sign_webhook_payload()
    # which uses separators=(",",":") AND sort_keys=True.
    # The server re-serializes with sort_keys=True for HMAC verification, so the
    # bytes sent over the wire must be produced with the same settings.
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
    }

    if verbose:
        print("=" * 68)
        print(f"  ProofPilot — Razorpay Webhook Simulation")
        print("=" * 68)
        print(f"  Event        : {event}")
        print(f"  Dispute ID   : {case.get('dispute_id')}")
        print(f"  Merchant     : {case.get('merchant_name')}")
        print(f"  Category     : {case.get('reason_category')}")
        network = case.get('network', 'CARD')
        amount = case.get('transaction', {}).get('amount', 0)
        print(f"  Network      : {network}  |  Amount: Rs {amount:,}")
        print(f"  HMAC-SHA256  : {signature[:24]}...{signature[-8:]}")
        print(f"  Endpoint     : POST {api_url}/webhook/dispute")
        print("-" * 68)

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{api_url}/webhook/dispute",
            data=payload_bytes,
            headers=headers,
            timeout=15,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        if verbose:
            status_icon = "✅" if resp.status_code == 200 else "❌"
            print(f"  Response     : {status_icon} HTTP {resp.status_code}  ({latency_ms}ms)")

        if resp.status_code == 200:
            result = resp.json()
            if verbose:
                routing = result.get("routing_decision", "N/A")
                readiness = result.get("completeness_pct", "N/A")
                p_win = result.get("win_probability_pct", "N/A")
                action = result.get("economic_recommendation", {}).get("action", "N/A")
                ev = result.get("expected_financial_value", {}).get("expected_value_inr", 0)
                hmac_ok = result.get("webhook_verified", True)

                print(f"  HMAC Verified: {'✅ Yes' if hmac_ok else '❌ No'}")
                print(f"  Readiness    : {readiness}")
                print(f"  P(Win)       : {p_win}")
                print(f"  EV           : Rs {ev:,.2f}")
                print(f"  Decision     : {action}")
                print(f"  Route        : {routing}")
                print("=" * 68)
            return result
        else:
            if verbose:
                print(f"  Error body   : {resp.text[:300]}")
                print("=" * 68)
            return {"error": resp.text, "status_code": resp.status_code}

    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to {api_url}")
        print("        Start the FastAPI service first:")
        print("        python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)


def run_demo_scenarios(api_url: str, secret: str | None) -> None:
    """Fire all three canonical demo scenarios from DEMO.md in sequence."""
    scenarios = [
        {
            "label": "Scenario A — Blocked Case (goods_not_received, Rs2,499)",
            "dispute_id": "disp_1000000000001",
            "event": "dispute.created",
        },
        {
            "label": "Scenario B — Doomed Case (unauthorized_fraud, Rs1,200)",
            "dispute_id": "disp_1000000000036",
            "event": "dispute.action_required",
        },
        {
            "label": "Scenario C — High-Value Safety Net (unauthorized_fraud, Rs42,000)",
            "dispute_id": "disp_1000000000004",
            "event": "dispute.action_required",
        },
    ]

    for i, s in enumerate(scenarios, 1):
        print(f"\n{'─' * 68}")
        print(f"  DEMO SCENARIO {i}: {s['label']}")
        print(f"{'─' * 68}")
        simulate_and_fire(
            dispute_id=s["dispute_id"],
            event=s["event"],
            api_url=api_url,
            webhook_secret=secret,
            verbose=True,
        )
        if i < len(scenarios):
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProofPilot — Razorpay Webhook Simulation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dispute-id",
        default="disp_1000000000001",
        help="Dispute ID to simulate (default: disp_1000000000001 — Scenario A)",
    )
    parser.add_argument(
        "--event",
        default="dispute.created",
        choices=["dispute.created", "dispute.action_required", "dispute.under_review"],
        help="Razorpay webhook event type (default: dispute.created)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="FastAPI base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--secret",
        default=None,
        help="Webhook HMAC secret (default: WEBHOOK_SECRET env var or 'test_webhook_secret_proofpilot')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fire all three demo scenarios (A, B, C) in sequence",
    )

    args = parser.parse_args()

    if args.all:
        run_demo_scenarios(api_url=args.api_url, secret=args.secret)
    else:
        simulate_and_fire(
            dispute_id=args.dispute_id,
            event=args.event,
            api_url=args.api_url,
            webhook_secret=args.secret,
            verbose=True,
        )


if __name__ == "__main__":
    main()
