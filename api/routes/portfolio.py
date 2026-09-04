"""
ProofPilot — Portfolio Intelligence & Merchant Risk Router
-----------------------------------------------------------
Provides endpoints for portfolio macro analytics and merchant cohort risk profiles.
"""

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter

from intelligence.merchant_risk_profiler import MerchantRiskProfiler
from intelligence.portfolio_analytics import PortfolioAnalytics

router = APIRouter(prefix="/portfolio", tags=["Portfolio Intelligence"])

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"


def _load_cases() -> list[dict[str, Any]]:
    if DATASET_PATH.exists():
        try:
            ds = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            return ds.get("cases", [])
        except Exception:
            pass
    return []


@router.get("/report")
def get_portfolio_report() -> dict[str, Any]:
    """Generate macro portfolio report including amount at risk, fees saved, and top gaps."""
    cases = _load_cases()
    analytics = PortfolioAnalytics()
    report = analytics.generate_portfolio_report(cases)
    return report.to_dict()


@router.get("/merchants")
def get_merchant_profiles() -> list[dict[str, Any]]:
    """Build merchant risk cohort profiles with VAMP/VCMP risk tiers."""
    cases = _load_cases()
    profiler = MerchantRiskProfiler()
    profiles = profiler.profile_all_merchants(cases)
    return [p.to_dict() for p in profiles]
