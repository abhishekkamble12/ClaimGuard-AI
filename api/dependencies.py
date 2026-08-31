"""
ProofPilot — FastAPI Shared Dependency Ingestion
"""

from pathlib import Path
from typing import Any

from intelligence.merchant_risk_profiler import MerchantRiskProfiler
from intelligence.portfolio_analytics import PortfolioAnalytics
from ml.win_predictor import DisputeWinPredictor, get_win_predictor
from scoring.scorer import load_reason_code_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config" / "reason_codes"

_cached_reason_codes: dict[str, Any] | None = None


def get_all_reason_codes() -> dict[str, Any]:
    global _cached_reason_codes
    if _cached_reason_codes is None:
        _cached_reason_codes = load_reason_code_config(CONFIG_DIR)
    return _cached_reason_codes


def get_active_predictor() -> DisputeWinPredictor:
    return get_win_predictor()


def get_portfolio_analytics() -> PortfolioAnalytics:
    return PortfolioAnalytics()


def get_merchant_profiler() -> MerchantRiskProfiler:
    return MerchantRiskProfiler()
