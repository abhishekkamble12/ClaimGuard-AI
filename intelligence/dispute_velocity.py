"""
ProofPilot — Dispute Velocity & Surge Analyzer
------------------------------------------------
Monitors dispute arrival velocity over time, detects velocity spikes,
and triggers early warning alerts for merchant risk operations.
"""

from collections import Counter
from datetime import datetime, timedelta
from typing import Any


class DisputeVelocityAnalyzer:
    """
    Analyzes temporal patterns, velocity surges, and category spikes
    in dispute streams.
    """

    def compute_daily_timeline(self, disputes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group dispute volume and disputed monetary amounts by transaction/dispute date."""
        dates_volume: Counter[str] = Counter()
        dates_amount: dict[str, float] = {}

        for d in disputes:
            tx_date = d.get("transaction", {}).get("payment_date", "2026-08-15")
            # Parse or extract YYYY-MM-DD
            d_str = str(tx_date)[:10]
            amt = float(d.get("transaction", {}).get("amount", 1000.0))
            
            dates_volume[d_str] += 1
            dates_amount[d_str] = dates_amount.get(d_str, 0.0) + amt

        sorted_dates = sorted(dates_volume.keys())
        timeline = []
        for d_str in sorted_dates:
            timeline.append({
                "date": d_str,
                "dispute_count": dates_volume[d_str],
                "total_amount_inr": round(dates_amount[d_str], 2),
            })
        return timeline

    def detect_velocity_anomalies(
        self,
        disputes: list[dict[str, Any]],
        threshold_multiplier: float = 1.8,
    ) -> list[dict[str, Any]]:
        """Detect days with dispute counts significantly exceeding average velocity."""
        timeline = self.compute_daily_timeline(disputes)
        if not timeline or len(timeline) < 3:
            return []

        counts = [item["dispute_count"] for item in timeline]
        avg_count = sum(counts) / len(counts)

        anomalies = []
        for item in timeline:
            if item["dispute_count"] > avg_count * threshold_multiplier and item["dispute_count"] >= 3:
                anomalies.append({
                    "date": item["date"],
                    "dispute_count": item["dispute_count"],
                    "average_count": round(avg_count, 1),
                    "surge_factor": round(item["dispute_count"] / max(avg_count, 1.0), 2),
                    "severity": "CRITICAL" if item["dispute_count"] > avg_count * 2.5 else "WARNING",
                })
        return anomalies


def compute_merchant_velocity_context(disputes: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """
    Computes rolling merchant dispute velocity features for each dispute in a sequence:
    - disputes_last_7d: number of disputes by this merchant in the 7 days preceding this dispute
    - rolling_win_rate_30d: win rate for this merchant over past 30 days (default 0.35 if no history)
    - days_since_last_dispute: days elapsed since the merchant's prior dispute
    """
    def _parse_date(d: dict[str, Any]) -> datetime:
        dt_str = d.get("created_at") or d.get("transaction", {}).get("payment_date") or "2026-08-01"
        try:
            return datetime.fromisoformat(str(dt_str)[:10])
        except Exception:
            return datetime(2026, 8, 1)

    sorted_disputes = sorted(disputes, key=_parse_date)
    merchant_history: dict[str, list[tuple[datetime, str]]] = {}
    result_map: dict[str, dict[str, float]] = {}

    for d in sorted_disputes:
        d_id = d.get("dispute_id", "")
        m_name = d.get("merchant_name") or d.get("transaction", {}).get("merchant_name", "Unknown")
        d_date = _parse_date(d)
        outcome = d.get("expected_outcome", "lost")

        history = merchant_history.get(m_name, [])
        if not history:
            result_map[d_id] = {
                "disputes_last_7d": 1.0,
                "rolling_win_rate_30d": 0.35,
                "days_since_last_dispute": 14.0,
            }
        else:
            cutoff_7d = d_date - timedelta(days=7)
            recent_7d = [h for h in history if h[0] >= cutoff_7d]
            disputes_last_7d = float(len(recent_7d) + 1)

            cutoff_30d = d_date - timedelta(days=30)
            recent_30d = [h for h in history if h[0] >= cutoff_30d]
            if recent_30d:
                wins = sum(1 for h in recent_30d if h[1] == "won")
                win_rate_30d = wins / len(recent_30d)
            else:
                win_rate_30d = 0.35

            last_date = history[-1][0]
            days_since = max(0.0, float((d_date - last_date).days))

            result_map[d_id] = {
                "disputes_last_7d": disputes_last_7d,
                "rolling_win_rate_30d": round(win_rate_30d, 4),
                "days_since_last_dispute": round(days_since, 1),
            }

        history.append((d_date, outcome))
        merchant_history[m_name] = history

    return result_map
