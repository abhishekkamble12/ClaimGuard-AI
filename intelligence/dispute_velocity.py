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
