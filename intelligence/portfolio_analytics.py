"""
ProofPilot — Portfolio Analytics & Systemic Risk Intelligence
--------------------------------------------------------------
Aggregates and analyzes dispute decisions across entire merchant portfolios,
quantifying financial ROI, fees saved through intelligent abstention,
and systemic evidence bottlenecks across card and UPI networks.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

DISPUTE_FEE_INR = 500.0


@dataclass
class PortfolioReport:
    total_disputes: int
    total_amount_at_risk_inr: float
    avg_dispute_amount_inr: float
    total_won_disputes: int
    portfolio_win_rate: float
    amount_recovered_by_contesting_inr: float
    fees_saved_by_abstaining_inr: float
    net_financial_benefit_inr: float
    projected_annual_savings_inr: float
    win_rate_by_category: dict[str, float]
    readiness_by_category: dict[str, float]
    network_breakdown: dict[str, int]
    top_systemic_evidence_gaps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PortfolioAnalytics:
    """
    Computes enterprise-grade portfolio metrics and macro dispute trends
    for payment risk operations teams.
    """

    def generate_portfolio_report(self, disputes: list[dict[str, Any]]) -> PortfolioReport:
        total = len(disputes)
        if total == 0:
            return PortfolioReport(
                total_disputes=0,
                total_amount_at_risk_inr=0.0,
                avg_dispute_amount_inr=0.0,
                total_won_disputes=0,
                portfolio_win_rate=0.0,
                amount_recovered_by_contesting_inr=0.0,
                fees_saved_by_abstaining_inr=0.0,
                net_financial_benefit_inr=0.0,
                projected_annual_savings_inr=0.0,
                win_rate_by_category={},
                readiness_by_category={},
                network_breakdown={},
                top_systemic_evidence_gaps=[],
            )

        amounts = [float(d.get("transaction", {}).get("amount", 1000.0)) for d in disputes]
        total_at_risk = sum(amounts)
        avg_amount = total_at_risk / total

        won_cases = [d for d in disputes if d.get("expected_outcome") == "won"]
        won_count = len(won_cases)
        win_rate = won_count / total

        # Financial recovery: Won cases where evidence readiness is high (>= 0.70)
        recovered_amount = sum(
            float(d.get("transaction", {}).get("amount", 0.0))
            for d in won_cases
            if float(d.get("expected_completeness_score", d.get("completeness_score", 0.5))) >= 0.70
        )

        # Fees saved: Doomed cases (score < 0.50 or outcome lost) where system abstained
        abstained_cases = [
            d for d in disputes
            if d.get("expected_route") in {"human_review", "request_more_evidence"}
            or float(d.get("expected_completeness_score", d.get("completeness_score", 0.5))) < 0.50
        ]
        fees_saved = len(abstained_cases) * DISPUTE_FEE_INR

        net_benefit = recovered_amount + fees_saved
        projected_annual = net_benefit * 12.0

        # Category level aggregation
        cat_disputes: dict[str, list[dict[str, Any]]] = {}
        for d in disputes:
            cat = d.get("reason_category", "other")
            cat_disputes.setdefault(cat, []).append(d)

        win_rate_by_cat = {}
        readiness_by_cat = {}
        for cat, c_list in cat_disputes.items():
            c_won = sum(1 for d in c_list if d.get("expected_outcome") == "won")
            win_rate_by_cat[cat] = round(c_won / len(c_list), 4)
            c_readiness = [
                float(d.get("expected_completeness_score", d.get("completeness_score", 0.5)))
                for d in c_list
            ]
            readiness_by_cat[cat] = round(sum(c_readiness) / len(c_list), 4)

        # Network breakdown
        network_breakdown: dict[str, int] = {}
        for d in disputes:
            net = d.get("network", "CARD")
            network_breakdown[net] = network_breakdown.get(net, 0) + 1

        # Systemic evidence gaps ranking
        gap_frequencies: dict[str, int] = {}
        gap_financial_loss: dict[str, float] = {}
        for d in disputes:
            amt = float(d.get("transaction", {}).get("amount", 0.0))
            for gap in d.get("missing_evidence", []):
                gap_frequencies[gap] = gap_frequencies.get(gap, 0) + 1
                gap_financial_loss[gap] = gap_financial_loss.get(gap, 0.0) + amt

        top_gaps = []
        for gap, count in sorted(gap_frequencies.items(), key=lambda x: x[1], reverse=True)[:6]:
            top_gaps.append({
                "evidence_id": gap,
                "evidence_name": gap.replace("_", " ").title(),
                "occurrence_count": count,
                "portfolio_gap_pct": round(count / total, 4),
                "total_amount_exposed_inr": round(gap_financial_loss.get(gap, 0.0), 2),
            })

        return PortfolioReport(
            total_disputes=total,
            total_amount_at_risk_inr=round(total_at_risk, 2),
            avg_dispute_amount_inr=round(avg_amount, 2),
            total_won_disputes=won_count,
            portfolio_win_rate=round(win_rate, 4),
            amount_recovered_by_contesting_inr=round(recovered_amount, 2),
            fees_saved_by_abstaining_inr=round(fees_saved, 2),
            net_financial_benefit_inr=round(net_benefit, 2),
            projected_annual_savings_inr=round(projected_annual, 2),
            win_rate_by_category=win_rate_by_cat,
            readiness_by_category=readiness_by_cat,
            network_breakdown=network_breakdown,
            top_systemic_evidence_gaps=top_gaps,
        )
