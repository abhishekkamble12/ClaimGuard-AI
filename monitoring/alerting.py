"""
ProofPilot — Operational Risk Alerting Rules Engine
----------------------------------------------------
Defines and evaluates threshold alerts for VAMP chargeback limits,
model drift spikes, and systemic merchant win-rate drops.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class AlertRule:
    rule_id: str
    name: str
    threshold_value: float
    severity: str  # INFO, WARNING, CRITICAL
    description: str


@dataclass
class TriggeredAlert:
    rule_id: str
    rule_name: str
    current_value: float
    threshold_value: float
    severity: str
    timestamp: str
    message: str
    remediation_action: str


class AlertManager:
    """
    Evaluates operational health indicators against risk tolerance thresholds.
    """

    def __init__(self):
        self.rules = [
            AlertRule(
                rule_id="RULE_VAMP_LIMIT",
                name="VAMP/VCMP Excessive Chargeback Ratio",
                threshold_value=0.90,
                severity="CRITICAL",
                description="Merchant chargeback volume exceeds Visa/Mastercard 0.90% excessive penalty tier.",
            ),
            AlertRule(
                rule_id="RULE_EARLY_WARNING",
                name="VAMP Early Warning Zone",
                threshold_value=0.65,
                severity="WARNING",
                description="Merchant chargeback volume exceeds 0.65% early warning threshold.",
            ),
            AlertRule(
                rule_id="RULE_PSI_DRIFT",
                name="Model Feature Population Drift (PSI)",
                threshold_value=0.25,
                severity="CRITICAL",
                description="Population Stability Index exceeds 0.25 on active feature distributions.",
            ),
            AlertRule(
                rule_id="RULE_WIN_RATE_DROP",
                name="Merchant Win Rate Degradation",
                threshold_value=0.40,
                severity="WARNING",
                description="Merchant dispute win rate dropped below 40%.",
            ),
        ]

    def evaluate_merchant(self, chargeback_ratio_pct: float, win_rate: float) -> list[TriggeredAlert]:
        now_str = datetime.now(timezone.utc).isoformat()
        alerts = []

        if chargeback_ratio_pct > 0.90:
            alerts.append(TriggeredAlert(
                rule_id="RULE_VAMP_LIMIT",
                rule_name="VAMP/VCMP Excessive Chargeback Ratio",
                current_value=chargeback_ratio_pct,
                threshold_value=0.90,
                severity="CRITICAL",
                timestamp=now_str,
                message=f"Chargeback ratio of {chargeback_ratio_pct:.2f}% exceeds card network 0.90% ceiling!",
                remediation_action="Halt automated response submissions on weak cases; enforce pre-submission evidence quality gating.",
            ))
        elif chargeback_ratio_pct > 0.65:
            alerts.append(TriggeredAlert(
                rule_id="RULE_EARLY_WARNING",
                rule_name="VAMP Early Warning Zone",
                current_value=chargeback_ratio_pct,
                threshold_value=0.65,
                severity="WARNING",
                timestamp=now_str,
                message=f"Chargeback ratio of {chargeback_ratio_pct:.2f}% entered warning zone (>0.65%).",
                remediation_action="Review high-frequency reason codes and request additional proof for partial disputes.",
            ))

        if win_rate < 0.40:
            alerts.append(TriggeredAlert(
                rule_id="RULE_WIN_RATE_DROP",
                rule_name="Merchant Win Rate Degradation",
                current_value=win_rate,
                threshold_value=0.40,
                severity="WARNING",
                timestamp=now_str,
                message=f"Merchant dispute win rate ({win_rate:.1%}) is critically low.",
                remediation_action="Audit evidence collection workflows and identify top missing documents.",
            ))

        return alerts
