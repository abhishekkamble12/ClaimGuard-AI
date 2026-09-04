import MetricCard from '../common/MetricCard';
import './MetricStrip.css';

/**
 * MetricStrip: top-level KPI cards + deadline urgency badge for the active dispute case.
 */
export default function MetricStrip({ scoringResult, disputeCase }) {
  if (!scoringResult) return null;

  const {
    completeness_score = 0,
    confidence = 0,
    win_probability = 0,
    expected_financial_value = {},
    risk_level = 'MEDIUM',
    economic_recommendation = {},
  } = scoringResult;

  const evInr = expected_financial_value.expected_value_inr ?? 0;
  const action = economic_recommendation.action || 'CONTEST';

  const riskScore = scoringResult.risk_score_100 ?? Math.round((1 - win_probability) * 100);
  const riskTier = scoringResult.risk_tier || risk_level;
  const scoreVariant = riskScore <= 35 ? 'success' : riskScore <= 65 ? 'warning' : 'danger';

  const readinessVariant = completeness_score >= 0.8 ? 'success' : completeness_score >= 0.5 ? 'warning' : 'danger';
  const confVariant = confidence >= 0.75 ? 'success' : 'warning';
  const riskVariant = riskTier === 'LOW' ? 'success' : riskTier === 'MEDIUM' ? 'warning' : 'danger';
  const evVariant = evInr >= 0 ? 'success' : 'danger';

  // Deadline urgency from dispute case metadata
  const daysRemaining = disputeCase?.days_remaining ?? null;
  const evidenceDueBy = disputeCase?.evidence_due_by ?? null;
  let deadlineLabel = null;
  let deadlineVariant = 'info';
  if (daysRemaining !== null) {
    if (daysRemaining <= 0) {
      deadlineLabel = 'OVERDUE';
      deadlineVariant = 'danger';
    } else if (daysRemaining <= 3) {
      deadlineLabel = `${daysRemaining}d left`;
      deadlineVariant = 'danger';
    } else if (daysRemaining <= 7) {
      deadlineLabel = `${daysRemaining}d left`;
      deadlineVariant = 'warning';
    } else {
      deadlineLabel = `${daysRemaining}d left`;
      deadlineVariant = 'success';
    }
  }

  return (
    <>
      {/* Deadline urgency banner — shown only when days_remaining is available */}
      {deadlineLabel && (
        <div className={`deadline-badge deadline-badge--${deadlineVariant}`}>
          <span className="deadline-badge__icon">
            {deadlineVariant === 'danger' ? '🚨' : deadlineVariant === 'warning' ? '⏰' : '📅'}
          </span>
          <span className="deadline-badge__text">
            <strong>Response Deadline:</strong>{' '}
            {daysRemaining <= 0
              ? 'Evidence submission window has closed'
              : `${deadlineLabel} to submit evidence`}
            {evidenceDueBy
              ? ` · Due ${new Date(evidenceDueBy * 1000).toLocaleDateString('en-IN', {
                  day: 'numeric', month: 'short', year: 'numeric',
                })}`
              : ''}
          </span>
        </div>
      )}

      <div className="metric-strip grid-6">
        <MetricCard
          label="ProofPilot Risk Score"
          value={riskScore}
          suffix="/100"
          variant={scoreVariant}
        />
        <MetricCard
          label="Evidence Readiness"
          value={Math.round(completeness_score * 100)}
          suffix="%"
          variant={readinessVariant}
        />
        <MetricCard
          label="Confidence Score"
          value={Math.round(confidence * 100)}
          suffix="%"
          variant={confVariant}
        />
        <MetricCard
          label="ML Win Prob P(Win)"
          value={Math.round(win_probability * 100)}
          suffix="%"
          variant="info"
        />
        <MetricCard
          label="Expected ROI (EV)"
          value={Math.round(evInr)}
          prefix="₹"
          variant={evVariant}
        />
        <MetricCard
          label="Economic Action"
          value={action}
          variant={action === 'CONTEST' ? 'success' : 'warning'}
        />
      </div>
    </>
  );
}
