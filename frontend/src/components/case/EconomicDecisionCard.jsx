import { Coins, CheckCircle, AlertOctagon } from 'lucide-react';
import './EconomicDecisionCard.css';

/**
 * EconomicDecisionCard: Monetary Expected Value (EV ₹) and Honest Financial Recommendation.
 */
export default function EconomicDecisionCard({ economicRecommendation, expectedFinancialValue }) {
  if (!economicRecommendation || !expectedFinancialValue) return null;

  const { action, reason, fee_saved_if_accepted } = economicRecommendation;
  const { expected_value_inr, amount_inr, win_probability, dispute_fee_inr } = expectedFinancialValue;

  const isContest = action === 'CONTEST';

  return (
    <div className={`economic-card card ${isContest ? 'economic-card--contest' : 'economic-card--accept'}`}>
      <div className="economic-header">
        <div className="economic-title">
          <Coins size={18} />
          <h4>Monetary Expected Value (₹) Engine</h4>
        </div>
        <span className={`economic-action-badge ${isContest ? 'badge-contest' : 'badge-accept'}`}>
          {isContest ? <CheckCircle size={13} /> : <AlertOctagon size={13} />}
          {action}
        </span>
      </div>

      <p className="economic-reason">
        <strong>Decision Logic:</strong> {reason}
      </p>

      <div className="economic-metrics-row">
        <div className="economic-metric">
          <span className="economic-metric-label">Disputed Amount</span>
          <span className="economic-metric-val">₹{Math.round(amount_inr || 0).toLocaleString()}</span>
        </div>
        <div className="economic-metric">
          <span className="economic-metric-label">Predicted P(Win)</span>
          <span className="economic-metric-val">{Math.round((win_probability || 0) * 100)}%</span>
        </div>
        <div className="economic-metric">
          <span className="economic-metric-label">Penalty Fee</span>
          <span className="economic-metric-val text-muted">₹{dispute_fee_inr || 500}</span>
        </div>
        <div className="economic-metric">
          <span className="economic-metric-label">Net EV (₹)</span>
          <span className={`economic-metric-val ${expected_value_inr >= 0 ? 'text-success' : 'text-danger'}`}>
            ₹{Math.round(expected_value_inr || 0).toLocaleString()}
          </span>
        </div>
      </div>

      {!isContest && fee_saved_if_accepted > 0 && (
        <div className="economic-fee-callout">
          💡 <strong>Honest Abstention:</strong> By accepting the loss, merchant saves the non-refundable ₹{fee_saved_if_accepted} penalty fee.
        </div>
      )}
    </div>
  );
}
