import { Brain, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import './XAISummaryCard.css';

/**
 * XAISummaryCard: Plain-English explanation of all AI decisions made for the active dispute.
 */
export default function XAISummaryCard({ scoringResult, disputeCase }) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!scoringResult || !disputeCase) return null;

  const completeness = scoringResult.completeness_pct || '0%';
  const confidence = scoringResult.confidence_pct || '0%';
  const winProb = scoringResult.win_probability_pct || '0%';
  const risk = scoringResult.risk_level || 'MEDIUM';
  const route = scoringResult.routing_decision || 'human_review';
  const action = scoringResult.economic_recommendation?.action || 'CONTEST';
  const missingCount = scoringResult.missing_evidence?.length || 0;
  const weakCount = scoringResult.weak_evidence?.length || 0;
  const merchant = disputeCase.merchant_name || 'Merchant';
  const reasonTitle = disputeCase.reason_title || disputeCase.reason_category || 'Dispute';

  return (
    <div className="xai-summary-card card">
      <div
        className="xai-summary-header"
        onClick={() => setIsExpanded(!isExpanded)}
        role="button"
        tabIndex={0}
      >
        <div className="xai-summary-title">
          <Brain size={18} className="text-primary" />
          <h4>ProofPilot XAI — Decision Narrative Summary</h4>
        </div>
        <button className="btn-icon">
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {isExpanded && (
        <div className="xai-summary-content animate-fade-in">
          <p>
            <strong>Executive Summary:</strong> Dispute case for <strong>{merchant}</strong> regarding <em>{reasonTitle}</em> was scored with an overall evidence readiness of <strong>{completeness}</strong> and model confidence of <strong>{confidence}</strong>.
          </p>

          <div className="xai-summary-grid">
            <div className="xai-point">
              <strong>1. Evidence Analysis:</strong>
              <span>
                {missingCount === 0 && weakCount === 0
                  ? 'All required evidence documents were identified and verified.'
                  : `${missingCount} missing document(s) and ${weakCount} weak evidence item(s) penalized readiness.`}
              </span>
            </div>
            <div className="xai-point">
              <strong>2. Calibrated ML Win Prob:</strong>
              <span>
                Stacked Ensemble (GBT + LR + RF) predicts <strong>{winProb}</strong> win probability based on 22 engineered features.
              </span>
            </div>
            <div className="xai-point">
              <strong>3. Economic Decision:</strong>
              <span>
                Economic engine recommends <strong>{action}</strong> ({scoringResult.economic_recommendation?.reason || 'Calculated positive ROI'}).
              </span>
            </div>
            <div className="xai-point">
              <strong>4. Decision Routing:</strong>
              <span>
                Routed to <code>{route}</code> with an assigned <strong>{risk}</strong> risk tier.
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
