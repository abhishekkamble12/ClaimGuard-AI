import { AlertTriangle, ArrowUpRight } from 'lucide-react';
import './GapGuidanceCards.css';

/**
 * Actionable gap guidance cards showing missing evidence and projected readiness gain.
 */
export default function GapGuidanceCards({ gaps = [], onSelectEvidence }) {
  if (!gaps || gaps.length === 0) return null;

  return (
    <div className="gap-guidance-container">
      <div className="gap-guidance-header">
        <AlertTriangle size={16} className="text-warning" />
        <h5>Actionable Evidence Recommendations</h5>
      </div>
      <div className="gap-guidance-grid">
        {gaps.map((gap, index) => {
          const evidenceName = gap.evidence_id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          const gainPct = Math.round((gap.potential_score_gain || 0) * 100);
          const projectedScore = Math.round((gap.score_if_added || 0) * 100);

          return (
            <div key={gap.evidence_id || index} className="gap-card card animate-fade-in-up">
              <div className="gap-card-top">
                <span className="gap-evidence-name">{evidenceName}</span>
                <span className="gap-gain-badge">+{gainPct}% Gain</span>
              </div>
              <p className="gap-card-desc">
                Current status: <strong className="text-danger">{gap.current_status}</strong>. Uploading this document elevates readiness to <strong>{projectedScore}%</strong> (Projected Tier: <span className="text-success">{gap.projected_risk}</span>).
              </p>
              {onSelectEvidence && (
                <button
                  className="btn btn-ghost btn-sm gap-action-btn"
                  onClick={() => onSelectEvidence(gap.evidence_id)}
                >
                  <span>Simulate Upload</span>
                  <ArrowUpRight size={13} />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
