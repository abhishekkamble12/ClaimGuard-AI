import { Sparkles, RotateCcw } from 'lucide-react';
import './CounterfactualSim.css';

/**
 * CounterfactualSim: Hero interactive simulator allowing merchants to simulate
 * uploading missing documents to project real-time score elevations.
 */
export default function CounterfactualSim({
  missingEvidence = [],
  weakEvidence = [],
  selectedGaps = [],
  onToggleGap,
  onApplyGaps,
  onResetCase,
  hasSimulatedEvidence = false,
  isScoring = false,
}) {
  const fixable = [...new Set([...missingEvidence, ...weakEvidence])];

  return (
    <div className="counterfactual-card card">
      <div className="counterfactual-header">
        <div className="counterfactual-title">
          <Sparkles size={18} className="text-primary" />
          <h4>Counterfactual Risk Simulator</h4>
        </div>
        <span className="counterfactual-badge">Hero Feature</span>
      </div>

      <p className="counterfactual-sub">
        Simulate merchant uploading missing evidence to dynamically project score, P(Win), and risk tier upgrades.
      </p>

      {fixable.length > 0 ? (
        <div className="counterfactual-checklist">
          <label className="checklist-label">Select missing / weak evidence to provide:</label>
          <div className="checklist-items">
            {fixable.map((eid) => {
              const label = eid.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
              const isChecked = selectedGaps.includes(eid);
              const isMissing = missingEvidence.includes(eid);

              return (
                <label key={eid} className={`checklist-item ${isChecked ? 'checklist-item--checked' : ''}`}>
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => onToggleGap(eid)}
                  />
                  <span className="checklist-item-name">{label}</span>
                  <span className={`checklist-tag ${isMissing ? 'checklist-tag--missing' : 'checklist-tag--weak'}`}>
                    {isMissing ? 'Missing' : 'Weak'}
                  </span>
                </label>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="text-success text-sm font-medium">
          ✅ All required evidence elements are already present!
        </p>
      )}

      <div className="counterfactual-actions">
        {fixable.length > 0 && (
          <button
            className="btn btn-primary w-full"
            onClick={onApplyGaps}
            disabled={selectedGaps.length === 0 || isScoring}
          >
            <Sparkles size={15} />
            <span>{isScoring ? 'Re-Scoring...' : 'Apply Evidence & Re-Score Case'}</span>
          </button>
        )}

        {hasSimulatedEvidence && (
          <button
            className="btn btn-ghost btn-sm w-full"
            onClick={onResetCase}
            disabled={isScoring}
          >
            <RotateCcw size={13} />
            <span>Reset Case to Original State</span>
          </button>
        )}
      </div>
    </div>
  );
}
