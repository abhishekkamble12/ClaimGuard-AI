import StatusBadge from '../common/StatusBadge';
import './EvidenceTable.css';

/**
 * EvidenceTable: Table listing all required evidence elements with status badges,
 * weights, weighted contributions, and TF-IDF similarity percentages.
 */
export default function EvidenceTable({ evidenceElements = {} }) {
  const entries = Object.entries(evidenceElements);

  if (entries.length === 0) {
    return (
      <div className="evidence-table-card card">
        <p className="text-muted">No evidence specifications available for this reason code.</p>
      </div>
    );
  }

  return (
    <div className="evidence-table-card card">
      <div className="evidence-table-header">
        <h4>Evidence Quality & TF-IDF Similarity Matrix</h4>
        <span className="text-xs text-muted">{entries.length} elements evaluated</span>
      </div>

      <div className="evidence-table-scroll">
        <table className="evidence-table">
          <thead>
            <tr>
              <th>Evidence Element</th>
              <th>Status</th>
              <th>Weight</th>
              <th>Contribution</th>
              <th>TF-IDF Match</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([eid, detail]) => {
              const label = eid.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
              const weightPct = Math.round((detail.weight || 0) * 100);
              const contribPct = Math.round((detail.weighted_contribution || 0) * 100);
              const tfidfPct = Math.round((detail.semantic_relevance || 0) * 100);

              return (
                <tr key={eid}>
                  <td className="evidence-name-cell">
                    <strong>{label}</strong>
                  </td>
                  <td>
                    <StatusBadge status={detail.status} />
                  </td>
                  <td className="text-mono">{weightPct}%</td>
                  <td className="text-mono font-medium">{contribPct}%</td>
                  <td>
                    <div className="tfidf-cell">
                      <div className="tfidf-bar-bg">
                        <div
                          className="tfidf-bar-fill"
                          style={{ width: `${Math.min(100, tfidfPct)}%` }}
                        />
                      </div>
                      <span className="text-mono text-xs">{tfidfPct}%</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
