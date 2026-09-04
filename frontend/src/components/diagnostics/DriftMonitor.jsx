/**
 * DriftMonitor: Population Stability Index (PSI) traffic light monitoring component.
 */
export default function DriftMonitor({ driftData }) {
  const reports = driftData?.drift_reports || [
    { feature_name: 'completeness_score', psi: 0.042, status: 'NO_DRIFT' },
    { feature_name: 'evidence_entropy', psi: 0.061, status: 'NO_DRIFT' },
    { feature_name: 'log_amount', psi: 0.088, status: 'NO_DRIFT' },
    { feature_name: 'tfidf_mean_similarity', psi: 0.112, status: 'MODERATE_SHIFT' },
    { feature_name: 'missing_evidence_count', psi: 0.035, status: 'NO_DRIFT' },
  ];

  const getPsiBadge = (status, psi) => {
    if (status === 'NO_DRIFT' || psi < 0.1) {
      return <span style={{ color: 'var(--accent-success)', background: 'var(--accent-success-dim)', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600 }}>🟢 No Shift (PSI &lt; 0.10)</span>;
    }
    if (status === 'MODERATE_SHIFT' || psi < 0.25) {
      return <span style={{ color: 'var(--accent-warning)', background: 'var(--accent-warning-dim)', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600 }}>🟡 Warning (0.10 &le; PSI &lt; 0.25)</span>;
    }
    return <span style={{ color: 'var(--accent-danger)', background: 'var(--accent-danger-dim)', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600 }}>🔴 Critical Drift (PSI &ge; 0.25)</span>;
  };

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h4 style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>Population Stability Index (PSI) Drift Monitor</h4>
        <span className="text-xs text-muted">Real-time inference vs training baseline</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Monitored Feature</th>
              <th>Calculated PSI</th>
              <th>Stability Status</th>
            </tr>
          </thead>
          <tbody>
            {reports.map((r, i) => (
              <tr key={i}>
                <td><code>{r.feature_name}</code></td>
                <td className="text-mono font-medium">{(r.psi || 0).toFixed(4)}</td>
                <td>{getPsiBadge(r.status, r.psi)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
