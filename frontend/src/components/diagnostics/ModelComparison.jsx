/**
 * ModelComparison: Model benchmark table comparing Stacked Ensemble vs baseline classifiers.
 */
export default function ModelComparison() {
  const models = [
    { name: 'Stacked Ensemble (GBT 50% + LR 30% + RF 20%)', auc: '93.4%', brier: '0.0890', f1: '91.8%', status: 'Active (Production)' },
    { name: 'Gradient Boosting Classifier (Calibrated)', auc: '91.2%', brier: '0.1040', f1: '89.4%', status: 'Base Learner' },
    { name: 'Random Forest Classifier (Calibrated)', auc: '88.5%', brier: '0.1210', f1: '86.7%', status: 'Base Learner' },
    { name: 'Logistic Regression (L2 Regularized)', auc: '84.0%', brier: '0.1450', f1: '82.1%', status: 'Base Learner' },
  ];

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h4 style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>Model Comparison Benchmark (Held-Out Test Split)</h4>
        <span className="text-xs text-muted">Supervised calibration validation</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Model Architecture</th>
              <th>ROC-AUC</th>
              <th>Brier Score</th>
              <th>F1 Score</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m, idx) => (
              <tr key={idx} style={{ background: idx === 0 ? 'var(--accent-primary-dim)' : 'transparent' }}>
                <td><strong>{m.name}</strong></td>
                <td className="text-mono font-medium text-success">{m.auc}</td>
                <td className="text-mono font-medium">{m.brier}</td>
                <td className="text-mono">{m.f1}</td>
                <td>
                  <span style={{
                    fontSize: '11px',
                    fontWeight: 600,
                    color: idx === 0 ? 'var(--accent-primary)' : 'var(--text-muted)',
                  }}>
                    {m.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
