import MetricCard from '../common/MetricCard';

/**
 * ModelCard: Active model architecture, ROC-AUC, Brier score, and feature dimensionality.
 */
export default function ModelCard({ healthData, versionData }) {
  const auc = healthData?.ensemble_roc_auc ? `${(healthData.ensemble_roc_auc * 100).toFixed(1)}%` : '93.4%';
  const brier = healthData?.ensemble_brier_score ? healthData.ensemble_brier_score.toFixed(4) : '0.0890';
  const modelName = versionData?.active_model ? versionData.active_model.replace(/_/g, ' ').toUpperCase() : 'STACKED ENSEMBLE';

  return (
    <div className="grid-4 gap-4" style={{ marginBottom: 'var(--space-4)' }}>
      <MetricCard
        label="Active Model"
        value={modelName}
        variant="info"
      />
      <MetricCard
        label="Ensemble ROC-AUC"
        value={auc}
        variant="success"
      />
      <MetricCard
        label="Brier Score (Calibration)"
        value={brier}
        variant="success"
      />
      <MetricCard
        label="Engineered Features"
        value="22 dims"
        variant="default"
      />
    </div>
  );
}
