import MetricCard from '../common/MetricCard';

/**
 * PortfolioKPIs: 4 top-level macro metrics for the merchant portfolio.
 */
export default function PortfolioKPIs({ report }) {
  if (!report) return null;

  const totalDisputes = report.total_disputes || 0;
  const amountAtRisk = report.total_amount_at_risk_inr || 0;
  const recovered = report.amount_recovered_by_contesting_inr || 0;
  const feesSaved = report.fees_saved_by_abstaining_inr || 0;

  return (
    <div className="grid-4 gap-4" style={{ marginBottom: 'var(--space-4)' }}>
      <MetricCard
        label="Total Portfolio Disputes"
        value={totalDisputes}
        variant="default"
      />
      <MetricCard
        label="Total Amount at Risk"
        value={Math.round(amountAtRisk)}
        prefix="₹"
        variant="warning"
      />
      <MetricCard
        label="Recovered by Contesting"
        value={Math.round(recovered)}
        prefix="₹"
        variant="success"
      />
      <MetricCard
        label="Penalty Fees Saved"
        value={Math.round(feesSaved)}
        prefix="₹"
        variant="info"
      />
    </div>
  );
}
