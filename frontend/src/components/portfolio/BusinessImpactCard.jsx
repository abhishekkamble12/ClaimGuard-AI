import { ShieldAlert, ShieldCheck, TrendingUp, AlertTriangle, CheckCircle, IndianRupee } from 'lucide-react';
import './BusinessImpactCard.css';

/**
 * BusinessImpactCard: Displays macro financial recovery impact, FP fees avoided,
 * FN revenue won, and Visa/Mastercard VAMP/VCMP regulatory health gauge.
 */
export default function BusinessImpactCard({ report, merchants = [] }) {
  if (!report) return null;

  const netBenefit = report.net_financial_benefit_inr || 0;
  const feesSaved = report.fees_saved_by_abstaining_inr || 0;
  const amountsRecovered = report.amount_recovered_by_contesting_inr || 0;
  const annualSavings = report.projected_annual_savings_inr || netBenefit * 12;

  // Compute VAMP tier counts from merchants
  const totalMerchants = merchants.length || 1;
  const healthyCount = merchants.filter((m) => m.vamp_risk_tier === 'HEALTHY').length;
  const warningCount = merchants.filter((m) => m.vamp_risk_tier === 'WARNING').length;
  const penaltyCount = merchants.filter((m) => m.vamp_risk_tier === 'PENALTY' || m.vamp_risk_tier === 'CRITICAL_EXCESSIVE').length;

  const healthyPct = Math.round((healthyCount / totalMerchants) * 100);
  const warningPct = Math.round((warningCount / totalMerchants) * 100);
  const penaltyPct = Math.max(0, 100 - healthyPct - warningPct);

  return (
    <div className="impact-card card animate-fade-in">
      <div className="impact-header">
        <div className="impact-title-group">
          <div className="impact-icon-badge">
            <TrendingUp size={22} className="text-success" />
          </div>
          <div>
            <h3 className="impact-title">Business Impact & Financial Value Engine</h3>
            <p className="impact-subtitle">
              Cost-asymmetric Expected Value (EV) optimization vs. Visa & Mastercard Regulatory Risk
            </p>
          </div>
        </div>
        <div className="impact-annual-pill">
          <IndianRupee size={16} />
          <span>Annual Run-Rate: <strong>₹{Math.round(annualSavings).toLocaleString('en-IN')}</strong></span>
        </div>
      </div>

      <div className="impact-metrics-grid">
        <div className="impact-metric-box">
          <span className="impact-metric-label">Net EV Profit / Savings</span>
          <span className="impact-metric-value text-success">
            ₹{Math.round(netBenefit).toLocaleString('en-IN')}
          </span>
          <span className="impact-metric-hint">Risk-adjusted return after penalty costs</span>
        </div>

        <div className="impact-metric-box">
          <span className="impact-metric-label">FP Penalty Fees Avoided</span>
          <span className="impact-metric-value text-info">
            ₹{Math.round(feesSaved).toLocaleString('en-IN')}
          </span>
          <span className="impact-metric-hint">₹500 fees prevented via abstention on doomed claims</span>
        </div>

        <div className="impact-metric-box">
          <span className="impact-metric-label">FN Amounts Recovered</span>
          <span className="impact-metric-value text-warning">
            ₹{Math.round(amountsRecovered).toLocaleString('en-IN')}
          </span>
          <span className="impact-metric-hint">Winnable revenue reclaimed through complete proof</span>
        </div>

        <div className="impact-metric-box">
          <span className="impact-metric-label">VAMP / VCMP Portfolio Status</span>
          <span className="impact-metric-value text-white">
            {penaltyCount === 0 ? 'Compliant' : `${penaltyCount} In Penalty`}
          </span>
          <span className="impact-metric-hint">{healthyCount}/{totalMerchants} merchants within &lt;0.65% safe ceiling</span>
        </div>
      </div>

      {/* VAMP / VCMP Regulatory Health Gauge */}
      <div className="vamp-gauge-section">
        <div className="vamp-gauge-header">
          <div className="vamp-gauge-title">
            <ShieldCheck size={16} className="text-info" />
            <span>VAMP / VCMP Merchant Health Distribution (Visa &lt;0.90% Ceiling)</span>
          </div>
          <div className="vamp-badges">
            <span className="vamp-badge vamp-badge--healthy">
              <CheckCircle size={12} /> {healthyCount} Healthy (≤0.65%)
            </span>
            <span className="vamp-badge vamp-badge--warning">
              <AlertTriangle size={12} /> {warningCount} Warning (0.65-0.90%)
            </span>
            <span className="vamp-badge vamp-badge--penalty">
              <ShieldAlert size={12} /> {penaltyCount} Excessive (&gt;0.90%)
            </span>
          </div>
        </div>

        <div className="vamp-bar-track">
          <div
            className="vamp-bar-fill vamp-bar-fill--healthy"
            style={{ width: `${healthyPct}%` }}
            title={`Healthy: ${healthyPct}%`}
          />
          <div
            className="vamp-bar-fill vamp-bar-fill--warning"
            style={{ width: `${warningPct}%` }}
            title={`Warning: ${warningPct}%`}
          />
          <div
            className="vamp-bar-fill vamp-bar-fill--penalty"
            style={{ width: `${penaltyPct}%` }}
            title={`Penalty: ${penaltyPct}%`}
          />
        </div>
        <div className="vamp-bar-labels">
          <span>0.00%</span>
          <span>Safe Zone (≤0.65%)</span>
          <span>Monitoring Watchlist (0.65% - 0.90%)</span>
          <span>Visa/MC Penalty Zone (&gt;0.90%)</span>
        </div>
      </div>
    </div>
  );
}
