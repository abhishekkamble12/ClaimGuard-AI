import { useState } from 'react';
import './MerchantRiskTable.css';

/**
 * MerchantRiskTable: Sortable table displaying merchant cohort risk profiles with VAMP tiers.
 */
export default function MerchantRiskTable({ merchants = [] }) {
  const [sortBy, setSortBy] = useState('total_disputes');
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(field);
      setSortAsc(false);
    }
  };

  const sorted = [...merchants].sort((a, b) => {
    const aVal = a[sortBy] ?? 0;
    const bVal = b[sortBy] ?? 0;
    return sortAsc ? (aVal > bVal ? 1 : -1) : (aVal < bVal ? 1 : -1);
  });

  const getTierBadge = (tier) => {
    if (tier === 'HEALTHY') return <span className="vamp-badge vamp-healthy">🟢 Healthy</span>;
    if (tier === 'WARNING') return <span className="vamp-badge vamp-warning">🟡 Warning</span>;
    return <span className="vamp-badge vamp-penalty">🔴 Penalty Zone</span>;
  };

  return (
    <div className="merchant-table-card card">
      <div className="merchant-table-header">
        <h4>Merchant Risk Cohort Profiles (VAMP / VCMP Tiers)</h4>
        <span className="text-xs text-muted">{merchants.length} active merchants monitored</span>
      </div>

      <div className="merchant-table-scroll">
        <table>
          <thead>
            <tr>
              <th onClick={() => handleSort('merchant_name')} className="cursor-pointer">
                Merchant
              </th>
              <th onClick={() => handleSort('total_disputes')} className="cursor-pointer">
                Disputes
              </th>
              <th onClick={() => handleSort('total_disputed_amount_inr')} className="cursor-pointer">
                At Risk (₹)
              </th>
              <th onClick={() => handleSort('win_rate')} className="cursor-pointer">
                Win Rate
              </th>
              <th onClick={() => handleSort('avg_evidence_readiness')} className="cursor-pointer">
                Readiness
              </th>
              <th onClick={() => handleSort('chargeback_ratio_pct')} className="cursor-pointer">
                CB Ratio
              </th>
              <th>VAMP Tier</th>
              <th onClick={() => handleSort('merchant_health_score')} className="cursor-pointer">
                Health Score
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => (
              <tr key={m.merchant_id || m.merchant_name}>
                <td><strong>{m.merchant_name}</strong></td>
                <td className="text-mono">{m.total_disputes}</td>
                <td className="text-mono">₹{Math.round(m.total_disputed_amount_inr || 0).toLocaleString()}</td>
                <td className="text-mono">{Math.round((m.win_rate || 0) * 100)}%</td>
                <td className="text-mono">{Math.round((m.avg_evidence_readiness || 0) * 100)}%</td>
                <td className="text-mono font-medium">{(m.chargeback_ratio_pct || 0).toFixed(2)}%</td>
                <td>{getTierBadge(m.vamp_risk_tier)}</td>
                <td className="text-mono font-medium">
                  {Math.round(m.merchant_health_score || 0)} / 100
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
