import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './EvidenceContribChart.css';

/**
 * EvidenceContribChart: Horizontal bar chart showing evidence weighted contributions.
 */
export default function EvidenceContribChart({ evidenceElements = {} }) {
  const data = Object.entries(evidenceElements).map(([eid, detail]) => {
    const label = eid.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const contrib = Math.round((detail.weighted_contribution || 0) * 100);
    const maxPotential = Math.round((detail.weight || 0) * 100);
    return {
      name: label,
      contribution: contrib,
      maxPotential,
      status: detail.status,
    };
  });

  const getBarColor = (status) => {
    if (status === 'present') return '#34d399';
    if (status === 'weak') return '#fbbf24';
    return '#f87171';
  };

  return (
    <div className="evidence-contrib-card card">
      <div className="evidence-contrib-header">
        <h4>Evidence Contribution Decomposition</h4>
        <span className="text-xs text-muted">Weighted readiness impact</span>
      </div>

      <div className="evidence-contrib-chart-wrapper">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 40, bottom: 5 }}>
            <XAxis type="number" domain={[0, 40]} unit="%" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#f1f5f9', fontSize: 11 }}
              width={120}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#131c30',
                borderColor: 'rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9',
                fontSize: '12px',
              }}
              formatter={(value, name) => [`${value}%`, name === 'contribution' ? 'Current Contribution' : 'Max Potential']}
            />
            <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.status)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
