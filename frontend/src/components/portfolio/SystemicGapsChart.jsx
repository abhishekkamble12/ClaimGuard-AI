import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

/**
 * SystemicGapsChart: Visualizes the top missing evidence items across the portfolio.
 */
export default function SystemicGapsChart({ gaps = [] }) {
  const data = gaps.slice(0, 6).map((g) => ({
    name: g.evidence_name || g.evidence_id || 'Evidence',
    disputes: g.occurrence_count || 0,
    amountExposed: g.total_amount_exposed_inr || 0,
  }));

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h4 style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>Top Systemic Evidence Gaps Across Portfolio</h4>
        <span className="text-xs text-muted">Most frequent missing documents</span>
      </div>

      <div style={{ width: '100%', height: '220px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 50, bottom: 5 }}>
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#f1f5f9', fontSize: 11 }}
              width={130}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#131c30',
                borderColor: 'rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9',
                fontSize: '12px',
              }}
              formatter={(value, name) => [
                name === 'disputes' ? `${value} disputes` : `₹${value.toLocaleString()}`,
                name === 'disputes' ? 'Affected Disputes' : 'Exposed Amount',
              ]}
            />
            <Bar dataKey="disputes" fill="#fbbf24" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
