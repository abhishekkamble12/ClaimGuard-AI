import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

/**
 * VelocityTimeline: Temporal dispute volume influx timeline.
 */
export default function VelocityTimeline({ timeline = [] }) {
  const dummyTimeline = [
    { date: '2026-08-01', count: 4 },
    { date: '2026-08-04', count: 7 },
    { date: '2026-08-07', count: 3 },
    { date: '2026-08-10', count: 9 },
    { date: '2026-08-13', count: 14 },
    { date: '2026-08-16', count: 8 },
    { date: '2026-08-19', count: 5 },
    { date: '2026-08-22', count: 11 },
    { date: '2026-08-25', count: 6 },
    { date: '2026-08-28', count: 12 },
  ];

  const data = timeline.length > 0 ? timeline : dummyTimeline;

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h4 style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>Dispute Influx Velocity & Anomaly Timeline</h4>
        <span className="text-xs text-muted">Daily incoming dispute velocity</span>
      </div>

      <div style={{ width: '100%', height: '200px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#131c30',
                borderColor: 'rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9',
                fontSize: '12px',
              }}
              formatter={(value) => [`${value} disputes`, 'Volume']}
            />
            <Line
              type="monotone"
              dataKey="count"
              stroke="#818cf8"
              strokeWidth={2}
              dot={{ fill: '#818cf8', r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
