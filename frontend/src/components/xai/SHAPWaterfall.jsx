import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { Layers } from 'lucide-react';
import './SHAPWaterfall.css';

/**
 * SHAPWaterfall: Interactive feature attribution visualization (Local SHAP & Global feature importance).
 */
export default function SHAPWaterfall({ featureImportances = {}, localShap = null }) {
  const [mode, setMode] = useState('local');

  // Convert feature importances to array
  const globalData = Object.entries(featureImportances)
    .slice(0, 8)
    .map(([feature, importance]) => ({
      name: feature.replace(/_/g, ' '),
      value: parseFloat(importance) || 0,
      direction: 'positive',
    }));

  // Local SHAP data if available
  const localData = localShap?.feature_names
    ? localShap.feature_names
        .map((name, i) => {
          const val = localShap.shap_values?.[i] ?? localShap.feature_vector?.[i] ?? 0.0;
          return {
            name: name.replace(/_/g, ' '),
            value: parseFloat(Number(val).toFixed(4)),
            direction: val >= 0 ? 'positive' : 'negative',
            absVal: Math.abs(val),
          };
        })
        .sort((a, b) => b.absVal - a.absVal)
        .slice(0, 8)
    : globalData;

  const activeData = mode === 'local' ? localData : globalData;

  return (
    <div className="shap-card card">
      <div className="shap-header">
        <div className="shap-title">
          <Layers size={18} className="text-info" />
          <h4>SHAP Model Explainability (Feature Attribution)</h4>
        </div>
        <div className="shap-mode-toggle">
          <button
            className={`shap-mode-btn ${mode === 'local' ? 'shap-mode-btn--active' : ''}`}
            onClick={() => setMode('local')}
          >
            This Dispute (Local)
          </button>
          <button
            className={`shap-mode-btn ${mode === 'global' ? 'shap-mode-btn--active' : ''}`}
            onClick={() => setMode('global')}
          >
            Global Model (All Cases)
          </button>
        </div>
      </div>

      <p className="shap-caption">
        {mode === 'local'
          ? 'Features pushing P(Win) up (green) or down (red) for this specific dispute.'
          : 'Average feature attribution across all 500 training dispute cases.'}
      </p>

      <div className="shap-chart-wrapper">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={activeData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#f1f5f9', fontSize: 11 }}
              width={140}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#131c30',
                borderColor: 'rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9',
                fontSize: '12px',
              }}
              formatter={(value) => [value, 'SHAP Value / Importance']}
            />
            <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {activeData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.direction === 'positive' || entry.value >= 0 ? '#34d399' : '#f87171'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
