import './RiskBadge.css';

const RISK_CONFIG = {
  LOW:    { label: 'LOW',    className: 'risk-badge--low' },
  MEDIUM: { label: 'MEDIUM', className: 'risk-badge--medium' },
  HIGH:   { label: 'HIGH',   className: 'risk-badge--high' },
};

/**
 * Risk level badge.
 * Props: level ('LOW' | 'MEDIUM' | 'HIGH')
 */
export default function RiskBadge({ level }) {
  const config = RISK_CONFIG[level] || RISK_CONFIG.HIGH;
  return (
    <span className={`risk-badge ${config.className}`}>
      {config.label}
    </span>
  );
}
