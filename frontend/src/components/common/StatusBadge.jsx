import { CheckCircle2, AlertTriangle, XCircle, Ban } from 'lucide-react';
import './StatusBadge.css';

const STATUS_CONFIG = {
  present:    { icon: CheckCircle2,  label: 'Present',    className: 'status-badge--success' },
  weak:       { icon: AlertTriangle, label: 'Weak',       className: 'status-badge--warning' },
  missing:    { icon: XCircle,       label: 'Missing',    className: 'status-badge--danger' },
  irrelevant: { icon: Ban,           label: 'Irrelevant', className: 'status-badge--muted' },
};

/**
 * Evidence status badge with icon.
 * Props: status ('present' | 'weak' | 'missing' | 'irrelevant')
 */
export default function StatusBadge({ status }) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.missing;
  const Icon = config.icon;

  return (
    <span className={`status-badge ${config.className} animate-scale-bounce`}>
      <Icon size={13} />
      <span>{config.label}</span>
    </span>
  );
}
