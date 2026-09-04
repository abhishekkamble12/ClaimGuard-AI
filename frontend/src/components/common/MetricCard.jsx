import { useEffect, useRef, useState } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import './MetricCard.css';

/**
 * Animated metric card with label, value, optional delta, and trend indicator.
 * Props: label, value, prefix, suffix, delta, deltaLabel, variant (default|success|warning|danger)
 */
export default function MetricCard({
  label,
  value,
  prefix = '',
  suffix = '',
  delta = null,
  deltaLabel = '',
  variant = 'default',
  className = '',
}) {
  const [displayValue, setDisplayValue] = useState(0);
  const animatedRef = useRef(null);

  useEffect(() => {
    const numValue = typeof value === 'number' ? value : parseFloat(value) || 0;
    const duration = 800;
    const start = performance.now();
    const startVal = displayValue;

    function animate(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); /* easeOutCubic */
      setDisplayValue(startVal + (numValue - startVal) * eased);
      if (progress < 1) {
        animatedRef.current = requestAnimationFrame(animate);
      }
    }

    animatedRef.current = requestAnimationFrame(animate);
    return () => {
      if (animatedRef.current) cancelAnimationFrame(animatedRef.current);
    };
  }, [value]);

  const formattedValue = typeof value === 'string' && isNaN(value)
    ? value
    : `${prefix}${Number.isInteger(value) ? Math.round(displayValue) : displayValue.toFixed(value?.toString().includes('.') ? 1 : 0)}${suffix}`;

  const trendIcon = delta > 0
    ? <TrendingUp size={14} />
    : delta < 0
    ? <TrendingDown size={14} />
    : delta !== null
    ? <Minus size={14} />
    : null;

  const deltaClass = delta > 0 ? 'metric-delta--positive' : delta < 0 ? 'metric-delta--negative' : '';

  return (
    <div className={`metric-card metric-card--${variant} ${className} animate-fade-in-up`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{formattedValue}</span>
      {delta !== null && (
        <span className={`metric-delta ${deltaClass}`}>
          {trendIcon}
          {delta > 0 ? '+' : ''}{delta}
          {deltaLabel && <span className="metric-delta-label">{deltaLabel}</span>}
        </span>
      )}
    </div>
  );
}
