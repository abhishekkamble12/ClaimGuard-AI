import './LoadingState.css';

/**
 * Skeleton loading state with shimmer animation.
 * Props: lines (number of skeleton lines), height
 */
export default function LoadingState({ lines = 4, height = '16px', className = '' }) {
  return (
    <div className={`loading-state ${className}`}>
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className="skeleton-line animate-shimmer"
          style={{
            height,
            width: `${85 - i * 10}%`,
            animationDelay: `${i * 100}ms`,
          }}
        />
      ))}
    </div>
  );
}

/**
 * Spinner loading indicator.
 */
export function Spinner({ size = 24, className = '' }) {
  return (
    <div className={`spinner ${className}`} style={{ width: size, height: size }}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <circle cx="12" cy="12" r="10" strokeOpacity="0.2" />
        <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" className="animate-spin" />
      </svg>
    </div>
  );
}
