import { AlertCircle, RefreshCw } from 'lucide-react';
import './ErrorState.css';

/**
 * Error state with polite message and retry button.
 * Props: message, onRetry
 */
export default function ErrorState({ message, onRetry }) {
  return (
    <div className="error-state animate-fade-in">
      <div className="error-state-icon">
        <AlertCircle size={32} />
      </div>
      <h4>Something went wrong</h4>
      <p>{message || 'An unexpected error occurred. Please try again.'}</p>
      {onRetry && (
        <button className="btn btn-ghost" onClick={onRetry}>
          <RefreshCw size={15} />
          <span>Try Again</span>
        </button>
      )}
    </div>
  );
}
