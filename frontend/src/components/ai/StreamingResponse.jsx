import { StopCircle, Bot } from 'lucide-react';
import FeedbackButtons from './FeedbackButtons';
import './StreamingResponse.css';

/**
 * Typewriter-style streaming AI response display.
 * Props: text, isStreaming, error, onStop, onRetry
 */
export default function StreamingResponse({ text, isStreaming, error, onStop, onRetry }) {
  return (
    <div className="streaming-response card">
      <div className="streaming-header">
        <div className="streaming-label">
          <Bot size={16} />
          <span>AI Response</span>
          {isStreaming && <span className="streaming-live-badge">LIVE</span>}
        </div>
        {isStreaming && (
          <button className="btn btn-sm btn-danger" onClick={onStop}>
            <StopCircle size={14} />
            Stop
          </button>
        )}
      </div>

      <div className="streaming-content">
        {text ? (
          <pre className="streaming-text">
            {text}
            {isStreaming && <span className="streaming-cursor">▌</span>}
          </pre>
        ) : isStreaming ? (
          <div className="streaming-waiting">
            <span className="streaming-dot" />
            <span className="streaming-dot" />
            <span className="streaming-dot" />
          </div>
        ) : error ? (
          <p className="text-danger">{error}</p>
        ) : null}
      </div>

      {text && !isStreaming && (
        <div className="streaming-footer">
          <span className="streaming-meta text-muted">
            {text.length} characters generated
          </span>
          <FeedbackButtons text={text} onRetry={onRetry} />
        </div>
      )}
    </div>
  );
}
