import { Copy, RefreshCw, ThumbsUp, ThumbsDown, Edit3 } from 'lucide-react';
import { useState } from 'react';
import './FeedbackButtons.css';

/**
 * Feedback buttons for AI-generated content: Copy, Retry, Edit, 👍/👎
 * Props: text (content to copy), onRetry, onEdit, onFeedback
 */
export default function FeedbackButtons({ text, onRetry, onEdit, onFeedback }) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard not available */
    }
  };

  const handleFeedback = (type) => {
    setFeedback(type);
    onFeedback?.(type);
  };

  return (
    <div className="feedback-buttons">
      <button className="btn-icon" onClick={handleCopy} title="Copy to clipboard">
        <Copy size={15} />
        {copied && <span className="feedback-toast">Copied!</span>}
      </button>

      {onRetry && (
        <button className="btn-icon" onClick={onRetry} title="Regenerate">
          <RefreshCw size={15} />
        </button>
      )}

      {onEdit && (
        <button className="btn-icon" onClick={onEdit} title="Edit">
          <Edit3 size={15} />
        </button>
      )}

      <div className="feedback-divider" />

      <button
        className={`btn-icon ${feedback === 'up' ? 'feedback-active--up' : ''}`}
        onClick={() => handleFeedback('up')}
        title="Helpful"
      >
        <ThumbsUp size={15} />
      </button>

      <button
        className={`btn-icon ${feedback === 'down' ? 'feedback-active--down' : ''}`}
        onClick={() => handleFeedback('down')}
        title="Not helpful"
      >
        <ThumbsDown size={15} />
      </button>
    </div>
  );
}
