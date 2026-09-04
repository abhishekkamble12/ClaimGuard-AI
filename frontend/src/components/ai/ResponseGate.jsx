import { CheckCircle2, ShieldAlert, Zap, FileText } from 'lucide-react';
import './ResponseGate.css';

/**
 * Gate status banner indicating whether automated response generation is unlocked or restricted.
 */
export default function ResponseGate({
  routingDecision,
  readinessPct,
  confidencePct,
  onStreamClick,
  onGenerateStaticClick,
  isStreaming,
}) {
  const isPassed = routingDecision === 'auto_draft_response';

  return (
    <div className={`response-gate card ${isPassed ? 'response-gate--passed' : 'response-gate--blocked'}`}>
      <div className="response-gate-header">
        <div className="response-gate-status">
          {isPassed ? (
            <>
              <CheckCircle2 size={20} className="text-success" />
              <div>
                <h4 className="text-success">Gate Status: PASSED</h4>
                <p className="response-gate-sub">
                  Readiness ({readinessPct}) &ge; 80% & Confidence ({confidencePct}) &ge; 75%. Safe for automated submission.
                </p>
              </div>
            </>
          ) : (
            <>
              <ShieldAlert size={20} className="text-warning" />
              <div>
                <h4 className="text-warning">Gate Status: BLOCKED</h4>
                <p className="response-gate-sub">
                  Automated drafting restricted to protect merchant win rates and prevent non-refundable penalty fees.
                </p>
              </div>
            </>
          )}
        </div>

        {isPassed && (
          <div className="response-gate-actions">
            <button
              className="btn btn-primary"
              onClick={onStreamClick}
              disabled={isStreaming}
            >
              <Zap size={15} />
              <span>{isStreaming ? 'Streaming...' : 'Stream AI Draft'}</span>
            </button>
            <button
              className="btn btn-ghost"
              onClick={onGenerateStaticClick}
              disabled={isStreaming}
            >
              <FileText size={15} />
              <span>Full Response</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
